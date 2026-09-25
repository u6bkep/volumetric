//! 2.5D solids: a top and a bottom height field over a footprint.
//!
//! A [`Solid`] is `inside(x, y) && bot(x, y) <= z <= top(x, y)` sampled on
//! a regular grid, plus the footprint as exact 2D regions
//! ([`sketch_regions_core::Region`]) so vertical walls keep sub-cell
//! accuracy. Every cell also carries the label of the feature that last set
//! its top / bottom height, which is how blends find the edges they round.
//!
//! Operations (all per column, all exact for prisms):
//! - [`Solid::extrude`]: add or remove a prism (a region swept along a
//!   direction between two distances), on any plane;
//! - [`Solid::apply_body`]: add or remove a general [`Body`] given as its
//!   own lower / upper height fields (lofts, imported meshes);
//! - [`kernel`]: rolling-ball fillets and cone chamfers as band-limited
//!   grey-scale morphology on the height fields;
//! - [`payload`]: the sampled model payload the wasm template reads.
//!
//! Heights are `NaN` where a column holds no material. Units are whatever
//! the caller uses (metres for the Onshape import).

pub mod edt;
pub mod kernel;
pub mod vertical;
pub mod loft;
pub mod payload;
pub mod raster;

use sketch_regions_core::Region;

/// A regular grid of cell centres: cell `(i, j)` is at
/// `(x0 + (i + 0.5) pitch, y0 + (j + 0.5) pitch)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub nx: usize,
    pub ny: usize,
    pub x0: f64,
    pub y0: f64,
    pub pitch: f64,
}

impl Grid {
    /// A grid covering `[x0, x1] x [y0, y1]` at `pitch`, with `margin`
    /// cells of padding on every side.
    pub fn covering(x0: f64, y0: f64, x1: f64, y1: f64, pitch: f64, margin: usize) -> Grid {
        let nx = ((x1 - x0) / pitch).ceil() as usize + 2 * margin;
        let ny = ((y1 - y0) / pitch).ceil() as usize + 2 * margin;
        Grid { nx, ny, x0: x0 - margin as f64 * pitch, y0: y0 - margin as f64 * pitch, pitch }
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.nx * self.ny
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[inline]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }
    #[inline]
    pub fn x(&self, i: usize) -> f64 {
        self.x0 + (i as f64 + 0.5) * self.pitch
    }
    #[inline]
    pub fn y(&self, j: usize) -> f64 {
        self.y0 + (j as f64 + 0.5) * self.pitch
    }
    #[inline]
    pub fn ij(&self, k: usize) -> (usize, usize) {
        (k % self.nx, k / self.nx)
    }
    /// Cell index range `[i0, i1) x [j0, j1)` overlapping the box
    /// `[bx0, by0, bx1, by1]` (clamped to the grid).
    pub fn cells_in(&self, bb: [f64; 4]) -> (usize, usize, usize, usize) {
        let c = |v: f64, o: f64, n: usize| -> usize { ((v - o) / self.pitch).floor().clamp(0.0, n as f64) as usize };
        let i0 = c(bb[0], self.x0, self.nx);
        let j0 = c(bb[1], self.y0, self.ny);
        let i1 = (((bb[2] - self.x0) / self.pitch).ceil().max(0.0) as usize).min(self.nx);
        let j1 = (((bb[3] - self.y0) / self.pitch).ceil().max(0.0) as usize).min(self.ny);
        (i0, i1, j0, j1)
    }
}

/// A closed 2.5D volume as lower / upper heights per column (`NaN` = the
/// column misses the body). The intermediate form of lofts, prisms on
/// oblique planes and imported meshes before they are combined into a
/// [`Solid`].
#[derive(Clone, Debug)]
pub struct Body {
    pub grid: Grid,
    pub lo: Vec<f32>,
    pub hi: Vec<f32>,
    /// Optional per-column labels for the upper and lower surfaces (which
    /// face of the body they are), used instead of the operation's label
    /// when present.
    pub hi_label: Option<Vec<u32>>,
    pub lo_label: Option<Vec<u32>>,
}

impl Body {
    pub fn empty(grid: &Grid) -> Body {
        Body { grid: grid.clone(), lo: vec![f32::NAN; grid.len()], hi: vec![f32::NAN; grid.len()], hi_label: None, lo_label: None }
    }
    /// Widen column `k` to include `z` and remember which face set the
    /// extreme it changed.
    #[inline]
    pub fn include_labelled(&mut self, k: usize, z: f32, label: u32) {
        if self.hi_label.is_none() {
            self.hi_label = Some(vec![u32::MAX; self.grid.len()]);
            self.lo_label = Some(vec![u32::MAX; self.grid.len()]);
        }
        let (hl, ll) = (self.hi_label.as_mut().unwrap(), self.lo_label.as_mut().unwrap());
        if self.lo[k].is_nan() {
            self.lo[k] = z;
            self.hi[k] = z;
            hl[k] = label;
            ll[k] = label;
        } else {
            if z > self.hi[k] {
                self.hi[k] = z;
                hl[k] = label;
            }
            if z < self.lo[k] {
                self.lo[k] = z;
                ll[k] = label;
            }
        }
    }
    /// Widen column `k` to include `z`.
    #[inline]
    pub fn include(&mut self, k: usize, z: f32) {
        if self.lo[k].is_nan() {
            self.lo[k] = z;
            self.hi[k] = z;
        } else {
            self.lo[k] = self.lo[k].min(z);
            self.hi[k] = self.hi[k].max(z);
        }
    }
    /// Widen column `k` to include `[z0, z1]`.
    #[inline]
    pub fn include_span(&mut self, k: usize, z0: f32, z1: f32) {
        self.include(k, z0);
        self.include(k, z1);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Add,
    Remove,
}

/// Which height field an operation addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Top,
    Bottom,
}

/// Per-column trouble an operation ran into: the column would have needed
/// two material intervals (a cut through the middle, a floating addition).
#[derive(Clone, Debug, Default)]
pub struct OpReport {
    pub cells_changed: usize,
    /// Columns where the result was forced back to one interval.
    pub split_columns: usize,
    /// Columns that were emptied entirely.
    pub emptied: usize,
}

#[derive(Clone, Debug)]
pub struct Solid {
    pub grid: Grid,
    pub top: Vec<f32>,
    pub bot: Vec<f32>,
    pub top_label: Vec<u32>,
    pub bot_label: Vec<u32>,
    /// Feature ordinal that last set the top / bottom of each cell, used to
    /// attribute the vertical wall between two cells to the newer feature.
    pub top_order: Vec<u32>,
    pub bot_order: Vec<u32>,
    /// Exact footprint (unioned regions in the grid's x/y coordinates),
    /// when the caller maintains it; `None` falls back to the sampled mask.
    pub outline: Option<Vec<Region>>,
}

impl Solid {
    pub fn empty(grid: Grid) -> Solid {
        let n = grid.len();
        Solid {
            grid,
            top: vec![f32::NAN; n],
            bot: vec![f32::NAN; n],
            top_label: vec![0; n],
            bot_label: vec![0; n],
            top_order: vec![0; n],
            bot_order: vec![0; n],
            outline: None,
        }
    }

    #[inline]
    pub fn has_material(&self, k: usize) -> bool {
        !self.top[k].is_nan()
    }

    pub fn mask(&self) -> Vec<bool> {
        self.top.iter().map(|t| !t.is_nan()).collect()
    }

    /// Combine a body into the solid column by column.
    pub fn apply_body(&mut self, body: &Body, op: Op, label: u32, order: u32) -> OpReport {
        assert_eq!(body.grid, self.grid, "body grid must match the solid grid");
        let mut rep = OpReport::default();
        for k in 0..self.grid.len() {
            let (lo, hi) = (body.lo[k], body.hi[k]);
            if lo.is_nan() {
                continue;
            }
            let pick = |labels: &Option<Vec<u32>>| labels.as_ref().map(|l| l[k]).filter(|&l| l != u32::MAX).unwrap_or(label);
            let (hi_label, lo_label) = (pick(&body.hi_label), pick(&body.lo_label));
            match op {
                Op::Add => {
                    if !self.has_material(k) {
                        self.top[k] = hi;
                        self.bot[k] = lo;
                        self.top_label[k] = hi_label;
                        self.bot_label[k] = lo_label;
                        self.top_order[k] = order;
                        self.bot_order[k] = order;
                        rep.cells_changed += 1;
                        continue;
                    }
                    let (t, b) = (self.top[k], self.bot[k]);
                    if lo > t || hi < b {
                        rep.split_columns += 1; // floating: bridge the gap
                    }
                    let mut changed = false;
                    if hi > t {
                        self.top[k] = hi;
                        self.top_label[k] = hi_label;
                        self.top_order[k] = order;
                        changed = true;
                    }
                    if lo < b {
                        self.bot[k] = lo;
                        self.bot_label[k] = lo_label;
                        self.bot_order[k] = order;
                        changed = true;
                    }
                    rep.cells_changed += changed as usize;
                }
                Op::Remove => {
                    if !self.has_material(k) {
                        continue;
                    }
                    let (t, b) = (self.top[k], self.bot[k]);
                    if hi <= b || lo >= t {
                        continue;
                    }
                    if lo <= b && hi >= t {
                        self.top[k] = f32::NAN;
                        self.bot[k] = f32::NAN;
                        rep.emptied += 1;
                        rep.cells_changed += 1;
                        continue;
                    }
                    if lo > b && hi < t {
                        // Cut through the middle: keep the thicker part.
                        rep.split_columns += 1;
                        if t - hi >= lo - b {
                            self.bot[k] = hi;
                            self.bot_label[k] = hi_label;
                            self.bot_order[k] = order;
                        } else {
                            self.top[k] = lo;
                            self.top_label[k] = lo_label;
                            self.top_order[k] = order;
                        }
                        rep.cells_changed += 1;
                        continue;
                    }
                    if hi >= t {
                        self.top[k] = lo;
                        self.top_label[k] = lo_label;
                        self.top_order[k] = order;
                    } else {
                        self.bot[k] = hi;
                        self.bot_label[k] = hi_label;
                        self.bot_order[k] = order;
                    }
                    rep.cells_changed += 1;
                }
            }
        }
        rep
    }

    /// Extrude: sweep `prism` and combine it.
    pub fn extrude(&mut self, prism: &Prism, op: Op, label: u32, order: u32) -> OpReport {
        let body = prism.body(&self.grid);
        self.apply_body(&body, op, label, order)
    }

    /// Empty every column whose centre lies outside the exact outline (after
    /// the outline changed). Returns the number of columns emptied.
    pub fn clip_to_outline(&mut self) -> usize {
        let Some(outline) = self.outline.clone() else { return 0 };
        let g = self.grid.clone();
        let mut n = 0;
        for j in 0..g.ny {
            for i in 0..g.nx {
                let k = g.idx(i, j);
                if self.top[k].is_nan() {
                    continue;
                }
                let p = [g.x(i), g.y(j)];
                if !outline.iter().any(|r| r.contains(p)) {
                    self.top[k] = f32::NAN;
                    self.bot[k] = f32::NAN;
                    n += 1;
                }
            }
        }
        n
    }

    /// Fill the heights of empty columns from the nearest filled column so
    /// bilinear sampling near the footprint boundary never touches `NaN`.
    /// Returns the filled copies without changing the solid.
    pub fn filled_fields(&self) -> (Vec<f32>, Vec<f32>) {
        let mask = self.mask();
        let (_, nearest) = edt::distance_to_sites(self.grid.nx, self.grid.ny, |k| mask[k]);
        let mut top = self.top.clone();
        let mut bot = self.bot.clone();
        for k in 0..self.grid.len() {
            if !mask[k] && nearest[k] != usize::MAX {
                top[k] = self.top[nearest[k]];
                bot[k] = self.bot[nearest[k]];
            }
        }
        (top, bot)
    }

    /// Signed distance (in length units, positive inside) to the footprint
    /// boundary at every cell centre: the exact outline where one is kept,
    /// tightened by the sampled mask where columns were emptied.
    pub fn footprint_sdf(&self) -> Vec<f32> {
        let mask = self.mask();
        let g = &self.grid;
        let grid_sdf: Vec<f64> = edt::signed_distance(g.nx, g.ny, &mask).into_iter().map(|d| d * g.pitch).collect();
        let Some(outline) = &self.outline else {
            return grid_sdf.into_iter().map(|d| d as f32).collect();
        };
        let mut out = vec![0f32; g.len()];
        for j in 0..g.ny {
            for i in 0..g.nx {
                let k = g.idx(i, j);
                let p = [g.x(i), g.y(j)];
                let exact = outline.iter().map(|r| r.signed_distance(p)).fold(f64::NEG_INFINITY, f64::max);
                // A column the exact outline contains but the mask lost was
                // emptied by a cut, and one the mask holds beyond the outline
                // was added by a sweep the outline does not track; the sampled
                // distance stands in for both.
                let d = if exact > 0.0 && !mask[k] {
                    grid_sdf[k].min(exact)
                } else if exact <= 0.0 && mask[k] {
                    grid_sdf[k]
                } else {
                    exact
                };
                out[k] = d as f32;
            }
        }
        out
    }
}

/// A region on a plane swept along a direction.
///
/// Plane points are `origin + u a + v b` for region coordinates `(a, b)`;
/// the prism is `{plane point + dir w : (a, b) in region, w0 <= w <= w1}`.
/// `w0` / `w1` may be infinite (through-all).
#[derive(Clone, Debug)]
pub struct Prism {
    pub region: Region,
    pub origin: [f64; 3],
    pub u: [f64; 3],
    pub v: [f64; 3],
    pub dir: [f64; 3],
    pub w0: f64,
    pub w1: f64,
}

fn inv3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-18 {
        return None;
    }
    let d = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * d,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * d,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * d,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * d,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * d,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * d,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * d,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * d,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * d,
        ],
    ])
}

fn mul3(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

impl Prism {
    /// The prism's occupancy per column of `grid`.
    pub fn body(&self, grid: &Grid) -> Body {
        let mut body = Body::empty(grid);
        // (a, b, w) = M^-1 (p - origin) with M = [u v dir] as columns.
        let m = [
            [self.u[0], self.v[0], self.dir[0]],
            [self.u[1], self.v[1], self.dir[1]],
            [self.u[2], self.v[2], self.dir[2]],
        ];
        let Some(inv) = inv3(m) else { return body };
        // Along a column, (a, b, w) = q0(x, y) + z * q1.
        let q1 = mul3(&inv, [0.0, 0.0, 1.0]);
        let vertical = q1[0].abs() < 1e-12 && q1[1].abs() < 1e-12;
        // Footprint bounding box in the grid: sample the region's bbox corners
        // at both sweep ends (exact for the prism's convex hull).
        let bb = self.region.bbox();
        let mut fx = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        let ws: [f64; 2] = [
            if self.w0.is_finite() { self.w0 } else { -1e3 },
            if self.w1.is_finite() { self.w1 } else { 1e3 },
        ];
        for a in [bb[0], bb[2]] {
            for b in [bb[1], bb[3]] {
                for w in ws {
                    let p = [
                        self.origin[0] + self.u[0] * a + self.v[0] * b + self.dir[0] * w,
                        self.origin[1] + self.u[1] * a + self.v[1] * b + self.dir[1] * w,
                    ];
                    fx[0] = fx[0].min(p[0]);
                    fx[1] = fx[1].min(p[1]);
                    fx[2] = fx[2].max(p[0]);
                    fx[3] = fx[3].max(p[1]);
                }
            }
        }
        let (i0, i1, j0, j1) = grid.cells_in([fx[0] - grid.pitch, fx[1] - grid.pitch, fx[2] + grid.pitch, fx[3] + grid.pitch]);
        for j in j0..j1 {
            for i in i0..i1 {
                let k = grid.idx(i, j);
                let p = [grid.x(i) - self.origin[0], grid.y(j) - self.origin[1], -self.origin[2]];
                let q0 = mul3(&inv, p);
                if vertical {
                    if q1[2].abs() < 1e-300 || !self.region.contains([q0[0], q0[1]]) {
                        continue;
                    }
                    let z0 = (self.w0 - q0[2]) / q1[2];
                    let z1 = (self.w1 - q0[2]) / q1[2];
                    let (za, zb) = if z0 <= z1 { (z0, z1) } else { (z1, z0) };
                    body.include_span(k, za as f32, zb as f32);
                } else {
                    // The column traces the line (a, b)(z) = q0 + z q1 in the
                    // plane; material where it is inside the region and the
                    // sweep coordinate is within range.
                    let ivs = self.region.line_intervals([q0[0], q0[1]], [q1[0], q1[1]]);
                    if ivs.is_empty() {
                        continue;
                    }
                    let (mut wa, mut wb) = (f64::NEG_INFINITY, f64::INFINITY);
                    if q1[2].abs() > 1e-300 {
                        let z0 = (self.w0 - q0[2]) / q1[2];
                        let z1 = (self.w1 - q0[2]) / q1[2];
                        wa = z0.min(z1);
                        wb = z0.max(z1);
                    } else if q0[2] < self.w0 || q0[2] > self.w1 {
                        continue;
                    }
                    for (s0, s1) in ivs {
                        let za = s0.max(wa);
                        let zb = s1.min(wb);
                        if za < zb {
                            body.include_span(k, za as f32, zb as f32);
                        }
                    }
                }
            }
        }
        body
    }
}

// ---------------------------------------------------------------------------
// Edges between faces
// ---------------------------------------------------------------------------

/// A face of the solid as the blend machinery sees it: the cells a feature
/// labelled, or the vertical wall a feature made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceSel {
    Cap(u32),
    Wall(u32),
    /// The footprint's vertical side wall.
    Outline,
    /// Any cap face at all.
    AnyCap,
}

/// Cells on the selected edge, on the side of the first face selector, with
/// the neighbouring cell across the edge.
#[derive(Clone, Debug, Default)]
pub struct EdgeCells {
    pub cells: Vec<usize>,
    pub across: Vec<usize>,
    /// Whether the solid is convex across the edge (a ridge / the top of a
    /// wall) as opposed to concave (a valley / the foot of a wall).
    pub convex: bool,
    /// Whether the second face has cells of its own (a cap) rather than
    /// being a wall or the footprint wall.
    pub b_has_cells: bool,
}

const NEIGHBOURS: [(isize, isize); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

impl Solid {
    fn field(&self, side: Side) -> (&[f32], &[u32], &[u32]) {
        match side {
            Side::Top => (&self.top, &self.top_label, &self.top_order),
            Side::Bottom => (&self.bot, &self.bot_label, &self.bot_order),
        }
    }

    /// Find the cells of the edge where face `a` meets face `b` on `side`.
    ///
    /// One of the two must be a cap (cells carry no wall), and the edge is
    /// reported on that cap's cells: each cell whose interface with a
    /// 4-neighbour is the other face. An interface is the footprint wall
    /// when the neighbour is empty, a wall owned by the newer of the two
    /// features when the heights differ by at least `jump`, and otherwise a
    /// crease between two differently labelled caps.
    pub fn edge_cells(&self, side: Side, a: FaceSel, b: FaceSel, jump: f32) -> EdgeCells {
        let (cap, other) = match (a, b) {
            (FaceSel::Cap(_) | FaceSel::AnyCap, _) => (a, b),
            (_, FaceSel::Cap(_) | FaceSel::AnyCap) => (b, a),
            _ => return EdgeCells::default(),
        };
        let g = &self.grid;
        let (h, label, order) = self.field(side);
        let sign = if side == Side::Top { 1.0f32 } else { -1.0 };
        let mut out = EdgeCells { b_has_cells: matches!(other, FaceSel::Cap(_) | FaceSel::AnyCap), ..Default::default() };
        let mut convex_votes: i64 = 0;
        for j in 0..g.ny {
            for i in 0..g.nx {
                let k = g.idx(i, j);
                if h[k].is_nan() {
                    continue;
                }
                if let FaceSel::Cap(l) = cap {
                    if label[k] != l {
                        continue;
                    }
                }
                for (di, dj) in NEIGHBOURS {
                    let (ii, jj) = (i as isize + di, j as isize + dj);
                    if ii < 0 || jj < 0 || ii >= g.nx as isize || jj >= g.ny as isize {
                        continue;
                    }
                    let n = g.idx(ii as usize, jj as usize);
                    let n_empty = h[n].is_nan();
                    let dz = if n_empty { 0.0 } else { (h[k] - h[n]) * sign }; // > 0: k higher
                    let is_wall = !n_empty && dz.abs() >= jump;
                    let hit = match other {
                        FaceSel::Outline => n_empty,
                        FaceSel::Wall(l) => is_wall && (if order[k] >= order[n] { label[k] } else { label[n] }) == l,
                        FaceSel::Cap(l) => !n_empty && !is_wall && label[n] == l && label[n] != label[k],
                        FaceSel::AnyCap => !n_empty && !is_wall && label[n] != label[k],
                    };
                    if !hit {
                        continue;
                    }
                    out.cells.push(k);
                    out.across.push(n);
                    if n_empty {
                        convex_votes += 1;
                    } else if is_wall {
                        convex_votes += if dz > 0.0 { 1 } else { -1 };
                    } else {
                        let (bi, bj) = (i as isize - di, j as isize - dj);
                        if bi >= 0 && bj >= 0 && bi < g.nx as isize && bj < g.ny as isize {
                            let bk = g.idx(bi as usize, bj as usize);
                            if !h[bk].is_nan() {
                                let second = (h[n] + h[bk] - 2.0 * h[k]) * sign;
                                convex_votes += if second < 0.0 { 1 } else { -1 };
                            }
                        }
                    }
                    break;
                }
            }
        }
        out.convex = convex_votes >= 0;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sketch_regions_core::{Curve, Piece, arrange};

    fn rect_region(x0: f64, y0: f64, x1: f64, y1: f64) -> Region {
        let p = |a, b| Piece { id: "r".into(), curve: Curve::Line { a, b } };
        let pieces = vec![p([x0, y0], [x1, y0]), p([x1, y0], [x1, y1]), p([x1, y1], [x0, y1]), p([x0, y1], [x0, y0])];
        arrange(&pieces, 1e-9).remove(0)
    }

    #[test]
    fn slab_with_pocket_and_boss() {
        let grid = Grid::covering(0.0, 0.0, 10.0, 8.0, 0.5, 2);
        let mut s = Solid::empty(grid.clone());
        let slab = Prism {
            region: rect_region(0.0, 0.0, 10.0, 8.0),
            origin: [0.0; 3],
            u: [1.0, 0.0, 0.0],
            v: [0.0, 1.0, 0.0],
            dir: [0.0, 0.0, 1.0],
            w0: 0.0,
            w1: 2.0,
        };
        let rep = s.extrude(&slab, Op::Add, 1, 1);
        assert_eq!(rep.cells_changed, 20 * 16);
        // Pocket 1 deep from the top face.
        let pocket = Prism { region: rect_region(2.0, 2.0, 4.0, 4.0), w0: 1.0, w1: 2.0, ..slab.clone() };
        let rep = s.extrude(&pocket, Op::Remove, 2, 2);
        assert_eq!(rep.cells_changed, 16);
        assert_eq!(rep.split_columns, 0);
        let k = grid.idx(2 + 5, 2 + 5); // centre of the pocket region (2.5, 2.5)
        assert_eq!(s.top[k], 1.0);
        assert_eq!(s.top_label[k], 2);
        // Boss on top.
        let boss = Prism { region: rect_region(6.0, 5.0, 8.0, 7.0), w0: 2.0, w1: 3.0, ..slab.clone() };
        s.extrude(&boss, Op::Add, 3, 3);
        let k = grid.idx(2 + 13, 2 + 11);
        assert_eq!(s.top[k], 3.0);
        // Through cut empties columns.
        let hole = Prism { region: rect_region(0.5, 6.0, 1.5, 7.0), w0: f64::NEG_INFINITY, w1: f64::INFINITY, ..slab.clone() };
        let rep = s.extrude(&hole, Op::Remove, 4, 4);
        assert_eq!(rep.emptied, 4);
        assert!(s.top[grid.idx(2 + 2, 2 + 13)].is_nan());
        // Edges: pocket floor / pocket wall (concave), pocket wall / slab top (convex).
        let e = s.edge_cells(Side::Top, FaceSel::Cap(2), FaceSel::Wall(2), 0.01);
        assert_eq!(e.cells.len(), 12, "{:?}", e.cells);
        assert!(!e.convex);
        let e = s.edge_cells(Side::Top, FaceSel::Cap(1), FaceSel::Wall(2), 0.01);
        assert_eq!(e.cells.len(), 16);
        assert!(e.convex);
        let e = s.edge_cells(Side::Top, FaceSel::Cap(3), FaceSel::Wall(3), 0.01);
        assert_eq!(e.cells.len(), 12);
        assert!(e.convex);
        let e = s.edge_cells(Side::Top, FaceSel::Cap(1), FaceSel::Wall(3), 0.01);
        assert!(!e.convex);
        // Top rim of the slab against the footprint.
        let e = s.edge_cells(Side::Top, FaceSel::Cap(1), FaceSel::Outline, 0.01);
        assert!(e.convex);
        assert!(e.cells.len() >= 2 * 20 + 2 * 16 - 4);
        // Footprint distance: exact outline when set, sampled otherwise.
        let d = s.footprint_sdf();
        assert!((d[grid.idx(2 + 10, 2 + 8)] - 3.75).abs() < 1e-6);
        s.outline = Some(vec![rect_region(0.0, 0.0, 10.0, 8.0)]);
        let d = s.footprint_sdf();
        assert!((d[grid.idx(2 + 10, 2 + 8)] - 3.75).abs() < 1e-6);
        assert!((d[grid.idx(2, 2)] - 0.25).abs() < 1e-6);
        assert!((d[grid.idx(1, 2)] + 0.25).abs() < 1e-6);
        // The through hole shows in the sampled part.
        assert!(d[grid.idx(2 + 2, 2 + 13)] < 0.0);
    }

    #[test]
    fn oblique_prism_on_a_vertical_plane() {
        // A triangle profile in the x = 0 plane (coordinates (y, z)) swept
        // along +x: a ridge whose height varies with y.
        let grid = Grid::covering(0.0, 0.0, 4.0, 4.0, 0.5, 0);
        let mut s = Solid::empty(grid.clone());
        let p = |a, b| Piece { id: "t".into(), curve: Curve::Line { a, b } };
        let tri = arrange(&[p([0.0, 0.0], [4.0, 0.0]), p([4.0, 0.0], [2.0, 2.0]), p([2.0, 2.0], [0.0, 0.0])], 1e-9).remove(0);
        let ridge = Prism {
            region: tri,
            origin: [0.0; 3],
            u: [0.0, 1.0, 0.0],
            v: [0.0, 0.0, 1.0],
            dir: [1.0, 0.0, 0.0],
            w0: 0.0,
            w1: 4.0,
        };
        s.extrude(&ridge, Op::Add, 1, 1);
        let k = grid.idx(4, 4); // (2.25, 2.25): height 2 - |y - 2| = 1.75
        assert!((s.top[k] - 1.75).abs() < 1e-6, "{}", s.top[k]);
        assert_eq!(s.bot[k], 0.0);
        let k = grid.idx(4, 0); // y = 0.25 -> 0.25
        assert!((s.top[k] - 0.25).abs() < 1e-6);
    }
}
