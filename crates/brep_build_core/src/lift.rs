//! The lift: a plan map as a boundary representation.
//!
//! Every boundary piece of the plan has a cell on each side (or nothing),
//! and over it the heights at which those cells' columns begin and end:
//! the *levels* of the piece. A level is an edge in space where a face
//! ends on it. Between two levels there is a wall where exactly one side
//! has material. Over a vertex of the plan the walls that meet there
//! share vertical edges. The faces on the sheets are what is left: for
//! each sheet and each side of it (top of the columns or bottom), the
//! loops of level edges around the cells that end on it.
//!
//! Edges get the exact curve where there is one (a line or an arc of the
//! plan on a plane is a line, a circle or an ellipse) and a fitted cubic
//! B-spline otherwise; walls are planes, cylinders, or for a traced curve
//! its spline swept upwards.
//!
//! What a 2.5D solid can hold but a manifold boundary cannot (two lumps
//! touching along an edge, four walls on one vertical) is reported, not
//! written.

use crate::conic::conic_through;
use crate::fit::spline_through;
use crate::planmap::{PlanMap, deep_point};
use crate::trace::FREE;
use brep_core::ir::{NurbsSurface, Surface};
use brep_core::math::{Frame, Vec3, norm, sub as sub3};
use brep_core::nurbs::curve_eval;
use brep_core::topo::{Coedge, Curve3, Edge, Face, NurbsCurve, Solid};
use core::f64::consts::TAU;
use sketch_regions_core::geom::{add, cross, dist, dot, scale, sub};
use sketch_regions_core::{Curve, P2, Piece, arrangement};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct LiftOptions {
    /// Fitted curves stay this close to what they follow, metres.
    pub fit: f64,
}

impl Default for LiftOptions {
    fn default() -> Self {
        LiftOptions { fit: 5e-8 }
    }
}

/// Points of an edge are taken this far apart at most, metres.
const SPACING: f64 = 2.5e-4;

/// Points within this of a line, a circle or an ellipse are on it.
const EXACT: f64 = 1e-10;

/// A boundary curve of the plan between two vertices.
#[derive(Clone, Debug)]
enum Geom {
    Line {
        a: P2,
        b: P2,
    },
    Arc {
        c: P2,
        r: f64,
        a0: f64,
        sweep: f64,
    },
    /// A traced curve: its points and the spline through them, over
    /// `[0, 1]`.
    Free {
        pts: Vec<P2>,
        spline: NurbsCurve,
    },
}

impl Geom {
    fn point(&self, t: f64) -> P2 {
        match self {
            Geom::Line { a, b } => add(*a, scale(sub(*b, *a), t)),
            Geom::Arc { c, r, a0, sweep } => {
                let th = a0 + sweep * t;
                [c[0] + r * th.cos(), c[1] + r * th.sin()]
            }
            Geom::Free { spline, .. } => {
                let p = curve_eval(spline, t).0;
                [p[0], p[1]]
            }
        }
    }

    /// Unit direction of travel.
    fn tangent(&self, t: f64) -> P2 {
        let d = match self {
            Geom::Line { a, b } => sub(*b, *a),
            Geom::Arc { a0, sweep, .. } => {
                let th = a0 + sweep * t;
                scale([-th.sin(), th.cos()], sweep.signum())
            }
            Geom::Free { spline, .. } => {
                let d = curve_eval(spline, t).1;
                [d[0], d[1]]
            }
        };
        let n = (d[0] * d[0] + d[1] * d[1]).sqrt();
        if n > 0.0 {
            scale(d, 1.0 / n)
        } else {
            [1.0, 0.0]
        }
    }

    fn flatten(&self) -> Vec<P2> {
        match self {
            Geom::Line { a, b } => vec![*a, *b],
            Geom::Arc { sweep, .. } => {
                let n = ((sweep.abs() / 0.1).ceil() as usize).max(2);
                (0..=n).map(|i| self.point(i as f64 / n as f64)).collect()
            }
            Geom::Free { pts, .. } => pts.clone(),
        }
    }

    fn closed_circle(&self) -> bool {
        matches!(self, Geom::Arc { sweep, .. } if sweep.abs() >= TAU - 1e-9)
    }
}

/// One use of a sheet by a side of a plan edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Entry {
    /// 0 the cell on the left of the edge, 1 the one on the right.
    side: usize,
    top: bool,
    sheet: usize,
}

struct Level {
    entries: Vec<Entry>,
    /// Index into the 3D edges, when a face ends here.
    edge: Option<usize>,
}

struct PlanEdge {
    id: String,
    geom: Geom,
    v0: usize,
    v1: usize,
    levels: Vec<Level>,
    /// Per gap between consecutive levels: the side with material, when
    /// only one has.
    walls: Vec<Option<usize>>,
}

struct Atom {
    id: String,
    curve: Curve,
    from: usize,
    to: usize,
    cells: [Option<usize>; 2],
}

struct Builder<'a> {
    map: &'a PlanMap,
    opts: LiftOptions,
    plan: Vec<P2>,
    solid: Solid,
    /// 3D vertices over each plan vertex: `(height, index)`.
    over: HashMap<usize, Vec<(f64, usize)>>,
}

impl Builder<'_> {
    fn z(&self, sheet: usize, p: P2) -> Result<f64, String> {
        self.map.sheets[sheet].z(p[0], p[1]).ok_or_else(|| {
            format!(
                "sheet {sheet} ({}) does not reach ({:.6}, {:.6})",
                self.map.sheets[sheet].name, p[0], p[1]
            )
        })
    }

    fn vertex(&mut self, plan: usize, z: f64) -> usize {
        let tol = 10.0 * self.map.tol.height;
        let list = self.over.entry(plan).or_default();
        if let Some(&(_, v)) = list.iter().find(|(h, _)| (h - z).abs() <= tol) {
            return v;
        }
        let p = self.plan[plan];
        self.solid.vertices.push([p[0], p[1], z]);
        let v = self.solid.vertices.len() - 1;
        list.push((z, v));
        v
    }

    /// The sheets of the columns on both sides: `[bot, top]` per side.
    fn sheets_of(&self, cells: [Option<usize>; 2]) -> [Option<[usize; 2]>; 2] {
        cells.map(|c| c.map(|c| [self.map.cells[c].bot, self.map.cells[c].top]))
    }

    /// Where along an atom the order of the two sides' heights changes.
    fn relation_changes(&self, atom: &Atom) -> Result<Vec<f64>, String> {
        let [Some(l), Some(r)] = self.sheets_of(atom.cells) else {
            return Ok(Vec::new());
        };
        let tol = self.map.tol.height;
        let mut cuts: Vec<f64> = Vec::new();
        for (a, b) in [(l[1], r[1]), (l[0], r[0]), (l[1], r[0]), (l[0], r[1])] {
            if a == b {
                continue;
            }
            let (sa, sb) = (&self.map.sheets[a], &self.map.sheets[b]);
            if let (Some(pa), Some(pb)) = (sa.as_plane(), sb.as_plane())
                && (pa.1 - pb.1).abs() < 1e-14
                && (pa.2 - pb.2).abs() < 1e-14
            {
                continue;
            }
            let diff = |t: f64| -> Result<f64, String> {
                let p = atom.curve.point(t);
                Ok(self.z(a, p)? - self.z(b, p)?)
            };
            let sign = |d: f64| {
                if d.abs() <= tol {
                    0
                } else if d > 0.0 {
                    1
                } else {
                    -1
                }
            };
            const N: usize = 8;
            let mut c = [0i32; N + 1];
            for (k, slot) in c.iter_mut().enumerate() {
                *slot = sign(diff(k as f64 / N as f64)?);
            }
            for k in 1..N - 1 {
                let (c0, c1) = (c[k], c[k + 1]);
                if c0 == c1 {
                    continue;
                }
                // A change of sign, or the start or end of a stretch
                // where the two are equal (two samples at least).
                let stretch = (c0 == 0 && c[k - 1] == 0) || (c1 == 0 && c[k + 2] == 0);
                if !(c0 * c1 < 0 || stretch) {
                    continue;
                }
                let (mut lo, mut hi) = (k as f64 / N as f64, (k + 1) as f64 / N as f64);
                for _ in 0..48 {
                    let mid = 0.5 * (lo + hi);
                    if sign(diff(mid)?) == c0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                cuts.push(0.5 * (lo + hi));
            }
        }
        cuts.sort_by(|a, b| a.total_cmp(b));
        let len = atom.curve.length();
        cuts.dedup_by(|a, b| (*a - *b).abs() * len <= self.map.tol.snap);
        Ok(cuts)
    }

    /// The curve in space over `geom` on `sheet`, running with `geom`.
    fn lift_curve(&self, geom: &Geom, sheet: usize) -> Result<(Curve3, f64, f64), String> {
        let s = &self.map.sheets[sheet];
        let up = |p: P2| -> Result<Vec3, String> { Ok([p[0], p[1], self.z(sheet, p)?]) };
        if let Some((_, gx, gy)) = s.as_plane() {
            match *geom {
                Geom::Line { a, b } => {
                    let (p, q) = (up(a)?, up(b)?);
                    return Ok((Curve3::line_through(p, q), 0.0, norm(sub3(q, p))));
                }
                Geom::Arc { c, r, a0, sweep } => {
                    // Conjugate half-diameters over the plan's x and y,
                    // turned onto the ellipse's own axes.
                    let centre = up(c)?;
                    let (a, b) = ([r, 0.0, r * gx], [0.0, r, r * gy]);
                    let (aa, bb, ab) = (
                        a[0] * a[0] + a[2] * a[2],
                        b[1] * b[1] + b[2] * b[2],
                        a[2] * b[2],
                    );
                    let phi = 0.5 * (2.0 * ab).atan2(aa - bb);
                    let (cp, sp) = (phi.cos(), phi.sin());
                    let major = [
                        a[0] * cp + b[0] * sp,
                        a[1] * cp + b[1] * sp,
                        a[2] * cp + b[2] * sp,
                    ];
                    let minor = [
                        b[0] * cp - a[0] * sp,
                        b[1] * cp - a[1] * sp,
                        b[2] * cp - a[2] * sp,
                    ];
                    let (la, lb) = (norm(major), norm(minor));
                    let x = [major[0] / la, major[1] / la, major[2] / la];
                    let mut y = [minor[0] / lb, minor[1] / lb, minor[2] / lb];
                    // Against the clock the curve runs about the opposite
                    // axis.
                    let (t0, t1) = if sweep >= 0.0 {
                        (a0 - phi, a0 - phi + sweep)
                    } else {
                        y = [-y[0], -y[1], -y[2]];
                        (-(a0 - phi), -(a0 - phi) - sweep)
                    };
                    let z = brep_core::math::cross(x, y);
                    let frame = Frame {
                        origin: centre,
                        x,
                        y,
                        z,
                    };
                    let curve = if (la - lb).abs() <= 1e-15 * la {
                        Curve3::Circle { frame, radius: la }
                    } else {
                        Curve3::Ellipse {
                            frame,
                            a: la,
                            b: lb,
                        }
                    };
                    return Ok((curve, t0, t1));
                }
                Geom::Free { .. } => {}
            }
        }
        // Points of the curve, close enough together in space that
        // three of them give its direction, however steeply the sheet
        // rises.
        let mut pts: Vec<Vec3> = vec![up(geom.point(0.0))?];
        let mut stack: Vec<(f64, Vec3, usize)> = Vec::new();
        for k in (1..=16).rev() {
            let t = k as f64 / 16.0;
            stack.push((t, up(geom.point(t))?, 0));
        }
        let mut at = 0.0;
        while let Some(&(t, p, depth)) = stack.last() {
            let from = *pts.last().unwrap();
            let mid = 0.5 * (at + t);
            if norm(sub3(p, from)) > SPACING && depth < 20 {
                stack.push((mid, up(geom.point(mid))?, depth + 1));
            } else {
                pts.push(p);
                at = t;
                stack.pop();
            }
        }
        let (p, q) = (pts[0], pts[pts.len() - 1]);
        let len = norm(sub3(q, p));
        if len > 0.0 {
            let d = [
                (q[0] - p[0]) / len,
                (q[1] - p[1]) / len,
                (q[2] - p[2]) / len,
            ];
            if pts
                .iter()
                .all(|c| norm(brep_core::math::cross(sub3(*c, p), d)) <= EXACT)
            {
                return Ok((Curve3::line_through(p, q), 0.0, len));
            }
        }
        if let Some(conic) = conic_through(&pts, EXACT) {
            return Ok(conic);
        }
        let closed = norm(sub3(p, q)) <= EXACT;
        Ok((
            Curve3::Nurbs(spline_through(&pts, self.opts.fit, closed)),
            0.0,
            1.0,
        ))
    }

    /// The wall over `geom` and whether its normal points to the right of
    /// travel... it always does; `between` is the range of heights it
    /// has to cover.
    fn wall_surface(&self, geom: &Geom, between: (f64, f64)) -> (Surface, bool) {
        match geom {
            Geom::Line { a, .. } => {
                let t = geom.tangent(0.0);
                (
                    Surface::Plane {
                        frame: Frame {
                            origin: [a[0], a[1], 0.0],
                            x: [t[0], t[1], 0.0],
                            y: [0.0, 0.0, 1.0],
                            z: [t[1], -t[0], 0.0],
                        },
                    },
                    true,
                )
            }
            Geom::Arc { c, r, sweep, .. } => (
                Surface::Cylinder {
                    frame: Frame {
                        origin: [c[0], c[1], 0.0],
                        ..Frame::IDENTITY
                    },
                    radius: *r,
                },
                // The normal is away from the axis: to the right when
                // turning left.
                *sweep >= 0.0,
            ),
            Geom::Free { spline, .. } => {
                let (z0, z1) = (between.0 - 1e-3, between.1 + 1e-3);
                let mut ctrl = Vec::with_capacity(2 * spline.ctrl.len());
                for c in &spline.ctrl {
                    ctrl.push([c[0], c[1], z0, 1.0]);
                    ctrl.push([c[0], c[1], z1, 1.0]);
                }
                (
                    Surface::Nurbs(NurbsSurface {
                        degree_u: 3,
                        degree_v: 1,
                        nctrl_u: spline.ctrl.len(),
                        nctrl_v: 2,
                        knots_u: spline.knots.clone(),
                        knots_v: vec![z0, z0, z1, z1],
                        ctrl,
                    }),
                    true,
                )
            }
        }
    }
}

/// `(id, curve, from, to, cells on the left and right)`.
type Chain = (String, Geom, usize, usize, [Option<usize>; 2]);

/// Chain atoms into plan edges: through every vertex where one curve,
/// with the same cells on its sides, simply goes on.
fn chain(atoms: &[Atom], plan: &[P2], snap: f64, fit: f64) -> Vec<Chain> {
    let mut at: Vec<Vec<(usize, bool)>> = vec![Vec::new(); plan.len()];
    for (i, a) in atoms.iter().enumerate() {
        at[a.from].push((i, false));
        at[a.to].push((i, true));
    }
    // An atom as travelled: forwards or backwards.
    let ends = |i: usize, fwd: bool| {
        if fwd {
            (atoms[i].from, atoms[i].to)
        } else {
            (atoms[i].to, atoms[i].from)
        }
    };
    let sides = |i: usize, fwd: bool| {
        let c = atoms[i].cells;
        if fwd { c } else { [c[1], c[0]] }
    };
    let curve = |i: usize, fwd: bool| {
        if fwd {
            atoms[i].curve
        } else {
            atoms[i].curve.reversed()
        }
    };
    // The atom that goes on from `(i, fwd)` through its end.
    let next = |i: usize, fwd: bool| -> Option<(usize, bool)> {
        let v = ends(i, fwd).1;
        if at[v].len() != 2 {
            return None;
        }
        let &(j, j_ends_here) = at[v].iter().find(|&&(j, e)| (j, e) != (i, fwd))?;
        if j == i {
            return None;
        }
        let jf = !j_ends_here;
        if atoms[j].id != atoms[i].id || sides(j, jf) != sides(i, fwd) {
            return None;
        }
        let ok = atoms[i].id.starts_with(FREE)
            || match (curve(i, fwd), curve(j, jf)) {
                (Curve::Line { .. }, Curve::Line { .. }) => {
                    let (a, b) = (curve(i, fwd).tangent(1.0), curve(j, jf).tangent(0.0));
                    cross(a, b).abs() < 1e-9 && dot(a, b) > 0.0
                }
                (
                    Curve::Arc {
                        c: c1,
                        r: r1,
                        sweep: s1,
                        ..
                    },
                    Curve::Arc {
                        c: c2,
                        r: r2,
                        sweep: s2,
                        ..
                    },
                ) => dist(c1, c2) <= snap && (r1 - r2).abs() <= snap && (s1 > 0.0) == (s2 > 0.0),
                _ => false,
            };
        ok.then_some((j, jf))
    };

    let mut used = vec![false; atoms.len()];
    let mut out = Vec::new();
    for start in 0..atoms.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let mut run: Vec<(usize, bool)> = vec![(start, true)];
        let mut closed = false;
        loop {
            let &(i, f) = run.last().unwrap();
            match next(i, f) {
                Some((j, _)) if j == start => {
                    closed = true;
                    break;
                }
                Some((j, jf)) if !used[j] => {
                    used[j] = true;
                    run.push((j, jf));
                }
                _ => break,
            }
        }
        if !closed {
            // Backwards from the start: forwards along the reverse.
            let mut back: Vec<(usize, bool)> = Vec::new();
            let (mut i, mut f) = (start, false);
            while let Some((j, jf)) = next(i, f) {
                if used[j] {
                    break;
                }
                used[j] = true;
                back.push((j, !jf));
                (i, f) = (j, jf);
            }
            back.reverse();
            back.extend(run);
            run = back;
        }
        let (first, last) = (run[0], run[run.len() - 1]);
        let (v0, v1) = (ends(first.0, first.1).0, ends(last.0, last.1).1);
        let id = atoms[first.0].id.clone();
        let geom = if id.starts_with(FREE) {
            let mut pts = vec![plan[v0]];
            for &(i, f) in &run {
                pts.push(plan[ends(i, f).1]);
            }
            let d = sub(plan[v1], plan[v0]);
            let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
            let straight = v0 != v1
                && pts
                    .iter()
                    .all(|p| (cross(d, sub(*p, plan[v0])) / len).abs() <= 1e-10);
            if straight {
                Geom::Line {
                    a: plan[v0],
                    b: plan[v1],
                }
            } else {
                let p3: Vec<Vec3> = pts.iter().map(|p| [p[0], p[1], 0.0]).collect();
                Geom::Free {
                    spline: spline_through(&p3, fit, v0 == v1),
                    pts,
                }
            }
        } else {
            match curve(first.0, first.1) {
                Curve::Line { .. } => Geom::Line {
                    a: plan[v0],
                    b: plan[v1],
                },
                Curve::Arc { c, r, a0, .. } => {
                    let mut sweep: f64 = run
                        .iter()
                        .map(|&(i, f)| match curve(i, f) {
                            Curve::Arc { sweep, .. } => sweep,
                            _ => 0.0,
                        })
                        .sum();
                    if sweep.abs() >= TAU - 1e-9 {
                        sweep = TAU * sweep.signum();
                    }
                    Geom::Arc { c, r, a0, sweep }
                }
            }
        };
        out.push((id, geom, v0, v1, sides(first.0, first.1)));
    }
    out
}

/// Signed area of a closed polygon.
fn area(poly: &[P2]) -> f64 {
    let mut a = 0.0;
    for i in 0..poly.len() {
        a += cross(poly[i], poly[(i + 1) % poly.len()]);
    }
    0.5 * a
}

fn inside(poly: &[P2], q: P2) -> bool {
    let mut odd = false;
    let mut prev = poly[poly.len() - 1];
    for &cur in poly {
        if (prev[1] <= q[1]) != (cur[1] <= q[1]) {
            let t = (q[1] - prev[1]) / (cur[1] - prev[1]);
            if prev[0] + t * (cur[0] - prev[0]) > q[0] {
                odd = !odd;
            }
        }
        prev = cur;
    }
    odd
}

/// A use of a level edge by a face on a sheet, in the direction that has
/// the cell on its left in the plan.
struct Use {
    edge: usize,
    plan_edge: usize,
    forward: bool,
}

/// The solids of `map`: one per connected lump.
pub fn lift(map: &PlanMap, opts: &LiftOptions) -> Result<Vec<Solid>, String> {
    if map.cells.is_empty() {
        return Err("the plan map holds nothing".into());
    }
    let snap = map.tol.snap;
    let tol = map.tol.height;

    // The plan as one arrangement.
    let mut pieces: Vec<Piece> = Vec::new();
    for c in &map.cells {
        pieces.extend(c.region.pieces().cloned());
    }
    let arr = arrangement(&pieces, snap);
    let cell_of: Vec<Option<usize>> = arr
        .faces
        .iter()
        .map(|f| deep_point(f).and_then(|p| map.cell_at(p)))
        .collect();
    let mut b = Builder {
        map,
        opts: *opts,
        plan: arr.vertices.clone(),
        solid: Solid::default(),
        over: HashMap::new(),
    };

    // Boundary atoms, cut where the order of heights changes.
    let column = |c: Option<usize>| c.map(|c| (map.cells[c].top, map.cells[c].bot));
    let mut atoms: Vec<Atom> = Vec::new();
    for e in &arr.edges {
        let cells = [
            e.left.and_then(|f| cell_of[f]),
            e.right.and_then(|f| cell_of[f]),
        ];
        if column(cells[0]) == column(cells[1]) {
            continue;
        }
        let atom = Atom {
            id: e.id.clone(),
            curve: e.curve,
            from: e.from,
            to: e.to,
            cells,
        };
        let cuts = b.relation_changes(&atom)?;
        if cuts.is_empty() {
            atoms.push(atom);
            continue;
        }
        let mut stops = vec![0.0];
        stops.extend(cuts.iter().filter(|t| **t > 1e-9 && **t < 1.0 - 1e-9));
        stops.push(1.0);
        let mut from = atom.from;
        for w in stops.windows(2) {
            let curve = atom.curve.sub(w[0], w[1]);
            let to = if w[1] == 1.0 {
                atom.to
            } else {
                b.plan.push(curve.end());
                b.plan.len() - 1
            };
            atoms.push(Atom {
                id: atom.id.clone(),
                curve,
                from,
                to,
                cells,
            });
            from = to;
        }
    }

    // Plan edges and their levels.
    let mut edges: Vec<PlanEdge> = Vec::new();
    for (id, geom, v0, v1, cells) in chain(&atoms, &b.plan, snap, opts.fit) {
        let sheets = b.sheets_of(cells);
        let mut entries: Vec<(Entry, [f64; 3])> = Vec::new();
        for (side, s) in sheets.iter().enumerate() {
            let Some(s) = s else { continue };
            for (top, sheet) in [(false, s[0]), (true, s[1])] {
                let mut z = [0.0; 3];
                for (k, t) in [0.25, 0.5, 0.75].into_iter().enumerate() {
                    z[k] = b.z(sheet, geom.point(t))?;
                }
                entries.push((Entry { side, top, sheet }, z));
            }
        }
        entries.sort_by(|x, y| x.1[1].total_cmp(&y.1[1]));
        let mut levels: Vec<(Level, [f64; 3])> = Vec::new();
        for (entry, z) in entries {
            match levels.last_mut() {
                Some((level, at)) if (0..3).all(|k| (at[k] - z[k]).abs() <= tol) => {
                    level.entries.push(entry)
                }
                _ => levels.push((
                    Level {
                        entries: vec![entry],
                        edge: None,
                    },
                    z,
                )),
            }
        }
        let levels: Vec<Level> = levels.into_iter().map(|(l, _)| l).collect();
        // Material per side in each gap: between its bottom's level and
        // its top's.
        let level_of = |side: usize, top: bool| {
            levels
                .iter()
                .position(|l| l.entries.iter().any(|e| e.side == side && e.top == top))
        };
        let mut walls = Vec::new();
        for gap in 0..levels.len().saturating_sub(1) {
            let has = |side: usize| match (level_of(side, false), level_of(side, true)) {
                (Some(lo), Some(hi)) => lo <= gap && hi > gap,
                _ => false,
            };
            walls.push(match (has(0), has(1)) {
                (true, false) => Some(0),
                (false, true) => Some(1),
                _ => None,
            });
        }
        edges.push(PlanEdge {
            id,
            geom,
            v0,
            v1,
            levels,
            walls,
        });
    }

    // Level edges: where exactly two faces end.
    let mut uses: HashMap<(bool, usize), Vec<Use>> = HashMap::new();
    for (ei, e) in edges.iter_mut().enumerate() {
        for li in 0..e.levels.len() {
            let entries = e.levels[li].entries.clone();
            // A face that simply goes on across the edge does not end.
            let faces: Vec<Entry> = entries
                .iter()
                .filter(|x| {
                    !entries
                        .iter()
                        .any(|y| y.side != x.side && y.top == x.top && y.sheet == x.sheet)
                })
                .copied()
                .collect();
            let below = li > 0 && e.walls[li - 1].is_some();
            let above = li < e.walls.len() && e.walls[li].is_some();
            let ending = faces.len() + usize::from(below) + usize::from(above);
            if ending == 0 {
                continue;
            }
            if ending != 2 {
                let p = e.geom.point(0.5);
                return Err(format!(
                    "{ending} faces end on one edge over '{}' near ({:.6}, {:.6}): the solid touches itself there",
                    e.id, p[0], p[1]
                ));
            }
            // Planes give exact curves: follow one if there is one.
            let sheet = entries
                .iter()
                .map(|x| x.sheet)
                .find(|s| map.sheets[*s].as_plane().is_some())
                .unwrap_or(entries[0].sheet);
            let (curve, t0, t1) = b.lift_curve(&e.geom, sheet)?;
            let (z0, z1) = (b.z(sheet, b.plan[e.v0])?, b.z(sheet, b.plan[e.v1])?);
            let (start, end) = (b.vertex(e.v0, z0), b.vertex(e.v1, z1));
            b.solid.edges.push(Edge {
                curve,
                t0,
                t1,
                start,
                end,
            });
            let edge = b.solid.edges.len() - 1;
            e.levels[li].edge = Some(edge);
            for x in faces {
                uses.entry((x.top, x.sheet)).or_default().push(Use {
                    edge,
                    plan_edge: ei,
                    forward: x.side == 0,
                });
            }
        }
    }

    // Vertical edges: over each plan vertex, between consecutive heights,
    // where two wall ends cover the stretch.
    struct WallEnd {
        edge: usize,
        gap: usize,
        lo: f64,
        hi: f64,
    }
    let mut ends_at: HashMap<usize, Vec<WallEnd>> = HashMap::new();
    for (ei, e) in edges.iter().enumerate() {
        for (gap, wall) in e.walls.iter().enumerate() {
            if wall.is_none() {
                continue;
            }
            for v in [e.v0, e.v1] {
                let height = |level: &Level| -> Result<f64, String> {
                    b.z(level.entries[0].sheet, b.plan[v])
                };
                ends_at.entry(v).or_default().push(WallEnd {
                    edge: ei,
                    gap,
                    lo: height(&e.levels[gap])?,
                    hi: height(&e.levels[gap + 1])?,
                });
            }
        }
    }
    // Per (plan vertex, index of the lower height): the vertical edge.
    let mut vertical: HashMap<(usize, usize), usize> = HashMap::new();
    let vtol = 10.0 * tol;
    for (&v, ends) in &ends_at {
        let mut heights = b.over.get(&v).cloned().unwrap_or_default();
        heights.sort_by(|x, y| x.0.total_cmp(&y.0));
        b.over.insert(v, heights.clone());
        for (j, w) in heights.windows(2).enumerate() {
            let covering: Vec<&WallEnd> = ends
                .iter()
                .filter(|e| e.lo <= w[0].0 + vtol && e.hi >= w[1].0 - vtol)
                .collect();
            match covering.len() {
                0 => continue,
                2 => {
                    // A circle's wall meets itself without a seam.
                    let same =
                        covering[0].edge == covering[1].edge && covering[0].gap == covering[1].gap;
                    if same && edges[covering[0].edge].geom.closed_circle() {
                        continue;
                    }
                }
                n => {
                    let p = b.plan[v];
                    return Err(format!(
                        "{n} walls meet on one vertical over ({:.6}, {:.6}): the solid touches itself there",
                        p[0], p[1]
                    ));
                }
            }
            let (lo, hi) = (b.solid.vertices[w[0].1], b.solid.vertices[w[1].1]);
            b.solid.edges.push(Edge {
                curve: Curve3::line_through(lo, hi),
                t0: 0.0,
                t1: hi[2] - lo[2],
                start: w[0].1,
                end: w[1].1,
            });
            vertical.insert((v, j), b.solid.edges.len() - 1);
        }
    }
    // The vertical edges of a wall's end, from the bottom up.
    let rise = |b: &Builder, v: usize, lo: f64, hi: f64| -> Vec<usize> {
        let Some(heights) = b.over.get(&v) else {
            return Vec::new();
        };
        (0..heights.len().saturating_sub(1))
            .filter(|&j| heights[j].0 >= lo - vtol && heights[j + 1].0 <= hi + vtol)
            .filter_map(|j| vertical.get(&(v, j)).copied())
            .collect()
    };

    // Walls.
    for e in &edges {
        for (gap, wall) in e.walls.iter().enumerate() {
            let Some(side) = *wall else { continue };
            let (Some(lower), Some(upper)) = (e.levels[gap].edge, e.levels[gap + 1].edge) else {
                return Err(format!("a wall over '{}' has no edge to stand on", e.id));
            };
            let at = |level: &Level, v: usize| b.z(level.entries[0].sheet, b.plan[v]);
            let (lo0, hi0) = (at(&e.levels[gap], e.v0)?, at(&e.levels[gap + 1], e.v0)?);
            let (lo1, hi1) = (at(&e.levels[gap], e.v1)?, at(&e.levels[gap + 1], e.v1)?);
            let mut reach = (lo0.min(lo1), hi0.max(hi1));
            for t in [0.25, 0.5, 0.75] {
                let p = e.geom.point(t);
                reach.0 = reach.0.min(b.z(e.levels[gap].entries[0].sheet, p)?);
                reach.1 = reach.1.max(b.z(e.levels[gap + 1].entries[0].sheet, p)?);
            }
            let (surface, right) = b.wall_surface(&e.geom, reach);
            let fwd = |edge: usize| Coedge {
                edge,
                forward: true,
            };
            let rev = |edge: usize| Coedge {
                edge,
                forward: false,
            };
            // With the material on the left.
            let mut loops: Vec<Vec<Coedge>> = if e.v0 == e.v1 && e.geom.closed_circle() {
                vec![vec![fwd(lower)], vec![rev(upper)]]
            } else {
                let mut lp = vec![fwd(lower)];
                lp.extend(rise(&b, e.v1, lo1, hi1).into_iter().map(fwd));
                lp.push(rev(upper));
                lp.extend(rise(&b, e.v0, lo0, hi0).into_iter().rev().map(rev));
                vec![lp]
            };
            if side == 1 {
                for lp in loops.iter_mut() {
                    lp.reverse();
                    for c in lp.iter_mut() {
                        c.forward = !c.forward;
                    }
                }
            }
            // Out of the material is to the right of travel when the
            // material is on the left.
            b.solid.faces.push(Face {
                surface,
                same_sense: right == (side == 0),
                loops,
                name: format!("wall:{}", e.id),
                color: None,
            });
        }
    }

    // Faces on the sheets.
    let mut keys: Vec<(bool, usize)> = uses.keys().copied().collect();
    keys.sort();
    for key in keys {
        let (top, sheet) = key;
        let list = &uses[&key];
        let start_of = |u: &Use| {
            let e = &b.solid.edges[u.edge];
            if u.forward {
                (e.start, e.end)
            } else {
                (e.end, e.start)
            }
        };
        let leaving = |u: &Use| {
            let g = &edges[u.plan_edge].geom;
            if u.forward {
                g.tangent(0.0)
            } else {
                scale(g.tangent(1.0), -1.0)
            }
        };
        let back = |u: &Use| {
            let g = &edges[u.plan_edge].geom;
            if u.forward {
                scale(g.tangent(1.0), -1.0)
            } else {
                g.tangent(0.0)
            }
        };
        let mut used = vec![false; list.len()];
        let mut loops: Vec<(Vec<Coedge>, Vec<P2>)> = Vec::new();
        for first in 0..list.len() {
            if used[first] {
                continue;
            }
            let mut lp: Vec<usize> = vec![first];
            used[first] = true;
            loop {
                let cur = &list[*lp.last().unwrap()];
                let here = start_of(cur).1;
                // The first way out clockwise from the way back in: the
                // face is on the left. Back out along the first edge
                // closes the loop.
                let from = back(cur);
                let turn = |u: &Use| {
                    let d = leaving(u);
                    let a = (-cross(from, d)).atan2(dot(from, d));
                    if a <= 1e-9 { a + TAU } else { a }
                };
                let next = (0..list.len())
                    .filter(|&k| (!used[k] || k == first) && start_of(&list[k]).0 == here)
                    .min_by(|&x, &y| turn(&list[x]).total_cmp(&turn(&list[y])));
                match next {
                    Some(k) if k == first => break,
                    Some(k) => {
                        used[k] = true;
                        lp.push(k);
                    }
                    None => {
                        let p = b.solid.vertices[here];
                        return Err(format!(
                            "the boundary of '{}' does not close at ({:.6}, {:.6}, {:.6})",
                            map.sheets[sheet].name, p[0], p[1], p[2]
                        ));
                    }
                }
            }
            let mut poly: Vec<P2> = Vec::new();
            for &k in &lp {
                let mut pts = edges[list[k].plan_edge].geom.flatten();
                if !list[k].forward {
                    pts.reverse();
                }
                pts.pop();
                poly.extend(pts);
            }
            let coedges = lp
                .iter()
                .map(|&k| Coedge {
                    edge: list[k].edge,
                    forward: list[k].forward,
                })
                .collect();
            loops.push((coedges, poly));
        }
        // Outer loops each bound a face; holes go to the smallest outer
        // loop around them.
        let outers: Vec<usize> = (0..loops.len())
            .filter(|&i| area(&loops[i].1) > 0.0)
            .collect();
        let mut faces: Vec<Vec<Vec<Coedge>>> =
            outers.iter().map(|&i| vec![loops[i].0.clone()]).collect();
        for (i, (coedges, poly)) in loops.iter().enumerate() {
            if outers.contains(&i) {
                continue;
            }
            let probe = poly[0];
            let host = outers
                .iter()
                .enumerate()
                .filter(|&(_, &o)| inside(&loops[o].1, probe))
                .min_by(|x, y| area(&loops[*x.1].1).total_cmp(&area(&loops[*y.1].1)))
                .map(|(slot, _)| slot);
            match host {
                Some(slot) => faces[slot].push(coedges.clone()),
                None => {
                    return Err(format!(
                        "a hole of '{}' near ({:.6}, {:.6}) is in no face",
                        map.sheets[sheet].name, probe[0], probe[1]
                    ));
                }
            }
        }
        for mut loops in faces {
            if !top {
                for lp in loops.iter_mut() {
                    lp.reverse();
                    for c in lp.iter_mut() {
                        c.forward = !c.forward;
                    }
                }
            }
            b.solid.faces.push(Face {
                surface: map.sheets[sheet].surface.clone(),
                same_sense: top == map.sheets[sheet].up,
                loops,
                name: map.sheets[sheet].name.clone(),
                color: None,
            });
        }
    }

    Ok(lumps(b.solid))
}

/// The connected parts of a solid's faces, each as a solid of its own.
fn lumps(all: Solid) -> Vec<Solid> {
    let mut group: Vec<usize> = (0..all.faces.len()).collect();
    fn root(group: &mut [usize], mut i: usize) -> usize {
        while group[i] != i {
            group[i] = group[group[i]];
            i = group[i];
        }
        i
    }
    let mut first_use: Vec<Option<usize>> = vec![None; all.edges.len()];
    for (fi, f) in all.faces.iter().enumerate() {
        for c in f.loops.iter().flatten() {
            match first_use[c.edge] {
                None => first_use[c.edge] = Some(fi),
                Some(other) => {
                    let (a, b) = (root(&mut group, fi), root(&mut group, other));
                    group[a] = b;
                }
            }
        }
    }
    let mut out: Vec<Solid> = Vec::new();
    let mut slot: HashMap<usize, usize> = HashMap::new();
    let mut edge_map: Vec<HashMap<usize, usize>> = Vec::new();
    let mut vertex_map: Vec<HashMap<usize, usize>> = Vec::new();
    for fi in 0..all.faces.len() {
        let r = root(&mut group, fi);
        let s = *slot.entry(r).or_insert_with(|| {
            out.push(Solid::default());
            edge_map.push(HashMap::new());
            vertex_map.push(HashMap::new());
            out.len() - 1
        });
        let mut face = all.faces[fi].clone();
        for c in face.loops.iter_mut().flatten() {
            let next = out[s].edges.len();
            let e = *edge_map[s].entry(c.edge).or_insert(next);
            if e == next {
                let mut edge = all.edges[c.edge].clone();
                for v in [&mut edge.start, &mut edge.end] {
                    let nv = out[s].vertices.len();
                    let mapped = *vertex_map[s].entry(*v).or_insert(nv);
                    if mapped == nv {
                        out[s].vertices.push(all.vertices[*v]);
                    }
                    *v = mapped;
                }
                out[s].edges.push(edge);
            }
            c.edge = e;
        }
        out[s].faces.push(face);
    }
    out
}
