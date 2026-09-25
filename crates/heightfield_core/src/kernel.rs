//! Band-limited grey-scale morphology on a height field: rolling-ball
//! fillets and cone chamfers along a selected edge.
//!
//! With `k(d)` the kernel's elevation at horizontal distance `d` (a sphere
//! cap for a fillet, a cone for a chamfer), the ball resting on the surface
//! at column `c` has its centre at `D(c) = max_p h(p) + k(|c - p|)`, and the
//! surface swept from below by all resting balls is `min_c D(c) - k(|c -
//! p|)`: a closing, which fills concave edges. The opening (erode, then
//! dilate) trims convex edges the same way. Both are computed only around
//! the selected edge cells and written back only there, which is what
//! restricts the blend to that edge.
//!
//! Empty columns count as no material at any height, so a convex blend at
//! the footprint rim rounds the top into the side wall, and a blend never
//! adds material outside the footprint.

use crate::{EdgeCells, Side, Solid, edt};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kernel {
    /// Rolling ball of radius `r`.
    Ball { r: f64 },
    /// Cone with horizontal reach `radius` and apex height `height`: a
    /// chamfer with those two offsets along the cap and along the wall.
    Cone { radius: f64, height: f64 },
}

impl Kernel {
    pub fn radius(&self) -> f64 {
        match *self {
            Kernel::Ball { r } => r,
            Kernel::Cone { radius, .. } => radius,
        }
    }
    /// Elevation at horizontal distance `d` (`d <= radius`).
    #[inline]
    pub fn k(&self, d: f64) -> f64 {
        match *self {
            Kernel::Ball { r } => (r * r - d * d).max(0.0).sqrt(),
            Kernel::Cone { radius, height } => height * (1.0 - d / radius).max(0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BlendReport {
    pub cells_changed: usize,
    pub max_change: f64,
    /// Band cells where the opening left no material (a feature thinner
    /// than the ball); they keep their height.
    pub vanished: usize,
    pub edge_cells: usize,
}

/// Options for [`blend`].
#[derive(Clone, Copy, Debug)]
pub struct BlendOptions {
    /// Cells are written within `band_factor * radius` of the edge.
    pub band_factor: f64,
    /// Changes smaller than this are ignored.
    pub min_change: f64,
}

impl Default for BlendOptions {
    fn default() -> Self {
        BlendOptions { band_factor: 1.5, min_change: 1e-7 }
    }
}

struct Disc {
    offsets: Vec<(isize, isize, f64)>, // di, dj, k(d)
}

fn disc(kernel: &Kernel, pitch: f64) -> Disc {
    let rc = (kernel.radius() / pitch).ceil() as isize;
    let mut offsets = Vec::new();
    for dj in -rc..=rc {
        for di in -rc..=rc {
            let d = ((di * di + dj * dj) as f64).sqrt() * pitch;
            // Offsets at the kernel's rim add nothing but the raw height and
            // would make the resting height jump exactly where a ball touches
            // a wall's top corner; the limit from outside is the right value.
            if d < kernel.radius() {
                offsets.push((di, dj, kernel.k(d)));
            }
        }
    }
    Disc { offsets }
}

/// Blend the edge `edge` on `side` with `kernel`. `convex` selects the
/// opening (material removed) over the closing (material added).
pub fn blend(
    solid: &mut Solid,
    side: Side,
    edge: &EdgeCells,
    kernel: &Kernel,
    convex: bool,
    label: u32,
    order: u32,
    opts: BlendOptions,
) -> BlendReport {
    let mut rep = BlendReport { edge_cells: edge.cells.len(), ..Default::default() };
    if edge.cells.is_empty() {
        return rep;
    }
    let g = solid.grid.clone();
    let (nx, ny) = (g.nx, g.ny);
    let sign = if side == Side::Top { 1.0f32 } else { -1.0 };
    // The field as "top" values: negate the bottom so the same code serves.
    let h: Vec<f32> = match side {
        Side::Top => solid.top.clone(),
        Side::Bottom => solid.bot.iter().map(|v| -v).collect(),
    };
    let mut is_edge = vec![false; g.len()];
    for &k in &edge.cells {
        is_edge[k] = true;
    }
    let (d2, _) = edt::distance_to_sites(nx, ny, |k| is_edge[k]);
    let rc = kernel.radius() / g.pitch;
    let write_r = rc * opts.band_factor + 1.0;
    let centre_r = write_r + rc + 1.0;
    let dsc = disc(kernel, g.pitch);

    // Pass 1: the resting-ball centre heights (dilation for a closing,
    // erosion for an opening) at every cell within reach of the band.
    let first = |k: usize| -> f32 {
        let (i, j) = g.ij(k);
        let mut acc = if convex { f32::INFINITY } else { f32::NEG_INFINITY };
        let mut any = false;
        for &(di, dj, kv) in &dsc.offsets {
            let (ii, jj) = (i as isize + di, j as isize + dj);
            let v = if ii < 0 || jj < 0 || ii >= nx as isize || jj >= ny as isize {
                f32::NAN
            } else {
                h[g.idx(ii as usize, jj as usize)]
            };
            if convex {
                // The ball must fit under the surface everywhere it reaches:
                // an empty column means it cannot rest here at all.
                if v.is_nan() {
                    return f32::NEG_INFINITY;
                }
                acc = acc.min(v - kv as f32);
                any = true;
            } else if !v.is_nan() {
                acc = acc.max(v + kv as f32);
                any = true;
            }
        }
        if any { acc } else { f32::NAN }
    };
    let centre_cells: Vec<usize> = (0..g.len()).filter(|&k| d2[k].sqrt() <= centre_r).collect();
    let centre_vals: Vec<f32> = map_par(&centre_cells, |&k| first(k));
    let mut stage = vec![f32::NAN; g.len()];
    for (k, v) in centre_cells.iter().zip(centre_vals) {
        stage[*k] = v;
    }

    // Pass 2: the swept surface at the band cells.
    let second = |k: usize| -> f32 {
        if h[k].is_nan() {
            return f32::NAN;
        }
        let (i, j) = g.ij(k);
        let mut acc = if convex { f32::NEG_INFINITY } else { f32::INFINITY };
        for &(di, dj, kv) in &dsc.offsets {
            let (ii, jj) = (i as isize + di, j as isize + dj);
            if ii < 0 || jj < 0 || ii >= nx as isize || jj >= ny as isize {
                continue;
            }
            let c = stage[g.idx(ii as usize, jj as usize)];
            if c.is_nan() {
                continue;
            }
            if convex {
                if c.is_finite() {
                    acc = acc.max(c + kv as f32);
                }
            } else {
                acc = acc.min(c - kv as f32);
            }
        }
        acc
    };
    let band_cells: Vec<usize> = (0..g.len()).filter(|&k| d2[k].sqrt() <= write_r && !h[k].is_nan()).collect();
    let band_vals: Vec<f32> = map_par(&band_cells, |&k| second(k));

    for (&k, &v) in band_cells.iter().zip(band_vals.iter()) {
        if !v.is_finite() {
            if convex && v == f32::NEG_INFINITY {
                rep.vanished += 1;
            }
            continue;
        }
        let delta = v - h[k];
        // A closing only raises, an opening only lowers; anything else is
        // numerical noise from the discrete disc.
        let delta = if convex { delta.min(0.0) } else { delta.max(0.0) };
        if delta.abs() as f64 <= opts.min_change {
            continue;
        }
        let new = h[k] + delta;
        match side {
            Side::Top => {
                solid.top[k] = new * sign;
                solid.top_label[k] = label;
                solid.top_order[k] = order;
            }
            Side::Bottom => {
                solid.bot[k] = -new;
                solid.bot_label[k] = label;
                solid.bot_order[k] = order;
            }
        }
        rep.cells_changed += 1;
        rep.max_change = rep.max_change.max(delta.abs() as f64);
    }
    rep
}

#[cfg(feature = "parallel")]
fn map_par<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
    use rayon::prelude::*;
    items.par_iter().map(f).collect()
}

#[cfg(not(feature = "parallel"))]
fn map_par<T, R>(items: &[T], f: impl Fn(&T) -> R) -> Vec<R> {
    items.iter().map(f).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FaceSel, Grid, Op, Prism};
    use sketch_regions_core::{Curve, Piece, arrange};

    fn rect_region(x0: f64, y0: f64, x1: f64, y1: f64) -> sketch_regions_core::Region {
        let p = |a, b| Piece { id: "r".into(), curve: Curve::Line { a, b } };
        let pieces = vec![p([x0, y0], [x1, y0]), p([x1, y0], [x1, y1]), p([x1, y1], [x0, y1]), p([x0, y1], [x0, y0])];
        arrange(&pieces, 1e-9).remove(0)
    }

    fn slab_with_step(pitch: f64) -> (Solid, Grid) {
        let grid = Grid::covering(0.0, 0.0, 20.0, 10.0, pitch, 4);
        let mut s = Solid::empty(grid.clone());
        let slab = Prism {
            region: rect_region(0.0, 0.0, 20.0, 10.0),
            origin: [0.0; 3],
            u: [1.0, 0.0, 0.0],
            v: [0.0, 1.0, 0.0],
            dir: [0.0, 0.0, 1.0],
            w0: 0.0,
            w1: 4.0,
        };
        s.extrude(&slab, Op::Add, 1, 1);
        // A step: the right half is 3 units higher.
        let step = Prism { region: rect_region(10.0, 0.0, 20.0, 10.0), w0: 4.0, w1: 7.0, ..slab };
        s.extrude(&step, Op::Add, 2, 2);
        (s, grid)
    }

    /// Largest distance from the sampled profile points `(x, z)` to the
    /// ideal profile curve (given densely as points), in the x-z plane.
    fn profile_error(samples: &[(f64, f64)], ideal: &[(f64, f64)]) -> f64 {
        let mut worst: f64 = 0.0;
        for &(x, z) in samples {
            let d = ideal.iter().map(|&(ix, iz)| ((x - ix).powi(2) + (z - iz).powi(2)).sqrt()).fold(f64::INFINITY, f64::min);
            worst = worst.max(d);
        }
        worst
    }

    fn row(s: &Solid, grid: &Grid, y: f64, x0: f64, x1: f64, field: Side) -> Vec<(f64, f64)> {
        let j = ((y - grid.y0) / grid.pitch) as usize;
        (0..grid.nx)
            .map(|i| (grid.x(i), i))
            .filter(|(x, _)| *x >= x0 && *x <= x1)
            .map(|(x, i)| {
                let k = grid.idx(i, j);
                (x, if field == Side::Top { s.top[k] } else { s.bot[k] } as f64)
            })
            .collect()
    }

    #[test]
    fn concave_fillet_matches_circle_profile() {
        let (mut s, grid) = slab_with_step(0.1);
        let r = 2.0;
        let edge = s.edge_cells(Side::Top, FaceSel::Cap(1), FaceSel::Wall(2), 0.01);
        assert!(!edge.convex);
        let rep = blend(&mut s, Side::Top, &edge, &Kernel::Ball { r }, false, 9, 3, BlendOptions::default());
        assert!(rep.cells_changed > 0);
        // Ideal: floor at 4 up to x = 8, arc centred (8, 6) to the wall at (10, 6), wall up to 7.
        let mut ideal: Vec<(f64, f64)> = (0..2000).map(|q| (q as f64 * 0.005, 4.0)).collect();
        ideal.extend((0..=2000).map(|q| {
            let a = -std::f64::consts::FRAC_PI_2 + std::f64::consts::FRAC_PI_2 * q as f64 / 2000.0;
            (8.0 + r * a.cos(), 6.0 + r * a.sin())
        }));
        ideal.extend((0..=200).map(|q| (10.0, 6.0 + q as f64 * 0.005)));
        let err = profile_error(&row(&s, &grid, 5.0, 0.5, 9.99, Side::Top), &ideal);
        assert!(err < 0.15, "profile error {err}");
        // Nothing changed on the high side.
        let j = ((5.0 - grid.y0) / grid.pitch) as usize;
        let k = grid.idx(((15.0 - grid.x0) / grid.pitch) as usize, j);
        assert_eq!(s.top[k], 7.0);
    }

    #[test]
    fn convex_fillet_and_rim() {
        let (mut s, grid) = slab_with_step(0.1);
        let r = 1.5;
        let edge = s.edge_cells(Side::Top, FaceSel::Cap(2), FaceSel::Wall(2), 0.01);
        assert!(edge.convex);
        blend(&mut s, Side::Top, &edge, &Kernel::Ball { r }, true, 9, 3, BlendOptions::default());
        // Ideal: wall up to (10, 7 - r), arc centred (10 + r, 7 - r), top at 7 beyond.
        let mut ideal: Vec<(f64, f64)> = (0..=600).map(|q| (10.0, 4.0 + q as f64 * 0.005)).collect();
        ideal.extend((0..=2000).map(|q| {
            let a = std::f64::consts::PI - std::f64::consts::FRAC_PI_2 * q as f64 / 2000.0;
            (10.0 + r + r * a.cos(), 7.0 - r + r * a.sin())
        }));
        ideal.extend((0..2000).map(|q| (11.5 + q as f64 * 0.005, 7.0)));
        let err = profile_error(&row(&s, &grid, 5.0, 10.01, 19.5, Side::Top), &ideal);
        assert!(err < 0.15, "profile error {err}");
        // The outer rim of the low half rounds into the footprint wall.
        let rim = s.edge_cells(Side::Top, FaceSel::Cap(1), FaceSel::Outline, 0.01);
        assert!(rim.convex);
        blend(&mut s, Side::Top, &rim, &Kernel::Ball { r: 1.0 }, true, 10, 4, BlendOptions::default());
        let i = ((5.0 - grid.x0) / grid.pitch) as usize;
        let col: Vec<(f64, f64)> = (0..grid.ny)
            .filter(|&j| grid.y(j) > 0.0 && grid.y(j) < 5.0)
            .map(|j| (grid.y(j), s.top[grid.idx(i, j)] as f64))
            .collect();
        let mut ideal: Vec<(f64, f64)> = (0..=600).map(|q| (0.0, q as f64 * 0.005)).collect();
        ideal.extend((0..=2000).map(|q| {
            let a = std::f64::consts::PI - std::f64::consts::FRAC_PI_2 * q as f64 / 2000.0;
            (1.0 + a.cos(), 3.0 + a.sin())
        }));
        ideal.extend((0..1000).map(|q| (1.0 + q as f64 * 0.005, 4.0)));
        let err = profile_error(&col, &ideal);
        assert!(err < 0.15, "rim profile error {err}");
        let jj = ((0.05 - grid.y0) / grid.pitch) as usize;
        assert!(s.top[grid.idx(i, jj - 1)].is_nan());
    }

    #[test]
    fn chamfer_is_a_plane() {
        let (mut s, grid) = slab_with_step(0.1);
        let edge = s.edge_cells(Side::Top, FaceSel::Cap(2), FaceSel::Wall(2), 0.01);
        blend(&mut s, Side::Top, &edge, &Kernel::Cone { radius: 2.0, height: 1.0 }, true, 9, 3, BlendOptions::default());
        let j = ((5.0 - grid.y0) / grid.pitch) as usize;
        for (x, expect) in [(10.25, 6.125), (11.0, 6.5), (12.0, 7.0), (14.0, 7.0)] {
            let i = ((x - grid.x0) / grid.pitch) as usize;
            let got = s.top[grid.idx(i, j)] as f64;
            assert!((got - expect).abs() < 0.06, "x {x}: {got} vs {expect}");
        }
    }

    #[test]
    fn bottom_side_blend() {
        let (mut s, grid) = slab_with_step(0.1);
        // Cut a groove into the bottom of the low half and round its floor edges.
        let groove = Prism {
            region: rect_region(3.0, 0.0, 6.0, 10.0),
            origin: [0.0; 3],
            u: [1.0, 0.0, 0.0],
            v: [0.0, 1.0, 0.0],
            dir: [0.0, 0.0, 1.0],
            w0: 0.0,
            w1: 1.0,
        };
        s.extrude(&groove, Op::Remove, 3, 3);
        let edge = s.edge_cells(Side::Bottom, FaceSel::Cap(3), FaceSel::Wall(3), 0.01);
        assert!(!edge.convex);
        let rep = blend(&mut s, Side::Bottom, &edge, &Kernel::Ball { r: 0.5 }, false, 9, 4, BlendOptions::default());
        assert!(rep.cells_changed > 0);
        // Ideal (x, bot): wall down from (3, 0) to (3, 0.5), arc centred (3.5, 0.5)
        // from (3, 0.5) to (3.5, 1.0), ceiling at 1 beyond.
        let mut ideal: Vec<(f64, f64)> = (0..=100).map(|q| (3.0, q as f64 * 0.005)).collect();
        ideal.extend((0..=2000).map(|q| {
            let a = std::f64::consts::PI - std::f64::consts::FRAC_PI_2 * q as f64 / 2000.0;
            (3.5 + 0.5 * a.cos(), 0.5 + 0.5 * a.sin())
        }));
        ideal.extend((0..200).map(|q| (3.5 + q as f64 * 0.005, 1.0)));
        let err = profile_error(&row(&s, &grid, 5.0, 3.01, 4.5, Side::Bottom), &ideal);
        assert!(err < 0.15, "profile error {err}");
    }
}
