//! Vertical-edge blends: rounding a corner seen in plan view. The corner
//! of a pocket, a boss, a through hole or the footprint is a vertical edge
//! of the solid; its fillet is a 2D rounding of the region's outline, which
//! on the grid is a morphological opening (convex corners) or closing
//! (concave corners) of the region's cell mask by a disc of the radius.
//! Cells that leave the region take the height and label of the nearest
//! cell outside it (the wall moves inward); cells that join it take those
//! of the nearest cell inside.

use crate::{Grid, Side, Solid, edt};

#[derive(Clone, Debug, Default)]
pub struct RoundReport {
    pub removed: usize,
    pub added: usize,
}

/// Round the corners of `mask` (true = inside the region) that lie in the
/// xy box `bbox` (`[x0, y0, x1, y1]`) with radius `r`. `empty_outside` says
/// the region is the footprint: cells leaving it are emptied.
pub fn round_mask(solid: &mut Solid, side: Side, mask: &[bool], r: f64, bbox: [f64; 4], empty_outside: bool, label: u32, order: u32) -> RoundReport {
    let g: Grid = solid.grid.clone();
    let rc = r / g.pitch;
    let margin = 2.0 * r + 4.0 * g.pitch;
    let (i0, i1, j0, j1) = g.cells_in([bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin]);
    let (wx, wy) = (i1.saturating_sub(i0), j1.saturating_sub(j0));
    let mut rep = RoundReport::default();
    if wx < 3 || wy < 3 {
        return rep;
    }
    let win = |i: usize, j: usize| g.idx(i0 + i, j0 + j);
    let m: Vec<bool> = (0..wx * wy).map(|q| mask[win(q % wx, q / wx)]).collect();
    // Opening: erode then dilate.
    let (d_out, _) = edt::distance_to_sites(wx, wy, |q| !m[q]);
    let eroded: Vec<bool> = (0..wx * wy).map(|q| m[q] && d_out[q].sqrt() - 0.5 >= rc).collect();
    let (d_er, _) = edt::distance_to_sites(wx, wy, |q| eroded[q]);
    let opened: Vec<bool> = (0..wx * wy).map(|q| d_er[q].sqrt() - 0.5 < rc || eroded[q]).collect();
    let mut removed: Vec<usize> = (0..wx * wy).filter(|&q| m[q] && !opened[q]).collect();
    // Only corners inside the box proper (the window margin catches the
    // disc's reach, not other corners).
    let in_box = |q: usize| {
        let (i, j) = (i0 + q % wx, j0 + q / wx);
        let (x, y) = (g.x(i), g.y(j));
        x >= bbox[0] - g.pitch && x <= bbox[2] + g.pitch && y >= bbox[1] - g.pitch && y <= bbox[3] + g.pitch
    };
    removed.retain(|&q| in_box(q));
    let mut added: Vec<usize> = Vec::new();
    if removed.is_empty() {
        // Closing: dilate then erode.
        let (d_in, _) = edt::distance_to_sites(wx, wy, |q| m[q]);
        let dilated: Vec<bool> = (0..wx * wy).map(|q| m[q] || d_in[q].sqrt() - 0.5 < rc).collect();
        let (d_dil_out, _) = edt::distance_to_sites(wx, wy, |q| !dilated[q]);
        let closed: Vec<bool> = (0..wx * wy).map(|q| dilated[q] && d_dil_out[q].sqrt() - 0.5 >= rc).collect();
        added = (0..wx * wy).filter(|&q| !m[q] && closed[q] && in_box(q)).collect();
    }
    let (_, near_out) = edt::distance_to_sites(wx, wy, |q| !m[q]);
    let (_, near_in) = edt::distance_to_sites(wx, wy, |q| m[q]);
    let copy = |solid: &mut Solid, k: usize, from: usize, side: Side| match side {
        Side::Top => {
            solid.top[k] = solid.top[from];
            solid.top_label[k] = label;
            solid.top_order[k] = order;
            if solid.bot[k].is_nan() {
                solid.bot[k] = solid.bot[from];
                solid.bot_label[k] = solid.bot_label[from];
            }
        }
        Side::Bottom => {
            solid.bot[k] = solid.bot[from];
            solid.bot_label[k] = label;
            solid.bot_order[k] = order;
        }
    };
    for &q in &removed {
        let k = win(q % wx, q / wx);
        if empty_outside {
            solid.top[k] = f32::NAN;
            solid.bot[k] = f32::NAN;
        } else if near_out[q] != usize::MAX {
            let from = win(near_out[q] % wx, near_out[q] / wx);
            if !solid.top[from].is_nan() {
                copy(solid, k, from, side);
            } else {
                solid.top[k] = f32::NAN;
                solid.bot[k] = f32::NAN;
            }
        }
        rep.removed += 1;
    }
    for &q in &added {
        let k = win(q % wx, q / wx);
        if near_in[q] != usize::MAX {
            let from = win(near_in[q] % wx, near_in[q] / wx);
            copy(solid, k, from, side);
            rep.added += 1;
        }
    }
    rep
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Op, Prism};
    use sketch_regions_core::{Curve, Piece, arrange};

    fn rect_region(x0: f64, y0: f64, x1: f64, y1: f64) -> sketch_regions_core::Region {
        let p = |a, b| Piece { id: "r".into(), curve: Curve::Line { a, b } };
        arrange(&[p([x0, y0], [x1, y0]), p([x1, y0], [x1, y1]), p([x1, y1], [x0, y1]), p([x0, y1], [x0, y0])], 1e-9).remove(0)
    }

    #[test]
    fn pocket_corners_round_off() {
        let grid = Grid::covering(0.0, 0.0, 20.0, 20.0, 0.25, 2);
        let mut s = Solid::empty(grid.clone());
        let slab = Prism { region: rect_region(0.0, 0.0, 20.0, 20.0), origin: [0.0; 3], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], dir: [0.0, 0.0, 1.0], w0: 0.0, w1: 4.0, label: None };
        s.extrude(&slab, Op::Add, 16, 1);
        let pocket = Prism { region: rect_region(5.0, 5.0, 15.0, 15.0), w0: 3.0, w1: 4.0, ..slab };
        s.extrude(&pocket, Op::Remove, 32, 2);
        let mask: Vec<bool> = s.top_label.iter().map(|&l| l == 32).collect();
        let rep = round_mask(&mut s, Side::Top, &mask, 2.0, [5.0, 5.0, 15.0, 15.0], false, 48, 3);
        // Four corners, each 2^2 (1 - pi/4) = 0.86 area units = ~13.7 cells.
        assert!((rep.removed as f64 - 4.0 * 0.8584 / 0.0625).abs() < 12.0, "{}", rep.removed);
        assert_eq!(rep.added, 0);
        // The corner cell is back at slab height; the pocket middle is not.
        let k = grid.idx(((5.2 - grid.x0) / grid.pitch) as usize, ((5.2 - grid.y0) / grid.pitch) as usize);
        assert_eq!(s.top[k], 4.0);
        let k = grid.idx(((10.0 - grid.x0) / grid.pitch) as usize, ((5.2 - grid.y0) / grid.pitch) as usize);
        assert_eq!(s.top[k], 3.0);
    }
}
