//! The plan map: a 2.5D solid as an exact arrangement of the plan.
//!
//! The solid is `bot(x, y) <= z <= top(x, y)` over a footprint. The plan
//! is divided into [`Cell`]s, exact 2D regions bounded by lines, arcs and
//! traced curves, and each cell names the [`Sheet`] its columns end on
//! above and below. Walls are not stored: a wall is a cell boundary
//! across which the heights differ.
//!
//! Adding or removing a body (another plan map) is an overlay. The two
//! sets of cells are arranged together, each face of the arrangement
//! takes the column the two leave at a point inside it, and where two
//! sheets cross inside a face the crossing, a curve in the plan, is put
//! into the arrangement first, so that every face has one answer.
//! Coincident walls and flush faces, the hard cases of a boundary
//! evaluator in space, are here shared boundary pieces and equal
//! heights.
//!
//! The column rules are those of `heightfield_core::Solid::apply_body`,
//! so the two models of the same feature tree agree.

use crate::sheet::Sheet;
use crate::trace::{pieces_of, trace_zero};
use brep_core::math::Vec3;
use sketch_regions_core::geom::{add, scale};
use sketch_regions_core::{Curve, P2, Piece, Region, arrange, arrangement};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Add,
    Remove,
}

#[derive(Clone, Debug)]
pub struct Cell {
    pub region: Region,
    /// Indices into [`PlanMap::sheets`].
    pub top: usize,
    pub bot: usize,
}

/// Lengths in metres.
#[derive(Clone, Copy, Debug)]
pub struct Tolerances {
    /// Plan points closer than this are one vertex.
    pub snap: f64,
    /// Heights closer than this are equal (a cut from a face starts on
    /// it, not a rounding error below it).
    pub height: f64,
    /// Chords of traced curves stay this close to them.
    pub chord: f64,
    /// A line or an arc is recognized in a traced curve to this.
    pub exact: f64,
    /// Grid a trace starts from.
    pub pitch: f64,
}

impl Default for Tolerances {
    fn default() -> Self {
        Tolerances {
            snap: 1e-7,
            height: 1e-7,
            chord: 2e-7,
            exact: 1e-9,
            pitch: 5e-4,
        }
    }
}

/// What an overlay did that a 2.5D solid cannot say exactly.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// Faces where the body was added across a gap (bridged) or removed
    /// from the middle of the column (the thicker part kept).
    pub split_faces: usize,
    pub cells_before: usize,
    pub cells_after: usize,
    pub creases: usize,
}

#[derive(Clone, Debug, Default)]
pub struct PlanMap {
    pub sheets: Vec<Sheet>,
    pub cells: Vec<Cell>,
    pub tol: Tolerances,
    /// Counter for the names of traced curves.
    traced: usize,
}

fn bbox_overlap(a: [f64; 4], b: [f64; 4], pad: f64) -> Option<[f64; 4]> {
    let r = [
        a[0].max(b[0]) - pad,
        a[1].max(b[1]) - pad,
        a[2].min(b[2]) + pad,
        a[3].min(b[3]) + pad,
    ];
    (r[0] < r[2] && r[1] < r[3]).then_some(r)
}

/// The parts of `pieces` inside both regions: a crease is where two
/// sheets cross, but it divides a cell only where both are the column's.
fn clip(pieces: Vec<Piece>, a: &Region, b: &Region, snap: f64) -> Vec<Piece> {
    let mut out = Vec::new();
    for piece in pieces {
        let mut stops = vec![0.0, 1.0];
        for edge in a.pieces().chain(b.pieces()) {
            for (t, _) in sketch_regions_core::geom::intersect(&piece.curve, &edge.curve, snap) {
                stops.push(t);
            }
        }
        stops.sort_by(|x, y| x.total_cmp(y));
        let len = piece.curve.length();
        stops.dedup_by(|x, y| (*x - *y).abs() * len <= snap);
        for w in stops.windows(2) {
            let part = piece.curve.sub(w[0], w[1]);
            let mid = part.point(0.5);
            if part.length() > snap && a.contains(mid) && b.contains(mid) {
                out.push(Piece {
                    id: piece.id.clone(),
                    curve: part,
                });
            }
        }
    }
    out
}

/// A point of the region as far from its boundary as a handful of tries
/// find: the place to ask what the region is.
pub fn deep_point(region: &Region) -> Option<P2> {
    let size = region.area.abs().sqrt().max(1e-12);
    let mut best: Option<(P2, f64)> = None;
    let n = region.outer.len();
    let stride = n.div_ceil(16).max(1);
    for piece in region.outer.iter().step_by(stride) {
        for frac in [0.5, 0.2, 0.8] {
            let m = piece.curve.point(frac);
            let t = piece.curve.tangent(frac);
            let inward = [-t[1], t[0]];
            for depth in [0.3, 0.1, 0.03, 0.01, 0.003, 0.001] {
                let p = add(m, scale(inward, depth * size));
                if !region.contains(p) {
                    continue;
                }
                let d = region.boundary_distance(p);
                if best.is_none_or(|(_, b)| d > b) {
                    best = Some((p, d));
                }
            }
        }
    }
    best.map(|(p, _)| p).or_else(|| region.interior_point())
}

/// The cells of an arrangement: `column` says what `(top, bot)` stands
/// over a point, the faces of the arrangement of `pieces` are asked at a
/// point inside each, and faces with the same answer on both sides of an
/// edge are one cell.
pub fn cells_of(
    pieces: &[Piece],
    snap: f64,
    column: &mut dyn FnMut(P2) -> Result<Option<(usize, usize)>, String>,
) -> Result<Vec<Cell>, String> {
    let arr = arrangement(pieces, snap);
    let mut labels: Vec<Option<(usize, usize)>> = Vec::with_capacity(arr.faces.len());
    for face in &arr.faces {
        labels.push(match deep_point(face) {
            Some(p) => column(p)?,
            None => None,
        });
    }
    let label = |f: Option<usize>| f.and_then(|f| labels[f]);
    let boundary: Vec<Piece> = arr
        .edges
        .iter()
        .filter(|e| label(e.left) != label(e.right))
        .map(|e| Piece {
            id: e.id.clone(),
            curve: e.curve,
        })
        .collect();
    let face_boxes: Vec<[f64; 4]> = arr.faces.iter().map(|f| f.bbox()).collect();
    let mut cells = Vec::new();
    for region in arrange(&boundary, snap) {
        let Some(p) = deep_point(&region) else {
            continue;
        };
        let face = arr.faces.iter().zip(&face_boxes).position(|(f, b)| {
            p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3] && f.contains(p)
        });
        if let Some((top, bot)) = label(face) {
            cells.push(Cell { region, top, bot });
        }
    }
    Ok(cells)
}

impl PlanMap {
    pub fn new() -> PlanMap {
        PlanMap::default()
    }

    /// A body of one cell: `region` between two sheets.
    pub fn slab(region: Region, bot: Sheet, top: Sheet) -> PlanMap {
        PlanMap::slabs(vec![region], bot, top)
    }

    /// A body of several cells between the same two sheets.
    pub fn slabs(regions: Vec<Region>, bot: Sheet, top: Sheet) -> PlanMap {
        PlanMap {
            sheets: vec![bot, top],
            cells: regions
                .into_iter()
                .map(|region| Cell {
                    region,
                    top: 1,
                    bot: 0,
                })
                .collect(),
            ..PlanMap::default()
        }
    }

    pub fn cell_at(&self, p: P2) -> Option<usize> {
        self.cells.iter().position(|c| c.region.contains(p))
    }

    /// `(bot, top)` heights of the column over `p`.
    pub fn column(&self, p: P2) -> Option<(f64, f64)> {
        let c = &self.cells[self.cell_at(p)?];
        Some((
            self.sheets[c.bot].z(p[0], p[1])?,
            self.sheets[c.top].z(p[0], p[1])?,
        ))
    }

    /// `[min_x, min_y, max_x, max_y]` of the footprint.
    pub fn bbox(&self) -> [f64; 4] {
        let mut bb = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for c in &self.cells {
            let b = c.region.bbox();
            bb = [
                bb[0].min(b[0]),
                bb[1].min(b[1]),
                bb[2].max(b[2]),
                bb[3].max(b[3]),
            ];
        }
        bb
    }

    /// The volume by columns on a grid of `pitch` (a check, not a
    /// measurement: walls are as coarse as the grid).
    pub fn volume(&self, pitch: f64) -> f64 {
        let bb = self.bbox();
        if self.cells.is_empty() || bb[0] >= bb[2] {
            return 0.0;
        }
        let nx = ((bb[2] - bb[0]) / pitch).ceil() as usize;
        let ny = ((bb[3] - bb[1]) / pitch).ceil() as usize;
        let mut v = 0.0;
        for j in 0..ny {
            for i in 0..nx {
                let p = [
                    bb[0] + (i as f64 + 0.5) * pitch,
                    bb[1] + (j as f64 + 0.5) * pitch,
                ];
                if let Some((b, t)) = self.column(p) {
                    v += (t - b) * pitch * pitch;
                }
            }
        }
        v
    }

    /// The curves inside `rect` along which sheets `a` and `b` are at
    /// the same height, as boundary pieces.
    fn creases(&self, a: usize, b: usize, rect: [f64; 4], traced: &mut usize) -> Vec<Piece> {
        if a == b {
            return Vec::new();
        }
        let (sa, sb) = (&self.sheets[a], &self.sheets[b]);
        let name = |n: usize| format!("crease:{a}:{b}:{n}");
        if let (Some(pa), Some(pb)) = (sa.as_plane(), sb.as_plane()) {
            // z0 + gx x + gy y = 0: a line, or nothing.
            let (z0, gx, gy) = (pa.0 - pb.0, pa.1 - pb.1, pa.2 - pb.2);
            let g = (gx * gx + gy * gy).sqrt();
            if g < 1e-12 {
                return Vec::new();
            }
            // The point of the line nearest the rectangle's centre, and
            // as much of the line as can cross the rectangle.
            let c = [0.5 * (rect[0] + rect[2]), 0.5 * (rect[1] + rect[3])];
            let d = (z0 + gx * c[0] + gy * c[1]) / g;
            let reach = 0.5 * ((rect[2] - rect[0]).hypot(rect[3] - rect[1]));
            if d.abs() > reach {
                return Vec::new();
            }
            let foot = [c[0] - d * gx / g, c[1] - d * gy / g];
            let along = [-gy / g, gx / g];
            *traced += 1;
            return vec![Piece {
                id: name(*traced),
                curve: Curve::Line {
                    a: add(foot, scale(along, -reach)),
                    b: add(foot, scale(along, reach)),
                },
            }];
        }
        let g = |p: P2| Some(sa.z_open(p[0], p[1])? - sb.z_open(p[0], p[1])?);
        let mut lines = trace_zero(&g, rect, self.tol.pitch, self.tol.chord);
        // A crease that runs out where one sheet turns vertical touches
        // that sheet's rim, at the point of the rim as high as the other
        // sheet. It ends there exactly, as does the crease with the
        // sheet's other half.
        let pin = |end: P2| -> Option<P2> {
            let reach = 4.0 * self.tol.pitch;
            [(sa, sb), (sb, sa)]
                .into_iter()
                .find_map(|(folded, other)| {
                    let (mut on, mut n) = folded.rim_near(end[0], end[1], reach)?;
                    let miss = |on: Vec3| Some(other.z(on[0], on[1])? - on[2]);
                    let mut h = miss(on)?;
                    let mut step = 1e-5;
                    for _ in 0..40 {
                        if h.abs() < 1e-11 {
                            break;
                        }
                        // Along the rim: across its level normal.
                        let len = n[0].hypot(n[1]);
                        let along = [-n[1] / len, n[0] / len];
                        let (next, next_n) = folded.rim_near(
                            on[0] + step * along[0],
                            on[1] + step * along[1],
                            reach,
                        )?;
                        let h1 = miss(next)?;
                        if h1 == h {
                            return None;
                        }
                        // The secant's next step, from the new point.
                        let moved = (next[0] - on[0]) * along[0] + (next[1] - on[1]) * along[1];
                        step = -h1 * moved / (h1 - h);
                        (on, n, h) = (next, next_n, h1);
                        if step.abs() > reach {
                            return None;
                        }
                    }
                    let at = [on[0], on[1]];
                    (h.abs() < 1e-9 && sketch_regions_core::geom::dist(at, end) <= reach)
                        .then_some(at)
                })
        };
        let mut out = Vec::new();
        for line in lines.iter_mut() {
            let n = line.len();
            let closed =
                n > 3 && sketch_regions_core::geom::dist(line[0], line[n - 1]) <= self.tol.exact;
            let mut beyond = [20.0 * self.tol.snap; 2];
            if !closed {
                for (k, slot) in [(0, 0), (n - 1, 1)] {
                    if let Some(at) = pin(line[k]) {
                        line[k] = at;
                        beyond[slot] = 0.0;
                    }
                }
            }
            *traced += 1;
            out.extend(pieces_of(line, &name(*traced), self.tol.exact, beyond));
        }
        out
    }

    /// The column that `op` leaves of the solid's column `s` and the
    /// body's column `t` over `p`, as `(top, bot)` sheets.
    fn combine(
        &self,
        p: P2,
        s: Option<&Cell>,
        t: Option<&Cell>,
        op: Op,
        split: &mut bool,
    ) -> Result<Option<(usize, usize)>, String> {
        let z = |sheet: usize| -> Result<f64, String> {
            self.sheets[sheet].z(p[0], p[1]).ok_or_else(|| {
                format!(
                    "sheet {sheet} ({}) does not reach ({:.6}, {:.6})",
                    self.sheets[sheet].name, p[0], p[1]
                )
            })
        };
        let tol = self.tol.height;
        Ok(match (s, t, op) {
            (None, None, _) | (None, Some(_), Op::Remove) => None,
            (Some(s), None, _) => Some((s.top, s.bot)),
            (None, Some(t), Op::Add) => Some((t.top, t.bot)),
            (Some(s), Some(t), Op::Add) => {
                let (top, bot, hi, lo) = (z(s.top)?, z(s.bot)?, z(t.top)?, z(t.bot)?);
                if lo > top + tol || hi < bot - tol {
                    *split = true;
                }
                Some((
                    if hi > top + tol { t.top } else { s.top },
                    if lo < bot - tol { t.bot } else { s.bot },
                ))
            }
            (Some(s), Some(t), Op::Remove) => {
                let (top, bot, hi, lo) = (z(s.top)?, z(s.bot)?, z(t.top)?, z(t.bot)?);
                if hi <= bot + tol || lo >= top - tol {
                    Some((s.top, s.bot))
                } else if lo <= bot + tol && hi >= top - tol {
                    None
                } else if lo > bot + tol && hi < top - tol {
                    *split = true;
                    if top - hi >= lo - bot {
                        Some((s.top, t.top))
                    } else {
                        Some((t.bot, s.bot))
                    }
                } else if hi >= top - tol {
                    Some((t.bot, s.bot))
                } else {
                    Some((s.top, t.top))
                }
            }
        })
    }

    /// Add `body` to the solid or remove it.
    pub fn apply(&mut self, body: &PlanMap, op: Op) -> Result<Report, String> {
        let mut report = Report {
            cells_before: self.cells.len(),
            ..Report::default()
        };
        let base = self.sheets.len();
        self.sheets.extend(body.sheets.iter().cloned());
        let tools: Vec<Cell> = body
            .cells
            .iter()
            .map(|c| Cell {
                region: c.region.clone(),
                top: c.top + base,
                bot: c.bot + base,
            })
            .collect();
        let tool_boxes: Vec<[f64; 4]> = tools.iter().map(|c| c.region.bbox()).collect();
        let pad = 4.0 * self.tol.snap;

        let (touched, kept): (Vec<Cell>, Vec<Cell>) =
            std::mem::take(&mut self.cells).into_iter().partition(|c| {
                let b = c.region.bbox();
                tool_boxes
                    .iter()
                    .any(|t| bbox_overlap(b, *t, pad).is_some())
            });

        let mut traced = self.traced;
        let mut pieces: Vec<Piece> = Vec::new();
        for c in touched.iter().chain(tools.iter()) {
            pieces.extend(c.region.pieces().cloned());
        }
        for s in &touched {
            for (t, tb) in tools.iter().zip(&tool_boxes) {
                let Some(rect) = bbox_overlap(s.region.bbox(), *tb, pad) else {
                    continue;
                };
                let pairs: &[(usize, usize)] = match op {
                    Op::Add => &[(s.top, t.top), (s.bot, t.bot)],
                    Op::Remove => &[
                        (s.top, t.top),
                        (s.bot, t.bot),
                        (s.top, t.bot),
                        (s.bot, t.top),
                    ],
                };
                for &(a, b) in pairs {
                    let found = clip(
                        self.creases(a, b, rect, &mut traced),
                        &s.region,
                        &t.region,
                        self.tol.snap,
                    );
                    report.creases += usize::from(!found.is_empty());
                    pieces.extend(found);
                }
            }
        }

        self.traced = traced;

        let mut splits = 0;
        let made = cells_of(&pieces, self.tol.snap, &mut |p| {
            let s = touched.iter().find(|c| c.region.contains(p));
            let t = tools.iter().find(|c| c.region.contains(p));
            let mut split = false;
            let column = self.combine(p, s, t, op, &mut split)?;
            splits += usize::from(split);
            Ok(column)
        })?;
        report.split_faces = splits;
        self.cells = kept;
        self.cells.extend(made);
        report.cells_after = self.cells.len();
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64, id: &str) -> Region {
        let line = |k: usize, a: P2, b: P2| Piece {
            id: format!("{id}.{k}"),
            curve: Curve::Line { a, b },
        };
        arrange(
            &[
                line(0, [x0, y0], [x1, y0]),
                line(1, [x1, y0], [x1, y1]),
                line(2, [x1, y1], [x0, y1]),
                line(3, [x0, y1], [x0, y0]),
            ],
            1e-9,
        )
        .remove(0)
    }

    fn disc(c: P2, r: f64, id: &str) -> Region {
        arrange(
            &[Piece {
                id: id.into(),
                curve: Curve::circle(c, r),
            }],
            1e-9,
        )
        .remove(0)
    }

    fn under(z: f64, name: &str) -> Sheet {
        Sheet::plane([0.0, 0.0, z], [0.0, 0.0, -1.0], name)
    }

    fn block() -> PlanMap {
        let mut m = PlanMap::new();
        let body = PlanMap::slab(
            rect(0.0, 0.0, 0.10, 0.08, "base"),
            under(0.0, "base:bottom"),
            Sheet::level(0.03, "base:top"),
        );
        m.apply(&body, Op::Add).unwrap();
        m
    }

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs()
    }

    #[test]
    fn pocket_boss_and_hole() {
        let mut m = block();
        assert_eq!(m.cells.len(), 1);
        // A pocket from the top, 10 mm deep.
        let pocket = PlanMap::slab(
            rect(0.02, 0.02, 0.05, 0.06, "pocket"),
            under(0.02, "pocket:floor"),
            Sheet::level(1.0, "pocket:above"),
        );
        let r = m.apply(&pocket, Op::Remove).unwrap();
        assert_eq!((r.cells_after, r.split_faces), (2, 0));
        assert_eq!(m.column([0.03, 0.03]), Some((0.0, 0.02)));
        // A round boss, half on the block and half over its edge.
        let boss = PlanMap::slab(
            disc([0.10, 0.04], 0.01, "boss"),
            under(0.0, "boss:bottom"),
            Sheet::level(0.04, "boss:top"),
        );
        m.apply(&boss, Op::Add).unwrap();
        assert_eq!(m.column([0.105, 0.04]), Some((0.0, 0.04)));
        assert_eq!(m.column([0.095, 0.04]), Some((0.0, 0.04)));
        // A hole through everything, inside the pocket.
        let hole = PlanMap::slab(
            disc([0.035, 0.04], 0.005, "hole"),
            under(-1.0, "hole:below"),
            Sheet::level(1.0, "hole:above"),
        );
        m.apply(&hole, Op::Remove).unwrap();
        assert_eq!(m.column([0.035, 0.04]), None);
        // A channel in the underside, 5 mm up, across the whole block.
        let channel = PlanMap::slab(
            rect(-0.01, 0.065, 0.12, 0.075, "channel"),
            under(-1.0, "channel:below"),
            Sheet::level(0.005, "channel:roof"),
        );
        m.apply(&channel, Op::Remove).unwrap();
        assert_eq!(m.column([0.05, 0.07]), Some((0.005, 0.03)));

        let expect = 0.10 * 0.08 * 0.03 - 0.03 * 0.04 * 0.01
            + PI * 0.01 * 0.01 * (0.5 * 0.04 + 0.5 * 0.01)
            - PI * 0.005 * 0.005 * 0.02
            - 0.10 * 0.01 * 0.005;
        let v = m.volume(2.5e-4);
        assert!(close(v, expect, 2e-3), "{v} against {expect}");
    }

    #[test]
    fn flush_faces_are_not_cuts() {
        // A body whose top is the block's top leaves the top alone, and a
        // cut that starts on the top is a cut from the top.
        let mut m = block();
        let same = PlanMap::slab(
            rect(0.01, 0.01, 0.03, 0.03, "pad"),
            under(0.01, "pad:bottom"),
            Sheet::level(0.03 + 1e-12, "pad:top"),
        );
        m.apply(&same, Op::Add).unwrap();
        assert_eq!(m.cells.len(), 1);
        let cut = PlanMap::slab(
            rect(0.01, 0.01, 0.03, 0.03, "cut"),
            under(0.02, "cut:floor"),
            Sheet::level(0.03 - 1e-12, "cut:above"),
        );
        let r = m.apply(&cut, Op::Remove).unwrap();
        assert_eq!((r.cells_after, r.split_faces), (2, 0));
        assert_eq!(m.column([0.02, 0.02]), Some((0.0, 0.02)));
    }

    #[test]
    fn a_sloping_cut_leaves_a_crease() {
        // Everything above a plane rising along x from 10 mm to 50 mm is
        // removed: the plane meets the top at x = 50 mm.
        let mut m = block();
        let cut = PlanMap::slab(
            rect(-0.01, -0.01, 0.11, 0.09, "cut"),
            Sheet::plane([0.0, 0.0, 0.01], [0.4, 0.0, -1.0], "ramp"),
            Sheet::level(1.0, "cut:above"),
        );
        let r = m.apply(&cut, Op::Remove).unwrap();
        assert_eq!(r.cells_after, 2, "{r:?}");
        let (b, t) = m.column([0.025, 0.04]).unwrap();
        assert!(b == 0.0 && (t - 0.02).abs() < 1e-12);
        assert_eq!(m.column([0.07, 0.04]), Some((0.0, 0.03)));
        let expect = 0.08 * (0.05 * 0.5 * (0.01 + 0.03) + 0.05 * 0.03);
        let v = m.volume(2.5e-4);
        assert!(close(v, expect, 2e-3), "{v} against {expect}");
        // The crease is a boundary piece of both cells, at x = 50 mm.
        for c in &m.cells {
            assert!(c.region.pieces().any(
                |p| p.id.starts_with("crease:") && (p.curve.point(0.5)[0] - 0.05).abs() < 1e-9
            ));
        }
    }

    #[test]
    fn a_ridge_crosses_a_flat() {
        // A lying half cylinder of radius 20 mm, axis along y at height
        // 20 mm, added to the 30 mm block: it stands above the top
        // between the two lines where it is 30 mm high.
        let mut m = block();
        let frame = brep_core::math::Frame::from_axis_ref(
            [0.05, 0.0, 0.02],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        );
        let ridge = PlanMap::slab(
            rect(0.031, 0.01, 0.069, 0.07, "ridge"),
            under(0.0, "ridge:bottom"),
            Sheet::of(
                brep_core::ir::Surface::Cylinder {
                    frame,
                    radius: 0.02,
                },
                true,
                "ridge:top",
            ),
        );
        let r = m.apply(&ridge, Op::Add).unwrap();
        assert_eq!(r.cells_after, 2, "{r:?}");
        let half = (0.02f64 * 0.02 - 0.01 * 0.01).sqrt();
        let (_, t) = m.column([0.05, 0.04]).unwrap();
        assert!(
            (t - 0.04).abs() < 1e-12,
            "{t} {r:?} {:?}",
            m.cells
                .iter()
                .map(|c| (c.top, c.region.bbox(), c.region.area))
                .collect::<Vec<_>>()
        );
        let (_, t) = m.column([0.05 + half + 0.001, 0.04]).unwrap();
        assert!((t - 0.03).abs() < 1e-12);
        // The creases were recognized as lines.
        let top = m
            .cells
            .iter()
            .find(|c| m.sheets[c.top].name == "ridge:top")
            .unwrap();
        let creases: Vec<&Piece> = top
            .region
            .pieces()
            .filter(|p| p.id.starts_with("crease:"))
            .collect();
        assert_eq!(creases.len(), 2);
        for p in creases {
            assert!(matches!(p.curve, Curve::Line { .. }));
            assert!(((p.curve.point(0.5)[0] - 0.05).abs() - half).abs() < 1e-9);
        }
    }
}
