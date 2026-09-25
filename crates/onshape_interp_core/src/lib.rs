//! Interpreter for an Onshape feature list in 2.5D.
//!
//! The feature tree references geometry by opaque deterministic ids. The
//! interpreter never reads those directly: an [`IdTable`] maps each id to a
//! semantic reference ([`Ref`]) that survives configuration changes, such
//! as "the region of sketch S bounded by entities E", "the cap face of
//! feature F" or "the edge where the wall of F meets the cap of G". A
//! reference the table lacks is reported, not guessed.
//!
//! Inputs are the feature JSON (Features API, with ids), the evaluated
//! variable scope, and the sketches already solved at the configuration
//! ([`SolvedSketch`]: plane frame plus 2D curves). Output is a
//! [`heightfield_core::Solid`] and a per-feature report.

pub mod exec;
pub mod idtable;
pub mod plan;

use std::collections::HashMap;

use onshape_import_core::sketch3d::{Entity3d, Frame, Geom3d};
use sketch_regions_core::{Curve, Piece, Region, arrange, ellipse_pieces};

pub use idtable::{FaceRole, IdTable, Ref};

/// A sketch at one configuration: its plane and its curves in plane
/// coordinates, with the regions of their arrangement.
#[derive(Clone, Debug)]
pub struct SolvedSketch {
    pub feature_id: String,
    pub name: String,
    pub frame: Frame,
    /// Onshape's sketch normal (the frame's normal, possibly negated).
    pub normal: [f64; 3],
    pub pieces: Vec<Piece>,
    /// Construction curves (never bound regions; usable as directions / axes).
    pub construction: Vec<Piece>,
    pub regions: Vec<Region>,
    /// Point entities by id, in plane coordinates.
    pub points: HashMap<String, [f64; 2]>,
    /// Feature index of the sketch in the tree (for coplanar context).
    pub index: usize,
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn norm3(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// Mean of the boundary piece midpoints (rounded, for ordering).
fn region_centroid(r: &Region) -> (i64, i64) {
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut n = 0.0f64;
    for p in r.pieces() {
        let m = p.curve.point(0.5);
        sx += m[0];
        sy += m[1];
        n += 1.0;
    }
    ((sx / n.max(1.0) * 1e6).round() as i64, (sy / n.max(1.0) * 1e6).round() as i64)
}

pub(crate) fn unit3(a: [f64; 3]) -> [f64; 3] {
    let n = norm3(a).max(1e-300);
    [a[0] / n, a[1] / n, a[2] / n]
}

impl SolvedSketch {
    /// Build from world-space entities (a sketches-endpoint dump) and the
    /// fitted frame. `frame_flipped` says the frame's normal is opposite to
    /// Onshape's sketch normal (which decides arc directions in 2D).
    pub fn from_entities(
        feature_id: &str,
        name: &str,
        ents: &[Entity3d],
        frame: Frame,
        frame_flipped: bool,
        eps: f64,
    ) -> SolvedSketch {
        Self::from_entities_with_normal(feature_id, name, ents, frame, frame_flipped, None, eps)
    }

    /// [`from_entities`] with Onshape's sketch normal known (the plane's
    /// normal from the API): it overrides the seed fit's handedness call,
    /// which decides which way every arc turns.
    pub fn from_entities_with_normal(
        feature_id: &str,
        name: &str,
        ents: &[Entity3d],
        frame: Frame,
        frame_flipped: bool,
        normal_hint: Option<[f64; 3]>,
        eps: f64,
    ) -> SolvedSketch {
        let frame_flipped = match normal_hint {
            Some(nh) => {
                let n = cross3(frame.ex, frame.ey);
                n[0] * nh[0] + n[1] * nh[1] + n[2] * nh[2] < 0.0
            }
            None => frame_flipped,
        };
        let mut pieces = Vec::new();
        let mut construction = Vec::new();
        let mut points = HashMap::new();
        for e in ents {
            match &e.geom {
                Geom3d::Point(p) => {
                    points.insert(e.id.clone(), frame.to_2d(*p));
                }
                Geom3d::Line(a, b) if e.construction => {
                    construction.push(Piece { id: e.id.clone(), curve: Curve::Line { a: frame.to_2d(*a), b: frame.to_2d(*b) } });
                }
                Geom3d::Arc { center, clockwise, start, end, .. } if e.construction => {
                    let ccw = if frame_flipped { *clockwise } else { !clockwise };
                    construction.push(Piece { id: e.id.clone(), curve: Curve::arc_from_ends(frame.to_2d(*center), frame.to_2d(*start), frame.to_2d(*end), ccw) });
                }
                Geom3d::Circle { center, radius } if e.construction => {
                    construction.push(Piece { id: e.id.clone(), curve: Curve::circle(frame.to_2d(*center), *radius) });
                }
                _ if e.construction => {}
                Geom3d::Line(a, b) => {
                    pieces.push(Piece { id: e.id.clone(), curve: Curve::Line { a: frame.to_2d(*a), b: frame.to_2d(*b) } });
                }
                Geom3d::Arc { center, clockwise, start, end, .. } => {
                    let ccw_onshape = !clockwise;
                    let ccw = if frame_flipped { !ccw_onshape } else { ccw_onshape };
                    pieces.push(Piece {
                        id: e.id.clone(),
                        curve: Curve::arc_from_ends(frame.to_2d(*center), frame.to_2d(*start), frame.to_2d(*end), ccw),
                    });
                }
                Geom3d::Circle { center, radius } => {
                    pieces.push(Piece { id: e.id.clone(), curve: Curve::circle(frame.to_2d(*center), *radius) });
                }
                Geom3d::Ellipse { center, major, minor, direction } => {
                    let c = frame.to_2d(*center);
                    let d3 = [center[0] + direction[0], center[1] + direction[1], center[2] + direction[2]];
                    let d = frame.to_2d(d3);
                    let dir = [d[0] - c[0], d[1] - c[1]];
                    // A multiple of four keeps the axis extremes as vertices,
                    // where other curves are tangent to the ellipse.
                    pieces.extend(ellipse_pieces(&e.id, c, *major, *minor, dir, 256));
                }
            }
        }
        let regions = arrange(&pieces, eps);
        let n = unit3(cross3(frame.ex, frame.ey));
        let normal = if frame_flipped { [-n[0], -n[1], -n[2]] } else { n };
        SolvedSketch { feature_id: feature_id.to_string(), name: name.to_string(), frame, normal, pieces, construction, regions, points, index: 0 }
    }

    /// Whether `other` lies in the same plane.
    pub fn coplanar(&self, other: &SolvedSketch) -> bool {
        let n = self.normal;
        let d = [other.frame.origin[0] - self.frame.origin[0], other.frame.origin[1] - self.frame.origin[1], other.frame.origin[2] - self.frame.origin[2]];
        let cross = cross3(n, other.normal);
        norm3(cross) < 1e-6 && (d[0] * n[0] + d[1] * n[1] + d[2] * n[2]).abs() < 1e-6
    }

    /// Recompute the regions with the curves of earlier coplanar sketches
    /// added: a sketch on a body face is split by that face's edges, which
    /// in this model are the boundaries of earlier sketches' regions on the
    /// same plane. Only regions touching the sketch's own curves are kept.
    pub fn with_context(&mut self, earlier: &[&SolvedSketch]) {
        self.with_context_and(earlier, &[]);
    }

    /// [`with_context`] plus extra pieces already in this sketch's plane
    /// coordinates (the body's cross-section in the plane).
    pub fn with_context_and(&mut self, earlier: &[&SolvedSketch], extra: &[Piece]) {
        let mut all: Vec<Piece> = self.pieces.clone();
        all.extend(extra.iter().cloned());
        let own: std::collections::HashSet<&str> = self.pieces.iter().map(|p| p.id.as_str()).collect();
        for sk in earlier {
            for p in &sk.pieces {
                let q = [sk.frame.to_3d(p.curve.start()), sk.frame.to_3d(p.curve.point(0.5)), sk.frame.to_3d(p.curve.end())];
                let map = |x: [f64; 3]| self.frame.to_2d(x);
                let curve = match p.curve {
                    Curve::Line { .. } => Curve::Line { a: map(q[0]), b: map(q[2]) },
                    Curve::Arc { r, sweep, .. } => {
                        if sweep.abs() >= std::f64::consts::TAU - 1e-9 {
                            let c = match p.curve {
                                Curve::Arc { c, .. } => c,
                                _ => unreachable!(),
                            };
                            Curve::circle(map(sk.frame.to_3d(c)), r)
                        } else {
                            let c = match p.curve {
                                Curve::Arc { c, .. } => map(sk.frame.to_3d(c)),
                                _ => unreachable!(),
                            };
                            // Turning direction in this frame from the midpoint.
                            let a = map(q[0]);
                            let m = map(q[1]);
                            let b = map(q[2]);
                            // An arc turning left lies entirely to the right
                            // of its chord (any sweep), so a midpoint on the
                            // left means the arc turns right.
                            let left = (b[0] - a[0]) * (m[1] - a[1]) - (b[1] - a[1]) * (m[0] - a[0]) > 0.0;
                            let _ = sweep;
                            Curve::arc_from_ends(c, a, b, !left)
                        }
                    }
                };
                all.push(Piece { id: p.id.clone(), curve });
            }
        }
        if all.len() == self.pieces.len() {
            return;
        }
        // Keep the pieces of the sketch's own regions: a split region lies
        // inside one of them and touches the sketch's own curves.
        let own_regions = self.regions.clone();
        self.regions = arrange(&all, 1e-7)
            .into_iter()
            .filter(|r| r.ids.iter().any(|i| own.contains(i.as_str())))
            .filter(|r| r.interior_point().is_some_and(|p| own_regions.iter().any(|o| o.contains(p))))
            .collect();
    }

    /// The region whose boundary ids are exactly `ids` (the `pick`-th in
    /// centroid order when several share them), else the largest region
    /// whose boundary lies within `ids`.
    pub fn region_by_ids(&self, ids: &[String], pick: Option<usize>) -> Option<&Region> {
        let set: std::collections::BTreeSet<&str> = ids.iter().map(|s| s.as_str()).collect();
        let exact = self.regions_with_ids(&set);
        if !exact.is_empty() {
            return Some(exact[pick.unwrap_or(0).min(exact.len() - 1)]);
        }
        self.regions
            .iter()
            .filter(|r| r.ids.iter().all(|i| set.contains(i.as_str())))
            .max_by(|a, b| a.area.partial_cmp(&b.area).unwrap())
    }

    /// Regions whose boundary id set is exactly `set`, in centroid order.
    pub fn regions_with_ids(&self, set: &std::collections::BTreeSet<&str>) -> Vec<&Region> {
        let mut v: Vec<&Region> = self.regions.iter().filter(|r| r.ids.iter().map(|s| s.as_str()).collect::<std::collections::BTreeSet<_>>() == *set).collect();
        v.sort_by(|a, b| {
            let ca = region_centroid(a);
            let cb = region_centroid(b);
            ca.partial_cmp(&cb).unwrap()
        });
        v
    }

    /// The index of `region` among the regions sharing its boundary ids.
    pub fn region_pick(&self, region: &Region) -> Option<usize> {
        let set: std::collections::BTreeSet<&str> = region.ids.iter().map(|s| s.as_str()).collect();
        let same = self.regions_with_ids(&set);
        if same.len() < 2 {
            return None;
        }
        let c = region_centroid(region);
        same.iter().position(|r| region_centroid(r) == c)
    }

    /// The 3D point of a sketch point entity (or a curve end: `id.start` /
    /// `id.end` are point entities in the dumps already).
    pub fn point3(&self, id: &str) -> Option<[f64; 3]> {
        self.points.get(id).map(|p| self.frame.to_3d(*p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ent(id: &str, geom: Geom3d) -> Entity3d {
        Entity3d { id: id.into(), construction: false, geom }
    }

    #[test]
    fn context_arcs_keep_their_side() {
        // Source: a stadium (two lines, two semicircles) on z = 1, in a frame
        // whose y axis is flipped; Onshape's normal is +z.
        let z = 1.0;
        let src_frame = Frame { origin: [0.0, 0.0, z], ex: [1.0, 0.0, 0.0], ey: [0.0, -1.0, 0.0] };
        // Left cap: from (10, 25) to (10, 16), centre (10, 20.5), bulging to x < 10.
        // Counter-clockwise about +z from (10,25) to (10,16) passes through x < 10? CCW from
        // angle 90 deg to 270 deg goes through 180 deg = (5.5, 20.5): yes.
        let src = vec![
            ent("top", Geom3d::Line([10.0, 25.0, z], [16.0, 25.0, z])),
            ent("bottom", Geom3d::Line([10.0, 16.0, z], [16.0, 16.0, z])),
            ent("left", Geom3d::Arc { center: [10.0, 20.5, z], radius: 4.5, clockwise: false, start: [10.0, 25.0, z], end: [10.0, 16.0, z] }),
            ent("right", Geom3d::Arc { center: [16.0, 20.5, z], radius: 4.5, clockwise: true, start: [16.0, 25.0, z], end: [16.0, 16.0, z] }),
        ];
        let a = SolvedSketch::from_entities("A", "A", &src, src_frame, true, 1e-9);
        assert_eq!(a.regions.len(), 1);
        let stadium = 6.0 * 9.0 + std::f64::consts::PI * 4.5 * 4.5;
        assert!((a.regions[0].area - stadium).abs() < 1e-9, "{}", a.regions[0].area);
        // Target: a rectangle over the lower-left corner, identity frame.
        let tgt_frame = Frame { origin: [0.0, 0.0, z], ex: [1.0, 0.0, 0.0], ey: [0.0, 1.0, 0.0] };
        let tgt = vec![
            ent("r.b", Geom3d::Line([5.5, 16.0, z], [13.0, 16.0, z])),
            ent("r.r", Geom3d::Line([13.0, 16.0, z], [13.0, 20.5, z])),
            ent("r.t", Geom3d::Line([13.0, 20.5, z], [5.5, 20.5, z])),
            ent("r.l", Geom3d::Line([5.5, 20.5, z], [5.5, 16.0, z])),
        ];
        let mut b = SolvedSketch::from_entities("B", "B", &tgt, tgt_frame, false, 1e-9);
        b.with_context(&[&a]);
        // The rectangle is split by the left cap's arc into the corner piece
        // outside the stadium and the rest.
        assert_eq!(b.regions.len(), 2, "{:?}", b.regions.iter().map(|r| r.area).collect::<Vec<_>>());
        let corner = b.regions.iter().map(|r| r.area).fold(f64::INFINITY, f64::min);
        // Corner = rectangle 4.5 x 4.5 minus the quarter disc.
        let expect = 4.5 * 4.5 - std::f64::consts::PI * 4.5 * 4.5 / 4.0;
        assert!((corner - expect).abs() < 1e-9, "corner {corner} vs {expect}");
    }
}
