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
    pub regions: Vec<Region>,
    /// Point entities by id, in plane coordinates.
    pub points: HashMap<String, [f64; 2]>,
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn norm3(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
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
        let mut pieces = Vec::new();
        let mut points = HashMap::new();
        for e in ents {
            match &e.geom {
                Geom3d::Point(p) => {
                    points.insert(e.id.clone(), frame.to_2d(*p));
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
                    let n = ((*major * 64.0 / 0.001).sqrt() as usize).clamp(48, 512);
                    pieces.extend(ellipse_pieces(&e.id, c, *major, *minor, dir, n));
                }
            }
        }
        let regions = arrange(&pieces, eps);
        let n = unit3(cross3(frame.ex, frame.ey));
        let normal = if frame_flipped { [-n[0], -n[1], -n[2]] } else { n };
        SolvedSketch { feature_id: feature_id.to_string(), name: name.to_string(), frame, normal, pieces, regions, points }
    }

    /// The region whose boundary ids are exactly `ids`, else the largest
    /// region whose boundary lies within `ids`.
    pub fn region_by_ids(&self, ids: &[String]) -> Option<&Region> {
        let set: std::collections::BTreeSet<&str> = ids.iter().map(|s| s.as_str()).collect();
        if let Some(r) = self.regions.iter().find(|r| r.ids.iter().map(|s| s.as_str()).collect::<std::collections::BTreeSet<_>>() == set) {
            return Some(r);
        }
        self.regions
            .iter()
            .filter(|r| r.ids.iter().all(|i| set.contains(i.as_str())))
            .max_by(|a, b| a.area.partial_cmp(&b.area).unwrap())
    }

    /// The 3D point of a sketch point entity (or a curve end: `id.start` /
    /// `id.end` are point entities in the dumps already).
    pub fn point3(&self, id: &str) -> Option<[f64; 3]> {
        self.points.get(id).map(|p| self.frame.to_3d(*p))
    }
}
