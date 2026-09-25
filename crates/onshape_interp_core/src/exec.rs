//! Executing planned steps against height-field solids.

use std::collections::HashMap;

use heightfield_core::kernel::{BlendOptions, Kernel, blend};
use heightfield_core::loft::{Profile, match_profiles, ruled_triangles};
use heightfield_core::raster::{P3, include_cap, include_triangles, triangulate_loop};
use heightfield_core::{Body, FaceSel, Grid, Op, Prism, Side, Solid};
use onshape_import_core::sketch3d::Frame;
use sketch_regions_core::{Curve, Piece, Region, arrange, difference, union};

use crate::SolvedSketch;
use crate::idtable::{FaceRef, FaceRole, SideRef};
use crate::plan::{BoolOp, Bound, EdgeSel, Planned, RegionSel, Step};

#[derive(Clone, Debug)]
pub enum Status {
    Suppressed,
    Nothing,
    Applied(String),
    Failed(String),
}

#[derive(Clone, Debug)]
pub struct FeatureReport {
    pub index: usize,
    pub name: String,
    pub feature_type: String,
    pub status: Status,
}

/// A rigid motion recorded for a body (applied at output time).
#[derive(Clone, Debug)]
pub enum Rigid {
    Rotate { point: [f64; 3], dir: [f64; 3], angle: f64 },
    Translate([f64; 3]),
}

pub struct Exec {
    pub grid: Grid,
    /// Sketches by feature id.
    pub sketches: HashMap<String, SolvedSketch>,
    /// Bodies by the feature that created them.
    pub bodies: HashMap<String, Solid>,
    /// The first body created (what "default scope" operations act on).
    pub main: Option<String>,
    pub transforms: HashMap<String, Vec<Rigid>>,
    pub reports: Vec<FeatureReport>,
    /// Feature id -> feature index (labels).
    pub labels: HashMap<String, u32>,
    pub blend_options: BlendOptions,
    /// Height jump (m) from which a cell boundary counts as a wall.
    pub wall_jump: f32,
    /// Chord tolerance for flattening arcs into loft profiles (m).
    pub chord_tol: f64,
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// A region in sketch coordinates mapped into the world x/y plane (for a
/// horizontal sketch plane): rotations and reflections keep arcs as arcs.
fn region_to_world_xy(region: &Region, frame: &Frame) -> Vec<Region> {
    let map = |p: [f64; 2]| -> [f64; 2] {
        let q = frame.to_3d(p);
        [q[0], q[1]]
    };
    let det = frame.ex[0] * frame.ey[1] - frame.ex[1] * frame.ey[0];
    let mut pieces = Vec::new();
    for p in region.pieces() {
        let curve = match p.curve {
            Curve::Line { a, b } => Curve::Line { a: map(a), b: map(b) },
            Curve::Arc { sweep, .. } => {
                let s = p.curve.start();
                let e = p.curve.end();
                let c = match p.curve {
                    Curve::Arc { c, .. } => c,
                    _ => unreachable!(),
                };
                let ccw = (sweep > 0.0) == (det > 0.0);
                if sweep.abs() >= std::f64::consts::TAU - 1e-9 {
                    Curve::circle(map(c), match p.curve {
                        Curve::Arc { r, .. } => r,
                        _ => unreachable!(),
                    })
                } else {
                    Curve::arc_from_ends(map(c), map(s), map(e), ccw)
                }
            }
        };
        pieces.push(Piece { id: p.id.clone(), curve });
    }
    arrange(&pieces, 1e-7)
}

impl Exec {
    pub fn new(grid: Grid, sketches: Vec<SolvedSketch>, planned: &[Planned]) -> Exec {
        let labels = planned.iter().map(|p| (p.feature_id.clone(), p.index as u32)).collect();
        Exec {
            grid,
            sketches: sketches.into_iter().map(|s| (s.feature_id.clone(), s)).collect(),
            bodies: HashMap::new(),
            main: None,
            transforms: HashMap::new(),
            reports: Vec::new(),
            labels,
            blend_options: BlendOptions::default(),
            wall_jump: 1e-4,
            chord_tol: 2e-5,
        }
    }

    pub fn main_body(&self) -> Option<&Solid> {
        self.main.as_ref().and_then(|m| self.bodies.get(m))
    }

    fn label(&self, feature_id: &str) -> u32 {
        self.labels.get(feature_id).copied().unwrap_or(u32::MAX)
    }

    fn face_sel(&self, f: &FaceRef) -> FaceSel {
        match f.role {
            FaceRole::Outline => FaceSel::Outline,
            FaceRole::Wall => FaceSel::Wall(self.label(&f.feature)),
            _ => FaceSel::Cap(self.label(&f.feature)),
        }
    }

    fn region_of<'a>(&'a self, sel: &RegionSel) -> Result<(&'a SolvedSketch, &'a Region), String> {
        let sk = self.sketches.get(&sel.sketch).ok_or_else(|| format!("sketch {} not solved", sel.sketch))?;
        let r = sk.region_by_ids(&sel.ids).ok_or_else(|| format!("sketch {} has no region bounded by {:?}", sk.name, sel.ids))?;
        Ok((sk, r))
    }

    fn targets_of(&self, targets: &[String]) -> Result<Vec<String>, String> {
        if targets.is_empty() {
            return self.main.clone().map(|m| vec![m]).ok_or_else(|| "no body to operate on".to_string());
        }
        for t in targets {
            if !self.bodies.contains_key(t) {
                return Err(format!("target body of feature {t} does not exist"));
            }
        }
        Ok(targets.to_vec())
    }

    /// Sweep extents along `dir` for the bounds, from the plane origin.
    fn sweep(&self, origin: [f64; 3], dir: [f64; 3], end: &Bound, second: &Option<Bound>) -> Result<(f64, f64), String> {
        let along = |b: &Bound| -> Result<f64, String> {
            Ok(match b {
                Bound::Blind(d) => *d,
                Bound::ThroughAll => f64::INFINITY,
                Bound::UpToPlane { origin: po, normal } => {
                    let den = dot3(dir, *normal);
                    if den.abs() < 1e-9 {
                        return Err("up-to plane is parallel to the sweep".into());
                    }
                    dot3(sub3(*po, origin), *normal) / den
                }
                Bound::UpToPoint(p) => dot3(sub3(*p, origin), dir),
            })
        };
        let e = along(end)?;
        let (mut w0, mut w1) = (e.min(0.0), e.max(0.0));
        if let Some(s) = second {
            let v = -along(s)?;
            w0 = w0.min(v);
            w1 = w1.max(v);
        }
        Ok((w0, w1))
    }

    fn apply_to_targets(&mut self, targets: &[String], body: &Body, op: Op, label: u32) -> String {
        let mut summary = Vec::new();
        for t in targets {
            let solid = self.bodies.get_mut(t).unwrap();
            let rep = solid.apply_body(body, op, label, label);
            summary.push(format!("{} cells{}{}", rep.cells_changed, if rep.split_columns > 0 { format!(", {} split", rep.split_columns) } else { String::new() }, if rep.emptied > 0 { format!(", {} emptied", rep.emptied) } else { String::new() }));
        }
        summary.join("; ")
    }

    fn run_extrude(&mut self, p: &Planned, regions: &[RegionSel], op: BoolOp, direction: Option<[f64; 3]>, flip: bool, end: &Bound, second: &Option<Bound>, targets: &[String]) -> Result<String, String> {
        if regions.is_empty() {
            return Err("no regions".into());
        }
        let label = p.index as u32;
        let mut body = Body::empty(&self.grid);
        let mut footprint: Vec<Region> = Vec::new();
        let mut horizontal = true;
        for sel in regions {
            let (sk, region) = self.region_of(sel)?;
            let mut dir = direction.unwrap_or(sk.normal);
            if flip {
                dir = [-dir[0], -dir[1], -dir[2]];
            }
            let (w0, w1) = self.sweep(sk.frame.origin, dir, end, second)?;
            let prism = Prism { region: region.clone(), origin: sk.frame.origin, u: sk.frame.ex, v: sk.frame.ey, dir, w0, w1 };
            let pb = prism.body(&self.grid);
            for k in 0..self.grid.len() {
                if !pb.lo[k].is_nan() {
                    body.include_span(k, pb.lo[k], pb.hi[k]);
                }
            }
            if sk.normal[2].abs() > 1.0 - 1e-9 && dir[2].abs() > 1.0 - 1e-9 {
                footprint.extend(region_to_world_xy(region, &sk.frame));
            } else {
                horizontal = false;
            }
        }
        match op {
            BoolOp::New => {
                let mut solid = Solid::empty(self.grid.clone());
                let rep = solid.apply_body(&body, Op::Add, label, label);
                if horizontal {
                    solid.outline = Some(footprint);
                }
                if self.main.is_none() {
                    self.main = Some(p.feature_id.clone());
                }
                self.bodies.insert(p.feature_id.clone(), solid);
                Ok(format!("new body, {} cells", rep.cells_changed))
            }
            BoolOp::Add | BoolOp::Remove => {
                let targets = self.targets_of(targets)?;
                let hf_op = if op == BoolOp::Add { Op::Add } else { Op::Remove };
                let s = self.apply_to_targets(&targets, &body, hf_op, label);
                // Keep the exact footprint in step with horizontal sweeps.
                if horizontal {
                    for t in &targets {
                        let solid = self.bodies.get_mut(t).unwrap();
                        if let Some(outline) = solid.outline.take() {
                            solid.outline = Some(match (op, end, second) {
                                (BoolOp::Add, _, _) => union(&outline, &footprint, 1e-7),
                                (BoolOp::Remove, Bound::ThroughAll, _) | (BoolOp::Remove, _, Some(Bound::ThroughAll)) => difference(&outline, &footprint, 1e-7),
                                _ => outline,
                            });
                        }
                    }
                }
                Ok(s)
            }
            BoolOp::Intersect => Err("intersect extrude".into()),
        }
    }

    /// Sample a region's outer loop in 3D.
    fn profile_of(&self, sk: &SolvedSketch, region: &Region) -> Profile {
        let mut pieces: Vec<Vec<P3>> = Vec::new();
        for p in &region.outer {
            let n = match p.curve {
                Curve::Line { .. } => 1,
                Curve::Arc { r, sweep, .. } => {
                    let per = 2.0 * (1.0 - self.chord_tol / r).max(-1.0).acos();
                    ((sweep.abs() / per.max(1e-6)).ceil() as usize).clamp(1, 512)
                }
            };
            pieces.push((0..=n).map(|q| sk.frame.to_3d(p.curve.point(q as f64 / n as f64))).collect());
        }
        Profile::from_pieces(&pieces)
    }

    fn cap_triangles(&self, sk: &SolvedSketch, region: &Region) -> Vec<[P3; 3]> {
        let prof = self.profile_of(sk, region);
        let pts2: Vec<[f64; 2]> = prof.points.iter().map(|p| sk.frame.to_2d(*p)).collect();
        triangulate_loop(&pts2).into_iter().map(|[a, b, c]| [prof.points[a], prof.points[b], prof.points[c]]).collect()
    }

    fn run_loft(&mut self, p: &Planned, profiles: &[RegionSel], op: BoolOp, targets: &[String]) -> Result<String, String> {
        if profiles.len() != 2 {
            return Err(format!("loft with {} profiles", profiles.len()));
        }
        let (sa, ra) = self.region_of(&profiles[0])?;
        let (sb, rb) = self.region_of(&profiles[1])?;
        let pa = self.profile_of(sa, ra);
        let pb = self.profile_of(sb, rb);
        let (qa, qb) = match_profiles(&pa, &pb, 48);
        let mut tris = ruled_triangles(&qa, &qb);
        let mut body = Body::empty(&self.grid);
        for (sk, r) in [(sa, ra), (sb, rb)] {
            if sk.normal[2].abs() > 1.0 - 1e-9 {
                // Horizontal cap: exact region test at the plane height.
                let world = region_to_world_xy(r, &sk.frame);
                for w in &world {
                    include_cap(&mut body, w, sk.frame.origin[2]);
                }
            } else {
                tris.extend(self.cap_triangles(sk, r));
            }
        }
        include_triangles(&mut body, &tris);
        let label = p.index as u32;
        match op {
            BoolOp::New => {
                let mut solid = Solid::empty(self.grid.clone());
                let rep = solid.apply_body(&body, Op::Add, label, label);
                if self.main.is_none() {
                    self.main = Some(p.feature_id.clone());
                }
                self.bodies.insert(p.feature_id.clone(), solid);
                Ok(format!("new body, {} cells", rep.cells_changed))
            }
            BoolOp::Add | BoolOp::Remove => {
                let targets = self.targets_of(targets)?;
                let hf_op = if op == BoolOp::Add { Op::Add } else { Op::Remove };
                Ok(self.apply_to_targets(&targets, &body, hf_op, label))
            }
            BoolOp::Intersect => Err("intersect loft".into()),
        }
    }

    fn run_blend(&mut self, p: &Planned, edges: &[EdgeSel], kernel: Kernel) -> Result<String, String> {
        let main = self.main.clone().ok_or("no body")?;
        let label = p.index as u32;
        let mut parts = Vec::new();
        let mut failures = Vec::new();
        for e in edges {
            let a = self.face_sel(&e.a);
            let b = self.face_sel(&e.b);
            let side = if e.side == SideRef::Top { Side::Top } else { Side::Bottom };
            let vertical = !matches!(a, FaceSel::Cap(_) | FaceSel::AnyCap) && !matches!(b, FaceSel::Cap(_) | FaceSel::AnyCap);
            if vertical {
                failures.push(format!("vertical edge {:?}/{:?} not implemented", e.a.role, e.b.role));
                continue;
            }
            let solid = self.bodies.get_mut(&main).unwrap();
            let cells = solid.edge_cells(side, a, b, self.wall_jump);
            if cells.cells.is_empty() {
                failures.push(format!("edge {}:{:?} / {}:{:?} has no cells", e.a.feature, e.a.role, e.b.feature, e.b.role));
                continue;
            }
            let rep = blend(solid, side, &cells, &kernel, cells.convex, label, label, self.blend_options);
            parts.push(format!("{} edge cells ({}), {} changed, max {:.3} mm", cells.cells.len(), if cells.convex { "convex" } else { "concave" }, rep.cells_changed, rep.max_change * 1e3));
        }
        if !failures.is_empty() {
            if parts.is_empty() {
                return Err(failures.join("; "));
            }
            parts.push(format!("FAILED: {}", failures.join("; ")));
        }
        Ok(parts.join("; "))
    }

    fn run_boolean(&mut self, p: &Planned, op: BoolOp, tools: &[String], targets: &[String]) -> Result<String, String> {
        let targets = self.targets_of(targets)?;
        let label = p.index as u32;
        let mut summary = Vec::new();
        for tool in tools {
            let ts = self.bodies.remove(tool).ok_or_else(|| format!("tool body of feature {tool} does not exist"))?;
            let body = Body { grid: ts.grid.clone(), lo: ts.bot.clone(), hi: ts.top.clone() };
            let hf_op = match op {
                BoolOp::Add => Op::Add,
                BoolOp::Remove => Op::Remove,
                _ => return Err("boolean intersect / new".into()),
            };
            summary.push(self.apply_to_targets(&targets, &body, hf_op, label));
        }
        Ok(summary.join("; "))
    }

    /// Execute all planned steps in order.
    pub fn run(&mut self, planned: &[Planned]) {
        for p in planned {
            let status = if p.suppressed {
                Status::Suppressed
            } else {
                match &p.step {
                    Err(e) => Status::Failed(e.clone()),
                    Ok(Step::Nothing) => Status::Nothing,
                    Ok(Step::Extrude { regions, op, direction, flip, end, second, targets }) => {
                        match self.run_extrude(p, regions, *op, *direction, *flip, end, second, targets) {
                            Ok(s) => Status::Applied(s),
                            Err(e) => Status::Failed(e),
                        }
                    }
                    Ok(Step::Loft { profiles, op, targets }) => match self.run_loft(p, profiles, *op, targets) {
                        Ok(s) => Status::Applied(s),
                        Err(e) => Status::Failed(e),
                    },
                    Ok(Step::Fillet { edges, radius }) => match self.run_blend(p, edges, Kernel::Ball { r: *radius }) {
                        Ok(s) => Status::Applied(s),
                        Err(e) => Status::Failed(e),
                    },
                    Ok(Step::Chamfer { edges, width_cap, width_wall }) => {
                        match self.run_blend(p, edges, Kernel::Cone { radius: *width_cap, height: *width_wall }) {
                            Ok(s) => Status::Applied(s),
                            Err(e) => Status::Failed(e),
                        }
                    }
                    Ok(Step::Boolean { op, tools, targets }) => match self.run_boolean(p, *op, tools, targets) {
                        Ok(s) => Status::Applied(s),
                        Err(e) => Status::Failed(e),
                    },
                    Ok(Step::Rotate { body, axis_point, axis_dir, angle }) => {
                        self.transforms.entry(body.clone()).or_default().push(Rigid::Rotate { point: *axis_point, dir: *axis_dir, angle: *angle });
                        Status::Applied(format!("rotation {:.1} deg recorded", angle.to_degrees()))
                    }
                    Ok(Step::Translate { body, delta }) => {
                        self.transforms.entry(body.clone()).or_default().push(Rigid::Translate(*delta));
                        Status::Applied(format!("translation ({:.4}, {:.4}, {:.4}) m recorded", delta[0], delta[1], delta[2]))
                    }
                }
            };
            self.reports.push(FeatureReport { index: p.index, name: p.name.clone(), feature_type: p.feature_type.clone(), status });
        }
    }
}
