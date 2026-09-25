//! Executing planned steps against height-field solids.

use std::collections::HashMap;

use heightfield_core::kernel::{BlendOptions, Kernel, blend};
use heightfield_core::loft::{Profile, coons_triangles, match_profiles, ruled_triangles};
use heightfield_core::raster::{P3, include_cap_labelled, include_triangles_labelled, triangulate_loop};
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
    /// Columns emptied / changed by the last apply (footprint bookkeeping).
    pub last_emptied: usize,
    pub last_changed: usize,
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
    /// Cell labels are `feature index * 16 + face`: face 0 is a feature's
    /// cap (or its only surface), face 1 a loft's side surface.
    pub const FACE_CAP: u32 = 0;
    pub const FACE_SIDE: u32 = 1;

    pub fn new(grid: Grid, sketches: Vec<SolvedSketch>, planned: &[Planned]) -> Exec {
        let labels = planned.iter().map(|p| (p.feature_id.clone(), p.index as u32 * 16)).collect();
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
            last_emptied: 0,
            last_changed: 0,
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
            FaceRole::Loft => FaceSel::Cap(self.label(&f.feature) + Self::FACE_SIDE),
            _ => FaceSel::Cap(self.label(&f.feature)),
        }
    }

    fn region_of<'a>(&'a self, sel: &RegionSel) -> Result<(&'a SolvedSketch, &'a Region), String> {
        let sk = self.sketches.get(&sel.sketch).ok_or_else(|| format!("sketch {} not solved", sel.sketch))?;
        let r = sk.region_by_ids(&sel.ids, sel.pick).ok_or_else(|| format!("sketch {} has no region bounded by {:?}", sk.name, sel.ids))?;
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
    /// An up-to-vertex bound ends at the plane through the vertex parallel
    /// to the sketch plane (`normal`), whatever the sweep direction.
    fn sweep(&self, origin: [f64; 3], normal: [f64; 3], dir: [f64; 3], end: &Bound, second: &Option<Bound>) -> Result<(f64, f64), String> {
        let along = |b: &Bound| -> Result<f64, String> {
            Ok(match b {
                Bound::Blind(d) => *d,
                Bound::ThroughAll => f64::INFINITY,
                Bound::UpToPlane { origin: po, normal: pn } => {
                    let den = dot3(dir, *pn);
                    if den.abs() < 1e-9 {
                        return Err(format!("up-to plane (normal {pn:?}) is parallel to the sweep {dir:?}"));
                    }
                    dot3(sub3(*po, origin), *pn) / den
                }
                Bound::UpToPoint(p) => {
                    let den = dot3(dir, normal);
                    if den.abs() < 1e-9 {
                        return Err("sweep direction lies in the sketch plane".into());
                    }
                    dot3(sub3(*p, origin), normal) / den
                }
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
        self.last_emptied = 0;
        self.last_changed = 0;
        let mut summary = Vec::new();
        for t in targets {
            let solid = self.bodies.get_mut(t).unwrap();
            let rep = solid.apply_body(body, op, label, label);
            self.last_emptied += rep.emptied;
            self.last_changed += rep.cells_changed;
            summary.push(format!("{} cells{}{}", rep.cells_changed, if rep.split_columns > 0 { format!(", {} split", rep.split_columns) } else { String::new() }, if rep.emptied > 0 { format!(", {} emptied", rep.emptied) } else { String::new() }));
        }
        summary.join("; ")
    }

    fn run_extrude(&mut self, p: &Planned, regions: &[RegionSel], op: BoolOp, direction: Option<[f64; 3]>, flip: bool, end: &Bound, second: &Option<Bound>, targets: &[String]) -> Result<String, String> {
        if regions.is_empty() {
            return Err("no regions".into());
        }
        let label = p.index as u32 * 16;
        let mut body = Body::empty(&self.grid);
        let mut footprint: Vec<Region> = Vec::new();
        let mut horizontal = true;
        let mut sweep_desc = String::new();
        let mut missing = Vec::new();
        let mut region_areas: Vec<String> = Vec::new();
        for sel in regions {
            let (sk, region) = match self.region_of(sel) {
                Ok(x) => x,
                Err(e) => {
                    missing.push(e);
                    continue;
                }
            };
            region_areas.push(format!("{:.1}in2{}", region.area / (0.0254 * 0.0254), sel.pick.map(|p| format!("#{p}")).unwrap_or_default()));
            let mut dir = direction.unwrap_or(sk.normal);
            if flip {
                dir = [-dir[0], -dir[1], -dir[2]];
            }
            let (w0, w1) = self.sweep(sk.frame.origin, sk.normal, dir, end, second)?;
            sweep_desc = format!("sketch {} normal ({:.2},{:.2},{:.2}) dir ({:.2},{:.2},{:.2}) w [{:.4}, {:.4}] m; regions {}", sk.name, sk.normal[0], sk.normal[1], sk.normal[2], dir[0], dir[1], dir[2], w0, w1, region_areas.join("/"));
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
        if missing.len() == regions.len() {
            return Err(missing.join("; "));
        }
        let missing_note = if missing.is_empty() { String::new() } else { format!("; MISSING {} of {} regions: {}", missing.len(), regions.len(), missing.join("; ")) };
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
                let mut s = self.apply_to_targets(&targets, &body, hf_op, label);
                // A true through cut empties every column it touches.
                let emptied_any = self.last_emptied > 0 && self.last_emptied == self.last_changed;
                // Keep the exact footprint in step with horizontal sweeps.
                if horizontal {
                    for t in &targets {
                        let solid = self.bodies.get_mut(t).unwrap();
                        if let Some(outline) = solid.outline.take() {
                            let before: f64 = outline.iter().map(|r| r.area).sum();
                            // A cut leaves the footprint only where it emptied
                            // columns (a blind pocket with a through-all second
                            // direction into empty air is still a pocket).
                            let new_outline = match (op, emptied_any) {
                                (BoolOp::Add, _) => union(&outline, &footprint, 1e-7),
                                (BoolOp::Remove, true) => difference(&outline, &footprint, 1e-7),
                                _ => outline,
                            };
                            let after: f64 = new_outline.iter().map(|r| r.area).sum();
                            if (after - before).abs() > 1e-12 {
                                s.push_str(&format!("; outline {:.5} -> {:.5} m2 ({} regions)", before, after, new_outline.len()));
                            }
                            solid.outline = Some(new_outline);
                        }
                    }
                }
                s.push_str(&format!("; {sweep_desc}{missing_note}"));
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

    /// A profile is one or more regions of one sketch; several merge into
    /// one face by union.
    fn profile_region(&self, sels: &[RegionSel]) -> Result<(&SolvedSketch, Region), String> {
        let mut regions: Vec<Region> = Vec::new();
        let mut sketch: Option<&SolvedSketch> = None;
        let mut missing = Vec::new();
        for sel in sels {
            match self.region_of(sel) {
                Ok((sk, r)) => {
                    if sketch.is_some_and(|s| s.feature_id != sk.feature_id) {
                        return Err("profile regions from different sketches".into());
                    }
                    sketch = Some(sk);
                    regions.push(r.clone());
                }
                Err(e) => missing.push(e),
            }
        }
        let sk = sketch.ok_or_else(|| format!("profile without regions: {}", missing.join("; ")))?;
        let merged = if regions.len() == 1 {
            regions.pop().unwrap()
        } else {
            let mut acc = vec![regions[0].clone()];
            for r in &regions[1..] {
                acc = union(&acc, std::slice::from_ref(r), 1e-7);
            }
            if acc.len() != 1 {
                return Err(format!("profile regions do not merge into one face ({} pieces)", acc.len()));
            }
            acc.pop().unwrap()
        };
        Ok((sk, merged))
    }

    /// A guide as a 3D polyline: its curves chained end to end.
    fn guide_polyline(&self, curves: &[(String, String)]) -> Result<Vec<P3>, String> {
        let mut pieces: Vec<Vec<P3>> = Vec::new();
        for (sid, id) in curves {
            let sk = self.sketches.get(sid).ok_or_else(|| format!("guide sketch {sid} not solved"))?;
            let piece = sk.pieces.iter().chain(sk.construction.iter()).find(|p| &p.id == id).ok_or_else(|| format!("guide curve {id} not in sketch {}", sk.name))?;
            let n = match piece.curve {
                Curve::Line { .. } => 1,
                Curve::Arc { r, sweep, .. } => {
                    let per = 2.0 * (1.0 - self.chord_tol / r).max(-1.0).acos();
                    ((sweep.abs() / per.max(1e-6)).ceil() as usize).clamp(1, 512)
                }
            };
            pieces.push((0..=n).map(|q| sk.frame.to_3d(piece.curve.point(q as f64 / n as f64))).collect());
        }
        let mut chain = pieces.remove(0);
        while !pieces.is_empty() {
            let end = *chain.last().unwrap();
            let start = chain[0];
            let d = |a: P3, b: P3| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
            let mut best: Option<(usize, bool, bool, f64)> = None; // (index, reverse, prepend, dist)
            for (k, pc) in pieces.iter().enumerate() {
                let cands = [
                    (false, false, d(end, pc[0])),
                    (true, false, d(end, *pc.last().unwrap())),
                    (false, true, d(start, *pc.last().unwrap())),
                    (true, true, d(start, pc[0])),
                ];
                for (rev, pre, dist) in cands {
                    if best.is_none_or(|b| dist < b.3) {
                        best = Some((k, rev, pre, dist));
                    }
                }
            }
            let (k, rev, pre, dist) = best.unwrap();
            if dist > 1e-3 {
                return Err(format!("guide curves do not chain (gap {:.2} mm)", dist * 1e3));
            }
            let mut pc = pieces.remove(k);
            if rev {
                pc.reverse();
            }
            if pre {
                pc.pop();
                pc.extend(chain);
                chain = pc;
            } else {
                chain.extend(pc.into_iter().skip(1));
            }
        }
        Ok(chain)
    }

    fn run_loft(&mut self, p: &Planned, profiles: &[Vec<RegionSel>], guides: &[Vec<(String, String)>], op: BoolOp, targets: &[String]) -> Result<String, String> {
        if profiles.len() != 2 {
            return Err(format!("loft with {} profiles", profiles.len()));
        }
        let (sa, ra) = self.profile_region(&profiles[0])?;
        let (sb, rb) = self.profile_region(&profiles[1])?;
        let (ra, rb) = (&ra, &rb);
        let pa = self.profile_of(sa, ra);
        let pb = self.profile_of(sb, rb);
        let label = p.index as u32 * 16;
        let mut guide_note = String::new();
        let mut side_tris: Vec<[P3; 3]> = Vec::new();
        // Guides that reach both profiles define Coons panels between them.
        let mut attached: Vec<(f64, f64, Vec<P3>)> = Vec::new(); // (s on A, s on B, polyline from A to B)
        let ca = pa.cumulative();
        let cb = pb.cumulative();
        let (la, lb) = (*ca.last().unwrap(), *cb.last().unwrap());
        let mut skipped = 0;
        for g in guides {
            match self.guide_polyline(g) {
                Ok(mut poly) => {
                    let (s0a, d0a) = pa.nearest(&ca, poly[0]);
                    let (s0b, d0b) = pb.nearest(&cb, poly[0]);
                    let (s1a, d1a) = pa.nearest(&ca, *poly.last().unwrap());
                    let (s1b, d1b) = pb.nearest(&cb, *poly.last().unwrap());
                    // Orient from A to B.
                    let (sa_, sb_, da, db) = if d0a + d1b <= d1a + d0b {
                        (s0a, s1b, d0a, d1b)
                    } else {
                        poly.reverse();
                        (s1a, s0b, d1a, d0b)
                    };
                    if da > 5e-3 || db > 5e-3 {
                        skipped += 1;
                        continue;
                    }
                    attached.push((sa_, sb_, poly));
                }
                Err(e) => {
                    guide_note.push_str(&format!("; guide: {e}"));
                    skipped += 1;
                }
            }
        }
        if !attached.is_empty() {
            attached.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            // B runs the same way as A when its attachment parameters
            // increase along with A's; else flip B's parametrisation.
            let n = attached.len();
            let mut inc = 0i32;
            for i in 0..n {
                let (b0, b1) = (attached[i].1, attached[(i + 1) % n].1);
                let fwd = (b1 - b0).rem_euclid(lb);
                inc += if fwd < lb / 2.0 { 1 } else { -1 };
            }
            let flip_b = n >= 2 && inc < 0;
            let b_at = |s: f64| -> P3 { if flip_b { pb.at(&cb, lb - s) } else { pb.at(&cb, s) } };
            let sb_of = |s: f64| if flip_b { (lb - s).rem_euclid(lb) } else { s };
            for i in 0..n {
                let (s0a, s0b_raw, ref g0) = attached[i];
                let (s1a, s1b_raw, ref g1) = attached[(i + 1) % n];
                let s0b = sb_of(s0b_raw);
                let s1b = sb_of(s1b_raw);
                let mut da_ = s1a - s0a;
                if n == 1 || da_ <= 1e-9 {
                    da_ += la;
                }
                let mut db_ = s1b - s0b;
                if n == 1 || db_ <= 1e-9 {
                    db_ += lb;
                }
                let a = |u: f64| pa.at(&ca, s0a + u * da_);
                let b = |u: f64| b_at(s0b + u * db_);
                let glen = |g: &[P3]| g.windows(2).map(|w| ((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2) + (w[1][2] - w[0][2]).powi(2)).sqrt()).sum::<f64>();
                let nu = ((da_.max(db_) / 2e-3).ceil() as usize).clamp(4, 96);
                let nv = ((glen(g0).max(glen(g1)) / 2e-3).ceil() as usize).clamp(4, 96);
                side_tris.extend(coons_triangles(&a, &b, g0, g1, nu, nv));
            }
            guide_note.push_str(&format!("; {} guides -> {} Coons panels{}", n, n, if flip_b { " (B reversed)" } else { "" }));
        }
        if skipped > 0 {
            guide_note.push_str(&format!("; {skipped} guides skipped"));
        }
        if side_tris.is_empty() {
            let (qa, qb) = match_profiles(&pa, &pb, 48);
            side_tris = ruled_triangles(&qa, &qb);
        }
        let mut body = Body::empty(&self.grid);
        let mut cap_tris = Vec::new();
        for (sk, r) in [(sa, ra), (sb, rb)] {
            if sk.normal[2].abs() > 1.0 - 1e-9 {
                // Horizontal cap: exact region test at the plane height.
                let world = region_to_world_xy(r, &sk.frame);
                for w in &world {
                    include_cap_labelled(&mut body, w, sk.frame.origin[2], Some(label + Self::FACE_CAP));
                }
            } else {
                cap_tris.extend(self.cap_triangles(sk, r));
            }
        }
        include_triangles_labelled(&mut body, &cap_tris, Some(label + Self::FACE_CAP));
        include_triangles_labelled(&mut body, &side_tris, Some(label + Self::FACE_SIDE));
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
                Ok(format!("{}{}", self.apply_to_targets(&targets, &body, hf_op, label), guide_note))
            }
            BoolOp::Intersect => Err("intersect loft".into()),
        }
    }

    /// The body holding cells labelled by `feature` (a tool body before its
    /// boolean), else the main body.
    fn body_for(&self, feature: &str) -> Option<String> {
        let l = self.label(feature) / 16;
        for (fid, solid) in &self.bodies {
            if Some(fid) == self.main.as_ref() {
                continue;
            }
            if solid.top_label.iter().zip(solid.top.iter()).any(|(&lab, &t)| !t.is_nan() && lab / 16 == l) {
                return Some(fid.clone());
            }
        }
        self.main.clone()
    }

    fn run_blend(&mut self, p: &Planned, edges: &[EdgeSel], kernel: Kernel) -> Result<String, String> {
        let owner = edges.iter().flat_map(|e| [&e.a.feature, &e.b.feature]).find(|f| !f.is_empty()).cloned().unwrap_or_default();
        let main = self.body_for(&owner).ok_or("no body")?;
        let label = p.index as u32 * 16;
        let mut parts = Vec::new();
        let mut failures = Vec::new();
        for e in edges {
            let a = self.face_sel(&e.a);
            let b = self.face_sel(&e.b);
            let side = if e.side == SideRef::Top { Side::Top } else { Side::Bottom };
            let vertical = !matches!(a, FaceSel::Cap(_) | FaceSel::AnyCap) && !matches!(b, FaceSel::Cap(_) | FaceSel::AnyCap);
            if vertical {
                match self.run_vertical_blend(&main, e, kernel.radius(), label) {
                    Ok(s) => parts.push(s),
                    Err(err) => failures.push(err),
                }
                continue;
            }
            let solid = self.bodies.get_mut(&main).unwrap();
            let mut cells = solid.edge_cells(side, a, b, self.wall_jump);
            // A wall on the footprint boundary has no cells across it: it is
            // the outline wall.
            if cells.cells.is_empty() && matches!(b, FaceSel::Wall(_)) {
                cells = solid.edge_cells(side, a, FaceSel::Outline, self.wall_jump);
            }
            if cells.cells.is_empty() && matches!(a, FaceSel::Wall(_)) {
                cells = solid.edge_cells(side, FaceSel::Outline, b, self.wall_jump);
            }
            if let Some(hb) = e.hint_box {
                // Keep the cells around the blend face (grown by the reach of
                // the kernel plus a margin).
                let g = &solid.grid;
                let m = kernel.radius() * 2.0 + 4.0 * g.pitch;
                let keep: Vec<usize> = (0..cells.cells.len())
                    .filter(|&q| {
                        let (i, j) = g.ij(cells.cells[q]);
                        let (x, y) = (g.x(i), g.y(j));
                        x >= hb[0] - m && x <= hb[3] + m && y >= hb[1] - m && y <= hb[4] + m
                    })
                    .collect();
                cells.cells = keep.iter().map(|&q| cells.cells[q]).collect();
                cells.across = keep.iter().map(|&q| cells.across[q]).collect();
            }
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

    /// A vertical edge: round the corner of the region it belongs to in
    /// plan view. The region is the footprint (both faces on the outline),
    /// else the cells of the wall's owner (a pocket floor or a boss top),
    /// else the hole the owner cut through (no cells).
    fn run_vertical_blend(&mut self, main: &str, e: &EdgeSel, r: f64, label: u32) -> Result<String, String> {
        let hb = e.hint_box.ok_or("vertical edge without a location")?;
        let bbox = [hb[0], hb[1], hb[3], hb[4]];
        let solid = self.bodies.get_mut(main).unwrap();
        let g = solid.grid.clone();
        let side = if e.side == SideRef::Top { Side::Top } else { Side::Bottom };
        let both_outline = e.a.role == FaceRole::Outline && e.b.role == FaceRole::Outline;
        let owner = [&e.a, &e.b].into_iter().find(|f| f.role == FaceRole::Wall).map(|f| self.labels.get(&f.feature).copied().unwrap_or(u32::MAX));
        if both_outline {
            // Exact: round line-line corners of the outline inside the box,
            // then drop the columns that fell outside.
            let mut rounded = 0;
            if let Some(outline) = solid.outline.as_mut() {
                for reg in outline.iter_mut() {
                    let n = reg.outer.len();
                    let mut k = 0;
                    while k < reg.outer.len() && rounded < 4 * n {
                        let corner = reg.outer[k].curve.end();
                        let inside = corner[0] >= bbox[0] - 2.0 * g.pitch && corner[0] <= bbox[2] + 2.0 * g.pitch && corner[1] >= bbox[1] - 2.0 * g.pitch && corner[1] <= bbox[3] + 2.0 * g.pitch;
                        if inside && sketch_regions_core::fillet_corner(&mut reg.outer, k, r, "fillet") {
                            rounded += 1;
                            k += 2;
                        } else {
                            k += 1;
                        }
                    }
                    reg.ids.insert("fillet".into());
                }
            }
            let mask = solid.mask();
            let rep = heightfield_core::vertical::round_mask(solid, side, &mask, r, bbox, true, label, label);
            let clipped = solid.clip_to_outline();
            let outline_desc = solid.outline.as_ref().map(|o| o.iter().map(|r| format!("{:.4}m2/{}p/{}h", r.area, r.outer.len(), r.holes.len())).collect::<Vec<_>>().join(",")).unwrap_or_default();
            return Ok(format!("footprint corner: {} outline corners rounded, {} cells removed, {} clipped; outline [{}]", rounded, rep.removed, clipped, outline_desc));
        }
        let owner = owner.ok_or("vertical edge without a wall owner")?;
        let field_label = match side {
            Side::Top => &solid.top_label,
            Side::Bottom => &solid.bot_label,
        };
        let (i0, i1, j0, j1) = g.cells_in([bbox[0] - g.pitch, bbox[1] - g.pitch, bbox[2] + g.pitch, bbox[3] + g.pitch]);
        let mut mask: Vec<bool> = field_label.iter().zip(solid.top.iter()).map(|(&l, &t)| !t.is_nan() && l / 16 == owner / 16).collect();
        let mut any = false;
        for j in j0..j1 {
            for i in i0..i1 {
                any |= mask[g.idx(i, j)];
            }
        }
        let kind = if any {
            "owner cells"
        } else {
            // A through cut: the hole itself.
            mask = solid.top.iter().map(|t| t.is_nan()).collect();
            "hole"
        };
        let rep = heightfield_core::vertical::round_mask(solid, side, &mask, r, bbox, false, label, label);
        Ok(format!("plan corner ({kind}): {} cells removed, {} added", rep.removed, rep.added))
    }

    fn run_boolean(&mut self, p: &Planned, op: BoolOp, tools: &[String], targets: &[String]) -> Result<String, String> {
        let targets = self.targets_of(targets)?;
        let label = p.index as u32 * 16;
        let mut summary = Vec::new();
        for tool in tools {
            let ts = self.bodies.remove(tool).ok_or_else(|| format!("tool body of feature {tool} does not exist"))?;
            let (mut lo_min, mut lo_max, mut n) = (f32::INFINITY, f32::NEG_INFINITY, 0usize);
            for k in 0..ts.grid.len() {
                if !ts.bot[k].is_nan() {
                    lo_min = lo_min.min(ts.bot[k]);
                    lo_max = lo_max.max(ts.bot[k]);
                    n += 1;
                }
            }
            summary.push(format!("tool {} columns, bottom {:.4}..{:.4} m", n, lo_min, lo_max));
            let body = Body { grid: ts.grid.clone(), lo: ts.bot.clone(), hi: ts.top.clone(), hi_label: None, lo_label: None };
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
                    Ok(Step::Loft { profiles, guides, op, targets }) => match self.run_loft(p, profiles, guides, *op, targets) {
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
