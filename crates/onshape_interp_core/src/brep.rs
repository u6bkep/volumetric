//! Executing planned steps on exact surfaces.
//!
//! The same steps as [`crate::exec`], the same 2.5D reading of them and
//! the same column rules, but each body is a
//! [`brep_build_core::planmap::PlanMap`]: cells of the plan that name the
//! exact surfaces they end on. A plan map lifts to a boundary
//! representation ([`brep_build_core::lift`]), which is what a STEP file
//! is written from.
//!
//! Done: extrudes (any plane, any direction, all bounds), lofts (between
//! two profiles, with guides or ruled), booleans of bodies, the recorded
//! rigid motions. Blends report that they are not done here yet and
//! leave the body as it was.

use std::collections::HashMap;

use brep_build_core::body::{FaceSpec, Sweep, envelope, sweep};
use brep_build_core::chain::{Chain, Span};
use brep_build_core::lift::{LiftOptions, lift};
use brep_build_core::loft::{Profile, loft};
use brep_build_core::planmap::{Op, PlanMap, Tolerances, deep_point};
use brep_core::topo::Solid;
use sketch_regions_core::{Curve, Piece, Region, union};

use crate::SolvedSketch;
use crate::exec::{FeatureReport, Rigid, Status, sweep_extent};
use crate::plan::{BoolOp, Bound, Planned, RegionSel, Step};

type P3 = [f64; 3];

/// A piece of a sketch in space, `lift` taking the sketch's points
/// there.
fn span_of(sk: &SolvedSketch, curve: &Curve, lift: &dyn Fn(P3) -> P3) -> Span {
    match *curve {
        Curve::Line { a, b } => Span::Line {
            a: lift(sk.frame.to_3d(a)),
            b: lift(sk.frame.to_3d(b)),
        },
        Curve::Arc { c, r, a0, sweep } => Span::Arc {
            c: lift(sk.frame.to_3d(c)),
            x: sk.frame.ex,
            y: sk.frame.ey,
            r,
            a0,
            sweep,
        },
    }
}

/// Spans that follow each other on one line or circle as one, and
/// otherwise (an ellipse that came as short lines) as the smooth curve
/// through their ends.
fn one_side(spans: Vec<Span>) -> Vec<Span> {
    if spans.len() < 2 {
        return spans;
    }
    match Chain::joined(spans.clone(), 1e-6) {
        Ok(chain) if chain.spans.len() == 1 => chain.spans,
        _ => {
            let mut pts = vec![spans[0].start()];
            pts.extend(spans.iter().map(|s| s.end()));
            vec![Span::through(&pts)]
        }
    }
}

/// The closed chain around a loop of pieces, its corners where one
/// sketch entity ends and the next begins.
fn loop_chain(sk: &SolvedSketch, pieces: &[Piece]) -> Chain {
    let n = pieces.len();
    // From a change of entity, so that no side goes around the start.
    let start = (0..n)
        .find(|&i| pieces[i].id != pieces[(i + n - 1) % n].id)
        .unwrap_or(0);
    let mut sides: Vec<Vec<Span>> = Vec::new();
    for q in 0..n {
        let p = &pieces[(start + q) % n];
        // (Shorter than the plan can tell apart: no side of anything.)
        if p.curve.length() < 1e-6 {
            continue;
        }
        let span = span_of(sk, &p.curve, &|x| x);
        if !sides.is_empty() && q > 0 && p.id == pieces[(start + q - 1) % n].id {
            sides.last_mut().unwrap().push(span);
        } else {
            sides.push(vec![span]);
        }
    }
    let spans: Vec<Span> = sides.into_iter().flat_map(one_side).collect();
    let corners = vec![true; spans.len()];
    Chain::new(spans, corners, true)
}

pub struct BrepExec {
    /// Sketches by feature id.
    pub sketches: HashMap<String, SolvedSketch>,
    /// Bodies by the feature that created them.
    pub bodies: HashMap<String, PlanMap>,
    /// The first body created (what "default scope" operations act on).
    pub main: Option<String>,
    pub transforms: HashMap<String, Vec<Rigid>>,
    pub reports: Vec<FeatureReport>,
    pub tol: Tolerances,
    /// Regions smaller than this are artefacts of nearly coincident
    /// curves, not features (square metres).
    pub sliver: f64,
    /// Lowest and highest point of each extrude's body (by feature id),
    /// for the walls that lofts start from.
    pub extrude_z: HashMap<String, (f64, f64)>,
    /// Faces of lofts stay this close to the blend they are fitted to.
    pub fit: f64,
}

impl BrepExec {
    pub fn new(sketches: Vec<SolvedSketch>) -> BrepExec {
        BrepExec {
            sketches: sketches
                .into_iter()
                .map(|s| (s.feature_id.clone(), s))
                .collect(),
            bodies: HashMap::new(),
            main: None,
            transforms: HashMap::new(),
            reports: Vec::new(),
            tol: Tolerances::default(),
            sliver: 2e-6,
            extrude_z: HashMap::new(),
            fit: 5e-8,
        }
    }

    pub fn main_body(&self) -> Option<&PlanMap> {
        self.main.as_ref().and_then(|m| self.bodies.get(m))
    }

    /// The main body as solids, in the frame the features are in.
    pub fn solids(&self) -> Result<Vec<Solid>, String> {
        let body = self.main_body().ok_or("no body")?;
        lift(body, &LiftOptions::default())
    }

    fn targets_of(&self, targets: &[String]) -> Result<Vec<String>, String> {
        if targets.is_empty() {
            return self
                .main
                .clone()
                .map(|m| vec![m])
                .ok_or_else(|| "no body to operate on".to_string());
        }
        for t in targets {
            if !self.bodies.contains_key(t) {
                return Err(format!("target body of feature {t} does not exist"));
            }
        }
        Ok(targets.to_vec())
    }

    fn apply_to_targets(
        &mut self,
        targets: &[String],
        body: &PlanMap,
        op: Op,
    ) -> Result<String, String> {
        let mut summary = Vec::new();
        for t in targets {
            // Work on a copy: a failure leaves the body as it was.
            let mut solid = self.bodies[t].clone();
            let rep = solid.apply(body, op)?;
            let check = if rep.wrong.is_empty() {
                String::new()
            } else {
                let inch = |v: f64| (v / 0.0254 * 100.0).round() / 100.0;
                let groups: Vec<String> = rep
                    .wrong_groups()
                    .iter()
                    .take(4)
                    .map(|(what, n, bb, off)| {
                        format!(
                            "{n} by up to {:.2} mm in (in) {:?}: {what}",
                            off * 1e3,
                            bb.map(inch)
                        )
                    })
                    .collect();
                format!(
                    ", CHECK: {} of {} points wrong [{}]",
                    rep.wrong.len(),
                    rep.checked,
                    groups.join("; ")
                )
            };
            summary.push(format!(
                "{} -> {} cells{}{}{check}",
                rep.cells_before,
                rep.cells_after,
                if rep.creases > 0 {
                    format!(", {} creases", rep.creases)
                } else {
                    String::new()
                },
                if rep.split_faces > 0 {
                    format!(", {} split", rep.split_faces)
                } else {
                    String::new()
                },
            ));
            self.bodies.insert(t.clone(), solid);
        }
        Ok(summary.join("; "))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_extrude(
        &mut self,
        p: &Planned,
        regions: &[RegionSel],
        op: BoolOp,
        direction: Option<[f64; 3]>,
        flip: bool,
        end: &Bound,
        second: &Option<Bound>,
        targets: &[String],
    ) -> Result<String, String> {
        if regions.is_empty() {
            return Err("no regions".into());
        }
        let mut body: Option<PlanMap> = None;
        let mut missing = Vec::new();
        let mut taken = 0;
        for sel in regions {
            let found = self
                .sketches
                .get(&sel.sketch)
                .ok_or_else(|| format!("sketch {} not solved", sel.sketch))
                .and_then(|sk| {
                    sk.region_by_ids(&sel.ids, sel.pick)
                        .map(|r| (sk, r))
                        .ok_or_else(|| {
                            format!("sketch {} has no region bounded by {:?}", sk.name, sel.ids)
                        })
                });
            let (sk, region) = match found {
                Ok(x) => x,
                Err(e) => {
                    missing.push(e);
                    continue;
                }
            };
            if region.area < self.sliver {
                continue;
            }
            let mut dir = direction.unwrap_or(sk.normal);
            if flip {
                dir = [-dir[0], -dir[1], -dir[2]];
            }
            let (w0, w1) = sweep_extent(sk.frame.origin, sk.normal, dir, end, second)?;
            let part = sweep(
                &Sweep {
                    region: region.clone(),
                    origin: sk.frame.origin,
                    u: sk.frame.ex,
                    v: sk.frame.ey,
                    dir,
                    w0,
                    w1,
                    name: p.name.clone(),
                },
                &self.tol,
            )?;
            taken += 1;
            body = Some(match body {
                None => part,
                Some(mut all) => {
                    all.apply(&part, Op::Add)?;
                    all
                }
            });
        }
        if missing.len() == regions.len() {
            return Err(missing.join("; "));
        }
        let body = body.ok_or("every region is a sliver")?;
        let mut z_range = (f64::INFINITY, f64::NEG_INFINITY);
        for c in &body.cells {
            if let Some((lo, hi)) = deep_point(&c.region).and_then(|q| body.column(q)) {
                z_range = (z_range.0.min(lo), z_range.1.max(hi));
            }
        }
        self.extrude_z.insert(p.feature_id.clone(), z_range);
        let note = if missing.is_empty() {
            String::new()
        } else {
            format!(
                "; MISSING {} of {} regions: {}",
                missing.len(),
                regions.len(),
                missing.join("; ")
            )
        };
        match op {
            BoolOp::New => {
                let cells = body.cells.len();
                if self.main.is_none() {
                    self.main = Some(p.feature_id.clone());
                }
                self.bodies.insert(p.feature_id.clone(), body);
                Ok(format!("new body, {taken} regions, {cells} cells{note}"))
            }
            BoolOp::Add | BoolOp::Remove => {
                let targets = self.targets_of(targets)?;
                let op = if op == BoolOp::Add {
                    Op::Add
                } else {
                    Op::Remove
                };
                Ok(format!(
                    "{}{note}",
                    self.apply_to_targets(&targets, &body, op)?
                ))
            }
            BoolOp::Intersect => Err("intersect extrude".into()),
        }
    }

    fn region_of<'a>(&'a self, sel: &RegionSel) -> Result<(&'a SolvedSketch, &'a Region), String> {
        let sk = self
            .sketches
            .get(&sel.sketch)
            .ok_or_else(|| format!("sketch {} not solved", sel.sketch))?;
        let r = sk
            .region_by_ids(&sel.ids, sel.pick)
            .ok_or_else(|| format!("sketch {} has no region bounded by {:?}", sk.name, sel.ids))?;
        Ok((sk, r))
    }

    /// A profile of one or more regions of one sketch, merged.
    fn region_profile(&self, sels: &[RegionSel], name: &str) -> Result<(Profile, String), String> {
        let mut regions: Vec<Region> = Vec::new();
        let mut sketch: Option<&SolvedSketch> = None;
        let mut missing = 0;
        let mut why = Vec::new();
        for sel in sels {
            match self.region_of(sel) {
                Ok((sk, r)) => {
                    if sketch.is_some_and(|s| s.feature_id != sk.feature_id) {
                        return Err("profile regions from different sketches".into());
                    }
                    sketch = Some(sk);
                    regions.push(r.clone());
                }
                Err(e) => {
                    missing += 1;
                    why.push(e);
                }
            }
        }
        let sk = sketch.ok_or_else(|| format!("profile without regions: {}", why.join("; ")))?;
        let note = format!(
            "{} regions{}",
            regions.len(),
            if missing > 0 {
                format!(" ({missing} missing)")
            } else {
                String::new()
            }
        );
        let region = if regions.len() == 1 {
            regions.pop().unwrap()
        } else {
            let mut acc = vec![regions[0].clone()];
            for r in &regions[1..] {
                acc = union(&acc, std::slice::from_ref(r), 1e-7);
            }
            if acc.len() != 1 {
                return Err(format!(
                    "profile regions do not merge into one face ({} pieces)",
                    acc.len()
                ));
            }
            acc.pop().unwrap()
        };
        let chain = loop_chain(sk, &region.outer);
        Ok((
            Profile {
                chain,
                plane: (sk.frame.origin, sk.normal),
                cap: Some(FaceSpec::Flat {
                    region,
                    origin: sk.frame.origin,
                    u: sk.frame.ex,
                    v: sk.frame.ey,
                    name: name.to_string(),
                }),
            },
            note,
        ))
    }

    /// A profile that is a wall of an extrude: the stretch of a boundary
    /// entity of the extrude's region, at the bottom of the extrude and
    /// back along its top. The wall only stands as high as the solid
    /// did outside the cut, which its top follows.
    fn wall_profile(&self, sel: &RegionSel) -> Result<(Profile, String), String> {
        let w = sel.wall.clone().ok_or("not a wall")?;
        let (sk, region) = self.region_of(sel)?;
        let (zlo, zhi) = self
            .extrude_z
            .get(&w.extrude)
            .copied()
            .ok_or_else(|| format!("wall face: extrude {} not run yet", w.extrude))?;
        let pieces: Vec<&Piece> = region.pieces().filter(|p| p.id == w.entity).collect();
        if pieces.is_empty() {
            return Err(format!(
                "wall face: entity {} is not on the region boundary",
                w.entity
            ));
        }
        let level = |z: f64| move |p: P3| [p[0], p[1], z];
        let low = Chain::new(
            pieces
                .iter()
                .map(|p| span_of(sk, &p.curve, &level(zlo)))
                .collect(),
            vec![true; pieces.len()],
            false,
        );
        let total = low.total();
        let (mut t0, mut t1) = (w.t0, w.t1);
        if let Some(ends) = w.ends {
            let frac = |p: P3| low.nearest([p[0], p[1], zlo]).0 / total.max(1e-300);
            let (f0, f1) = (frac(ends[0]), frac(ends[1]));
            (t0, t1) = (f0.min(f1), f0.max(f1));
        }
        let bottom = low.part(t0.clamp(0.0, 1.0) * total, t1.clamp(0.0, 1.0) * total);
        if bottom.spans.is_empty() {
            return Err("wall face: nothing of the entity between its ends".into());
        }
        // The solid's height a little way outside the wall, along it.
        let main = self.main_body();
        let n = 24;
        let reach = bottom.total();
        let mut clipped = 0;
        let heights: Vec<f64> = (0..=n)
            .map(|q| {
                let s = reach * q as f64 / n as f64;
                let (here, ahead) = (bottom.at(s), bottom.at((s + 1e-6).min(reach)));
                let back = bottom.at((s - 1e-6).max(0.0));
                let d = [ahead[0] - back[0], ahead[1] - back[1]];
                let l = d[0].hypot(d[1]).max(1e-300);
                let across = [-d[1] / l, d[0] / l];
                let inside = sk.frame.to_2d([
                    here[0] + 2e-3 * across[0],
                    here[1] + 2e-3 * across[1],
                    sk.frame.origin[2],
                ]);
                let side = if region.contains(inside) { -1.0 } else { 1.0 };
                let out = [
                    here[0] + side * 2e-3 * across[0],
                    here[1] + side * 2e-3 * across[1],
                ];
                match main.and_then(|m| m.column(out)) {
                    Some((_, top)) if top < zhi - 1e-6 => {
                        clipped += 1;
                        top.max(zlo)
                    }
                    _ => zhi,
                }
            })
            .collect();
        let flat = heights.iter().all(|h| (h - heights[0]).abs() < 1e-9);
        let top: Vec<Span> = if flat {
            let lifted: Vec<Span> = bottom
                .spans
                .iter()
                .map(|s| match s.clone() {
                    Span::Line { a, b } => Span::Line {
                        a: [a[0], a[1], heights[0]],
                        b: [b[0], b[1], heights[0]],
                    },
                    Span::Arc {
                        c,
                        x,
                        y,
                        r,
                        a0,
                        sweep,
                    } => Span::Arc {
                        c: [c[0], c[1], heights[0]],
                        x,
                        y,
                        r,
                        a0,
                        sweep,
                    },
                    free => free,
                })
                .collect();
            lifted.iter().rev().map(Span::reversed).collect()
        } else {
            // Through the heights, from the far end back.
            let pts: Vec<P3> = (0..=n)
                .rev()
                .map(|q| {
                    let p = bottom.at(reach * q as f64 / n as f64);
                    [p[0], p[1], heights[q]]
                })
                .collect();
            let lines: Vec<Span> = pts
                .windows(2)
                .map(|w| Span::Line { a: w[0], b: w[1] })
                .filter(|s| s.len() > 1e-9)
                .collect();
            Chain::joined(lines, 1e-6)?.spans
        };
        let (first, last) = (bottom.start(), bottom.end());
        let mut spans = bottom.spans.clone();
        let mut corners = vec![false; spans.len()];
        corners[0] = true;
        let close = |from: P3, to: P3, spans: &mut Vec<Span>, corners: &mut Vec<bool>| {
            let line = Span::Line { a: from, b: to };
            if line.len() > 1e-9 {
                spans.push(line);
                corners.push(false);
            }
        };
        close(last, top[0].start(), &mut spans, &mut corners);
        let at = spans.len();
        spans.extend(top.iter().cloned());
        corners.extend(vec![false; top.len()]);
        corners[at] = true;
        close(top[top.len() - 1].end(), first, &mut spans, &mut corners);
        // The wall's plane: through its first point, across the wall.
        let d = [last[0] - first[0], last[1] - first[1]];
        let l = d[0].hypot(d[1]).max(1e-12);
        let note = format!(
            "wall of {} along {} ({t0:.2}..{t1:.2}), z {:.3}..{:.3} in{}",
            w.extrude,
            w.entity,
            zlo / 0.0254,
            zhi / 0.0254,
            if clipped > 0 {
                format!(", top clipped at {clipped}/{} points", n + 1)
            } else {
                String::new()
            }
        );
        Ok((
            Profile {
                chain: Chain::new(spans, corners, true),
                plane: (first, [-d[1] / l, d[0] / l, 0.0]),
                cap: None,
            },
            note,
        ))
    }

    /// A guide: its curves end to end.
    fn guide(&self, curves: &[(String, String)]) -> Result<Chain, String> {
        let mut spans = Vec::new();
        for (sid, id) in curves {
            let sk = self
                .sketches
                .get(sid)
                .ok_or_else(|| format!("guide sketch {sid} not solved"))?;
            let run: Vec<Span> = sk
                .pieces
                .iter()
                .chain(sk.construction.iter())
                .filter(|p| &p.id == id)
                .map(|p| span_of(sk, &p.curve, &|x| x))
                .collect();
            if run.is_empty() {
                return Err(format!("guide curve {id} not in sketch {}", sk.name));
            }
            spans.extend(one_side(run));
        }
        Chain::joined(spans, 1e-3)
    }

    fn run_loft(
        &mut self,
        p: &Planned,
        profiles: &[Vec<RegionSel>],
        guides: &[Vec<(String, String)>],
        op: BoolOp,
        targets: &[String],
    ) -> Result<String, String> {
        if profiles.len() != 2 {
            return Err(format!("loft with {} profiles", profiles.len()));
        }
        let make = |sels: &[RegionSel], end: &str| -> Result<(Profile, String), String> {
            if sels.len() == 1 && sels[0].wall.is_some() {
                self.wall_profile(&sels[0])
            } else {
                self.region_profile(sels, &format!("{}:{end}", p.name))
            }
        };
        let (a, note_a) = make(&profiles[0], "start")?;
        let (b, note_b) = make(&profiles[1], "end")?;
        let mut note = format!("; A {note_a}, B {note_b}");
        let mut chains = Vec::new();
        for g in guides {
            match self.guide(g) {
                Ok(c) => chains.push(c),
                Err(e) => note.push_str(&format!("; guide: {e}")),
            }
        }
        let (faces, how) = loft(&p.name, &a, &b, &chains, self.fit)?;
        note.push_str(&how);
        let body = envelope(&faces, &self.tol);
        let body = body?;
        match op {
            BoolOp::New => {
                let cells = body.cells.len();
                if self.main.is_none() {
                    self.main = Some(p.feature_id.clone());
                }
                self.bodies.insert(p.feature_id.clone(), body);
                Ok(format!("new body, {cells} cells{note}"))
            }
            BoolOp::Add | BoolOp::Remove => {
                let targets = self.targets_of(targets)?;
                let op = if op == BoolOp::Add {
                    Op::Add
                } else {
                    Op::Remove
                };
                Ok(format!(
                    "{}{note}",
                    self.apply_to_targets(&targets, &body, op)?
                ))
            }
            BoolOp::Intersect => Err("intersect loft".into()),
        }
    }

    fn run_boolean(
        &mut self,
        op: BoolOp,
        tools: &[String],
        targets: &[String],
    ) -> Result<String, String> {
        let targets = self.targets_of(targets)?;
        let op = match op {
            BoolOp::Add => Op::Add,
            BoolOp::Remove => Op::Remove,
            _ => return Err("boolean intersect / new".into()),
        };
        let mut summary = Vec::new();
        for tool in tools {
            let body = self
                .bodies
                .get(tool)
                .cloned()
                .ok_or_else(|| format!("tool body of feature {tool} does not exist"))?;
            summary.push(self.apply_to_targets(&targets, &body, op)?);
            self.bodies.remove(tool);
        }
        Ok(summary.join("; "))
    }

    /// Execute all planned steps in order.
    pub fn run(&mut self, planned: &[Planned]) {
        for p in planned {
            let started = std::time::Instant::now();
            let done = |r: Result<String, String>| match r {
                Ok(s) => Status::Applied(s),
                Err(e) => Status::Failed(e),
            };
            let status = if p.suppressed {
                Status::Suppressed
            } else {
                match &p.step {
                    Err(e) => Status::Failed(e.clone()),
                    Ok(Step::Nothing) => Status::Nothing,
                    Ok(Step::Extrude {
                        regions,
                        op,
                        direction,
                        flip,
                        end,
                        second,
                        targets,
                    }) => done(
                        self.run_extrude(p, regions, *op, *direction, *flip, end, second, targets),
                    ),
                    Ok(Step::Boolean { op, tools, targets }) => {
                        done(self.run_boolean(*op, tools, targets))
                    }
                    Ok(Step::Loft {
                        profiles,
                        guides,
                        op,
                        targets,
                    }) => done(self.run_loft(p, profiles, guides, *op, targets)),
                    Ok(Step::Fillet { .. }) | Ok(Step::Chamfer { .. }) => {
                        Status::Failed("blends are not done on exact surfaces yet".into())
                    }
                    Ok(Step::Rotate {
                        body,
                        axis_point,
                        axis_dir,
                        angle,
                    }) => {
                        self.transforms
                            .entry(body.clone())
                            .or_default()
                            .push(Rigid::Rotate {
                                point: *axis_point,
                                dir: *axis_dir,
                                angle: *angle,
                            });
                        Status::Applied(format!("rotation {:.1} deg recorded", angle.to_degrees()))
                    }
                    Ok(Step::Translate { body, delta }) => {
                        self.transforms
                            .entry(body.clone())
                            .or_default()
                            .push(Rigid::Translate(*delta));
                        Status::Applied(format!(
                            "translation ({:.4}, {:.4}, {:.4}) m recorded",
                            delta[0], delta[1], delta[2]
                        ))
                    }
                }
            };
            let status = match status {
                Status::Applied(s) if started.elapsed().as_secs_f64() > 0.5 => {
                    Status::Applied(format!("{s}; {:.1} s", started.elapsed().as_secs_f64()))
                }
                other => other,
            };
            self.reports.push(FeatureReport {
                index: p.index,
                name: p.name.clone(),
                feature_type: p.feature_type.clone(),
                status,
            });
        }
    }
}
