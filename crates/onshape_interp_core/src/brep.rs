//! Executing planned steps on exact surfaces.
//!
//! The same steps as [`crate::exec`], the same 2.5D reading of them and
//! the same column rules, but each body is a
//! [`brep_build_core::planmap::PlanMap`]: cells of the plan that name the
//! exact surfaces they end on. A plan map lifts to a boundary
//! representation ([`brep_build_core::lift`]), which is what a STEP file
//! is written from.
//!
//! Done: extrudes (any plane, any direction, all bounds), booleans of
//! bodies, the recorded rigid motions. Lofts and blends report that they
//! are not done here yet and leave the body as it was.

use std::collections::HashMap;

use brep_build_core::body::{Sweep, sweep};
use brep_build_core::lift::{LiftOptions, lift};
use brep_build_core::planmap::{Op, PlanMap, Tolerances};
use brep_core::topo::Solid;

use crate::SolvedSketch;
use crate::exec::{FeatureReport, Rigid, Status, sweep_extent};
use crate::plan::{BoolOp, Bound, Planned, RegionSel, Step};

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
            summary.push(format!(
                "{} -> {} cells{}{}",
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
        if std::env::var("BREP_DUMP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            == Some(p.index)
        {
            for c in &body.cells {
                eprintln!(
                    "BREP_DUMP {} body cell: top '{}' bot '{}' area {:.1} mm2, box (in) {:?}",
                    p.index,
                    body.sheets[c.top].name,
                    body.sheets[c.bot].name,
                    c.region.area * 1e6,
                    c.region
                        .bbox()
                        .map(|v| (v / 0.0254 * 100.0).round() / 100.0)
                );
            }
        }
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
                    Ok(Step::Loft { .. }) => {
                        Status::Failed("lofts are not done on exact surfaces yet".into())
                    }
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
            self.reports.push(FeatureReport {
                index: p.index,
                name: p.name.clone(),
                feature_type: p.feature_type.clone(),
                status,
            });
        }
    }
}
