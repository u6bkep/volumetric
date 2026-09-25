//! Feature JSON -> executable steps, with every geometry reference
//! resolved through the [`IdTable`].

use std::collections::HashMap;

use fs_expr::{Scope, Value};
use serde_json::Value as Json;

use crate::idtable::{FaceRef, IdTable, Ref, RegionEntry, SideRef};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoolOp {
    New,
    Add,
    Remove,
    Intersect,
}

/// One end of an extrude sweep, as a signed distance along the sketch
/// normal from the sketch plane.
#[derive(Clone, Debug, PartialEq)]
pub enum Bound {
    Blind(f64),
    ThroughAll,
    /// Up to a plane parallel to the sketch plane.
    UpToPlane { origin: [f64; 3], normal: [f64; 3] },
    UpToPoint([f64; 3]),
}

#[derive(Clone, Debug)]
pub struct RegionSel {
    pub sketch: String,
    pub ids: Vec<String>,
    pub pick: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct EdgeSel {
    pub a: FaceRef,
    pub b: FaceRef,
    pub side: SideRef,
    pub hint: Option<[f64; 3]>,
    /// Bounding box of the blend face at the resolved configuration
    /// (model frame, metres): edge cells are taken from around it.
    pub hint_box: Option<[f64; 6]>,
}

#[derive(Clone, Debug)]
pub enum Step {
    Extrude {
        regions: Vec<RegionSel>,
        op: BoolOp,
        /// Sweep direction: the sketch normal unless a line was given.
        direction: Option<[f64; 3]>,
        flip: bool,
        end: Bound,
        /// Second direction (opposite side of the plane), if any.
        second: Option<Bound>,
        targets: Vec<String>,
    },
    Loft {
        /// One entry per profile; a profile is the union of its regions.
        profiles: Vec<Vec<RegionSel>>,
        op: BoolOp,
        targets: Vec<String>,
    },
    Fillet {
        edges: Vec<EdgeSel>,
        radius: f64,
    },
    Chamfer {
        edges: Vec<EdgeSel>,
        /// Offset along the cap face and along the wall.
        width_cap: f64,
        width_wall: f64,
    },
    Boolean {
        op: BoolOp,
        tools: Vec<String>,
        targets: Vec<String>,
    },
    Rotate {
        body: String,
        axis_point: [f64; 3],
        axis_dir: [f64; 3],
        angle: f64,
    },
    Translate {
        body: String,
        delta: [f64; 3],
    },
    /// Nothing to execute (sketches, planes, variables, curves, measures).
    Nothing,
}

#[derive(Clone, Debug)]
pub struct Planned {
    pub index: usize,
    pub feature_id: String,
    pub name: String,
    pub feature_type: String,
    pub suppressed: bool,
    pub step: Result<Step, String>,
    /// Ids the table could not resolve (the step is an error then).
    pub unresolved: Vec<String>,
}

fn param<'a>(feature: &'a Json, id: &str) -> Option<&'a Json> {
    feature["message"]["parameters"]
        .as_array()?
        .iter()
        .find(|p| p["message"]["parameterId"] == id)
        .map(|p| &p["message"])
}

fn param_bool(feature: &Json, id: &str) -> Option<bool> {
    param(feature, id)?["value"].as_bool()
}

fn param_enum(feature: &Json, id: &str) -> Option<String> {
    param(feature, id)?["value"].as_str().map(|s| s.to_string())
}

fn param_expr(feature: &Json, id: &str) -> Option<String> {
    param(feature, id)?["expression"].as_str().map(|s| s.to_string())
}

fn param_length(feature: &Json, id: &str, scope: &dyn Scope) -> Result<f64, String> {
    let e = param_expr(feature, id).ok_or_else(|| format!("missing parameter {id}"))?;
    fs_expr::evaluate(&e, scope).map_err(|err| format!("{id} = {e}: {}", err.0))?.as_length_m().map_err(|err| format!("{id} = {e}: {}", err.0))
}

fn param_angle(feature: &Json, id: &str, scope: &dyn Scope) -> Result<f64, String> {
    let e = param_expr(feature, id).ok_or_else(|| format!("missing parameter {id}"))?;
    fs_expr::evaluate(&e, scope).map_err(|err| format!("{id} = {e}: {}", err.0))?.as_angle_rad().map_err(|err| format!("{id} = {e}: {}", err.0))
}

/// `(feature id of the query, geometry ids)` per query of a query-list
/// parameter.
fn queries(p: &Json) -> Vec<(Option<String>, Vec<String>)> {
    p["queries"]
        .as_array()
        .map(|qs| {
            qs.iter()
                .map(|q| {
                    let m = &q["message"];
                    (
                        m["featureId"].as_str().filter(|s| !s.is_empty()).map(|s| s.to_string()),
                        m["geometryIds"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()).unwrap_or_default(),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

fn query_ids(feature: &Json, id: &str) -> Vec<(Option<String>, Vec<String>)> {
    param(feature, id).map(queries).unwrap_or_default()
}

/// Whether the feature is suppressed at the configuration: the plain flag,
/// a configured boolean, or a suppression expression (active when true).
pub fn is_suppressed(feature: &Json, cfg: &HashMap<String, Value>, scope: &dyn Scope) -> Result<bool, String> {
    let m = &feature["message"];
    let ss = &m["suppressionState"];
    match ss["typeName"].as_str() {
        Some("BTMSuppressionStateConfigured") => {
            let pid = ss["message"]["configurationParameterId"].as_str().unwrap_or("");
            let current = cfg.get(pid).ok_or_else(|| format!("configuration input {pid} not set"))?;
            for v in ss["message"]["values"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
                let vm = &v["message"];
                let matches = match (vm.get("booleanValue"), current) {
                    (Some(b), Value::Bool(c)) => b.as_bool() == Some(*c),
                    _ => false,
                };
                if matches {
                    return Ok(vm["value"]["message"]["value"].as_bool().unwrap_or(false));
                }
            }
            Ok(m["suppressed"].as_bool().unwrap_or(false))
        }
        Some("BTMSuppressionStateExpression") => {
            // An expression that fails to evaluate (an undefined variable)
            // leaves the feature in an error state in Onshape: not applied.
            let e = ss["message"]["value"]["message"]["expression"].as_str().unwrap_or("true");
            match fs_expr::evaluate(e, scope).and_then(|v| v.as_bool()) {
                Ok(active) => Ok(!active),
                Err(_) => Ok(true),
            }
        }
        _ => Ok(m["suppressed"].as_bool().unwrap_or(false)),
    }
}

struct Resolver<'a> {
    table: &'a IdTable,
    feature: String,
    unresolved: Vec<String>,
}

/// `(parameter, query index, id index, id)` for every id of a query-list
/// parameter.
fn slot_ids(feature: &Json, param_id: &str) -> Vec<(usize, usize, String)> {
    let mut out = Vec::new();
    for (qi, (_, ids)) in query_ids(feature, param_id).into_iter().enumerate() {
        for (ii, id) in ids.into_iter().enumerate() {
            out.push((qi, ii, id));
        }
    }
    out
}

impl Resolver<'_> {
    fn get_slot(&mut self, param: &str, qi: usize, ii: usize, id: &str) -> Option<Ref> {
        let key = IdTable::slot_key(&self.feature, param, qi, ii);
        match self.table.lookup(&key, id) {
            Some(Ref::Unresolved { .. }) | None => {
                self.unresolved.push(if id.is_empty() { key } else { id.to_string() });
                None
            }
            Some(r) => Some(r.clone()),
        }
    }

    fn get(&mut self, param: &str, qi: usize, ii: usize, id: &str) -> Option<Ref> {
        self.get_slot(param, qi, ii, id)
    }

    /// First id of a parameter (a single-entity query).
    fn first(&mut self, feature: &Json, param: &str) -> Option<Ref> {
        let ids = slot_ids(feature, param);
        let (qi, ii, id) = ids.first().cloned().unwrap_or((0, 0, String::new()));
        self.get(param, qi, ii, &id)
    }

    fn list(&self, param: &str, item: Option<usize>) -> Option<Vec<RegionSel>> {
        match self.table.slots.get(&IdTable::list_key(&self.feature, param, item)).map(|e| &e.r) {
            Some(Ref::RegionList { regions }) => Some(regions.iter().map(|r: &RegionEntry| RegionSel { sketch: r.sketch.clone(), ids: r.ids.clone(), pick: r.pick }).collect()),
            _ => None,
        }
    }

    fn regions_of(&mut self, feature: &Json, param: &str) -> Vec<RegionSel> {
        if let Some(list) = self.list(param, None) {
            return list;
        }
        let mut out = Vec::new();
        for (qi, ii, id) in slot_ids(feature, param) {
            match self.get(param, qi, ii, &id) {
                Some(Ref::Region { sketch, ids, pick }) => out.push(RegionSel { sketch, ids, pick }),
                Some(_) => self.unresolved.push(format!("{id} (not a region)")),
                None => {}
            }
        }
        out
    }

    fn regions_in(&mut self, param: &str, qs: &[(Option<String>, Vec<String>)], q_offset: usize) -> Vec<RegionSel> {
        let mut out = Vec::new();
        for (qi, (_, ids)) in qs.iter().enumerate() {
            for (ii, id) in ids.iter().enumerate() {
                match self.get(param, qi + q_offset, ii, id) {
                    Some(Ref::Region { sketch, ids, pick }) => out.push(RegionSel { sketch, ids, pick }),
                    Some(_) => self.unresolved.push(format!("{id} (not a region)")),
                    None => {}
                }
            }
        }
        out
    }

    fn bodies_of(&mut self, feature: &Json, param: &str) -> Vec<String> {
        let mut out = Vec::new();
        for (qi, ii, id) in slot_ids(feature, param) {
            match self.get(param, qi, ii, &id) {
                Some(Ref::Body { feature }) => out.push(feature),
                Some(_) => self.unresolved.push(format!("{id} (not a body)")),
                None => {}
            }
        }
        out
    }

    fn edges_of(&mut self, feature: &Json, param: &str) -> Vec<EdgeSel> {
        // Per-feature edges from the blend-face resolution win.
        if let Some(list) = self.table.edges.get(&self.feature) {
            return list
                .iter()
                .map(|e| EdgeSel { a: e.a.clone(), b: e.b.clone(), side: e.side, hint: None, hint_box: e.hint_box })
                .collect();
        }
        let mut out = Vec::new();
        for (qi, ii, id) in slot_ids(feature, param) {
            match self.get(param, qi, ii, &id) {
                Some(Ref::Edge { a, b, side, hint }) => out.push(EdgeSel { a, b, side, hint, hint_box: None }),
                Some(_) => self.unresolved.push(format!("{id} (not an edge)")),
                None => {}
            }
        }
        out
    }
}

/// Resolve a sketch point / line reference to world geometry.
pub trait GeomLookup {
    fn point(&self, sketch: &str, id: &str) -> Option<[f64; 3]>;
    fn line(&self, sketch: &str, id: &str) -> Option<([f64; 3], [f64; 3])>;
}

fn bool_op(s: &str) -> Result<BoolOp, String> {
    match s {
        "NEW" => Ok(BoolOp::New),
        "ADD" => Ok(BoolOp::Add),
        "REMOVE" => Ok(BoolOp::Remove),
        "INTERSECT" => Ok(BoolOp::Intersect),
        "SUBTRACTION" => Ok(BoolOp::Remove),
        "UNION" => Ok(BoolOp::Add),
        "INTERSECTION" => Ok(BoolOp::Intersect),
        other => Err(format!("operation {other}")),
    }
}

fn bound(
    feature: &Json,
    kind: &str,
    depth_id: &str,
    face_id: &str,
    vertex_id: &str,
    scope: &dyn Scope,
    res: &mut Resolver,
    geom: &dyn GeomLookup,
) -> Result<Bound, String> {
    match kind {
        "BLIND" => Ok(Bound::Blind(param_length(feature, depth_id, scope)?)),
        "THROUGH_ALL" => Ok(Bound::ThroughAll),
        "UP_TO_SURFACE" | "UP_TO_FACE" => match res.first(feature, face_id) {
            Some(Ref::Plane { origin, normal, .. }) => Ok(Bound::UpToPlane { origin, normal }),
            Some(_) => Err("up-to-surface target is not a plane".into()),
            None => Err("unresolved up-to-surface target".into()),
        },
        "UP_TO_VERTEX" => match res.first(feature, vertex_id) {
            Some(Ref::SketchPoint { sketch, id: pid }) => {
                geom.point(&sketch, &pid).map(Bound::UpToPoint).ok_or_else(|| format!("sketch point {pid} of {sketch} not found"))
            }
            Some(_) => Err("up-to-vertex target is not a sketch point".into()),
            None => Err("unresolved up-to-vertex target".into()),
        },
        other => Err(format!("end bound {other}")),
    }
}

/// Plan one feature.
pub fn plan_feature(
    index: usize,
    feature: &Json,
    cfg: &HashMap<String, Value>,
    scope: &dyn Scope,
    table: &IdTable,
    geom: &dyn GeomLookup,
) -> Planned {
    let m = &feature["message"];
    let feature_id = m["featureId"].as_str().unwrap_or("").to_string();
    let name = m["name"].as_str().unwrap_or("").to_string();
    let feature_type = m["featureType"].as_str().unwrap_or("").to_string();
    let mut res = Resolver { table, feature: feature_id.clone(), unresolved: Vec::new() };
    let suppressed = match is_suppressed(feature, cfg, scope) {
        Ok(s) => s,
        Err(e) => {
            return Planned { index, feature_id, name, feature_type, suppressed: false, step: Err(e), unresolved: Vec::new() };
        }
    };
    let step: Result<Step, String> = if suppressed {
        Ok(Step::Nothing)
    } else {
        match feature_type.as_str() {
            "newSketch" | "cPlane" | "assignVariable" | "compositeCurve" | "measureDistance" => Ok(Step::Nothing),
            "extrude" => (|| {
                let op = bool_op(&param_enum(feature, "operationType").unwrap_or_default())?;
                let regions = res.regions_of(feature, "entities");
                let flip = param_bool(feature, "oppositeDirection").unwrap_or(false);
                let end = bound(feature, &param_enum(feature, "endBound").unwrap_or_default(), "depth", "endBoundEntityFace", "endBoundEntityVertex", scope, &mut res, geom)?;
                let second = if param_bool(feature, "hasSecondDirection").unwrap_or(false) {
                    Some(bound(
                        feature,
                        &param_enum(feature, "secondDirectionBound").unwrap_or_default(),
                        "secondDirectionDepth",
                        "secondDirectionBoundEntityFace",
                        "secondDirectionBoundEntityVertex",
                        scope,
                        &mut res,
                        geom,
                    )?)
                } else {
                    None
                };
                if param_bool(feature, "symmetric").unwrap_or(false) {
                    return Err("symmetric extrude".into());
                }
                if param_bool(feature, "startOffset").unwrap_or(false) || param_bool(feature, "hasOffset").unwrap_or(false) {
                    return Err("extrude with offset".into());
                }
                if param_bool(feature, "hasDraft").unwrap_or(false) {
                    return Err("extrude with draft".into());
                }
                let direction = if param_bool(feature, "hasExtrudeDirection").unwrap_or(false) {
                    match res.first(feature, "extrudeDirection") {
                        Some(Ref::SketchLine { sketch, id: lid }) => {
                            let (a, b) = geom.line(&sketch, &lid).ok_or_else(|| format!("sketch line {lid} of {sketch} not found"))?;
                            Some(crate::unit3([b[0] - a[0], b[1] - a[1], b[2] - a[2]]))
                        }
                        Some(_) => return Err("extrude direction is not a sketch line".into()),
                        None => return Err("unresolved extrude direction".into()),
                    }
                } else {
                    None
                };
                let targets = if param_bool(feature, "defaultScope").unwrap_or(true) { Vec::new() } else { res.bodies_of(feature, "booleanScope") };
                Ok(Step::Extrude { regions, op, direction, flip, end, second, targets })
            })(),
            "loft" => (|| {
                let op = bool_op(&param_enum(feature, "operationType").unwrap_or_default())?;
                // Profiles are one array item each; slot query indices run
                // over the whole array in order.
                let mut profiles: Vec<Vec<RegionSel>> = Vec::new();
                let mut q_offset = 0;
                if let Some(items) = param(feature, "sheetProfilesArray").and_then(|p| p["items"].as_array()) {
                    for (item, it) in items.iter().enumerate() {
                        let ps = it["message"]["parameters"].as_array().cloned().unwrap_or_default();
                        for p in ps {
                            if p["message"]["parameterId"] == "sheetProfileEntities" {
                                let qs = queries(&p["message"]);
                                let regs = match res.list("sheetProfileEntities", Some(item)) {
                                    Some(list) => list,
                                    None => res.regions_in("sheetProfileEntities", &qs, q_offset),
                                };
                                q_offset += qs.len();
                                if !regs.is_empty() {
                                    profiles.push(regs);
                                }
                            }
                        }
                    }
                }
                let targets = if param_bool(feature, "defaultScope").unwrap_or(true) { Vec::new() } else { res.bodies_of(feature, "booleanScope") };
                Ok(Step::Loft { profiles, op, targets })
            })(),
            "fillet" => (|| {
                if param_enum(feature, "filletType").as_deref() != Some("EDGE") {
                    return Err("non-edge fillet".into());
                }
                if param_bool(feature, "isVariable").unwrap_or(false) || param_bool(feature, "isPartial").unwrap_or(false) {
                    return Err("variable or partial fillet".into());
                }
                let radius = param_length(feature, "radius", scope)?;
                let edges = res.edges_of(feature, "entities");
                Ok(Step::Fillet { edges, radius })
            })(),
            "chamfer" => (|| {
                let ty = param_enum(feature, "chamferType").unwrap_or_default();
                let (w1, w2) = match ty.as_str() {
                    "EQUAL_OFFSETS" => {
                        let w = param_length(feature, "width", scope)?;
                        (w, w)
                    }
                    "TWO_OFFSETS" => (param_length(feature, "width1", scope)?, param_length(feature, "width2", scope)?),
                    "OFFSET_ANGLE" => {
                        let w = param_length(feature, "width1", scope)?;
                        let a = param_angle(feature, "angle", scope)?;
                        (w, w * a.tan())
                    }
                    other => return Err(format!("chamfer type {other}")),
                };
                let flip = param_bool(feature, "oppositeDirection").unwrap_or(false);
                let (width_cap, width_wall) = if flip { (w2, w1) } else { (w1, w2) };
                let edges = res.edges_of(feature, "entities");
                Ok(Step::Chamfer { edges, width_cap, width_wall })
            })(),
            "booleanBodies" => (|| {
                let op = bool_op(&param_enum(feature, "operationType").unwrap_or_default())?;
                let tools = res.bodies_of(feature, "tools");
                let targets = res.bodies_of(feature, "targets");
                Ok(Step::Boolean { op, tools, targets })
            })(),
            "transform" => (|| {
                let bodies = res.bodies_of(feature, "entities");
                let body = bodies.first().cloned().ok_or("transform without a body")?;
                match param_enum(feature, "transformType").unwrap_or_default().as_str() {
                    "ROTATION" => {
                        let (a, b) = match res.first(feature, "transformAxis") {
                            Some(Ref::SketchLine { sketch, id: lid }) => geom.line(&sketch, &lid).ok_or_else(|| format!("sketch line {lid} of {sketch} not found"))?,
                            Some(_) => return Err("rotation axis is not a sketch line".into()),
                            None => return Err("unresolved rotation axis".into()),
                        };
                        let mut angle = param_angle(feature, "angle", scope)?;
                        if param_bool(feature, "oppositeDirection").unwrap_or(false) {
                            angle = -angle;
                        }
                        Ok(Step::Rotate { body, axis_point: a, axis_dir: crate::unit3([b[0] - a[0], b[1] - a[1], b[2] - a[2]]), angle })
                    }
                    "TRANSLATION_3D" => {
                        let d = [param_length(feature, "dx", scope)?, param_length(feature, "dy", scope)?, param_length(feature, "dz", scope)?];
                        Ok(Step::Translate { body, delta: d })
                    }
                    other => Err(format!("transform type {other}")),
                }
            })(),
            other => Err(format!("feature type {other}")),
        }
    };
    let unresolved = res.unresolved;
    // Region lists tolerate gaps (the resolved regions still cut); anything
    // else unresolved fails the feature.
    let partial_ok = matches!(&step, Ok(Step::Extrude { regions, .. }) if !regions.is_empty()) || matches!(&step, Ok(Step::Loft { profiles, .. }) if profiles.len() >= 2);
    let step = if !unresolved.is_empty() && step.is_ok() && !partial_ok { Err(format!("unresolved ids: {}", unresolved.join(", "))) } else { step };
    Planned { index, feature_id, name, feature_type, suppressed, step, unresolved }
}

/// Plan every feature.
pub fn plan_all(features: &[Json], cfg: &HashMap<String, Value>, scope: &dyn Scope, table: &IdTable, geom: &dyn GeomLookup) -> Vec<Planned> {
    features.iter().enumerate().map(|(i, f)| plan_feature(i, f, cfg, scope, table, geom)).collect()
}

/// Every query slot of a feature: `(parameter, query index, id index, query
/// feature id, geometry id)`; array parameters count their queries in order.
pub fn feature_slots(feature: &Json) -> Vec<(String, usize, usize, Option<String>, String)> {
    let mut out = Vec::new();
    fn walk(p: &Json, counts: &mut HashMap<String, usize>, out: &mut Vec<(String, usize, usize, Option<String>, String)>) {
        let m = &p["message"];
        let pid = m["parameterId"].as_str().unwrap_or("").to_string();
        if p["typeName"] == "BTMParameterQueryList" {
            let base = *counts.get(&pid).unwrap_or(&0);
            let qs = queries(m);
            for (qi, (fid, ids)) in qs.iter().enumerate() {
                for (ii, id) in ids.iter().enumerate() {
                    out.push((pid.clone(), base + qi, ii, fid.clone(), id.clone()));
                }
            }
            counts.insert(pid, base + qs.len());
        } else if p["typeName"] == "BTMParameterArray" {
            for it in m["items"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
                for pp in it["message"]["parameters"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
                    walk(pp, counts, out);
                }
            }
        }
    }
    let mut counts = HashMap::new();
    for p in feature["message"]["parameters"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        walk(p, &mut counts, &mut out);
    }
    out
}

/// Every deterministic id a feature's queries mention, with the parameter
/// it appears in and the query's feature id (a sketch, for region queries).
pub fn feature_query_ids(feature: &Json) -> Vec<(String, Option<String>, String)> {
    let mut out = Vec::new();
    fn walk(p: &Json, out: &mut Vec<(String, Option<String>, String)>) {
        let m = &p["message"];
        let pid = m["parameterId"].as_str().unwrap_or("").to_string();
        if p["typeName"] == "BTMParameterQueryList" {
            for (fid, ids) in queries(m) {
                for id in ids {
                    out.push((pid.clone(), fid.clone(), id));
                }
            }
        } else if p["typeName"] == "BTMParameterArray" {
            for it in m["items"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
                for pp in it["message"]["parameters"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
                    walk(pp, out);
                }
            }
        }
    }
    for p in feature["message"]["parameters"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        walk(p, &mut out);
    }
    out
}
