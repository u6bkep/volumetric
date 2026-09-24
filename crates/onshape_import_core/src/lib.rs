//! Turns an Onshape Features API `newSketch` feature (the JSON returned by
//! `GET /partstudios/.../features`) into a [`sketch_solve_core::Sketch`].
//!
//! The JSON carries the solved geometry at the configuration it was fetched
//! at, which becomes the solver seed, and every constraint with its
//! dimension expression. Constraint side / direction ambiguities that
//! Onshape stores implicitly (which side of a line a distance is measured
//! to, external vs internal tangency, the branch of an angle) are chosen
//! so the seed satisfies the constraint, then kept as the dimensions move.
//!
//! References to geometry outside the sketch (deterministic ids of vertices
//! and edges of earlier features) are resolved by an [`ExternalResolver`]
//! into fixed entities in sketch coordinates. Unresolved references leave
//! their constraints out and are reported in [`SketchImport::skipped`].
//!
//! Sketch coordinates and dimension values are in metres and radians,
//! matching the JSON.

pub mod sketch3d;

use std::collections::HashMap;
use std::fmt;

use fs_expr::Scope;
use serde_json::Value as Json;
use sketch_solve_core::{Constraint, DistanceMode, EntId, Entity, PointRef, Sketch};

#[derive(Clone, Debug, PartialEq)]
pub struct ImportError(pub String);

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ImportError {}

/// Geometry of an external reference, in the sketch's own plane coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExternalGeom {
    Point([f64; 2]),
    Line([f64; 2], [f64; 2]),
}

pub trait ExternalResolver {
    fn resolve(&self, id: &str) -> Option<ExternalGeom>;
}

impl ExternalResolver for HashMap<String, ExternalGeom> {
    fn resolve(&self, id: &str) -> Option<ExternalGeom> {
        self.get(id).copied()
    }
}

/// Which parameter of a constraint a dimension expression feeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DimKind {
    Length,
    Distance,
    Radius,
    Diameter,
    MajorDiameter,
    MinorDiameter,
    Angle,
}

#[derive(Clone, Debug)]
pub struct Dimension {
    /// Index into `sketch.constraints`.
    pub constraint: usize,
    pub kind: DimKind,
    pub expression: String,
}

/// A constraint that could not be imported, with the reason.
#[derive(Clone, Debug)]
pub struct Skipped {
    pub constraint_index: usize,
    pub constraint_type: String,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct SketchImport {
    pub name: String,
    pub feature_id: String,
    pub sketch: Sketch,
    /// Onshape entity id -> solver entity, including sub-entities such as
    /// `rect.bottom`.
    pub entities: HashMap<String, EntId>,
    /// External deterministic id -> the fixed entity created for it.
    pub externals: HashMap<String, EntId>,
    pub dimensions: Vec<Dimension>,
    pub skipped: Vec<Skipped>,
    /// Solver constraint index -> Onshape constraint index in the feature.
    pub constraint_source: Vec<usize>,
    /// Constraints whose side / branch was chosen from the seed, with all
    /// candidates; [`SketchImport::reselect_variants`] re-chooses from the
    /// current geometry.
    pub variants: Vec<(usize, Vec<Constraint>)>,
}

impl SketchImport {
    /// Re-evaluate every dimension expression against `scope` and write the
    /// values into the constraints (the sketch is not re-solved here).
    pub fn set_dimensions(&mut self, scope: &dyn Scope) -> Result<(), ImportError> {
        for d in &self.dimensions {
            let v = fs_expr::evaluate(&d.expression, scope)
                .map_err(|e| ImportError(format!("{}: dimension {:?}: {e}", self.name, d.expression)))?;
            let c = &mut self.sketch.constraints[d.constraint];
            let len = |v: &fs_expr::Value| v.as_length_m().map_err(|e| ImportError(format!("{}: {:?}: {e}", self.name, d.expression)));
            match (d.kind, c) {
                (DimKind::Length, Constraint::Length { value, .. }) => *value = len(&v)?,
                (DimKind::Distance, Constraint::PointPointDistance { value, .. })
                | (DimKind::Distance, Constraint::PointLineDistance { value, .. })
                | (DimKind::Distance, Constraint::LineLineDistance { value, .. })
                | (DimKind::Distance, Constraint::PointCircleDistance { value, .. }) => *value = len(&v)?,
                (DimKind::Radius, Constraint::Radius { value, .. }) => *value = len(&v)?,
                (DimKind::Diameter, Constraint::Radius { value, .. }) => *value = len(&v)? / 2.0,
                (DimKind::MajorDiameter, Constraint::EllipseRadius { value, .. })
                | (DimKind::MinorDiameter, Constraint::EllipseRadius { value, .. }) => *value = len(&v)? / 2.0,
                (DimKind::Angle, Constraint::Angle { value, .. }) => {
                    *value = v.as_angle_rad().map_err(|e| ImportError(format!("{}: {:?}: {e}", self.name, d.expression)))?
                }
                (k, c) => return Err(ImportError(format!("{}: dimension kind {k:?} does not match constraint {c:?}", self.name))),
            }
        }
        Ok(())
    }

    /// Re-choose every ambiguous constraint variant from the current variable
    /// values (call after replacing the seed). Dimension values are carried
    /// over from the current choice.
    pub fn reselect_variants(&mut self) {
        for (idx, cands) in &self.variants {
            let current = self.sketch.constraints[*idx];
            let mut best = (f64::INFINITY, current);
            for c in cands {
                let mut c = *c;
                copy_value(&current, &mut c);
                self.sketch.constraints[*idx] = c;
                let r = self.sketch.constraint_residuals(*idx).iter().fold(0.0f64, |m, v| m.max(v.abs()));
                if r < best.0 {
                    best = (r, c);
                }
            }
            self.sketch.constraints[*idx] = best.1;
        }
    }

    /// [`Sketch::merge_coincident_points`] with this import's constraint
    /// bookkeeping kept in step. Returns the number of constraints removed.
    pub fn merge_coincident_points(&mut self) -> usize {
        let remap = self.sketch.merge_coincident_points();
        self.apply_remap(&remap);
        remap.iter().filter(|r| r.is_none()).count()
    }

    fn apply_remap(&mut self, remap: &[Option<usize>]) {
        self.constraint_source = self.constraint_source.iter().enumerate().filter_map(|(i, s)| remap[i].map(|_| *s)).collect();
        self.dimensions = self.dimensions.iter().filter_map(|d| remap[d.constraint].map(|c| Dimension { constraint: c, ..d.clone() })).collect();
        self.variants = self.variants.iter().filter_map(|(i, v)| remap[*i].map(|c| (c, v.clone()))).collect();
    }

    /// Remove every constraint that refers to one of the given entity ids
    /// (for entities the seed could not be set for). Returns the number
    /// removed.
    pub fn drop_constraints_touching(&mut self, ids: &[String]) -> usize {
        let ents: Vec<EntId> = ids.iter().filter_map(|id| self.entities.get(id).copied()).collect();
        if ents.is_empty() {
            return 0;
        }
        let mut removed = 0;
        let mut keep = Vec::new();
        let mut remap = vec![None; self.sketch.constraints.len()];
        for (i, c) in self.sketch.constraints.iter().enumerate() {
            if self.sketch.constraint_vars_entities(c).iter().any(|e| ents.contains(e)) {
                removed += 1;
                continue;
            }
            remap[i] = Some(keep.len());
            keep.push(*c);
        }
        self.sketch.constraints = keep;
        self.apply_remap(&remap);
        removed
    }

    /// Current values of all dimensions and fixed external variables, in a
    /// form [`solve_towards`](Self::solve_towards) can interpolate.
    pub fn targets(&self) -> Vec<f64> {
        let mut out: Vec<f64> = self.dimensions.iter().map(|d| dimension_value(&self.sketch.constraints[d.constraint])).collect();
        out.extend((0..self.sketch.x.len()).filter(|&v| self.sketch.fixed[v]).map(|v| self.sketch.x[v]));
        out
    }

    fn set_targets(&mut self, values: &[f64]) {
        let nd = self.dimensions.len();
        for (d, v) in self.dimensions.iter().zip(values) {
            set_dimension_value(&mut self.sketch.constraints[d.constraint], *v);
        }
        let fixed: Vec<usize> = (0..self.sketch.x.len()).filter(|&v| self.sketch.fixed[v]).collect();
        for (v, val) in fixed.iter().zip(&values[nd..]) {
            self.sketch.x[*v] = *val;
        }
    }

    /// Solve from the current (satisfied) state to the dimensions evaluated at
    /// `scope` (externals as already set), by continuation in dimension space.
    pub fn solve_towards(&mut self, scope: &dyn Scope, opts: sketch_solve_core::SolveOptions) -> Result<sketch_solve_core::SolveReport, ImportError> {
        let from = self.targets();
        self.set_dimensions(scope)?;
        let to = self.targets();
        self.set_targets(&from);
        self.solve_between(&from, &to, opts)
    }

    /// Solve by continuation along a path of configurations: `scope_at(t)`
    /// gives the variable scope and `externals_at(t)` the fixed external
    /// geometry at path parameter `t` in [0, 1] (0 = the state the current
    /// geometry satisfies, 1 = the wanted one). Dimensions are re-evaluated
    /// at every step so intermediate states are real configurations of the
    /// model. Where the expressions jump (tables, floor, branches) a step
    /// cannot be made small, so a stuck step is retried as a continuation in
    /// dimension space across the jump alone.
    pub fn solve_along(
        &mut self,
        scope_at: &dyn Fn(f64) -> Box<dyn Scope>,
        externals_at: &dyn Fn(f64) -> Vec<(String, ExternalGeom)>,
        opts: sketch_solve_core::SolveOptions,
    ) -> Result<sketch_solve_core::SolveReport, ImportError> {
        let trace = std::env::var_os("SKETCH_TRACE").is_some();
        let mut t: f64 = 0.0;
        // Start with a small step: a large first move can land on a different
        // root of a sensitive sub-system (a flat arc joining a fillet), which
        // is a locked state the path can never leave. Steps double on success.
        let mut step: f64 = std::env::var("SKETCH_STEP0").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0 / 64.0);
        let mut steps = 0;
        let mut last: Option<sketch_solve_core::SolveReport> = None;
        // The targets (dimensions + externals) the current geometry satisfies.
        let mut good = self.targets();
        while t < 1.0 {
            let target = (t + step).min(1.0);
            let saved = self.sketch.x.clone();
            self.set_dimensions(scope_at(target).as_ref())?;
            for (id, g) in externals_at(target) {
                self.set_external(&id, g)?;
            }
            let wanted = self.targets();
            let r = sketch_solve_core::solve(&mut self.sketch, opts).map_err(|e| ImportError(format!("{}: {e}", self.name)))?;
            steps += 1;
            if trace {
                eprintln!("  path t={t:.4} -> {target:.4}: converged {} in {} iterations, residual {:.2e}", r.converged, r.iterations, r.max_residual);
            }
            if r.converged {
                t = target;
                last = Some(r);
                good = wanted;
                step = (step * 2.0).min(1.0 - t).max(1e-6);
                continue;
            }
            // Retry a small step as a continuation in dimension space from the
            // last good state (fine for smooth changes, and the only way across
            // a jump in the expressions); larger steps are just halved.
            self.sketch.x = saved.clone();
            self.set_targets(&good);
            if step > 0.0625 {
                step /= 2.0;
                continue;
            }
            let mut r = self.solve_between(&good, &wanted, opts)?;
            if trace {
                eprintln!("  retry t={t:.4} -> {target:.4} in dimension space: converged {} in {} steps, residual {:.2e}", r.converged, r.iterations, r.max_residual);
            }
            steps += r.iterations;
            if r.converged {
                t = target;
                last = Some(r);
                good = wanted;
                step = (step * 2.0).min(1.0 - t).max(1e-6);
                continue;
            }
            self.sketch.x = saved;
            self.set_targets(&good);
            if step > 1e-4 {
                step /= 2.0;
                continue;
            }
            r.iterations = steps;
            r.progress = t;
            return Ok(r);
        }
        let mut r = last.unwrap_or(sketch_solve_core::SolveReport { converged: true, iterations: 0, max_residual: 0.0, worst_constraint: None, free_vars: 0, equations: 0, progress: 1.0 });
        r.iterations = steps;
        r.progress = 1.0;
        Ok(r)
    }

    /// Continuation in dimension space between two target vectors (see
    /// [`targets`](Self::targets)).
    fn solve_between(&mut self, from: &[f64], to: &[f64], opts: sketch_solve_core::SolveOptions) -> Result<sketch_solve_core::SolveReport, ImportError> {
        let mut sketch = std::mem::take(&mut self.sketch);
        let dims = self.dimensions.clone();
        let fixed: Vec<usize> = (0..sketch.x.len()).filter(|&v| sketch.fixed[v]).collect();
        let nd = dims.len();
        let r = sketch_solve_core::solve_continuation(&mut sketch, opts, |sk, t| {
            for (i, d) in dims.iter().enumerate() {
                set_dimension_value(&mut sk.constraints[d.constraint], from[i] + t * (to[i] - from[i]));
            }
            for (j, v) in fixed.iter().enumerate() {
                sk.x[*v] = from[nd + j] + t * (to[nd + j] - from[nd + j]);
            }
        });
        self.sketch = sketch;
        r.map_err(|e| ImportError(format!("{}: {e}", self.name)))
    }

    /// Move a fixed external entity to new geometry (same kind as when it was
    /// created).
    pub fn set_external(&mut self, id: &str, geom: ExternalGeom) -> Result<(), ImportError> {
        let e = *self.externals.get(id).ok_or_else(|| ImportError(format!("{}: unknown external {id}", self.name)))?;
        match (self.sketch.entities[e], geom) {
            (Entity::Point { x, y }, ExternalGeom::Point(p)) => {
                self.sketch.x[x] = p[0];
                self.sketch.x[y] = p[1];
            }
            (Entity::Line { ax, ay, bx, by }, ExternalGeom::Line(a, b)) => {
                self.sketch.x[ax] = a[0];
                self.sketch.x[ay] = a[1];
                self.sketch.x[bx] = b[0];
                self.sketch.x[by] = b[1];
            }
            (e, g) => return Err(ImportError(format!("{}: external {id} kind changed: {e:?} vs {g:?}", self.name))),
        }
        Ok(())
    }
}

fn dimension_value(c: &Constraint) -> f64 {
    match c {
        Constraint::Length { value, .. }
        | Constraint::PointPointDistance { value, .. }
        | Constraint::PointLineDistance { value, .. }
        | Constraint::LineLineDistance { value, .. }
        | Constraint::PointCircleDistance { value, .. }
        | Constraint::Radius { value, .. }
        | Constraint::EllipseRadius { value, .. }
        | Constraint::Angle { value, .. } => *value,
        _ => 0.0,
    }
}

fn set_dimension_value(c: &mut Constraint, v: f64) {
    match c {
        Constraint::Length { value, .. }
        | Constraint::PointPointDistance { value, .. }
        | Constraint::PointLineDistance { value, .. }
        | Constraint::LineLineDistance { value, .. }
        | Constraint::PointCircleDistance { value, .. }
        | Constraint::Radius { value, .. }
        | Constraint::EllipseRadius { value, .. }
        | Constraint::Angle { value, .. } => *value = v,
        _ => {}
    }
}

/// Copy the dimension value of `from` into `to` when both carry one.
fn copy_value(from: &Constraint, to: &mut Constraint) {
    let v = match from {
        Constraint::Length { value, .. }
        | Constraint::PointPointDistance { value, .. }
        | Constraint::PointLineDistance { value, .. }
        | Constraint::LineLineDistance { value, .. }
        | Constraint::PointCircleDistance { value, .. }
        | Constraint::Radius { value, .. }
        | Constraint::EllipseRadius { value, .. }
        | Constraint::Angle { value, .. } => *value,
        _ => return,
    };
    match to {
        Constraint::Length { value, .. }
        | Constraint::PointPointDistance { value, .. }
        | Constraint::PointLineDistance { value, .. }
        | Constraint::LineLineDistance { value, .. }
        | Constraint::PointCircleDistance { value, .. }
        | Constraint::Radius { value, .. }
        | Constraint::EllipseRadius { value, .. }
        | Constraint::Angle { value, .. } => *value = v,
        _ => {}
    }
}

// ---------------------------------------------------------------- JSON helpers

fn msg(v: &Json) -> &Json {
    &v["message"]
}

fn f(v: &Json, key: &str) -> Result<f64, ImportError> {
    v[key].as_f64().ok_or_else(|| ImportError(format!("missing number {key}")))
}

/// Parameters of a feature / constraint as a map parameterId -> parameter JSON.
fn params(m: &Json) -> HashMap<String, &Json> {
    let mut out = HashMap::new();
    if let Some(ps) = m["parameters"].as_array() {
        for p in ps {
            if let Some(id) = msg(p)["parameterId"].as_str() {
                out.insert(id.to_string(), p);
            }
        }
    }
    out
}

fn param_str<'a>(ps: &HashMap<String, &'a Json>, id: &str) -> Option<&'a str> {
    ps.get(id).and_then(|p| msg(p)["value"].as_str())
}

fn param_expr<'a>(ps: &HashMap<String, &'a Json>, id: &str) -> Option<&'a str> {
    ps.get(id).and_then(|p| msg(p)["expression"].as_str())
}


fn param_query_ids(ps: &HashMap<String, &Json>, id: &str) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(p) = ps.get(id) {
        if let Some(qs) = msg(p)["queries"].as_array() {
            for q in qs {
                if let Some(ids) = msg(q)["geometryIds"].as_array() {
                    out.extend(ids.iter().filter_map(|x| x.as_str().map(str::to_string)));
                }
                if let Some(ids) = msg(q)["deterministicIds"].as_array() {
                    out.extend(ids.iter().filter_map(|x| x.as_str().map(str::to_string)));
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------- references

#[derive(Clone, Copy, Debug)]
enum Ref {
    Point(PointRef),
    Curve(EntId),
}

struct Builder<'a> {
    name: String,
    sketch: Sketch,
    entities: HashMap<String, EntId>,
    externals: HashMap<String, EntId>,
    resolver: &'a dyn ExternalResolver,
    /// Candidate lists of the constraint currently being built.
    pending_variants: Option<Vec<Constraint>>,
    /// Coincident point pairs (pre-pass over all COINCIDENT constraints).
    coincident: Vec<(PointRef, PointRef)>,
    /// Points lying on curves (pre-pass).
    on_curve: Vec<(PointRef, EntId)>,
}

impl Builder<'_> {
    fn kind(&self, e: EntId) -> &'static str {
        match self.sketch.entities[e] {
            Entity::Point { .. } => "point",
            Entity::Line { .. } => "line",
            Entity::Arc { .. } | Entity::BulgeArc { .. } => "arc",
            Entity::Circle { .. } => "circle",
            Entity::Ellipse { .. } => "ellipse",
        }
    }

    fn is_circle(&self, e: EntId) -> bool {
        matches!(self.sketch.entities[e], Entity::Arc { .. } | Entity::BulgeArc { .. } | Entity::Circle { .. })
    }

    fn is_arc(&self, e: EntId) -> bool {
        matches!(self.sketch.entities[e], Entity::Arc { .. } | Entity::BulgeArc { .. })
    }

    fn local_ref(&self, id: &str) -> Result<Ref, ImportError> {
        if let Some(&e) = self.entities.get(id) {
            return Ok(match self.sketch.entities[e] {
                Entity::Point { .. } => Ref::Point(PointRef::Point(e)),
                _ => Ref::Curve(e),
            });
        }
        if let Some((base, suffix)) = id.rsplit_once('.') {
            if let Some(&e) = self.entities.get(base) {
                let is_point = matches!(self.sketch.entities[e], Entity::Point { .. });
                return match (suffix, is_point) {
                    ("start", false) => Ok(Ref::Point(PointRef::Start(e))),
                    ("end", false) => Ok(Ref::Point(PointRef::End(e))),
                    ("center", false) => Ok(Ref::Point(PointRef::Center(e))),
                    _ => Err(ImportError(format!("unknown sub-entity {id}"))),
                };
            }
        }
        Err(ImportError(format!("unknown entity {id}")))
    }

    fn external_ref(&mut self, id: &str) -> Result<Option<Ref>, ImportError> {
        if let Some(&e) = self.externals.get(id) {
            return Ok(Some(self.ref_of(e)));
        }
        let Some(g) = self.resolver.resolve(id) else { return Ok(None) };
        let e = match g {
            ExternalGeom::Point(p) => self.sketch.add_point(p, true),
            ExternalGeom::Line(a, b) => self.sketch.add_line(a, b, true),
        };
        self.externals.insert(id.to_string(), e);
        Ok(Some(self.ref_of(e)))
    }

    fn ref_of(&self, e: EntId) -> Ref {
        match self.sketch.entities[e] {
            Entity::Point { .. } => Ref::Point(PointRef::Point(e)),
            _ => Ref::Curve(e),
        }
    }

    /// Resolve the `localX` / `externalX` pair of a constraint slot. `Ok(None)`
    /// means an external reference the resolver does not know.
    fn slot(&mut self, ps: &HashMap<String, &Json>, local: &str, external: &str) -> Result<Option<(Ref, String)>, ImportError> {
        if let Some(id) = param_str(ps, local) {
            return Ok(Some((self.local_ref(id)?, id.to_string())));
        }
        let ids = param_query_ids(ps, external);
        match ids.as_slice() {
            [] => Ok(None),
            [id] => Ok(self.external_ref(id)?.map(|r| (r, id.clone()))),
            many => Err(ImportError(format!("external slot {external} has {} ids", many.len()))),
        }
    }

    /// Residual magnitude of a candidate constraint at the seed.
    fn seed_residual(&mut self, c: Constraint) -> f64 {
        let i = self.sketch.add_constraint(c);
        let r = self.sketch.constraint_residuals(i).iter().fold(0.0f64, |m, v| m.max(v.abs()));
        self.sketch.constraints.pop();
        r
    }

    /// Of several candidate constraints, the one the seed satisfies best;
    /// the list is recorded so the choice can be revisited after re-seeding.
    fn best(&mut self, candidates: Vec<Constraint>) -> Constraint {
        let mut best = (f64::INFINITY, candidates[0]);
        for &c in &candidates {
            let r = self.seed_residual(c);
            if r < best.0 {
                best = (r, c);
            }
        }
        self.pending_variants = Some(candidates);
        best.1
    }

    /// All points coincident with `p` (transitively), including `p`.
    fn coincident_class(&self, p: PointRef) -> Vec<PointRef> {
        let mut class = vec![p];
        let mut changed = true;
        while changed {
            changed = false;
            for (a, b) in &self.coincident {
                if class.contains(a) && !class.contains(b) {
                    class.push(*b);
                    changed = true;
                } else if class.contains(b) && !class.contains(a) {
                    class.push(*a);
                    changed = true;
                }
            }
        }
        class
    }

    /// The point where two curves meet, if an endpoint of one is coincident
    /// with an endpoint of the other or lies on it (through any chain of
    /// coincidences): returns (point on a, point on b), where the point on b
    /// may be a's endpoint itself when b has no endpoint there.
    fn shared_endpoint(&self, a: EntId, b: EntId) -> Option<(PointRef, PointRef)> {
        let ends = |e: EntId| -> Vec<PointRef> {
            match self.sketch.entities[e] {
                Entity::Line { .. } | Entity::Arc { .. } | Entity::BulgeArc { .. } => vec![PointRef::Start(e), PointRef::End(e)],
                _ => vec![],
            }
        };
        for (x, y) in [(a, b), (b, a)] {
            for px in ends(x) {
                let class = self.coincident_class(px);
                if let Some(py) = ends(y).into_iter().find(|py| class.contains(py)) {
                    return Some(if x == a { (px, py) } else { (py, px) });
                }
                if self.on_curve.iter().any(|(p, c)| *c == y && class.contains(p)) {
                    return Some(if x == a { (px, px) } else { (px, px) });
                }
            }
        }
        None
    }

    fn point_of_ref(&self, r: Ref, what: &str) -> Result<PointRef, ImportError> {
        match r {
            Ref::Point(p) => Ok(p),
            Ref::Curve(e) => Err(ImportError(format!("{what}: expected a point, got a {}", self.kind(e)))),
        }
    }
}

// ---------------------------------------------------------------- entities

fn add_entity(b: &mut Builder, e: &Json) -> Result<(), ImportError> {
    let m = msg(e);
    let id = m["entityId"].as_str().ok_or_else(|| ImportError("entity without id".into()))?.to_string();
    let g = &m["geometry"];
    let gm = msg(g);
    let ent = match (e["typeName"].as_str().unwrap_or(""), g["typeName"].as_str().unwrap_or("")) {
        ("BTMSketchPoint", _) => b.sketch.add_point([f(m, "x")?, f(m, "y")?], false),
        ("BTMSketchCurveSegment", "BTCurveGeometryLine") => {
            let p = [f(gm, "pntX")?, f(gm, "pntY")?];
            let d = [f(gm, "dirX")?, f(gm, "dirY")?];
            let (t0, t1) = (f(m, "startParam")?, f(m, "endParam")?);
            b.sketch.add_line([p[0] + d[0] * t0, p[1] + d[1] * t0], [p[0] + d[0] * t1, p[1] + d[1] * t1], false)
        }
        ("BTMSketchCurveSegment", "BTCurveGeometryCircle") => {
            let c = [f(gm, "xCenter")?, f(gm, "yCenter")?];
            let r = f(gm, "radius")?;
            let rot = f(gm, "yDir")?.atan2(f(gm, "xDir")?);
            let cw = gm["clockwise"].as_bool().unwrap_or(false);
            let (t0, t1) = (f(m, "startParam")?, f(m, "endParam")?);
            // Parameters are angles from the circle's x direction; a clockwise
            // circle's parameter runs the other way.
            let (a0, a1) = if cw { (rot - t0, rot - t1) } else { (rot + t0, rot + t1) };
            add_arc_entity(&mut b.sketch, c, r, a0, a1, !cw)
        }
        ("BTMSketchCurve", "BTCurveGeometryCircle") => {
            b.sketch.add_circle([f(gm, "xCenter")?, f(gm, "yCenter")?], f(gm, "radius")?, false)
        }
        ("BTMSketchCurve", "BTCurveGeometryEllipse") => {
            let ang = f(gm, "yDir")?.atan2(f(gm, "xDir")?);
            b.sketch.add_ellipse([f(gm, "xCenter")?, f(gm, "yCenter")?], f(gm, "radius")?, f(gm, "minorRadius")?, ang, false)
        }
        (t, g) => return Err(ImportError(format!("entity {id}: unsupported {t} / {g}"))),
    };
    b.entities.insert(id, ent);
    Ok(())
}

/// Arcs are stored as endpoints plus bulge unless the sweep is nearly a
/// full turn (a "circle drawn as an arc"), which the bulge form cannot
/// represent; those keep the centre / angle form.
fn add_arc_entity(sk: &mut Sketch, c: [f64; 2], r: f64, a0: f64, a1: f64, ccw: bool) -> EntId {
    let tau = std::f64::consts::TAU;
    let mut sweep = (a1 - a0) % tau;
    if ccw && sweep <= 0.0 {
        sweep += tau;
    } else if !ccw && sweep >= 0.0 {
        sweep -= tau;
    }
    if sweep.abs() > 1.9 * std::f64::consts::PI {
        return sk.add_arc(c, r, a0, a1, ccw, false);
    }
    let a = [c[0] + r * a0.cos(), c[1] + r * a0.sin()];
    let b = [c[0] + r * a1.cos(), c[1] + r * a1.sin()];
    sk.add_bulge_arc(a, b, sweep, false)
}

// ---------------------------------------------------------------- constraints

struct Built {
    constraint: Constraint,
    dim: Option<(DimKind, String)>,
}

/// `Ok(None)`: skip (unresolved external). `Err`: unsupported.
fn build_constraint(b: &mut Builder, c: &Json, scope: &dyn Scope) -> Result<Option<Built>, ImportError> {
    let m = msg(c);
    let ty = m["constraintType"].as_str().unwrap_or("");
    let ps = params(m);
    let length = |b: &Builder, id: &str| -> Result<(f64, String), ImportError> {
        let e = param_expr(&ps, id).ok_or_else(|| ImportError(format!("{ty}: missing {id} expression")))?;
        let v = fs_expr::evaluate(e, scope).map_err(|err| ImportError(format!("{}: {ty} {:?}: {err}", b.name, e)))?;
        Ok((v.as_length_m().map_err(|err| ImportError(format!("{}: {ty} {:?}: {err}", b.name, e)))?, e.to_string()))
    };
    let first = b.slot(&ps, "localFirst", "externalFirst")?;
    let second = b.slot(&ps, "localSecond", "externalSecond")?;
    let has_second = param_str(&ps, "localSecond").is_some() || ps.contains_key("externalSecond");
    let unresolved = first.is_none() || (has_second && second.is_none());
    let one = |b: &Builder, s: &Option<(Ref, String)>| -> Result<Ref, ImportError> {
        s.as_ref().map(|x| x.0).ok_or_else(|| ImportError(format!("{}: {ty}: missing entity", b.name)))
    };
    let built = match ty {
        "COINCIDENT" | "HORIZONTAL" | "VERTICAL" | "PARALLEL" | "PERPENDICULAR" | "TANGENT" | "EQUAL" | "LENGTH" | "RADIUS"
        | "DIAMETER" | "MAJOR_DIAMETER" | "MINOR_DIAMETER" | "DISTANCE" | "ANGLE" => {
            if unresolved {
                return Ok(None);
            }
            let a = one(b, &first)?;
            match ty {
                "COINCIDENT" => {
                    let bb = one(b, &second)?;
                    let c = match (a, bb) {
                        (Ref::Point(p), Ref::Point(q)) => Constraint::Coincident(p, q),
                        (Ref::Point(p), Ref::Curve(e)) | (Ref::Curve(e), Ref::Point(p)) => match b.sketch.entities[e] {
                            Entity::Line { .. } => Constraint::PointOnLine(p, e),
                            Entity::Arc { .. } | Entity::BulgeArc { .. } | Entity::Circle { .. } => Constraint::PointOnCircle(p, e),
                            Entity::Ellipse { .. } => Constraint::PointOnEllipse(p, e),
                            Entity::Point { .. } => unreachable!(),
                        },
                        (Ref::Curve(e), Ref::Curve(g)) => match (b.sketch.entities[e], b.sketch.entities[g]) {
                            (Entity::Line { .. }, Entity::Line { .. }) => Constraint::Collinear(e, g),
                            _ => return Err(ImportError(format!("COINCIDENT {} / {}", b.kind(e), b.kind(g)))),
                        },
                    };
                    Built { constraint: c, dim: None }
                }
                "HORIZONTAL" | "VERTICAL" => {
                    let horizontal = ty == "HORIZONTAL";
                    let c = match (a, second.map(|s| s.0)) {
                        (Ref::Curve(e), None) if matches!(b.sketch.entities[e], Entity::Line { .. }) => {
                            if horizontal { Constraint::HorizontalLine(e) } else { Constraint::VerticalLine(e) }
                        }
                        (Ref::Point(p), Some(Ref::Point(q))) => {
                            if horizontal { Constraint::HorizontalPoints(p, q) } else { Constraint::VerticalPoints(p, q) }
                        }
                        (Ref::Curve(e), None) if matches!(b.sketch.entities[e], Entity::Ellipse { .. }) => {
                            Constraint::EllipseAxis { ellipse: e, horizontal }
                        }
                        (Ref::Curve(e), None) => return Err(ImportError(format!("{ty} on a {}", b.kind(e)))),
                        (x, y) => return Err(ImportError(format!("{ty} on {x:?} / {y:?}"))),
                    };
                    Built { constraint: c, dim: None }
                }
                "PARALLEL" | "PERPENDICULAR" => {
                    let bb = one(b, &second)?;
                    let (Ref::Curve(e), Ref::Curve(g)) = (a, bb) else { return Err(ImportError(format!("{ty} on non-curves"))) };
                    if !matches!(b.sketch.entities[e], Entity::Line { .. }) || !matches!(b.sketch.entities[g], Entity::Line { .. }) {
                        return Err(ImportError(format!("{ty} on {} / {}", b.kind(e), b.kind(g))));
                    }
                    Built { constraint: if ty == "PARALLEL" { Constraint::Parallel(e, g) } else { Constraint::Perpendicular(e, g) }, dim: None }
                }
                "TANGENT" => {
                    let bb = one(b, &second)?;
                    let (Ref::Curve(e), Ref::Curve(g)) = (a, bb) else { return Err(ImportError("TANGENT on a point".into())) };
                    let (line, circ) = match (b.sketch.entities[e], b.sketch.entities[g]) {
                        (Entity::Line { .. }, _) if b.is_circle(g) => (Some(e), g),
                        (_, Entity::Line { .. }) if b.is_circle(e) => (Some(g), e),
                        _ if b.is_circle(e) && b.is_circle(g) => (None, e),
                        _ => return Err(ImportError(format!("TANGENT {} / {}", b.kind(e), b.kind(g)))),
                    };
                    let shared = b.shared_endpoint(e, g);
                    let c = match (shared, line) {
                        (Some((pa, pb)), _) => Constraint::TangentAt { a: e, at_a: pa, b: g, at_b: pb },
                        (None, Some(line)) => b.best(vec![
                            Constraint::TangentLineCircle { line, circle: circ, sign: 1.0 },
                            Constraint::TangentLineCircle { line, circle: circ, sign: -1.0 },
                        ]),
                        (None, None) => b.best(vec![
                            Constraint::TangentCircles { a: e, b: g, external: true, sign: 1.0 },
                            Constraint::TangentCircles { a: e, b: g, external: false, sign: 1.0 },
                            Constraint::TangentCircles { a: e, b: g, external: false, sign: -1.0 },
                        ]),
                    };
                    Built { constraint: c, dim: None }
                }
                "EQUAL" => {
                    let bb = one(b, &second)?;
                    let (Ref::Curve(e), Ref::Curve(g)) = (a, bb) else { return Err(ImportError("EQUAL on a point".into())) };
                    let c = match (b.sketch.entities[e], b.sketch.entities[g]) {
                        (Entity::Line { .. }, Entity::Line { .. }) => Constraint::EqualLength(e, g),
                        _ if b.is_circle(e) && b.is_circle(g) => Constraint::EqualRadius(e, g),
                        _ => return Err(ImportError(format!("EQUAL {} / {}", b.kind(e), b.kind(g)))),
                    };
                    Built { constraint: c, dim: None }
                }
                "LENGTH" => {
                    let Ref::Curve(e) = a else { return Err(ImportError("LENGTH on a point".into())) };
                    if !matches!(b.sketch.entities[e], Entity::Line { .. }) {
                        return Err(ImportError(format!("LENGTH on {}", b.kind(e))));
                    }
                    let (v, expr) = length(b, "length")?;
                    Built { constraint: Constraint::Length { line: e, value: v }, dim: Some((DimKind::Length, expr)) }
                }
                "RADIUS" | "DIAMETER" => {
                    let Ref::Curve(e) = a else { return Err(ImportError(format!("{ty} on a point"))) };
                    if !b.is_circle(e) {
                        return Err(ImportError(format!("{ty} on {}", b.kind(e))));
                    }
                    let (v, expr) = length(b, "length")?;
                    let (value, kind) = if ty == "RADIUS" { (v, DimKind::Radius) } else { (v / 2.0, DimKind::Diameter) };
                    Built { constraint: Constraint::Radius { circle: e, value }, dim: Some((kind, expr)) }
                }
                "MAJOR_DIAMETER" | "MINOR_DIAMETER" => {
                    let Ref::Curve(e) = a else { return Err(ImportError(format!("{ty} on a point"))) };
                    if !matches!(b.sketch.entities[e], Entity::Ellipse { .. }) {
                        return Err(ImportError(format!("{ty} on {}", b.kind(e))));
                    }
                    let (v, expr) = length(b, "length")?;
                    let major = ty == "MAJOR_DIAMETER";
                    Built {
                        constraint: Constraint::EllipseRadius { ellipse: e, major, value: v / 2.0 },
                        dim: Some((if major { DimKind::MajorDiameter } else { DimKind::MinorDiameter }, expr)),
                    }
                }
                "DISTANCE" => {
                    let bb = one(b, &second)?;
                    let (v, expr) = length(b, "length")?;
                    let mode = match param_str(&ps, "direction").unwrap_or("MINIMUM") {
                        "MINIMUM" => DistanceMode::Minimum,
                        "HORIZONTAL" => DistanceMode::Horizontal,
                        "VERTICAL" => DistanceMode::Vertical,
                        other => return Err(ImportError(format!("DISTANCE direction {other}"))),
                    };
                    // Any curve reference measured along an axis is taken at its start
                    // point (a horizontal distance to a vertical line, and so on).
                    let axis_point = |b: &Builder, r: Ref| -> Result<PointRef, ImportError> {
                        match r {
                            Ref::Point(p) => Ok(p),
                            Ref::Curve(e) => match b.sketch.entities[e] {
                                Entity::Line { .. } => Ok(PointRef::Start(e)),
                                _ => Ok(PointRef::Center(e)),
                            },
                        }
                    };
                    let c = match mode {
                        DistanceMode::Horizontal | DistanceMode::Vertical => {
                            let (p, q) = (axis_point(b, a)?, axis_point(b, bb)?);
                            b.best(vec![
                                Constraint::PointPointDistance { a: p, b: q, mode, sign: 1.0, value: v },
                                Constraint::PointPointDistance { a: p, b: q, mode, sign: -1.0, value: v },
                            ])
                        }
                        DistanceMode::Minimum => match (a, bb) {
                            (Ref::Point(p), Ref::Point(q)) => Constraint::PointPointDistance { a: p, b: q, mode, sign: 1.0, value: v },
                            (Ref::Point(p), Ref::Curve(e)) | (Ref::Curve(e), Ref::Point(p)) => match b.sketch.entities[e] {
                                Entity::Line { .. } => b.best(vec![
                                    Constraint::PointLineDistance { point: p, line: e, sign: 1.0, value: v },
                                    Constraint::PointLineDistance { point: p, line: e, sign: -1.0, value: v },
                                ]),
                                Entity::Arc { .. } | Entity::BulgeArc { .. } | Entity::Circle { .. } => b.best(vec![
                                    Constraint::PointCircleDistance { point: p, circle: e, sign: 1.0, value: v },
                                    Constraint::PointCircleDistance { point: p, circle: e, sign: -1.0, value: v },
                                ]),
                                other => return Err(ImportError(format!("DISTANCE point / {other:?}"))),
                            },
                            (Ref::Curve(e), Ref::Curve(g)) => match (b.sketch.entities[e], b.sketch.entities[g]) {
                                (Entity::Line { .. }, Entity::Line { .. }) => b.best(vec![
                                    Constraint::LineLineDistance { a: e, b: g, sign: 1.0, value: v },
                                    Constraint::LineLineDistance { a: e, b: g, sign: -1.0, value: v },
                                ]),
                                _ => return Err(ImportError(format!("DISTANCE {} / {}", b.kind(e), b.kind(g)))),
                            },
                        },
                    };
                    Built { constraint: c, dim: Some((DimKind::Distance, expr)) }
                }
                "ANGLE" => {
                    let bb = one(b, &second)?;
                    let (Ref::Curve(e), Ref::Curve(g)) = (a, bb) else { return Err(ImportError("ANGLE on a point".into())) };
                    let expr = param_expr(&ps, "angle").ok_or_else(|| ImportError("ANGLE without angle".into()))?.to_string();
                    let v = fs_expr::evaluate(&expr, scope)
                        .and_then(|v| v.as_angle_rad())
                        .map_err(|err| ImportError(format!("ANGLE {expr:?}: {err}")))?;
                    let pi = std::f64::consts::PI;
                    let mut cands = Vec::new();
                    for sign in [1.0, -1.0] {
                        for offset in [0.0, pi, -pi] {
                            cands.push(Constraint::Angle { a: e, b: g, value: v, sign, offset });
                        }
                    }
                    Built { constraint: b.best(cands), dim: Some((DimKind::Angle, expr)) }
                }
                _ => unreachable!(),
            }
        }
        "MIDPOINT" => {
            let mid = b.slot(&ps, "localMidpoint", "externalMidpoint")?;
            let e1 = b.slot(&ps, "localEntity1", "externalEntity1")?;
            let e2 = b.slot(&ps, "localEntity2", "externalEntity2")?;
            let want = |ps: &HashMap<String, &Json>, l: &str, x: &str| param_str(ps, l).is_some() || ps.contains_key(x);
            if (want(&ps, "localMidpoint", "externalMidpoint") && mid.is_none())
                || (want(&ps, "localEntity1", "externalEntity1") && e1.is_none())
                || (want(&ps, "localEntity2", "externalEntity2") && e2.is_none())
            {
                return Ok(None);
            }
            let c = match (mid, e1, e2) {
                (Some((m, _)), Some((p, _)), Some((q, _))) => Constraint::Midpoint {
                    mid: b.point_of_ref(m, "MIDPOINT midpoint")?,
                    a: b.point_of_ref(p, "MIDPOINT entity1")?,
                    b: b.point_of_ref(q, "MIDPOINT entity2")?,
                },
                (None, Some((p, _)), Some((q, _))) => match (p, q) {
                    (Ref::Point(m), Ref::Curve(l)) | (Ref::Curve(l), Ref::Point(m)) if matches!(b.sketch.entities[l], Entity::Line { .. }) => {
                        Constraint::Midpoint { mid: m, a: PointRef::Start(l), b: PointRef::End(l) }
                    }
                    (Ref::Point(m), Ref::Curve(l)) | (Ref::Curve(l), Ref::Point(m)) if b.is_arc(l) => b.best(vec![
                        Constraint::ArcMidpoint { point: m, arc: l, far: false },
                        Constraint::ArcMidpoint { point: m, arc: l, far: true },
                    ]),
                    _ => return Err(ImportError(format!("MIDPOINT {p:?} / {q:?}"))),
                },
                other => return Err(ImportError(format!("MIDPOINT slots {other:?}"))),
            };
            Built { constraint: c, dim: None }
        }
        other => return Err(ImportError(format!("unsupported constraint type {other}"))),
    };
    Ok(Some(built))
}

/// Import one `newSketch` feature.
pub fn import_sketch(feature: &Json, scope: &dyn Scope, resolver: &dyn ExternalResolver) -> Result<SketchImport, ImportError> {
    let m = msg(feature);
    if m["featureType"] != "newSketch" {
        return Err(ImportError(format!("not a sketch: {}", m["featureType"])));
    }
    let name = m["name"].as_str().unwrap_or("").to_string();
    let feature_id = m["featureId"].as_str().unwrap_or("").to_string();
    let mut b = Builder {
        name: name.clone(),
        sketch: Sketch::new(),
        entities: HashMap::new(),
        externals: HashMap::new(),
        resolver,
        pending_variants: None,
        coincident: Vec::new(),
        on_curve: Vec::new(),
    };
    for e in m["entities"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        add_entity(&mut b, e).map_err(|err| ImportError(format!("{name}: {err}")))?;
    }
    // Pre-pass: coincidences and point-on-curve facts, so tangencies built
    // earlier in the list can still find the meeting point.
    for c in m["constraints"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        if msg(c)["constraintType"] != "COINCIDENT" {
            continue;
        }
        let ps = params(msg(c));
        let (Ok(Some((a, _))), Ok(Some((bb, _)))) = (b.slot(&ps, "localFirst", "externalFirst"), b.slot(&ps, "localSecond", "externalSecond")) else { continue };
        match (a, bb) {
            (Ref::Point(p), Ref::Point(q)) => b.coincident.push((p, q)),
            (Ref::Point(p), Ref::Curve(e)) | (Ref::Curve(e), Ref::Point(p)) => b.on_curve.push((p, e)),
            _ => {}
        }
    }
    let mut dimensions = Vec::new();
    let mut skipped = Vec::new();
    let mut constraint_source = Vec::new();
    let mut variants = Vec::new();
    for (i, c) in m["constraints"].as_array().map(|a| a.as_slice()).unwrap_or(&[]).iter().enumerate() {
        let ty = msg(c)["constraintType"].as_str().unwrap_or("?").to_string();
        b.pending_variants = None;
        match build_constraint(&mut b, c, scope) {
            Ok(Some(built)) => {
                let idx = b.sketch.add_constraint(built.constraint);
                constraint_source.push(i);
                if let Some(v) = b.pending_variants.take() {
                    variants.push((idx, v));
                }
                if let Some((kind, expression)) = built.dim {
                    dimensions.push(Dimension { constraint: idx, kind, expression });
                }
            }
            Ok(None) => skipped.push(Skipped { constraint_index: i, constraint_type: ty, reason: "unresolved external reference".into() }),
            Err(err) => skipped.push(Skipped { constraint_index: i, constraint_type: ty, reason: err.0 }),
        }
    }
    Ok(SketchImport { name, feature_id, sketch: b.sketch, entities: b.entities, externals: b.externals, dimensions, skipped, constraint_source, variants })
}

/// External reference ids used by a sketch feature's constraints, in order of
/// first appearance.
pub fn external_ids(feature: &Json) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for c in msg(feature)["constraints"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        let ps = params(msg(c));
        for slot in ["externalFirst", "externalSecond", "externalMidpoint", "externalEntity1", "externalEntity2"] {
            for id in param_query_ids(&ps, slot) {
                if !out.contains(&id) {
                    out.push(id);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn quantity(id: &str, expr: &str) -> Json {
        json!({"typeName": "BTMParameterQuantity", "message": {"parameterId": id, "expression": expr}})
    }
    fn string(id: &str, v: &str) -> Json {
        json!({"typeName": "BTMParameterString", "message": {"parameterId": id, "value": v}})
    }
    fn enumv(id: &str, v: &str) -> Json {
        json!({"typeName": "BTMParameterEnum", "message": {"parameterId": id, "value": v}})
    }
    fn external(id: &str, det: &str) -> Json {
        json!({"typeName": "BTMParameterQueryList", "message": {"parameterId": id, "queries": [{"typeName": "BTMIndividualQuery", "message": {"geometryIds": [det]}}]}})
    }
    fn line(id: &str, a: [f64; 2], b: [f64; 2]) -> Json {
        let d = [b[0] - a[0], b[1] - a[1]];
        let n = (d[0] * d[0] + d[1] * d[1]).sqrt();
        json!({"typeName": "BTMSketchCurveSegment", "message": {"entityId": id, "startParam": 0.0, "endParam": n,
            "geometry": {"typeName": "BTCurveGeometryLine", "message": {"pntX": a[0], "pntY": a[1], "dirX": d[0] / n, "dirY": d[1] / n}}}})
    }
    fn constraint(ty: &str, params: Vec<Json>) -> Json {
        json!({"typeName": "BTMSketchConstraint", "message": {"constraintType": ty, "parameters": params}})
    }

    /// A rectangle seeded slightly off its dimensions, with the corner pinned
    /// to an external origin; importing and solving sizes it exactly.
    #[test]
    fn rectangle_round_trip() {
        let feature = json!({"typeName": "BTMFeature", "message": {"featureType": "newSketch", "name": "Rect", "featureId": "F1",
            "entities": [
                line("b", [0.01, 0.0], [0.1, 0.0]), line("r", [0.1, 0.0], [0.1, 0.05]),
                line("t", [0.1, 0.05], [0.01, 0.05]), line("l", [0.01, 0.05], [0.01, 0.0]),
            ],
            "constraints": [
                constraint("COINCIDENT", vec![string("localFirst", "b.end"), string("localSecond", "r.start")]),
                constraint("COINCIDENT", vec![string("localFirst", "r.end"), string("localSecond", "t.start")]),
                constraint("COINCIDENT", vec![string("localFirst", "t.end"), string("localSecond", "l.start")]),
                constraint("COINCIDENT", vec![string("localFirst", "l.end"), string("localSecond", "b.start")]),
                constraint("COINCIDENT", vec![string("localFirst", "b.start"), external("externalSecond", "ORIGIN")]),
                constraint("HORIZONTAL", vec![string("localFirst", "b")]),
                constraint("PARALLEL", vec![string("localFirst", "b"), string("localSecond", "t")]),
                constraint("PERPENDICULAR", vec![string("localFirst", "b"), string("localSecond", "r")]),
                constraint("VERTICAL", vec![string("localFirst", "l")]),
                constraint("LENGTH", vec![string("localFirst", "b"), quantity("length", "(#W) in"), enumv("direction", "MINIMUM")]),
                constraint("DISTANCE", vec![string("localFirst", "t"), string("localSecond", "b"), quantity("length", "2 in"), enumv("direction", "MINIMUM")]),
                constraint("COINCIDENT", vec![string("localFirst", "b.start"), external("externalSecond", "UNKNOWN")]),
            ]}});
        let mut scope: HashMap<String, fs_expr::Value> = HashMap::new();
        scope.insert("W".into(), fs_expr::Value::num(4.0));
        let mut ext: HashMap<String, ExternalGeom> = HashMap::new();
        ext.insert("ORIGIN".into(), ExternalGeom::Point([0.0, 0.0]));
        let mut imp = import_sketch(&feature, &scope, &ext).unwrap();
        assert_eq!(imp.skipped.len(), 1, "{:?}", imp.skipped);
        assert_eq!(imp.dimensions.len(), 2);
        assert_eq!(external_ids(&feature), vec!["ORIGIN".to_string(), "UNKNOWN".to_string()]);
        let r = sketch_solve_core::solve(&mut imp.sketch, Default::default()).unwrap();
        assert!(r.converged, "{r:?}");
        let (a, b) = imp.sketch.line_points(imp.entities["t"]);
        assert!((a[0] - 4.0 * 0.0254).abs() < 1e-9 && (a[1] - 0.0508).abs() < 1e-9, "{a:?}");
        assert!(b[0].abs() < 1e-9 && (b[1] - 0.0508).abs() < 1e-9, "{b:?}");
        // Re-dimension and re-solve.
        scope.insert("W".into(), fs_expr::Value::num(6.0));
        imp.set_dimensions(&scope).unwrap();
        let r = sketch_solve_core::solve(&mut imp.sketch, Default::default()).unwrap();
        assert!(r.converged);
        let (a, _) = imp.sketch.line_points(imp.entities["t"]);
        assert!((a[0] - 6.0 * 0.0254).abs() < 1e-9, "{a:?}");
    }

    #[test]
    fn arc_parameters_follow_direction() {
        // Clockwise arc from the top of a circle to its right: start (0, r), end (r, 0).
        let feature = json!({"typeName": "BTMFeature", "message": {"featureType": "newSketch", "name": "Arc", "featureId": "F2",
            "entities": [
                {"typeName": "BTMSketchCurveSegment", "message": {"entityId": "a", "startParam": -1.5707963267948966, "endParam": 0.0,
                    "geometry": {"typeName": "BTCurveGeometryCircle", "message": {"xCenter": 0.0, "yCenter": 0.0, "radius": 1.0, "xDir": 1.0, "yDir": 0.0, "clockwise": true}}}},
            ], "constraints": []}});
        let scope: HashMap<String, fs_expr::Value> = HashMap::new();
        let ext: HashMap<String, ExternalGeom> = HashMap::new();
        let imp = import_sketch(&feature, &scope, &ext).unwrap();
        let s = imp.sketch.point(PointRef::Start(imp.entities["a"]));
        let e = imp.sketch.point(PointRef::End(imp.entities["a"]));
        assert!(s[0].abs() < 1e-12 && (s[1] - 1.0).abs() < 1e-12, "{s:?}");
        assert!((e[0] - 1.0).abs() < 1e-12 && e[1].abs() < 1e-12, "{e:?}");
    }
}
