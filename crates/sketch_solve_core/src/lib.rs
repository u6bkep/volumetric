//! A 2D sketch constraint solver in the style of a CAD sketcher.
//!
//! A [`Sketch`] is a flat vector of variables (`x`) plus entities that name
//! those variables geometrically (a point is two variables, an arc is
//! centre, radius and two angles, ...) and constraints between entities.
//! [`solve`] moves the free variables so every constraint residual is ~0,
//! starting from the current values as the seed and staying close to it:
//! under-constrained sketches are the norm in real CAD models and the
//! damped least-squares step keeps unconstrained degrees of freedom where
//! the seed put them.
//!
//! Every residual has length units (angular conditions are scaled by a
//! length), so a single tolerance in model units applies throughout.
//!
//! Conventions: angles in radians, counter-clockwise positive. An
//! [`Entity::Arc`] runs from angle `a0` to `a1`; `ccw` says whether that run
//! is counter-clockwise (the endpoints are `c + r (cos a, sin a)` either
//! way, only the tangent direction at each end depends on it).

pub mod dual;

use dual::{Dual, Real};
use std::fmt;

pub type VarId = usize;
pub type EntId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entity {
    Point { x: VarId, y: VarId },
    Line { ax: VarId, ay: VarId, bx: VarId, by: VarId },
    Arc { cx: VarId, cy: VarId, r: VarId, a0: VarId, a1: VarId, ccw: bool },
    /// An arc by its endpoints and signed sagitta (the bow of the arc over
    /// its chord, positive counter-clockwise). Centre and radius are
    /// derived, so the arc passes smoothly through straight (sagitta 0) as
    /// dimensions move, and the sagitta is a length, so the solver's
    /// minimum-norm behaviour moves the bow by small distances rather than
    /// swinging a flat arc's radius around. Use it for every arc whose sweep
    /// is not near a full turn.
    BulgeArc { ax: VarId, ay: VarId, bx: VarId, by: VarId, sag: VarId },
    Circle { cx: VarId, cy: VarId, r: VarId },
    /// `ra` along the direction `ang`, `rb` across it.
    Ellipse { cx: VarId, cy: VarId, ra: VarId, rb: VarId, ang: VarId },
}

/// A point of an entity: a point entity itself, or the start / end / centre
/// of a curve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointRef {
    Point(EntId),
    Start(EntId),
    End(EntId),
    Center(EntId),
}

impl PointRef {
    pub fn entity(self) -> EntId {
        match self {
            PointRef::Point(e) | PointRef::Start(e) | PointRef::End(e) | PointRef::Center(e) => e,
        }
    }
}

/// How a distance between two things is measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistanceMode {
    /// Euclidean / perpendicular distance.
    Minimum,
    /// Difference in x only.
    Horizontal,
    /// Difference in y only.
    Vertical,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Constraint {
    Coincident(PointRef, PointRef),
    /// Point on the infinite line through a line entity.
    PointOnLine(PointRef, EntId),
    /// Point on an arc's or circle's circle.
    PointOnCircle(PointRef, EntId),
    PointOnEllipse(PointRef, EntId),
    /// Both endpoints of the second line on the first (collinear).
    Collinear(EntId, EntId),
    HorizontalLine(EntId),
    VerticalLine(EntId),
    HorizontalPoints(PointRef, PointRef),
    VerticalPoints(PointRef, PointRef),
    Parallel(EntId, EntId),
    Perpendicular(EntId, EntId),
    /// Line length.
    Length { line: EntId, value: f64 },
    /// Signed distance between two points; `sign` picks which way round (`a - b`
    /// along the mode axis, or +1 for Euclidean).
    PointPointDistance { a: PointRef, b: PointRef, mode: DistanceMode, sign: f64, value: f64 },
    /// Signed perpendicular distance from a point to a line's infinite line.
    PointLineDistance { point: PointRef, line: EntId, sign: f64, value: f64 },
    /// Distance between two lines: both endpoints of `b` at the signed
    /// distance from `a`'s infinite line, which makes them parallel (as a
    /// CAD line-line distance does).
    LineLineDistance { a: EntId, b: EntId, sign: f64, value: f64 },
    /// Distance from a point to a circle (`|p - c| - r`, signed by `sign`).
    PointCircleDistance { point: PointRef, circle: EntId, sign: f64, value: f64 },
    /// `mid` is the midpoint of `a` and `b`.
    Midpoint { mid: PointRef, a: PointRef, b: PointRef },
    /// Line tangent to an arc / circle: perpendicular distance of the centre to
    /// the line equals the radius (`sign` selects the side).
    TangentLineCircle { line: EntId, circle: EntId, sign: f64 },
    /// Two circles tangent: `|c1 - c2| = r1 + r2` (external) or
    /// `|c1 - c2| = sign (r1 - r2)` (internal).
    TangentCircles { a: EntId, b: EntId, external: bool, sign: f64 },
    EqualLength(EntId, EntId),
    EqualRadius(EntId, EntId),
    Radius { circle: EntId, value: f64 },
    /// Ellipse semi-axis: `ra` (major) or `rb` (minor).
    EllipseRadius { ellipse: EntId, major: bool, value: f64 },
    /// Angle from line `a`'s direction to line `b`'s direction equals
    /// `sign * value + offset` (offset in {0, ±π}); chosen at build time to
    /// match the seed.
    Angle { a: EntId, b: EntId, value: f64, sign: f64, offset: f64 },
    /// Two curves are tangent where they meet: the tangent direction of `a` at
    /// `at_a` is parallel to that of `b` at `at_b` (the points are expected to
    /// be coincident by another constraint). Branch-free, unlike the distance
    /// forms, so an arc can pass through straight as dimensions move.
    TangentAt { a: EntId, at_a: PointRef, b: EntId, at_b: PointRef },
    /// An ellipse's major axis is horizontal (or vertical).
    EllipseAxis { ellipse: EntId, horizontal: bool },
    /// `point` is the midpoint of `arc` in the CAD sense: on the arc's
    /// circle and on the perpendicular bisector of its chord. That has two
    /// solutions and Onshape accepts either; `far` says the point is on the
    /// opposite side of the chord from the arc (chosen from the seed).
    ArcMidpoint { point: PointRef, arc: EntId, far: bool },
    /// Pin a variable to a value (used for fixed external geometry that must
    /// still be part of the system).
    FixVar { var: VarId, value: f64 },
}

/// Length used to turn angular residuals (unitless) into length units.
const ANGLE_SCALE: f64 = 0.1;

#[derive(Clone, Debug, Default)]
pub struct Sketch {
    pub x: Vec<f64>,
    pub fixed: Vec<bool>,
    pub entities: Vec<Entity>,
    pub constraints: Vec<Constraint>,
    /// Optional anchor: a weak pull of every free variable toward these
    /// values with the given weight (rows `w (x - x0)` added to the least
    /// squares, not counted in convergence). It decides under-constrained
    /// degrees of freedom the way a CAD solver re-solving from its stored
    /// sketch does: the solution closest to the seed, independent of the
    /// path taken to reach the current dimensions.
    pub anchor: Option<(Vec<f64>, f64)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SolveError(pub String);

impl fmt::Display for SolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for SolveError {}

#[derive(Clone, Debug)]
pub struct SolveReport {
    pub converged: bool,
    pub iterations: usize,
    /// Largest absolute residual after solving.
    pub max_residual: f64,
    /// Index of the constraint holding the largest residual.
    pub worst_constraint: Option<usize>,
    /// Number of free variables and of residual equations.
    pub free_vars: usize,
    pub equations: usize,
    /// For continuation: the path parameter reached (1 when complete).
    pub progress: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct SolveOptions {
    pub max_iterations: usize,
    /// Convergence tolerance on the largest residual (model units).
    pub tolerance: f64,
    pub initial_damping: f64,
    /// Optional trust region: a step moving any variable by more than this
    /// is rejected and the damping increased. Off by default: the variables
    /// mix lengths and radians and under-constrained directions legitimately
    /// take large steps, so a cap mostly just slows convergence.
    pub max_step: f64,
}

impl Default for SolveOptions {
    fn default() -> Self {
        SolveOptions { max_iterations: 100, tolerance: 1e-7, initial_damping: 1e-3, max_step: f64::INFINITY }
    }
}

impl Sketch {
    pub fn new() -> Sketch {
        Sketch::default()
    }

    fn var(&mut self, v: f64, fixed: bool) -> VarId {
        self.x.push(v);
        self.fixed.push(fixed);
        self.x.len() - 1
    }

    pub fn add_point(&mut self, p: [f64; 2], fixed: bool) -> EntId {
        let x = self.var(p[0], fixed);
        let y = self.var(p[1], fixed);
        self.entities.push(Entity::Point { x, y });
        self.entities.len() - 1
    }

    pub fn add_line(&mut self, a: [f64; 2], b: [f64; 2], fixed: bool) -> EntId {
        let ax = self.var(a[0], fixed);
        let ay = self.var(a[1], fixed);
        let bx = self.var(b[0], fixed);
        let by = self.var(b[1], fixed);
        self.entities.push(Entity::Line { ax, ay, bx, by });
        self.entities.len() - 1
    }

    /// `a1` is normalised so that `a1 - a0` is in (0, 2π] for a
    /// counter-clockwise arc and in [-2π, 0) for a clockwise one.
    pub fn add_arc(&mut self, c: [f64; 2], r: f64, a0: f64, a1: f64, ccw: bool, fixed: bool) -> EntId {
        let tau = std::f64::consts::TAU;
        let mut sweep = (a1 - a0) % tau;
        if ccw && sweep <= 0.0 {
            sweep += tau;
        } else if !ccw && sweep >= 0.0 {
            sweep -= tau;
        }
        let a1 = a0 + sweep;
        let cx = self.var(c[0], fixed);
        let cy = self.var(c[1], fixed);
        let r = self.var(r, fixed);
        let a0 = self.var(a0, fixed);
        let a1 = self.var(a1, fixed);
        self.entities.push(Entity::Arc { cx, cy, r, a0, a1, ccw });
        self.entities.len() - 1
    }

    /// Arc from `a` to `b` with the given signed sweep (radians, positive
    /// counter-clockwise), stored as endpoints plus sagitta.
    pub fn add_bulge_arc(&mut self, a: [f64; 2], b: [f64; 2], sweep: f64, fixed: bool) -> EntId {
        let ax = self.var(a[0], fixed);
        let ay = self.var(a[1], fixed);
        let bx = self.var(b[0], fixed);
        let by = self.var(b[1], fixed);
        let chord = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        let sag = self.var(sagitta_from_sweep(chord, sweep), fixed);
        self.entities.push(Entity::BulgeArc { ax, ay, bx, by, sag });
        self.entities.len() - 1
    }

    /// Set a bulge arc's sagitta from a signed sweep at its current chord.
    pub fn set_arc_sweep(&mut self, e: EntId, sweep: f64) {
        let Entity::BulgeArc { ax, ay, bx, by, sag } = self.entities[e] else { panic!("not a bulge arc") };
        let chord = ((self.x[bx] - self.x[ax]).powi(2) + (self.x[by] - self.x[ay]).powi(2)).sqrt();
        self.x[sag] = sagitta_from_sweep(chord, sweep);
    }

    /// Signed sweep of a bulge arc at its current values.
    pub fn arc_sweep(&self, e: EntId) -> f64 {
        let Entity::BulgeArc { ax, ay, bx, by, sag } = self.entities[e] else { panic!("not a bulge arc") };
        let chord = ((self.x[bx] - self.x[ax]).powi(2) + (self.x[by] - self.x[ay]).powi(2)).sqrt();
        4.0 * (2.0 * self.x[sag] / chord.max(1e-12)).atan()
    }

    /// Centre and radius of an arc or circle entity at the current values.
    pub fn circle(&self, e: EntId) -> ([f64; 2], f64) {
        circle_of::<f64>(self, e, &self.x)
    }

    pub fn add_circle(&mut self, c: [f64; 2], r: f64, fixed: bool) -> EntId {
        let cx = self.var(c[0], fixed);
        let cy = self.var(c[1], fixed);
        let r = self.var(r, fixed);
        self.entities.push(Entity::Circle { cx, cy, r });
        self.entities.len() - 1
    }

    pub fn add_ellipse(&mut self, c: [f64; 2], ra: f64, rb: f64, ang: f64, fixed: bool) -> EntId {
        let cx = self.var(c[0], fixed);
        let cy = self.var(c[1], fixed);
        let ra = self.var(ra, fixed);
        let rb = self.var(rb, fixed);
        let ang = self.var(ang, fixed);
        self.entities.push(Entity::Ellipse { cx, cy, ra, rb, ang });
        self.entities.len() - 1
    }

    pub fn add_constraint(&mut self, c: Constraint) -> usize {
        self.constraints.push(c);
        self.constraints.len() - 1
    }

    /// Merge the variables of points joined by `Coincident` constraints when
    /// both are plain variables (point entities, line endpoints): the
    /// constraint is removed and the entities share one pair of variables.
    /// Pairs involving a fixed variable or a derived point (arc endpoints)
    /// are left as constraints. Returns the old -> new constraint index map
    /// (`None` for removed constraints). Orphaned variables are marked fixed
    /// so they leave the system.
    pub fn merge_coincident_points(&mut self) -> Vec<Option<usize>> {
        let n = self.x.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        let plain = |sk: &Sketch, p: PointRef| -> Option<(VarId, VarId)> {
            match (p, sk.entities[p.entity()]) {
                (PointRef::Point(_), Entity::Point { x, y }) => Some((x, y)),
                (PointRef::Start(_), Entity::Line { ax, ay, .. }) | (PointRef::Start(_), Entity::BulgeArc { ax, ay, .. }) => Some((ax, ay)),
                (PointRef::End(_), Entity::Line { bx, by, .. }) | (PointRef::End(_), Entity::BulgeArc { bx, by, .. }) => Some((bx, by)),
                (PointRef::Center(_), Entity::Arc { cx, cy, .. })
                | (PointRef::Center(_), Entity::Circle { cx, cy, .. })
                | (PointRef::Center(_), Entity::Ellipse { cx, cy, .. }) => Some((cx, cy)),
                _ => None,
            }
        };
        let mut removed = 0;
        let mut keep = Vec::with_capacity(self.constraints.len());
        let mut remap = Vec::with_capacity(self.constraints.len());
        for c in std::mem::take(&mut self.constraints) {
            if let Constraint::Coincident(p, q) = c {
                if let (Some((px, py)), Some((qx, qy))) = (plain(self, p), plain(self, q)) {
                    if !self.fixed[px] && !self.fixed[py] && !self.fixed[qx] && !self.fixed[qy] {
                        let (rx, sx) = (find(&mut parent, px), find(&mut parent, qx));
                        let (ry, sy) = (find(&mut parent, py), find(&mut parent, qy));
                        parent[sx] = rx;
                        parent[sy] = ry;
                        removed += 1;
                        remap.push(None);
                        continue;
                    }
                }
            }
            remap.push(Some(keep.len()));
            keep.push(c);
        }
        self.constraints = keep;
        if removed == 0 {
            return remap;
        }
        let rep: Vec<usize> = (0..n).map(|i| find(&mut parent, i)).collect();
        // Representative takes the mean of the merged seeds (they coincide anyway).
        let mut used = vec![false; n];
        for e in &mut self.entities {
            let m = |v: &mut VarId| *v = rep[*v];
            match e {
                Entity::Point { x, y } => {
                    m(x);
                    m(y);
                }
                Entity::Line { ax, ay, bx, by } => {
                    m(ax);
                    m(ay);
                    m(bx);
                    m(by);
                }
                Entity::Arc { cx, cy, r, a0, a1, .. } => {
                    m(cx);
                    m(cy);
                    m(r);
                    m(a0);
                    m(a1);
                }
                Entity::BulgeArc { ax, ay, bx, by, sag } => {
                    m(ax);
                    m(ay);
                    m(bx);
                    m(by);
                    m(sag);
                }
                Entity::Circle { cx, cy, r } => {
                    m(cx);
                    m(cy);
                    m(r);
                }
                Entity::Ellipse { cx, cy, ra, rb, ang } => {
                    m(cx);
                    m(cy);
                    m(ra);
                    m(rb);
                    m(ang);
                }
            }
        }
        for e in 0..self.entities.len() {
            for v in self.entity_vars(e) {
                used[v] = true;
            }
        }
        for c in &mut self.constraints {
            if let Constraint::FixVar { var, .. } = c {
                *var = rep[*var];
                used[*var] = true;
            }
        }
        for v in 0..n {
            if !used[v] {
                self.fixed[v] = true;
            }
        }
        remap
    }

    /// Fix the sweep direction of bulge arcs from their tangent junctions:
    /// where a curve ends and another starts (`TangentAt` with End/Start or
    /// Start/End points) the tangents must point the same way, where both
    /// start or both end they must oppose. An arc whose tangent has the
    /// wrong sense is replaced by its complement (sweep -/+ 2π: the same
    /// circle the other way round). Lines have a definite sense, so their
    /// junctions are decided first and the result propagates through
    /// arc-arc junctions; arcs joined only to other unconfirmed arcs (or to
    /// circles) keep their seeded direction. Returns the number flipped.
    pub fn fix_arc_sweeps(&mut self) -> usize {
        let mut confirmed: Vec<bool> = self.entities.iter().map(|e| !matches!(e, Entity::BulgeArc { .. })).collect();
        let mut flipped = 0;
        loop {
            let mut changed = false;
            for i in 0..self.constraints.len() {
                let Constraint::TangentAt { a, at_a, b, at_b } = self.constraints[i] else { continue };
                if at_a.entity() != a || at_b.entity() != b {
                    continue; // a foreign point (circle junction): no sense
                }
                let (known, unknown, at_k, at_u) = match (confirmed[a], confirmed[b]) {
                    (true, false) => (a, b, at_a, at_b),
                    (false, true) => (b, a, at_b, at_a),
                    _ => continue,
                };
                if !matches!(self.entities[unknown], Entity::BulgeArc { .. }) {
                    continue;
                }
                let is_start = |p: PointRef| matches!(p, PointRef::Start(_));
                let same_sense = is_start(at_k) != is_start(at_u);
                let tk = tangent_at::<f64>(self, known, at_k, &self.x);
                let tu = tangent_at::<f64>(self, unknown, at_u, &self.x);
                let d = tk[0] * tu[0] + tk[1] * tu[1];
                if d != 0.0 && (d > 0.0) != same_sense {
                    let sweep = self.arc_sweep(unknown);
                    let sweep = if sweep > 0.0 { sweep - std::f64::consts::TAU } else { sweep + std::f64::consts::TAU };
                    self.set_arc_sweep(unknown, sweep);
                    flipped += 1;
                }
                confirmed[unknown] = true;
                changed = true;
            }
            if !changed {
                break;
            }
        }
        flipped
    }

    /// Variables an entity is made of.
    pub fn entity_vars(&self, e: EntId) -> Vec<VarId> {
        match self.entities[e] {
            Entity::Point { x, y } => vec![x, y],
            Entity::Line { ax, ay, bx, by } => vec![ax, ay, bx, by],
            Entity::Arc { cx, cy, r, a0, a1, .. } => vec![cx, cy, r, a0, a1],
            Entity::BulgeArc { ax, ay, bx, by, sag } => vec![ax, ay, bx, by, sag],
            Entity::Circle { cx, cy, r } => vec![cx, cy, r],
            Entity::Ellipse { cx, cy, ra, rb, ang } => vec![cx, cy, ra, rb, ang],
        }
    }

    /// Current position of a point reference.
    pub fn point(&self, p: PointRef) -> [f64; 2] {
        point_of::<f64>(self, p, &self.x)
    }

    /// Endpoints of a line entity.
    pub fn line_points(&self, e: EntId) -> ([f64; 2], [f64; 2]) {
        (self.point(PointRef::Start(e)), self.point(PointRef::End(e)))
    }

    /// Residuals of one constraint (one or two numbers, length units).
    pub fn constraint_residuals(&self, i: usize) -> Vec<f64> {
        residuals_of::<f64>(self, &self.constraints[i], &self.x)
    }

    /// Analytic Jacobian rows of one constraint: for each variable it
    /// depends on, the derivative of each of its residuals.
    pub fn constraint_jacobian(&self, i: usize) -> Vec<(VarId, Vec<f64>)> {
        let c = &self.constraints[i];
        let mut xd: Vec<Dual> = self.x.iter().map(|&v| Dual::c(v)).collect();
        let mut out = Vec::new();
        for v in self.constraint_vars(c) {
            xd[v] = Dual::var(self.x[v]);
            let rd = residuals_of::<Dual>(self, c, &xd);
            xd[v] = Dual::c(self.x[v]);
            out.push((v, rd.iter().map(|d| d.d).collect()));
        }
        out
    }

    /// All residuals in constraint order.
    pub fn residuals(&self) -> Vec<f64> {
        (0..self.constraints.len()).flat_map(|i| self.constraint_residuals(i)).collect()
    }

    /// Largest absolute residual and the constraint holding it.
    pub fn worst_residual(&self) -> (f64, Option<usize>) {
        let mut worst = (0.0, None);
        for i in 0..self.constraints.len() {
            for r in self.constraint_residuals(i) {
                if r.abs() > worst.0 {
                    worst = (r.abs(), Some(i));
                }
            }
        }
        worst
    }

    fn constraint_vars(&self, c: &Constraint) -> Vec<VarId> {
        let mut v = Vec::new();
        for e in self.constraint_vars_entities(c) {
            v.extend(self.entity_vars(e));
        }
        if let Constraint::FixVar { var, .. } = *c {
            v.push(var);
        }
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Entities a constraint refers to.
    pub fn constraint_vars_entities(&self, c: &Constraint) -> Vec<EntId> {
        let mut v: Vec<EntId> = Vec::new();
        let mut pr = |p: PointRef| v.push(p.entity());
        match *c {
            Constraint::Coincident(a, b) | Constraint::HorizontalPoints(a, b) | Constraint::VerticalPoints(a, b) => {
                pr(a);
                pr(b);
            }
            Constraint::PointOnLine(p, e) | Constraint::PointOnCircle(p, e) | Constraint::PointOnEllipse(p, e) => {
                pr(p);
                v.push(e);
            }
            Constraint::Collinear(a, b)
            | Constraint::Parallel(a, b)
            | Constraint::Perpendicular(a, b)
            | Constraint::EqualLength(a, b)
            | Constraint::EqualRadius(a, b) => {
                v.push(a);
                v.push(b);
            }
            Constraint::HorizontalLine(e) | Constraint::VerticalLine(e) => v.push(e),
            Constraint::Length { line, .. } => v.push(line),
            Constraint::PointPointDistance { a, b, .. } => {
                pr(a);
                pr(b);
            }
            Constraint::PointLineDistance { point, line, .. } => {
                pr(point);
                v.push(line);
            }
            Constraint::LineLineDistance { a, b, .. } | Constraint::TangentCircles { a, b, .. } => {
                v.push(a);
                v.push(b);
            }
            Constraint::PointCircleDistance { point, circle, .. } => {
                pr(point);
                v.push(circle);
            }
            Constraint::Midpoint { mid, a, b } => {
                pr(mid);
                pr(a);
                pr(b);
            }
            Constraint::TangentLineCircle { line, circle, .. } => {
                v.push(line);
                v.push(circle);
            }
            Constraint::Radius { circle, .. } => v.push(circle),
            Constraint::EllipseRadius { ellipse, .. } => v.push(ellipse),
            Constraint::Angle { a, b, .. } => {
                v.push(a);
                v.push(b);
            }
            Constraint::EllipseAxis { ellipse, .. } => v.push(ellipse),
            Constraint::TangentAt { a, b, .. } => {
                v.push(a);
                v.push(b);
            }
            Constraint::ArcMidpoint { point, arc, .. } => {
                pr(point);
                v.push(arc);
            }
            Constraint::FixVar { .. } => {}
        }
        v.sort_unstable();
        v.dedup();
        v
    }
}

// ------------------------------------------------------------ geometry

fn point_of<T: Real>(sk: &Sketch, p: PointRef, x: &[T]) -> [T; 2] {
    let e = sk.entities[p.entity()];
    match (p, e) {
        (PointRef::Point(_), Entity::Point { x: px, y: py }) => [x[px], x[py]],
        (PointRef::Start(_), Entity::Line { ax, ay, .. }) => [x[ax], x[ay]],
        (PointRef::End(_), Entity::Line { bx, by, .. }) => [x[bx], x[by]],
        (PointRef::Start(_), Entity::Arc { cx, cy, r, a0, .. }) => [x[cx] + x[r] * x[a0].cos(), x[cy] + x[r] * x[a0].sin()],
        (PointRef::End(_), Entity::Arc { cx, cy, r, a1, .. }) => [x[cx] + x[r] * x[a1].cos(), x[cy] + x[r] * x[a1].sin()],
        (PointRef::Start(_), Entity::BulgeArc { ax, ay, .. }) => [x[ax], x[ay]],
        (PointRef::End(_), Entity::BulgeArc { bx, by, .. }) => [x[bx], x[by]],
        (PointRef::Center(_), Entity::BulgeArc { .. }) => bulge_center::<T>(sk, p.entity(), x).0,
        (PointRef::Center(_), Entity::Arc { cx, cy, .. })
        | (PointRef::Center(_), Entity::Circle { cx, cy, .. })
        | (PointRef::Center(_), Entity::Ellipse { cx, cy, .. }) => [x[cx], x[cy]],
        // Midpoint of a line, for callers that treat a line as its midpoint is
        // not a PointRef; any other combination is a builder error.
        (p, e) => panic!("point reference {p:?} does not apply to {e:?}"),
    }
}

fn line_of<T: Real>(sk: &Sketch, e: EntId, x: &[T]) -> ([T; 2], [T; 2]) {
    match sk.entities[e] {
        Entity::Line { ax, ay, bx, by } => ([x[ax], x[ay]], [x[bx], x[by]]),
        other => panic!("entity {e} is not a line: {other:?}"),
    }
}

fn circle_of<T: Real>(sk: &Sketch, e: EntId, x: &[T]) -> ([T; 2], T) {
    match sk.entities[e] {
        Entity::Arc { cx, cy, r, .. } | Entity::Circle { cx, cy, r } => ([x[cx], x[cy]], x[r]),
        Entity::BulgeArc { .. } => bulge_center::<T>(sk, e, x),
        other => panic!("entity {e} is not a circle or arc: {other:?}"),
    }
}

/// Centre and (positive) radius of a bulge arc. With chord length L and
/// bulge b: R = L (1 + b²) / (4 |b|); the centre sits at
/// (L / 2)(1 - b²) / (2 b) to the left of the chord's midpoint (negative
/// = right). Singular at b = 0 (a straight arc has no centre); callers
/// that need the centre only use it for arcs that are never straight.
/// Sagitta of an arc with chord `chord` and signed sweep: s = (L/2) tan(sweep/4).
pub fn sagitta_from_sweep(chord: f64, sweep: f64) -> f64 {
    chord * 0.5 * (sweep / 4.0).tan()
}

/// Bulge tan(sweep/4) of a bulge arc at values `x`: 2 s / L.
fn bulge_of<T: Real>(sk: &Sketch, e: EntId, x: &[T]) -> T {
    let Entity::BulgeArc { ax, ay, bx, by, sag } = sk.entities[e] else { panic!("not a bulge arc") };
    let l = norm(dir([x[ax], x[ay]], [x[bx], x[by]]));
    x[sag] * T::c(2.0) / longer(l, T::c(1e-12))
}

fn bulge_center<T: Real>(sk: &Sketch, e: EntId, x: &[T]) -> ([T; 2], T) {
    let Entity::BulgeArc { ax, ay, bx, by, .. } = sk.entities[e] else { panic!("not a bulge arc") };
    let (a, b) = ([x[ax], x[ay]], [x[bx], x[by]]);
    let c = dir(a, b);
    let l = norm(c);
    let bl = bulge_of::<T>(sk, e, x);
    let mid = [(a[0] + b[0]) * T::c(0.5), (a[1] + b[1]) * T::c(0.5)];
    let left = [-c[1] / longer(l, T::c(1e-12)), c[0] / longer(l, T::c(1e-12))];
    let two_b = bl * T::c(2.0);
    let two_b = if two_b.val().abs() < 1e-12 { T::c(1e-12) } else { two_b };
    let d = l * T::c(0.5) * (T::c(1.0) - bl * bl) / two_b;
    let r = l * (T::c(1.0) + bl * bl) / (two_b * T::c(2.0));
    ([mid[0] + left[0] * d, mid[1] + left[1] * d], r.abs())
}

/// The other intersection of the chord's bisector with the arc's circle:
/// at distance L² / (4 s) from the chord midpoint on the side opposite the
/// arc (the two bisector segments of the diameter multiply to (L/2)²).
fn bulge_far_point<T: Real>(sk: &Sketch, e: EntId, x: &[T]) -> [T; 2] {
    let Entity::BulgeArc { ax, ay, bx, by, sag } = sk.entities[e] else { panic!("not a bulge arc") };
    let (a, b) = ([x[ax], x[ay]], [x[bx], x[by]]);
    let c = dir(a, b);
    let l = longer(norm(c), T::c(1e-12));
    let mid = [(a[0] + b[0]) * T::c(0.5), (a[1] + b[1]) * T::c(0.5)];
    let sag = if x[sag].val().abs() < 1e-12 { T::c(1e-12) } else { x[sag] };
    // offset along -left normal is +s for the near point; the far point is at -(L²/(4s)) along the same direction
    let t = -(l * l) / (sag * T::c(4.0)) / l;
    [mid[0] + c[1] * t, mid[1] - c[0] * t]
}

/// Point at the middle of a bulge arc: chord midpoint minus the left unit
/// normal times the sagitta. Smooth through straight.
fn bulge_midpoint<T: Real>(sk: &Sketch, e: EntId, x: &[T]) -> [T; 2] {
    let Entity::BulgeArc { ax, ay, bx, by, sag } = sk.entities[e] else { panic!("not a bulge arc") };
    let (a, b) = ([x[ax], x[ay]], [x[bx], x[by]]);
    let c = dir(a, b);
    let l = longer(norm(c), T::c(1e-12));
    let mid = [(a[0] + b[0]) * T::c(0.5), (a[1] + b[1]) * T::c(0.5)];
    let s = x[sag] / l;
    [mid[0] + c[1] * s, mid[1] - c[0] * s]
}

/// Tangent direction of a curve at one of its points (unnormalised). For a
/// line it is the line direction; for an arc / circle the direction
/// perpendicular to the radius at the point's angle.
fn tangent_at<T: Real>(sk: &Sketch, e: EntId, at: PointRef, x: &[T]) -> [T; 2] {
    match sk.entities[e] {
        Entity::Line { ax, ay, bx, by } => [x[bx] - x[ax], x[by] - x[ay]],
        Entity::Arc { cx, cy, a0, a1, .. } => {
            if at.entity() != e {
                let p = point_of::<T>(sk, at, x);
                return [-(p[1] - x[cy]), p[0] - x[cx]];
            }
            let a = match at {
                PointRef::Start(_) => x[a0],
                PointRef::End(_) => x[a1],
                other => panic!("tangent of an arc at {other:?}"),
            };
            [-a.sin(), a.cos()]
        }
        Entity::BulgeArc { ax, ay, bx, by, .. } => {
            if at.entity() != e {
                // A point of another entity lying on this arc: perpendicular
                // to the radius (needs the centre, so not for straight arcs).
                let (c, _) = bulge_center::<T>(sk, e, x);
                let p = point_of::<T>(sk, at, x);
                return [-(p[1] - c[1]), p[0] - c[0]];
            }
            // Rotate the chord by -sweep/2 at the start, +sweep/2 at the end;
            // cos(sweep/2) = (1 - b²)/(1 + b²), sin(sweep/2) = 2b/(1 + b²).
            let c = [x[bx] - x[ax], x[by] - x[ay]];
            let b = bulge_of::<T>(sk, e, x);
            let den = T::c(1.0) + b * b;
            let (cs, sn) = ((T::c(1.0) - b * b) / den, b * T::c(2.0) / den);
            let sn = match at {
                PointRef::Start(_) => -sn,
                PointRef::End(_) => sn,
                other => panic!("tangent of an arc at {other:?}"),
            };
            [c[0] * cs - c[1] * sn, c[0] * sn + c[1] * cs]
        }
        Entity::Circle { cx, cy, .. } => {
            let p = point_of::<T>(sk, at, x);
            [-(p[1] - x[cy]), p[0] - x[cx]]
        }
        Entity::Ellipse { cx, cy, ra, rb, ang } => {
            // Parameter of the point on the ellipse, then the derivative there.
            let p = point_of::<T>(sk, at, x);
            let (dx, dy) = (p[0] - x[cx], p[1] - x[cy]);
            let (ca, sa) = (x[ang].cos(), x[ang].sin());
            let u = (dx * ca + dy * sa) / x[ra];
            let v = (-dx * sa + dy * ca) / x[rb];
            let t = v.atan2(u);
            let (du, dv) = (-t.sin() * x[ra], t.cos() * x[rb]);
            [du * ca - dv * sa, du * sa + dv * ca]
        }
        Entity::Point { .. } => panic!("tangent of a point"),
    }
}

/// The larger of two values, with a floor so a division by it stays finite.
fn longer<T: Real>(a: T, b: T) -> T {
    let m = if a.val() >= b.val() { a } else { b };
    if m.val() < 1e-12 { T::c(1e-12) } else { m }
}

fn dir<T: Real>(a: [T; 2], b: [T; 2]) -> [T; 2] {
    [b[0] - a[0], b[1] - a[1]]
}

fn norm<T: Real>(v: [T; 2]) -> T {
    v[0].hypot(v[1])
}

fn cross<T: Real>(a: [T; 2], b: [T; 2]) -> T {
    a[0] * b[1] - a[1] * b[0]
}

fn dot<T: Real>(a: [T; 2], b: [T; 2]) -> T {
    a[0] * b[0] + a[1] * b[1]
}

/// Signed perpendicular distance of `p` from the infinite line `a b`
/// (positive on the left of a->b).
fn line_signed_distance<T: Real>(p: [T; 2], a: [T; 2], b: [T; 2]) -> T {
    let d = dir(a, b);
    cross(d, dir(a, p)) / norm(d)
}

/// Signed distance of `p` from line `a b` times the line's length over
/// ANGLE_SCALE: equal to the distance for a line of length ANGLE_SCALE,
/// weaker for shorter lines, and smooth when the line collapses.
fn line_weighted_distance<T: Real>(p: [T; 2], a: [T; 2], b: [T; 2]) -> T {
    let d = dir(a, b);
    cross(d, dir(a, p)) / T::c(ANGLE_SCALE)
}

fn residuals_of<T: Real>(sk: &Sketch, c: &Constraint, x: &[T]) -> Vec<T> {
    let pt = |p: PointRef| point_of::<T>(sk, p, x);
    let ln = |e: EntId| line_of::<T>(sk, e, x);
    let ci = |e: EntId| circle_of::<T>(sk, e, x);
    let k = T::c;
    match *c {
        Constraint::Coincident(a, b) => {
            let (p, q) = (pt(a), pt(b));
            vec![p[0] - q[0], p[1] - q[1]]
        }
        // Point-on-line and collinearity are weighted by the line's length
        // relative to ANGLE_SCALE (cross product / reference length) so a
        // collapsing line does not make the condition violently nonlinear.
        Constraint::PointOnLine(p, e) => {
            let (a, b) = ln(e);
            vec![line_weighted_distance(pt(p), a, b)]
        }
        Constraint::PointOnCircle(p, e) => {
            let (c, r) = ci(e);
            vec![norm(dir(c, pt(p))) - r]
        }
        Constraint::PointOnEllipse(p, e) => {
            let Entity::Ellipse { cx, cy, ra, rb, ang } = sk.entities[e] else { panic!("not an ellipse") };
            let q = pt(p);
            let (dx, dy) = (q[0] - x[cx], q[1] - x[cy]);
            let (ca, sa) = (x[ang].cos(), x[ang].sin());
            let u = (dx * ca + dy * sa) / x[ra];
            let v = (-dx * sa + dy * ca) / x[rb];
            // (|q|_ellipse - 1) scaled by the smaller radius approximates a distance.
            let rmin = if x[ra].val() < x[rb].val() { x[ra] } else { x[rb] };
            vec![((u * u + v * v).sqrt() - k(1.0)) * rmin]
        }
        Constraint::Collinear(a, b) => {
            let (p, q) = ln(a);
            let (s, t) = ln(b);
            vec![line_weighted_distance(s, p, q), line_weighted_distance(t, p, q)]
        }
        Constraint::HorizontalLine(e) => {
            let (a, b) = ln(e);
            vec![a[1] - b[1]]
        }
        Constraint::VerticalLine(e) => {
            let (a, b) = ln(e);
            vec![a[0] - b[0]]
        }
        Constraint::HorizontalPoints(a, b) => vec![pt(a)[1] - pt(b)[1]],
        Constraint::VerticalPoints(a, b) => vec![pt(a)[0] - pt(b)[0]],
        // Direction conditions are scaled by the shorter line (residual =
        // shorter length x sin / cos of the angle) so a line collapsing to
        // zero length, which real sketches do at some sizes, degrades smoothly
        // instead of dividing by zero.
        Constraint::Parallel(a, b) => {
            let (p, q) = ln(a);
            let (s, t) = ln(b);
            let (d1, d2) = (dir(p, q), dir(s, t));
            vec![cross(d1, d2) / longer(norm(d1), norm(d2))]
        }
        Constraint::Perpendicular(a, b) => {
            let (p, q) = ln(a);
            let (s, t) = ln(b);
            let (d1, d2) = (dir(p, q), dir(s, t));
            vec![dot(d1, d2) / longer(norm(d1), norm(d2))]
        }
        Constraint::Length { line, value } => {
            let (a, b) = ln(line);
            vec![norm(dir(a, b)) - k(value)]
        }
        Constraint::PointPointDistance { a, b, mode, sign, value } => {
            let (p, q) = (pt(a), pt(b));
            let d = match mode {
                DistanceMode::Minimum => norm(dir(q, p)),
                DistanceMode::Horizontal => (p[0] - q[0]) * k(sign),
                DistanceMode::Vertical => (p[1] - q[1]) * k(sign),
            };
            vec![d - k(value)]
        }
        Constraint::PointLineDistance { point, line, sign, value } => {
            let (a, b) = ln(line);
            vec![line_signed_distance(pt(point), a, b) * k(sign) - k(value)]
        }
        Constraint::LineLineDistance { a, b, sign, value } => {
            let (p, q) = ln(a);
            let (s, t) = ln(b);
            vec![line_signed_distance(s, p, q) * k(sign) - k(value), line_signed_distance(t, p, q) * k(sign) - k(value)]
        }
        Constraint::PointCircleDistance { point, circle, sign, value } => {
            let (c, r) = ci(circle);
            vec![(norm(dir(c, pt(point))) - r) * k(sign) - k(value)]
        }
        Constraint::Midpoint { mid, a, b } => {
            let (m, p, q) = (pt(mid), pt(a), pt(b));
            vec![m[0] - (p[0] + q[0]) * k(0.5), m[1] - (p[1] + q[1]) * k(0.5)]
        }
        Constraint::TangentLineCircle { line, circle, sign } => {
            let (a, b) = ln(line);
            let (c, r) = ci(circle);
            vec![line_signed_distance(c, a, b) * k(sign) - r]
        }
        Constraint::TangentCircles { a, b, external, sign } => {
            let (c1, r1) = ci(a);
            let (c2, r2) = ci(b);
            let d = norm(dir(c1, c2));
            vec![if external { d - (r1 + r2) } else { d - (r1 - r2) * k(sign) }]
        }
        Constraint::EqualLength(a, b) => {
            let (p, q) = ln(a);
            let (s, t) = ln(b);
            vec![norm(dir(p, q)) - norm(dir(s, t))]
        }
        Constraint::EqualRadius(a, b) => vec![ci(a).1 - ci(b).1],
        Constraint::Radius { circle, value } => vec![ci(circle).1 - k(value)],
        Constraint::EllipseRadius { ellipse, major, value } => {
            let Entity::Ellipse { ra, rb, .. } = sk.entities[ellipse] else { panic!("not an ellipse") };
            vec![(if major { x[ra] } else { x[rb] }) - k(value)]
        }
        Constraint::Angle { a, b, value, sign, offset } => {
            let (p, q) = ln(a);
            let (s, t) = ln(b);
            let (d1, d2) = (dir(p, q), dir(s, t));
            let phi = cross(d1, d2).atan2(dot(d1, d2));
            let target = sign * value + offset;
            // Wrap the difference into (-π, π] so the residual is continuous near the seed.
            let mut diff = phi.val() - target;
            let two_pi = std::f64::consts::TAU;
            diff -= (diff / two_pi).round() * two_pi;
            // phi - (target + wrap correction): keep derivative through phi.
            let corr = (phi.val() - target) - diff;
            vec![(phi - k(target + corr)) * k(ANGLE_SCALE)]
        }
        Constraint::TangentAt { a, at_a, b, at_b } => {
            // Arc tangents are unit vectors, a line's tangent has its length:
            // the cross product over the longer of the two (or ANGLE_SCALE)
            // weights the condition by the shorter length, so a collapsing
            // line's tangency fades smoothly.
            let ta = tangent_at::<T>(sk, a, at_a, x);
            let tb = tangent_at::<T>(sk, b, at_b, x);
            let m = longer(longer(norm(ta), norm(tb)), k(ANGLE_SCALE));
            vec![cross(ta, tb) / m * k(ANGLE_SCALE)]
        }
        Constraint::EllipseAxis { ellipse, horizontal } => {
            let Entity::Ellipse { ra, ang, .. } = sk.entities[ellipse] else { panic!("not an ellipse") };
            vec![x[ra] * if horizontal { x[ang].sin() } else { x[ang].cos() }]
        }
        Constraint::ArcMidpoint { point, arc, far } if matches!(sk.entities[arc], Entity::BulgeArc { .. }) => {
            let p = pt(point);
            let m = if far { bulge_far_point::<T>(sk, arc, x) } else { bulge_midpoint::<T>(sk, arc, x) };
            vec![p[0] - m[0], p[1] - m[1]]
        }
        Constraint::ArcMidpoint { point, arc, .. } => {
            let Entity::Arc { cx, cy, r, .. } = sk.entities[arc] else { panic!("not an arc") };
            let p = pt(point);
            let s = pt(PointRef::Start(arc));
            let e = pt(PointRef::End(arc));
            let chord = dir(s, e);
            let mid = [(s[0] + e[0]) * k(0.5), (s[1] + e[1]) * k(0.5)];
            let on_circle = norm(dir([x[cx], x[cy]], p)) - x[r];
            let on_bisector = dot(dir(mid, p), chord) / norm(chord);
            vec![on_circle, on_bisector]
        }
        Constraint::FixVar { var, value } => vec![x[var] - k(value)],
    }
}

// ------------------------------------------------------------ solver

/// Solve the sketch in place with Levenberg-Marquardt from the current
/// variable values. Free variables move; fixed ones never do.
pub fn solve(sk: &mut Sketch, opts: SolveOptions) -> Result<SolveReport, SolveError> {
    let free: Vec<VarId> = (0..sk.x.len()).filter(|&i| !sk.fixed[i]).collect();
    let col_of: Vec<Option<usize>> = {
        let mut m = vec![None; sk.x.len()];
        for (c, &v) in free.iter().enumerate() {
            m[v] = Some(c);
        }
        m
    };
    let n = free.len();
    let cvars: Vec<Vec<VarId>> = sk.constraints.iter().map(|c| sk.constraint_vars(c)).collect();
    let equations: usize = (0..sk.constraints.len()).map(|i| sk.constraint_residuals(i).len()).sum();

    let mut lambda = opts.initial_damping;
    let mut iterations = 0;
    let (mut worst, mut worst_i) = sk.worst_residual();
    if n == 0 {
        return Ok(SolveReport { converged: worst <= opts.tolerance, iterations: 0, max_residual: worst, worst_constraint: worst_i, free_vars: 0, equations, progress: 1.0 });
    }
    let mut cost = total_cost(sk);
    let sparse = n > dense_limit();
    while worst > opts.tolerance && iterations < opts.max_iterations {
        iterations += 1;
        // Jacobian rows: (residual value, [(column, derivative)]).
        let mut rows: Vec<(f64, Vec<(usize, f64)>)> = Vec::with_capacity(equations);
        let mut xd: Vec<Dual> = sk.x.iter().map(|&v| Dual::c(v)).collect();
        for (ci, c) in sk.constraints.iter().enumerate() {
            let r0 = residuals_of::<f64>(sk, c, &sk.x);
            let m = r0.len();
            let mut cols: Vec<(usize, Vec<f64>)> = Vec::new();
            for &v in &cvars[ci] {
                let Some(col) = col_of[v] else { continue };
                xd[v] = Dual::var(sk.x[v]);
                let rd = residuals_of::<Dual>(sk, c, &xd);
                xd[v] = Dual::c(sk.x[v]);
                cols.push((col, rd.iter().map(|d| d.d).collect()));
            }
            for k in 0..m {
                rows.push((r0[k], cols.iter().map(|(c, d)| (*c, d[k])).collect()));
            }
        }
        if std::env::var("SKETCH_TRACE").is_ok_and(|v| v == "2") {
            let bad: Vec<usize> = rows.iter().enumerate().filter(|(_, (r, cols))| !r.is_finite() || cols.iter().any(|(_, d)| !d.is_finite())).map(|(i, _)| i).collect();
            let maxd = rows.iter().flat_map(|(_, c)| c.iter().map(|(_, d)| d.abs())).fold(0.0f64, f64::max);
            eprintln!("    jacobian: {} rows, {} non-finite rows {:?}, max |d| {maxd:.2e}, cost {cost:.3e}", rows.len(), bad.len(), &bad[..bad.len().min(5)]);
        }
        if let Some((x0, w)) = &sk.anchor {
            for (c, &v) in free.iter().enumerate() {
                rows.push((w * (sk.x[v] - x0[v]), vec![(c, *w)]));
            }
        }
        // g = Jᵀr and the diagonal of JᵀJ; the full dense JᵀJ only when small.
        let mut g = vec![0.0; n];
        let mut diag = vec![0.0; n];
        for (r, cols) in &rows {
            for (c, d) in cols {
                g[*c] += d * r;
                diag[*c] += d * d;
            }
        }
        // No free variable influences any residual: nothing to do.
        if g.iter().all(|v| v.abs() < 1e-20) {
            break;
        }
        // Under-constrained sketches have near-null directions where the
        // curvature is ~0 and the gradient is rounding noise; a small absolute
        // Tikhonov term (NULL_DAMPING, independent of lambda) keeps their
        // steps tiny without slowing the well-conditioned directions.
        let mut a = Vec::new();
        if !sparse {
            a = vec![0.0; n * n];
            for (_, cols) in &rows {
                for (ca, da) in cols {
                    for (cb, db) in cols {
                        a[ca * n + cb] += da * db;
                    }
                }
            }
        }
        // Try steps with increasing damping until the cost drops.
        let mut accepted = false;
        for _ in 0..20 {
            let rhs: Vec<f64> = g.iter().map(|v| -v).collect();
            let delta = if sparse {
                pcg_normal(&rows, &diag, lambda, &rhs, n)
            } else {
                let mut m = a.clone();
                for i in 0..n {
                    m[i * n + i] += lambda * diag[i].max(1e-12) + NULL_DAMPING;
                }
                cholesky_solve(&mut m, n, &rhs)
            };
            let Some(delta) = delta else {
                lambda *= 10.0;
                continue;
            };
            if delta.iter().any(|d| d.abs() > opts.max_step || !d.is_finite()) {
                lambda *= 10.0;
                continue;
            }
            let saved = sk.x.clone();
            for (c, &v) in free.iter().enumerate() {
                sk.x[v] += delta[c];
            }
            let new_cost = total_cost(sk);
            if std::env::var("SKETCH_TRACE").is_ok_and(|v| v == "2") {
                eprintln!("    iter {iterations} lambda {lambda:.1e} cost {cost:.3e} -> {new_cost:.3e} step {:.2e}", delta.iter().map(|d| d.abs()).fold(0.0f64, f64::max));
            }
            if new_cost.is_finite() && new_cost < cost {
                cost = new_cost;
                lambda = (lambda / 10.0).max(1e-15);
                accepted = true;
                break;
            }
            sk.x = saved;
            lambda *= 10.0;
        }
        let (w, wi) = sk.worst_residual();
        worst = w;
        worst_i = wi;
        if !accepted {
            break;
        }
    }
    Ok(SolveReport { converged: worst <= opts.tolerance, iterations, max_residual: worst, worst_constraint: worst_i, free_vars: n, equations, progress: 1.0 })
}

/// Solve by continuation: `set(t)` writes the constraint targets for
/// parameter `t` in [0, 1] (0 = where the seed is known to satisfy them,
/// 1 = the wanted targets); the sketch is solved at increasing `t`, halving
/// the step whenever a solve fails to converge, so the solution tracks the
/// branch the seed is on through large parameter moves.
pub fn solve_continuation(
    sk: &mut Sketch,
    opts: SolveOptions,
    mut set: impl FnMut(&mut Sketch, f64),
) -> Result<SolveReport, SolveError> {
    let mut t: f64 = 0.0;
    let mut step: f64 = 1.0;
    let mut last = None;
    let mut steps = 0;
    while t < 1.0 {
        let target = (t + step).min(1.0);
        set(sk, target);
        let saved = sk.x.clone();
        let r = solve(sk, opts)?;
        steps += 1;
        if std::env::var_os("SKETCH_TRACE").is_some() {
            eprintln!("  continuation t={t:.4} -> {target:.4}: converged {} in {} iterations, residual {:.2e}", r.converged, r.iterations, r.max_residual);
        }
        if r.converged {
            t = target;
            last = Some(r);
            step = (step * 2.0).min(1.0 - t).max(1e-6);
        } else {
            sk.x = saved;
            step /= 2.0;
            if step < 1e-4 {
                // Stuck: report the state at the last converged t, with the
                // failing step's target set so the worst constraint is visible.
                set(sk, target);
                let mut r = solve(sk, opts)?;
                r.iterations = steps;
                r.progress = t;
                r.converged = false;
                return Ok(r);
            }
        }
        if steps > 10_000 {
            return Err(SolveError("continuation did not finish".into()));
        }
    }
    let mut r = last.unwrap_or(SolveReport { converged: true, iterations: 0, max_residual: 0.0, worst_constraint: None, free_vars: 0, equations: 0, progress: 1.0 });
    r.iterations = steps;
    r.progress = 1.0;
    Ok(r)
}

/// Absolute damping added to every diagonal entry of the normal matrix (in
/// the units of JᵀJ, i.e. residual² per variable²; residuals and variables
/// are both in model length units or radians).
const NULL_DAMPING: f64 = 1e-12;

/// Systems with more free variables than this use the sparse path.
fn dense_limit() -> usize {
    std::env::var("SKETCH_DENSE_LIMIT").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

/// Preconditioned conjugate gradients on (JᵀJ + λ diag) δ = rhs with J given
/// as sparse rows. Jacobi preconditioning; returns `None` if it fails to make
/// progress.
fn pcg_normal(rows: &[(f64, Vec<(usize, f64)>)], diag: &[f64], lambda: f64, rhs: &[f64], n: usize) -> Option<Vec<f64>> {
    let damp: Vec<f64> = diag.iter().map(|d| lambda * d.max(1e-12) + NULL_DAMPING).collect();
    let apply = |p: &[f64], out: &mut Vec<f64>| {
        for (o, (pi, di)) in out.iter_mut().zip(p.iter().zip(&damp)) {
            *o = pi * di;
        }
        for (_, cols) in rows {
            let mut jp = 0.0;
            for (c, d) in cols {
                jp += d * p[*c];
            }
            if jp != 0.0 {
                for (c, d) in cols {
                    out[*c] += d * jp;
                }
            }
        }
    };
    let minv: Vec<f64> = diag.iter().zip(&damp).map(|(d, l)| 1.0 / (d + l)).collect();
    let mut x = vec![0.0; n];
    let mut r = rhs.to_vec();
    let mut z: Vec<f64> = r.iter().zip(&minv).map(|(a, b)| a * b).collect();
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let rhs_norm: f64 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    if rhs_norm == 0.0 {
        return Some(x);
    }
    let mut ap = vec![0.0; n];
    // Inexact solves are fine for a damped step; cap the work per solve.
    for _ in 0..(2 * n).clamp(50, 2000) {
        apply(&p, &mut ap);
        let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        if pap <= 0.0 || !pap.is_finite() {
            if std::env::var("SKETCH_TRACE").is_ok_and(|v| v == "2") {
                eprintln!("    pcg stop: pAp {pap:.3e} rhs norm {rhs_norm:.3e}");
            }
            break;
        }
        let alpha = rz / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rn: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if rn <= 1e-12 * rhs_norm {
            break;
        }
        for i in 0..n {
            z[i] = r[i] * minv[i];
        }
        let rz_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    if x.iter().all(|v| v.is_finite()) { Some(x) } else { None }
}

fn sum_sq(r: &[f64]) -> f64 {
    r.iter().map(|v| v * v).sum()
}

/// Least-squares cost including the anchor rows.
fn total_cost(sk: &Sketch) -> f64 {
    let mut c = sum_sq(&sk.residuals());
    if let Some((x0, w)) = &sk.anchor {
        for v in 0..sk.x.len() {
            if !sk.fixed[v] {
                c += (w * (sk.x[v] - x0[v])).powi(2);
            }
        }
    }
    c
}

/// In-place Cholesky solve of the symmetric positive definite `m` (n×n, row
/// major). Returns `None` if the matrix is not positive definite.
fn cholesky_solve(m: &mut [f64], n: usize, b: &[f64]) -> Option<Vec<f64>> {
    for j in 0..n {
        let mut d = m[j * n + j];
        for k in 0..j {
            d -= m[j * n + k] * m[j * n + k];
        }
        if d <= 0.0 || !d.is_finite() {
            return None;
        }
        let d = d.sqrt();
        m[j * n + j] = d;
        for i in j + 1..n {
            let mut s = m[i * n + j];
            for k in 0..j {
                s -= m[i * n + k] * m[j * n + k];
            }
            m[i * n + j] = s / d;
        }
    }
    // Forward substitution L y = b, then back substitution Lᵀ x = y.
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= m[i * n + k] * y[k];
        }
        y[i] = s / m[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in i + 1..n {
            s -= m[k * n + i] * x[k];
        }
        x[i] = s / m[i * n + i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solved(sk: &mut Sketch) -> SolveReport {
        // The tests work at unit scale, so allow unit-scale steps.
        let r = solve(sk, SolveOptions { tolerance: 1e-11, ..Default::default() }).unwrap();
        assert!(r.converged, "did not converge: {r:?}");
        r
    }

    #[test]
    fn rectangle_with_dimensions() {
        let mut sk = Sketch::new();
        let o = sk.add_point([0.0, 0.0], true);
        // Rough rectangle seed, to be sized to 4 x 2 with a corner at the origin.
        let bottom = sk.add_line([0.1, 0.1], [3.0, 0.2], false);
        let right = sk.add_line([3.0, 0.2], [3.1, 1.5], false);
        let top = sk.add_line([3.1, 1.5], [0.0, 1.6], false);
        let left = sk.add_line([0.0, 1.6], [0.1, 0.1], false);
        sk.add_constraint(Constraint::Coincident(PointRef::End(bottom), PointRef::Start(right)));
        sk.add_constraint(Constraint::Coincident(PointRef::End(right), PointRef::Start(top)));
        sk.add_constraint(Constraint::Coincident(PointRef::End(top), PointRef::Start(left)));
        sk.add_constraint(Constraint::Coincident(PointRef::End(left), PointRef::Start(bottom)));
        sk.add_constraint(Constraint::Coincident(PointRef::Start(bottom), PointRef::Point(o)));
        sk.add_constraint(Constraint::HorizontalLine(bottom));
        sk.add_constraint(Constraint::HorizontalLine(top));
        sk.add_constraint(Constraint::VerticalLine(left));
        sk.add_constraint(Constraint::VerticalLine(right));
        sk.add_constraint(Constraint::Length { line: bottom, value: 4.0 });
        sk.add_constraint(Constraint::Length { line: right, value: 2.0 });
        solved(&mut sk);
        let (a, b) = sk.line_points(top);
        assert!((a[0] - 4.0).abs() < 1e-8 && (a[1] - 2.0).abs() < 1e-8, "{a:?}");
        assert!(b[0].abs() < 1e-8 && (b[1] - 2.0).abs() < 1e-8, "{b:?}");
        assert_eq!(sk.point(PointRef::Point(o)), [0.0, 0.0]);
    }

    #[test]
    fn slot_with_tangent_arcs() {
        // Two horizontal lines joined by two semicircular arcs; radius and
        // centre distance dimensioned.
        let mut sk = Sketch::new();
        let top = sk.add_line([0.0, 1.0], [5.0, 1.0], false);
        let bottom = sk.add_line([5.0, -1.0], [0.0, -1.0], false);
        let pi = std::f64::consts::PI;
        let right = sk.add_arc([5.0, 0.0], 1.0, pi / 2.0, -pi / 2.0, false, false); // clockwise from top to bottom
        let left = sk.add_arc([0.0, 0.0], 1.0, -pi / 2.0, pi / 2.0, true, false); // hmm: from bottom to top going through the left side
        // Left arc: start at bottom (-π/2), end at top (π/2) counter-clockwise passes through the right side;
        // for a slot it must pass through the left side, i.e. clockwise: use angles 3π/2 -> π/2 with ccw=false.
        let _ = left;
        let left = sk.add_arc([0.0, 0.0], 1.0, 3.0 * pi / 2.0, pi / 2.0, false, false);
        sk.add_constraint(Constraint::Coincident(PointRef::End(top), PointRef::Start(right)));
        sk.add_constraint(Constraint::Coincident(PointRef::End(right), PointRef::Start(bottom)));
        sk.add_constraint(Constraint::Coincident(PointRef::End(bottom), PointRef::Start(left)));
        sk.add_constraint(Constraint::Coincident(PointRef::End(left), PointRef::Start(top)));
        sk.add_constraint(Constraint::TangentLineCircle { line: top, circle: right, sign: -1.0 });
        sk.add_constraint(Constraint::TangentLineCircle { line: bottom, circle: right, sign: -1.0 });
        sk.add_constraint(Constraint::TangentLineCircle { line: top, circle: left, sign: -1.0 });
        sk.add_constraint(Constraint::TangentLineCircle { line: bottom, circle: left, sign: -1.0 });
        sk.add_constraint(Constraint::HorizontalLine(top));
        sk.add_constraint(Constraint::Radius { circle: right, value: 1.5 });
        sk.add_constraint(Constraint::EqualRadius(right, left));
        sk.add_constraint(Constraint::PointPointDistance { a: PointRef::Center(right), b: PointRef::Center(left), mode: DistanceMode::Horizontal, sign: 1.0, value: 8.0 });
        sk.add_constraint(Constraint::FixVar { var: 0, value: 0.0 }); // pin top.start.x
        sk.add_constraint(Constraint::FixVar { var: 1, value: 1.5 }); // pin top.start.y
        solved(&mut sk);
        let (c_r, r_r) = circle_of::<f64>(&sk, right, &sk.x);
        let (c_l, r_l) = circle_of::<f64>(&sk, left, &sk.x);
        assert!((r_r - 1.5).abs() < 1e-8 && (r_l - 1.5).abs() < 1e-8);
        assert!((c_r[0] - c_l[0] - 8.0).abs() < 1e-8, "{c_r:?} {c_l:?}");
        let (a, b) = sk.line_points(bottom);
        assert!((a[1] + 1.5).abs() < 1e-8 && (b[1] + 1.5).abs() < 1e-8, "{a:?} {b:?}");
    }

    #[test]
    fn under_constrained_stays_near_seed() {
        let mut sk = Sketch::new();
        let l = sk.add_line([1.0, 1.0], [2.0, 1.0], false);
        sk.add_constraint(Constraint::Length { line: l, value: 3.0 });
        solved(&mut sk);
        let (a, b) = sk.line_points(l);
        // The line grows symmetrically about its seed rather than jumping.
        assert!((a[0] + b[0] - 3.0).abs() < 1e-6, "{a:?} {b:?}");
        assert!((a[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn angle_and_tangent_circles() {
        let mut sk = Sketch::new();
        let base = sk.add_line([0.0, 0.0], [1.0, 0.0], true);
        let l = sk.add_line([0.0, 0.0], [1.0, 0.9], false);
        sk.add_constraint(Constraint::Coincident(PointRef::Start(l), PointRef::Start(base)));
        sk.add_constraint(Constraint::Length { line: l, value: 2.0 });
        sk.add_constraint(Constraint::Angle { a: base, b: l, value: 60f64.to_radians(), sign: 1.0, offset: 0.0 });
        let c1 = sk.add_circle([0.0, 0.0], 1.0, true);
        let c2 = sk.add_circle([2.5, 0.0], 0.8, false);
        sk.add_constraint(Constraint::TangentCircles { a: c1, b: c2, external: true, sign: 1.0 });
        sk.add_constraint(Constraint::Radius { circle: c2, value: 0.5 });
        sk.add_constraint(Constraint::HorizontalPoints(PointRef::Center(c1), PointRef::Center(c2)));
        solved(&mut sk);
        let (_, b) = sk.line_points(l);
        assert!((b[0] - 1.0).abs() < 1e-8 && (b[1] - 3f64.sqrt()).abs() < 1e-8, "{b:?}");
        let (c, r) = circle_of::<f64>(&sk, c2, &sk.x);
        assert!((r - 0.5).abs() < 1e-8 && (c[0] - 1.5).abs() < 1e-8 && c[1].abs() < 1e-8, "{c:?} {r}");
    }

    #[test]
    fn dual_derivatives() {
        let x = Dual::var(2.0);
        let y = x * x + Dual::c(3.0) * x; // 10, d = 7
        assert!((y.v - 10.0).abs() < 1e-12 && (y.d - 7.0).abs() < 1e-12);
        let s = x.sqrt();
        assert!((s.d - 1.0 / (2.0 * 2f64.sqrt())).abs() < 1e-12);
        let a = Dual::var(1.0).atan2(Dual::c(1.0));
        assert!((a.d - 0.5).abs() < 1e-12);
    }
}
