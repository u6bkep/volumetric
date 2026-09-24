//! Solved sketch geometry from Onshape's sketches endpoint
//! (`GET /partstudios/.../sketches?output3D=true`), which reports every
//! sketch entity in world coordinates at the requested configuration.
//!
//! The feature-tree JSON stores each sketch as it was last edited, at
//! whatever configuration was active then, so it is not a consistent seed.
//! This module fits each sketch's plane frame from the 3D dump plus the
//! sketch's own horizontal / vertical constraints, maps the dump into sketch
//! coordinates and writes it over the import's seed. With the dimension
//! expressions evaluated at the same configuration every residual should
//! then vanish, which is the check that the import reads Onshape correctly.

use std::collections::HashMap;

use serde_json::Value as Json;
use sketch_solve_core::{Constraint, Entity, PointRef, SolveOptions, solve};

use crate::{ImportError, SketchImport};

#[derive(Clone, Debug, PartialEq)]
pub enum Geom3d {
    Point([f64; 3]),
    Line([f64; 3], [f64; 3]),
    Arc { center: [f64; 3], radius: f64, clockwise: bool, start: [f64; 3], end: [f64; 3] },
    Circle { center: [f64; 3], radius: f64 },
    Ellipse { center: [f64; 3], major: f64, minor: f64, direction: [f64; 3] },
}

#[derive(Clone, Debug)]
pub struct Entity3d {
    pub id: String,
    pub construction: bool,
    pub geom: Geom3d,
}

fn v3(j: &Json) -> Option<[f64; 3]> {
    let a = j.as_array()?;
    Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?, a.get(2)?.as_f64()?])
}

/// Parse the `sketches` array of the endpoint response into
/// sketch name -> entities. (`unreportedSketches` is ignored.)
pub fn parse_sketches_dump(dump: &Json) -> HashMap<String, Vec<Entity3d>> {
    let mut out = HashMap::new();
    for sk in dump["sketches"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        let name = sk["sketch"].as_str().unwrap_or("").to_string();
        let mut ents = Vec::new();
        for e in sk["geomEntities"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
            let id = e["id"].as_str().unwrap_or("").to_string();
            let construction = e["isConstruction"].as_bool().unwrap_or(false);
            let geom = match e["entityType"].as_str().unwrap_or("") {
                "point" => v3(&e["point"]).map(Geom3d::Point),
                "lineSegment" => Some(Geom3d::Line(v3(&e["startPoint"]).unwrap(), v3(&e["endPoint"]).unwrap())),
                "arc" => Some(Geom3d::Arc {
                    center: v3(&e["center"]).unwrap(),
                    radius: e["radius"].as_f64().unwrap_or(0.0),
                    clockwise: e["clockwise"].as_bool().unwrap_or(false),
                    start: v3(&e["startPoint"]).unwrap(),
                    end: v3(&e["endPoint"]).unwrap(),
                }),
                "circle" => Some(Geom3d::Circle { center: v3(&e["center"]).unwrap(), radius: e["radius"].as_f64().unwrap_or(0.0) }),
                "ellipse" => Some(Geom3d::Ellipse {
                    center: v3(&e["center"]).unwrap(),
                    major: e["majorRadius"].as_f64().unwrap_or(0.0),
                    minor: e["minorRadius"].as_f64().unwrap_or(0.0),
                    direction: v3(&e["direction"]).unwrap_or([1.0, 0.0, 0.0]),
                }),
                _ => None,
            };
            if let Some(geom) = geom {
                ents.push(Entity3d { id, construction, geom });
            }
        }
        out.insert(name, ents);
    }
    out
}

/// A sketch plane frame: sketch (u, v) = ((p - origin)·ex, (p - origin)·ey).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub origin: [f64; 3],
    pub ex: [f64; 3],
    pub ey: [f64; 3],
}

impl Frame {
    pub fn normal(&self) -> [f64; 3] {
        cross(self.ex, self.ey)
    }
    pub fn to_2d(&self, p: [f64; 3]) -> [f64; 2] {
        let d = sub(p, self.origin);
        [dot(d, self.ex), dot(d, self.ey)]
    }
    pub fn to_3d(&self, p: [f64; 2]) -> [f64; 3] {
        add(self.origin, add(scale(self.ex, p[0]), scale(self.ey, p[1])))
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: [f64; 3]) -> [f64; 3] {
    let n = norm(a);
    if n > 0.0 { scale(a, 1.0 / n) } else { a }
}

fn sample_points(ents: &[Entity3d]) -> Vec<[f64; 3]> {
    let mut pts = Vec::new();
    for e in ents {
        match &e.geom {
            Geom3d::Point(p) => pts.push(*p),
            Geom3d::Line(a, b) => pts.extend([*a, *b]),
            Geom3d::Arc { center, start, end, .. } => pts.extend([*center, *start, *end]),
            Geom3d::Circle { center, .. } | Geom3d::Ellipse { center, .. } => pts.push(*center),
        }
    }
    pts
}

/// Unit normal of the best-fit plane through the points (principal
/// directions by power iteration on the covariance).
fn plane_normal(pts: &[[f64; 3]]) -> Option<[f64; 3]> {
    if pts.len() < 3 {
        return None;
    }
    let n = pts.len() as f64;
    let mut c = [0.0; 3];
    for p in pts {
        c = add(c, *p);
    }
    let c = scale(c, 1.0 / n);
    let mut cov = [[0.0; 3]; 3];
    for p in pts {
        let d = sub(*p, c);
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    let mul = |m: &[[f64; 3]; 3], v: [f64; 3]| [dot(m[0], v), dot(m[1], v), dot(m[2], v)];
    let power = |m: &[[f64; 3]; 3], seed: [f64; 3]| {
        let mut v = unit(seed);
        for _ in 0..200 {
            let w = mul(m, v);
            if norm(w) < 1e-300 {
                break;
            }
            v = unit(w);
        }
        v
    };
    let e1 = power(&cov, [1.0, 0.3, 0.2]);
    // Deflate and find the second direction.
    let mut cov2 = cov;
    let l1 = dot(mul(&cov, e1), e1);
    for i in 0..3 {
        for j in 0..3 {
            cov2[i][j] -= l1 * e1[i] * e1[j];
        }
    }
    let mut e2 = power(&cov2, [0.2, 1.0, 0.3]);
    // If the sketch is degenerate (all points on a line) e2 is arbitrary; fine.
    e2 = unit(sub(e2, scale(e1, dot(e2, e1))));
    let nrm = unit(cross(e1, e2));
    if norm(nrm) < 0.5 { None } else { Some(nrm) }
}

fn angle_of(c: [f64; 2], p: [f64; 2]) -> f64 {
    (p[1] - c[1]).atan2(p[0] - c[0])
}

/// Write the 3D geometry into the import's seed through `frame`. The dump's
/// arc direction is about Onshape's sketch normal; `flip_arcs` says the
/// frame's normal is the opposite one. Returns the entity ids present in
/// the import but missing from the dump.
pub fn apply_seed(imp: &mut SketchImport, ents: &[Entity3d], frame: &Frame, flip_arcs: bool) -> Vec<String> {
    let by_id: HashMap<&str, &Entity3d> = ents.iter().map(|e| (e.id.as_str(), e)).collect();
    // The dump's segment startPoint / endPoint follow the curve's parameter
    // direction, which the solver may have flipped; the `<id>.start` /
    // `<id>.end` point entities carry the sketch's own endpoint identity.
    let end_point = |id: &str, suffix: &str, fallback: [f64; 3]| -> [f64; 3] {
        match by_id.get(format!("{id}.{suffix}").as_str()) {
            Some(Entity3d { geom: Geom3d::Point(p), .. }) => *p,
            _ => fallback,
        }
    };
    let mut missing = Vec::new();
    let ids: Vec<(String, usize)> = imp.entities.iter().map(|(k, v)| (k.clone(), *v)).collect();
    for (id, e) in ids {
        let Some(g) = by_id.get(id.as_str()) else {
            missing.push(id);
            continue;
        };
        let sk = &mut imp.sketch;
        match (sk.entities[e], &g.geom) {
            (Entity::Point { x, y }, Geom3d::Point(p)) => {
                let q = frame.to_2d(*p);
                sk.x[x] = q[0];
                sk.x[y] = q[1];
            }
            (Entity::Line { ax, ay, bx, by }, Geom3d::Line(a, b)) => {
                let (a, b) = (end_point(&id, "start", *a), end_point(&id, "end", *b));
                let (p, q) = (frame.to_2d(a), frame.to_2d(b));
                sk.x[ax] = p[0];
                sk.x[ay] = p[1];
                sk.x[bx] = q[0];
                sk.x[by] = q[1];
            }
            (Entity::Arc { cx, cy, r, a0, a1, .. }, Geom3d::Arc { center, radius, start, end, clockwise }) => {
                let (start, end) = (end_point(&id, "start", *start), end_point(&id, "end", *end));
                let ccw = (!*clockwise) != flip_arcs;
                if let Entity::Arc { ccw: c, .. } = &mut sk.entities[e] {
                    *c = ccw;
                }
                let c = frame.to_2d(*center);
                let s0 = angle_of(c, frame.to_2d(start));
                let s1 = angle_of(c, frame.to_2d(end));
                let tau = std::f64::consts::TAU;
                let mut sweep = (s1 - s0) % tau;
                if ccw && sweep <= 0.0 {
                    sweep += tau;
                } else if !ccw && sweep >= 0.0 {
                    sweep -= tau;
                }
                sk.x[cx] = c[0];
                sk.x[cy] = c[1];
                sk.x[r] = *radius;
                sk.x[a0] = s0;
                sk.x[a1] = s0 + sweep;
            }
            (Entity::Circle { cx, cy, r }, Geom3d::Circle { center, radius }) => {
                let c = frame.to_2d(*center);
                sk.x[cx] = c[0];
                sk.x[cy] = c[1];
                sk.x[r] = *radius;
            }
            (Entity::Ellipse { cx, cy, ra, rb, ang }, Geom3d::Ellipse { center, major, minor, direction }) => {
                let c = frame.to_2d(*center);
                let d = [dot(*direction, frame.ex), dot(*direction, frame.ey)];
                sk.x[cx] = c[0];
                sk.x[cy] = c[1];
                sk.x[ra] = *major;
                sk.x[rb] = *minor;
                sk.x[ang] = d[1].atan2(d[0]);
            }
            _ => missing.push(id),
        }
    }
    missing
}

/// Largest residual over constraints that do not touch a fixed (external)
/// entity.
fn internal_residual(imp: &SketchImport) -> f64 {
    let sk = &imp.sketch;
    let touches_fixed = |c: &Constraint| -> bool {
        let ents: Vec<usize> = match *c {
            Constraint::Coincident(a, b) | Constraint::HorizontalPoints(a, b) | Constraint::VerticalPoints(a, b) => vec![a.entity(), b.entity()],
            Constraint::PointOnLine(p, e) | Constraint::PointOnCircle(p, e) | Constraint::PointOnEllipse(p, e) => vec![p.entity(), e],
            Constraint::Collinear(a, b)
            | Constraint::Parallel(a, b)
            | Constraint::Perpendicular(a, b)
            | Constraint::EqualLength(a, b)
            | Constraint::EqualRadius(a, b) => vec![a, b],
            Constraint::HorizontalLine(e) | Constraint::VerticalLine(e) => vec![e],
            Constraint::Length { line, .. } => vec![line],
            Constraint::PointPointDistance { a, b, .. } => vec![a.entity(), b.entity()],
            Constraint::PointLineDistance { point, line, .. } => vec![point.entity(), line],
            Constraint::LineLineDistance { a, b, .. } | Constraint::TangentCircles { a, b, .. } => vec![a, b],
            Constraint::PointCircleDistance { point, circle, .. } => vec![point.entity(), circle],
            Constraint::Midpoint { mid, a, b } => vec![mid.entity(), a.entity(), b.entity()],
            Constraint::TangentLineCircle { line, circle, .. } => vec![line, circle],
            Constraint::Radius { circle, .. } => vec![circle],
            Constraint::EllipseRadius { ellipse, .. } | Constraint::EllipseAxis { ellipse, .. } => vec![ellipse],
            Constraint::Angle { a, b, .. } | Constraint::TangentAt { a, b, .. } => vec![a, b],
            Constraint::ArcMidpoint { point, arc } => vec![point.entity(), arc],
            Constraint::FixVar { var, .. } => return sk.fixed[var],
        };
        ents.iter().any(|&e| sk.entity_vars(e).iter().any(|&v| sk.fixed[v]))
    };
    let mut worst = 0.0f64;
    for (i, c) in sk.constraints.iter().enumerate() {
        if touches_fixed(c) {
            continue;
        }
        for r in sk.constraint_residuals(i) {
            worst = worst.max(r.abs());
        }
    }
    worst
}

fn signed_area(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// The largest-area triangle of line endpoints present in both the seed and
/// the dump: (signed area in the seed frame, the three dump points).
fn handedness_triangle(imp: &SketchImport, by_id: &HashMap<&str, &Entity3d>) -> Option<(f64, [f64; 3], [f64; 3], [f64; 3])> {
    let mut pairs: Vec<([f64; 2], [f64; 3])> = Vec::new();
    for (id, &e) in &imp.entities {
        if let (Entity::Line { .. }, Some(Entity3d { geom: Geom3d::Line(a, b), .. })) = (imp.sketch.entities[e], by_id.get(id.as_str())) {
            pairs.push((imp.sketch.point(PointRef::Start(e)), *a));
            pairs.push((imp.sketch.point(PointRef::End(e)), *b));
        }
    }
    if pairs.len() < 3 {
        return None;
    }
    let mut best: Option<(f64, usize, usize, usize)> = None;
    for i in 0..pairs.len() {
        for j in i + 1..pairs.len() {
            for k in j + 1..pairs.len() {
                let area = signed_area(pairs[i].0, pairs[j].0, pairs[k].0);
                if best.is_none_or(|b| area.abs() > b.0.abs()) {
                    best = Some((area, i, j, k));
                }
            }
        }
    }
    let (area, i, j, k) = best?;
    if area.abs() < 1e-12 {
        return None;
    }
    Some((area, pairs[i].1, pairs[j].1, pairs[k].1))
}

/// Where the seeded geometry puts the external origin point `e`: free it,
/// fix everything else, solve.
fn origin_shift(trial: &mut SketchImport, e: usize) -> [f64; 2] {
    let saved = trial.sketch.fixed.clone();
    for v in trial.sketch.fixed.iter_mut() {
        *v = true;
    }
    for v in trial.sketch.entity_vars(e) {
        trial.sketch.fixed[v] = false;
    }
    let _ = solve(&mut trial.sketch, SolveOptions { max_iterations: 50, ..Default::default() });
    let o = trial.sketch.point(PointRef::Point(e));
    trial.sketch.fixed = saved;
    o
}

#[derive(Clone, Debug)]
pub struct FrameFit {
    pub frame: Frame,
    /// The frame's normal is opposite to Onshape's sketch normal.
    pub arcs_flipped: bool,
    /// Largest residual over internal constraints after seeding.
    pub internal_residual: f64,
    /// Largest residual over all imported constraints after seeding and the
    /// origin fix.
    pub residual: f64,
    pub missing: Vec<String>,
}

/// Fit the sketch plane frame from the 3D dump and seed the import with it.
///
/// The plane comes from the points; the in-plane x direction from a
/// horizontal (or vertical) constrained line, trying both signs and both
/// handednesses; the origin is the world origin projected onto the plane,
/// then corrected so an external origin point (`origin_id`, if the sketch
/// references it) lands at (0, 0).
pub fn fit_frame(imp: &mut SketchImport, ents: &[Entity3d], origin_id: Option<&str>) -> Result<FrameFit, ImportError> {
    let pts = sample_points(ents);
    let n = match plane_normal(&pts) {
        Some(n) => n,
        None if pts.len() == 2 => {
            // A two-point sketch: take the world axis most perpendicular to the segment.
            let d = unit(sub(pts[1], pts[0]));
            let mut axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
            axes.sort_by(|a, b| dot(*a, d).abs().partial_cmp(&dot(*b, d).abs()).unwrap());
            axes[0]
        }
        None => return Err(ImportError(format!("{}: cannot fit a plane to {} points", imp.name, pts.len()))),
    };
    let by_id: HashMap<&str, &Entity3d> = ents.iter().map(|e| (e.id.as_str(), e)).collect();
    let id_of: HashMap<usize, &str> = imp.entities.iter().map(|(k, v)| (*v, k.as_str())).collect();
    // Candidate x directions from horizontal / vertical lines in the dump.
    let mut xs: Vec<[f64; 3]> = Vec::new();
    for c in &imp.sketch.constraints {
        let (e, horizontal) = match *c {
            Constraint::HorizontalLine(e) => (e, true),
            Constraint::VerticalLine(e) => (e, false),
            _ => continue,
        };
        let Some(id) = id_of.get(&e) else { continue };
        let Some(Entity3d { geom: Geom3d::Line(a, b), .. }) = by_id.get(id) else { continue };
        let d = unit(sub(*b, *a));
        let d = unit(sub(d, scale(n, dot(d, n))));
        if norm(d) < 0.5 {
            continue;
        }
        let dx = if horizontal { d } else { cross(n, d) };
        if !xs.iter().any(|x| dot(*x, dx).abs() > 1.0 - 1e-9) {
            xs.push(dx);
        }
        if xs.len() >= 2 {
            break;
        }
    }
    // Always also try the world axes projected into the plane: a horizontal
    // line in the dump can be off-axis when its constraint is to an external
    // reference the import skipped.
    {
        for w in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            let d = unit(sub(w, scale(n, dot(w, n))));
            if norm(d) > 0.5 && !xs.iter().any(|x| dot(*x, d).abs() > 1.0 - 1e-9) {
                xs.push(d);
            }
        }
    }
    let p0 = pts[0];
    let origin0 = scale(n, dot(p0, n));
    let origin_ent = origin_id.and_then(|oid| imp.externals.get(oid).copied());
    // Handedness reference: the largest triangle of line endpoints in the
    // feature-tree seed (Onshape's own sketch frame) and the same points in
    // the dump. A candidate frame whose triangle has the opposite sign is
    // mirrored relative to Onshape's frame, so arc directions flip.
    let tri = handedness_triangle(imp, &by_id);
    // Every candidate is seeded, origin-corrected, and scored on all constraints.
    let mut best: Option<(f64, f64, Frame, bool)> = None;
    for dx in &xs {
        for sx in [1.0, -1.0] {
            for hand in [1.0, -1.0] {
                let ex = scale(*dx, sx);
                let ey = scale(cross(n, ex), hand);
                let mut frame = Frame { origin: origin0, ex, ey };
                let flip = tri.map(|(seed_area, a, b, c)| {
                    let area = signed_area(frame.to_2d(a), frame.to_2d(b), frame.to_2d(c));
                    (area < 0.0) != (seed_area < 0.0)
                }).unwrap_or(false);
                let mut trial = imp.clone();
                apply_seed(&mut trial, ents, &frame, flip);
                trial.reselect_variants();
                let internal = internal_residual(&trial);
                if let Some(e) = origin_ent {
                    frame.origin = frame.to_3d(origin_shift(&mut trial, e));
                    apply_seed(&mut trial, ents, &frame, flip);
                    trial.reselect_variants();
                }
                let all = trial.sketch.worst_residual().0;
                if best.as_ref().is_none_or(|b| all < b.1) {
                    best = Some((internal, all, frame, flip));
                }
            }
        }
    }
    let (internal, _, frame, flip) = best.unwrap();
    let missing = apply_seed(imp, ents, &frame, flip);
    imp.reselect_variants();
    let residual = imp.sketch.worst_residual().0;
    Ok(FrameFit { frame, arcs_flipped: flip, internal_residual: internal, residual, missing })
}
