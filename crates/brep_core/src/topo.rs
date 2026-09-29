//! Topological boundary representation: solids as faces bounded by loops
//! of shared edges, with exact edge curves.
//!
//! [`crate::ir`] is what a *sampler* needs: a surface and the UV region of
//! it that belongs to the boundary, nothing shared between faces. This
//! module is what a *writer* or a constructing kernel needs: vertices,
//! edges and the faces that share them, so a solid can be checked for
//! closedness and written out as STEP topology. [`Solid::to_ir`] lowers
//! one into the other, which is also how a constructed solid reaches the
//! sampling model and the viewer.
//!
//! # Conventions
//!
//! - An edge runs along its curve from `t0` to `t1` (`t0 < t1`), from
//!   vertex `start` to vertex `end`. A closed edge (a full circle) has
//!   `start == end` and spans one period of its curve.
//! - A face's outward normal (out of the material) is its surface's
//!   normal `dS/du x dS/dv` when `same_sense`, the opposite otherwise.
//! - Every loop runs with the face on its left, seen from outside the
//!   material: outer bounds counter-clockwise, holes clockwise. On a
//!   periodic surface (a full cylinder band between two circles) there
//!   is no outer loop and the rule still holds.
//! - A closed solid uses every edge exactly twice, once each way.
//!
//! [`Solid::defects`] checks all of this geometrically; a solid with no
//! defects is what `step_export_core` writes.

use crate::ir::{self, Surface};
use crate::math::{Frame, Vec3, add, cross, norm, normalize, scale, sub};
use crate::nurbs::{self, CurveData};
use crate::project;
use crate::surface::SurfaceView;
use crate::trim::Region;
use core::f64::consts::TAU;

/// `a < b`, false when either is not a number.
fn increasing(a: f64, b: f64) -> bool {
    matches!(a.partial_cmp(&b), Some(core::cmp::Ordering::Less))
}

/// A (possibly rational) B-spline curve, clamped knot vector, xyzw
/// control points.
#[derive(Clone, Debug)]
pub struct NurbsCurve {
    pub degree: usize,
    /// Length `ctrl.len() + degree + 1`.
    pub knots: Vec<f64>,
    pub ctrl: Vec<[f64; 4]>,
}

impl NurbsCurve {
    pub fn validate(&self) -> Result<(), String> {
        if self.degree == 0 || self.degree > nurbs::MAX_DEGREE {
            return Err(format!("nurbs curve degree {} unsupported", self.degree));
        }
        if self.ctrl.len() <= self.degree {
            return Err(format!(
                "nurbs curve with {} control points for degree {}",
                self.ctrl.len(),
                self.degree
            ));
        }
        if self.knots.len() != self.ctrl.len() + self.degree + 1 {
            return Err(format!(
                "knot vector length {} != {} control points + degree {} + 1",
                self.knots.len(),
                self.ctrl.len(),
                self.degree
            ));
        }
        if self
            .knots
            .windows(2)
            .any(|k| k[1] < k[0] || k[1].is_nan() || k[0].is_nan())
        {
            return Err("knot vector not non-decreasing".into());
        }
        if self
            .ctrl
            .iter()
            .any(|c| c[3] <= 0.0 || c.iter().any(|v| !v.is_finite()))
        {
            return Err("control point non-finite or weight <= 0".into());
        }
        Ok(())
    }
}

impl CurveData for NurbsCurve {
    fn degree(&self) -> usize {
        self.degree
    }
    fn nctrl(&self) -> usize {
        self.ctrl.len()
    }
    fn knot(&self, i: usize) -> f64 {
        self.knots[i]
    }
    fn ctrl(&self, i: usize) -> [f64; 4] {
        self.ctrl[i]
    }
}

/// An exact 3D curve with its natural parameterization.
#[derive(Clone, Debug)]
pub enum Curve3 {
    /// `P(t) = origin + t * dir`, `dir` a unit vector (t is length).
    Line {
        origin: Vec3,
        dir: Vec3,
    },
    /// `P(t) = origin + radius * (cos t * x + sin t * y)`; period 2π.
    Circle {
        frame: Frame,
        radius: f64,
    },
    /// `P(t) = origin + a cos t * x + b sin t * y`; period 2π.
    Ellipse {
        frame: Frame,
        a: f64,
        b: f64,
    },
    Nurbs(NurbsCurve),
}

impl Curve3 {
    /// The line through `a` and `b`, parameterized by length from `a`.
    pub fn line_through(a: Vec3, b: Vec3) -> Curve3 {
        Curve3::Line {
            origin: a,
            dir: normalize(sub(b, a)),
        }
    }

    pub fn eval(&self, t: f64) -> Vec3 {
        match self {
            Curve3::Line { origin, dir } => add(*origin, scale(*dir, t)),
            Curve3::Circle { frame, radius } => {
                frame.to_world([radius * t.cos(), radius * t.sin(), 0.0])
            }
            Curve3::Ellipse { frame, a, b } => frame.to_world([a * t.cos(), b * t.sin(), 0.0]),
            Curve3::Nurbs(c) => nurbs::curve_eval(c, t).0,
        }
    }

    /// Unit tangent in the direction of increasing `t`.
    pub fn tangent(&self, t: f64) -> Vec3 {
        match self {
            Curve3::Line { dir, .. } => *dir,
            Curve3::Circle { frame, .. } => frame.dir_to_world([-t.sin(), t.cos(), 0.0]),
            Curve3::Ellipse { frame, a, b } => {
                normalize(frame.dir_to_world([-a * t.sin(), b * t.cos(), 0.0]))
            }
            Curve3::Nurbs(c) => normalize(nurbs::curve_eval(c, t).1),
        }
    }

    /// Parameter period, 0.0 for an aperiodic curve.
    pub fn period(&self) -> f64 {
        match self {
            Curve3::Circle { .. } | Curve3::Ellipse { .. } => TAU,
            Curve3::Line { .. } | Curve3::Nurbs(_) => 0.0,
        }
    }

    /// Points from `t0` to `t1` (`t0 < t1`) at chordal tolerance `tol`,
    /// ends included.
    pub fn flatten(&self, t0: f64, t1: f64, tol: f64) -> Vec<Vec3> {
        match self {
            Curve3::Line { .. } => vec![self.eval(t0), self.eval(t1)],
            Curve3::Circle { frame, radius } => {
                project::flatten_circle(frame, *radius, t0, t1, tol)
            }
            Curve3::Ellipse { frame, a, b } => project::flatten_ellipse(frame, *a, *b, t0, t1, tol),
            Curve3::Nurbs(c) => project::flatten_bspline(c, t0, t1, tol),
        }
    }
}

/// An edge: a span of a curve between two vertices.
#[derive(Clone, Debug)]
pub struct Edge {
    pub curve: Curve3,
    pub t0: f64,
    pub t1: f64,
    /// Indices into [`Solid::vertices`].
    pub start: usize,
    pub end: usize,
}

/// One use of an edge by a loop; `forward` when the loop runs the edge
/// from its start to its end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coedge {
    pub edge: usize,
    pub forward: bool,
}

#[derive(Clone, Debug)]
pub struct Face {
    pub surface: Surface,
    /// Outward normal agrees with the surface normal.
    pub same_sense: bool,
    pub loops: Vec<Vec<Coedge>>,
    /// Free text carried into the written file (the feature and role that
    /// made the face); empty if unnamed.
    pub name: String,
    pub color: Option<[f32; 3]>,
}

#[derive(Clone, Debug, Default)]
pub struct Solid {
    pub name: String,
    pub vertices: Vec<Vec3>,
    pub edges: Vec<Edge>,
    pub faces: Vec<Face>,
    pub color: Option<[f32; 3]>,
}

/// Tolerances of [`Solid::defects`], metres.
#[derive(Clone, Copy, Debug)]
pub struct CheckOptions {
    /// How far a vertex may sit off its edge's curve, or an edge off its
    /// faces' surfaces.
    pub tol: f64,
    /// Flattening tolerance of the trim loops used by the side tests.
    pub chord_tol: f64,
    /// Offset of the probe points of the side tests; faces narrower than
    /// twice this are reported as wrongly oriented.
    pub probe: f64,
    /// Samples per edge for the on-surface test.
    pub samples: usize,
}

impl Default for CheckOptions {
    fn default() -> Self {
        CheckOptions {
            tol: 1e-6,
            chord_tol: 5e-6,
            probe: 1e-4,
            samples: 9,
        }
    }
}

/// What is wrong with a frame as a right-handed orthonormal basis.
pub fn frame_defect(f: &Frame) -> Option<String> {
    const TOL: f64 = 1e-9;
    let all = [f.origin, f.x, f.y, f.z];
    if all.iter().flatten().any(|v| !v.is_finite()) {
        return Some("frame is not finite".into());
    }
    for (name, a) in [("x", f.x), ("y", f.y), ("z", f.z)] {
        if (norm(a) - 1.0).abs() > TOL {
            return Some(format!("frame {name} axis has length {}", norm(a)));
        }
    }
    // Right-handed and orthogonal at once: y is z cross x.
    let d = norm(sub(cross(f.z, f.x), f.y));
    (d > TOL).then(|| format!("frame y axis is {d:.3e} from z cross x (left-handed or skew)"))
}

fn positive(what: &str, v: f64) -> Option<String> {
    (!(v.is_finite() && v > 0.0)).then(|| format!("{what} {v} is not positive"))
}

/// What is wrong with a curve's own definition.
pub fn curve_defect(c: &Curve3) -> Option<String> {
    match c {
        Curve3::Line { origin, dir } => {
            if origin.iter().chain(dir).any(|v| !v.is_finite()) {
                Some("line is not finite".into())
            } else if (norm(*dir) - 1.0).abs() > 1e-9 {
                Some(format!("line direction has length {}", norm(*dir)))
            } else {
                None
            }
        }
        Curve3::Circle { frame, radius } => {
            frame_defect(frame).or_else(|| positive("circle radius", *radius))
        }
        Curve3::Ellipse { frame, a, b } => frame_defect(frame)
            .or_else(|| positive("ellipse semi-axis", *a))
            .or_else(|| positive("ellipse semi-axis", *b)),
        Curve3::Nurbs(n) => n.validate().err(),
    }
}

/// What is wrong with a surface's own definition.
pub fn surface_defect(s: &Surface) -> Option<String> {
    match s {
        Surface::Plane { frame } => frame_defect(frame),
        Surface::Cylinder { frame, radius } | Surface::Sphere { frame, radius } => {
            frame_defect(frame).or_else(|| positive("radius", *radius))
        }
        Surface::Cone {
            frame,
            radius,
            half_angle,
        } => frame_defect(frame).or_else(|| {
            if !(radius.is_finite() && *radius >= 0.0) {
                Some(format!("cone radius {radius} is negative"))
            } else if !(half_angle.is_finite()
                && half_angle.abs() > 1e-12
                && half_angle.abs() < core::f64::consts::FRAC_PI_2)
            {
                Some(format!(
                    "cone half angle {half_angle} is not within 0..90 degrees either way"
                ))
            } else {
                None
            }
        }),
        Surface::Torus {
            frame,
            major,
            minor,
        } => frame_defect(frame)
            .or_else(|| positive("torus major radius", *major))
            .or_else(|| positive("torus minor radius", *minor)),
        Surface::Nurbs(n) => n.validate().err(),
        Surface::ExtrusionPolyline { .. } | Surface::Mesh(_) => {
            Some("approximate surface in a topological solid".into())
        }
    }
}

/// The surface normal `dS/du x dS/dv` (unit) at `uv`.
pub fn surface_normal(surface: &Surface, uv: [f64; 2]) -> Result<Vec3, String> {
    let (u, v) = (uv[0], uv[1]);
    Ok(match surface {
        Surface::Plane { frame } => frame.z,
        Surface::Cylinder { frame, .. } => frame.dir_to_world([u.cos(), u.sin(), 0.0]),
        Surface::Cone {
            frame, half_angle, ..
        } => frame.dir_to_world([
            half_angle.cos() * u.cos(),
            half_angle.cos() * u.sin(),
            -half_angle.sin(),
        ]),
        Surface::Sphere { frame, .. } | Surface::Torus { frame, .. } => {
            frame.dir_to_world([v.cos() * u.cos(), v.cos() * u.sin(), v.sin()])
        }
        Surface::Nurbs(n) => {
            let (_, du, dv) = nurbs::surface_eval(n, u, v);
            let c = cross(du, dv);
            if norm(c) == 0.0 {
                return Err(format!("nurbs normal undefined at ({u}, {v})"));
            }
            normalize(c)
        }
        Surface::ExtrusionPolyline { .. } | Surface::Mesh(_) => {
            return Err("approximate surface in a topological solid".into());
        }
    })
}

impl Solid {
    /// The 3D points of one coedge in loop direction, ends snapped onto
    /// the vertices.
    fn coedge_points(&self, c: Coedge, chord_tol: f64) -> Result<Vec<Vec3>, String> {
        let e = self
            .edges
            .get(c.edge)
            .ok_or_else(|| format!("coedge names edge {} of {}", c.edge, self.edges.len()))?;
        if !increasing(e.t0, e.t1) {
            return Err(format!("edge {}: empty span {}..{}", c.edge, e.t0, e.t1));
        }
        let mut pts = e.curve.flatten(e.t0, e.t1, chord_tol);
        let n = pts.len();
        pts[0] = self.vertices[e.start];
        pts[n - 1] = self.vertices[e.end];
        if !c.forward {
            pts.reverse();
        }
        Ok(pts)
    }

    /// One loop as a closed 3D polyline (closing point omitted).
    fn loop_points(&self, lp: &[Coedge], chord_tol: f64) -> Result<Vec<Vec3>, String> {
        let mut out: Vec<Vec3> = Vec::new();
        for c in lp {
            let pts = self.coedge_points(*c, chord_tol)?;
            let skip = usize::from(!out.is_empty());
            out.extend(&pts[skip..]);
        }
        out.pop();
        if out.len() < 3 {
            return Err("loop with fewer than 3 points".into());
        }
        Ok(out)
    }

    /// The UV trim loops of a face (see [`ir::Face::trims`]).
    pub fn face_trims(&self, face: &Face, chord_tol: f64) -> Result<Vec<Vec<[f64; 2]>>, String> {
        let proj_tol = chord_tol * 4.0 + 1e-6;
        let mut trims = Vec::with_capacity(face.loops.len());
        for (li, lp) in face.loops.iter().enumerate() {
            let pts = self
                .loop_points(lp, chord_tol)
                .map_err(|e| format!("loop {li}: {e}"))?;
            let mut uv = Vec::with_capacity(pts.len());
            let mut hint = None;
            for p in &pts {
                let q = project::project_point(&face.surface, *p, hint, proj_tol)
                    .map_err(|e| format!("loop {li}: {e}"))?;
                uv.push(q);
                hint = Some(q);
            }
            project::verify_loop(&face.surface, &pts, &uv, proj_tol * 2.0)
                .map_err(|e| format!("loop {li}: {e}"))?;
            trims.push(uv);
        }
        project::settle_periodic(&face.surface, face.same_sense, &mut trims, &[])?;
        Ok(trims)
    }

    /// Lower to the sampling representation.
    pub fn to_ir(&self, chord_tol: f64) -> Result<ir::Solid, String> {
        let mut faces = Vec::with_capacity(self.faces.len());
        for (fi, f) in self.faces.iter().enumerate() {
            let trims = self
                .face_trims(f, chord_tol)
                .map_err(|e| format!("face {fi} ({}): {e}", f.name))?;
            faces.push(ir::Face {
                surface: f.surface.clone(),
                trims,
                color: f.color.or(self.color),
            });
        }
        Ok(ir::Solid { faces })
    }

    /// Everything wrong with the solid as a closed, consistently oriented
    /// boundary; empty when it is sound.
    pub fn defects(&self, opts: &CheckOptions) -> Vec<String> {
        let mut out = Vec::new();

        // Edges: vertices on the curve.
        for (ei, e) in self.edges.iter().enumerate() {
            if e.start >= self.vertices.len() || e.end >= self.vertices.len() {
                out.push(format!("edge {ei}: vertex index out of range"));
                continue;
            }
            if !increasing(e.t0, e.t1) {
                out.push(format!("edge {ei}: empty span {}..{}", e.t0, e.t1));
                continue;
            }
            if let Some(err) = curve_defect(&e.curve) {
                out.push(format!("edge {ei}: {err}"));
                continue;
            }
            let period = e.curve.period();
            if period > 0.0 && e.t1 - e.t0 > period * (1.0 + 1e-12) {
                out.push(format!(
                    "edge {ei}: spans {} of a curve of period {period}",
                    e.t1 - e.t0
                ));
                continue;
            }
            let d0 = norm(sub(e.curve.eval(e.t0), self.vertices[e.start]));
            let d1 = norm(sub(e.curve.eval(e.t1), self.vertices[e.end]));
            if d0 > opts.tol || d1 > opts.tol {
                out.push(format!(
                    "edge {ei}: vertices off the curve by {d0:.3e} / {d1:.3e}"
                ));
            }
        }
        if !out.is_empty() {
            return out;
        }

        // Loops chain and close; every edge is used once each way.
        let mut uses = vec![[0usize; 2]; self.edges.len()];
        for (fi, f) in self.faces.iter().enumerate() {
            if f.loops.is_empty() {
                out.push(format!("face {fi} ({}): no loops", f.name));
            }
            for (li, lp) in f.loops.iter().enumerate() {
                if lp.is_empty() {
                    out.push(format!("face {fi} ({}) loop {li}: empty", f.name));
                    continue;
                }
                let ends = |c: &Coedge| -> Option<(usize, usize)> {
                    let e = self.edges.get(c.edge)?;
                    Some(if c.forward {
                        (e.start, e.end)
                    } else {
                        (e.end, e.start)
                    })
                };
                for (ci, c) in lp.iter().enumerate() {
                    let next = &lp[(ci + 1) % lp.len()];
                    match (ends(c), ends(next)) {
                        (Some((_, b)), Some((a, _))) => {
                            if a != b {
                                out.push(format!(
                                    "face {fi} ({}) loop {li}: coedge {ci} ends at vertex {b}, the next starts at {a}",
                                    f.name
                                ));
                            }
                            uses[c.edge][usize::from(!c.forward)] += 1;
                        }
                        _ => out.push(format!(
                            "face {fi} ({}) loop {li}: edge index out of range",
                            f.name
                        )),
                    }
                }
            }
        }
        // One shell: every face reached from the first across edges.
        let mut group: Vec<usize> = (0..self.faces.len()).collect();
        fn root(group: &mut [usize], mut i: usize) -> usize {
            while group[i] != i {
                group[i] = group[group[i]];
                i = group[i];
            }
            i
        }
        let mut edge_face: Vec<Option<usize>> = vec![None; self.edges.len()];
        for (fi, f) in self.faces.iter().enumerate() {
            for c in f.loops.iter().flatten() {
                let Some(slot) = edge_face.get_mut(c.edge) else {
                    continue;
                };
                match *slot {
                    None => *slot = Some(fi),
                    Some(other) => {
                        let (a, b) = (root(&mut group, fi), root(&mut group, other));
                        group[a] = b;
                    }
                }
            }
        }
        let shells = (0..self.faces.len())
            .filter(|&i| root(&mut group, i) == i)
            .count();
        if shells > 1 {
            out.push(format!(
                "{shells} separate shells: lumps are solids of their own, and voids are not written"
            ));
        }
        for (ei, u) in uses.iter().enumerate() {
            if *u != [1, 1] {
                out.push(format!(
                    "edge {ei}: used {} times forward and {} reversed (a closed solid uses each once)",
                    u[0], u[1]
                ));
            }
        }
        if !out.is_empty() {
            return out;
        }

        // Edges lie on their faces; loops keep the face on their left.
        let mut ir_faces = Vec::with_capacity(self.faces.len());
        for (fi, f) in self.faces.iter().enumerate() {
            let tag = format!("face {fi} ({})", f.name);
            if let Some(err) = surface_defect(&f.surface) {
                out.push(format!("{tag}: {err}"));
                continue;
            }
            let view = SurfaceView::from_ir(&f.surface);
            for lp in &f.loops {
                for c in lp {
                    let e = &self.edges[c.edge];
                    let mut worst = 0.0f64;
                    let samples = opts.samples.max(1);
                    for k in 0..samples {
                        let t = e.t0 + (e.t1 - e.t0) * (k as f64 + 0.5) / samples as f64;
                        worst = worst.max(view.closest(e.curve.eval(t)).1);
                    }
                    if worst > opts.tol {
                        out.push(format!(
                            "{tag}: edge {} is {worst:.3e} off the surface",
                            c.edge
                        ));
                    }
                }
            }
            let trims = match self.face_trims(f, opts.chord_tol) {
                Ok(t) => t,
                Err(err) => {
                    out.push(format!("{tag}: {err}"));
                    continue;
                }
            };
            let (u_period, v_period) = f.surface.periods();
            let mut aabb = [
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ];
            for q in trims.iter().flatten() {
                aabb[0] = aabb[0].min(q[0]);
                aabb[1] = aabb[1].max(q[0]);
                aabb[2] = aabb[2].min(q[1]);
                aabb[3] = aabb[3].max(q[1]);
            }
            let region = Region {
                loops: &trims[..],
                uv_aabb: aabb,
                u_period,
                v_period,
                uv_eps: [0.0, 0.0],
            };
            let proj_tol = opts.probe * 4.0;
            for (li, lp) in f.loops.iter().enumerate() {
                let mut wrong = 0usize;
                for c in lp {
                    let e = &self.edges[c.edge];
                    let t = 0.5 * (e.t0 + e.t1);
                    let p = e.curve.eval(t);
                    let tangent = if c.forward {
                        e.curve.tangent(t)
                    } else {
                        scale(e.curve.tangent(t), -1.0)
                    };
                    let Ok(uv) = project::project_point(&f.surface, p, None, proj_tol) else {
                        wrong += 1;
                        continue;
                    };
                    let Ok(n) = surface_normal(&f.surface, uv) else {
                        wrong += 1;
                        continue;
                    };
                    let outward = if f.same_sense { n } else { scale(n, -1.0) };
                    let left = normalize(cross(outward, tangent));
                    // A probe beyond the end of the surface itself (a
                    // patch trimmed along its own boundary) projects back
                    // onto that boundary from a probe's length away.
                    let side = |s: f64| -> bool {
                        let q = add(p, scale(left, s * opts.probe));
                        let (quv, d) = view.closest(q);
                        d < 0.5 * opts.probe && region.contains(quv[0], quv[1]).inside
                    };
                    if !side(1.0) || side(-1.0) {
                        wrong += 1;
                    }
                }
                if wrong > 0 {
                    out.push(format!(
                        "{tag} loop {li}: {wrong} of {} coedges do not have the face on their left",
                        lp.len()
                    ));
                }
            }
            ir_faces.push((
                fi,
                ir::Face {
                    surface: f.surface.clone(),
                    trims,
                    color: None,
                },
            ));
        }
        if !out.is_empty() {
            return out;
        }

        // Outward normals point out of the material.
        let model = ir::BRepModel {
            solids: vec![ir::Solid {
                faces: ir_faces.iter().map(|(_, f)| f.clone()).collect(),
            }],
            instances: vec![ir::Instance {
                solid: 0,
                local_to_world: crate::math::Affine::IDENTITY,
                label: String::new(),
            }],
        };
        let payload = match crate::payload::build_payload(&model) {
            Ok(p) => p,
            Err(err) => {
                out.push(format!("payload: {err}"));
                return out;
            }
        };
        let view = match crate::payload::PayloadView::new(&payload) {
            Ok(v) => v,
            Err(err) => {
                out.push(format!("payload: {err}"));
                return out;
            }
        };
        for (fi, f) in self.faces.iter().enumerate() {
            // A point just inside the face: off the middle of its first
            // coedge, to the left.
            let Some(c) = f.loops.first().and_then(|lp| lp.first()) else {
                continue;
            };
            let e = &self.edges[c.edge];
            let t = 0.5 * (e.t0 + e.t1);
            let p = e.curve.eval(t);
            let tangent = if c.forward {
                e.curve.tangent(t)
            } else {
                scale(e.curve.tangent(t), -1.0)
            };
            let Ok(uv) = project::project_point(&f.surface, p, None, opts.probe * 4.0) else {
                continue;
            };
            let Ok(n) = surface_normal(&f.surface, uv) else {
                continue;
            };
            let outward = if f.same_sense { n } else { scale(n, -1.0) };
            let left = normalize(cross(outward, tangent));
            let q = add(p, scale(left, opts.probe));
            let sview = SurfaceView::from_ir(&f.surface);
            let (quv, _) = sview.closest(q);
            let on = sview.eval(quv[0], quv[1]);
            let Ok(n) = surface_normal(&f.surface, quv) else {
                continue;
            };
            let outward = if f.same_sense { n } else { scale(n, -1.0) };
            let above = view.is_inside(add(on, scale(outward, opts.probe)));
            let below = view.is_inside(sub(on, scale(outward, opts.probe)));
            if above || !below {
                out.push(format!(
                    "face {fi} ({}): material is not behind the face (outside probe {}, inside probe {})",
                    f.name,
                    if above { "in" } else { "out" },
                    if below { "in" } else { "out" },
                ));
            }
        }
        out
    }

    /// Bounding box of the vertices and flattened edges,
    /// `[min_x, min_y, min_z, max_x, max_y, max_z]`.
    pub fn bbox(&self, chord_tol: f64) -> [f64; 6] {
        let mut bb = [
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        let mut take = |p: Vec3| {
            for c in 0..3 {
                bb[c] = bb[c].min(p[c]);
                bb[c + 3] = bb[c + 3].max(p[c]);
            }
        };
        for v in &self.vertices {
            take(*v);
        }
        for e in &self.edges {
            for p in e.curve.flatten(e.t0, e.t1, chord_tol) {
                take(p);
            }
        }
        bb
    }
}

/// A model of solids as instances at the identity, for the sampler.
pub fn to_ir_model(solids: &[Solid], chord_tol: f64) -> Result<ir::BRepModel, String> {
    let mut model = ir::BRepModel::default();
    for (si, s) in solids.iter().enumerate() {
        let lowered = s
            .to_ir(chord_tol)
            .map_err(|e| format!("solid {si} ({}): {e}", s.name))?;
        model.solids.push(lowered);
        model.instances.push(ir::Instance {
            solid: si,
            local_to_world: crate::math::Affine::IDENTITY,
            label: s.name.clone(),
        });
    }
    Ok(model)
}
