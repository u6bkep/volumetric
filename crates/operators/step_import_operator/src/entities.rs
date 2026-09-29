//! Typed extraction of the AP214 entities the importer consumes:
//! geometry (points, directions, placements, curves, surfaces), topology
//! (vertices through shells), product structure, and units. Every
//! accessor reports the entity id on failure so import errors point at
//! the offending line of the STEP file.

use crate::p21::{Arg, DataSection, Record};
use brep_core::math::{Frame, Vec3, normalize, scale as vscale};
use brep_core::nurbs::CurveData;

/// Extraction context: the parsed table plus the file's length unit as
/// a factor to metres (applied to every coordinate and measure at
/// extraction time; engine geometry is canonically in metres).
pub struct Ctx<'a> {
    pub data: &'a DataSection,
    pub scale: f64,
    /// Plane-angle unit factor to radians (degree-based files exist).
    pub angle_scale: f64,
}

impl<'a> Ctx<'a> {
    /// The single record of entity `id`, requiring its type name.
    fn simple(&self, id: u64, expect: &[&str]) -> Result<&'a Record, String> {
        let e = self.data.get(id)?;
        let r = e
            .simple()
            .ok_or_else(|| format!("#{id}: expected a simple instance"))?;
        if expect.iter().any(|n| r.name == *n) {
            Ok(r)
        } else {
            Err(format!(
                "#{id}: expected {}, found {}",
                expect.join("/"),
                r.name
            ))
        }
    }

    fn ref_arg(&self, r: &Record, idx: usize, id: u64) -> Result<u64, String> {
        r.args
            .get(idx)
            .and_then(Arg::as_ref_id)
            .ok_or_else(|| format!("#{id} {}: argument {idx} is not a reference", r.name))
    }

    pub fn point(&self, id: u64) -> Result<Vec3, String> {
        let r = self.simple(id, &["CARTESIAN_POINT"])?;
        let c = r
            .args
            .get(1)
            .and_then(Arg::as_list)
            .ok_or_else(|| format!("#{id} CARTESIAN_POINT: missing coordinate list"))?;
        if c.len() != 3 {
            return Err(format!(
                "#{id} CARTESIAN_POINT: expected 3 coordinates, got {} (2D geometry is unsupported)",
                c.len()
            ));
        }
        let mut p = [0.0; 3];
        for (i, a) in c.iter().enumerate() {
            p[i] = a
                .as_f64()
                .ok_or_else(|| format!("#{id}: non-numeric coordinate"))?
                * self.scale;
        }
        Ok(p)
    }

    pub fn direction(&self, id: u64) -> Result<Vec3, String> {
        let r = self.simple(id, &["DIRECTION"])?;
        let c = r
            .args
            .get(1)
            .and_then(Arg::as_list)
            .ok_or_else(|| format!("#{id} DIRECTION: missing ratio list"))?;
        let mut d = [0.0; 3];
        for (i, a) in c.iter().take(3).enumerate() {
            d[i] = a
                .as_f64()
                .ok_or_else(|| format!("#{id}: non-numeric direction"))?;
        }
        Ok(normalize(d))
    }

    /// `VECTOR(name, orientation, magnitude)` as a scaled 3-vector.
    pub fn vector(&self, id: u64) -> Result<Vec3, String> {
        let r = self.simple(id, &["VECTOR"])?;
        let dir = self.direction(self.ref_arg(r, 1, id)?)?;
        let mag = r
            .args
            .get(2)
            .and_then(Arg::as_f64)
            .ok_or_else(|| format!("#{id} VECTOR: bad magnitude"))?;
        Ok(vscale(dir, mag * self.scale))
    }

    /// `AXIS2_PLACEMENT_3D` as an orthonormal right-handed frame:
    /// z = axis (default +z), x = ref_direction orthogonalized against z
    /// (default derived), y completing the frame.
    pub fn axis2(&self, id: u64) -> Result<Frame, String> {
        let r = self.simple(id, &["AXIS2_PLACEMENT_3D"])?;
        let origin = self.point(self.ref_arg(r, 1, id)?)?;
        let axis = match r.args.get(2) {
            Some(Arg::Ref(a)) => self.direction(*a)?,
            _ => [0.0, 0.0, 1.0],
        };
        let x_ref = match r.args.get(3) {
            Some(Arg::Ref(a)) => self.direction(*a)?,
            _ => [1.0, 0.0, 0.0],
        };
        Ok(Frame::from_axis_ref(origin, axis, x_ref))
    }
}

/// A bounded 3D curve in importer form (pre-flattening).
pub enum Curve {
    Line {
        point: Vec3,
        dir: Vec3,
    },
    Circle {
        frame: Frame,
        radius: f64,
    },
    /// `(a cos t, b sin t)` in `frame`: `a` along x, `b` along y (STEP
    /// semi_axis_1/semi_axis_2, in that order — not necessarily
    /// major/minor).
    Ellipse {
        frame: Frame,
        a: f64,
        b: f64,
    },
    Bspline(BsplineCurve),
}

/// B-spline curve with expanded knots, xyzw control points.
pub struct BsplineCurve {
    pub degree: usize,
    pub knots: Vec<f64>,
    pub ctrl: Vec<[f64; 4]>,
}

impl CurveData for BsplineCurve {
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

/// Expand `(multiplicities, values)` into a flat knot vector.
fn expand_knots(mults: &[Arg], knots: &[Arg], id: u64) -> Result<Vec<f64>, String> {
    if mults.len() != knots.len() {
        return Err(format!(
            "#{id}: {} multiplicities for {} knots",
            mults.len(),
            knots.len()
        ));
    }
    let mut out = Vec::new();
    for (m, k) in mults.iter().zip(knots) {
        let m = m
            .as_usize()
            .ok_or_else(|| format!("#{id}: non-integer knot multiplicity"))?;
        let k = k
            .as_f64()
            .ok_or_else(|| format!("#{id}: non-numeric knot"))?;
        for _ in 0..m {
            out.push(k);
        }
    }
    Ok(out)
}

impl<'a> Ctx<'a> {
    pub fn curve(&self, id: u64) -> Result<Curve, String> {
        let e = self.data.get(id)?;
        if let Some(r) = e.simple() {
            return match r.name.as_str() {
                // Wrappers carrying a 3D curve plus pcurves: use the 3D
                // curve (trim UVs come from projection, not pcurves).
                "SURFACE_CURVE" | "SEAM_CURVE" | "INTERSECTION_CURVE" | "BOUNDED_CURVE" => {
                    self.curve(self.ref_arg(r, 1, id)?)
                }
                "LINE" => {
                    let point = self.point(self.ref_arg(r, 1, id)?)?;
                    let dir = self.vector(self.ref_arg(r, 2, id)?)?;
                    Ok(Curve::Line { point, dir })
                }
                "CIRCLE" => {
                    let frame = self.axis2(self.ref_arg(r, 1, id)?)?;
                    let radius = r
                        .args
                        .get(2)
                        .and_then(Arg::as_f64)
                        .ok_or_else(|| format!("#{id} CIRCLE: bad radius"))?
                        * self.scale;
                    Ok(Curve::Circle { frame, radius })
                }
                "ELLIPSE" => {
                    // (name, position, semi_axis_1, semi_axis_2)
                    let frame = self.axis2(self.ref_arg(r, 1, id)?)?;
                    let semi = |idx: usize| -> Result<f64, String> {
                        r.args
                            .get(idx)
                            .and_then(Arg::as_f64)
                            .filter(|v| *v > 0.0)
                            .map(|v| v * self.scale)
                            .ok_or_else(|| format!("#{id} ELLIPSE: bad semi-axis"))
                    };
                    Ok(Curve::Ellipse {
                        frame,
                        a: semi(2)?,
                        b: semi(3)?,
                    })
                }
                "B_SPLINE_CURVE_WITH_KNOTS" => {
                    // (name, degree, ctrl, form, closed, self_int,
                    //  mults, knots, spec)
                    let degree = r
                        .args
                        .get(1)
                        .and_then(Arg::as_usize)
                        .ok_or_else(|| format!("#{id}: bad b-spline degree"))?;
                    let ctrl = self.curve_ctrl(r.args.get(2), id, None)?;
                    let knots = expand_knots(
                        r.args.get(6).and_then(Arg::as_list).unwrap_or(&[]),
                        r.args.get(7).and_then(Arg::as_list).unwrap_or(&[]),
                        id,
                    )?;
                    Ok(Curve::Bspline(BsplineCurve {
                        degree,
                        knots,
                        ctrl,
                    }))
                }
                other => Err(format!("#{id}: unsupported curve type {other}")),
            };
        }
        // Complex instance: rational B-spline. Degree and control points
        // live in B_SPLINE_CURVE, knots in B_SPLINE_CURVE_WITH_KNOTS,
        // weights in RATIONAL_B_SPLINE_CURVE.
        let bc = e
            .record("B_SPLINE_CURVE")
            .ok_or_else(|| format!("#{id}: unsupported complex curve"))?;
        let degree = bc
            .args
            .first()
            .and_then(Arg::as_usize)
            .ok_or_else(|| format!("#{id}: bad b-spline degree"))?;
        let weights = e
            .record("RATIONAL_B_SPLINE_CURVE")
            .and_then(|r| r.args.first())
            .and_then(Arg::as_list);
        let ctrl = self.curve_ctrl(bc.args.get(1), id, weights)?;
        let bk = e
            .record("B_SPLINE_CURVE_WITH_KNOTS")
            .ok_or_else(|| format!("#{id}: complex b-spline curve without knots"))?;
        let knots = expand_knots(
            bk.args.first().and_then(Arg::as_list).unwrap_or(&[]),
            bk.args.get(1).and_then(Arg::as_list).unwrap_or(&[]),
            id,
        )?;
        Ok(Curve::Bspline(BsplineCurve {
            degree,
            knots,
            ctrl,
        }))
    }

    fn curve_ctrl(
        &self,
        arg: Option<&Arg>,
        id: u64,
        weights: Option<&[Arg]>,
    ) -> Result<Vec<[f64; 4]>, String> {
        let refs = arg
            .and_then(Arg::as_list)
            .ok_or_else(|| format!("#{id}: missing control point list"))?;
        let mut ctrl = Vec::with_capacity(refs.len());
        for (i, a) in refs.iter().enumerate() {
            let p = self.point(
                a.as_ref_id()
                    .ok_or_else(|| format!("#{id}: control point {i} is not a reference"))?,
            )?;
            let w = match weights {
                Some(ws) => ws
                    .get(i)
                    .and_then(Arg::as_f64)
                    .ok_or_else(|| format!("#{id}: missing weight {i}"))?,
                None => 1.0,
            };
            ctrl.push([p[0], p[1], p[2], w]);
        }
        Ok(ctrl)
    }

    /// A surface entity in importer form. Extrusion profiles stay as
    /// curves here; lowering happens in `convert`.
    pub fn surface(&self, id: u64) -> Result<StepSurface, String> {
        let e = self.data.get(id)?;
        if let Some(r) = e.simple() {
            let radius = |idx: usize| -> Result<f64, String> {
                r.args
                    .get(idx)
                    .and_then(Arg::as_f64)
                    .map(|v| v * self.scale)
                    .ok_or_else(|| format!("#{id} {}: bad measure", r.name))
            };
            return match r.name.as_str() {
                "PLANE" => Ok(StepSurface::Plane(self.axis2(self.ref_arg(r, 1, id)?)?)),
                "CYLINDRICAL_SURFACE" => Ok(StepSurface::Cylinder(
                    self.axis2(self.ref_arg(r, 1, id)?)?,
                    radius(2)?,
                )),
                "CONICAL_SURFACE" => {
                    let frame = self.axis2(self.ref_arg(r, 1, id)?)?;
                    let rad = radius(2)?;
                    let angle = r
                        .args
                        .get(3)
                        .and_then(Arg::as_f64)
                        .ok_or_else(|| format!("#{id}: bad cone angle"))?
                        * self.angle_scale;
                    Ok(StepSurface::Cone(frame, rad, angle))
                }
                "SPHERICAL_SURFACE" => Ok(StepSurface::Sphere(
                    self.axis2(self.ref_arg(r, 1, id)?)?,
                    radius(2)?,
                )),
                "TOROIDAL_SURFACE" => Ok(StepSurface::Torus(
                    self.axis2(self.ref_arg(r, 1, id)?)?,
                    radius(2)?,
                    radius(3)?,
                )),
                "SURFACE_OF_LINEAR_EXTRUSION" => {
                    let profile = self.curve(self.ref_arg(r, 1, id)?)?;
                    let dir = self.vector(self.ref_arg(r, 2, id)?)?;
                    Ok(StepSurface::Extrusion(Box::new(profile), dir))
                }
                "B_SPLINE_SURFACE_WITH_KNOTS" => {
                    // (name, deg_u, deg_v, ctrl, form, u_closed, v_closed,
                    //  self_int, u_mults, v_mults, u_knots, v_knots, spec)
                    let nurbs = self.bspline_surface(
                        id,
                        r.args.get(1),
                        r.args.get(2),
                        r.args.get(3),
                        r.args.get(8),
                        r.args.get(9),
                        r.args.get(10),
                        r.args.get(11),
                        None,
                    )?;
                    Ok(StepSurface::Nurbs(nurbs))
                }
                other => Err(format!("#{id}: unsupported surface type {other}")),
            };
        }
        // Complex rational B-spline surface.
        let bs = e
            .record("B_SPLINE_SURFACE")
            .ok_or_else(|| format!("#{id}: unsupported complex surface"))?;
        let bk = e
            .record("B_SPLINE_SURFACE_WITH_KNOTS")
            .ok_or_else(|| format!("#{id}: complex b-spline surface without knots"))?;
        let weights = e
            .record("RATIONAL_B_SPLINE_SURFACE")
            .and_then(|r| r.args.first());
        let nurbs = self.bspline_surface(
            id,
            bs.args.first(),
            bs.args.get(1),
            bs.args.get(2),
            bk.args.first(),
            bk.args.get(1),
            bk.args.get(2),
            bk.args.get(3),
            weights,
        )?;
        Ok(StepSurface::Nurbs(nurbs))
    }

    #[allow(clippy::too_many_arguments)]
    fn bspline_surface(
        &self,
        id: u64,
        deg_u: Option<&Arg>,
        deg_v: Option<&Arg>,
        ctrl: Option<&Arg>,
        u_mults: Option<&Arg>,
        v_mults: Option<&Arg>,
        u_knots: Option<&Arg>,
        v_knots: Option<&Arg>,
        weights: Option<&Arg>,
    ) -> Result<brep_core::ir::NurbsSurface, String> {
        let degree_u = deg_u
            .and_then(Arg::as_usize)
            .ok_or_else(|| format!("#{id}: bad u degree"))?;
        let degree_v = deg_v
            .and_then(Arg::as_usize)
            .ok_or_else(|| format!("#{id}: bad v degree"))?;
        let rows = ctrl
            .and_then(Arg::as_list)
            .ok_or_else(|| format!("#{id}: missing control net"))?;
        let nctrl_u = rows.len();
        let mut nctrl_v = 0;
        let mut points = Vec::new();
        let weight_rows = weights.and_then(Arg::as_list);
        for (i, row) in rows.iter().enumerate() {
            let row = row
                .as_list()
                .ok_or_else(|| format!("#{id}: control net row {i} is not a list"))?;
            if i == 0 {
                nctrl_v = row.len();
            } else if row.len() != nctrl_v {
                return Err(format!("#{id}: ragged control net"));
            }
            let wrow = weight_rows.map(|ws| ws.get(i).and_then(Arg::as_list));
            for (j, a) in row.iter().enumerate() {
                let p = self.point(
                    a.as_ref_id()
                        .ok_or_else(|| format!("#{id}: control point is not a reference"))?,
                )?;
                let w = match wrow {
                    None => 1.0,
                    Some(wr) => wr
                        .and_then(|wr| wr.get(j))
                        .and_then(Arg::as_f64)
                        .ok_or_else(|| format!("#{id}: missing weight ({i}, {j})"))?,
                };
                points.push([p[0], p[1], p[2], w]);
            }
        }
        let knots_u = expand_knots(
            u_mults.and_then(Arg::as_list).unwrap_or(&[]),
            u_knots.and_then(Arg::as_list).unwrap_or(&[]),
            id,
        )?;
        let knots_v = expand_knots(
            v_mults.and_then(Arg::as_list).unwrap_or(&[]),
            v_knots.and_then(Arg::as_list).unwrap_or(&[]),
            id,
        )?;
        let s = brep_core::ir::NurbsSurface {
            degree_u,
            degree_v,
            nctrl_u,
            nctrl_v,
            knots_u,
            knots_v,
            ctrl: points,
        };
        s.validate().map_err(|e| format!("#{id}: {e}"))?;
        Ok(s)
    }
}

/// A resolved surface entity, pre-lowering.
pub enum StepSurface {
    Plane(Frame),
    Cylinder(Frame, f64),
    /// frame, reference radius, half angle (radians).
    Cone(Frame, f64, f64),
    Sphere(Frame, f64),
    /// frame, major, minor.
    Torus(Frame, f64, f64),
    /// profile curve, extrusion vector.
    Extrusion(Box<Curve>, Vec3),
    Nurbs(brep_core::ir::NurbsSurface),
}

// -------------------------------------------------------------------
// Topology
// -------------------------------------------------------------------

pub struct StepEdge {
    /// Start/end vertex points, in oriented-edge order.
    pub start: Vec3,
    pub end: Vec3,
    pub curve_id: u64,
    /// EDGE_CURVE.same_sense XOR ORIENTED_EDGE.orientation applied:
    /// true when the curve runs start -> end.
    pub forward: bool,
}

impl<'a> Ctx<'a> {
    pub fn vertex(&self, id: u64) -> Result<Vec3, String> {
        let r = self.simple(id, &["VERTEX_POINT"])?;
        self.point(self.ref_arg(r, 1, id)?)
    }

    /// Resolve an ORIENTED_EDGE into endpoint geometry.
    pub fn oriented_edge(&self, id: u64) -> Result<StepEdge, String> {
        let r = self.simple(id, &["ORIENTED_EDGE"])?;
        let edge_id = self.ref_arg(r, 3, id)?;
        let orientation = r
            .args
            .get(4)
            .and_then(Arg::as_bool)
            .ok_or_else(|| format!("#{id} ORIENTED_EDGE: bad orientation"))?;
        let er = self.simple(edge_id, &["EDGE_CURVE"])?;
        let v1 = self.vertex(self.ref_arg(er, 1, edge_id)?)?;
        let v2 = self.vertex(self.ref_arg(er, 2, edge_id)?)?;
        let curve_id = self.ref_arg(er, 3, edge_id)?;
        let same_sense = er
            .args
            .get(4)
            .and_then(Arg::as_bool)
            .ok_or_else(|| format!("#{edge_id} EDGE_CURVE: bad same_sense"))?;
        // Curve direction relative to traversal: EDGE_CURVE.same_sense
        // relates curve to edge; ORIENTED_EDGE.orientation relates edge
        // to loop traversal.
        let forward = same_sense == orientation;
        let (start, end) = if orientation { (v1, v2) } else { (v2, v1) };
        Ok(StepEdge {
            start,
            end,
            curve_id,
            forward,
        })
    }

    /// The loops of a face bound: each FACE_BOUND -> EDGE_LOOP ->
    /// oriented edges.
    pub fn face_bounds(&self, face_id: u64) -> Result<Vec<Vec<StepEdge>>, String> {
        let r = self.simple(face_id, &["ADVANCED_FACE", "FACE_SURFACE"])?;
        let bounds = r
            .args
            .get(1)
            .and_then(Arg::as_list)
            .ok_or_else(|| format!("#{face_id}: missing bounds list"))?;
        let mut loops = Vec::new();
        for b in bounds {
            let bid = b
                .as_ref_id()
                .ok_or_else(|| format!("#{face_id}: bound is not a reference"))?;
            let br = self.simple(bid, &["FACE_BOUND", "FACE_OUTER_BOUND"])?;
            let loop_id = self.ref_arg(br, 1, bid)?;
            let lr = self.simple(loop_id, &["EDGE_LOOP"])?;
            let edges = lr
                .args
                .get(1)
                .and_then(Arg::as_list)
                .ok_or_else(|| format!("#{loop_id} EDGE_LOOP: missing edge list"))?;
            let mut lp = Vec::with_capacity(edges.len());
            for e in edges {
                let eid = e
                    .as_ref_id()
                    .ok_or_else(|| format!("#{loop_id}: edge is not a reference"))?;
                lp.push(self.oriented_edge(eid)?);
            }
            loops.push(lp);
        }
        Ok(loops)
    }

    /// `(surface_id, loops)` for an ADVANCED_FACE.
    pub fn advanced_face(&self, face_id: u64) -> Result<(u64, Vec<Vec<StepEdge>>), String> {
        let r = self.simple(face_id, &["ADVANCED_FACE", "FACE_SURFACE"])?;
        let surface_id = self.ref_arg(r, 2, face_id)?;
        Ok((surface_id, self.face_bounds(face_id)?))
    }

    /// The sense flags of a face: whether its normal is the surface's
    /// (`ADVANCED_FACE.same_sense`), and per bound whether the loop runs
    /// as listed (`FACE_BOUND.orientation`).
    pub fn face_senses(&self, face_id: u64) -> Result<(bool, Vec<bool>), String> {
        let r = self.simple(face_id, &["ADVANCED_FACE", "FACE_SURFACE"])?;
        let same_sense = r
            .args
            .get(3)
            .and_then(Arg::as_bool)
            .ok_or_else(|| format!("#{face_id}: bad same_sense"))?;
        let bounds = r
            .args
            .get(1)
            .and_then(Arg::as_list)
            .ok_or_else(|| format!("#{face_id}: missing bounds list"))?;
        let mut as_listed = Vec::with_capacity(bounds.len());
        for b in bounds {
            let bid = b
                .as_ref_id()
                .ok_or_else(|| format!("#{face_id}: bound is not a reference"))?;
            let br = self.simple(bid, &["FACE_BOUND", "FACE_OUTER_BOUND"])?;
            as_listed.push(
                br.args
                    .get(2)
                    .and_then(Arg::as_bool)
                    .ok_or_else(|| format!("#{bid}: bad bound orientation"))?,
            );
        }
        Ok((same_sense, as_listed))
    }

    /// Face ids of a MANIFOLD_SOLID_BREP's shell.
    pub fn solid_faces(&self, msb_id: u64) -> Result<Vec<u64>, String> {
        let r = self.simple(msb_id, &["MANIFOLD_SOLID_BREP"])?;
        let shell_id = self.ref_arg(r, 1, msb_id)?;
        self.shell_faces(shell_id)
    }

    /// Face ids across all shells of a SHELL_BASED_SURFACE_MODEL.
    pub fn surface_model_faces(&self, sbsm_id: u64) -> Result<Vec<u64>, String> {
        let r = self.simple(sbsm_id, &["SHELL_BASED_SURFACE_MODEL"])?;
        let shells = r
            .args
            .get(1)
            .and_then(Arg::as_list)
            .ok_or_else(|| format!("#{sbsm_id}: missing shell list"))?;
        let mut faces = Vec::new();
        for s in shells {
            let shell_id = s
                .as_ref_id()
                .ok_or_else(|| format!("#{sbsm_id}: shell is not a reference"))?;
            faces.extend(self.shell_faces(shell_id)?);
        }
        Ok(faces)
    }

    fn shell_faces(&self, shell_id: u64) -> Result<Vec<u64>, String> {
        let sr = self.simple(shell_id, &["CLOSED_SHELL", "OPEN_SHELL"])?;
        let faces = sr
            .args
            .get(1)
            .and_then(Arg::as_list)
            .ok_or_else(|| format!("#{shell_id}: missing face list"))?;
        faces
            .iter()
            .map(|a| {
                a.as_ref_id()
                    .ok_or_else(|| format!("#{shell_id}: face is not a reference"))
            })
            .collect()
    }
}

// -------------------------------------------------------------------
// Tessellated geometry (AP242)
// -------------------------------------------------------------------

/// Vertices (scaled to metres) and triangles of a tessellated face.
pub struct MeshData {
    pub verts: Vec<Vec3>,
    pub tris: Vec<[u32; 3]>,
}

impl<'a> Ctx<'a> {
    /// Item ids of a TESSELLATED_SHELL: `(name, (items), topological_link?)`.
    pub fn tessellated_shell_items(&self, id: u64) -> Result<Vec<u64>, String> {
        let r = self.simple(id, &["TESSELLATED_SHELL"])?;
        let items = r
            .args
            .get(1)
            .and_then(Arg::as_list)
            .ok_or_else(|| format!("#{id} TESSELLATED_SHELL: missing item list"))?;
        items
            .iter()
            .map(|a| {
                a.as_ref_id()
                    .ok_or_else(|| format!("#{id}: shell item is not a reference"))
            })
            .collect()
    }

    /// A TRIANGULATED_FACE or COMPLEX_TRIANGULATED_FACE as an indexed
    /// triangle set. Handles the `pnindex` indirection (triangle indices
    /// name per-face points, which map onto the shared coordinate list)
    /// and lowers strips/fans to plain triangles — winding is irrelevant
    /// downstream (parity is orientation-blind). Degenerate triangles
    /// are dropped.
    pub fn triangulated_face(&self, id: u64) -> Result<MeshData, String> {
        let r = self.simple(id, &["TRIANGULATED_FACE", "COMPLEX_TRIANGULATED_FACE"])?;
        let verts = self.coordinates_list(self.ref_arg(r, 1, id)?)?;
        // tessellated_face attributes: name, coordinates, pnmax,
        // normals, geometric_link (OPTIONAL — some exporters omit the
        // slot entirely), pnindex, then triangles / strips + fans.
        let complex = r.name == "COMPLEX_TRIANGULATED_FACE";
        let with_link = if complex { 8 } else { 7 };
        let pnindex_at = match r.args.len() {
            n if n == with_link => 5,
            n if n == with_link - 1 => 4,
            n => {
                return Err(format!("#{id} {}: unexpected argument count {n}", r.name));
            }
        };
        let pnindex = r
            .args
            .get(pnindex_at)
            .and_then(Arg::as_list)
            .ok_or_else(|| format!("#{id}: missing pnindex list"))?;
        let pnindex: Vec<usize> = pnindex
            .iter()
            .map(|a| {
                a.as_usize()
                    .ok_or_else(|| format!("#{id}: non-integer pnindex entry"))
            })
            .collect::<Result<_, _>>()?;
        // 1-based triangle vertex -> 0-based coordinate index.
        let map = |v: usize| -> Result<u32, String> {
            let k = if pnindex.is_empty() {
                v
            } else {
                *pnindex
                    .get(v.wrapping_sub(1))
                    .ok_or_else(|| format!("#{id}: pnindex reference {v} out of range"))?
            };
            if k == 0 || k > verts.len() {
                return Err(format!("#{id}: coordinate index {k} out of range"));
            }
            Ok((k - 1) as u32)
        };
        let int_at = |a: &Arg| -> Result<usize, String> {
            a.as_usize()
                .ok_or_else(|| format!("#{id}: non-integer vertex index"))
        };

        let mut tris: Vec<[u32; 3]> = Vec::new();
        let mut push = |a: u32, b: u32, c: u32| {
            if a != b && b != c && a != c {
                tris.push([a, b, c]);
            }
        };
        if complex {
            let strips = r
                .args
                .get(pnindex_at + 1)
                .and_then(Arg::as_list)
                .ok_or_else(|| format!("#{id}: missing triangle strips"))?;
            for strip in strips {
                let strip = strip
                    .as_list()
                    .ok_or_else(|| format!("#{id}: strip is not a list"))?;
                let idx: Vec<u32> = strip
                    .iter()
                    .map(|a| map(int_at(a)?))
                    .collect::<Result<_, _>>()?;
                for w in idx.windows(3) {
                    push(w[0], w[1], w[2]);
                }
            }
            let fans = r
                .args
                .get(pnindex_at + 2)
                .and_then(Arg::as_list)
                .ok_or_else(|| format!("#{id}: missing triangle fans"))?;
            for fan in fans {
                let fan = fan
                    .as_list()
                    .ok_or_else(|| format!("#{id}: fan is not a list"))?;
                let idx: Vec<u32> = fan
                    .iter()
                    .map(|a| map(int_at(a)?))
                    .collect::<Result<_, _>>()?;
                if let Some((&center, rest)) = idx.split_first() {
                    for w in rest.windows(2) {
                        push(center, w[0], w[1]);
                    }
                }
            }
        } else {
            let triangles = r
                .args
                .get(pnindex_at + 1)
                .and_then(Arg::as_list)
                .ok_or_else(|| format!("#{id}: missing triangle list"))?;
            for t in triangles {
                let t = t
                    .as_list()
                    .ok_or_else(|| format!("#{id}: triangle is not a list"))?;
                if t.len() != 3 {
                    return Err(format!("#{id}: triangle with {} vertices", t.len()));
                }
                push(
                    map(int_at(&t[0])?)?,
                    map(int_at(&t[1])?)?,
                    map(int_at(&t[2])?)?,
                );
            }
        }
        if tris.is_empty() {
            return Err(format!("#{id}: tessellated face with no triangles"));
        }
        Ok(MeshData { verts, tris })
    }

    /// `COORDINATES_LIST(name, npoints, ((x, y, z), ...))`, scaled to
    /// metres. Points are literal coordinate triples, not references.
    fn coordinates_list(&self, id: u64) -> Result<Vec<Vec3>, String> {
        let r = self.simple(id, &["COORDINATES_LIST"])?;
        let pts = r
            .args
            .get(2)
            .and_then(Arg::as_list)
            .ok_or_else(|| format!("#{id} COORDINATES_LIST: missing point list"))?;
        let mut out = Vec::with_capacity(pts.len());
        for (i, p) in pts.iter().enumerate() {
            let p = p
                .as_list()
                .ok_or_else(|| format!("#{id}: point {i} is not a list"))?;
            if p.len() != 3 {
                return Err(format!("#{id}: point {i} has {} coordinates", p.len()));
            }
            let mut v = [0.0; 3];
            for (k, c) in p.iter().enumerate() {
                v[k] = c
                    .as_f64()
                    .ok_or_else(|| format!("#{id}: non-numeric coordinate in point {i}"))?
                    * self.scale;
            }
            out.push(v);
        }
        Ok(out)
    }
}

// -------------------------------------------------------------------
// Units
// -------------------------------------------------------------------

/// Scan the whole file for the geometric context's length unit and
/// return its factor to metres (the engine's canonical unit). OCCT
/// exports carry exactly one global length unit; when several distinct
/// factors appear the file is rejected rather than silently mis-scaled.
pub fn length_unit_scale(data: &DataSection) -> Result<f64, String> {
    let mut found: Option<f64> = None;
    for (id, e) in &data.entities {
        // Length units are complex instances: (LENGTH_UNIT() NAMED_UNIT(*)
        // SI_UNIT(.MILLI., .METRE.)) or a CONVERSION_BASED_UNIT complex.
        if e.record("LENGTH_UNIT").is_none() {
            continue;
        }
        let factor = if let Some(si) = e.record("SI_UNIT") {
            let prefix = match si.args.first() {
                Some(Arg::Enum(p)) => {
                    si_prefix(p).ok_or_else(|| format!("#{id}: unknown SI prefix .{p}."))?
                }
                _ => 1.0,
            };
            match si.args.last() {
                Some(Arg::Enum(n)) if n == "METRE" => prefix,
                other => return Err(format!("#{id}: unsupported SI length unit {other:?}")),
            }
        } else if let Some(cbu) = e.record("CONVERSION_BASED_UNIT") {
            // (name, measure_with_unit): factor times the inner unit.
            let mwu_id = cbu
                .args
                .get(1)
                .and_then(Arg::as_ref_id)
                .ok_or_else(|| format!("#{id}: conversion unit without measure"))?;
            let mr = data.get(mwu_id)?;
            let rec = mr
                .records
                .iter()
                .find(|r| r.name.contains("MEASURE_WITH_UNIT"))
                .ok_or_else(|| format!("#{mwu_id}: expected MEASURE_WITH_UNIT"))?;
            let value = rec
                .args
                .first()
                .and_then(Arg::as_f64)
                .ok_or_else(|| format!("#{mwu_id}: bad measure value"))?;
            let inner_id = rec
                .args
                .get(1)
                .and_then(Arg::as_ref_id)
                .ok_or_else(|| format!("#{mwu_id}: measure without unit"))?;
            let inner = data.get(inner_id)?;
            let inner_si = inner
                .record("SI_UNIT")
                .ok_or_else(|| format!("#{inner_id}: conversion base is not an SI unit"))?;
            match inner_si.args.last() {
                Some(Arg::Enum(n)) if n == "METRE" => {}
                other => {
                    return Err(format!(
                        "#{inner_id}: conversion base is not a metre unit ({other:?})"
                    ));
                }
            }
            let prefix = match inner_si.args.first() {
                Some(Arg::Enum(p)) => {
                    si_prefix(p).ok_or_else(|| format!("#{inner_id}: unknown SI prefix .{p}."))?
                }
                _ => 1.0,
            };
            value * prefix
        } else {
            continue;
        };
        match found {
            None => found = Some(factor),
            Some(f) if (f - factor).abs() < f * 1e-12 => {}
            Some(f) => {
                return Err(format!(
                    "multiple length units in file ({f} m and {factor} m)"
                ));
            }
        }
    }
    found.ok_or_else(|| "no length unit declared".to_string())
}

fn si_prefix(name: &str) -> Option<f64> {
    Some(match name {
        "EXA" => 1e18,
        "PETA" => 1e15,
        "TERA" => 1e12,
        "GIGA" => 1e9,
        "MEGA" => 1e6,
        "KILO" => 1e3,
        "HECTO" => 1e2,
        "DECA" => 1e1,
        "DECI" => 1e-1,
        "CENTI" => 1e-2,
        "MILLI" => 1e-3,
        "MICRO" => 1e-6,
        "NANO" => 1e-9,
        "PICO" => 1e-12,
        "FEMTO" => 1e-15,
        "ATTO" => 1e-18,
        _ => return None,
    })
}

/// The plane-angle unit's factor to radians: 1.0 for SI radian files
/// (with prefix support), the declared conversion for degree-based
/// exporters, and 1.0 when no plane-angle unit is declared.
pub fn plane_angle_scale(data: &DataSection) -> Result<f64, String> {
    let mut found: Option<f64> = None;
    for (id, e) in &data.entities {
        if e.record("PLANE_ANGLE_UNIT").is_none() {
            continue;
        }
        let factor = if let Some(si) = e.record("SI_UNIT") {
            let prefix = match si.args.first() {
                Some(Arg::Enum(p)) => {
                    si_prefix(p).ok_or_else(|| format!("#{id}: unknown SI prefix .{p}."))?
                }
                _ => 1.0,
            };
            match si.args.last() {
                Some(Arg::Enum(n)) if n == "RADIAN" => prefix,
                other => {
                    return Err(format!("#{id}: unsupported SI angle unit {other:?}"));
                }
            }
        } else if let Some(cbu) = e.record("CONVERSION_BASED_UNIT") {
            let mwu_id = cbu
                .args
                .get(1)
                .and_then(Arg::as_ref_id)
                .ok_or_else(|| format!("#{id}: conversion unit without measure"))?;
            let mr = data.get(mwu_id)?;
            let rec = mr
                .records
                .iter()
                .find(|r| r.name.contains("MEASURE_WITH_UNIT"))
                .ok_or_else(|| format!("#{mwu_id}: expected MEASURE_WITH_UNIT"))?;
            rec.args
                .first()
                .and_then(Arg::as_f64)
                .ok_or_else(|| format!("#{mwu_id}: bad angle measure"))?
        } else {
            continue;
        };
        match found {
            None => found = Some(factor),
            Some(f) if (f - factor).abs() < f.abs() * 1e-12 => {}
            Some(f) => {
                return Err(format!(
                    "multiple plane-angle units in file ({f} rad and {factor} rad)"
                ));
            }
        }
    }
    Ok(found.unwrap_or(1.0))
}
