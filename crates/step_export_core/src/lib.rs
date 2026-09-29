//! ISO 10303-21 (STEP AP214) writer for [`brep_core::topo`] solids.
//!
//! Each solid becomes one product with a `MANIFOLD_SOLID_BREP`: exact
//! surfaces (plane, cylinder, cone, sphere, torus, B-spline), exact edge
//! curves (line, circle, ellipse, B-spline) and shared topology, so a
//! receiving CAD or CAM system gets a sewn solid rather than loose faces.
//! The entity layout follows what Parasolid-based systems write (a shape
//! representation per product related to the advanced B-rep
//! representation, `FACE_BOUND` loops, no seam edges on periodic faces).
//!
//! The writer does not judge the solid: run
//! [`brep_core::topo::Solid::defects`] first. It only refuses what the
//! file cannot say or a reader would misread: approximate surfaces,
//! indices out of range, numbers that are not finite, and frames that
//! are not right-handed and orthonormal (a placement carries two of the
//! three axes).
//!
//! Geometry is in metres in `brep_core`; [`StepOptions::unit`] picks the
//! unit of the file.

use brep_core::ir::{NurbsSurface, Surface};
use brep_core::math::{Frame, Vec3};
use brep_core::topo::{Curve3, NurbsCurve, Solid, curve_defect, surface_defect};
use std::collections::HashMap;
use std::fmt::Write as _;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LengthUnit {
    Metre,
    Millimetre,
}

impl LengthUnit {
    /// File units per metre.
    fn per_metre(self) -> f64 {
        match self {
            LengthUnit::Metre => 1.0,
            LengthUnit::Millimetre => 1000.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StepOptions {
    pub unit: LengthUnit,
    /// The model's distance accuracy, metres (how far apart two entities
    /// that should coincide may be).
    pub uncertainty: f64,
    /// `FILE_NAME` fields. The time stamp is the caller's so the output
    /// is reproducible.
    pub file_name: String,
    pub time_stamp: String,
    pub author: String,
    pub organization: String,
    pub originating_system: String,
}

impl Default for StepOptions {
    fn default() -> Self {
        StepOptions {
            unit: LengthUnit::Metre,
            uncertainty: 2e-5,
            file_name: String::new(),
            time_stamp: "1970-01-01T00:00:00".into(),
            author: String::new(),
            organization: String::new(),
            originating_system: "volumetric".into(),
        }
    }
}

/// A REAL in the exchange structure: always with a decimal point, the
/// shortest form that reads back to the same value.
fn real(x: f64) -> String {
    let x = if x == 0.0 { 0.0 } else { x };
    let s = format!("{x:?}");
    match s.split_once('e') {
        Some((mantissa, exp)) => {
            if mantissa.contains('.') {
                format!("{mantissa}E{exp}")
            } else {
                format!("{mantissa}.E{exp}")
            }
        }
        None => match s.strip_suffix(".0") {
            Some(whole) => format!("{whole}."),
            None => s,
        },
    }
}

/// A STRING: apostrophes doubled, anything outside printable ASCII
/// replaced (the basic alphabet of the exchange structure).
fn string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        match c {
            '\'' => out.push_str("''"),
            '\\' => out.push_str("\\\\"),
            ' '..='~' => out.push(c),
            _ => out.push('_'),
        }
    }
    out.push('\'');
    out
}

fn refs(ids: &[u64]) -> String {
    let mut out = String::from("(");
    for (i, id) in ids.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "#{id}");
    }
    out.push(')');
    out
}

fn reals(values: &[f64]) -> String {
    let mut out = String::from("(");
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&real(*v));
    }
    out.push(')');
    out
}

fn ints(values: &[usize]) -> String {
    let mut out = String::from("(");
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{v}");
    }
    out.push(')');
    out
}

fn flag(b: bool) -> &'static str {
    if b { ".T." } else { ".F." }
}

/// Distinct knots with their multiplicities.
fn knot_runs(knots: &[f64]) -> (Vec<usize>, Vec<f64>) {
    let mut mults: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    for &k in knots {
        match values.last() {
            Some(&last) if last == k => *mults.last_mut().unwrap() += 1,
            _ => {
                values.push(k);
                mults.push(1);
            }
        }
    }
    (mults, values)
}

struct Writer {
    data: String,
    next: u64,
    /// File units per metre.
    scale: f64,
    /// Directions repeat (axis-aligned frames); share them.
    directions: HashMap<[u64; 3], u64>,
}

impl Writer {
    fn add(&mut self, body: &str) -> u64 {
        let id = self.next;
        self.next += 1;
        // Lines of the exchange structure are conventionally kept short;
        // break at commas.
        let text = format!("#{id}={body};");
        let mut col = 0;
        let mut in_string = false;
        for c in text.chars() {
            self.data.push(c);
            col += 1;
            if c == '\'' {
                in_string = !in_string;
            }
            if c == ',' && !in_string && col >= 72 {
                self.data.push('\n');
                col = 0;
            }
        }
        self.data.push('\n');
        id
    }

    fn point(&mut self, p: Vec3) -> u64 {
        let s = self.scale;
        self.add(&format!(
            "CARTESIAN_POINT('',{})",
            reals(&[p[0] * s, p[1] * s, p[2] * s])
        ))
    }

    fn direction(&mut self, d: Vec3) -> u64 {
        let key = [d[0].to_bits(), d[1].to_bits(), d[2].to_bits()];
        if let Some(&id) = self.directions.get(&key) {
            return id;
        }
        let id = self.add(&format!("DIRECTION('',{})", reals(&d)));
        self.directions.insert(key, id);
        id
    }

    /// A placement carries the axis and the reference direction only;
    /// the reader completes the frame right-handed.
    fn axis2(&mut self, f: &Frame) -> u64 {
        let p = self.point(f.origin);
        let z = self.direction(f.z);
        let x = self.direction(f.x);
        self.add(&format!("AXIS2_PLACEMENT_3D('',#{p},#{z},#{x})"))
    }

    fn length(&self, v: f64) -> String {
        real(v * self.scale)
    }

    fn nurbs_curve(&mut self, c: &NurbsCurve) -> Result<u64, String> {
        c.validate()?;
        let pts: Vec<u64> = c
            .ctrl
            .iter()
            .map(|q| self.point([q[0], q[1], q[2]]))
            .collect();
        let (mults, knots) = knot_runs(&c.knots);
        let rational = c.ctrl.iter().any(|q| q[3] != 1.0);
        let head = format!("{},{},.UNSPECIFIED.,.F.,.F.", c.degree, refs(&pts));
        let tail = format!("{},{},.UNSPECIFIED.", ints(&mults), reals(&knots));
        Ok(if rational {
            let w: Vec<f64> = c.ctrl.iter().map(|q| q[3]).collect();
            self.add(&format!(
                "(BOUNDED_CURVE() B_SPLINE_CURVE({head}) B_SPLINE_CURVE_WITH_KNOTS({tail}) CURVE() GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_CURVE({}) REPRESENTATION_ITEM(''))",
                reals(&w)
            ))
        } else {
            self.add(&format!("B_SPLINE_CURVE_WITH_KNOTS('',{head},{tail})"))
        })
    }

    fn curve(&mut self, c: &Curve3) -> Result<u64, String> {
        Ok(match c {
            Curve3::Line { origin, dir } => {
                let p = self.point(*origin);
                let d = self.direction(*dir);
                let v = self.add(&format!("VECTOR('',#{d},1.)"));
                self.add(&format!("LINE('',#{p},#{v})"))
            }
            Curve3::Circle { frame, radius } => {
                let a = self.axis2(frame);
                let r = self.length(*radius);
                self.add(&format!("CIRCLE('',#{a},{r})"))
            }
            Curve3::Ellipse { frame, a, b } => {
                // The first semi-axis is written as the major one: a
                // quarter turn of the frame about its axis keeps the
                // curve and its direction.
                let (frame, a, b) = if a >= b {
                    (*frame, *a, *b)
                } else {
                    let turned = Frame {
                        origin: frame.origin,
                        x: frame.y,
                        y: [-frame.x[0], -frame.x[1], -frame.x[2]],
                        z: frame.z,
                    };
                    (turned, *b, *a)
                };
                let ax = self.axis2(&frame);
                let (a, b) = (self.length(a), self.length(b));
                self.add(&format!("ELLIPSE('',#{ax},{a},{b})"))
            }
            Curve3::Nurbs(n) => self.nurbs_curve(n)?,
        })
    }

    fn nurbs_surface(&mut self, n: &NurbsSurface) -> Result<u64, String> {
        n.validate()?;
        let mut rows = String::from("(");
        let mut weights = String::from("(");
        let mut rational = false;
        for i in 0..n.nctrl_u {
            if i > 0 {
                rows.push(',');
                weights.push(',');
            }
            let row = &n.ctrl[i * n.nctrl_v..(i + 1) * n.nctrl_v];
            let ids: Vec<u64> = row.iter().map(|q| self.point([q[0], q[1], q[2]])).collect();
            rows.push_str(&refs(&ids));
            let w: Vec<f64> = row.iter().map(|q| q[3]).collect();
            rational |= w.iter().any(|w| *w != 1.0);
            weights.push_str(&reals(&w));
        }
        rows.push(')');
        weights.push(')');
        let (mu, ku) = knot_runs(&n.knots_u);
        let (mv, kv) = knot_runs(&n.knots_v);
        let head = format!(
            "{},{},{rows},.UNSPECIFIED.,.F.,.F.,.F.",
            n.degree_u, n.degree_v
        );
        let tail = format!(
            "{},{},{},{},.UNSPECIFIED.",
            ints(&mu),
            ints(&mv),
            reals(&ku),
            reals(&kv)
        );
        Ok(if rational {
            self.add(&format!(
                "(BOUNDED_SURFACE() B_SPLINE_SURFACE({head}) B_SPLINE_SURFACE_WITH_KNOTS({tail}) GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_SURFACE({weights}) REPRESENTATION_ITEM('') SURFACE())"
            ))
        } else {
            self.add(&format!("B_SPLINE_SURFACE_WITH_KNOTS('',{head},{tail})"))
        })
    }

    fn surface(&mut self, s: &Surface) -> Result<u64, String> {
        Ok(match s {
            Surface::Plane { frame } => {
                let a = self.axis2(frame);
                self.add(&format!("PLANE('',#{a})"))
            }
            Surface::Cylinder { frame, radius } => {
                let a = self.axis2(frame);
                let r = self.length(*radius);
                self.add(&format!("CYLINDRICAL_SURFACE('',#{a},{r})"))
            }
            Surface::Cone {
                frame,
                radius,
                half_angle,
            } => {
                // Readers take the semi-angle as positive. A cone that
                // narrows along its axis is the same surface, with the
                // same normal, about the opposite axis.
                let (frame, half_angle) = if *half_angle < 0.0 {
                    let neg = |v: Vec3| [-v[0], -v[1], -v[2]];
                    (
                        Frame {
                            origin: frame.origin,
                            x: frame.x,
                            y: neg(frame.y),
                            z: neg(frame.z),
                        },
                        -*half_angle,
                    )
                } else {
                    (*frame, *half_angle)
                };
                let (frame, half_angle) = (&frame, &half_angle);
                let a = self.axis2(frame);
                let r = self.length(*radius);
                self.add(&format!(
                    "CONICAL_SURFACE('',#{a},{r},{})",
                    real(*half_angle)
                ))
            }
            Surface::Sphere { frame, radius } => {
                let a = self.axis2(frame);
                let r = self.length(*radius);
                self.add(&format!("SPHERICAL_SURFACE('',#{a},{r})"))
            }
            Surface::Torus {
                frame,
                major,
                minor,
            } => {
                let a = self.axis2(frame);
                let (major, minor) = (self.length(*major), self.length(*minor));
                self.add(&format!("TOROIDAL_SURFACE('',#{a},{major},{minor})"))
            }
            Surface::Nurbs(n) => self.nurbs_surface(n)?,
            Surface::ExtrusionPolyline { .. } => {
                return Err("polyline extrusion surfaces are approximate; not written".into());
            }
            Surface::Mesh(_) => return Err("mesh faces are not written".into()),
        })
    }

    fn colour_style(&mut self, rgb: [f32; 3]) -> u64 {
        // The shortest form of the f32, within 0..1.
        let part = |v: f32| -> String {
            let v = if v.is_finite() {
                v.clamp(0.0, 1.0)
            } else {
                0.0
            };
            real(format!("{v}").parse::<f64>().unwrap_or(0.0))
        };
        let c = self.add(&format!(
            "COLOUR_RGB('',{},{},{})",
            part(rgb[0]),
            part(rgb[1]),
            part(rgb[2])
        ));
        let fc = self.add(&format!("FILL_AREA_STYLE_COLOUR('',#{c})"));
        let fs = self.add(&format!("FILL_AREA_STYLE('',(#{fc}))"));
        let sf = self.add(&format!("SURFACE_STYLE_FILL_AREA(#{fs})"));
        let ss = self.add(&format!("SURFACE_SIDE_STYLE('',(#{sf}))"));
        let su = self.add(&format!("SURFACE_STYLE_USAGE(.BOTH.,#{ss})"));
        self.add(&format!("PRESENTATION_STYLE_ASSIGNMENT((#{su}))"))
    }

    /// The geometric context of one product: units and accuracy.
    fn context(&mut self, name: &str, unit: LengthUnit, uncertainty: f64) -> u64 {
        let length = match unit {
            LengthUnit::Metre => self.add("(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.))"),
            LengthUnit::Millimetre => {
                self.add("(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.))")
            }
        };
        let angle = self.add("(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.))");
        let solid = self.add("(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT())");
        let acc = self.add(&format!(
            "UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE({}),#{length},'DISTANCE_ACCURACY_VALUE','Maximum Tolerance applied to model')",
            real(uncertainty * self.scale)
        ));
        self.add(&format!(
            "(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{acc})) GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length},#{angle},#{solid})) REPRESENTATION_CONTEXT({},'TOP_LEVEL_ASSEMBLY_PART'))",
            string(name)
        ))
    }

    fn solid(&mut self, s: &Solid, name: &str, opts: &StepOptions) -> Result<(), String> {
        let name = name.to_string();
        let mut vertices = Vec::with_capacity(s.vertices.len());
        for v in &s.vertices {
            let p = self.point(*v);
            vertices.push(self.add(&format!("VERTEX_POINT('',#{p})")));
        }
        let mut edges = Vec::with_capacity(s.edges.len());
        for (ei, e) in s.edges.iter().enumerate() {
            let (Some(v0), Some(v1)) = (vertices.get(e.start), vertices.get(e.end)) else {
                return Err(format!("edge {ei}: vertex index out of range"));
            };
            let (v0, v1) = (*v0, *v1);
            if let Some(err) = curve_defect(&e.curve) {
                return Err(format!("edge {ei}: {err}"));
            }
            let c = self
                .curve(&e.curve)
                .map_err(|err| format!("edge {ei}: {err}"))?;
            edges.push(self.add(&format!("EDGE_CURVE('',#{v0},#{v1},#{c},.T.)")));
        }
        let mut faces = Vec::with_capacity(s.faces.len());
        let mut styled: Vec<u64> = Vec::new();
        for (fi, f) in s.faces.iter().enumerate() {
            if let Some(err) = surface_defect(&f.surface) {
                return Err(format!("face {fi} ({}): {err}", f.name));
            }
            let surface = self
                .surface(&f.surface)
                .map_err(|err| format!("face {fi} ({}): {err}", f.name))?;
            let mut bounds = Vec::with_capacity(f.loops.len());
            for lp in &f.loops {
                let mut oriented = Vec::with_capacity(lp.len());
                for c in lp {
                    let e = edges.get(c.edge).ok_or_else(|| {
                        format!("face {fi} ({}): edge index out of range", f.name)
                    })?;
                    oriented
                        .push(self.add(&format!("ORIENTED_EDGE('',*,*,#{e},{})", flag(c.forward))));
                }
                let el = self.add(&format!("EDGE_LOOP('',{})", refs(&oriented)));
                bounds.push(self.add(&format!("FACE_BOUND('',#{el},.T.)")));
            }
            let face = self.add(&format!(
                "ADVANCED_FACE({},{},#{surface},{})",
                string(&f.name),
                refs(&bounds),
                flag(f.same_sense)
            ));
            if let Some(rgb) = f.color {
                let style = self.colour_style(rgb);
                styled.push(self.add(&format!("STYLED_ITEM('',(#{style}),#{face})")));
            }
            faces.push(face);
        }
        let shell = self.add(&format!("CLOSED_SHELL('',{})", refs(&faces)));
        let msb = self.add(&format!("MANIFOLD_SOLID_BREP({},#{shell})", string(&name)));
        if let Some(rgb) = s.color {
            let style = self.colour_style(rgb);
            styled.push(self.add(&format!("STYLED_ITEM('',(#{style}),#{msb})")));
        }

        let ctx = self.context(&name, opts.unit, opts.uncertainty);
        let brep = self.add(&format!(
            "ADVANCED_BREP_SHAPE_REPRESENTATION('',(#{msb}),#{ctx})"
        ));
        if !styled.is_empty() {
            self.add(&format!(
                "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',{},#{ctx})",
                refs(&styled)
            ));
        }
        let placement = self.axis2(&Frame::IDENTITY);
        let shape = self.add(&format!(
            "SHAPE_REPRESENTATION({},(#{placement}),#{ctx})",
            string(&name)
        ));
        self.add(&format!(
            "SHAPE_REPRESENTATION_RELATIONSHIP('','',#{shape},#{brep})"
        ));

        let app =
            self.add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')");
        self.add(&format!(
            "APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2010,#{app})"
        ));
        let pc = self.add(&format!("PRODUCT_CONTEXT('',#{app},'mechanical')"));
        let n = string(&name);
        let product = self.add(&format!("PRODUCT({n},{n},{n},(#{pc}))"));
        self.add(&format!(
            "PRODUCT_RELATED_PRODUCT_CATEGORY('part','',(#{product}))"
        ));
        let formation = self.add(&format!(
            "PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE('','',#{product},.NOT_KNOWN.)"
        ));
        let pdc = self.add(&format!("PRODUCT_DEFINITION_CONTEXT('',#{app},'design')"));
        let pd = self.add(&format!("PRODUCT_DEFINITION('','',#{formation},#{pdc})"));
        let pds = self.add(&format!("PRODUCT_DEFINITION_SHAPE('','',#{pd})"));
        self.add(&format!("SHAPE_DEFINITION_REPRESENTATION(#{pds},#{shape})"));
        Ok(())
    }
}

/// Write `solids` as one STEP file, a product per solid.
pub fn write_step(solids: &[Solid], opts: &StepOptions) -> Result<String, String> {
    if solids.is_empty() {
        return Err("no solids to write".into());
    }
    if !(opts.uncertainty > 0.0 && opts.uncertainty.is_finite()) {
        return Err(format!("uncertainty must be > 0, got {}", opts.uncertainty));
    }
    let mut w = Writer {
        data: String::new(),
        next: 10,
        scale: opts.unit.per_metre(),
        directions: HashMap::new(),
    };
    // Product ids are the names: unnamed solids are numbered and a
    // repeated name gets its position appended.
    let mut names: Vec<String> = Vec::with_capacity(solids.len());
    for (i, s) in solids.iter().enumerate() {
        let mut name = if s.name.is_empty() {
            format!("Part {}", i + 1)
        } else {
            s.name.clone()
        };
        if names.contains(&name) {
            name = format!("{name} ({})", i + 1);
        }
        names.push(name);
    }
    for (i, s) in solids.iter().enumerate() {
        if s.vertices.iter().flatten().any(|v| !v.is_finite()) {
            return Err(format!("solid {i} ({}): vertex is not finite", s.name));
        }
        w.solid(s, &names[i], opts)
            .map_err(|e| format!("solid {i} ({}): {e}", s.name))?;
    }

    let mut out = String::with_capacity(w.data.len() + 1024);
    out.push_str("ISO-10303-21;\nHEADER;\n");
    out.push_str("FILE_DESCRIPTION(('STEP AP214'),'2;1');\n");
    let _ = writeln!(
        out,
        "FILE_NAME({},{},({}),({}),{},{},'');",
        string(&opts.file_name),
        string(&opts.time_stamp),
        string(&opts.author),
        string(&opts.organization),
        string(concat!("step_export_core ", env!("CARGO_PKG_VERSION"))),
        string(&opts.originating_system),
    );
    out.push_str("FILE_SCHEMA(('AUTOMOTIVE_DESIGN {1 0 10303 214 3 1 1}'));\n");
    out.push_str("ENDSEC;\nDATA;\n");
    out.push_str(&w.data);
    out.push_str("ENDSEC;\nEND-ISO-10303-21;\n");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reals_always_carry_a_point() {
        assert_eq!(real(0.0), "0.");
        assert_eq!(real(-0.0), "0.");
        assert_eq!(real(1.0), "1.");
        assert_eq!(real(-12.0), "-12.");
        assert_eq!(real(0.0127), "0.0127");
        assert_eq!(real(1e-8), "1.E-8");
        assert_eq!(real(2.5e-7), "2.5E-7");
        assert_eq!(real(1e21), "1.E21");
    }

    #[test]
    fn strings_are_escaped() {
        assert_eq!(string("it's"), "'it''s'");
        assert_eq!(string("é"), "'_'");
    }

    #[test]
    fn knots_compress_into_runs() {
        let (m, k) = knot_runs(&[0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]);
        assert_eq!(m, vec![3, 1, 3]);
        assert_eq!(k, vec![0.0, 0.5, 1.0]);
    }
}
