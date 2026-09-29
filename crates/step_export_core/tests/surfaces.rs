//! Hand-built solids covering the surface and curve kinds the prisms do
//! not reach: torus, cone, B-spline surfaces and B-spline edge curves,
//! rational and not. Each is checked by the topological checker and by
//! the sampler, on the lowered solid and on the written file read back.
//!
//! `STEP_EXPORT_DUMP=<dir>` also writes each file there.

use brep_core::ir::{BRepModel, NurbsSurface, Surface};
use brep_core::math::{Frame, Vec3};
use brep_core::nurbs::surface_eval;
use brep_core::payload::{PayloadView, build_payload};
use brep_core::topo::{CheckOptions, Coedge, Curve3, Edge, Face, NurbsCurve, Solid, to_ir_model};
use core::f64::consts::{FRAC_PI_2, TAU};
use step_export_core::{LengthUnit, StepOptions, write_step};
use step_import_operator::{StepConfig, import};

const CHORD_TOL: f64 = 2e-7;

fn fwd(edge: usize) -> Coedge {
    Coedge {
        edge,
        forward: true,
    }
}

fn rev(edge: usize) -> Coedge {
    Coedge {
        edge,
        forward: false,
    }
}

fn face(name: &str, surface: Surface, same_sense: bool, loops: Vec<Vec<Coedge>>) -> Face {
    Face {
        surface,
        same_sense,
        loops,
        name: name.into(),
        color: None,
    }
}

fn at_height(z: f64) -> Frame {
    Frame {
        origin: [0.0, 0.0, z],
        ..Frame::IDENTITY
    }
}

/// A full circle about the z axis at height `z`, as a closed edge on a
/// vertex of its own (pushed onto `solid`).
fn rim(solid: &mut Solid, z: f64, radius: f64) -> usize {
    solid.vertices.push([radius, 0.0, z]);
    let v = solid.vertices.len() - 1;
    solid.edges.push(Edge {
        curve: Curve3::Circle {
            frame: at_height(z),
            radius,
        },
        t0: 0.0,
        t1: TAU,
        start: v,
        end: v,
    });
    solid.edges.len() - 1
}

/// A disc of radius 20 mm, 12 mm tall, its top rim rounded at 4 mm.
fn puck() -> Solid {
    let (r, h, rf) = (0.020, 0.012, 0.004);
    let mut s = Solid {
        name: "puck".into(),
        color: Some([0.2, 0.6, 0.9]),
        ..Solid::default()
    };
    let c0 = rim(&mut s, 0.0, r);
    let c1 = rim(&mut s, h - rf, r);
    let c2 = rim(&mut s, h, r - rf);
    s.faces = vec![
        face(
            "bottom",
            Surface::Plane {
                frame: Frame::IDENTITY,
            },
            false,
            vec![vec![rev(c0)]],
        ),
        face(
            "wall",
            Surface::Cylinder {
                frame: Frame::IDENTITY,
                radius: r,
            },
            true,
            vec![vec![fwd(c0)], vec![rev(c1)]],
        ),
        face(
            "round",
            Surface::Torus {
                frame: at_height(h - rf),
                major: r - rf,
                minor: rf,
            },
            true,
            vec![vec![fwd(c1)], vec![rev(c2)]],
        ),
        face(
            "top",
            Surface::Plane {
                frame: at_height(h),
            },
            true,
            vec![vec![fwd(c2)]],
        ),
    ];
    s.faces[2].color = Some([0.9, 0.3, 0.1]);
    s
}

fn puck_truth(p: Vec3) -> Option<bool> {
    let (r, h, rf) = (0.020, 0.012, 0.004);
    let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
    // Signed distance of the profile in the (rho, z) half plane.
    let d = if rho > r - rf && p[2] > h - rf {
        ((rho - (r - rf)).powi(2) + (p[2] - (h - rf)).powi(2)).sqrt() - rf
    } else {
        (rho - r).max(p[2] - h)
    }
    .max(-p[2]);
    (d.abs() > 2e-5).then_some(d < 0.0)
}

/// A cone frustum widening upward: radius 10 mm at the base, 15 degrees.
fn frustum() -> Solid {
    let (r0, h, a) = (0.010, 0.015, 15f64.to_radians());
    let mut s = Solid {
        name: "frustum".into(),
        ..Solid::default()
    };
    let c0 = rim(&mut s, 0.0, r0);
    let c1 = rim(&mut s, h, r0 + h * a.tan());
    s.faces = vec![
        face(
            "bottom",
            Surface::Plane {
                frame: Frame::IDENTITY,
            },
            false,
            vec![vec![rev(c0)]],
        ),
        face(
            "wall",
            Surface::Cone {
                frame: Frame::IDENTITY,
                radius: r0,
                half_angle: a,
            },
            true,
            vec![vec![fwd(c0)], vec![rev(c1)]],
        ),
        face(
            "top",
            Surface::Plane {
                frame: at_height(h),
            },
            true,
            vec![vec![fwd(c1)]],
        ),
    ];
    s
}

fn frustum_truth(p: Vec3) -> Option<bool> {
    let (r0, h, a) = (0.010, 0.015, 15f64.to_radians());
    let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
    let d = ((rho - (r0 + p[2] * a.tan())) * a.cos())
        .max(p[2] - h)
        .max(-p[2]);
    (d.abs() > 2e-5).then_some(d < 0.0)
}

const PILLOW: [f64; 2] = [0.040, 0.030];

/// The top of the pillow: a biquadratic patch over the 40 x 30 mm
/// rectangle, 8 mm high at the corners. `w` weights the middles of the
/// two sides along x and the centre.
fn pillow_top(w: f64) -> NurbsSurface {
    let z = [
        [0.008, 0.012, 0.008],
        [0.010, 0.020, 0.011],
        [0.008, 0.013, 0.008],
    ];
    let weight = [[1.0, 1.0, 1.0], [w, w, w], [1.0, 1.0, 1.0]];
    let mut ctrl = Vec::new();
    for i in 0..3 {
        for j in 0..3 {
            ctrl.push([
                PILLOW[0] * i as f64 / 2.0,
                PILLOW[1] * j as f64 / 2.0,
                z[i][j],
                weight[i][j],
            ]);
        }
    }
    NurbsSurface {
        degree_u: 2,
        degree_v: 2,
        nctrl_u: 3,
        nctrl_v: 3,
        knots_u: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        knots_v: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ctrl,
    }
}

/// A box whose top is [`pillow_top`]: the top's four edges are its
/// boundary curves, B-splines lying in the walls.
fn pillow(w: f64, name: &str) -> Solid {
    let top = pillow_top(w);
    let mut s = Solid {
        name: name.into(),
        ..Solid::default()
    };
    // Corners in loop order (counter-clockwise from above), bottom 0..4
    // then top 4..8.
    let corners = [[0usize, 0usize], [2, 0], [2, 2], [0, 2]];
    for c in corners {
        let q = top.ctrl[c[0] * 3 + c[1]];
        s.vertices.push([q[0], q[1], 0.0]);
    }
    for c in corners {
        let q = top.ctrl[c[0] * 3 + c[1]];
        s.vertices.push([q[0], q[1], q[2]]);
    }
    let line = |s: &Solid, a: usize, b: usize| Edge {
        curve: Curve3::line_through(s.vertices[a], s.vertices[b]),
        t0: 0.0,
        t1: brep_core::math::norm(brep_core::math::sub(s.vertices[b], s.vertices[a])),
        start: a,
        end: b,
    };
    // Edges 0..4 bottom, 4..8 vertical, 8..12 top.
    for k in 0..4 {
        let e = line(&s, k, (k + 1) % 4);
        s.edges.push(e);
    }
    for k in 0..4 {
        let e = line(&s, k, k + 4);
        s.edges.push(e);
    }
    for k in 0..4 {
        let (a, b) = (corners[k], corners[(k + 1) % 4]);
        let mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
        let ctrl = [a, mid, b].map(|c| top.ctrl[c[0] * 3 + c[1]]).to_vec();
        s.edges.push(Edge {
            curve: Curve3::Nurbs(NurbsCurve {
                degree: 2,
                knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                ctrl,
            }),
            t0: 0.0,
            t1: 1.0,
            start: 4 + k,
            end: 4 + (k + 1) % 4,
        });
    }
    for k in 0..4 {
        let a = s.vertices[k];
        let b = s.vertices[(k + 1) % 4];
        let along = brep_core::math::normalize(brep_core::math::sub(b, a));
        let out = [along[1], -along[0], 0.0];
        s.faces.push(face(
            &format!("wall {k}"),
            Surface::Plane {
                frame: Frame {
                    origin: a,
                    x: along,
                    y: [0.0, 0.0, 1.0],
                    z: out,
                },
            },
            true,
            vec![vec![fwd(k), fwd(4 + (k + 1) % 4), rev(8 + k), rev(4 + k)]],
        ));
    }
    s.faces.push(face(
        "bottom",
        Surface::Plane {
            frame: Frame::IDENTITY,
        },
        false,
        vec![vec![rev(3), rev(2), rev(1), rev(0)]],
    ));
    s.faces.push(face(
        "top",
        Surface::Nurbs(top),
        true,
        vec![vec![fwd(8), fwd(9), fwd(10), fwd(11)]],
    ));
    s
}

/// The top's height over `(x, y)`: Newton on the patch's plan position.
fn pillow_height(top: &NurbsSurface, x: f64, y: f64) -> f64 {
    let (mut u, mut v) = (x / PILLOW[0], y / PILLOW[1]);
    for _ in 0..30 {
        let (p, du, dv) = surface_eval(top, u, v);
        let (rx, ry) = (x - p[0], y - p[1]);
        let det = du[0] * dv[1] - du[1] * dv[0];
        u = (u + (rx * dv[1] - ry * dv[0]) / det).clamp(0.0, 1.0);
        v = (v + (du[0] * ry - du[1] * rx) / det).clamp(0.0, 1.0);
    }
    surface_eval(top, u, v).0[2]
}

fn pillow_truth(w: f64) -> impl Fn(Vec3) -> Option<bool> {
    let top = pillow_top(w);
    move |p: Vec3| {
        let side = (-p[0])
            .max(p[0] - PILLOW[0])
            .max(-p[1])
            .max(p[1] - PILLOW[1]);
        let d = if side < 0.0 {
            side.max(-p[2]).max(p[2] - pillow_height(&top, p[0], p[1]))
        } else {
            side
        };
        (d.abs() > 2e-5).then_some(d < 0.0)
    }
}

fn assert_matches(
    name: &str,
    what: &str,
    model: &BRepModel,
    bb: [f64; 6],
    truth: &dyn Fn(Vec3) -> Option<bool>,
) {
    let payload = build_payload(model).expect("payload");
    let view = PayloadView::new(&payload).expect("view");
    let n = 26;
    let margin = 0.003;
    let mut tested = 0;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let f = |c: usize, t: usize| {
                    let (a, b) = (bb[c] - margin, bb[c + 3] + margin);
                    a + (b - a) * (t as f64 + 0.37) / n as f64
                };
                let p = [f(0, i), f(1, j), f(2, k)];
                let Some(expect) = truth(p) else { continue };
                assert_eq!(
                    view.is_inside(p),
                    expect,
                    "{name} ({what}): {p:?} misclassified"
                );
                tested += 1;
            }
        }
    }
    assert!(tested > 12_000, "{name}: only {tested} points");
}

fn check(solid: Solid, truth: &dyn Fn(Vec3) -> Option<bool>) {
    let name = solid.name.clone();
    let defects = solid.defects(&CheckOptions::default());
    assert!(defects.is_empty(), "{name}: {defects:#?}");
    let bb = solid.bbox(1e-5);

    let lowered = to_ir_model(std::slice::from_ref(&solid), CHORD_TOL).expect("lower");
    assert_matches(&name, "lowered", &lowered, bb, truth);

    let opts = StepOptions {
        unit: LengthUnit::Millimetre,
        file_name: format!("{name}.step"),
        ..StepOptions::default()
    };
    let text = write_step(std::slice::from_ref(&solid), &opts).expect("write");
    if let Ok(dir) = std::env::var("STEP_EXPORT_DUMP") {
        std::fs::write(format!("{dir}/{name}.step"), &text).expect("dump");
    }
    let cfg = StepConfig {
        chord_tol: CHORD_TOL,
        ..StepConfig::default()
    };
    let back = import(&text, &cfg).unwrap_or_else(|e| panic!("{name}: import: {e}"));
    assert_eq!(back.solids[0].faces.len(), solid.faces.len());
    for (f, g) in solid.faces.iter().zip(&back.solids[0].faces) {
        let expect = f.color.or(solid.color);
        match (expect, g.color) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                assert!(
                    (0..3).all(|c| (a[c] - b[c]).abs() < 1e-6),
                    "{name} {}: colour",
                    f.name
                )
            }
            other => panic!("{name} {}: colour {other:?}", f.name),
        }
    }
    assert_matches(&name, "read back", &back, bb, truth);
}

#[test]
fn torus_round() {
    check(puck(), &puck_truth);
}

#[test]
fn cone_wall() {
    check(frustum(), &frustum_truth);
}

#[test]
fn bspline_top() {
    check(pillow(1.0, "pillow"), &pillow_truth(1.0));
}

#[test]
fn rational_bspline_top() {
    check(pillow(0.7, "pillow_rational"), &pillow_truth(0.7));
}

#[test]
fn quarter_turn_keeps_an_ellipse() {
    // An ellipse with its longer axis along y is written with the frame
    // turned; the curve a reader rebuilds must be the same one.
    let e = Curve3::Ellipse {
        frame: Frame::from_axis_ref([0.001, 0.002, 0.003], [0.2, 0.1, 1.0], [1.0, 0.0, 0.0]),
        a: 0.004,
        b: 0.009,
    };
    let mut s = puck();
    s.edges[0].curve = e.clone();
    let text = write_step(&[s], &StepOptions::default()).expect("write");
    let line = text
        .lines()
        .find(|l| l.contains("=ELLIPSE("))
        .expect("an ellipse");
    assert!(line.ends_with(",0.009,0.004);"), "{line}");
    let _ = FRAC_PI_2;
}

/// A ball of radius 20 mm cut flat at height `z0` (as a fraction of the
/// radius), keeping the part above or below: a cap bounded by one rim
/// that runs around the sphere, with no seam.
fn dome(z0: f64, above: bool, name: &str) -> Solid {
    let r = 0.020;
    let mut s = Solid {
        name: name.into(),
        ..Solid::default()
    };
    let c = rim(&mut s, z0 * r, r * (1.0 - z0 * z0).sqrt());
    s.faces = vec![
        face(
            "flat",
            Surface::Plane {
                frame: at_height(z0 * r),
            },
            !above,
            vec![vec![if above { rev(c) } else { fwd(c) }]],
        ),
        face(
            "cap",
            Surface::Sphere {
                frame: Frame::IDENTITY,
                radius: r,
            },
            true,
            vec![vec![if above { fwd(c) } else { rev(c) }]],
        ),
    ];
    s
}

fn dome_truth(z0: f64, above: bool) -> impl Fn(Vec3) -> Option<bool> {
    move |p: Vec3| {
        let r = 0.020;
        let cut = if above { z0 * r - p[2] } else { p[2] - z0 * r };
        let d = (brep_core::math::norm(p) - r).max(cut);
        (d.abs() > 2e-5).then_some(d < 0.0)
    }
}

#[test]
fn sphere_caps_without_a_seam() {
    for (z0, above) in [(0.4, true), (0.4, false), (-0.4, true), (-0.4, false)] {
        let name = format!(
            "dome_{}_{}",
            if above { "above" } else { "below" },
            if z0 > 0.0 { "high" } else { "low" }
        );
        check(dome(z0, above, &name), &dome_truth(z0, above));
    }
}

/// A cone on its tip, or standing on its base with the tip up.
fn cone(tip_up: bool, name: &str) -> Solid {
    let (h, a) = (0.020, 25f64.to_radians());
    let r = h * a.tan();
    let mut s = Solid {
        name: name.into(),
        ..Solid::default()
    };
    if tip_up {
        let c = rim(&mut s, 0.0, r);
        s.faces = vec![
            face(
                "base",
                Surface::Plane {
                    frame: Frame::IDENTITY,
                },
                false,
                vec![vec![rev(c)]],
            ),
            face(
                "side",
                Surface::Cone {
                    frame: Frame::IDENTITY,
                    radius: r,
                    half_angle: -a,
                },
                true,
                vec![vec![fwd(c)]],
            ),
        ];
    } else {
        let c = rim(&mut s, h, r);
        s.faces = vec![
            face(
                "top",
                Surface::Plane {
                    frame: at_height(h),
                },
                true,
                vec![vec![fwd(c)]],
            ),
            face(
                "side",
                Surface::Cone {
                    frame: Frame::IDENTITY,
                    radius: 0.0,
                    half_angle: a,
                },
                true,
                vec![vec![rev(c)]],
            ),
        ];
    }
    s
}

fn cone_truth(tip_up: bool) -> impl Fn(Vec3) -> Option<bool> {
    move |p: Vec3| {
        let (h, a) = (0.020, 25f64.to_radians());
        let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
        let z = if tip_up { h - p[2] } else { p[2] };
        let d = ((rho - z * a.tan()) * a.cos()).max(z - h);
        (d.abs() > 2e-5).then_some(d < 0.0)
    }
}

#[test]
fn cones_to_their_apex() {
    check(cone(false, "cone_tip_down"), &cone_truth(false));
    check(cone(true, "cone_tip_up"), &cone_truth(true));
}

/// A spool: a disc 8 mm thick whose rim is hollowed by a torus of tube
/// radius 6 mm centred 20 mm out. The waist is a band of the torus's
/// inner half, across the torus's own v = pi.
fn spool() -> Solid {
    let (major, minor, h): (f64, f64, f64) = (0.020, 0.006, 0.004);
    let r = major - f64::sqrt(minor * minor - h * h);
    let mut s = Solid {
        name: "spool".into(),
        ..Solid::default()
    };
    let c0 = rim(&mut s, -h, r);
    let c1 = rim(&mut s, h, r);
    s.faces = vec![
        face(
            "bottom",
            Surface::Plane {
                frame: at_height(-h),
            },
            false,
            vec![vec![rev(c0)]],
        ),
        face(
            "waist",
            Surface::Torus {
                frame: Frame::IDENTITY,
                major,
                minor,
            },
            false,
            vec![vec![fwd(c0)], vec![rev(c1)]],
        ),
        face(
            "top",
            Surface::Plane {
                frame: at_height(h),
            },
            true,
            vec![vec![fwd(c1)]],
        ),
    ];
    s
}

fn spool_truth(p: Vec3) -> Option<bool> {
    let (major, minor, h) = (0.020, 0.006, 0.004);
    let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
    let tube = minor - ((rho - major).powi(2) + p[2] * p[2]).sqrt();
    let d = (rho - major).max(tube).max(p[2].abs() - h);
    (d.abs() > 2e-5).then_some(d < 0.0)
}

#[test]
fn torus_band_across_its_cut() {
    check(spool(), &spool_truth);
}

#[test]
fn what_the_writer_and_the_checker_refuse() {
    // A left-handed frame would be read back mirrored.
    let mut s = puck();
    if let Surface::Cylinder { frame, .. } = &mut s.faces[1].surface {
        frame.y = [0.0, -1.0, 0.0];
    }
    assert!(
        s.defects(&CheckOptions::default())
            .iter()
            .any(|d| d.contains("left-handed"))
    );
    assert!(write_step(&[s], &StepOptions::default()).is_err());

    // Two lumps in one solid.
    let mut s = puck();
    let mut other = puck();
    let (nv, ne) = (s.vertices.len(), s.edges.len());
    for v in &mut other.vertices {
        v[0] += 0.1;
    }
    for e in &mut other.edges {
        e.start += nv;
        e.end += nv;
        if let Curve3::Circle { frame, .. } = &mut e.curve {
            frame.origin[0] += 0.1;
        }
    }
    for f in &mut other.faces {
        match &mut f.surface {
            Surface::Plane { frame }
            | Surface::Cylinder { frame, .. }
            | Surface::Torus { frame, .. } => frame.origin[0] += 0.1,
            _ => unreachable!(),
        }
        for c in f.loops.iter_mut().flatten() {
            c.edge += ne;
        }
    }
    s.vertices.extend(other.vertices);
    s.edges.extend(other.edges);
    s.faces.extend(other.faces);
    assert!(
        s.defects(&CheckOptions::default())
            .iter()
            .any(|d| d.contains("separate shells"))
    );

    // Numbers that are not numbers.
    let mut s = puck();
    s.vertices[0][0] = f64::NAN;
    assert!(write_step(&[s], &StepOptions::default()).is_err());
}

#[test]
fn a_narrowing_cone_is_written_about_the_opposite_axis() {
    let text = write_step(&[cone(true, "cone")], &StepOptions::default()).expect("write");
    let line = text
        .lines()
        .find(|l| l.contains("=CONICAL_SURFACE("))
        .expect("a cone");
    assert!(line.contains(",0.4363323129985824)"), "{line}");
}
