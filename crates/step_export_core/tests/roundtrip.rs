//! Prisms built by `brep_build_core`, checked three ways against the
//! analytic solid: the topological checker, the sampler on the lowered
//! solid, and the sampler on what the STEP importer reads back from the
//! written file.
//!
//! `STEP_EXPORT_DUMP=<dir>` also writes each file there, for checking in
//! another CAD kernel (see `tools/step_check.py`).

use brep_build_core::prism;
use brep_core::ir::BRepModel;
use brep_core::math::Frame;
use brep_core::payload::{PayloadView, build_payload};
use brep_core::topo::{CheckOptions, Solid, to_ir_model};
use sketch_regions_core::{Curve, Piece, Region, arrange};
use step_export_core::{LengthUnit, StepOptions, write_step};
use step_import_operator::{StepConfig, import};

fn piece(id: &str, curve: Curve) -> Piece {
    Piece {
        id: id.into(),
        curve,
    }
}

fn line(id: &str, a: [f64; 2], b: [f64; 2]) -> Piece {
    piece(id, Curve::Line { a, b })
}

/// The region of the arrangement holding `inside`.
fn region_at(pieces: &[Piece], inside: [f64; 2]) -> Region {
    arrange(pieces, 1e-9)
        .into_iter()
        .find(|r| r.contains(inside))
        .expect("a region at the probe point")
}

fn rectangle(w: f64, h: f64) -> Vec<Piece> {
    vec![
        line("s0", [0.0, 0.0], [w, 0.0]),
        line("s1", [w, 0.0], [w, h]),
        line("s2", [w, h], [0.0, h]),
        line("s3", [0.0, h], [0.0, 0.0]),
    ]
}

/// A 60 x 40 mm plate with 8 mm corner rounds.
fn rounded_plate() -> Vec<Piece> {
    let (w, h, r) = (0.060, 0.040, 0.008);
    let arc = |id: &str, c: [f64; 2], s: [f64; 2], e: [f64; 2]| {
        piece(id, Curve::arc_from_ends(c, s, e, true))
    };
    vec![
        line("b", [r, 0.0], [w - r, 0.0]),
        arc("br", [w - r, r], [w - r, 0.0], [w, r]),
        line("r", [w, r], [w, h - r]),
        arc("tr", [w - r, h - r], [w, h - r], [w - r, h]),
        line("t", [w - r, h], [r, h]),
        arc("tl", [r, h - r], [r, h], [0.0, h - r]),
        line("l", [0.0, h - r], [0.0, r]),
        arc("bl", [r, r], [0.0, r], [r, 0.0]),
    ]
}

fn tilted_frame() -> Frame {
    Frame::from_axis_ref([0.010, -0.020, 0.030], [0.3, -0.2, 0.9], [1.0, 0.4, 0.1])
}

struct Case {
    name: &'static str,
    region: Region,
    frame: Frame,
    h0: f64,
    h1: f64,
}

fn cases() -> Vec<Case> {
    let mut plate = rounded_plate();
    plate.push(piece("hole", Curve::circle([0.020, 0.020], 0.006)));
    plate.extend(rectangle(0.010, 0.012).into_iter().map(|p| match p.curve {
        Curve::Line { a, b } => line(
            &format!("w{}", p.id),
            [a[0] + 0.040, a[1] + 0.014],
            [b[0] + 0.040, b[1] + 0.014],
        ),
        _ => unreachable!(),
    }));

    // A bar with a semicircle on its top side, cut out of it (the arc
    // runs clockwise along the outer loop) or added to it.
    let bar = |ccw: bool| {
        vec![
            line("b", [0.0, 0.0], [0.050, 0.0]),
            line("r", [0.050, 0.0], [0.050, 0.020]),
            line("t1", [0.050, 0.020], [0.035, 0.020]),
            piece(
                "round",
                Curve::arc_from_ends([0.025, 0.020], [0.035, 0.020], [0.015, 0.020], ccw),
            ),
            line("t0", [0.015, 0.020], [0.0, 0.020]),
            line("l", [0.0, 0.020], [0.0, 0.0]),
        ]
    };

    let ring = vec![
        piece("outer", Curve::circle([0.0, 0.0], 0.020)),
        piece("inner", Curve::circle([0.003, 0.0], 0.008)),
    ];

    vec![
        Case {
            name: "box",
            region: region_at(&rectangle(0.040, 0.030), [0.010, 0.010]),
            frame: Frame::IDENTITY,
            h0: 0.0,
            h1: 0.010,
        },
        Case {
            name: "plate",
            region: region_at(&plate, [0.005, 0.020]),
            frame: Frame::IDENTITY,
            h0: -0.004,
            h1: 0.008,
        },
        Case {
            name: "bite",
            region: region_at(&bar(false), [0.005, 0.005]),
            frame: tilted_frame(),
            h0: 0.002,
            h1: 0.015,
        },
        Case {
            name: "bump",
            region: region_at(&bar(true), [0.005, 0.005]),
            frame: tilted_frame(),
            h0: 0.002,
            h1: 0.015,
        },
        Case {
            name: "ring",
            region: region_at(&ring, [0.015, 0.0]),
            frame: tilted_frame(),
            h0: -0.005,
            h1: 0.005,
        },
    ]
}

fn build(case: &Case) -> Solid {
    prism(&case.region, &case.frame, case.h0, case.h1, case.name).expect("prism")
}

/// Trim flattening tolerance of the comparisons. The sampler trusts its
/// trim polylines, so a ray that grazes a curved edge within this
/// distance can be miscounted; kept far below the grid's spacing.
const CHORD_TOL: f64 = 2e-7;

/// Compare the sampler on `model` with the analytic prism over a grid
/// that overhangs it; returns how many points were decided.
fn assert_matches(case: &Case, model: &BRepModel, what: &str) -> usize {
    let payload = build_payload(model).expect("payload");
    let view = PayloadView::new(&payload).expect("view");
    let bb = case.region.bbox();
    let margin = 0.004;
    let n = 26;
    let mut tested = 0;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let f = |a: f64, b: f64, t: usize| a + (b - a) * (t as f64 + 0.37) / n as f64;
                let x = f(bb[0] - margin, bb[2] + margin, i);
                let y = f(bb[1] - margin, bb[3] + margin, j);
                let h = f(case.h0 - margin, case.h1 + margin, k);
                let edge = case
                    .region
                    .boundary_distance([x, y])
                    .min((h - case.h0).abs())
                    .min((h - case.h1).abs());
                if edge < 2e-5 {
                    continue;
                }
                let expect = case.region.contains([x, y]) && h > case.h0 && h < case.h1;
                let p = case.frame.to_world([x, y, h]);
                assert_eq!(
                    view.is_inside(p),
                    expect,
                    "{} ({what}): plane point ({x}, {y}, {h}) misclassified",
                    case.name
                );
                tested += 1;
            }
        }
    }
    tested
}

#[test]
fn prisms_are_sound() {
    for case in cases() {
        let solid = build(&case);
        let defects = solid.defects(&CheckOptions::default());
        assert!(defects.is_empty(), "{}: {defects:#?}", case.name);
    }
}

#[test]
fn defects_are_found() {
    let case = &cases()[1];
    // A face turned inside out.
    let mut solid = build(case);
    solid.faces[0].same_sense = !solid.faces[0].same_sense;
    assert!(!solid.defects(&CheckOptions::default()).is_empty());
    // A loop run the wrong way.
    let mut solid = build(case);
    let cap = solid.faces.len() - 2;
    for lp in &mut solid.faces[cap].loops {
        lp.reverse();
        for c in lp.iter_mut() {
            c.forward = !c.forward;
        }
    }
    let defects = solid.defects(&CheckOptions::default());
    assert!(!defects.is_empty());
    // A missing face leaves edges used once.
    let mut solid = build(case);
    solid.faces.pop();
    assert!(
        solid
            .defects(&CheckOptions::default())
            .iter()
            .any(|d| d.contains("closed solid"))
    );
    // A vertex off its curve.
    let mut solid = build(case);
    solid.vertices[0][2] += 1e-4;
    assert!(
        solid
            .defects(&CheckOptions::default())
            .iter()
            .any(|d| d.contains("off the curve"))
    );
}

#[test]
fn lowered_solid_classifies_like_the_prism() {
    for case in cases() {
        let model = to_ir_model(&[build(&case)], CHORD_TOL).expect("lower");
        let tested = assert_matches(&case, &model, "lowered");
        assert!(tested > 12_000, "{}: only {tested} points", case.name);
    }
}

#[test]
fn written_file_reads_back_as_the_prism() {
    for unit in [LengthUnit::Metre, LengthUnit::Millimetre] {
        for case in cases() {
            let solid = build(&case);
            let opts = StepOptions {
                unit,
                file_name: format!("{}.step", case.name),
                ..StepOptions::default()
            };
            let text = write_step(std::slice::from_ref(&solid), &opts).expect("write");
            if unit == LengthUnit::Millimetre
                && let Ok(dir) = std::env::var("STEP_EXPORT_DUMP")
            {
                std::fs::write(format!("{dir}/{}.step", case.name), &text).expect("dump");
            }
            let cfg = StepConfig {
                chord_tol: CHORD_TOL,
                ..StepConfig::default()
            };
            let model =
                import(&text, &cfg).unwrap_or_else(|e| panic!("{}: import: {e}", case.name));
            assert_eq!(model.solids.len(), 1);
            assert_eq!(model.solids[0].faces.len(), solid.faces.len());
            let tested = assert_matches(&case, &model, "read back");
            assert!(tested > 12_000, "{}: only {tested} points", case.name);
        }
    }
}

#[test]
fn two_solids_make_two_products() {
    let all: Vec<Solid> = cases().iter().take(2).map(build).collect();
    let text = write_step(&all, &StepOptions::default()).expect("write");
    let model = import(&text, &StepConfig::default()).expect("import");
    assert_eq!(model.solids.len(), 2);
    assert_eq!(model.instances.len(), 2);
    assert_eq!(model.instances[0].label, "box");
    assert_eq!(model.instances[1].label, "plate");
}
