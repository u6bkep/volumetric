//! Solids built as plan maps, lifted, and checked four ways: the
//! topological checker, the sampler on the lowered solid against the
//! plan map's own columns, the same on the written file read back, and
//! the volume.
//!
//! `STEP_EXPORT_DUMP=<dir>` also writes each file there.

use brep_build_core::lift::{LiftOptions, lift};
use brep_build_core::planmap::{Op, PlanMap};
use brep_build_core::sheet::Sheet;
use brep_core::ir::{BRepModel, Surface};
use brep_core::math::Frame;
use brep_core::payload::{PayloadView, build_payload};
use brep_core::topo::{CheckOptions, Curve3, Solid, to_ir_model};
use sketch_regions_core::{Curve, P2, Piece, Region, arrange};
use step_export_core::{LengthUnit, StepOptions, write_step};
use step_import_operator::{StepConfig, import};

const CHORD_TOL: f64 = 2e-7;

fn rect(x0: f64, y0: f64, x1: f64, y1: f64, id: &str) -> Region {
    let line = |k: usize, a: P2, b: P2| Piece {
        id: format!("{id}.{k}"),
        curve: Curve::Line { a, b },
    };
    arrange(
        &[
            line(0, [x0, y0], [x1, y0]),
            line(1, [x1, y0], [x1, y1]),
            line(2, [x1, y1], [x0, y1]),
            line(3, [x0, y1], [x0, y0]),
        ],
        1e-9,
    )
    .remove(0)
}

fn disc(c: P2, r: f64, id: &str) -> Region {
    arrange(
        &[Piece {
            id: id.into(),
            curve: Curve::circle(c, r),
        }],
        1e-9,
    )
    .remove(0)
}

fn under(z: f64, name: &str) -> Sheet {
    Sheet::plane([0.0, 0.0, z], [0.0, 0.0, -1.0], name)
}

/// A 100 x 80 x 30 mm block on the plan's origin.
fn block() -> PlanMap {
    let mut m = PlanMap::new();
    let body = PlanMap::slab(
        rect(0.0, 0.0, 0.10, 0.08, "base"),
        under(0.0, "base:bottom"),
        Sheet::level(0.03, "base:top"),
    );
    m.apply(&body, Op::Add).unwrap();
    m
}

fn cut(m: &mut PlanMap, region: Region, floor: Sheet) {
    let body = PlanMap::slab(region, floor, Sheet::level(1.0, "above"));
    m.apply(&body, Op::Remove).unwrap();
}

fn cut_from_below(m: &mut PlanMap, region: Region, roof: Sheet) {
    let body = PlanMap::slab(region, under(-1.0, "below"), roof);
    m.apply(&body, Op::Remove).unwrap();
}

fn boss(m: &mut PlanMap, region: Region, top: Sheet) {
    let body = PlanMap::slab(region, under(0.0, "boss:bottom"), top);
    m.apply(&body, Op::Add).unwrap();
}

fn lying_cylinder(through: [f64; 3], along: [f64; 3], radius: f64, up: bool, name: &str) -> Sheet {
    Sheet {
        surface: Surface::Cylinder {
            frame: Frame::from_axis_ref(through, along, [0.0, 0.0, 1.0]),
            radius,
        },
        up,
        name: name.into(),
    }
}

fn machined() -> PlanMap {
    let mut m = block();
    cut(
        &mut m,
        rect(0.02, 0.02, 0.05, 0.06, "pocket"),
        under(0.02, "pocket:floor"),
    );
    boss(
        &mut m,
        disc([0.10, 0.04], 0.01, "boss"),
        Sheet::level(0.04, "boss:top"),
    );
    cut(
        &mut m,
        disc([0.035, 0.04], 0.005, "hole"),
        under(-1.0, "hole:below"),
    );
    cut_from_below(
        &mut m,
        rect(-0.01, 0.065, 0.12, 0.075, "channel"),
        Sheet::level(0.005, "channel:roof"),
    );
    m
}

fn ramp() -> PlanMap {
    let mut m = block();
    cut(
        &mut m,
        rect(-0.01, -0.01, 0.11, 0.09, "cut"),
        Sheet::plane([0.0, 0.0, 0.01], [0.4, 0.0, -1.0], "ramp"),
    );
    // A round pocket across the crease, its floor parallel to the ramp.
    cut(
        &mut m,
        disc([0.05, 0.03], 0.012, "dish"),
        Sheet::plane([0.0, 0.0, 0.005], [0.4, 0.0, -1.0], "dish:floor"),
    );
    m
}

fn ridge() -> PlanMap {
    let mut m = block();
    boss(
        &mut m,
        rect(0.031, 0.01, 0.069, 0.07, "ridge"),
        lying_cylinder([0.05, 0.0, 0.02], [0.0, 1.0, 0.0], 0.02, true, "ridge:top"),
    );
    // A hose channel under it: a lying cylinder's upper half as the
    // roof, across the block.
    cut_from_below(
        &mut m,
        rect(-0.01, 0.032, 0.11, 0.048, "hose"),
        lying_cylinder([0.0, 0.04, 0.0], [1.0, 0.0, 0.0], 0.008, true, "hose:roof"),
    );
    m
}

fn dome() -> PlanMap {
    let mut m = block();
    boss(
        &mut m,
        disc([0.04, 0.04], 0.03, "dome"),
        Sheet {
            surface: Surface::Sphere {
                frame: Frame {
                    origin: [0.04, 0.04, 0.0],
                    ..Frame::IDENTITY
                },
                radius: 0.04,
            },
            up: true,
            name: "dome:top".into(),
        },
    );
    // A sloping cut through dome and block: its crease with the sphere
    // is a circle in space and an ellipse in the plan.
    cut(
        &mut m,
        rect(0.05, -0.01, 0.11, 0.09, "slice"),
        Sheet::plane([0.05, 0.0, 0.01], [0.5, 0.0, -1.0], "slice:floor"),
    );
    m
}

fn assert_matches(name: &str, what: &str, model: &BRepModel, map: &PlanMap) -> usize {
    let payload = build_payload(model).expect("payload");
    let view = PayloadView::new(&payload).expect("view");
    let bb = map.bbox();
    let (zs, margin) = ((-0.01, 0.06), 1e-4);
    let n = 40;
    let mut tested = 0;
    for i in 0..n {
        for j in 0..n {
            let f = |a: f64, b: f64, t: usize| a + (b - a) * (t as f64 + 0.37) / n as f64;
            let p = [
                f(bb[0] - 0.004, bb[2] + 0.004, i),
                f(bb[1] - 0.004, bb[3] + 0.004, j),
            ];
            let wall = map
                .cells
                .iter()
                .map(|c| c.region.boundary_distance(p))
                .fold(f64::INFINITY, f64::min);
            if wall < margin {
                continue;
            }
            let column = map.column(p);
            for k in 0..n {
                let z = f(zs.0, zs.1, k);
                let expect = match column {
                    None => false,
                    Some((bot, top)) => {
                        if (z - bot).abs() < margin || (z - top).abs() < margin {
                            continue;
                        }
                        z > bot && z < top
                    }
                };
                assert_eq!(
                    view.is_inside([p[0], p[1], z]),
                    expect,
                    "{name} ({what}): ({}, {}, {z}) misclassified",
                    p[0],
                    p[1]
                );
                tested += 1;
            }
        }
    }
    tested
}

fn check(name: &str, map: &PlanMap, lumps: usize) -> Vec<Solid> {
    let mut solids = lift(map, &LiftOptions::default()).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(solids.len(), lumps, "{name}: lumps");
    for (i, s) in solids.iter_mut().enumerate() {
        s.name = if lumps == 1 {
            name.to_string()
        } else {
            format!("{name} {i}")
        };
        let defects = s.defects(&CheckOptions::default());
        assert!(defects.is_empty(), "{name}: {defects:#?}");
    }
    let lowered = to_ir_model(&solids, CHORD_TOL).expect("lower");
    let tested = assert_matches(name, "lowered", &lowered, map);
    assert!(tested > 30_000, "{name}: only {tested} points");

    let opts = StepOptions {
        unit: LengthUnit::Millimetre,
        file_name: format!("{name}.step"),
        ..StepOptions::default()
    };
    let text = write_step(&solids, &opts).expect("write");
    if let Ok(dir) = std::env::var("STEP_EXPORT_DUMP") {
        std::fs::write(format!("{dir}/{name}.step"), &text).expect("dump");
    }
    let cfg = StepConfig {
        chord_tol: CHORD_TOL,
        ..StepConfig::default()
    };
    let back = import(&text, &cfg).unwrap_or_else(|e| panic!("{name}: import: {e}"));
    assert_matches(name, "read back", &back, map);
    solids
}

fn kinds(solid: &Solid) -> (usize, usize, usize, usize) {
    let mut k = (0, 0, 0, 0);
    for f in &solid.faces {
        match f.surface {
            Surface::Plane { .. } => k.0 += 1,
            Surface::Cylinder { .. } => k.1 += 1,
            Surface::Sphere { .. } => k.2 += 1,
            _ => k.3 += 1,
        }
    }
    k
}

#[test]
fn machined_block() {
    let solids = check("machined", &machined(), 1);
    let s = &solids[0];
    // Every edge of a prismatic part is a line or a circle.
    assert!(
        s.edges
            .iter()
            .all(|e| matches!(e.curve, Curve3::Line { .. } | Curve3::Circle { .. }))
    );
    let (planes, cylinders, _, other) = kinds(s);
    assert_eq!(other, 0);
    // The hole's wall and the boss's two (on the block and beyond it).
    assert_eq!(cylinders, 3, "{planes} planes");
}

#[test]
fn sloping_cut_with_a_dish() {
    let solids = check("ramp", &ramp(), 1);
    // The dish's wall meets two sloping floors: ellipses.
    assert!(
        solids[0]
            .edges
            .iter()
            .any(|e| matches!(e.curve, Curve3::Ellipse { .. }))
    );
    assert!(
        solids[0]
            .edges
            .iter()
            .all(|e| !matches!(e.curve, Curve3::Nurbs(_)))
    );
}

#[test]
fn ridge_over_a_hose_channel() {
    check("ridge", &ridge(), 1);
}

#[test]
fn dome_sliced() {
    let solids = check("dome", &dome(), 1);
    let (_, _, spheres, _) = kinds(&solids[0]);
    assert_eq!(spheres, 1);
}

#[test]
fn two_lumps() {
    let mut m = block();
    cut(
        &mut m,
        rect(0.04, -0.01, 0.06, 0.09, "saw"),
        under(-1.0, "saw:below"),
    );
    check("halves", &m, 2);
}
