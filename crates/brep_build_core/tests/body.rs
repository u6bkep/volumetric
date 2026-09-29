//! Swept bodies against the height field's prism, column by column: the
//! two are written apart and must agree.

use brep_build_core::body::{Sweep, sweep};
use brep_build_core::planmap::Tolerances;
use heightfield_core::{Grid, Prism};
use sketch_regions_core::{Curve, P2, Piece, Region, arrange};

fn piece(id: &str, curve: Curve) -> Piece {
    Piece {
        id: id.into(),
        curve,
    }
}

fn line(id: &str, a: P2, b: P2) -> Piece {
    piece(id, Curve::Line { a, b })
}

fn region_at(pieces: &[Piece], inside: P2) -> Region {
    arrange(pieces, 1e-9)
        .into_iter()
        .find(|r| r.contains(inside))
        .expect("a region")
}

/// A slot 60 x 20 mm with round ends, about the origin.
fn slot() -> Region {
    region_at(
        &[
            line("bottom", [-0.02, -0.01], [0.02, -0.01]),
            piece(
                "right",
                Curve::arc_from_ends([0.02, 0.0], [0.02, -0.01], [0.02, 0.01], true),
            ),
            line("top", [0.02, 0.01], [-0.02, 0.01]),
            piece(
                "left",
                Curve::arc_from_ends([-0.02, 0.0], [-0.02, 0.01], [-0.02, -0.01], true),
            ),
        ],
        [0.0, 0.0],
    )
}

/// A house: a 40 x 20 mm wall under a round roof of radius 25 mm, with a
/// round window.
fn house() -> Region {
    let rise = 0.025 - (0.025f64 * 0.025 - 0.02 * 0.02).sqrt();
    let _ = rise;
    region_at(
        &[
            line("floor", [-0.02, 0.0], [0.02, 0.0]),
            line("east", [0.02, 0.0], [0.02, 0.02]),
            piece(
                "roof",
                Curve::arc_from_ends([0.0, 0.005], [0.02, 0.02], [-0.02, 0.02], true),
            ),
            line("west", [-0.02, 0.02], [-0.02, 0.0]),
            piece("window", Curve::circle([0.0, 0.012], 0.005)),
        ],
        [0.015, 0.003],
    )
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

fn agree(name: &str, s: &Sweep) {
    let tol = Tolerances::default();
    let map = sweep(s, &tol).unwrap_or_else(|e| panic!("{name}: {e}"));
    let bb = map.bbox();
    let pitch = 1e-3;
    let grid = Grid::covering(bb[0], bb[1], bb[2], bb[3], pitch, 3);
    let prism = Prism {
        region: s.region.clone(),
        origin: s.origin,
        u: s.u,
        v: s.v,
        dir: s.dir,
        w0: s.w0,
        w1: s.w1,
        label: None,
    };
    let body = prism.body(&grid);
    let mut both = 0;
    for k in 0..grid.len() {
        let (i, j) = grid.ij(k);
        let p = [grid.x(i), grid.y(j)];
        // Columns next to a boundary of the plan are either's to decide.
        let near = map
            .cells
            .iter()
            .map(|c| c.region.boundary_distance(p))
            .fold(f64::INFINITY, f64::min);
        if near < 1e-5 {
            continue;
        }
        match (map.column(p), body.lo[k].is_nan()) {
            (Some((bot, top)), false) => {
                both += 1;
                let (lo, hi) = (body.lo[k] as f64, body.hi[k] as f64);
                let ok = |a: f64, b: f64| (a - b).abs() < 1e-6 * (1.0 + b.abs());
                // The field holds infinite sweeps as far as it likes.
                let far = |v: f64| v.abs() > 1.0;
                assert!(
                    (far(lo) && far(bot)) || ok(bot, lo),
                    "{name}: bottom {bot} against {lo} at {p:?}"
                );
                assert!(
                    (far(hi) && far(top)) || ok(top, hi),
                    "{name}: top {top} against {hi} at {p:?}"
                );
            }
            (None, true) => {}
            (ours, _) => {
                panic!(
                    "{name}: at {p:?} the plan map has {ours:?}, the field {} {}",
                    body.lo[k], body.hi[k]
                );
            }
        }
    }
    assert!(both > 200, "{name}: {both} columns compared");
}

#[test]
fn upright() {
    agree(
        "upright",
        &Sweep {
            region: slot(),
            origin: [0.1, 0.2, 0.01],
            u: [0.0, 1.0, 0.0],
            v: [-1.0, 0.0, 0.0],
            dir: [0.0, 0.0, 1.0],
            w0: -0.004,
            w1: 0.02,
            name: "slot".into(),
        },
    );
}

#[test]
fn sideways() {
    // The house on an upright plane, swept level along the plane's
    // normal and then askew in the plan.
    for dir in [[0.0, 1.0, 0.0], unit([0.4, 1.0, 0.0])] {
        agree(
            "sideways",
            &Sweep {
                region: house(),
                origin: [0.0, 0.0, 0.0],
                u: [1.0, 0.0, 0.0],
                v: [0.0, 0.0, 1.0],
                dir,
                w0: 0.0,
                w1: 0.05,
                name: "house".into(),
            },
        );
    }
}

#[test]
fn slanted() {
    // The slot on the ground, swept up at a slant, and the house on an
    // upright plane swept downhill.
    agree(
        "slanted slot",
        &Sweep {
            region: slot(),
            origin: [0.0, 0.0, 0.0],
            u: [1.0, 0.0, 0.0],
            v: [0.0, 1.0, 0.0],
            dir: unit([0.3, 0.5, 1.0]),
            w0: 0.0,
            w1: 0.03,
            name: "slot".into(),
        },
    );
    agree(
        "downhill house",
        &Sweep {
            region: house(),
            origin: [0.0, 0.0, 0.0],
            u: [1.0, 0.0, 0.0],
            v: [0.0, 0.0, 1.0],
            dir: unit([0.0, 1.0, -0.3]),
            w0: -0.01,
            w1: 0.05,
            name: "house".into(),
        },
    );
}

#[test]
fn through_everything() {
    agree(
        "through",
        &Sweep {
            region: slot(),
            origin: [0.0, 0.0, 0.0],
            u: [1.0, 0.0, 0.0],
            v: [0.0, 1.0, 0.0],
            dir: [0.0, 0.0, 1.0],
            w0: f64::NEG_INFINITY,
            w1: 0.01,
            name: "slot".into(),
        },
    );
}
