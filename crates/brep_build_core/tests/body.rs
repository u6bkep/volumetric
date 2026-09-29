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

use brep_build_core::body::{FaceSpec, envelope};
use brep_build_core::fit::fit_surface;
use brep_core::ir::NurbsSurface;

fn square(half: f64, id: &str) -> Region {
    region_at(
        &[
            line(&format!("{id}0"), [-half, -half], [half, -half]),
            line(&format!("{id}1"), [half, -half], [half, half]),
            line(&format!("{id}2"), [half, half], [-half, half]),
            line(&format!("{id}3"), [-half, half], [-half, -half]),
        ],
        [0.0, 0.0],
    )
}

/// The flat patch with the given corners: `(u, v)` from `a` along to `b`
/// and up to `d`.
fn bilinear(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> NurbsSurface {
    let w = |p: [f64; 3]| [p[0], p[1], p[2], 1.0];
    NurbsSurface {
        degree_u: 1,
        degree_v: 1,
        nctrl_u: 2,
        nctrl_v: 2,
        knots_u: vec![0.0, 0.0, 1.0, 1.0],
        knots_v: vec![0.0, 0.0, 1.0, 1.0],
        ctrl: vec![w(a), w(d), w(b), w(c)],
    }
}

#[test]
fn a_frustum_from_its_faces() {
    // A 40 mm square below, a 20 mm square 20 mm above it and 5 mm to
    // the side.
    let (lo, hi, shift, h) = (0.02, 0.01, 0.005, 0.02);
    let low = |k: usize| {
        let c = [[-lo, -lo], [lo, -lo], [lo, lo], [-lo, lo]][k % 4];
        [c[0], c[1], 0.0]
    };
    let high = |k: usize| {
        let c = [[-hi, -hi], [hi, -hi], [hi, hi], [-hi, hi]][k % 4];
        [c[0] + shift, c[1], h]
    };
    let mut faces = vec![
        FaceSpec::Flat {
            region: square(lo, "a"),
            origin: [0.0, 0.0, 0.0],
            u: [1.0, 0.0, 0.0],
            v: [0.0, 1.0, 0.0],
            name: "loft:start".into(),
        },
        FaceSpec::Flat {
            region: square(hi, "b"),
            origin: [shift, 0.0, h],
            u: [1.0, 0.0, 0.0],
            v: [0.0, 1.0, 0.0],
            name: "loft:end".into(),
        },
    ];
    for k in 0..4 {
        faces.push(FaceSpec::Patch {
            surface: bilinear(low(k), low(k + 1), high(k + 1), high(k)),
            name: format!("loft:side{k}"),
            edges: Vec::new(),
        });
    }
    let map = envelope(&faces, &Tolerances::default()).expect("envelope");
    assert_eq!(map.cells.len(), 5);
    // The top over a point: the end face, or the side it is under.
    let top = |p: P2| -> f64 {
        // Height at which the square through the point is as large as
        // the frustum's there.
        let mut best = 0.0f64;
        for k in 0..=2000 {
            let z = h * k as f64 / 2000.0;
            let half = lo + (hi - lo) * z / h;
            let cx = shift * z / h;
            if (p[0] - cx).abs() <= half && p[1].abs() <= half {
                best = z;
            }
        }
        best
    };
    let mut n = 0;
    for i in 0..40 {
        for j in 0..40 {
            let p = [-0.0195 + 0.001 * i as f64, -0.0195 + 0.001 * j as f64];
            let (bot, t) = map
                .column(p)
                .unwrap_or_else(|| panic!("no column at {p:?}"));
            assert!(bot.abs() < 1e-12);
            assert!((t - top(p)).abs() < 2e-5, "{t} against {} at {p:?}", top(p));
            n += 1;
        }
    }
    assert_eq!(n, 1600);
    assert!(map.column([0.021, 0.0]).is_none());
}

#[test]
fn a_tube_from_its_faces() {
    // A lying tube of radius 20 mm, 50 mm long, its axis 30 mm up: one
    // patch, closed around, and two upright ends.
    let f = |u: f64, v: f64| [0.05 * v, 0.02 * u.cos(), 0.03 + 0.02 * u.sin()];
    let surface = fit_surface(&f, (0.0, core::f64::consts::TAU), (0.0, 1.0), 2e-8);
    let end = |x: f64, name: &str| FaceSpec::Flat {
        region: region_at(
            &[piece("rim", Curve::circle([0.0, 0.03], 0.02))],
            [0.0, 0.03],
        ),
        origin: [x, 0.0, 0.0],
        u: [0.0, 1.0, 0.0],
        v: [0.0, 0.0, 1.0],
        name: name.into(),
    };
    let faces = vec![
        FaceSpec::Patch {
            surface,
            name: "tube".into(),
            edges: Vec::new(),
        },
        end(0.0, "start"),
        end(0.05, "end"),
    ];
    let map = envelope(&faces, &Tolerances::default()).expect("envelope");
    assert_eq!(
        map.cells.len(),
        1,
        "{:?}",
        map.cells
            .iter()
            .map(|c| c.region.bbox())
            .collect::<Vec<_>>()
    );
    for i in 0..20 {
        for j in 0..20 {
            let p = [0.00125 + 0.0025 * i as f64, -0.019 + 0.0019 * j as f64];
            let (bot, top) = map
                .column(p)
                .unwrap_or_else(|| panic!("no column at {p:?}"));
            let half = (0.02f64 * 0.02 - p[1] * p[1]).sqrt();
            assert!((top - 0.03 - half).abs() < 1e-7 && (bot - 0.03 + half).abs() < 1e-7);
        }
    }
    assert!(map.column([0.025, 0.0201]).is_none());
}

use brep_build_core::chain::{Chain, Span};
use brep_build_core::loft::{Profile, loft};

/// The boundary of a region of the plane at height `z`, as a closed
/// chain with a corner at every piece.
fn profile_of(region: &Region, shift: [f64; 2], z: f64, name: &str) -> Profile {
    let up = |q: P2| [q[0] + shift[0], q[1] + shift[1], z];
    let spans: Vec<Span> = region
        .outer
        .iter()
        .map(|p| match p.curve {
            Curve::Line { a, b } => Span::Line { a: up(a), b: up(b) },
            Curve::Arc { c, r, a0, sweep } => Span::Arc {
                c: up(c),
                x: [1.0, 0.0, 0.0],
                y: [0.0, 1.0, 0.0],
                r,
                a0,
                sweep,
            },
        })
        .collect();
    let n = spans.len();
    Profile {
        chain: Chain::new(spans, vec![true; n], true),
        plane: ([0.0, 0.0, z], [0.0, 0.0, 1.0]),
        cap: Some(FaceSpec::Flat {
            region: region.clone(),
            origin: [shift[0], shift[1], z],
            u: [1.0, 0.0, 0.0],
            v: [0.0, 1.0, 0.0],
            name: name.into(),
        }),
    }
}

#[test]
fn a_ruled_loft_is_the_frustum() {
    let (lo, hi, shift, h) = (0.02, 0.01, 0.005, 0.02);
    let a = profile_of(&square(lo, "a"), [0.0, 0.0], 0.0, "loft:start");
    // The second square's loop from another corner.
    let mut upper = square(hi, "b");
    upper.outer.rotate_left(2);
    let b = profile_of(&upper, [shift, 0.0], h, "loft:end");
    let (faces, note) = loft("loft", &a, &b, &[], 5e-8).expect("loft");
    assert!(note.contains("corner to corner"), "{note}");
    assert_eq!(faces.len(), 6);
    let map = envelope(&faces, &Tolerances::default()).expect("envelope");
    for i in 0..40 {
        for j in 0..40 {
            let p = [-0.0195 + 0.001 * i as f64, -0.0195 + 0.001 * j as f64];
            let (bot, top) = map
                .column(p)
                .unwrap_or_else(|| panic!("no column at {p:?}"));
            // The side over the point: the highest level at which the
            // frustum's square still holds it.
            let mut expect = 0.0f64;
            for k in 0..=4000 {
                let z = h * k as f64 / 4000.0;
                if (p[0] - shift * z / h).abs() <= lo + (hi - lo) * z / h
                    && p[1].abs() <= lo + (hi - lo) * z / h
                {
                    expect = z;
                }
            }
            assert!(
                bot.abs() < 1e-9 && (top - expect).abs() < 1e-5,
                "{top} against {expect} at {p:?}"
            );
        }
    }
}

/// From (20, 0) on the lower circle to (10, 30) on the upper, bulging
/// 4 mm outwards, in the plane y = 0 (millimetres).
fn bulging_guide(side: f64) -> Chain {
    let (p, q): ([f64; 2], [f64; 2]) = ([0.02, 0.0], [0.01, 0.03]);
    let m = [
        0.015 + 0.004 * 0.9486832980505138,
        0.015 + 0.004 * 0.31622776601683794,
    ];
    let (ab, ac) = ([m[0] - p[0], m[1] - p[1]], [q[0] - p[0], q[1] - p[1]]);
    let den = 2.0 * (ab[0] * ac[1] - ab[1] * ac[0]);
    let (l1, l2) = (ab[0] * ab[0] + ab[1] * ab[1], ac[0] * ac[0] + ac[1] * ac[1]);
    let c = [
        p[0] + (ac[1] * l1 - ab[1] * l2) / den,
        p[1] + (ab[0] * l2 - ac[0] * l1) / den,
    ];
    let r = (p[0] - c[0]).hypot(p[1] - c[1]);
    let (t0, t1) = (
        (p[1] - c[1]).atan2(p[0] - c[0]),
        (q[1] - c[1]).atan2(q[0] - c[0]),
    );
    Chain::new(
        vec![Span::Arc {
            c: [side * c[0], 0.0, c[1]],
            x: [side, 0.0, 0.0],
            y: [0.0, 0.0, 1.0],
            r,
            a0: t0,
            sweep: t1 - t0,
        }],
        vec![true],
        false,
    )
}

#[test]
fn a_guided_loft_follows_its_guides() {
    // From a circle of radius 20 mm to one of 10 mm, 30 mm above, along
    // two arcs that bulge outwards in the plane y = 0.
    let disc = |r: f64| region_at(&[piece("c", Curve::circle([0.0, 0.0], r))], [0.0, 0.0]);
    let a = profile_of(&disc(0.02), [0.0, 0.0], 0.0, "loft:start");
    let b = profile_of(&disc(0.01), [0.0, 0.0], 0.03, "loft:end");
    // The circle through (20, 0), (10, 30) and a point 4 mm outside the
    // middle of the chord between them, in millimetres.
    let guides = [bulging_guide(1.0), bulging_guide(-1.0)];
    let (faces, note) = loft("loft", &a, &b, &guides, 5e-8).expect("loft");
    assert!(note.contains("2 guides"), "{note}");
    let map = envelope(&faces, &Tolerances::default()).expect("envelope");
    // Over the plane of the guides the top is the guide.
    for k in 1..40 {
        let s = guides[0].total() * k as f64 / 40.0;
        let g = guides[0].at(s);
        // (Not at the profiles, nor where the bulge is at its widest:
        // the loft ends there.)
        if g[0] < 0.0101 || g[0] > 0.02045 || g[2] < 1e-4 {
            continue;
        }
        for side in [1.0, -1.0] {
            // (Beside the guides' plane, which divides the plan.)
            let Some((_, top)) = map.column([side * g[0], 1e-5]) else {
                let cells: Vec<String> = map
                    .cells
                    .iter()
                    .map(|c| {
                        format!(
                            "{} / {} {:?}",
                            map.sheets[c.top].name,
                            map.sheets[c.bot].name,
                            c.region.bbox()
                        )
                    })
                    .collect();
                panic!("no column at {:?} (side {side}); cells {cells:#?}", g);
            };
            // Where the guide bulges past the lower profile the column
            // meets it twice, and the top is the upper crossing.
            if g[0] <= 0.02 {
                assert!(top >= g[2] - 1e-5, "{top} under the guide at {g:?}");
            }
        }
    }
    let (bot, top) = map.column([0.001, 0.002]).unwrap();
    assert!(bot.abs() < 1e-9 && (top - 0.03).abs() < 1e-9);
    // Half way between the guides their bulges, which are opposite,
    // cancel: the loft is as wide as the lower profile there.
    assert!(map.column([0.0, 0.0199]).is_some());
    assert!(map.column([0.0, 0.0201]).is_none());
    // Beside a guide it overhangs the lower profile: the column there
    // is between two crossings of the same face.
    let (bot, top) = map.column([0.0203, 0.0005]).expect("the overhang");
    assert!(bot > 1e-3 && top > bot + 1e-3 && top < 0.02, "{bot} {top}");
    let v = map.volume(5e-4);
    assert!(v > 2.2e-5 && v < 3.6e-5, "{v}");
}
