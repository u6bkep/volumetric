//! Constructors of [`brep_core::topo`] solids.
//!
//! [`prism`] sweeps an exact 2D region (lines and circular arcs, from
//! `sketch_regions_core`) along the normal of its plane: caps are planes,
//! walls are planes and cylinders, every edge is an exact line or circle
//! and every wall is named after the sketch entity it was swept from.

pub mod conic;
pub mod fit;
pub mod lift;
pub mod planmap;
pub mod sheet;
pub mod trace;

use brep_core::ir::Surface;
use brep_core::math::{Frame, Vec3, scale};
use brep_core::topo::{Coedge, Curve3, Edge, Face, Solid};
use core::f64::consts::TAU;
use sketch_regions_core::{Curve, Piece, Region};

fn neg(v: Vec3) -> Vec3 {
    scale(v, -1.0)
}

fn is_full_circle(c: &Curve) -> bool {
    matches!(c, Curve::Arc { sweep, .. } if sweep.abs() >= TAU - 1e-9)
}

/// The curve of a piece lifted to height `h` above the plane, with the
/// parameter span that runs it from the piece's start to its end.
fn lift(curve: &Curve, frame: &Frame, h: f64) -> (Curve3, f64, f64) {
    match *curve {
        Curve::Line { a, b } => {
            let p = frame.to_world([a[0], a[1], h]);
            let q = frame.to_world([b[0], b[1], h]);
            let len = brep_core::math::norm(brep_core::math::sub(q, p));
            (Curve3::line_through(p, q), 0.0, len)
        }
        Curve::Arc { c, r, a0, sweep } => {
            let origin = frame.to_world([c[0], c[1], h]);
            if sweep >= 0.0 {
                let f = Frame {
                    origin,
                    x: frame.x,
                    y: frame.y,
                    z: frame.z,
                };
                (
                    Curve3::Circle {
                        frame: f,
                        radius: r,
                    },
                    a0,
                    a0 + sweep,
                )
            } else {
                // Clockwise in the plane: the same circle about the
                // opposite axis, where plane angle `th` is parameter `-th`.
                let f = Frame {
                    origin,
                    x: frame.x,
                    y: neg(frame.y),
                    z: neg(frame.z),
                };
                (
                    Curve3::Circle {
                        frame: f,
                        radius: r,
                    },
                    -a0,
                    -a0 - sweep,
                )
            }
        }
    }
}

/// The wall swept from one piece: its surface and whether the material
/// lies behind the surface normal. The region is on the left of the
/// piece, so the wall faces right.
fn wall_surface(curve: &Curve, frame: &Frame) -> (Surface, bool) {
    match *curve {
        Curve::Line { a, .. } => {
            let t = curve.tangent(0.0);
            let along = frame.dir_to_world([t[0], t[1], 0.0]);
            let out = frame.dir_to_world([t[1], -t[0], 0.0]);
            let f = Frame {
                origin: frame.to_world([a[0], a[1], 0.0]),
                x: along,
                y: frame.z,
                z: out,
            };
            (Surface::Plane { frame: f }, true)
        }
        Curve::Arc { c, r, sweep, .. } => {
            let f = Frame {
                origin: frame.to_world([c[0], c[1], 0.0]),
                x: frame.x,
                y: frame.y,
                z: frame.z,
            };
            // Counter-clockwise keeps the centre on the left: the wall
            // faces away from the axis, as the cylinder's normal does.
            (
                Surface::Cylinder {
                    frame: f,
                    radius: r,
                },
                sweep >= 0.0,
            )
        }
    }
}

/// The solid swept from `region` (plane coordinates of `frame`) along
/// `frame.z` between heights `h0 < h1`.
pub fn prism(
    region: &Region,
    frame: &Frame,
    h0: f64,
    h1: f64,
    name: &str,
) -> Result<Solid, String> {
    if h1 <= h0 || h0.is_nan() || h1.is_nan() {
        return Err(format!("prism heights {h0}..{h1} are empty"));
    }
    let mut solid = Solid {
        name: name.to_string(),
        ..Solid::default()
    };
    let mut top_loops: Vec<Vec<Coedge>> = Vec::new();
    let mut bottom_loops: Vec<Vec<Coedge>> = Vec::new();

    let loops: Vec<&Vec<Piece>> = std::iter::once(&region.outer)
        .chain(region.holes.iter())
        .collect();
    for (li, pieces) in loops.iter().enumerate() {
        if pieces.is_empty() {
            return Err(format!("loop {li} of the region is empty"));
        }
        let n = pieces.len();
        let closed_circle = n == 1 && is_full_circle(&pieces[0].curve);
        if n == 1 && !closed_circle {
            return Err(format!("loop {li} of the region is one open piece"));
        }
        // Vertices at each piece's start, bottom then top.
        let v0 = solid.vertices.len();
        for p in pieces.iter() {
            let s = p.curve.start();
            solid.vertices.push(frame.to_world([s[0], s[1], h0]));
            solid.vertices.push(frame.to_world([s[0], s[1], h1]));
        }
        let bottom_v = |i: usize| v0 + 2 * (i % n);
        let top_v = |i: usize| v0 + 2 * (i % n) + 1;

        // Edges: per piece its bottom and top curve, and the vertical
        // line at its start (none on a seamless full circle).
        let e0 = solid.edges.len();
        let per = if closed_circle { 2 } else { 3 };
        for (i, p) in pieces.iter().enumerate() {
            let (cb, t0, t1) = lift(&p.curve, frame, h0);
            solid.edges.push(Edge {
                curve: cb,
                t0,
                t1,
                start: bottom_v(i),
                end: bottom_v(i + 1),
            });
            let (ct, t0, t1) = lift(&p.curve, frame, h1);
            solid.edges.push(Edge {
                curve: ct,
                t0,
                t1,
                start: top_v(i),
                end: top_v(i + 1),
            });
            if !closed_circle {
                let a = solid.vertices[bottom_v(i)];
                let b = solid.vertices[top_v(i)];
                solid.edges.push(Edge {
                    curve: Curve3::line_through(a, b),
                    t0: 0.0,
                    t1: h1 - h0,
                    start: bottom_v(i),
                    end: top_v(i),
                });
            }
        }
        let bottom_e = |i: usize| e0 + per * (i % n);
        let top_e = |i: usize| e0 + per * (i % n) + 1;
        let side_e = |i: usize| e0 + per * (i % n) + 2;

        // Walls.
        for (i, p) in pieces.iter().enumerate() {
            let (surface, same_sense) = wall_surface(&p.curve, frame);
            let loops = if closed_circle {
                vec![
                    vec![Coedge {
                        edge: bottom_e(i),
                        forward: true,
                    }],
                    vec![Coedge {
                        edge: top_e(i),
                        forward: false,
                    }],
                ]
            } else {
                vec![vec![
                    Coedge {
                        edge: bottom_e(i),
                        forward: true,
                    },
                    Coedge {
                        edge: side_e(i + 1),
                        forward: true,
                    },
                    Coedge {
                        edge: top_e(i),
                        forward: false,
                    },
                    Coedge {
                        edge: side_e(i),
                        forward: false,
                    },
                ]]
            };
            solid.faces.push(Face {
                surface,
                same_sense,
                loops,
                name: format!("wall:{}", p.id),
                color: None,
            });
        }

        // The caps' loops: the top runs with the region's loop, the
        // bottom against it.
        top_loops.push(
            (0..n)
                .map(|i| Coedge {
                    edge: top_e(i),
                    forward: true,
                })
                .collect(),
        );
        bottom_loops.push(
            (0..n)
                .rev()
                .map(|i| Coedge {
                    edge: bottom_e(i),
                    forward: false,
                })
                .collect(),
        );
    }

    let cap = |h: f64| Surface::Plane {
        frame: Frame {
            origin: frame.to_world([0.0, 0.0, h]),
            x: frame.x,
            y: frame.y,
            z: frame.z,
        },
    };
    solid.faces.push(Face {
        surface: cap(h1),
        same_sense: true,
        loops: top_loops,
        name: "cap:top".into(),
        color: None,
    });
    solid.faces.push(Face {
        surface: cap(h0),
        same_sense: false,
        loops: bottom_loops,
        name: "cap:bottom".into(),
        color: None,
    });
    Ok(solid)
}
