//! Bodies for the plan map: what a feature sweeps, as cells and sheets.
//!
//! [`sweep`] is a region of any plane moved along any direction, which
//! covers the extrude in all its forms: a sketch on the ground swept
//! upwards (the region itself, between two levels), a profile on an
//! upright plane swept sideways (strips of the plan under the profile's
//! lines and arcs, which become planes and lying cylinders), and a sweep
//! at a slant.
//!
//! The plan is divided by the shadows of the body's own edges: the
//! region's boundary at both ends of the sweep, the path of every corner
//! of the region, and on a curved face the line where it turns vertical.
//! Within one face of that division the column through the body enters
//! and leaves by the same two faces, which a point inside it tells.

use crate::chain::Span;
use crate::planmap::{Cell, PlanMap, Tolerances, cells_of};
use crate::sheet::Sheet;
use crate::trace::{FREE, pieces_of, trace_zero};
use brep_core::ir::{NurbsSurface, Surface};
use brep_core::math::{Frame, Vec3, add, cross, dot, norm, normalize, scale, sub};
use core::f64::consts::{FRAC_PI_2, PI, TAU};
use sketch_regions_core::{Curve, P2, Piece, Region};
use std::collections::HashMap;

/// A region on a plane swept along a direction: the points
/// `origin + a u + b v + w dir` with `(a, b)` in the region and
/// `w0 <= w <= w1`.
#[derive(Clone, Debug)]
pub struct Sweep {
    pub region: Region,
    pub origin: Vec3,
    /// The plane's axes: unit vectors at right angles.
    pub u: Vec3,
    pub v: Vec3,
    /// Unit direction of the sweep, not in the plane.
    pub dir: Vec3,
    /// Either may be infinite (through everything).
    pub w0: f64,
    pub w1: f64,
    /// Begins the names of the body's faces (the feature).
    pub name: String,
}

/// An infinite sweep ends this far out, metres.
const REACH: f64 = 8.0;

/// A face of the swept body.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Side {
    Start,
    End,
    /// The face swept by the boundary piece of this index.
    Swept(usize),
}

fn inv3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let c = |a: usize, b: usize, c: usize, d: usize| m[a][b] * m[c][d] - m[a][d] * m[c][b];
    let det = m[0][0] * c(1, 1, 2, 2) - m[0][1] * c(1, 0, 2, 2) + m[0][2] * c(1, 0, 2, 1);
    if det.abs() < 1e-14 {
        return None;
    }
    let d = 1.0 / det;
    Some([
        [c(1, 1, 2, 2) * d, -c(0, 1, 2, 2) * d, c(0, 1, 1, 2) * d],
        [-c(1, 0, 2, 2) * d, c(0, 0, 2, 2) * d, -c(0, 0, 1, 2) * d],
        [c(1, 0, 2, 1) * d, -c(0, 0, 2, 1) * d, c(0, 0, 1, 1) * d],
    ])
}

fn mul3(m: &[[f64; 3]; 3], v: Vec3) -> Vec3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

/// The surface swept by an arc of the plane along `dir`, exactly: the
/// arc as rational quadratic pieces of a quarter turn at most, by a
/// line.
fn swept_arc(s: &Sweep, c: P2, r: f64, a0: f64, turn: f64, w: (f64, f64)) -> NurbsSurface {
    let n = ((turn.abs() / FRAC_PI_2).ceil() as usize).max(1);
    let step = turn / n as f64;
    let at = |th: f64, radius: f64| -> Vec3 {
        add(
            s.origin,
            add(
                scale(s.u, c[0] + radius * th.cos()),
                scale(s.v, c[1] + radius * th.sin()),
            ),
        )
    };
    let weight = (0.5 * step).cos();
    let mut row: Vec<(Vec3, f64)> = vec![(at(a0, r), 1.0)];
    for i in 0..n {
        let from = a0 + step * i as f64;
        row.push((at(from + 0.5 * step, r / weight), weight));
        row.push((at(from + step, r), 1.0));
    }
    let mut knots_u = vec![0.0; 3];
    for i in 1..n {
        knots_u.extend([i as f64; 2]);
    }
    knots_u.extend([n as f64; 3]);
    let mut ctrl = Vec::with_capacity(2 * row.len());
    for (p, weight) in &row {
        for end in [w.0, w.1] {
            let q = add(*p, scale(s.dir, end));
            ctrl.push([q[0], q[1], q[2], *weight]);
        }
    }
    NurbsSurface {
        degree_u: 2,
        degree_v: 1,
        nctrl_u: row.len(),
        nctrl_v: 2,
        knots_u,
        knots_v: vec![w.0, w.0, w.1, w.1],
        ctrl,
    }
}

/// The shadow on the plan of a curve of a plane whose axes have the
/// shadows `ux` and `vx`, `to_plan` taking the plane's points to the
/// plan: a line's is a line, an arc's an arc (the plane level), a line
/// (the plane upright, seen edge on) or an ellipse, which comes as short
/// lines under a [`FREE`] id.
fn shadow(
    curve: &Curve,
    to_plan: &dyn Fn(P2) -> P2,
    ux: P2,
    vx: P2,
    id: &str,
    tol: &Tolerances,
) -> Vec<(String, Curve)> {
    let det = ux[0] * vx[1] - ux[1] * vx[0];
    let rigid = (ux[0] * ux[0] + ux[1] * ux[1] - 1.0).abs() < 1e-12
        && (vx[0] * vx[0] + vx[1] * vx[1] - 1.0).abs() < 1e-12
        && (ux[0] * vx[0] + ux[1] * vx[1]).abs() < 1e-12;
    match *curve {
        Curve::Line { a, b } => vec![(
            id.to_string(),
            Curve::Line {
                a: to_plan(a),
                b: to_plan(b),
            },
        )],
        Curve::Arc { c, r, sweep, .. } => {
            if rigid {
                let full = sweep.abs() >= TAU - 1e-9;
                let arc = if full {
                    Curve::circle(to_plan(c), r)
                } else {
                    Curve::arc_from_ends(
                        to_plan(c),
                        to_plan(curve.start()),
                        to_plan(curve.end()),
                        (sweep > 0.0) == (det > 0.0),
                    )
                };
                vec![(id.to_string(), arc)]
            } else if det.abs() < 1e-12 {
                // Seen edge on: the stretch of the plane's trace the arc
                // covers.
                let trace = if ux[0].hypot(ux[1]) > vx[0].hypot(vx[1]) {
                    ux
                } else {
                    vx
                };
                let along = |q: P2| {
                    let f = to_plan(q);
                    f[0] * trace[0] + f[1] * trace[1]
                };
                let (al, be) = (
                    r * (ux[0] * trace[0] + ux[1] * trace[1]),
                    r * (vx[0] * trace[0] + vx[1] * trace[1]),
                );
                let mut stops = vec![curve.start(), curve.end()];
                for th in [be.atan2(al), be.atan2(al) + PI] {
                    if let Some(t) = curve.arc_param(th, 0.0) {
                        stops.push(curve.point(t));
                    }
                }
                stops.sort_by(|x, y| along(*x).total_cmp(&along(*y)));
                vec![(
                    id.to_string(),
                    Curve::Line {
                        a: to_plan(stops[0]),
                        b: to_plan(stops[stops.len() - 1]),
                    },
                )]
            } else {
                // An ellipse in the plan.
                let step = 2.0 * (1.0 - (tol.chord / r).min(0.5)).acos();
                let n = ((sweep.abs() / step).ceil() as usize).max(8);
                (0..n)
                    .map(|i| {
                        let (t0, t1) = (i as f64 / n as f64, (i + 1) as f64 / n as f64);
                        (
                            format!("{FREE}{id}"),
                            Curve::Line {
                                a: to_plan(curve.point(t0)),
                                b: to_plan(curve.point(t1)),
                            },
                        )
                    })
                    .collect()
            }
        }
    }
}

/// A face of a body given by its faces.
#[derive(Clone, Debug)]
pub enum FaceSpec {
    /// A region of the plane `origin + a u + b v`.
    Flat {
        region: Region,
        origin: Vec3,
        u: Vec3,
        v: Vec3,
        name: String,
    },
    /// A whole patch of surface. `edges` are the curves around it where
    /// they are known better than the surface has them (a fitted patch
    /// follows its edges to a tolerance, and its neighbour follows the
    /// same edge to the same tolerance elsewhere); without any, the
    /// patch's own four edges are taken.
    Patch {
        surface: NurbsSurface,
        name: String,
        edges: Vec<Span>,
    },
}

/// The body enclosed by `faces`, as far as a plan map holds it: over
/// every plan point the column from the lowest of the faces to the
/// highest.
///
/// The plan is divided by the shadows of the faces' boundaries and, on a
/// patch, of the curves where it turns vertical.
pub fn envelope(faces: &[FaceSpec], tol: &Tolerances) -> Result<PlanMap, String> {
    let flat = |p: Vec3| -> P2 { [p[0], p[1]] };
    let mut plan: Vec<Piece> = Vec::new();
    let mut push = |id: String, curve: Curve| {
        if curve.length() > tol.snap {
            plan.push(Piece { id, curve });
        }
    };
    // Both halves of every patch, the plane of every flat face.
    let mut sheets: Vec<Vec<Sheet>> = Vec::with_capacity(faces.len());
    for face in faces {
        match face {
            FaceSpec::Flat {
                region,
                origin,
                u,
                v,
                name,
            } => {
                let to_plan = |q: P2| flat(add(*origin, add(scale(*u, q[0]), scale(*v, q[1]))));
                for p in region.pieces() {
                    let id = format!("{name}:{}", p.id);
                    for (id, curve) in shadow(&p.curve, &to_plan, flat(*u), flat(*v), &id, tol) {
                        push(id, curve);
                    }
                }
                let n = normalize(cross(*u, *v));
                sheets.push(if n[2].abs() < 1e-9 {
                    Vec::new()
                } else {
                    let n = if n[2] > 0.0 { n } else { scale(n, -1.0) };
                    vec![Sheet::of(
                        Surface::Plane {
                            frame: Frame::from_axis_ref(*origin, n, *u),
                        },
                        true,
                        name,
                    )]
                });
            }
            FaceSpec::Patch {
                surface,
                name,
                edges,
            } => {
                surface.validate()?;
                let dom = surface.domain();
                let at = |u: f64, v: f64| flat(brep_core::nurbs::surface_eval(surface, u, v).0);
                let mut edge = |curve: &dyn Fn(f64) -> P2, which: &str| {
                    for line in polyline(curve, tol.chord) {
                        for p in pieces_of(&line, &format!("{name}:{which}"), tol.exact, [0.0; 2]) {
                            push(p.id, p.curve);
                        }
                    }
                };
                if edges.is_empty() {
                    edge(&|t| at(dom[0] + t * (dom[1] - dom[0]), dom[2]), "v0");
                    edge(&|t| at(dom[0] + t * (dom[1] - dom[0]), dom[3]), "v1");
                    edge(&|t| at(dom[0], dom[2] + t * (dom[3] - dom[2])), "u0");
                    edge(&|t| at(dom[1], dom[2] + t * (dom[3] - dom[2])), "u1");
                }
                for (k, span) in edges.iter().enumerate() {
                    let len = span.len();
                    if len > tol.snap {
                        edge(&|t| flat(span.at(t * len)), &format!("edge{k}"));
                    }
                }
                let (du, dv) = (dom[1] - dom[0], dom[3] - dom[2]);
                let own = |q: P2| (dom[0] + q[0] * du, dom[2] + q[1] * dv);
                // A patch that stands upright all over is a wall: its
                // edges' shadows are all there is of it in the plan.
                let upright = (0..=8).all(|i| {
                    (0..=8).all(|j| {
                        let (u, v) = own([i as f64 / 8.0, j as f64 / 8.0]);
                        let (_, su, sv) = brep_core::nurbs::surface_eval(surface, u, v);
                        let n = cross(su, sv);
                        n[2].abs() <= 1e-6 * norm(n)
                    })
                });
                if upright {
                    sheets.push(Vec::new());
                    continue;
                }
                // Where it turns vertical: the level normal, followed in
                // the patch's own parameters brought to the unit square.
                let level = |q: P2| -> Option<f64> {
                    if q[0] < 0.0 || q[0] > 1.0 || q[1] < 0.0 || q[1] > 1.0 {
                        return None;
                    }
                    let (u, v) = own(q);
                    let (_, su, sv) = brep_core::nurbs::surface_eval(surface, u, v);
                    let n = cross(su, sv);
                    let len = norm(n);
                    (len > 0.0).then(|| n[2] / len)
                };
                // A chord of the curve in the plan is about as long as
                // the patch is wide times its length in parameters.
                let size = {
                    let (a, b, c) = (at(dom[0], dom[2]), at(dom[1], dom[2]), at(dom[0], dom[3]));
                    ((b[0] - a[0]).hypot(b[1] - a[1]))
                        .max((c[0] - a[0]).hypot(c[1] - a[1]))
                        .max(1e-6)
                };
                let rims = trace_zero(&level, [0.0, 0.0, 1.0, 1.0], 1.0 / 48.0, tol.chord / size);
                for (k, rim) in rims.iter().enumerate() {
                    let line: Vec<P2> = rim
                        .iter()
                        .map(|q| {
                            let (u, v) = own(*q);
                            at(u, v)
                        })
                        .collect();
                    for p in pieces_of(&line, &format!("{name}:rim{k}"), tol.exact, [0.0; 2]) {
                        push(p.id, p.curve);
                    }
                }
                sheets.push(
                    [true, false]
                        .into_iter()
                        .map(|up| Sheet::of(Surface::Nurbs(surface.clone()), up, name))
                        .collect(),
                );
            }
        }
    }

    let mut map = PlanMap::new();
    map.tol = *tol;
    let mut made: HashMap<(usize, usize), usize> = HashMap::new();
    let cells: Vec<Cell> = cells_of(&plan, tol.snap, &mut |p| {
        // Every face the column meets.
        let mut lo: Option<(f64, usize, usize)> = None;
        let mut hi: Option<(f64, usize, usize)> = None;
        let mut met = 0;
        for (f, (face, halves)) in faces.iter().zip(&sheets).enumerate() {
            for (h, sheet) in halves.iter().enumerate() {
                let Some(z) = sheet.z_open(p[0], p[1]) else {
                    continue;
                };
                if let FaceSpec::Flat {
                    region,
                    origin,
                    u,
                    v,
                    ..
                } = face
                {
                    let d = sub([p[0], p[1], z], *origin);
                    if !region.contains([dot(d, *u), dot(d, *v)]) {
                        continue;
                    }
                }
                met += 1;
                if lo.is_none_or(|l| z < l.0) {
                    lo = Some((z, f, h));
                }
                if hi.is_none_or(|l| z > l.0) {
                    hi = Some((z, f, h));
                }
            }
        }
        let (Some(lo), Some(hi)) = (lo, hi) else {
            return Ok(None);
        };
        if met < 2 || hi.0 - lo.0 <= tol.height {
            return Ok(None);
        }
        let mut sheet = |(_, f, h): (f64, usize, usize)| {
            *made.entry((f, h)).or_insert_with(|| {
                map.sheets.push(sheets[f][h].clone());
                map.sheets.len() - 1
            })
        };
        Ok(Some((sheet(hi), sheet(lo))))
    })?;
    if cells.is_empty() {
        return Err("the faces enclose nothing the plan can hold".into());
    }
    map.cells = cells;
    Ok(map)
}

/// Points of a curve over `[0, 1]` whose chords stay within `tol` of it,
/// in runs that do not turn back on themselves: the shadow of a curve
/// seen edge on goes out and back along one line, and turns at a point
/// of its own.
fn polyline(curve: &dyn Fn(f64) -> P2, tol: f64) -> Vec<Vec<P2>> {
    // Halved until the middle of each part is on its chord, and between
    // the chord's ends (a curve can swing back onto its chord).
    fn halve(
        curve: &dyn Fn(f64) -> P2,
        tol: f64,
        from: (f64, P2),
        to: (f64, P2),
        depth: usize,
        out: &mut Vec<(f64, P2)>,
    ) {
        let mid = 0.5 * (from.0 + to.0);
        let q = curve(mid);
        let chord = [to.1[0] - from.1[0], to.1[1] - from.1[1]];
        let len = chord[0].hypot(chord[1]);
        let (off, along) = if len > 0.0 {
            let d = [q[0] - from.1[0], q[1] - from.1[1]];
            (
                (d[0] * chord[1] - d[1] * chord[0]).abs() / len,
                (d[0] * chord[0] + d[1] * chord[1]) / (len * len),
            )
        } else {
            ((q[0] - from.1[0]).hypot(q[1] - from.1[1]), 0.5)
        };
        if (off > tol || !(0.2..=0.8).contains(&along)) && depth < 20 {
            halve(curve, tol, from, (mid, q), depth + 1, out);
            halve(curve, tol, (mid, q), to, depth + 1, out);
        } else {
            out.push(to);
        }
    }
    let mut out: Vec<(f64, P2)> = vec![(0.0, curve(0.0))];
    for k in 1..=16 {
        let from = *out.last().unwrap();
        let t = k as f64 / 16.0;
        halve(curve, tol, from, (t, curve(t)), 0, &mut out);
    }
    // Turns: where a chord runs against the one before. The point of
    // the turn is the furthest the curve gets along the way in.
    let mut runs: Vec<Vec<P2>> = vec![vec![out[0].1]];
    for i in 1..out.len() {
        let (t, p) = out[i];
        let turned = i + 1 < out.len() && {
            let (a, b) = (out[i - 1].1, out[i + 1].1);
            (p[0] - a[0]) * (b[0] - p[0]) + (p[1] - a[1]) * (b[1] - p[1]) < 0.0
        };
        if !turned {
            runs.last_mut().unwrap().push(p);
            continue;
        }
        let from = out[i - 1].1;
        let way = [p[0] - from[0], p[1] - from[1]];
        let reach = |t: f64| {
            let q = curve(t);
            (q[0] - from[0]) * way[0] + (q[1] - from[1]) * way[1]
        };
        const G: f64 = 0.618_033_988_749_894_9;
        let (mut lo, mut hi) = (out[i - 1].0, out[i + 1].0);
        for _ in 0..64 {
            let (a, b) = (hi - G * (hi - lo), lo + G * (hi - lo));
            if reach(a) > reach(b) {
                hi = b;
            } else {
                lo = a;
            }
        }
        let _ = t;
        let turn = curve(0.5 * (lo + hi));
        runs.last_mut().unwrap().push(turn);
        runs.push(vec![turn]);
    }
    runs.retain(|r| r.len() >= 2);
    runs
}

/// The body of a sweep.
pub fn sweep(s: &Sweep, tol: &Tolerances) -> Result<PlanMap, String> {
    let normal = normalize(cross(s.u, s.v));
    let m = [
        [s.u[0], s.v[0], s.dir[0]],
        [s.u[1], s.v[1], s.dir[1]],
        [s.u[2], s.v[2], s.dir[2]],
    ];
    let inv = inv3(m).ok_or("the sweep's direction lies in its plane")?;
    let (w0, w1) = (s.w0.max(-REACH), s.w1.min(REACH));
    if w0.is_nan() || w1.is_nan() || w0 >= w1 {
        return Err(format!("the sweep {}..{} is empty", s.w0, s.w1));
    }
    let point = |q: P2, w: f64| -> Vec3 {
        add(
            s.origin,
            add(add(scale(s.u, q[0]), scale(s.v, q[1])), scale(s.dir, w)),
        )
    };
    let flat = |p: Vec3| -> P2 { [p[0], p[1]] };
    let level = |v: Vec3| v[2].abs() < 1e-9;
    let upright = s.dir[2].abs() > 1.0 - 1e-12;
    let pieces: Vec<&Piece> = s.region.pieces().collect();

    // The faces' surfaces. `None` for a face that stands upright: a wall
    // of the plan, not a sheet.
    let cap = |w: f64, which: &str| -> Option<Surface> {
        (!level(normal)).then(|| {
            let n = if normal[2] > 0.0 {
                normal
            } else {
                scale(normal, -1.0)
            };
            let _ = which;
            Surface::Plane {
                frame: Frame::from_axis_ref(point([0.0, 0.0], w), n, s.u),
            }
        })
    };
    let mut surfaces: HashMap<Side, Option<Surface>> = HashMap::new();
    surfaces.insert(Side::Start, cap(w0, "start"));
    surfaces.insert(Side::End, cap(w1, "end"));
    for (k, p) in pieces.iter().enumerate() {
        let surface = match p.curve {
            Curve::Line { a, b } => {
                let along = sub(point(b, 0.0), point(a, 0.0));
                let n = normalize(cross(along, s.dir));
                (!level(n)).then(|| {
                    let n = if n[2] > 0.0 { n } else { scale(n, -1.0) };
                    Surface::Plane {
                        frame: Frame::from_axis_ref(point(a, 0.0), n, s.dir),
                    }
                })
            }
            Curve::Arc { c, r, a0, sweep } => {
                if dot(s.dir, normal).abs() > 1.0 - 1e-12 {
                    (!upright).then(|| Surface::Cylinder {
                        frame: Frame::from_axis_ref(point(c, 0.0), s.dir, s.u),
                        radius: r,
                    })
                } else {
                    // A little more than the face, so that its edges are
                    // inside the surface.
                    let full = sweep.abs() >= TAU - 1e-9;
                    let more = if full { 0.0 } else { 1e-3 * sweep.signum() };
                    Some(Surface::Nurbs(swept_arc(
                        s,
                        c,
                        r,
                        a0 - more,
                        sweep + 2.0 * more,
                        (w0 - 1e-3, w1 + 1e-3),
                    )))
                }
            }
        };
        surfaces.insert(Side::Swept(k), surface);
    }

    // The plan's division.
    let named = |what: &str| format!("{}:{what}", s.name);
    let mut plan: Vec<Piece> = Vec::new();
    let mut push = |id: String, curve: Curve| {
        if curve.length() > tol.snap {
            plan.push(Piece { id, curve });
        }
    };
    for (w, end) in [(w0, "start"), (w1, "end")] {
        for p in &pieces {
            // A piece of an upright sweep is its wall; elsewhere it is
            // the edge of an end face.
            let id = if upright {
                named(&p.id)
            } else {
                named(&format!("{end}:{}", p.id))
            };
            for (id, curve) in shadow(
                &p.curve,
                &|q| flat(point(q, w)),
                flat(s.u),
                flat(s.v),
                &id,
                tol,
            ) {
                push(id, curve);
            }
        }
    }
    if !upright {
        let n = pieces.len();
        for (k, p) in pieces.iter().enumerate() {
            // The path of the corner the piece starts at.
            let q = p.curve.start();
            let before = pieces
                .iter()
                .find(|o| {
                    let e = o.curve.end();
                    (e[0] - q[0]).hypot(e[1] - q[1]) <= tol.snap
                })
                .map(|o| o.id.as_str())
                .unwrap_or("");
            let path = Curve::Line {
                a: flat(point(q, w0)),
                b: flat(point(q, w1)),
            };
            if n > 1 || !matches!(p.curve, Curve::Arc { sweep, .. } if sweep.abs() >= TAU - 1e-9) {
                push(named(&format!("{before}|{}", p.id)), path);
            }
            // Where a curved face turns vertical.
            if let Curve::Arc { c, r, .. } = p.curve {
                let (a, b) = (cross(s.u, s.dir)[2], cross(s.v, s.dir)[2]);
                for th in [b.atan2(a), b.atan2(a) + PI] {
                    // At an arc's ends the corner's path is the rim; a
                    // whole circle has no corner.
                    let whole =
                        matches!(p.curve, Curve::Arc { sweep, .. } if sweep.abs() >= TAU - 1e-9);
                    if let Some(t) = p.curve.arc_param(th, 0.0)
                        && (whole || (t > 1e-9 && t < 1.0 - 1e-9))
                    {
                        let q = [c[0] + r * th.cos(), c[1] + r * th.sin()];
                        push(
                            named(&format!("{}:rim", p.id)),
                            Curve::Line {
                                a: flat(point(q, w0)),
                                b: flat(point(q, w1)),
                            },
                        );
                    }
                }
            }
            let _ = k;
        }
    }

    // What a column meets: along it `(a, b, w) = q0 + z q1`.
    let q1 = mul3(&inv, [0.0, 0.0, 1.0]);
    let column = |p: P2| -> Option<((f64, Side), (f64, Side))> {
        let q0 = mul3(&inv, sub([p[0], p[1], 0.0], s.origin));
        // The heights at which the column passes the ends of the sweep.
        let ends = (q1[2].abs() > 1e-300).then(|| {
            let (z0, z1) = ((w0 - q0[2]) / q1[2], (w1 - q0[2]) / q1[2]);
            if z0 <= z1 {
                ((z0, Side::Start), (z1, Side::End))
            } else {
                ((z1, Side::End), (z0, Side::Start))
            }
        });
        if q1[0].abs() < 1e-12 && q1[1].abs() < 1e-12 {
            return s.region.contains([q0[0], q0[1]]).then_some(ends?);
        }
        if ends.is_none() && (q0[2] < w0 || q0[2] > w1) {
            return None;
        }
        let mut lo: Option<(f64, Side)> = None;
        let mut hi: Option<(f64, Side)> = None;
        for (z0, z1, p0, p1) in s.region.line_intervals_with([q0[0], q0[1]], [q1[0], q1[1]]) {
            let mut from = (z0, Side::Swept(p0));
            let mut to = (z1, Side::Swept(p1));
            if let Some((first, last)) = ends {
                if first.0 > from.0 {
                    from = first;
                }
                if last.0 < to.0 {
                    to = last;
                }
            }
            if from.0 >= to.0 {
                continue;
            }
            if lo.is_none_or(|l| from.0 < l.0) {
                lo = Some(from);
            }
            if hi.is_none_or(|h| to.0 > h.0) {
                hi = Some(to);
            }
        }
        Some((lo?, hi?))
    };

    // Sheets as the columns call for them: a curved face has an upper
    // and a lower one.
    let mut map = PlanMap::new();
    map.tol = *tol;
    let mut made: HashMap<(Side, bool), usize> = HashMap::new();
    let mut sheet_for = |map: &mut PlanMap, p: P2, z: f64, side: Side| -> Result<usize, String> {
        let what = match side {
            Side::Start => "start".to_string(),
            Side::End => "end".to_string(),
            Side::Swept(k) => pieces[k].id.clone(),
        };
        let Some(surface) = surfaces.get(&side).cloned().flatten() else {
            return Err(format!(
                "a column at ({:.6}, {:.6}) ends on the upright face '{}'",
                p[0],
                p[1],
                named(&what)
            ));
        };
        for up in [true, false] {
            let sheet = Sheet::of(surface.clone(), up, &named(&what));
            if sheet.z(p[0], p[1]).is_some_and(|h| (h - z).abs() <= 1e-7) {
                return Ok(*made.entry((side, up)).or_insert_with(|| {
                    map.sheets.push(sheet);
                    map.sheets.len() - 1
                }));
            }
        }
        let found: Vec<Option<f64>> = [true, false]
            .into_iter()
            .map(|up| Sheet::of(surface.clone(), up, "").z(p[0], p[1]))
            .collect();
        Err(format!(
            "the face '{}' is not at {z:.6} over ({:.6}, {:.6}) but at {found:?}",
            named(&what),
            p[0],
            p[1]
        ))
    };
    let cells: Vec<Cell> = cells_of(&plan, tol.snap, &mut |p| {
        let Some((lo, hi)) = column(p) else {
            return Ok(None);
        };
        let top = sheet_for(&mut map, p, hi.0, hi.1)?;
        let bot = sheet_for(&mut map, p, lo.0, lo.1)?;
        Ok(Some((top, bot)))
    })?;
    if cells.is_empty() {
        return Err(format!("the sweep '{}' has no footprint", s.name));
    }
    map.cells = cells;
    Ok(map)
}
