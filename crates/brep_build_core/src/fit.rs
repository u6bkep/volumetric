//! Curves through computed points: cubic B-splines that follow a function
//! to a tolerance.
//!
//! The curve is piecewise cubic Hermite (values and first derivatives at
//! the nodes, C1 across them), written as a B-spline with double interior
//! knots. Nodes are added where the curve strays from the function, so a
//! cubic polynomial piece is reproduced exactly from its own ends.

use brep_core::math::{Vec3, add, norm, scale, sub};
use brep_core::nurbs::curve_eval;
use brep_core::topo::NurbsCurve;

/// The derivative of `f` at `t` by differences inside `[t0, t1]`. At a
/// `kink` (where `f` is only C1) the two sides are differenced apart and
/// averaged, so neither reaches across.
fn derivative(f: &dyn Fn(f64) -> Vec3, t: f64, t0: f64, t1: f64, kink: bool) -> Vec3 {
    let h = 1e-5 * (t1 - t0);
    let ahead = |sign: f64| -> Vec3 {
        let (p0, p1, p2) = (f(t), f(t + sign * h), f(t + sign * 2.0 * h));
        scale(
            add(add(scale(p0, -3.0), scale(p1, 4.0)), scale(p2, -1.0)),
            sign * 0.5 / h,
        )
    };
    let (room_below, room_above) = (t - t0 >= 2.0 * h, t1 - t >= 2.0 * h);
    match (room_below, room_above) {
        (false, _) => ahead(1.0),
        (_, false) => ahead(-1.0),
        _ if kink => scale(add(ahead(1.0), ahead(-1.0)), 0.5),
        _ => scale(sub(f(t + h), f(t - h)), 0.5 / h),
    }
}

fn hermite(p0: Vec3, d0: Vec3, p1: Vec3, d1: Vec3, h: f64, s: f64) -> Vec3 {
    let s2 = s * s;
    let s3 = s2 * s;
    let (a, b, c, d) = (
        2.0 * s3 - 3.0 * s2 + 1.0,
        (s3 - 2.0 * s2 + s) * h,
        -2.0 * s3 + 3.0 * s2,
        (s3 - s2) * h,
    );
    add(
        add(scale(p0, a), scale(d0, b)),
        add(scale(p1, c), scale(d1, d)),
    )
}

/// A cubic B-spline over `[t0, t1]` within `tol` of `f`. `must` are
/// parameters the curve has nodes at (the places `f` is only C1).
pub fn fit_curve(f: &dyn Fn(f64) -> Vec3, t0: f64, t1: f64, tol: f64, must: &[f64]) -> NurbsCurve {
    let mut nodes: Vec<f64> = vec![t0, t1];
    for &m in must {
        if m > t0 && m < t1 {
            nodes.push(m);
        }
    }
    nodes.sort_by(|a, b| a.total_cmp(b));
    nodes.dedup_by(|a, b| (*a - *b).abs() <= 1e-12 * (t1 - t0));

    struct Node {
        t: f64,
        p: Vec3,
        d: Vec3,
    }
    let node = |t: f64, kink: bool| Node {
        t,
        p: f(t),
        d: derivative(f, t, t0, t1, kink),
    };
    let mut out: Vec<Node> = vec![node(nodes[0], true)];
    // Depth-first over the spans, so the nodes come out in order.
    let mut stack: Vec<(Node, usize)> = nodes[1..]
        .iter()
        .rev()
        .map(|&t| (node(t, true), 0))
        .collect();
    while let Some((b, depth)) = stack.pop() {
        let a = out.last().unwrap();
        let h = b.t - a.t;
        let mut worst = 0.0f64;
        for s in [0.25, 0.5, 0.75] {
            let q = hermite(a.p, a.d, b.p, b.d, h, s);
            worst = worst.max(norm(sub(q, f(a.t + s * h))));
        }
        if worst > tol && depth < 24 {
            let mid = node(a.t + 0.5 * h, false);
            stack.push((b, depth + 1));
            stack.push((mid, depth + 1));
        } else {
            out.push(b);
        }
    }

    let nodes: Vec<(f64, Vec3, Vec3)> = out.into_iter().map(|n| (n.t, n.p, n.d)).collect();
    hermite_spline(&nodes)
}

/// The C1 piecewise cubic with the given `(parameter, point, derivative)`
/// at its nodes (two at least, parameters increasing), as a B-spline.
pub fn hermite_spline(nodes: &[(f64, Vec3, Vec3)]) -> NurbsCurve {
    let n = nodes.len() - 1;
    let (t0, t1) = (nodes[0].0, nodes[n].0);
    let mut knots = vec![t0; 4];
    for nd in &nodes[1..n] {
        knots.push(nd.0);
        knots.push(nd.0);
    }
    knots.extend([t1; 4]);
    let mut ctrl: Vec<[f64; 4]> = Vec::with_capacity(2 * n + 2);
    let w = |p: Vec3| [p[0], p[1], p[2], 1.0];
    for (i, &(t, p, d)) in nodes.iter().enumerate() {
        if i > 0 {
            let h = t - nodes[i - 1].0;
            ctrl.push(w(sub(p, scale(d, h / 3.0))));
        }
        if i == 0 || i == n {
            ctrl.push(w(p));
        }
        if i < n {
            let h = nodes[i + 1].0 - t;
            ctrl.push(w(add(p, scale(d, h / 3.0))));
        }
    }
    NurbsCurve {
        degree: 3,
        knots,
        ctrl,
    }
}

/// A C1 cubic through points of a smooth curve (dense enough that three
/// neighbours give its direction), over `[0, 1]` by chord length. As few
/// of the points as keep the others within `tol` become nodes. With
/// `closed` the first point is the last and the curve is smooth there.
pub fn spline_through(points: &[Vec3], tol: f64, closed: bool) -> NurbsCurve {
    let n = points.len();
    assert!(n >= 2, "a curve needs two points");
    let mut s = vec![0.0; n];
    for i in 1..n {
        s[i] = s[i - 1] + norm(sub(points[i], points[i - 1]));
    }
    let total = s[n - 1].max(1e-300);
    for v in s.iter_mut() {
        *v /= total;
    }
    // The derivative of the parabola through three points, at the middle
    // one (`at` = 1) or at an end one.
    let three = |a: (f64, Vec3), b: (f64, Vec3), c: (f64, Vec3), at: usize| -> Vec3 {
        let t = [a.0, b.0, c.0][at];
        let (la, lb, lc) = (
            (2.0 * t - b.0 - c.0) / ((a.0 - b.0) * (a.0 - c.0)),
            (2.0 * t - a.0 - c.0) / ((b.0 - a.0) * (b.0 - c.0)),
            (2.0 * t - a.0 - b.0) / ((c.0 - a.0) * (c.0 - b.0)),
        );
        add(add(scale(a.1, la), scale(b.1, lb)), scale(c.1, lc))
    };
    let at = |i: usize| (s[i], points[i]);
    let slope = |i: usize| -> Vec3 {
        if n == 2 {
            return sub(points[1], points[0]);
        }
        if i > 0 && i < n - 1 {
            three(at(i - 1), at(i), at(i + 1), 1)
        } else if closed {
            // Across the seam: the neighbour on the far side, moved by
            // the whole length.
            let before = (s[n - 2] - 1.0, points[n - 2]);
            let after = (s[1] + 1.0, points[1]);
            if i == 0 {
                three(before, at(0), at(1), 1)
            } else {
                three(at(n - 2), at(n - 1), after, 1)
            }
        } else if i == 0 {
            three(at(0), at(1), at(2), 0)
        } else {
            three(at(n - 3), at(n - 2), at(n - 1), 2)
        }
    };
    let mut keep: Vec<usize> = vec![0];
    let mut stack: Vec<usize> = vec![n - 1];
    while let Some(&b) = stack.last() {
        let a = *keep.last().unwrap();
        let (h, da, db) = (s[b] - s[a], slope(a), slope(b));
        let mut worst = 0.0f64;
        for i in a + 1..b {
            let q = hermite(points[a], da, points[b], db, h, (s[i] - s[a]) / h);
            worst = worst.max(norm(sub(q, points[i])));
        }
        if worst > tol && b - a > 1 {
            stack.push(a + (b - a) / 2);
        } else {
            keep.push(b);
            stack.pop();
        }
    }
    let nodes: Vec<(f64, Vec3, Vec3)> = keep.iter().map(|&i| (s[i], points[i], slope(i))).collect();
    hermite_spline(&nodes)
}

/// The largest distance from `curve` to `f` over a sampling of its spans.
pub fn deviation(curve: &NurbsCurve, f: &dyn Fn(f64) -> Vec3, per_span: usize) -> f64 {
    let mut worst = 0.0f64;
    for k in curve.knots.windows(2) {
        if k[1] <= k[0] {
            continue;
        }
        for i in 0..=per_span {
            let t = k[0] + (k[1] - k[0]) * i as f64 / per_span as f64;
            worst = worst.max(norm(sub(curve_eval(curve, t).0, f(t))));
        }
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_a_helix() {
        let f = |t: f64| [0.05 * t.cos(), 0.05 * t.sin(), 0.01 * t];
        let c = fit_curve(&f, 0.0, 5.0, 1e-8, &[]);
        c.validate().unwrap();
        assert!(deviation(&c, &f, 16) < 2e-8);
        assert!(c.ctrl.len() < 200, "{} control points", c.ctrl.len());
    }

    #[test]
    fn a_cubic_is_its_own_fit() {
        let f = |t: f64| [t, t * t - 0.3 * t * t * t, 2.0 - t];
        let c = fit_curve(&f, -1.0, 2.0, 1e-9, &[]);
        assert_eq!(c.ctrl.len(), 4);
        assert!(deviation(&c, &f, 16) < 1e-9);
    }

    #[test]
    fn through_the_points_of_a_circle() {
        let pts: Vec<Vec3> = (0..=400)
            .map(|i| {
                let a = core::f64::consts::TAU * i as f64 / 400.0;
                [0.05 * a.cos(), 0.05 * a.sin(), 0.0]
            })
            .collect();
        let c = spline_through(&pts, 5e-8, true);
        c.validate().unwrap();
        assert!(c.ctrl.len() < 200, "{} control points", c.ctrl.len());
        for k in 0..2000 {
            let (p, _) = curve_eval(&c, k as f64 / 2000.0);
            assert!((norm(p) - 0.05).abs() < 1e-7, "{}", norm(p) - 0.05);
        }
        // Smooth across the seam.
        let (_, d0) = curve_eval(&c, 0.0);
        let (_, d1) = curve_eval(&c, 1.0);
        assert!(norm(sub(d0, d1)) < 1e-6 * norm(d0));
    }

    #[test]
    fn a_kink_gets_a_node() {
        // Two cubics meeting with a jump in the second derivative.
        let f = |t: f64| {
            if t < 1.0 {
                [t, t * t, 0.0]
            } else {
                [t, 2.0 * t - 1.0 + 3.0 * (t - 1.0).powi(3), 0.0]
            }
        };
        let c = fit_curve(&f, 0.0, 2.0, 1e-9, &[1.0]);
        assert_eq!(c.ctrl.len(), 6);
        assert!(deviation(&c, &f, 16) < 1e-8);
    }
}

/// A bicubic B-spline surface over `[u0, u1] x [v0, v1]` within `tol` of
/// `f`, which must be smooth there and a little beyond (its derivatives
/// are taken by differences across the nodes, the edges included, two
/// thousandths of the range out).
///
/// Piecewise bicubic Hermite on a grid of nodes that is halved, in the
/// direction the error is worse in, wherever the patch strays.
pub fn fit_surface(
    f: &dyn Fn(f64, f64) -> Vec3,
    u: (f64, f64),
    v: (f64, f64),
    tol: f64,
) -> brep_core::ir::NurbsSurface {
    let (hu, hv) = (1e-3 * (u.1 - u.0), 1e-3 * (v.1 - v.0));
    // Value, d/du, d/dv and d2/dudv at a node, by differences of fourth
    // order (exact on the cubics the patch is made of).
    const STENCIL: [(f64, f64); 4] = [
        (-2.0, 1.0 / 12.0),
        (-1.0, -8.0 / 12.0),
        (1.0, 8.0 / 12.0),
        (2.0, -1.0 / 12.0),
    ];
    let node = |a: f64, b: f64| -> [Vec3; 4] {
        let mut out = [f(a, b), [0.0; 3], [0.0; 3], [0.0; 3]];
        for (i, wi) in STENCIL {
            out[1] = add(out[1], scale(f(a + i * hu, b), wi / hu));
            out[2] = add(out[2], scale(f(a, b + i * hv), wi / hv));
            for (j, wj) in STENCIL {
                out[3] = add(
                    out[3],
                    scale(f(a + i * hu, b + j * hv), wi * wj / (hu * hv)),
                );
            }
        }
        out
    };
    // One direction of Hermite: values and derivatives at the ends.
    let h1 = |p0: Vec3, d0: Vec3, p1: Vec3, d1: Vec3, h: f64, s: f64| hermite(p0, d0, p1, d1, h, s);
    let mut us = vec![u.0, u.1];
    let mut vs = vec![v.0, v.1];
    let mut grid: Vec<Vec<[Vec3; 4]>>;
    for _ in 0..12 {
        grid = us
            .iter()
            .map(|&a| vs.iter().map(|&b| node(a, b)).collect())
            .collect();
        let at = |i: usize, j: usize, s: f64, t: f64| -> Vec3 {
            let (du, dv) = (us[i + 1] - us[i], vs[j + 1] - vs[j]);
            // Along v at both u ends, for the value and for d/du.
            let (n00, n01, n10, n11) = (
                grid[i][j],
                grid[i][j + 1],
                grid[i + 1][j],
                grid[i + 1][j + 1],
            );
            let p0 = h1(n00[0], n00[2], n01[0], n01[2], dv, t);
            let p1 = h1(n10[0], n10[2], n11[0], n11[2], dv, t);
            let d0 = h1(n00[1], n00[3], n01[1], n01[3], dv, t);
            let d1 = h1(n10[1], n10[3], n11[1], n11[3], dv, t);
            h1(p0, d0, p1, d1, du, s)
        };
        let mut split_u = vec![false; us.len() - 1];
        let mut split_v = vec![false; vs.len() - 1];
        for i in 0..us.len() - 1 {
            for j in 0..vs.len() - 1 {
                let err = |s: f64, t: f64| {
                    let q = f(
                        us[i] + s * (us[i + 1] - us[i]),
                        vs[j] + t * (vs[j + 1] - vs[j]),
                    );
                    norm(sub(at(i, j, s, t), q))
                };
                let across_u = err(0.5, 0.0)
                    .max(err(0.5, 1.0))
                    .max(err(0.25, 0.5))
                    .max(err(0.75, 0.5));
                let across_v = err(0.0, 0.5)
                    .max(err(1.0, 0.5))
                    .max(err(0.5, 0.25))
                    .max(err(0.5, 0.75));
                let middle = err(0.5, 0.5);
                if across_u.max(across_v).max(middle) <= tol {
                    continue;
                }
                if across_u >= across_v || middle > tol {
                    split_u[i] = true;
                }
                if across_v > across_u || middle > tol {
                    split_v[j] = true;
                }
            }
        }
        if !split_u.contains(&true) && !split_v.contains(&true) {
            break;
        }
        let halve = |xs: &[f64], split: &[bool]| -> Vec<f64> {
            let mut out = vec![xs[0]];
            for (k, w) in xs.windows(2).enumerate() {
                if split[k] {
                    out.push(0.5 * (w[0] + w[1]));
                }
                out.push(w[1]);
            }
            out
        };
        us = halve(&us, &split_u);
        vs = halve(&vs, &split_v);
    }
    let grid: Vec<Vec<[Vec3; 4]>> = us
        .iter()
        .map(|&a| vs.iter().map(|&b| node(a, b)).collect())
        .collect();

    // To control points: first along u (values and d/dv alike), then
    // along v.
    let along = |xs: &[f64], p: &[Vec3], d: &[Vec3]| -> Vec<Vec3> {
        let n = xs.len() - 1;
        let mut out = Vec::with_capacity(2 * n + 2);
        for i in 0..=n {
            if i > 0 {
                out.push(sub(p[i], scale(d[i], (xs[i] - xs[i - 1]) / 3.0)));
            }
            if i == 0 || i == n {
                out.push(p[i]);
            }
            if i < n {
                out.push(add(p[i], scale(d[i], (xs[i + 1] - xs[i]) / 3.0)));
            }
        }
        out
    };
    let knots = |xs: &[f64]| -> Vec<f64> {
        let n = xs.len() - 1;
        let mut k = vec![xs[0]; 4];
        for x in &xs[1..n] {
            k.extend([*x; 2]);
        }
        k.extend([xs[n]; 4]);
        k
    };
    let (nu, nv) = (2 * us.len(), 2 * vs.len());
    // Per v node: control points along u of the value and of d/dv.
    let mut rows: Vec<(Vec<Vec3>, Vec<Vec3>)> = Vec::with_capacity(vs.len());
    for j in 0..vs.len() {
        let pick = |k: usize| -> Vec<Vec3> { (0..us.len()).map(|i| grid[i][j][k]).collect() };
        rows.push((
            along(&us, &pick(0), &pick(1)),
            along(&us, &pick(2), &pick(3)),
        ));
    }
    let mut ctrl = vec![[0.0; 4]; nu * nv];
    for i in 0..nu {
        let p: Vec<Vec3> = rows.iter().map(|r| r.0[i]).collect();
        let d: Vec<Vec3> = rows.iter().map(|r| r.1[i]).collect();
        for (j, c) in along(&vs, &p, &d).into_iter().enumerate() {
            ctrl[i * nv + j] = [c[0], c[1], c[2], 1.0];
        }
    }
    brep_core::ir::NurbsSurface {
        degree_u: 3,
        degree_v: 3,
        nctrl_u: nu,
        nctrl_v: nv,
        knots_u: knots(&us),
        knots_v: knots(&vs),
        ctrl,
    }
}

#[cfg(test)]
mod surface_tests {
    use super::*;
    use brep_core::nurbs::surface_eval;

    fn worst(
        s: &brep_core::ir::NurbsSurface,
        f: &dyn Fn(f64, f64) -> Vec3,
        u: (f64, f64),
        v: (f64, f64),
    ) -> f64 {
        let mut w = 0.0f64;
        for i in 0..=60 {
            for j in 0..=60 {
                let (a, b) = (
                    u.0 + (u.1 - u.0) * i as f64 / 60.0,
                    v.0 + (v.1 - v.0) * j as f64 / 60.0,
                );
                w = w.max(norm(sub(surface_eval(s, a, b).0, f(a, b))));
            }
        }
        w
    }

    #[test]
    fn a_bicubic_is_its_own_fit() {
        let f = |u: f64, v: f64| [u, v, u * u * u - 2.0 * u * v * v + 0.5 * v * v * v * u];
        let s = fit_surface(&f, (0.0, 1.0), (-1.0, 2.0), 1e-9);
        s.validate().unwrap();
        assert_eq!((s.nctrl_u, s.nctrl_v), (4, 4));
        assert!(worst(&s, &f, (0.0, 1.0), (-1.0, 2.0)) < 1e-9);
    }

    #[test]
    fn a_blend_of_arcs() {
        // Between a quarter circle of radius 50 mm and one of 80 mm, 30 mm
        // higher and turned.
        let f = |u: f64, v: f64| {
            let a = [0.05 * u.cos(), 0.05 * u.sin(), 0.0];
            let b = [0.08 * (u + 0.3).cos(), 0.08 * (u + 0.3).sin(), 0.03];
            let bulge = 0.01 * (core::f64::consts::PI * v).sin();
            [
                a[0] + v * (b[0] - a[0]) + bulge * u.cos(),
                a[1] + v * (b[1] - a[1]) + bulge * u.sin(),
                a[2] + v * (b[2] - a[2]),
            ]
        };
        let s = fit_surface(&f, (0.0, 1.5), (0.0, 1.0), 5e-8);
        s.validate().unwrap();
        let w = worst(&s, &f, (0.0, 1.5), (0.0, 1.0));
        assert!(w < 1e-7, "{w}");
        assert!(s.ctrl.len() < 3000, "{} control points", s.ctrl.len());
    }
}
