//! Recognizing a circle or an ellipse in the points of a curve.
//!
//! A vertical wall cuts a lying cylinder in an ellipse and a sphere in a
//! circle, a plane cuts them likewise; the edge there is that curve, not
//! a spline near it. Rather than a case per pair of surfaces, the points
//! themselves are asked: in one plane, and on one conic in it.

use brep_core::math::{Frame, Vec3, add, cross, dot, norm, normalize, scale, sub};
use brep_core::topo::Curve3;
use core::f64::consts::TAU;

/// Eigenvalues and eigenvectors (columns) of a symmetric matrix, by
/// Jacobi rotations.
#[allow(clippy::needless_range_loop)]
fn eigen<const N: usize>(mut a: [[f64; N]; N]) -> ([f64; N], [[f64; N]; N]) {
    let mut v = [[0.0; N]; N];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..64 {
        let mut off = 0.0;
        for i in 0..N {
            for j in i + 1..N {
                off += a[i][j] * a[i][j];
            }
        }
        if off < 1e-300 {
            break;
        }
        for p in 0..N {
            for q in p + 1..N {
                if a[p][q] == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..N {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..N {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    let mut values = [0.0; N];
    for i in 0..N {
        values[i] = a[i][i];
    }
    (values, v)
}

fn wrap(a: f64) -> f64 {
    let r = a.rem_euclid(TAU);
    if r >= TAU { 0.0 } else { r }
}

/// The circle or ellipse through `points` (in order along the curve, six
/// at least, the first repeated at the end if the curve is closed), with
/// the parameters of the first and the last, when all of them are within
/// `tol` of one.
pub fn conic_through(points: &[Vec3], tol: f64) -> Option<(Curve3, f64, f64)> {
    let n = points.len();
    if n < 6 {
        return None;
    }
    let closed = norm(sub(points[0], points[n - 1])) <= tol;
    // The plane: through the centroid, across the direction the points
    // spread least in.
    let mut centroid = [0.0; 3];
    for p in points {
        centroid = add(centroid, scale(*p, 1.0 / n as f64));
    }
    let mut spread = [[0.0; 3]; 3];
    for p in points {
        let d = sub(*p, centroid);
        for i in 0..3 {
            for j in 0..3 {
                spread[i][j] += d[i] * d[j];
            }
        }
    }
    let (values, vectors) = eigen(spread);
    let least = (0..3).min_by(|a, b| values[*a].total_cmp(&values[*b]))?;
    let most = (0..3).max_by(|a, b| values[*a].total_cmp(&values[*b]))?;
    let column = |k: usize| [vectors[0][k], vectors[1][k], vectors[2][k]];
    let normal = normalize(column(least));
    let e1 = normalize(column(most));
    let e2 = cross(normal, e1);
    if points
        .iter()
        .any(|p| dot(sub(*p, centroid), normal).abs() > tol)
    {
        return None;
    }
    let size = (values[most] / n as f64).sqrt();
    if size.is_nan() || size <= 0.0 {
        return None;
    }
    // In the plane, lengths in units of the points' spread.
    let flat: Vec<[f64; 2]> = points
        .iter()
        .map(|p| {
            let d = sub(*p, centroid);
            [dot(d, e1) / size, dot(d, e2) / size]
        })
        .collect();
    let mut m = [[0.0; 6]; 6];
    for q in &flat {
        let row = [q[0] * q[0], q[0] * q[1], q[1] * q[1], q[0], q[1], 1.0];
        for i in 0..6 {
            for j in 0..6 {
                m[i][j] += row[i] * row[j];
            }
        }
    }
    let (values, vectors) = eigen(m);
    let k = (0..6).min_by(|a, b| values[*a].total_cmp(&values[*b]))?;
    let [a, b, c, d, e, f] = core::array::from_fn(|i| vectors[i][k]);
    let disc = b * b - 4.0 * a * c;
    if disc.is_nan() || disc >= 0.0 {
        return None;
    }
    let centre = [(2.0 * c * d - b * e) / disc, (2.0 * a * e - b * d) / disc];
    let f0 = f + 0.5 * (d * centre[0] + e * centre[1]);
    let (lambda, axes) = eigen([[a, 0.5 * b], [0.5 * b, c]]);
    let half = [(-f0 / lambda[0]).sqrt(), (-f0 / lambda[1]).sqrt()];
    if !(half[0].is_finite() && half[1].is_finite() && half[0] > 0.0 && half[1] > 0.0) {
        return None;
    }
    // The longer half-axis first.
    let (i, j) = if half[0] >= half[1] { (0, 1) } else { (1, 0) };
    let dir = |k: usize| [axes[0][k], axes[1][k]];
    let (major, minor) = (half[i] * size, half[j] * size);
    let lift = |q: [f64; 2]| add(scale(e1, q[0]), scale(e2, q[1]));
    let origin = add(centroid, scale(lift(centre), size));
    let x = normalize(lift(dir(i)));
    let mut y = normalize(sub(lift(dir(j)), scale(x, dot(lift(dir(j)), x))));
    let angle = |p: Vec3, y: Vec3| {
        let w = sub(p, origin);
        (dot(w, y) / minor).atan2(dot(w, x) / major)
    };
    for p in points {
        let t = angle(*p, y);
        let on = add(
            origin,
            add(scale(x, major * t.cos()), scale(y, minor * t.sin())),
        );
        if norm(sub(on, *p)) > tol {
            return None;
        }
    }
    // Run the way the points do: two of them on the way come in their
    // order, before the last.
    let (first, one, two, last) = (points[0], points[n / 3], points[2 * n / 3], points[n - 1]);
    let ahead = |y: Vec3| {
        let t0 = angle(first, y);
        let to_last = if closed {
            TAU
        } else {
            wrap(angle(last, y) - t0)
        };
        let in_order = wrap(angle(one, y) - t0) < wrap(angle(two, y) - t0)
            && wrap(angle(two, y) - t0) <= to_last;
        (t0, to_last, in_order)
    };
    let (mut t0, mut to_last, in_order) = ahead(y);
    if !in_order {
        y = scale(y, -1.0);
        (t0, to_last, _) = ahead(y);
    }
    let frame = Frame {
        origin,
        x,
        y,
        z: cross(x, y),
    };
    let curve = if (major - minor).abs() <= 1e-12 * major {
        Curve3::Circle {
            frame,
            radius: major,
        }
    } else {
        Curve3::Ellipse {
            frame,
            a: major,
            b: minor,
        }
    };
    Some((curve, t0, t0 + to_last))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn along(curve: &Curve3, t0: f64, t1: f64, points: &[Vec3]) {
        assert!(t1 > t0);
        assert!(norm(sub(curve.eval(t0), points[0])) < 1e-12);
        assert!(norm(sub(curve.eval(t1), points[points.len() - 1])) < 1e-12);
        let mid = points[points.len() / 2];
        let near = (0..=1000)
            .map(|k| norm(sub(curve.eval(t0 + (t1 - t0) * k as f64 / 1000.0), mid)))
            .fold(f64::INFINITY, f64::min);
        assert!(near < 1e-3 * norm(sub(points[0], mid)).max(1e-3), "{near}");
    }

    #[test]
    fn a_tilted_ellipse_either_way_round() {
        let frame = Frame::from_axis_ref([0.1, -0.2, 0.05], [0.3, 0.2, 1.0], [1.0, 0.5, 0.0]);
        for (from, to) in [(0.4, 2.9), (2.9, 0.4), (5.5, 7.0), (-1.0, -4.5)] {
            let pts: Vec<Vec3> = (0..=20)
                .map(|i| {
                    let t: f64 = from + (to - from) * i as f64 / 20.0;
                    frame.to_world([0.03 * t.cos(), 0.012 * t.sin(), 0.0])
                })
                .collect();
            let (curve, t0, t1) = conic_through(&pts, 1e-10).expect("an ellipse");
            let Curve3::Ellipse { a, b, .. } = curve else {
                panic!("not an ellipse")
            };
            assert!((a - 0.03).abs() < 1e-12 && (b - 0.012).abs() < 1e-12);
            along(&curve, t0, t1, &pts);
        }
    }

    #[test]
    fn a_whole_circle() {
        let frame = Frame::from_axis_ref([0.0, 0.0, 0.3], [0.0, 1.0, 0.2], [1.0, 0.0, 0.0]);
        let pts: Vec<Vec3> = (0..=24)
            .map(|i| {
                let t = 1.0 - TAU * i as f64 / 24.0;
                frame.to_world([0.02 * t.cos(), 0.02 * t.sin(), 0.0])
            })
            .collect();
        let (curve, t0, t1) = conic_through(&pts, 1e-10).expect("a circle");
        assert!(matches!(curve, Curve3::Circle { radius, .. } if (radius - 0.02).abs() < 1e-12));
        assert!((t1 - t0 - TAU).abs() < 1e-12);
        assert!(norm(sub(curve.eval(t0 + 0.25 * TAU), pts[6])) < 1e-12);
    }

    #[test]
    fn not_a_conic() {
        // A helix is in no plane; a cubic in a plane is no conic.
        let helix: Vec<Vec3> = (0..=20)
            .map(|i| {
                let t = 0.2 * i as f64;
                [0.05 * t.cos(), 0.05 * t.sin(), 0.01 * t]
            })
            .collect();
        assert!(conic_through(&helix, 1e-10).is_none());
        let cubic: Vec<Vec3> = (0..=20)
            .map(|i| {
                let t = 0.1 * i as f64 - 1.0;
                [0.05 * t, 0.05 * t * t * t, 0.0]
            })
            .collect();
        assert!(conic_through(&cubic, 1e-10).is_none());
        // Part of a parabola.
        let parabola: Vec<Vec3> = (0..=20)
            .map(|i| {
                let t = 0.1 * i as f64 - 1.0;
                [0.05 * t, 0.05 * t * t, 0.0]
            })
            .collect();
        assert!(conic_through(&parabola, 1e-10).is_none());
    }
}
