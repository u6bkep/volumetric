//! The curve in the plan along which two height functions are equal:
//! the zero set of their difference, traced inside a rectangle.
//!
//! The zero set of a scalar function on the plane has no topology to get
//! wrong that a fine enough grid does not show: signs at the grid nodes
//! give the crossings of the grid lines, each crossing is polished onto
//! the zero set, and the polylines between them are filled in until no
//! chord is further than the tolerance from the curve.

use sketch_regions_core::geom::{add, cross, dist, dot, norm, scale, sub};
use sketch_regions_core::{Curve, P2, Piece};
use std::collections::HashMap;

/// Ids of pieces that stand for a curve that is neither a line nor an
/// arc begin with this; consecutive pieces of one id are one curve.
pub const FREE: &str = "free:";

/// A root of `g` on the segment `a..b`, whose ends have opposite signs.
fn root_on(g: &dyn Fn(P2) -> Option<f64>, a: P2, b: P2, ga: f64) -> P2 {
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    let at = |s: f64| add(a, scale(sub(b, a), s));
    for _ in 0..52 {
        let s = 0.5 * (lo + hi);
        match g(at(s)) {
            Some(v) if (v < 0.0) == (ga < 0.0) => lo = s,
            Some(_) => hi = s,
            None => break,
        }
    }
    at(0.5 * (lo + hi))
}

/// `p` moved onto the zero set along the gradient.
fn polish(g: &dyn Fn(P2) -> Option<f64>, mut p: P2, h: f64) -> Option<P2> {
    for _ in 0..8 {
        let v = g(p)?;
        let gx = (g([p[0] + h, p[1]])? - g([p[0] - h, p[1]])?) / (2.0 * h);
        let gy = (g([p[0], p[1] + h])? - g([p[0], p[1] - h])?) / (2.0 * h);
        let n2 = gx * gx + gy * gy;
        if n2 == 0.0 {
            return None;
        }
        let step = [-v * gx / n2, -v * gy / n2];
        p = add(p, step);
        if norm(step) < 1e-13 {
            break;
        }
    }
    Some(p)
}

fn fill(g: &dyn Fn(P2) -> Option<f64>, a: P2, b: P2, tol: f64, depth: usize, out: &mut Vec<P2>) {
    let mid = scale(add(a, b), 0.5);
    let h = (1e-3 * dist(a, b)).max(1e-9);
    match polish(g, mid, h) {
        Some(m) if depth < 16 && dist(m, mid) > tol && dist(m, mid) < dist(a, b) => {
            fill(g, a, m, tol, depth + 1, out);
            fill(g, m, b, tol, depth + 1, out);
        }
        _ => out.push(b),
    }
}

/// The zero set of `g` inside `rect` (`[min_x, min_y, max_x, max_y]`) as
/// polylines whose points are on it and whose chords are within `tol` of
/// it. `pitch` is the grid the search starts from: parts of the curve
/// that enter and leave a grid square through the same side are not
/// seen. A closed curve comes back with its first point repeated.
pub fn trace_zero(
    g: &dyn Fn(P2) -> Option<f64>,
    rect: [f64; 4],
    pitch: f64,
    tol: f64,
) -> Vec<Vec<P2>> {
    let (w, h) = (rect[2] - rect[0], rect[3] - rect[1]);
    if !(w > 0.0 && h > 0.0) {
        return Vec::new();
    }
    let nx = ((w / pitch).ceil() as usize).clamp(2, 1024);
    let ny = ((h / pitch).ceil() as usize).clamp(2, 1024);
    let node = |i: usize, j: usize| -> P2 {
        [
            rect[0] + w * i as f64 / nx as f64,
            rect[1] + h * j as f64 / ny as f64,
        ]
    };
    let mut vals: Vec<Option<f64>> = Vec::with_capacity((nx + 1) * (ny + 1));
    for j in 0..=ny {
        for i in 0..=nx {
            vals.push(g(node(i, j)));
        }
    }
    let val = |i: usize, j: usize| vals[j * (nx + 1) + i];
    // Crossings of the grid lines: key (i, j, 0) is the edge from node
    // (i, j) along x, (i, j, 1) along y.
    let mut points: Vec<P2> = Vec::new();
    let mut crossing: HashMap<(usize, usize, u8), usize> = HashMap::new();
    for j in 0..=ny {
        for i in 0..=nx {
            for (dir, (i2, j2)) in [(0u8, (i + 1, j)), (1u8, (i, j + 1))] {
                if i2 > nx || j2 > ny {
                    continue;
                }
                let (Some(a), Some(b)) = (val(i, j), val(i2, j2)) else {
                    continue;
                };
                if (a < 0.0) != (b < 0.0) {
                    crossing.insert((i, j, dir), points.len());
                    points.push(root_on(g, node(i, j), node(i2, j2), a));
                }
            }
        }
    }
    // Segments per grid square.
    let mut links: Vec<Vec<usize>> = vec![Vec::new(); points.len()];
    let mut link = |a: usize, b: usize| {
        links[a].push(b);
        links[b].push(a);
    };
    for j in 0..ny {
        for i in 0..nx {
            // Bottom, right, top, left.
            let sides = [(i, j, 0u8), (i + 1, j, 1), (i, j + 1, 0), (i, j, 1)];
            let hit: Vec<Option<usize>> = sides.iter().map(|k| crossing.get(k).copied()).collect();
            let found: Vec<usize> = hit.iter().flatten().copied().collect();
            match found.len() {
                2 => link(found[0], found[1]),
                4 => {
                    // A saddle: the centre's sign says which corners are
                    // joined.
                    let centre = g(scale(add(node(i, j), node(i + 1, j + 1)), 0.5));
                    let corner = val(i, j);
                    let same =
                        matches!((centre, corner), (Some(c), Some(k)) if (c < 0.0) == (k < 0.0));
                    let (b, r, t, l) = (found[0], found[1], found[2], found[3]);
                    if same {
                        // The near corner joins the centre: cut off the
                        // other two.
                        link(b, r);
                        link(t, l);
                    } else {
                        link(b, l);
                        link(r, t);
                    }
                }
                _ => {}
            }
        }
    }
    // Chains: from loose ends first, then what is left is closed.
    let mut used = vec![false; points.len()];
    let mut out: Vec<Vec<P2>> = Vec::new();
    let walk = |start: usize, used: &mut Vec<bool>| -> Vec<usize> {
        let mut chain = vec![start];
        used[start] = true;
        let mut cur = start;
        while let Some(&next) = links[cur].iter().find(|&&n| !used[n]) {
            used[next] = true;
            chain.push(next);
            cur = next;
        }
        if chain.len() > 2 && links[cur].contains(&start) {
            chain.push(start);
        }
        chain
    };
    let mut chains: Vec<Vec<usize>> = Vec::new();
    for s in 0..points.len() {
        if !used[s] && links[s].len() == 1 {
            chains.push(walk(s, &mut used));
        }
    }
    for s in 0..points.len() {
        if !used[s] && !links[s].is_empty() {
            chains.push(walk(s, &mut used));
        }
    }
    for chain in chains {
        if chain.len() < 2 {
            continue;
        }
        let mut line = vec![points[chain[0]]];
        for k in chain.windows(2) {
            fill(g, points[k[0]], points[k[1]], tol, 0, &mut line);
        }
        out.push(line);
    }
    out
}

/// A traced polyline as boundary pieces: one line or arc when the points
/// lie on one within `exact`, otherwise short lines under a [`FREE`] id.
pub fn pieces_of(points: &[P2], name: &str, exact: f64) -> Vec<Piece> {
    let n = points.len();
    if n < 2 {
        return Vec::new();
    }
    let (first, last) = (points[0], points[n - 1]);
    let closed = n > 3 && dist(first, last) <= exact;
    if !closed {
        let d = sub(last, first);
        let len = norm(d);
        if len > 0.0
            && points
                .iter()
                .all(|p| (cross(d, sub(*p, first)) / len).abs() <= exact)
        {
            return vec![Piece {
                id: name.to_string(),
                curve: Curve::Line { a: first, b: last },
            }];
        }
    }
    // The circle through three spread points.
    let (a, b, c) = if closed {
        (points[0], points[n / 3], points[2 * n / 3])
    } else {
        (first, points[n / 2], last)
    };
    let (ab, ac) = (sub(b, a), sub(c, a));
    let den = 2.0 * cross(ab, ac);
    if den.abs() > 1e-300 {
        let (la, lc) = (dot(ab, ab), dot(ac, ac));
        let centre = add(
            a,
            [
                (ac[1] * la - ab[1] * lc) / den,
                (ab[0] * lc - ac[0] * la) / den,
            ],
        );
        let r = dist(centre, a);
        if r.is_finite() && points.iter().all(|p| (dist(*p, centre) - r).abs() <= exact) {
            let curve = if closed {
                Curve::circle(centre, r)
            } else {
                Curve::arc_from_ends(centre, first, last, den > 0.0)
            };
            return vec![Piece {
                id: name.to_string(),
                curve,
            }];
        }
    }
    points
        .windows(2)
        .filter(|w| dist(w[0], w[1]) > 0.0)
        .map(|w| Piece {
            id: format!("{FREE}{name}"),
            curve: Curve::Line { a: w[0], b: w[1] },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_circle_comes_back_as_one() {
        let g = |p: P2| Some(p[0] * p[0] + p[1] * p[1] - 0.04 * 0.04);
        let lines = trace_zero(&g, [-0.1, -0.1, 0.1, 0.1], 0.004, 1e-7);
        assert_eq!(lines.len(), 1);
        let line = &lines[0];
        assert!(dist(line[0], line[line.len() - 1]) < 1e-12);
        for p in line {
            assert!((norm(*p) - 0.04).abs() < 1e-12);
        }
        for w in line.windows(2) {
            let mid = scale(add(w[0], w[1]), 0.5);
            assert!(0.04 - norm(mid) < 2e-7, "chord {}", 0.04 - norm(mid));
        }
        let pieces = pieces_of(line, "c", 1e-9);
        assert_eq!(pieces.len(), 1);
        assert!(matches!(pieces[0].curve, Curve::Arc { r, .. } if (r - 0.04).abs() < 1e-12));
    }

    #[test]
    fn a_line_and_a_free_curve() {
        let g = |p: P2| Some(p[0] + 2.0 * p[1] - 0.01);
        let lines = trace_zero(&g, [0.0, -0.05, 0.1, 0.05], 0.01, 1e-7);
        assert_eq!(lines.len(), 1);
        let pieces = pieces_of(&lines[0], "l", 1e-9);
        assert_eq!(pieces.len(), 1);
        assert!(matches!(pieces[0].curve, Curve::Line { .. }));

        let g = |p: P2| Some(p[1] - 0.02 * (30.0 * p[0]).sin());
        let lines = trace_zero(&g, [0.0, -0.05, 0.3, 0.05], 0.005, 1e-7);
        assert_eq!(lines.len(), 1);
        let pieces = pieces_of(&lines[0], "s", 1e-9);
        assert!(pieces.len() > 50);
        assert!(pieces.iter().all(|p| p.id == "free:s"));
        for p in &pieces {
            let m = p.curve.point(0.5);
            assert!((m[1] - 0.02 * (30.0 * m[0]).sin()).abs() < 3e-7);
        }
    }

    #[test]
    fn two_branches_and_a_hole_in_the_domain() {
        // A hyperbola, the function undefined in a disc between the
        // branches.
        let g = |p: P2| (norm(p) > 0.005).then(|| p[0] * p[0] - p[1] * p[1] - 0.0004);
        let lines = trace_zero(&g, [-0.1, -0.1, 0.1, 0.1], 0.004, 1e-7);
        assert_eq!(lines.len(), 2);
    }
}
