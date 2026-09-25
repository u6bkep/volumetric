//! Exact 2D curve pieces: line segments and circular arcs.
//!
//! An arc is `centre + r * (cos, sin)(a0 + sweep * t)` for `t` in `[0, 1]`;
//! the sign of `sweep` is the turning direction (positive = counter-
//! clockwise), and a full circle is `|sweep| = 2π`.

use std::f64::consts::{PI, TAU};

pub type P2 = [f64; 2];

#[inline]
pub fn sub(a: P2, b: P2) -> P2 {
    [a[0] - b[0], a[1] - b[1]]
}
#[inline]
pub fn add(a: P2, b: P2) -> P2 {
    [a[0] + b[0], a[1] + b[1]]
}
#[inline]
pub fn scale(a: P2, s: f64) -> P2 {
    [a[0] * s, a[1] * s]
}
#[inline]
pub fn dot(a: P2, b: P2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
#[inline]
pub fn cross(a: P2, b: P2) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
#[inline]
pub fn norm(a: P2) -> f64 {
    a[0].hypot(a[1])
}
#[inline]
pub fn dist(a: P2, b: P2) -> f64 {
    norm(sub(a, b))
}

/// Wrap an angle difference into `[0, 2π)`.
#[inline]
pub fn wrap_pos(a: f64) -> f64 {
    let r = a.rem_euclid(TAU);
    if r >= TAU { 0.0 } else { r }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Curve {
    Line { a: P2, b: P2 },
    Arc { c: P2, r: f64, a0: f64, sweep: f64 },
}

impl Curve {
    /// A circle as a full counter-clockwise arc starting at angle 0.
    pub fn circle(c: P2, r: f64) -> Curve {
        Curve::Arc { c, r, a0: 0.0, sweep: TAU }
    }

    /// An arc through `start` and `end` about `c`, turning the given way.
    /// If the ends coincide the arc is a full circle.
    pub fn arc_from_ends(c: P2, start: P2, end: P2, ccw: bool) -> Curve {
        let r = 0.5 * (dist(start, c) + dist(end, c));
        let a0 = (start[1] - c[1]).atan2(start[0] - c[0]);
        let a1 = (end[1] - c[1]).atan2(end[0] - c[0]);
        let mut sweep = if ccw { wrap_pos(a1 - a0) } else { -wrap_pos(a0 - a1) };
        if sweep.abs() < 1e-12 && dist(start, end) < 1e-9 * (1.0 + r) {
            sweep = if ccw { TAU } else { -TAU };
        }
        Curve::Arc { c, r, a0, sweep }
    }

    pub fn start(&self) -> P2 {
        self.point(0.0)
    }

    pub fn end(&self) -> P2 {
        self.point(1.0)
    }

    pub fn point(&self, t: f64) -> P2 {
        match *self {
            Curve::Line { a, b } => add(a, scale(sub(b, a), t)),
            Curve::Arc { c, r, a0, sweep } => {
                let th = a0 + sweep * t;
                [c[0] + r * th.cos(), c[1] + r * th.sin()]
            }
        }
    }

    /// Unit tangent in the direction of increasing `t`.
    pub fn tangent(&self, t: f64) -> P2 {
        match *self {
            Curve::Line { a, b } => {
                let d = sub(b, a);
                let n = norm(d);
                if n > 0.0 { scale(d, 1.0 / n) } else { [1.0, 0.0] }
            }
            Curve::Arc { a0, sweep, .. } => {
                let th = a0 + sweep * t;
                let s = if sweep >= 0.0 { 1.0 } else { -1.0 };
                [-th.sin() * s, th.cos() * s]
            }
        }
    }

    /// Signed curvature along increasing `t` (positive turns left).
    pub fn curvature(&self) -> f64 {
        match *self {
            Curve::Line { .. } => 0.0,
            Curve::Arc { r, sweep, .. } => {
                if sweep >= 0.0 { 1.0 / r } else { -1.0 / r }
            }
        }
    }

    pub fn length(&self) -> f64 {
        match *self {
            Curve::Line { a, b } => dist(a, b),
            Curve::Arc { r, sweep, .. } => r * sweep.abs(),
        }
    }

    pub fn reversed(&self) -> Curve {
        match *self {
            Curve::Line { a, b } => Curve::Line { a: b, b: a },
            Curve::Arc { c, r, a0, sweep } => Curve::Arc { c, r, a0: a0 + sweep, sweep: -sweep },
        }
    }

    /// The sub-curve for `t` in `[t0, t1]`.
    pub fn sub(&self, t0: f64, t1: f64) -> Curve {
        match *self {
            Curve::Line { .. } => Curve::Line { a: self.point(t0), b: self.point(t1) },
            Curve::Arc { c, r, a0, sweep } => Curve::Arc { c, r, a0: a0 + sweep * t0, sweep: sweep * (t1 - t0) },
        }
    }

    /// Parameter of the arc angle `th` if it lies on the arc (within `eps`
    /// radians of its ends), else `None`.
    pub fn arc_param(&self, th: f64, eps: f64) -> Option<f64> {
        let Curve::Arc { a0, sweep, .. } = *self else { return None };
        if sweep >= 0.0 {
            let d = wrap_pos(th - a0);
            if d <= sweep + eps {
                return Some((d / sweep).clamp(0.0, 1.0));
            }
            if d >= TAU - eps {
                return Some(0.0);
            }
            None
        } else {
            let d = wrap_pos(a0 - th);
            if d <= -sweep + eps {
                return Some((d / -sweep).clamp(0.0, 1.0));
            }
            if d >= TAU - eps {
                return Some(0.0);
            }
            None
        }
    }

    /// Parameter of the point on the curve nearest `p` (clamped to the
    /// piece), and the unsigned distance.
    pub fn nearest(&self, p: P2) -> (f64, f64) {
        match *self {
            Curve::Line { a, b } => {
                let d = sub(b, a);
                let l2 = dot(d, d);
                let t = if l2 > 0.0 { (dot(sub(p, a), d) / l2).clamp(0.0, 1.0) } else { 0.0 };
                (t, dist(p, self.point(t)))
            }
            Curve::Arc { c, r, .. } => {
                let th = (p[1] - c[1]).atan2(p[0] - c[0]);
                if let Some(t) = self.arc_param(th, 0.0) {
                    return (t, (dist(p, c) - r).abs());
                }
                let d0 = dist(p, self.start());
                let d1 = dist(p, self.end());
                if d0 <= d1 { (0.0, d0) } else { (1.0, d1) }
            }
        }
    }

    /// Axis-aligned bounding box `[min_x, min_y, max_x, max_y]`.
    pub fn bbox(&self) -> [f64; 4] {
        match *self {
            Curve::Line { a, b } => [a[0].min(b[0]), a[1].min(b[1]), a[0].max(b[0]), a[1].max(b[1])],
            Curve::Arc { c, r, .. } => {
                let s = self.start();
                let e = self.end();
                let mut bb = [s[0].min(e[0]), s[1].min(e[1]), s[0].max(e[0]), s[1].max(e[1])];
                for (k, th) in [0.0, 0.5 * PI, PI, 1.5 * PI].into_iter().enumerate() {
                    if self.arc_param(th, 0.0).is_some() {
                        let q = [c[0] + r * th.cos(), c[1] + r * th.sin()];
                        let _ = k;
                        bb[0] = bb[0].min(q[0]);
                        bb[1] = bb[1].min(q[1]);
                        bb[2] = bb[2].max(q[0]);
                        bb[3] = bb[3].max(q[1]);
                    }
                }
                bb
            }
        }
    }

    /// Signed area between the chord and the curve (the circular segment),
    /// positive when the piece turns left. Zero for lines.
    pub fn segment_area(&self) -> f64 {
        match *self {
            Curve::Line { .. } => 0.0,
            Curve::Arc { r, sweep, .. } => 0.5 * r * r * (sweep - sweep.sin()),
        }
    }

    /// Winding angle of this piece about `p` (the angle the piece subtends
    /// as it is traversed, with full turns counted).
    pub fn winding(&self, p: P2) -> f64 {
        let a = sub(self.start(), p);
        let b = sub(self.end(), p);
        let chord = cross(a, b).atan2(dot(a, b));
        match *self {
            Curve::Line { .. } => chord,
            Curve::Arc { c, r, sweep, .. } => {
                if sweep.abs() >= TAU - 1e-12 {
                    return if dist(p, c) < r { sweep.signum() * TAU } else { 0.0 };
                }
                let inside_circle = dist(p, c) < r;
                if !inside_circle {
                    return chord;
                }
                // The chord's line cuts the circle in two; the arc lies
                // entirely on one side (the right when it turns left), and
                // the segment between chord and arc is that side of the disc.
                let left = cross(sub(self.end(), self.start()), sub(p, self.start())) > 0.0;
                let enclosed = if sweep > 0.0 { !left } else { left };
                if enclosed { chord + sweep.signum() * TAU } else { chord }
            }
        }
    }

    /// Intersections of the horizontal line `y = y0` with this piece,
    /// as `(x, t)` pairs.
    pub fn horizontal_hits(&self, y0: f64) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        match *self {
            Curve::Line { a, b } => {
                let dy = b[1] - a[1];
                if dy != 0.0 {
                    let t = (y0 - a[1]) / dy;
                    if (0.0..=1.0).contains(&t) {
                        out.push((a[0] + (b[0] - a[0]) * t, t));
                    }
                }
            }
            Curve::Arc { c, r, .. } => {
                let dy = y0 - c[1];
                if dy.abs() <= r {
                    let dx = (r * r - dy * dy).max(0.0).sqrt();
                    for x in [c[0] - dx, c[0] + dx] {
                        let th = dy.atan2(x - c[0]);
                        if let Some(t) = self.arc_param(th, 0.0) {
                            out.push((x, t));
                        }
                        if dx == 0.0 {
                            break;
                        }
                    }
                }
            }
        }
        out
    }

    /// Intersections with the infinite line through `o` along `d`, as
    /// `(s, t)` with the line point `o + s d` and this piece's parameter.
    pub fn line_hits(&self, o: P2, d: P2) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        match *self {
            Curve::Line { a, b } => {
                let e = sub(b, a);
                let den = cross(d, e);
                if den.abs() > 1e-300 {
                    let w = sub(a, o);
                    let s = cross(w, e) / den;
                    let t = cross(w, d) / den;
                    if (-1e-9..=1.0 + 1e-9).contains(&t) {
                        out.push((s, t.clamp(0.0, 1.0)));
                    }
                }
            }
            Curve::Arc { c, r, .. } => {
                let dd = dot(d, d);
                if dd > 0.0 {
                    let w = sub(o, c);
                    let b = dot(w, d) / dd;
                    let cc = (dot(w, w) - r * r) / dd;
                    let disc = b * b - cc;
                    if disc >= 0.0 {
                        let q = disc.sqrt();
                        for s in [-b - q, -b + q] {
                            let p = add(o, scale(d, s));
                            let th = (p[1] - c[1]).atan2(p[0] - c[0]);
                            if let Some(t) = self.arc_param(th, 1e-9) {
                                out.push((s, t));
                            }
                            if q == 0.0 {
                                break;
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

/// Intersection parameters `(t_a, t_b)` of two pieces, including touching
/// ends. `eps` is the absolute distance tolerance.
pub fn intersect(a: &Curve, b: &Curve, eps: f64) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    match (a, b) {
        (Curve::Line { a: p, b: q }, Curve::Line { a: r, b: s }) => {
            let d1 = sub(*q, *p);
            let d2 = sub(*s, *r);
            let den = cross(d1, d2);
            let l1 = norm(d1).max(1e-300);
            let l2 = norm(d2).max(1e-300);
            if den.abs() > 1e-12 * l1 * l2 {
                let w = sub(*r, *p);
                let t = cross(w, d2) / den;
                let u = cross(w, d1) / den;
                let e1 = eps / l1;
                let e2 = eps / l2;
                if (-e1..=1.0 + e1).contains(&t) && (-e2..=1.0 + e2).contains(&u) {
                    out.push((t.clamp(0.0, 1.0), u.clamp(0.0, 1.0)));
                }
            } else {
                // Parallel: overlapping collinear pieces meet at their
                // shared end points at most (overlaps are left to the caller).
                for (t, pa) in [(0.0, *p), (1.0, *q)] {
                    let (u, d) = b.nearest(pa);
                    if d <= eps {
                        out.push((t, u));
                    }
                }
                for (u, pb) in [(0.0, *r), (1.0, *s)] {
                    let (t, d) = a.nearest(pb);
                    if d <= eps {
                        out.push((t, u));
                    }
                }
            }
        }
        (Curve::Line { a: p, b: q }, Curve::Arc { .. }) => {
            let d = sub(*q, *p);
            let l = norm(d);
            if l > 0.0 {
                for (s, t) in b.line_hits(*p, d) {
                    let e = eps / l;
                    if (-e..=1.0 + e).contains(&s) {
                        out.push((s.clamp(0.0, 1.0), t));
                    }
                }
            }
            // Tangent touches slip through the discriminant; catch them by
            // the nearest-point test at the arc's ends and the line's ends.
            for (t, pa) in [(0.0, *p), (1.0, *q)] {
                let (u, dd) = b.nearest(pa);
                if dd <= eps {
                    out.push((t, u));
                }
            }
            for u in [0.0, 1.0] {
                let pb = b.point(u);
                let (t, dd) = a.nearest(pb);
                if dd <= eps {
                    out.push((t, u));
                }
            }
        }
        (Curve::Arc { .. }, Curve::Line { .. }) => {
            for (u, t) in intersect(b, a, eps) {
                out.push((t, u));
            }
            return dedupe(out, a, b, eps);
        }
        (Curve::Arc { c: c1, r: r1, .. }, Curve::Arc { c: c2, r: r2, .. }) => {
            let d = dist(*c1, *c2);
            if d > 1e-12 * (r1 + r2) && d <= r1 + r2 + eps && d >= (r1 - r2).abs() - eps {
                let x = (d * d + r1 * r1 - r2 * r2) / (2.0 * d);
                let h2 = r1 * r1 - x * x;
                let h = h2.max(0.0).sqrt();
                let u = scale(sub(*c2, *c1), 1.0 / d);
                let base = add(*c1, scale(u, x));
                let perp = [-u[1], u[0]];
                for pnt in [add(base, scale(perp, h)), add(base, scale(perp, -h))] {
                    let th1 = (pnt[1] - c1[1]).atan2(pnt[0] - c1[0]);
                    let th2 = (pnt[1] - c2[1]).atan2(pnt[0] - c2[0]);
                    let e1 = eps / r1;
                    let e2 = eps / r2;
                    if let (Some(t), Some(v)) = (a.arc_param(th1, e1), b.arc_param(th2, e2)) {
                        out.push((t, v));
                    }
                    if h == 0.0 {
                        break;
                    }
                }
            }
            for t in [0.0, 1.0] {
                let (u, dd) = b.nearest(a.point(t));
                if dd <= eps {
                    out.push((t, u));
                }
            }
            for u in [0.0, 1.0] {
                let (t, dd) = a.nearest(b.point(u));
                if dd <= eps {
                    out.push((t, u));
                }
            }
        }
    }
    dedupe(out, a, b, eps)
}

fn dedupe(mut hits: Vec<(f64, f64)>, a: &Curve, b: &Curve, eps: f64) -> Vec<(f64, f64)> {
    hits.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mut out: Vec<(f64, f64)> = Vec::new();
    for h in hits {
        let p = a.point(h.0);
        if out.iter().any(|o| dist(a.point(o.0), p) <= 2.0 * eps || dist(b.point(o.1), b.point(h.1)) <= 2.0 * eps) {
            continue;
        }
        out.push(h);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arc_params_and_points() {
        let a = Curve::arc_from_ends([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], true);
        assert!((a.length() - PI / 2.0).abs() < 1e-12);
        let m = a.point(0.5);
        assert!((m[0] - 0.5f64.sqrt()).abs() < 1e-12 && (m[1] - 0.5f64.sqrt()).abs() < 1e-12);
        assert!(a.arc_param(PI / 4.0, 0.0).is_some());
        assert!(a.arc_param(-PI / 4.0, 0.0).is_none());
        let cw = Curve::arc_from_ends([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], false);
        assert!((cw.length() - 1.5 * PI).abs() < 1e-12);
        assert!(cw.arc_param(-PI / 4.0, 0.0).is_some());
    }

    #[test]
    fn line_arc_intersections() {
        let l = Curve::Line { a: [-2.0, 0.0], b: [2.0, 0.0] };
        let c = Curve::circle([0.0, 0.0], 1.0);
        let hits = intersect(&l, &c, 1e-9);
        assert_eq!(hits.len(), 2);
        // Tangent touch.
        let t = Curve::Line { a: [-2.0, 1.0], b: [2.0, 1.0] };
        let hits = intersect(&t, &c, 1e-9);
        assert_eq!(hits.len(), 1);
        assert!((t.point(hits[0].0)[0]).abs() < 1e-6);
    }

    #[test]
    fn winding_of_loops() {
        // Unit square, CCW.
        let sq = [
            Curve::Line { a: [0.0, 0.0], b: [1.0, 0.0] },
            Curve::Line { a: [1.0, 0.0], b: [1.0, 1.0] },
            Curve::Line { a: [1.0, 1.0], b: [0.0, 1.0] },
            Curve::Line { a: [0.0, 1.0], b: [0.0, 0.0] },
        ];
        let w: f64 = sq.iter().map(|c| c.winding([0.3, 0.6])).sum();
        assert!((w - TAU).abs() < 1e-9);
        let w: f64 = sq.iter().map(|c| c.winding([1.3, 0.6])).sum();
        assert!(w.abs() < 1e-9);
        // Circle as two half arcs; point inside and outside; point in the
        // circular segment of a short arc.
        let h1 = Curve::arc_from_ends([0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], true);
        let h2 = Curve::arc_from_ends([0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], true);
        let w = h1.winding([0.0, 0.5]) + h2.winding([0.0, 0.5]);
        assert!((w - TAU).abs() < 1e-9, "{w}");
        let w = h1.winding([0.0, 0.9]) + h2.winding([0.0, 0.9]);
        assert!((w - TAU).abs() < 1e-9, "{w}");
        let w = h1.winding([0.0, -0.9]) + h2.winding([0.0, -0.9]);
        assert!((w - TAU).abs() < 1e-9, "{w}");
        let w = h1.winding([2.0, 0.0]) + h2.winding([2.0, 0.0]);
        assert!(w.abs() < 1e-9, "{w}");
        // Long arc (270 degrees) closed by a chord.
        let long = Curve::arc_from_ends([0.0, 0.0], [1.0, 0.0], [0.0, -1.0], true);
        let chord = Curve::Line { a: [0.0, -1.0], b: [1.0, 0.0] };
        for (p, expect) in [([-0.5, 0.0], TAU), ([0.0, 0.5], TAU), ([0.6, -0.6], 0.0), ([0.2, -0.2], TAU)] {
            let w = long.winding(p) + chord.winding(p);
            assert!((w - expect).abs() < 1e-9, "p {p:?}: {w}");
        }
    }
}
