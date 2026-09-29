//! Curves in space by arc length: the profiles and guides of a loft.
//!
//! A [`Span`] is a line, a circular arc, or a smooth curve through
//! points; a [`Chain`] is spans end to end, open (a guide) or closed (a
//! profile). Both are evaluated at a length along them. A span goes on
//! past its ends the way it was going, which is what lets a surface made
//! from it be differentiated at its edge.

use crate::fit::spline_through;
use brep_core::math::{Vec3, add, cross, dot, norm, normalize, scale, sub};
use brep_core::nurbs::curve_eval;
use brep_core::topo::NurbsCurve;

#[derive(Clone, Debug)]
pub enum Span {
    Line {
        a: Vec3,
        b: Vec3,
    },
    /// `c + r (cos t x + sin t y)` for `t` from `a0` over `sweep`.
    Arc {
        c: Vec3,
        x: Vec3,
        y: Vec3,
        r: f64,
        a0: f64,
        sweep: f64,
    },
    /// A smooth curve through points, over `[0, 1]` by chord length.
    Free {
        curve: NurbsCurve,
        len: f64,
    },
}

impl Span {
    /// A smooth curve through `points`, or the line between two.
    pub fn through(points: &[Vec3]) -> Span {
        if points.len() <= 2 {
            return Span::Line {
                a: points[0],
                b: points[points.len() - 1],
            };
        }
        let len = points.windows(2).map(|w| norm(sub(w[1], w[0]))).sum();
        Span::Free {
            curve: spline_through(points, 1e-8, false),
            len,
        }
    }

    pub fn len(&self) -> f64 {
        match self {
            Span::Line { a, b } => norm(sub(*b, *a)),
            Span::Arc { r, sweep, .. } => r * sweep.abs(),
            Span::Free { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() <= 0.0
    }

    /// The point at length `s` from the start, also before 0 and past
    /// the end.
    pub fn at(&self, s: f64) -> Vec3 {
        match self {
            Span::Line { a, b } => {
                let d = sub(*b, *a);
                let l = norm(d).max(1e-300);
                add(*a, scale(d, s / l))
            }
            Span::Arc {
                c,
                x,
                y,
                r,
                a0,
                sweep,
            } => {
                let t = a0 + sweep.signum() * s / r;
                add(*c, add(scale(*x, r * t.cos()), scale(*y, r * t.sin())))
            }
            Span::Free { curve, len } => {
                let t = s / len;
                if (0.0..=1.0).contains(&t) {
                    curve_eval(curve, t).0
                } else {
                    // Straight on from the end.
                    let end = if t < 0.0 { 0.0 } else { 1.0 };
                    let (p, d) = curve_eval(curve, end);
                    add(p, scale(d, t - end))
                }
            }
        }
    }

    pub fn start(&self) -> Vec3 {
        self.at(0.0)
    }

    pub fn end(&self) -> Vec3 {
        self.at(self.len())
    }

    /// The length along the span of its point nearest `p`, and the
    /// distance.
    pub fn nearest(&self, p: Vec3) -> (f64, f64) {
        let len = self.len();
        let s = match self {
            Span::Line { a, b } => {
                let d = sub(*b, *a);
                let l2 = dot(d, d);
                if l2 > 0.0 {
                    (dot(sub(p, *a), d) / l2).clamp(0.0, 1.0) * len
                } else {
                    0.0
                }
            }
            Span::Arc {
                c,
                x,
                y,
                a0,
                sweep,
                r,
            } => {
                let w = sub(p, *c);
                let t = dot(w, *y).atan2(dot(w, *x));
                // Along the arc's own turn from its start.
                let turn = ((t - a0) * sweep.signum()).rem_euclid(core::f64::consts::TAU);
                if turn <= sweep.abs() {
                    turn * r
                } else if norm(sub(p, self.start())) <= norm(sub(p, self.end())) {
                    0.0
                } else {
                    len
                }
            }
            Span::Free { .. } => {
                // The best of a scan, then halved steps about it.
                let n = 64;
                let mut best = (0.0, f64::INFINITY);
                for k in 0..=n {
                    let s = len * k as f64 / n as f64;
                    let d = norm(sub(self.at(s), p));
                    if d < best.1 {
                        best = (s, d);
                    }
                }
                let mut step = len / n as f64;
                for _ in 0..40 {
                    step *= 0.5;
                    for s in [(best.0 - step).max(0.0), (best.0 + step).min(len)] {
                        let d = norm(sub(self.at(s), p));
                        if d < best.1 {
                            best = (s, d);
                        }
                    }
                }
                best.0
            }
        };
        (s, norm(sub(self.at(s), p)))
    }

    /// The part from length `s0` to `s1`; backwards when `s1 < s0`.
    pub fn part(&self, s0: f64, s1: f64) -> Span {
        match self {
            Span::Line { .. } => Span::Line {
                a: self.at(s0),
                b: self.at(s1),
            },
            Span::Arc {
                c,
                x,
                y,
                r,
                a0,
                sweep,
            } => Span::Arc {
                c: *c,
                x: *x,
                y: *y,
                r: *r,
                a0: a0 + sweep.signum() * s0 / r,
                sweep: sweep.signum() * (s1 - s0) / r,
            },
            Span::Free { .. } => {
                let n = (((s1 - s0).abs() / 2.5e-4).ceil() as usize).clamp(2, 4096);
                let pts: Vec<Vec3> = (0..=n)
                    .map(|k| self.at(s0 + (s1 - s0) * k as f64 / n as f64))
                    .collect();
                Span::through(&pts)
            }
        }
    }

    pub fn reversed(&self) -> Span {
        self.part(self.len(), 0.0)
    }

    /// Whether `next` goes on along the same line or circle from this
    /// span's end.
    fn goes_on(&self, next: &Span, tol: f64) -> Option<Span> {
        if norm(sub(self.end(), next.start())) > tol {
            return None;
        }
        match (self, next) {
            (Span::Line { a, b }, Span::Line { b: far, .. }) => {
                let (d, e) = (normalize(sub(*b, *a)), normalize(sub(*far, *b)));
                (norm(cross(d, e)) < 1e-9 && dot(d, e) > 0.0)
                    .then_some(Span::Line { a: *a, b: *far })
            }
            (
                Span::Arc {
                    c,
                    x,
                    y,
                    r,
                    a0,
                    sweep,
                },
                Span::Arc {
                    c: c2,
                    r: r2,
                    sweep: sweep2,
                    x: x2,
                    y: y2,
                    ..
                },
            ) => {
                let same = norm(sub(*c, *c2)) <= tol
                    && (r - r2).abs() <= tol
                    && (sweep > &0.0) == (sweep2 > &0.0)
                    && norm(cross(cross(*x, *y), cross(*x2, *y2))) < 1e-9
                    && dot(cross(*x, *y), cross(*x2, *y2)) > 0.0;
                same.then_some(Span::Arc {
                    c: *c,
                    x: *x,
                    y: *y,
                    r: *r,
                    a0: *a0,
                    sweep: sweep + sweep2,
                })
            }
            _ => None,
        }
    }
}

/// Spans end to end.
#[derive(Clone, Debug)]
pub struct Chain {
    pub spans: Vec<Span>,
    /// Whether each span starts at a vertex: a corner of the profile
    /// that is paired with a corner of the other one. (The others are
    /// joints within what is one side of the profile.)
    pub vertex: Vec<bool>,
    /// The length up to each span, and the whole at the end.
    pub cum: Vec<f64>,
    pub closed: bool,
}

impl Chain {
    pub fn new(spans: Vec<Span>, vertex: Vec<bool>, closed: bool) -> Chain {
        let mut cum = vec![0.0];
        for s in &spans {
            cum.push(cum.last().unwrap() + s.len());
        }
        Chain {
            spans,
            vertex,
            cum,
            closed,
        }
    }

    /// An open chain of `spans` in the order and the direction in which
    /// they join end to end, consecutive ones on one line or circle made
    /// one.
    pub fn joined(mut spans: Vec<Span>, tol: f64) -> Result<Chain, String> {
        spans.retain(|s| s.len() > 0.0);
        if spans.is_empty() {
            return Err("a chain of nothing".into());
        }
        let mut chain = vec![spans.remove(0)];
        while !spans.is_empty() {
            let (start, end) = (chain[0].start(), chain[chain.len() - 1].end());
            let mut best: Option<(usize, bool, bool, f64)> = None;
            for (k, s) in spans.iter().enumerate() {
                for (reverse, prepend, d) in [
                    (false, false, norm(sub(end, s.start()))),
                    (true, false, norm(sub(end, s.end()))),
                    (false, true, norm(sub(start, s.end()))),
                    (true, true, norm(sub(start, s.start()))),
                ] {
                    if best.is_none_or(|b| d < b.3) {
                        best = Some((k, reverse, prepend, d));
                    }
                }
            }
            let (k, reverse, prepend, d) = best.unwrap();
            if d > tol {
                return Err(format!(
                    "the curves do not join (a gap of {:.2} mm)",
                    d * 1e3
                ));
            }
            let mut s = spans.remove(k);
            if reverse {
                s = s.reversed();
            }
            if prepend {
                chain.insert(0, s);
            } else {
                chain.push(s);
            }
        }
        let mut merged: Vec<Span> = Vec::new();
        for s in chain {
            match merged.last().and_then(|last| last.goes_on(&s, 1e-7)) {
                Some(one) => *merged.last_mut().unwrap() = one,
                None => merged.push(s),
            }
        }
        let n = merged.len();
        Ok(Chain::new(merged, vec![true; n], false))
    }

    pub fn total(&self) -> f64 {
        *self.cum.last().unwrap()
    }

    /// The span that holds length `s` (around, for a closed chain) and
    /// the length within it.
    pub fn locate(&self, s: f64) -> (usize, f64) {
        let total = self.total();
        let s = if self.closed {
            s.rem_euclid(total.max(1e-300))
        } else {
            s
        };
        let n = self.spans.len();
        let k = (0..n).rev().find(|&k| self.cum[k] <= s).unwrap_or(0);
        (k, s - self.cum[k])
    }

    pub fn at(&self, s: f64) -> Vec3 {
        let (k, within) = self.locate(s);
        self.spans[k].at(within)
    }

    pub fn start(&self) -> Vec3 {
        self.spans[0].start()
    }

    pub fn end(&self) -> Vec3 {
        self.spans[self.spans.len() - 1].end()
    }

    /// The length along the chain of its point nearest `p`, and the
    /// distance.
    pub fn nearest(&self, p: Vec3) -> (f64, f64) {
        let mut best = (0.0, f64::INFINITY);
        for (k, span) in self.spans.iter().enumerate() {
            let (s, d) = span.nearest(p);
            if d < best.1 {
                best = (self.cum[k] + s, d);
            }
        }
        best
    }

    /// The lengths at which the vertices are.
    pub fn vertices(&self) -> Vec<f64> {
        (0..self.spans.len())
            .filter(|&k| self.vertex[k])
            .map(|k| self.cum[k])
            .collect()
    }

    /// The lengths at which one span ends and the next begins.
    pub fn joints(&self) -> Vec<f64> {
        self.cum[..self.spans.len()].to_vec()
    }

    /// The part of an open chain from length `s0` to `s1 > s0`.
    pub fn part(&self, s0: f64, s1: f64) -> Chain {
        let mut spans = Vec::new();
        for (k, span) in self.spans.iter().enumerate() {
            let (from, to) = (s0.max(self.cum[k]), s1.min(self.cum[k + 1]));
            if to - from > 1e-12 {
                spans.push(span.part(from - self.cum[k], to - self.cum[k]));
            }
        }
        let n = spans.len();
        Chain::new(spans, vec![true; n], false)
    }

    pub fn reversed(&self) -> Chain {
        let spans: Vec<Span> = self.spans.iter().rev().map(Span::reversed).collect();
        let n = spans.len();
        Chain::new(spans, vec![true; n], self.closed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::FRAC_PI_2;

    fn quarter() -> Span {
        Span::Arc {
            c: [0.0, 0.0, 1.0],
            x: [1.0, 0.0, 0.0],
            y: [0.0, 1.0, 0.0],
            r: 2.0,
            a0: 0.0,
            sweep: -FRAC_PI_2,
        }
    }

    #[test]
    fn by_length_and_back() {
        let arc = quarter();
        assert!((arc.len() - core::f64::consts::PI).abs() < 1e-12);
        let end = arc.end();
        assert!(norm(sub(end, [0.0, -2.0, 1.0])) < 1e-12);
        let (s, d) = arc.nearest([3.0, -3.0, 1.0]);
        assert!((s - 0.5 * arc.len()).abs() < 1e-12 && (d - (18f64.sqrt() - 2.0)).abs() < 1e-12);
        // Off its end the nearest point is the end.
        assert!((arc.nearest([-1.0, -2.0, 1.0]).0 - arc.len()).abs() < 1e-12);
        let back = arc.reversed();
        assert!(norm(sub(back.start(), end)) < 1e-12);
        assert!(norm(sub(back.at(1.0), arc.at(arc.len() - 1.0))) < 1e-12);
        // Past its end it goes on around.
        assert!(
            norm(sub(
                arc.at(arc.len() + 0.2),
                [-2.0 * 0.1f64.sin(), -2.0 * 0.1f64.cos(), 1.0]
            )) < 1e-12
        );
    }

    #[test]
    fn joined_in_any_order() {
        let a = Span::Line {
            a: [0.0, 0.0, 1.0],
            b: [2.0, 0.0, 1.0],
        };
        let b = quarter().reversed();
        let c = Span::Line {
            a: [-3.0, -2.0, 1.0],
            b: [0.0, -2.0, 1.0],
        };
        let half = Span::Line {
            a: [-5.0, -2.0, 1.0],
            b: [-3.0, -2.0, 1.0],
        };
        let chain = Chain::joined(vec![b, half, a, c], 1e-6).unwrap();
        // The two lines in line are one span.
        assert_eq!(chain.spans.len(), 3);
        assert!((chain.total() - (2.0 + core::f64::consts::PI + 5.0)).abs() < 1e-12);
        let ends = [chain.start(), chain.end()];
        assert!(ends.iter().any(|p| norm(sub(*p, [0.0, 0.0, 1.0])) < 1e-12));
        assert!(
            ends.iter()
                .any(|p| norm(sub(*p, [-5.0, -2.0, 1.0])) < 1e-12)
        );
        let part = chain.part(1.0, chain.total() - 1.0);
        assert!((part.total() - (chain.total() - 2.0)).abs() < 1e-12);
        assert!(norm(sub(part.start(), chain.at(1.0))) < 1e-12);
    }
}
