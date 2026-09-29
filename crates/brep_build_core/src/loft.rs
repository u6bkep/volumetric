//! Lofts: the faces between two profiles.
//!
//! The construction is `heightfield_core::loft`'s and
//! `onshape_interp_core::exec`'s, on exact curves. Guides that reach both
//! profiles cut the loft into strips, each a bilinearly blended (Coons)
//! patch between its two guides and the stretches of the profiles
//! between them; within a strip the profiles' corners pair up when they
//! are as many on both, else the profiles are run at even speed. Without
//! two guides the loft is ruled: corner to corner when the profiles have
//! as many, else at even speed from the best start.
//!
//! A strip is smooth only between the corners of its profiles and the
//! joints of its guides, so it is cut there into blocks, and each block
//! is one face: a bicubic B-spline fitted to the blend.

use crate::body::FaceSpec;
use crate::chain::{Chain, Span};
use crate::fit::fit_surface;
use brep_core::math::{Vec3, add, dot, norm, scale, sub};

/// One end of a loft.
#[derive(Clone, Debug)]
pub struct Profile {
    /// Closed.
    pub chain: Chain,
    /// The plane the profile is in: a point and the normal. Guides are
    /// cut where they pass it.
    pub plane: (Vec3, Vec3),
    /// The face that closes the loft there, if one does.
    pub cap: Option<FaceSpec>,
}

/// A profile as a strip runs it: the chain's length at `u` is
/// `from + rate * map(u)`.
#[derive(Clone)]
struct Run<'a> {
    chain: &'a Chain,
    from: f64,
    /// 1 with the chain, -1 against it.
    rate: f64,
    /// The stretch's length.
    reach: f64,
    /// Lengths along the stretch of the corners that are paired, when
    /// they are.
    paired: Option<Vec<f64>>,
}

impl Run<'_> {
    /// Length along the stretch at `u`: through the paired corners at
    /// even steps of `u`, else at even speed.
    fn map(&self, u: f64) -> f64 {
        let Some(corners) = &self.paired else {
            return u * self.reach;
        };
        let n = corners.len() + 1;
        let f = u * n as f64;
        let k = (f.floor().max(0.0) as usize).min(n - 1);
        let lo = if k == 0 { 0.0 } else { corners[k - 1] };
        let hi = if k + 1 == n { self.reach } else { corners[k] };
        lo + (f - k as f64) * (hi - lo)
    }

    /// The `u` at which the stretch is at length `t` along it.
    fn unmap(&self, t: f64) -> f64 {
        let Some(corners) = &self.paired else {
            return t / self.reach.max(1e-300);
        };
        let n = corners.len() + 1;
        let k = corners.iter().filter(|c| **c <= t).count();
        let lo = if k == 0 { 0.0 } else { corners[k - 1] };
        let hi = if k + 1 == n { self.reach } else { corners[k] };
        (k as f64 + (t - lo) / (hi - lo).max(1e-300)) / n as f64
    }

    fn at(&self, u: f64) -> Vec3 {
        self.chain.at(self.from + self.rate * self.map(u))
    }

    /// The values of `u` inside the stretch at which the chain passes
    /// from one span to the next or the map from one rate to the next.
    fn breaks(&self) -> Vec<f64> {
        let total = self.chain.total();
        let mut out = Vec::new();
        for joint in self.chain.joints() {
            // Along the stretch, in its direction.
            let t = (self.rate * (joint - self.from)).rem_euclid(total);
            if t > 1e-9 && t < self.reach - 1e-9 {
                out.push(self.unmap(t));
            }
        }
        if let Some(corners) = &self.paired {
            let n = corners.len() + 1;
            out.extend((1..n).map(|k| k as f64 / n as f64));
        }
        out
    }

    /// The run between `u0` and `u1`, where it is on one span at one
    /// rate, as a function that goes on past both.
    fn smooth(&self, u0: f64, u1: f64) -> impl Fn(f64) -> Vec3 + '_ {
        let mid = 0.5 * (u0 + u1);
        let (k, within) = self.chain.locate(self.from + self.rate * self.map(mid));
        let rate = self.rate * (self.map(u1) - self.map(u0)) / (u1 - u0);
        move |u: f64| self.chain.spans[k].at(within + rate * (u - mid))
    }
}

/// A guide from one profile to the other, over `[0, 1]` by its length,
/// moved so that its ends are on the profiles.
struct Guide {
    chain: Chain,
    /// What is added at the start and at the end.
    shift: (Vec3, Vec3),
}

impl Guide {
    fn new(chain: Chain, from: Vec3, to: Vec3) -> Guide {
        let shift = (sub(from, chain.start()), sub(to, chain.end()));
        Guide { chain, shift }
    }

    fn straight(from: Vec3, to: Vec3) -> Guide {
        let chain = Chain::new(vec![Span::Line { a: from, b: to }], vec![true], false);
        Guide::new(chain, from, to)
    }

    fn breaks(&self) -> Vec<f64> {
        let total = self.chain.total().max(1e-300);
        self.chain.joints()[1..].iter().map(|j| j / total).collect()
    }

    fn smooth(&self, v0: f64, v1: f64) -> impl Fn(f64) -> Vec3 + '_ {
        let total = self.chain.total();
        let mid = 0.5 * (v0 + v1);
        let (k, within) = self.chain.locate(mid * total);
        move |v: f64| {
            let raw = self.chain.spans[k].at(within + (v - mid) * total);
            add(
                raw,
                add(scale(self.shift.0, 1.0 - v), scale(self.shift.1, v)),
            )
        }
    }
}

/// A strip of the loft between two guides.
struct Strip<'a> {
    a: Run<'a>,
    b: Run<'a>,
    g0: Guide,
    g1: Guide,
}

/// A guide cut where it passes the planes of the profiles.
fn trimmed(guide: &Chain, planes: [(Vec3, Vec3); 2]) -> Chain {
    let total = guide.total();
    let passing = |plane: &(Vec3, Vec3)| -> Option<f64> {
        let side = |s: f64| dot(sub(guide.at(s), plane.0), plane.1);
        let n = 512;
        let mut before = side(0.0);
        if before == 0.0 {
            return Some(0.0);
        }
        for k in 1..=n {
            let s = total * k as f64 / n as f64;
            let here = side(s);
            if (before < 0.0) != (here < 0.0) || here == 0.0 {
                let (mut lo, mut hi) = (total * (k - 1) as f64 / n as f64, s);
                for _ in 0..60 {
                    let mid = 0.5 * (lo + hi);
                    if (side(mid) < 0.0) == (before < 0.0) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                return Some(0.5 * (lo + hi));
            }
            before = here;
        }
        None
    };
    match (passing(&planes[0]), passing(&planes[1])) {
        (Some(x), Some(y)) if (x - y).abs() > 1e-9 => guide.part(x.min(y), x.max(y)),
        _ => guide.clone(),
    }
}

/// The line or the arc the points lie on, or the smooth curve through
/// them.
fn exact_span(pts: &[Vec3]) -> Span {
    let (first, last) = (pts[0], pts[pts.len() - 1]);
    let chord = sub(last, first);
    let len = norm(chord);
    const EXACT: f64 = 1e-10;
    if len > 0.0
        && pts
            .iter()
            .all(|p| norm(brep_core::math::cross(sub(*p, first), chord)) / len <= EXACT)
    {
        return Span::Line { a: first, b: last };
    }
    if let Some((brep_core::topo::Curve3::Circle { frame, radius }, t0, t1)) =
        crate::conic::conic_through(pts, EXACT)
    {
        return Span::Arc {
            c: frame.origin,
            x: frame.x,
            y: frame.y,
            r: radius,
            a0: t0,
            sweep: t1 - t0,
        };
    }
    Span::through(pts)
}

/// The faces of the loft from `a` to `b` along `guides`, and what was
/// done. `fit` is how close the faces stay to the blend.
pub fn loft(
    name: &str,
    a: &Profile,
    b: &Profile,
    guides: &[Chain],
    fit: f64,
) -> Result<(Vec<FaceSpec>, String), String> {
    let (ca, cb) = (&a.chain, &b.chain);
    let (la, lb) = (ca.total(), cb.total());
    if !(la > 0.0 && lb > 0.0) {
        return Err("a profile without length".into());
    }
    let mut note = String::new();
    // Guides that reach both profiles: where on each, and the guide from
    // the first to the second.
    let mut attached: Vec<(f64, f64, Chain)> = Vec::new();
    let mut skipped = 0;
    for g in guides {
        let mut g = trimmed(g, [a.plane, b.plane]);
        let (s0a, d0a) = ca.nearest(g.start());
        let (s0b, d0b) = cb.nearest(g.start());
        let (s1a, d1a) = ca.nearest(g.end());
        let (s1b, d1b) = cb.nearest(g.end());
        let (sa, sb, da, db) = if d0a + d1b <= d1a + d0b {
            (s0a, s1b, d0a, d1b)
        } else {
            g = g.reversed();
            (s1a, s0b, d1a, d0b)
        };
        if da > 5e-3 || db > 5e-3 {
            note.push_str(&format!(
                "; guide off the profiles by {:.1} / {:.1} mm",
                da * 1e3,
                db * 1e3
            ));
            skipped += 1;
            continue;
        }
        attached.push((sa, sb, g));
    }
    if attached.len() == 1 {
        // One guide fixes no pairing around the loops.
        note.push_str("; 1 guide: ruled");
        attached.clear();
    }

    let mut strips: Vec<Strip> = Vec::new();
    if !attached.is_empty() {
        attached.sort_by(|x, y| x.0.total_cmp(&y.0));
        let n = attached.len();
        // The second profile runs with the first when the guides reach
        // it in the same order; two guides cannot tell, and then the
        // stretches of like length are paired.
        let mut vote = 0i32;
        for i in 0..n {
            let (b0, b1) = (attached[i].1, attached[(i + 1) % n].1);
            // (Half way around is no vote either way.)
            let ahead = (b1 - b0).rem_euclid(lb) / lb;
            vote += if ahead < 0.5 - 1e-9 {
                1
            } else if ahead > 0.5 + 1e-9 {
                -1
            } else {
                0
            };
        }
        let along = |flip: bool, s: f64| if flip { (lb - s).rem_euclid(lb) } else { s };
        let flip = if vote != 0 {
            vote < 0
        } else {
            let twist = |flip: bool| -> f64 {
                (0..n)
                    .map(|i| {
                        let (s0a, s0b, _) = &attached[i];
                        let (s1a, s1b, _) = &attached[(i + 1) % n];
                        let mut da = s1a - s0a;
                        if da <= 1e-9 {
                            da += la;
                        }
                        let mut db = along(flip, *s1b) - along(flip, *s0b);
                        if db <= 1e-9 {
                            db += lb;
                        }
                        ((da / la) / (db / lb)).ln().abs()
                    })
                    .sum()
            };
            // (Against it only when that is better by more than
            // rounding.)
            twist(true) < twist(false) - 1e-9
        };
        for i in 0..n {
            let (s0a, s0b_raw, g0) = &attached[i];
            let (s1a, s1b_raw, g1) = &attached[(i + 1) % n];
            let (s0b, s1b) = (along(flip, *s0b_raw), along(flip, *s1b_raw));
            let mut da = s1a - s0a;
            if da <= 1e-9 {
                da += la;
            }
            let mut db = s1b - s0b;
            if db <= 1e-9 {
                db += lb;
            }
            // Corners inside the stretch, as lengths along it.
            let inner = |corners: Vec<f64>, reach: f64| -> Vec<f64> {
                let mut out: Vec<f64> = corners
                    .into_iter()
                    .filter(|t| *t > 1e-6 && *t < reach - 1e-6)
                    .collect();
                out.sort_by(|x, y| x.total_cmp(y));
                out
            };
            let va = inner(
                ca.vertices()
                    .iter()
                    .map(|v| (v - s0a).rem_euclid(la))
                    .collect(),
                da,
            );
            let vb = inner(
                cb.vertices()
                    .iter()
                    .map(|v| (along(flip, *v) - s0b).rem_euclid(lb))
                    .collect(),
                db,
            );
            let paired = !va.is_empty() && va.len() == vb.len();
            if paired {
                note.push_str(&format!("; strip {i}: {} corners paired", va.len()));
            }
            let run_a = Run {
                chain: ca,
                from: *s0a,
                rate: 1.0,
                reach: da,
                paired: paired.then_some(va),
            };
            // Against the chain the stretch starts at the chain's own
            // length for the flipped one.
            let run_b = Run {
                chain: cb,
                from: if flip { lb - s0b } else { s0b },
                rate: if flip { -1.0 } else { 1.0 },
                reach: db,
                paired: paired.then_some(vb),
            };
            let (a0, a1, b0, b1) = (run_a.at(0.0), run_a.at(1.0), run_b.at(0.0), run_b.at(1.0));
            strips.push(Strip {
                a: run_a,
                b: run_b,
                g0: Guide::new(g0.clone(), a0, b0),
                g1: Guide::new(g1.clone(), a1, b1),
            });
        }
        note.push_str(&format!(
            "; {n} guides, {n} strips{}",
            if flip {
                " (second profile against the first)"
            } else {
                ""
            }
        ));
    } else {
        // Ruled. The second profile from each of its corners, both
        // ways: the start that brings its corners, or without as many
        // corners its points at even speed, nearest the first's.
        let (va, vb) = (ca.vertices(), cb.vertices());
        let same = va.len() == vb.len() && va.len() >= 2;
        let mut best: Option<(f64, f64, f64)> = None;
        for rate in [1.0, -1.0] {
            for start in &vb {
                // Its corners in the order it is run in from there.
                let mut order: Vec<f64> = vb
                    .iter()
                    .map(|v| (rate * (v - start)).rem_euclid(lb))
                    .collect();
                order.sort_by(|x, y| x.total_cmp(y));
                let score: f64 = if same {
                    va.iter()
                        .zip(&order)
                        .map(|(s, t)| norm(sub(ca.at(*s), cb.at(start + rate * t))))
                        .sum()
                } else {
                    (0..16)
                        .map(|q| {
                            let f = q as f64 / 16.0;
                            norm(sub(ca.at(va[0] + f * la), cb.at(start + rate * f * lb)))
                        })
                        .sum()
                };
                if best.is_none_or(|b| score < b.0) {
                    best = Some((score, *start, rate));
                }
            }
        }
        let (_, start, rate) = best.ok_or("a profile without corners")?;
        let mut order: Vec<f64> = vb
            .iter()
            .map(|v| (rate * (v - start)).rem_euclid(lb))
            .collect();
        order.sort_by(|x, y| x.total_cmp(y));
        let stretches: Vec<(f64, f64, f64, f64)> = if same {
            (0..va.len())
                .map(|k| {
                    let next = (k + 1) % va.len();
                    let da = if next == 0 {
                        la - (va[k] - va[0])
                    } else {
                        va[next] - va[k]
                    };
                    let db = if next == 0 {
                        lb - order[k]
                    } else {
                        order[next] - order[k]
                    };
                    (va[k], da, order[k], db)
                })
                .collect()
        } else {
            vec![(va[0], la, 0.0, lb)]
        };
        for (sa, da, tb, db) in stretches {
            let run_a = Run {
                chain: ca,
                from: sa,
                rate: 1.0,
                reach: da,
                paired: None,
            };
            let run_b = Run {
                chain: cb,
                from: start + rate * tb,
                rate,
                reach: db,
                paired: None,
            };
            let (a0, a1, b0, b1) = (run_a.at(0.0), run_a.at(1.0), run_b.at(0.0), run_b.at(1.0));
            strips.push(Strip {
                a: run_a,
                b: run_b,
                g0: Guide::straight(a0, b0),
                g1: Guide::straight(a1, b1),
            });
        }
        note.push_str(&format!(
            "; ruled, {}",
            if same {
                "corner to corner"
            } else {
                "at even speed"
            }
        ));
    }
    if skipped > 0 {
        note.push_str(&format!("; {skipped} guides left out"));
    }

    // The sides of the first profile, to name the faces after.
    let sides = ca.vertices();
    let side_of = |s: f64| -> usize {
        let s = s.rem_euclid(la);
        sides.iter().rposition(|v| *v <= s + 1e-9).unwrap_or(0)
    };
    let mut faces: Vec<FaceSpec> = Vec::new();
    for (i, strip) in strips.iter().enumerate() {
        let cuts = |mut at: Vec<f64>| -> Vec<f64> {
            at.extend([0.0, 1.0]);
            at.retain(|x| (0.0..=1.0).contains(x));
            at.sort_by(|x, y| x.total_cmp(y));
            at.dedup_by(|x, y| (*x - *y).abs() < 1e-9);
            at
        };
        let us = cuts([strip.a.breaks(), strip.b.breaks()].concat());
        let vs = cuts([strip.g0.breaks(), strip.g1.breaks()].concat());
        let corner = [
            strip.a.at(0.0),
            strip.a.at(1.0),
            strip.b.at(0.0),
            strip.b.at(1.0),
        ];
        for u in us.windows(2) {
            let (a, b) = (strip.a.smooth(u[0], u[1]), strip.b.smooth(u[0], u[1]));
            for v in vs.windows(2) {
                let (g0, g1) = (strip.g0.smooth(v[0], v[1]), strip.g1.smooth(v[0], v[1]));
                let blend = |u: f64, v: f64| -> Vec3 {
                    let (pa, pb, p0, p1) = (a(u), b(u), g0(v), g1(v));
                    let mut p = [0.0; 3];
                    for k in 0..3 {
                        p[k] = (1.0 - v) * pa[k] + v * pb[k] + (1.0 - u) * p0[k] + u * p1[k]
                            - ((1.0 - u) * (1.0 - v) * corner[0][k]
                                + u * (1.0 - v) * corner[1][k]
                                + (1.0 - u) * v * corner[2][k]
                                + u * v * corner[3][k]);
                    }
                    p
                };
                let mid = 0.5 * (u[0] + u[1]);
                // The edges from the blend itself, which a neighbour has
                // the same: through points of it, a line or an arc when
                // they lie on one.
                let edge = |f: &dyn Fn(f64) -> Vec3| -> Span {
                    let first = f(0.0);
                    let mut pts = vec![first];
                    let n = 256;
                    for k in 1..=n {
                        pts.push(f(k as f64 / n as f64));
                    }
                    exact_span(&pts)
                };
                let edges = vec![
                    edge(&|t| blend(u[0] + t * (u[1] - u[0]), v[0])),
                    edge(&|t| blend(u[0] + t * (u[1] - u[0]), v[1])),
                    edge(&|t| blend(u[0], v[0] + t * (v[1] - v[0]))),
                    edge(&|t| blend(u[1], v[0] + t * (v[1] - v[0]))),
                ];
                faces.push(FaceSpec::Patch {
                    edges,
                    surface: fit_surface(&blend, (u[0], u[1]), (v[0], v[1]), fit),
                    name: format!(
                        "{name}:side:{i}/{}",
                        side_of(strip.a.from + strip.a.map(mid))
                    ),
                });
            }
        }
    }
    note.push_str(&format!("; {} faces", faces.len()));
    for cap in [&a.cap, &b.cap].into_iter().flatten() {
        faces.push(cap.clone());
    }
    Ok((faces, note))
}
