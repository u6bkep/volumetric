//! Exact surfaces as height functions over the plan.
//!
//! A [`Sheet`] is the part of an exact surface that faces one way
//! vertically: where the surface normal `dS/du x dS/dv` has a positive z
//! component (`up`), or a negative one. Over a plan point it has at most
//! one height (the upper half of a lying cylinder, the lower half of a
//! torus's tube), which is what makes it usable as the top or the bottom
//! of a column of material.

use brep_core::ir::{NurbsSurface, Surface};
use brep_core::math::Vec3;
use brep_core::nurbs::surface_eval;
use brep_core::surface::SurfaceView;
use brep_core::topo::surface_normal;
use std::sync::{Arc, OnceLock};

/// A free-form surface as triangles over the plan, in buckets: where to
/// start looking for the surface over a plan point.
#[derive(Debug)]
struct Shadow {
    /// Corners in the plan and in the surface's parameters.
    triangles: Vec<[([f64; 2], [f64; 2]); 3]>,
    /// The plan's bucket grid: origin, bucket size, counts.
    origin: [f64; 2],
    size: f64,
    count: [usize; 2],
    /// Triangles by bucket.
    buckets: Vec<Vec<u32>>,
}

impl Shadow {
    fn of(n: &NurbsSurface) -> Shadow {
        let dom = n.domain();
        // A few steps per span of the knots.
        let spans = |knots: &[f64]| knots.windows(2).filter(|k| k[1] > k[0]).count();
        let nu = (3 * spans(&n.knots_u)).clamp(32, 192);
        let nv = (3 * spans(&n.knots_v)).clamp(32, 192);
        let at = |i: usize, j: usize| -> ([f64; 2], [f64; 2]) {
            let u = dom[0] + (dom[1] - dom[0]) * i as f64 / nu as f64;
            let v = dom[2] + (dom[3] - dom[2]) * j as f64 / nv as f64;
            let p = surface_eval(n, u, v).0;
            ([p[0], p[1]], [u, v])
        };
        let grid: Vec<Vec<([f64; 2], [f64; 2])>> = (0..=nu)
            .map(|i| (0..=nv).map(|j| at(i, j)).collect())
            .collect();
        let mut triangles = Vec::with_capacity(2 * nu * nv);
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for i in 0..nu {
            for j in 0..nv {
                let (a, b, c, d) = (
                    grid[i][j],
                    grid[i + 1][j],
                    grid[i + 1][j + 1],
                    grid[i][j + 1],
                );
                triangles.push([a, b, c]);
                triangles.push([a, c, d]);
                for q in [a.0, b.0, c.0, d.0] {
                    for k in 0..2 {
                        lo[k] = lo[k].min(q[k]);
                        hi[k] = hi[k].max(q[k]);
                    }
                }
            }
        }
        // About one triangle's width per bucket.
        let area = (hi[0] - lo[0]) * (hi[1] - lo[1]);
        let size = (area / triangles.len().max(1) as f64).sqrt().max(1e-6) * 1.5;
        let pad = 2.0 * EDGE_REACH;
        let origin = [lo[0] - pad, lo[1] - pad];
        let count = [
            (((hi[0] - origin[0] + pad) / size).ceil() as usize).clamp(1, 2048),
            (((hi[1] - origin[1] + pad) / size).ceil() as usize).clamp(1, 2048),
        ];
        let mut shadow = Shadow {
            triangles,
            origin,
            size,
            count,
            buckets: vec![Vec::new(); count[0] * count[1]],
        };
        for (t, tri) in shadow.triangles.iter().enumerate() {
            let (mut a, mut b) = ([usize::MAX; 2], [0usize; 2]);
            for (q, _) in tri {
                for k in 0..2 {
                    let at =
                        (((q[k] - origin[k]) / size).floor().max(0.0) as usize).min(count[k] - 1);
                    a[k] = a[k].min(at);
                    b[k] = b[k].max(at);
                }
            }
            for i in a[0]..=b[0] {
                for j in a[1]..=b[1] {
                    shadow.buckets[j * count[0] + i].push(t as u32);
                }
            }
        }
        shadow
    }

    /// Parameters to start from for the crossings over `(x, y)` on the
    /// side of the surface that faces `up`: of every triangle facing
    /// that way that holds the point, or is within reach of it.
    fn starts(&self, x: f64, y: f64, up: bool) -> Vec<[f64; 2]> {
        let mut out = Vec::new();
        let at = |q: f64, k: usize| ((q - self.origin[k]) / self.size).floor();
        let (i, j) = (at(x, 0), at(y, 1));
        if i < -1.0 || j < -1.0 || i > self.count[0] as f64 || j > self.count[1] as f64 {
            return out;
        }
        let mut near: Option<(f64, [f64; 2])> = None;
        let bucket = |i: f64, j: f64| -> Option<&Vec<u32>> {
            (i >= 0.0 && j >= 0.0 && (i as usize) < self.count[0] && (j as usize) < self.count[1])
                .then(|| &self.buckets[j as usize * self.count[0] + i as usize])
        };
        // The triangles are flat where the surface is not: beside a fold
        // the point can be on the surface and in none of them, and then
        // the nearest one is where to start.
        for (di, dj) in [
            (0.0, 0.0),
            (-1.0, 0.0),
            (1.0, 0.0),
            (0.0, -1.0),
            (0.0, 1.0),
            (-1.0, -1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (1.0, 1.0),
        ] {
            let Some(list) = bucket(i + di, j + dj) else {
                continue;
            };
            for &t in list {
                let [a, b, c] = self.triangles[t as usize];
                let (e1, e2) = (
                    [b.0[0] - a.0[0], b.0[1] - a.0[1]],
                    [c.0[0] - a.0[0], c.0[1] - a.0[1]],
                );
                let det = e1[0] * e2[1] - e1[1] * e2[0];
                if det == 0.0 || (det > 0.0) != up {
                    continue;
                }
                let d = [x - a.0[0], y - a.0[1]];
                let (s, t) = (
                    (d[0] * e2[1] - d[1] * e2[0]) / det,
                    (e1[0] * d[1] - e1[1] * d[0]) / det,
                );
                let uv = |s: f64, t: f64| {
                    [
                        a.1[0] + s * (b.1[0] - a.1[0]) + t * (c.1[0] - a.1[0]),
                        a.1[1] + s * (b.1[1] - a.1[1]) + t * (c.1[1] - a.1[1]),
                    ]
                };
                const SLACK: f64 = 1e-9;
                if s >= -SLACK && t >= -SLACK && s + t <= 1.0 + SLACK {
                    out.push(uv(s, t));
                    continue;
                }
                // Outside it: how far, and the nearest point of it.
                let (cs, ct) = {
                    let (s, t) = (s.clamp(0.0, 1.0), t.clamp(0.0, 1.0));
                    let sum = s + t;
                    if sum > 1.0 {
                        (s / sum, t / sum)
                    } else {
                        (s, t)
                    }
                };
                let off = ((cs - s) * e1[0] + (ct - t) * e2[0])
                    .hypot((cs - s) * e1[1] + (ct - t) * e2[1]);
                if off <= self.size && near.is_none_or(|n| off < n.0) {
                    near = Some((off, uv(cs, ct)));
                }
            }
        }
        if out.is_empty()
            && let Some((_, uv)) = near
        {
            out.push(uv);
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct Sheet {
    pub surface: Surface,
    /// The sheet is where the surface normal points up.
    pub up: bool,
    /// What made it (a feature and a role); goes into face names.
    pub name: String,
    /// A free-form surface's shadow, to start from; made when first
    /// asked for and shared by copies.
    shadow: Arc<OnceLock<Shadow>>,
}

/// Normals flatter than this are the sheet's rim, not the sheet.
const RIM: f64 = 1e-9;

/// A plan point this close to the rim is on it: the rim in the plan is
/// chords of the curve, between vertices that are where the arrangement
/// has them, within its tolerance
/// ([`Tolerances::snap`](crate::planmap::Tolerances::snap)) of where
/// they were computed.
const RIM_REACH: f64 = 2.5e-6;

/// How far a sheet is taken to reach.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reach {
    /// As far as the surface has it.
    Open,
    /// Just beyond its rim and its edges as well.
    Beyond,
    /// And within reach of the rim, on either side, the rim.
    Rim,
}

/// A plan point this far beyond an edge of the sheet's patch is on the
/// edge: the edge in the plan is where the arrangement has it, within
/// its tolerance ([`Tolerances::snap`](crate::planmap::Tolerances::snap))
/// and the fit's of where the patch has it. (Across an edge that is not
/// the rim the height changes as the slope has it.)
const EDGE_REACH: f64 = 4e-6;

/// Where the normal is at least this far from level the sheet's height
/// is well defined by the plan point.
const STEEP: f64 = 0.1;

impl Sheet {
    /// The part of `surface` whose normal points up, or down.
    pub fn of(surface: Surface, up: bool, name: &str) -> Sheet {
        Sheet {
            surface,
            up,
            name: name.to_string(),
            shadow: Arc::default(),
        }
    }

    /// The plane through `p` with normal `n` (pointing up or down as
    /// given; `n` must not be horizontal).
    pub fn plane(p: Vec3, n: Vec3, name: &str) -> Sheet {
        let frame = brep_core::math::Frame::from_axis_ref(p, n, [1.0, 0.0, 0.0]);
        Sheet::of(Surface::Plane { frame }, n[2] > 0.0, name)
    }

    /// The horizontal plane at height `z`.
    pub fn level(z: f64, name: &str) -> Sheet {
        Sheet::of(
            Surface::Plane {
                frame: brep_core::math::Frame {
                    origin: [0.0, 0.0, z],
                    ..brep_core::math::Frame::IDENTITY
                },
            },
            true,
            name,
        )
    }

    /// Where the vertical line over `(x, y)` meets a free-form surface on
    /// the side that faces the sheet's way: `(z, u, v)` of every
    /// crossing.
    fn crossings(&self, n: &NurbsSurface, x: f64, y: f64) -> Vec<(f64, f64, f64)> {
        let dom = n.domain();
        // Which way the surface faces: over a plan point a surface that
        // folds has a crossing on either side of the fold, and the
        // sheet's is the one facing its way.
        let facing = |su: Vec3, sv: Vec3| su[0] * sv[1] - su[1] * sv[0];
        let ours = |f: f64| (f > 0.0) == self.up;
        let shadow = self.shadow.get_or_init(|| Shadow::of(n));
        let (du, dv) = (dom[1] - dom[0], dom[3] - dom[2]);
        let mut out: Vec<(f64, f64, f64)> = Vec::new();
        for [mut u, mut v] in shadow.starts(x, y, self.up) {
            let mut found = false;
            for _ in 0..32 {
                let (p, su, sv) = surface_eval(n, u, v);
                let (rx, ry) = (x - p[0], y - p[1]);
                if rx.hypot(ry) < 1e-13 {
                    found = true;
                    break;
                }
                let det = facing(su, sv);
                if det == 0.0 || !det.is_finite() {
                    break;
                }
                let (mu, mv) = (
                    (rx * sv[1] - ry * sv[0]) / det,
                    (su[0] * ry - su[1] * rx) / det,
                );
                // The whole step, or as much of it as stays on the
                // sheet's side of the fold.
                let mut part = 1.0;
                let mut to = None;
                for _ in 0..12 {
                    let (nu, nv) = (
                        (u + part * mu).clamp(dom[0], dom[1]),
                        (v + part * mv).clamp(dom[2], dom[3]),
                    );
                    let (_, su, sv) = surface_eval(n, nu, nv);
                    if ours(facing(su, sv)) {
                        to = Some((nu, nv));
                        break;
                    }
                    part *= 0.5;
                }
                let Some((nu, nv)) = to else {
                    break;
                };
                // Held at an edge with the point just beyond it: on the
                // edge. (Held, not passing: on its way along an edge the
                // search is as near the point as this many times.)
                let held = (nu - u).abs() <= 1e-12 * du && (nv - v).abs() <= 1e-12 * dv;
                (u, v) = (nu, nv);
                if held {
                    let edge = u == dom[0] || u == dom[1] || v == dom[2] || v == dom[3];
                    let off = rx.hypot(ry);
                    found = off < 1e-11 || (edge && off < EDGE_REACH);
                    break;
                }
            }
            if !found {
                continue;
            }
            let z = surface_eval(n, u, v).0[2];
            let seen = out.iter().any(|o| {
                (o.0 - z).abs() < 1e-9 && (o.1 - u).abs() < 1e-6 * du && (o.2 - v).abs() < 1e-6 * dv
            });
            if !seen {
                out.push((z, u, v));
            }
        }
        out
    }

    /// `(z, normal)` of the sheet over `(x, y)`: the point where the
    /// vertical line meets the sheet, and the surface normal there.
    /// `None` where the sheet does not reach. A point within a
    /// micrometre of the sheet's rim, on either side, is on the rim.
    pub fn at(&self, x: f64, y: f64) -> Option<(f64, Vec3)> {
        self.over(x, y, Reach::Rim)
    }

    /// The height as the surface has it over a point of its shadow, and
    /// the rim's or the edge's over a point just beyond: for a curve
    /// that crosses the rim, whose points within reach of the rim are
    /// not the rim's (a point as far inside the rim as `d` is below or
    /// above it by the square root of `d` times the diameter).
    pub fn z_within(&self, x: f64, y: f64) -> Option<f64> {
        self.over(x, y, Reach::Beyond).map(|(z, _)| z)
    }

    /// Whether the plan point is on the sheet's rim.
    pub fn on_rim(&self, x: f64, y: f64) -> bool {
        self.rim_near(x, y, RIM_REACH)
            .is_some_and(|(on, _)| (on[0] - x).hypot(on[1] - y) <= RIM_REACH)
    }

    /// The height as the surface has it, without [`Sheet::at`]'s reach
    /// for the rim: for following the sheet as a function, where the rim
    /// is its end and not a place.
    pub fn z_open(&self, x: f64, y: f64) -> Option<f64> {
        self.over(x, y, Reach::Open).map(|(z, _)| z)
    }

    fn over(&self, x: f64, y: f64, reach: Reach) -> Option<(f64, Vec3)> {
        if let Surface::Plane { frame } = &self.surface {
            let n = frame.z;
            if n[2].abs() < RIM {
                return None;
            }
            let o = frame.origin;
            let z = o[2] - (n[0] * (x - o[0]) + n[1] * (y - o[1])) / n[2];
            return ((n[2] > 0.0) == self.up).then_some((z, n));
        }
        let view = SurfaceView::from_ir(&self.surface);
        // `(z, normal, uv)` of the crossing on this sheet.
        let cast = |x: f64, y: f64, base: f64| -> Option<(f64, Vec3, [f64; 2])> {
            let mut best: Option<(f64, Vec3, [f64; 2])> = None;
            let mut take = |z: f64, uv: [f64; 2]| {
                let Ok(n) = surface_normal(&self.surface, uv) else {
                    return;
                };
                if n[2].abs() < RIM || (n[2] > 0.0) != self.up {
                    return;
                }
                // A surface folded over itself offers several: the
                // outermost in the sheet's direction.
                let better = match best {
                    None => true,
                    Some((b, _, _)) => (z > b) == self.up,
                };
                if better {
                    best = Some((z, n, uv));
                }
            };
            match &self.surface {
                Surface::Nurbs(n) => {
                    for (z, u, v) in self.crossings(n, x, y) {
                        take(z, [u, v]);
                    }
                }
                _ => view.ray_hits([x, y, base], [0.0, 0.0, 1.0], 1e-9, &mut |hit| {
                    take(base + hit.t, [hit.u, hit.v])
                }),
            }
            best
        };
        // From far below, where every crossing is ahead, then again from
        // just under the one found: a crossing is as exact as it is
        // near.
        let hit = cast(x, y, -64.0).map(|(z, n, uv)| {
            cast(x, y, z - 1e-3)
                .filter(|(again, _, _)| (again - z).abs() < 1e-6)
                .unwrap_or((z, n, uv))
        });
        if reach == Reach::Open {
            return hit.map(|(z, n, _)| (z, n));
        }
        if let Some((z, n, _)) = hit
            && (n[2].abs() >= STEEP || reach == Reach::Beyond)
        {
            return Some((z, n));
        }
        // Near where the sheet turns vertical its height is as uncertain
        // as the square root of the plan point's error: a point within
        // reach of the rim is on it.
        // (From the crossing, or from a crossing nearby: the first from
        // which the rim is found within reach.)
        let nearby = [1e-6, 1e-5, 1e-4, 1e-3].into_iter().flat_map(|reach| {
            (0..8).filter_map(move |k| {
                let a = core::f64::consts::FRAC_PI_4 * k as f64;
                cast(x + reach * a.cos(), y + reach * a.sin(), -64.0).map(|h| h.2)
            })
        });
        hit.map(|h| h.2)
            .into_iter()
            .chain(nearby)
            .find_map(|start| {
                self.rim(&view, [x, y], start)
                    .filter(|(on, _)| (on[0] - x).hypot(on[1] - y) <= RIM_REACH)
            })
            .map(|(on, n)| (on[2], n))
            .or(hit.map(|(z, n, _)| (z, n)))
            .or_else(|| self.on_edge(x, y))
    }

    /// The sheet over a plan point on the shadow of one of the patch's
    /// own edges, where the edge is the sheet's or the rim's: `(z,
    /// normal)`. Where an edge runs into the rim the plan says nothing
    /// of the patch's parameters in either direction (across the edge
    /// there is no patch, across the rim no way), but along the edge it
    /// does.
    fn on_edge(&self, x: f64, y: f64) -> Option<(f64, Vec3)> {
        let Surface::Nurbs(n) = &self.surface else {
            return None;
        };
        let dom = n.domain();
        let (du, dv) = (dom[1] - dom[0], dom[3] - dom[2]);
        let edges: [&dyn Fn(f64) -> [f64; 2]; 4] = [
            &|t| [dom[0] + t * du, dom[2]],
            &|t| [dom[0] + t * du, dom[3]],
            &|t| [dom[0], dom[2] + t * dv],
            &|t| [dom[1], dom[2] + t * dv],
        ];
        const N: usize = 128;
        let mut best: Option<(f64, [f64; 2])> = None;
        for edge in edges {
            let off = |t: f64| {
                let [u, v] = edge(t);
                let p = surface_eval(n, u, v).0;
                (p[0] - x).hypot(p[1] - y)
            };
            // (An edge that stands upright is over one plan point from
            // end to end, and at no height in particular.)
            let plan = |t: f64| {
                let [u, v] = edge(t);
                let p = surface_eval(n, u, v).0;
                [p[0], p[1]]
            };
            let (a, m, b) = (plan(0.0), plan(0.5), plan(1.0));
            if (a[0] - m[0]).hypot(a[1] - m[1]) + (b[0] - m[0]).hypot(b[1] - m[1])
                <= 4.0 * EDGE_REACH
            {
                continue;
            }
            let Some((k, _)) = (0..=N)
                .map(|k| (k, off(k as f64 / N as f64)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
            else {
                continue;
            };
            // The least between the neighbours of the nearest sample.
            let (mut lo, mut hi) = (
                k.saturating_sub(1) as f64 / N as f64,
                (k + 1).min(N) as f64 / N as f64,
            );
            let golden = 0.5 * (5f64.sqrt() - 1.0);
            for _ in 0..64 {
                let (a, b) = (hi - golden * (hi - lo), lo + golden * (hi - lo));
                if off(a) <= off(b) {
                    hi = b;
                } else {
                    lo = a;
                }
            }
            let t = 0.5 * (lo + hi);
            let d = off(t);
            if best.is_none_or(|(b, _)| d < b) {
                best = Some((d, edge(t)));
            }
        }
        let (d, uv) = best?;
        if d > EDGE_REACH {
            return None;
        }
        let normal = surface_normal(&self.surface, uv).ok()?;
        (normal[2].abs() < STEEP || (normal[2] > 0.0) == self.up)
            .then(|| (surface_eval(n, uv[0], uv[1]).0[2], normal))
    }

    /// The point of the sheet's rim nearest the plan point `(x, y)`,
    /// with the (level) normal there, if the sheet is within `within` of
    /// the point at all.
    pub fn rim_near(&self, x: f64, y: f64, within: f64) -> Option<(Vec3, Vec3)> {
        if matches!(self.surface, Surface::Plane { .. }) {
            return None;
        }
        let view = SurfaceView::from_ir(&self.surface);
        let crossing = |x: f64, y: f64| -> Option<[f64; 2]> {
            let mut uv = None;
            match &self.surface {
                Surface::Nurbs(n) => {
                    if let Some(&(_, u, v)) = self.crossings(n, x, y).first() {
                        uv = Some([u, v]);
                    }
                }
                _ => view.ray_hits([x, y, -64.0], [0.0, 0.0, 1.0], 1e-9, &mut |hit| {
                    uv = Some([hit.u, hit.v])
                }),
            }
            uv
        };
        let start = crossing(x, y).or_else(|| {
            [0.01, 0.1, 1.0].into_iter().find_map(|part| {
                (0..8).find_map(|k| {
                    let a = core::f64::consts::FRAC_PI_4 * k as f64;
                    crossing(x + part * within * a.cos(), y + part * within * a.sin())
                })
            })
        })?;
        self.rim(&view, [x, y], start)
            .filter(|(on, _)| (on[0] - x).hypot(on[1] - y) <= within)
    }

    /// The point of the sheet's rim (where the surface normal is
    /// horizontal) nearest the plan point `p`, found from `uv`, and the
    /// normal there.
    fn rim(&self, view: &SurfaceView, p: [f64; 2], mut uv: [f64; 2]) -> Option<(Vec3, Vec3)> {
        // Zero where the normal is level and the way from `p` to the
        // point is along it.
        let f = |uv: [f64; 2]| -> Option<[f64; 2]> {
            let n = surface_normal(&self.surface, uv).ok()?;
            let s = view.eval(uv[0], uv[1]);
            let h = (n[0] * n[0] + n[1] * n[1]).sqrt().max(1e-300);
            Some([n[2], ((s[0] - p[0]) * n[1] - (s[1] - p[1]) * n[0]) / h])
        };
        // Steps in u and v of about a micrometre on the surface.
        let here = view.eval(uv[0], uv[1]);
        let scale_of = |d: [f64; 2]| {
            let q = view.eval(uv[0] + 1e-6 * d[0], uv[1] + 1e-6 * d[1]);
            let l = brep_core::math::norm(brep_core::math::sub(q, here)) / 1e-6;
            if l > 0.0 { 1e-6 / l } else { 1e-6 }
        };
        let h = [scale_of([1.0, 0.0]), scale_of([0.0, 1.0])];
        for _ in 0..40 {
            let f0 = f(uv)?;
            let fu = f([uv[0] + h[0], uv[1]])?;
            let fv = f([uv[0], uv[1] + h[1]])?;
            // Columns of the Jacobian, the second equation brought to
            // the first one's size (it is a length).
            let j = [
                [(fu[0] - f0[0]) / h[0], (fv[0] - f0[0]) / h[1]],
                [(fu[1] - f0[1]) / h[0], (fv[1] - f0[1]) / h[1]],
            ];
            let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
            if det == 0.0 || !det.is_finite() {
                return None;
            }
            let du = -(f0[0] * j[1][1] - f0[1] * j[0][1]) / det;
            let dv = -(j[0][0] * f0[1] - j[1][0] * f0[0]) / det;
            uv = [uv[0] + du, uv[1] + dv];
            if (du / h[0]).abs() + (dv / h[1]).abs() < 1e-6 {
                break;
            }
        }
        let on = view.eval(uv[0], uv[1]);
        let n = surface_normal(&self.surface, uv).ok()?;
        (n[2].abs() < 1e-6).then_some((on, n))
    }

    pub fn z(&self, x: f64, y: f64) -> Option<f64> {
        self.at(x, y).map(|(z, _)| z)
    }

    /// Height and slope `(z, dz/dx, dz/dy)`; `None` also on the rim,
    /// where there is no slope.
    pub fn slope(&self, x: f64, y: f64) -> Option<(f64, f64, f64)> {
        let (z, n) = self.at(x, y)?;
        (n[2].abs() >= 1e-3).then(|| (z, -n[0] / n[2], -n[1] / n[2]))
    }

    /// The plane's `(z0, gx, gy)` with `z = z0 + gx x + gy y`, if the
    /// sheet is a plane.
    pub fn as_plane(&self) -> Option<(f64, f64, f64)> {
        let Surface::Plane { frame } = &self.surface else {
            return None;
        };
        let n = frame.z;
        if n[2].abs() < RIM {
            return None;
        }
        let (gx, gy) = (-n[0] / n[2], -n[1] / n[2]);
        let o = frame.origin;
        Some((o[2] - gx * o[0] - gy * o[1], gx, gy))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brep_core::math::Frame;

    #[test]
    fn planes() {
        let s = Sheet::plane([1.0, 2.0, 3.0], [-1.0, 0.0, 1.0], "slope");
        assert!(s.up);
        assert!((s.z(1.0, 7.0).unwrap() - 3.0).abs() < 1e-12);
        assert!((s.z(2.0, 0.0).unwrap() - 4.0).abs() < 1e-12);
        let (z0, gx, gy) = s.as_plane().unwrap();
        assert!((z0 - 2.0).abs() < 1e-12 && (gx - 1.0).abs() < 1e-12 && gy.abs() < 1e-12);
        let down = Sheet::plane([0.0, 0.0, 1.0], [0.0, 0.0, -1.0], "under");
        assert!(!down.up);
        assert!((down.z(5.0, 5.0).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn halves_of_a_lying_cylinder() {
        // Axis along y through (0, 0, 1), radius 0.5.
        let frame = Frame::from_axis_ref([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]);
        let surface = Surface::Cylinder { frame, radius: 0.5 };
        let upper = Sheet::of(surface.clone(), true, "upper");
        let lower = Sheet::of(surface, false, "lower");
        let h = (0.25f64 - 0.09).sqrt();
        assert!((upper.z(0.3, 4.0).unwrap() - (1.0 + h)).abs() < 1e-12);
        assert!((lower.z(0.3, 4.0).unwrap() - (1.0 - h)).abs() < 1e-12);
        assert!(upper.z(0.6, 0.0).is_none());
        // On the rim, and a rounding error beyond it.
        assert!((upper.z(0.5, 3.0).unwrap() - 1.0).abs() < 1e-9);
        assert!((lower.z(-0.5 - 1e-13, 3.0).unwrap() - 1.0).abs() < 1e-9);
        assert!(upper.z(0.5 + 4e-6, 3.0).is_none());
        let (_, gx, gy) = upper.slope(0.3, 0.0).unwrap();
        assert!((gx + 0.3 / h).abs() < 1e-9 && gy.abs() < 1e-12);
    }

    #[test]
    fn tube_of_a_torus() {
        let surface = Surface::Torus {
            frame: Frame::IDENTITY,
            major: 2.0,
            minor: 0.5,
        };
        let top = Sheet::of(surface.clone(), true, "");
        let under = Sheet::of(surface, false, "");
        let h = (0.25f64 - 0.04).sqrt();
        assert!((top.z(2.2, 0.0).unwrap() - h).abs() < 1e-9);
        assert!((under.z(0.0, -1.8).unwrap() + h).abs() < 1e-9);
        assert!(top.z(0.0, 0.0).is_none());
    }
}
