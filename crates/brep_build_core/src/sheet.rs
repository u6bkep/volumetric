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

/// A plan point of a surface and its parameters.
type Seed = ([f64; 2], [f64; 2]);

#[derive(Clone, Debug)]
pub struct Sheet {
    pub surface: Surface,
    /// The sheet is where the surface normal points up.
    pub up: bool,
    /// What made it (a feature and a role); goes into face names.
    pub name: String,
    /// Points of a free-form surface with their parameters, to start
    /// from; made when first asked for and shared by copies.
    seeds: Arc<OnceLock<Vec<Seed>>>,
}

/// Normals flatter than this are the sheet's rim, not the sheet.
const RIM: f64 = 1e-9;

/// A plan point this close to the rim is on it.
const RIM_REACH: f64 = 1e-6;

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
            seeds: Arc::default(),
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

    /// Where the vertical line over `(x, y)` meets a free-form surface:
    /// `(z, u, v)` of every crossing found from the nearest seeds.
    fn crossings(&self, n: &NurbsSurface, x: f64, y: f64) -> Vec<(f64, f64, f64)> {
        let dom = n.domain();
        let seeds = self.seeds.get_or_init(|| {
            // A few points per span of the knots.
            let spans = |knots: &[f64]| knots.windows(2).filter(|k| k[1] > k[0]).count();
            let nu = (4 * spans(&n.knots_u)).clamp(8, 96);
            let nv = (4 * spans(&n.knots_v)).clamp(8, 96);
            let mut seeds = Vec::with_capacity((nu + 1) * (nv + 1));
            for i in 0..=nu {
                for j in 0..=nv {
                    let u = dom[0] + (dom[1] - dom[0]) * i as f64 / nu as f64;
                    let v = dom[2] + (dom[3] - dom[2]) * j as f64 / nv as f64;
                    let p = surface_eval(n, u, v).0;
                    seeds.push(([p[0], p[1]], [u, v]));
                }
            }
            seeds
        });
        const STARTS: usize = 8;
        let mut nearest: [(f64, usize); STARTS] = [(f64::INFINITY, 0); STARTS];
        for (k, (p, _)) in seeds.iter().enumerate() {
            let d = (p[0] - x).powi(2) + (p[1] - y).powi(2);
            if d < nearest[STARTS - 1].0 {
                let at = nearest.iter().position(|e| d < e.0).unwrap();
                nearest[at..].rotate_right(1);
                nearest[at] = (d, k);
            }
        }
        let (du, dv) = (dom[1] - dom[0], dom[3] - dom[2]);
        let mut out: Vec<(f64, f64, f64)> = Vec::new();
        for &(d, k) in &nearest {
            if !d.is_finite() {
                continue;
            }
            let [mut u, mut v] = seeds[k].1;
            let mut found = false;
            for step in 0..32 {
                let (p, su, sv) = surface_eval(n, u, v);
                let (rx, ry) = (x - p[0], y - p[1]);
                // On the surface, or at its edge with the point just
                // beyond it.
                let edge = u == dom[0] || u == dom[1] || v == dom[2] || v == dom[3];
                if rx.hypot(ry) < 1e-13 || (step > 8 && edge && rx.hypot(ry) < RIM_REACH) {
                    found = true;
                    break;
                }
                let det = su[0] * sv[1] - su[1] * sv[0];
                if det == 0.0 || !det.is_finite() {
                    break;
                }
                u = (u + (rx * sv[1] - ry * sv[0]) / det).clamp(dom[0], dom[1]);
                v = (v + (su[0] * ry - su[1] * rx) / det).clamp(dom[2], dom[3]);
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
        self.over(x, y, true)
    }

    /// The height as the surface has it, without [`Sheet::at`]'s reach
    /// for the rim: for following the sheet as a function, where the rim
    /// is its end and not a place.
    pub fn z_open(&self, x: f64, y: f64) -> Option<f64> {
        self.over(x, y, false).map(|(z, _)| z)
    }

    fn over(&self, x: f64, y: f64, reach: bool) -> Option<(f64, Vec3)> {
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
        if !reach {
            return hit.map(|(z, n, _)| (z, n));
        }
        if let Some((z, n, _)) = hit
            && n[2].abs() >= STEEP
        {
            return Some((z, n));
        }
        // Near where the sheet turns vertical its height is as uncertain
        // as the square root of the plan point's error: a point within
        // reach of the rim is on it.
        let start = hit.map(|h| h.2).or_else(|| {
            [1e-6, 1e-5, 1e-4].into_iter().find_map(|reach| {
                (0..8).find_map(|k| {
                    let a = core::f64::consts::FRAC_PI_4 * k as f64;
                    cast(x + reach * a.cos(), y + reach * a.sin(), -64.0).map(|h| h.2)
                })
            })
        })?;
        match self.rim(&view, [x, y], start) {
            Some((on, n)) if (on[0] - x).hypot(on[1] - y) <= RIM_REACH => Some((on[2], n)),
            _ => hit.map(|(z, n, _)| (z, n)),
        }
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
        assert!(upper.z(0.5 + 2e-6, 3.0).is_none());
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
