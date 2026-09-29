//! Exact surfaces as height functions over the plan.
//!
//! A [`Sheet`] is the part of an exact surface that faces one way
//! vertically: where the surface normal `dS/du x dS/dv` has a positive z
//! component (`up`), or a negative one. Over a plan point it has at most
//! one height (the upper half of a lying cylinder, the lower half of a
//! torus's tube), which is what makes it usable as the top or the bottom
//! of a column of material.

use brep_core::ir::Surface;
use brep_core::math::Vec3;
use brep_core::surface::SurfaceView;
use brep_core::topo::surface_normal;

#[derive(Clone, Debug)]
pub struct Sheet {
    pub surface: Surface,
    /// The sheet is where the surface normal points up.
    pub up: bool,
    /// What made it (a feature and a role); goes into face names.
    pub name: String,
}

/// Normals flatter than this are the sheet's rim, not the sheet.
const RIM: f64 = 1e-9;

/// A plan point this close to the rim is on it.
const RIM_REACH: f64 = 1e-6;

/// Where the normal is at least this far from level the sheet's height
/// is well defined by the plan point.
const STEEP: f64 = 0.1;

impl Sheet {
    /// The plane through `p` with normal `n` (pointing up or down as
    /// given; `n` must not be horizontal).
    pub fn plane(p: Vec3, n: Vec3, name: &str) -> Sheet {
        let frame = brep_core::math::Frame::from_axis_ref(p, n, [1.0, 0.0, 0.0]);
        Sheet {
            surface: Surface::Plane { frame },
            up: n[2] > 0.0,
            name: name.to_string(),
        }
    }

    /// The horizontal plane at height `z`.
    pub fn level(z: f64, name: &str) -> Sheet {
        Sheet {
            surface: Surface::Plane {
                frame: brep_core::math::Frame {
                    origin: [0.0, 0.0, z],
                    ..brep_core::math::Frame::IDENTITY
                },
            },
            up: true,
            name: name.to_string(),
        }
    }

    /// `(z, normal)` of the sheet over `(x, y)`: the point where the
    /// vertical line meets the sheet, and the surface normal there.
    /// `None` where the sheet does not reach.
    pub fn at(&self, x: f64, y: f64) -> Option<(f64, Vec3)> {
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
            view.ray_hits([x, y, base], [0.0, 0.0, 1.0], 1e-9, &mut |hit| {
                let Ok(n) = surface_normal(&self.surface, [hit.u, hit.v]) else {
                    return;
                };
                if n[2].abs() < RIM || (n[2] > 0.0) != self.up {
                    return;
                }
                let z = base + hit.t;
                // A surface folded over itself offers several: the
                // outermost in the sheet's direction.
                let better = match best {
                    None => true,
                    Some((b, _, _)) => (z > b) == self.up,
                };
                if better {
                    best = Some((z, n, [hit.u, hit.v]));
                }
            });
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
            Some(on) => Some(on),
            None => hit.map(|(z, n, _)| (z, n)),
        }
    }

    /// The point of the sheet's rim (where the surface normal is
    /// horizontal) nearest the plan point `p`, if within reach, found
    /// from `uv`: its height and normal.
    fn rim(&self, view: &SurfaceView, p: [f64; 2], mut uv: [f64; 2]) -> Option<(f64, Vec3)> {
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
        let off = (on[0] - p[0]).hypot(on[1] - p[1]);
        (n[2].abs() < 1e-6 && off <= RIM_REACH).then_some((on[2], n))
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
        let upper = Sheet {
            surface: surface.clone(),
            up: true,
            name: "upper".into(),
        };
        let lower = Sheet {
            surface,
            up: false,
            name: "lower".into(),
        };
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
        let top = Sheet {
            surface: surface.clone(),
            up: true,
            name: String::new(),
        };
        let under = Sheet {
            surface,
            up: false,
            name: String::new(),
        };
        let h = (0.25f64 - 0.04).sqrt();
        assert!((top.z(2.2, 0.0).unwrap() - h).abs() < 1e-9);
        assert!((under.z(0.0, -1.8).unwrap() + h).abs() < 1e-9);
        assert!(top.z(0.0, 0.0).is_none());
    }
}
