//! The height-field model payload: the data contract between whatever
//! builds a [`Solid`] and `heightfield_model_template`, which samples it
//! at model sample time. Same pattern as the grid-field payload: all work
//! happens up front, the template only reads.
//!
//! # Payload layout (little-endian)
//!
//! ```text
//! Header (56 bytes):
//!    0  magic       u32   "HF01" (0x3130_4648)
//!    4  nx          u32   columns (>= 2)
//!    8  ny          u32   rows (>= 2)
//!   12  payload_len u32   total byte length, header included
//!   16  x0          f64   x of the first cell centre
//!   24  y0          f64   y of the first cell centre
//!   32  pitch       f64   cell spacing
//!   40  z_min       f64   lowest bottom height
//!   48  z_max       f64   highest top height
//! top  nx*ny x f32 at 56: row-major, row 0 at y0
//! bot  nx*ny x f32
//! sdf  nx*ny x f32: signed distance to the footprint boundary, positive inside
//! ```
//!
//! # Sampling
//!
//! Fields are bilinear between cell centres. A point is inside iff the
//! interpolated `sdf > 0` and `bot <= z <= top`. Outside the grid of cell
//! centres the model is empty. Heights are filled beyond the footprint by
//! the builder so interpolation next to the boundary never sees `NaN`.

use crate::Solid;

pub const MAGIC: u32 = 0x3130_4648; // "HF01"
const HEADER_LEN: usize = 56;

/// Serialize a solid. Empty columns get the heights of the nearest filled
/// column; the footprint distance comes from [`Solid::footprint_sdf`].
pub fn build_payload(solid: &Solid) -> Result<Vec<u8>, String> {
    let g = &solid.grid;
    if g.nx < 2 || g.ny < 2 {
        return Err("height field needs at least 2 x 2 cells".into());
    }
    let (top, bot) = solid.filled_fields();
    let sdf = solid.footprint_sdf();
    let n = g.len();
    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;
    for k in 0..n {
        if !solid.top[k].is_nan() {
            z_min = z_min.min(solid.bot[k] as f64);
            z_max = z_max.max(solid.top[k] as f64);
        }
    }
    if !z_min.is_finite() {
        return Err("solid has no material".into());
    }
    let payload_len = HEADER_LEN + 3 * n * 4;
    let mut out = Vec::with_capacity(payload_len);
    out.extend(MAGIC.to_le_bytes());
    out.extend((g.nx as u32).to_le_bytes());
    out.extend((g.ny as u32).to_le_bytes());
    out.extend((payload_len as u32).to_le_bytes());
    for v in [g.x(0), g.y(0), g.pitch, z_min, z_max] {
        out.extend(v.to_le_bytes());
    }
    for field in [&top, &bot, &sdf] {
        for v in field {
            let v = if v.is_nan() { 0.0 } else { *v };
            out.extend(v.to_le_bytes());
        }
    }
    debug_assert_eq!(out.len(), payload_len);
    Ok(out)
}

/// Read-only view over a payload (the template's side).
pub struct PayloadView<'a> {
    bytes: &'a [u8],
    nx: usize,
    ny: usize,
}

impl<'a> PayloadView<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self, &'static str> {
        if bytes.len() < HEADER_LEN {
            return Err("payload shorter than header");
        }
        let u32_at = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        if u32_at(0) != MAGIC {
            return Err("bad payload magic");
        }
        let nx = u32_at(4) as usize;
        let ny = u32_at(8) as usize;
        let payload_len = u32_at(12) as usize;
        let expected = HEADER_LEN
            + nx.checked_mul(ny).and_then(|n| n.checked_mul(12)).ok_or("grid dimensions overflow")?;
        if nx < 2 || ny < 2 || payload_len != expected || bytes.len() < expected {
            return Err("payload length mismatch");
        }
        Ok(Self { bytes, nx, ny })
    }

    fn f64_at(&self, off: usize) -> f64 {
        f64::from_le_bytes(self.bytes[off..off + 8].try_into().unwrap())
    }

    /// `[x0, y0, pitch, z_min, z_max]`.
    pub fn header(&self) -> [f64; 5] {
        std::array::from_fn(|i| self.f64_at(16 + i * 8))
    }

    /// `[min_x, max_x, min_y, max_y, min_z, max_z]` of the sampled domain.
    pub fn bounds(&self) -> [f64; 6] {
        let [x0, y0, pitch, z_min, z_max] = self.header();
        [x0, x0 + (self.nx - 1) as f64 * pitch, y0, y0 + (self.ny - 1) as f64 * pitch, z_min, z_max]
    }

    fn value(&self, field: usize, col: usize, row: usize) -> f64 {
        let off = HEADER_LEN + (field * self.nx * self.ny + row * self.nx + col) * 4;
        f32::from_le_bytes(self.bytes[off..off + 4].try_into().unwrap()) as f64
    }

    /// Bilinear `(top, bot, sdf)` at `(x, y)`, `None` outside the grid.
    pub fn fields(&self, x: f64, y: f64) -> Option<(f64, f64, f64)> {
        let [x0, y0, pitch, _, _] = self.header();
        let gx = (x - x0) / pitch;
        let gy = (y - y0) / pitch;
        if !(gx >= 0.0 && gy >= 0.0 && gx <= (self.nx - 1) as f64 && gy <= (self.ny - 1) as f64) {
            return None;
        }
        let c0 = (gx.floor() as usize).min(self.nx - 2);
        let r0 = (gy.floor() as usize).min(self.ny - 2);
        let fx = gx - c0 as f64;
        let fy = gy - r0 as f64;
        let lerp = |field: usize| {
            (1.0 - fx) * (1.0 - fy) * self.value(field, c0, r0)
                + fx * (1.0 - fy) * self.value(field, c0 + 1, r0)
                + (1.0 - fx) * fy * self.value(field, c0, r0 + 1)
                + fx * fy * self.value(field, c0 + 1, r0 + 1)
        };
        Some((lerp(0), lerp(1), lerp(2)))
    }

    pub fn inside(&self, x: f64, y: f64, z: f64) -> bool {
        match self.fields(x, y) {
            Some((top, bot, sdf)) => sdf > 0.0 && z >= bot && z <= top,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Grid, Op, Prism};
    use sketch_regions_core::{Curve, Piece, arrange};

    #[test]
    fn round_trip_slab() {
        let grid = Grid::covering(0.0, 0.0, 4.0, 3.0, 0.5, 2);
        let mut s = Solid::empty(grid.clone());
        let p = |a, b| Piece { id: "r".into(), curve: Curve::Line { a, b } };
        let region = arrange(&[p([0.0, 0.0], [4.0, 0.0]), p([4.0, 0.0], [4.0, 3.0]), p([4.0, 3.0], [0.0, 3.0]), p([0.0, 3.0], [0.0, 0.0])], 1e-9)
            .remove(0);
        s.outline = Some(vec![region.clone()]);
        let prism = Prism { region, origin: [0.0; 3], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], dir: [0.0, 0.0, 1.0], w0: 0.5, w1: 2.0, label: None };
        s.extrude(&prism, Op::Add, 1, 1);
        let bytes = build_payload(&s).unwrap();
        let view = PayloadView::new(&bytes).unwrap();
        let b = view.bounds();
        assert_eq!(b[4], 0.5);
        assert_eq!(b[5], 2.0);
        assert!(view.inside(2.0, 1.5, 1.0));
        assert!(!view.inside(2.0, 1.5, 2.1));
        assert!(!view.inside(2.0, 1.5, 0.4));
        // The exact outline decides the wall position between cell centres.
        assert!(view.inside(3.9, 1.5, 1.0));
        assert!(!view.inside(4.1, 1.5, 1.0));
        assert!(!view.inside(-0.2, 1.5, 1.0));
        assert!(PayloadView::new(&bytes[..40]).is_err());
    }
}
