//! Rasterizing surfaces into a [`Body`]: every triangle (or planar cap)
//! records its height at the cell centres it covers, and a closed surface's
//! lowest / highest hits per column are the column's material interval.

use crate::{Body, Grid};
use sketch_regions_core::Region;

pub type P3 = [f64; 3];

/// Add the triangles' heights to `body`. Vertical triangles (no projected
/// area) contribute nothing; their neighbours cover their columns.
pub fn include_triangles(body: &mut Body, tris: &[[P3; 3]]) {
    include_triangles_labelled(body, tris, None)
}

/// [`include_triangles`] with a face label recorded on the columns whose
/// extreme the triangles set.
pub fn include_triangles_labelled(body: &mut Body, tris: &[[P3; 3]], label: Option<u32>) {
    let g = body.grid.clone();
    for t in tris {
        let [a, b, c] = *t;
        let bb = [
            a[0].min(b[0]).min(c[0]),
            a[1].min(b[1]).min(c[1]),
            a[0].max(b[0]).max(c[0]),
            a[1].max(b[1]).max(c[1]),
        ];
        let (i0, i1, j0, j1) = g.cells_in([bb[0] - g.pitch, bb[1] - g.pitch, bb[2] + g.pitch, bb[3] + g.pitch]);
        let det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
        if det.abs() < 1e-300 {
            continue;
        }
        let inv = 1.0 / det;
        // Tolerance so cell centres on shared edges are hit by at least one
        // side.
        let tol = 1e-9;
        for j in j0..j1 {
            let y = g.y(j);
            for i in i0..i1 {
                let x = g.x(i);
                let l1 = ((b[0] - x) * (c[1] - y) - (c[0] - x) * (b[1] - y)) * inv;
                let l2 = ((c[0] - x) * (a[1] - y) - (a[0] - x) * (c[1] - y)) * inv;
                let l3 = 1.0 - l1 - l2;
                if l1 < -tol || l2 < -tol || l3 < -tol {
                    continue;
                }
                let z = l1 * a[2] + l2 * b[2] + l3 * c[2];
                match label {
                    Some(l) => body.include_labelled(g.idx(i, j), z as f32, l),
                    None => body.include(g.idx(i, j), z as f32),
                }
            }
        }
    }
}

/// Add a planar cap: height `z` over every cell centre inside `region`
/// (region coordinates are the grid's x / y).
pub fn include_cap(body: &mut Body, region: &Region, z: f64) {
    include_cap_labelled(body, region, z, None)
}

pub fn include_cap_labelled(body: &mut Body, region: &Region, z: f64, label: Option<u32>) {
    let g = body.grid.clone();
    let bb = region.bbox();
    let (i0, i1, j0, j1) = g.cells_in([bb[0] - g.pitch, bb[1] - g.pitch, bb[2] + g.pitch, bb[3] + g.pitch]);
    for j in j0..j1 {
        for i in i0..i1 {
            if region.contains([g.x(i), g.y(j)]) {
                match label {
                    Some(l) => body.include_labelled(g.idx(i, j), z as f32, l),
                    None => body.include(g.idx(i, j), z as f32),
                }
            }
        }
    }
}

/// Ear-clipping triangulation of a simple polygon (any orientation),
/// returning index triples into `pts`. Degenerate (collinear) ears are
/// clipped too so noisy loops still terminate.
pub fn triangulate_loop(pts: &[[f64; 2]]) -> Vec<[usize; 3]> {
    let n = pts.len();
    let mut tris = Vec::new();
    if n < 3 {
        return tris;
    }
    // Orientation.
    let area: f64 = (0..n).map(|i| {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        a[0] * b[1] - b[0] * a[1]
    }).sum();
    let sign = if area >= 0.0 { 1.0 } else { -1.0 };
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    // A vertex on the candidate ear's boundary blocks it too (a reflex
    // vertex lying on the ear's chord would otherwise be cut through);
    // only points coinciding with the ear's own corners are ignored.
    let inside = |p: [f64; 2], a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
        let scale = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).max((c[0] - a[0]).powi(2) + (c[1] - a[1]).powi(2));
        let eps = 1e-12 * scale;
        let near = |q: [f64; 2]| (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) <= eps;
        if near(a) || near(b) || near(c) {
            return false;
        }
        let d1 = cross(a, b, p) * sign;
        let d2 = cross(b, c, p) * sign;
        let d3 = cross(c, a, p) * sign;
        d1 >= -eps && d2 >= -eps && d3 >= -eps
    };
    let mut idx: Vec<usize> = (0..n).collect();
    let mut guard = 0usize;
    while idx.len() > 3 && guard < 4 * n * n {
        guard += 1;
        let m = idx.len();
        let mut clipped = false;
        for k in 0..m {
            let (i0, i1, i2) = (idx[(k + m - 1) % m], idx[k], idx[(k + 1) % m]);
            let (a, b, c) = (pts[i0], pts[i1], pts[i2]);
            let convex = cross(a, b, c) * sign;
            if convex < 0.0 {
                continue;
            }
            let mut ear = true;
            if convex > 0.0 {
                for &j in &idx {
                    if j != i0 && j != i1 && j != i2 && inside(pts[j], a, b, c) {
                        ear = false;
                        break;
                    }
                }
            }
            if ear {
                if convex > 0.0 {
                    tris.push([i0, i1, i2]);
                }
                idx.remove(k);
                clipped = true;
                break;
            }
        }
        if !clipped {
            // Fall back: clip the first vertex regardless.
            let (i0, i1, i2) = (idx[idx.len() - 1], idx[0], idx[1]);
            tris.push([i0, i1, i2]);
            idx.remove(0);
        }
    }
    if idx.len() == 3 {
        tris.push([idx[0], idx[1], idx[2]]);
    }
    tris
}

/// The body enclosed by a closed triangle mesh (assumed 2.5D: one material
/// interval per column).
pub fn body_from_mesh(grid: &Grid, tris: &[[P3; 3]]) -> Body {
    let mut body = Body::empty(grid);
    include_triangles(&mut body, tris);
    body
}

/// Read a binary or ASCII STL file into triangles.
pub fn read_stl(bytes: &[u8]) -> Result<Vec<[P3; 3]>, String> {
    let is_ascii = bytes.len() >= 5 && &bytes[..5] == b"solid" && {
        // Binary files can start with "solid" too: check the triangle count.
        if bytes.len() >= 84 {
            let n = u32::from_le_bytes(bytes[80..84].try_into().unwrap()) as usize;
            84 + n * 50 != bytes.len()
        } else {
            true
        }
    };
    if is_ascii {
        let text = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
        let mut tris = Vec::new();
        let mut cur: Vec<P3> = Vec::new();
        for line in text.lines() {
            let mut it = line.split_whitespace();
            if it.next() == Some("vertex") {
                let v: Vec<f64> = it.take(3).map(|s| s.parse::<f64>().unwrap_or(f64::NAN)).collect();
                if v.len() == 3 {
                    cur.push([v[0], v[1], v[2]]);
                    if cur.len() == 3 {
                        tris.push([cur[0], cur[1], cur[2]]);
                        cur.clear();
                    }
                }
            }
        }
        return Ok(tris);
    }
    if bytes.len() < 84 {
        return Err("STL shorter than its header".into());
    }
    let n = u32::from_le_bytes(bytes[80..84].try_into().unwrap()) as usize;
    if bytes.len() < 84 + n * 50 {
        return Err("STL truncated".into());
    }
    let f = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as f64;
    let mut tris = Vec::with_capacity(n);
    for t in 0..n {
        let o = 84 + t * 50 + 12;
        tris.push([[f(o), f(o + 4), f(o + 8)], [f(o + 12), f(o + 16), f(o + 20)], [f(o + 24), f(o + 28), f(o + 32)]]);
    }
    Ok(tris)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangulates_a_concave_polygon() {
        // An L shape.
        let pts = [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]];
        let tris = triangulate_loop(&pts);
        assert_eq!(tris.len(), 4);
        let area: f64 = tris
            .iter()
            .map(|[a, b, c]| {
                let (p, q, r) = (pts[*a], pts[*b], pts[*c]);
                0.5 * ((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1])).abs()
            })
            .sum();
        assert!((area - 3.0).abs() < 1e-12);
    }

    #[test]
    fn box_mesh_gives_its_columns() {
        // A unit cube [0,1]^3 as 12 triangles.
        let v = |x: f64, y: f64, z: f64| [x, y, z];
        let q = |a: P3, b: P3, c: P3, d: P3| [[a, b, c], [a, c, d]];
        let mut tris = Vec::new();
        tris.extend(q(v(0., 0., 0.), v(1., 0., 0.), v(1., 1., 0.), v(0., 1., 0.)));
        tris.extend(q(v(0., 0., 1.), v(1., 0., 1.), v(1., 1., 1.), v(0., 1., 1.)));
        tris.extend(q(v(0., 0., 0.), v(1., 0., 0.), v(1., 0., 1.), v(0., 0., 1.)));
        tris.extend(q(v(0., 1., 0.), v(1., 1., 0.), v(1., 1., 1.), v(0., 1., 1.)));
        tris.extend(q(v(0., 0., 0.), v(0., 1., 0.), v(0., 1., 1.), v(0., 0., 1.)));
        tris.extend(q(v(1., 0., 0.), v(1., 1., 0.), v(1., 1., 1.), v(1., 0., 1.)));
        let grid = Grid::covering(0.0, 0.0, 1.0, 1.0, 0.25, 1);
        let body = body_from_mesh(&grid, &tris);
        let k = grid.idx(2, 2);
        assert_eq!(body.lo[k], 0.0);
        assert_eq!(body.hi[k], 1.0);
        assert!(body.lo[grid.idx(0, 0)].is_nan());
    }
}
