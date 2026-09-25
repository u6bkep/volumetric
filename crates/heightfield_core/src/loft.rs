//! Loft surfaces between two closed profiles as triangles.
//!
//! A profile is a closed 3D polyline that keeps track of which points are
//! the original sketch vertices. Two profiles are matched vertex to vertex
//! when they have the same number of vertices (the cyclic shift and the
//! direction minimising the summed vertex distance), otherwise by relative
//! arc length from the nearest pair of vertices. Matched profiles are
//! resampled to the same point count and joined by a ruled surface.
//!
//! Guides (curves from a vertex of one profile to a vertex of the other)
//! are not used yet: the ruled surface is the design-equivalent loft.

use crate::raster::P3;

fn sub3(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn norm3(a: P3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
fn lerp3(a: P3, b: P3, t: f64) -> P3 {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

/// A closed profile: `points` in order (closed implicitly), `vertices` =
/// indices into `points` of the original sketch vertices, ascending,
/// starting with 0.
#[derive(Clone, Debug)]
pub struct Profile {
    pub points: Vec<P3>,
    pub vertices: Vec<usize>,
}

impl Profile {
    /// Build from the pieces of a loop: each piece sampled at `n_per(len)`
    /// points (its start included, its end excluded).
    pub fn from_pieces(pieces: &[Vec<P3>]) -> Profile {
        let mut points = Vec::new();
        let mut vertices = Vec::new();
        for p in pieces {
            vertices.push(points.len());
            points.extend(p.iter().take(p.len().saturating_sub(1)).copied());
        }
        Profile { points, vertices }
    }

    pub fn len(&self) -> f64 {
        let n = self.points.len();
        (0..n).map(|i| norm3(sub3(self.points[(i + 1) % n], self.points[i]))).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Cumulative arc length at each point (and the total at the end).
    pub fn cumulative(&self) -> Vec<f64> {
        let n = self.points.len();
        let mut s = Vec::with_capacity(n + 1);
        let mut acc = 0.0;
        for i in 0..n {
            s.push(acc);
            acc += norm3(sub3(self.points[(i + 1) % n], self.points[i]));
        }
        s.push(acc);
        s
    }

    /// Arc-length position of the point on the loop nearest `p`, and the
    /// distance to it.
    pub fn nearest(&self, cum: &[f64], p: P3) -> (f64, f64) {
        let n = self.points.len();
        let mut best = (0.0, f64::INFINITY);
        for i in 0..n {
            let a = self.points[i];
            let b = self.points[(i + 1) % n];
            let d = sub3(b, a);
            let l2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let t = if l2 > 0.0 { (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1] + (p[2] - a[2]) * d[2]) / l2).clamp(0.0, 1.0) } else { 0.0 };
            let q = lerp3(a, b, t);
            let dist = norm3(sub3(p, q));
            if dist < best.1 {
                best = (cum[i] + t * (cum[i + 1] - cum[i]), dist);
            }
        }
        best
    }

    /// Point at arc length `s` (wrapping).
    pub fn at(&self, cum: &[f64], s: f64) -> P3 {
        let total = *cum.last().unwrap();
        let s = s.rem_euclid(total.max(1e-300));
        let n = self.points.len();
        let mut i = match cum.binary_search_by(|v| v.partial_cmp(&s).unwrap()) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        if i >= n {
            i = n - 1;
        }
        let seg = cum[i + 1] - cum[i];
        let t = if seg > 0.0 { (s - cum[i]) / seg } else { 0.0 };
        lerp3(self.points[i], self.points[(i + 1) % n], t)
    }

    /// The profile with its point order reversed (same start vertex).
    fn reversed(&self) -> Profile {
        let n = self.points.len();
        let mut points: Vec<P3> = Vec::with_capacity(n);
        points.push(self.points[0]);
        for i in (1..n).rev() {
            points.push(self.points[i]);
        }
        let vertices: Vec<usize> = self.vertices.iter().map(|&v| if v == 0 { 0 } else { n - v }).collect();
        let mut vertices = vertices;
        vertices.sort_unstable();
        Profile { points, vertices }
    }

    /// The profile rotated so that vertex number `k` comes first.
    fn shifted(&self, k: usize) -> Profile {
        let n = self.points.len();
        let start = self.vertices[k % self.vertices.len()];
        let points: Vec<P3> = (0..n).map(|i| self.points[(start + i) % n]).collect();
        let mut vertices: Vec<usize> = self.vertices.iter().map(|&v| (v + n - start) % n).collect();
        vertices.sort_unstable();
        Profile { points, vertices }
    }

    /// Points along vertex span `k` (from vertex `k` to vertex `k + 1`) at
    /// `m + 1` evenly spaced arc-length fractions (both ends included).
    fn span_samples(&self, cum: &[f64], k: usize, m: usize) -> Vec<P3> {
        let nv = self.vertices.len();
        let s0 = cum[self.vertices[k]];
        let s1 = if k + 1 < nv { cum[self.vertices[k + 1]] } else { *cum.last().unwrap() };
        (0..=m).map(|q| self.at(cum, s0 + (s1 - s0) * q as f64 / m as f64)).collect()
    }
}

/// Match two profiles into equal-length point lists.
pub fn match_profiles(a: &Profile, b: &Profile, samples_per_span: usize) -> (Vec<P3>, Vec<P3>) {
    let same_count = a.vertices.len() == b.vertices.len() && a.vertices.len() >= 2;
    // Candidate alignments of b: every start vertex, both directions; pick
    // the one whose vertices lie nearest a's.
    let mut best: Option<(f64, Profile)> = None;
    for rev in [false, true] {
        let bb = if rev { b.reversed() } else { b.clone() };
        for k in 0..bb.vertices.len().max(1) {
            let cand = bb.shifted(k);
            let score = if same_count {
                a.vertices.iter().zip(cand.vertices.iter()).map(|(&i, &j)| norm3(sub3(a.points[i], cand.points[j]))).sum::<f64>()
            } else {
                // Arc-length pairing from the aligned start.
                let ca = a.cumulative();
                let cb = cand.cumulative();
                let la = *ca.last().unwrap();
                let lb = *cb.last().unwrap();
                (0..16).map(|q| norm3(sub3(a.at(&ca, la * q as f64 / 16.0), cand.at(&cb, lb * q as f64 / 16.0)))).sum::<f64>()
            };
            if best.as_ref().is_none_or(|(s, _)| score < *s) {
                best = Some((score, cand));
            }
        }
    }
    let (_, bb) = best.expect("profiles have points");
    let ca = a.cumulative();
    let cb = bb.cumulative();
    let mut pa = Vec::new();
    let mut pb = Vec::new();
    if same_count {
        for k in 0..a.vertices.len() {
            let sa = a.span_samples(&ca, k, samples_per_span);
            let sb = bb.span_samples(&cb, k, samples_per_span);
            pa.extend(sa.into_iter().take(samples_per_span));
            pb.extend(sb.into_iter().take(samples_per_span));
        }
    } else {
        let n = samples_per_span * a.vertices.len().max(b.vertices.len()).max(1);
        let la = *ca.last().unwrap();
        let lb = *cb.last().unwrap();
        for q in 0..n {
            pa.push(a.at(&ca, la * q as f64 / n as f64));
            pb.push(bb.at(&cb, lb * q as f64 / n as f64));
        }
    }
    (pa, pb)
}

/// Evaluate an open polyline at fraction `f` of its length.
pub fn polyline_at(pts: &[P3], f: f64) -> P3 {
    if pts.len() < 2 {
        return pts.first().copied().unwrap_or([0.0; 3]);
    }
    let mut cum = vec![0.0];
    for w in pts.windows(2) {
        cum.push(cum.last().unwrap() + norm3(sub3(w[1], w[0])));
    }
    let total = *cum.last().unwrap();
    let s = (f.clamp(0.0, 1.0) * total).min(total);
    let i = match cum.binary_search_by(|v| v.partial_cmp(&s).unwrap()) {
        Ok(i) => i.min(pts.len() - 2),
        Err(i) => i.saturating_sub(1).min(pts.len() - 2),
    };
    let seg = cum[i + 1] - cum[i];
    let t = if seg > 0.0 { (s - cum[i]) / seg } else { 0.0 };
    lerp3(pts[i], pts[i + 1], t)
}

/// A bilinearly blended (Coons) patch between two profile arcs and two
/// guide curves: `a(u)` runs along the first profile, `b(u)` along the
/// second, `g0(v)` from `a(0)` to `b(0)` and `g1(v)` from `a(1)` to `b(1)`.
/// The guides are shifted linearly so their ends meet the profiles.
pub fn coons_triangles(a: &dyn Fn(f64) -> P3, b: &dyn Fn(f64) -> P3, g0: &[P3], g1: &[P3], nu: usize, nv: usize) -> Vec<[P3; 3]> {
    let (a0, a1, b0, b1) = (a(0.0), a(1.0), b(0.0), b(1.0));
    let corrected = |g: &[P3], pa: P3, pb: P3, v: f64| -> P3 {
        let raw = polyline_at(g, v);
        let (s, e) = (g[0], g[g.len() - 1]);
        let ds = sub3(pa, s);
        let de = sub3(pb, e);
        [raw[0] + (1.0 - v) * ds[0] + v * de[0], raw[1] + (1.0 - v) * ds[1] + v * de[1], raw[2] + (1.0 - v) * ds[2] + v * de[2]]
    };
    let mut grid: Vec<P3> = Vec::with_capacity((nu + 1) * (nv + 1));
    for j in 0..=nv {
        let v = j as f64 / nv as f64;
        let ga = corrected(g0, a0, b0, v);
        let gb = corrected(g1, a1, b1, v);
        for i in 0..=nu {
            let u = i as f64 / nu as f64;
            let pa = a(u);
            let pb = b(u);
            let mut p = [0.0; 3];
            for k in 0..3 {
                p[k] = (1.0 - v) * pa[k] + v * pb[k] + (1.0 - u) * ga[k] + u * gb[k]
                    - ((1.0 - u) * (1.0 - v) * a0[k] + u * (1.0 - v) * a1[k] + (1.0 - u) * v * b0[k] + u * v * b1[k]);
            }
            grid.push(p);
        }
    }
    let mut tris = Vec::with_capacity(2 * nu * nv);
    let idx = |i: usize, j: usize| j * (nu + 1) + i;
    for j in 0..nv {
        for i in 0..nu {
            tris.push([grid[idx(i, j)], grid[idx(i + 1, j)], grid[idx(i, j + 1)]]);
            tris.push([grid[idx(i + 1, j)], grid[idx(i + 1, j + 1)], grid[idx(i, j + 1)]]);
        }
    }
    tris
}

/// Ruled surface between two equal-length closed point lists.
pub fn ruled_triangles(a: &[P3], b: &[P3]) -> Vec<[P3; 3]> {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut tris = Vec::with_capacity(2 * n);
    for i in 0..n {
        let j = (i + 1) % n;
        tris.push([a[i], a[j], b[i]]);
        tris.push([a[j], b[j], b[i]]);
    }
    tris
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(z: f64, half: f64, n: usize) -> Profile {
        let c = [[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]];
        let pieces: Vec<Vec<P3>> = (0..4)
            .map(|k| (0..=n).map(|q| lerp3(c[k], c[(k + 1) % 4], q as f64 / n as f64)).collect())
            .collect();
        Profile::from_pieces(&pieces)
    }

    #[test]
    fn matched_squares_form_a_frustum() {
        let a = square(0.0, 1.0, 4);
        let mut b = square(1.0, 2.0, 2);
        // Rotate b's point order to start elsewhere and reverse it.
        b.points.rotate_left(3);
        b.vertices = vec![0, 2, 4, 6].into_iter().map(|v| (v + 8 - 3) % 8).collect();
        b.vertices.sort_unstable();
        let b = b.reversed();
        let (pa, pb) = match_profiles(&a, &b, 8);
        assert_eq!(pa.len(), pb.len());
        for (p, q) in pa.iter().zip(pb.iter()) {
            // Each rule connects a point on the small square to the point
            // twice as far from the centre on the big one.
            assert!((q[0] - 2.0 * p[0]).abs() < 1e-9 && (q[1] - 2.0 * p[1]).abs() < 1e-9, "{p:?} {q:?}");
        }
        let tris = ruled_triangles(&pa, &pb);
        assert_eq!(tris.len(), 2 * pa.len());
    }
}
