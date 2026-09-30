//! Regions of a planar sketch: the faces of the arrangement of its curves.
//!
//! Input is a set of line and arc pieces, each tagged with the id of the
//! sketch entity it came from. The pieces are split at every mutual
//! intersection, their end points snapped together, and the bounded faces
//! of the resulting planar graph walked out as [`Region`]s: an outer loop,
//! zero or more hole loops, and the set of entity ids on the boundary,
//! which is how Onshape itself identifies a sketch region across
//! configurations.
//!
//! The same machinery gives exact 2D booleans of regions ([`union`],
//! [`difference`]): the arrangement of both operands' boundaries, faces
//! classified by a sample point, and boundary pieces between two kept faces
//! dropped.
//!
//! Everything is exact for lines and circular arcs (no flattening); ellipses
//! and splines are for the caller to approximate by short pieces.

pub mod geom;

use std::collections::{BTreeSet, HashMap};

pub use geom::{Curve, P2};
use geom::{add, cross, dist, dot, norm, scale, sub};

/// One boundary piece with the id of the sketch entity it lies on.
#[derive(Clone, Debug)]
pub struct Piece {
    pub id: String,
    pub curve: Curve,
}

/// A bounded face of the arrangement.
#[derive(Clone, Debug)]
pub struct Region {
    /// Counter-clockwise boundary.
    pub outer: Vec<Piece>,
    /// Clockwise hole boundaries.
    pub holes: Vec<Vec<Piece>>,
    /// Ids of every entity on the boundary (outer and holes).
    pub ids: BTreeSet<String>,
    pub area: f64,
}

fn loop_area(pieces: &[Piece]) -> f64 {
    let mut a = 0.0;
    for p in pieces {
        let s = p.curve.start();
        let e = p.curve.end();
        a += 0.5 * cross(s, e) + p.curve.segment_area();
    }
    a
}

fn loop_winding(pieces: &[Piece], p: P2) -> f64 {
    pieces.iter().map(|q| q.curve.winding(p)).sum()
}

impl Region {
    pub fn pieces(&self) -> impl Iterator<Item = &Piece> {
        self.outer.iter().chain(self.holes.iter().flatten())
    }

    /// `[min_x, min_y, max_x, max_y]`.
    pub fn bbox(&self) -> [f64; 4] {
        let mut bb = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for p in &self.outer {
            let b = p.curve.bbox();
            bb[0] = bb[0].min(b[0]);
            bb[1] = bb[1].min(b[1]);
            bb[2] = bb[2].max(b[2]);
            bb[3] = bb[3].max(b[3]);
        }
        bb
    }

    /// Winding-number containment (boundary points are unspecified).
    pub fn contains(&self, p: P2) -> bool {
        let w: f64 = self.pieces().map(|q| q.curve.winding(p)).sum();
        w > std::f64::consts::PI
    }

    /// Unsigned distance to the boundary.
    pub fn boundary_distance(&self, p: P2) -> f64 {
        self.pieces().map(|q| q.curve.nearest(p).1).fold(f64::INFINITY, f64::min)
    }

    /// Distance to the boundary, positive inside.
    pub fn signed_distance(&self, p: P2) -> f64 {
        let d = self.boundary_distance(p);
        if self.contains(p) { d } else { -d }
    }

    /// A point strictly inside the region.
    pub fn interior_point(&self) -> Option<P2> {
        let scale_len = self.area.abs().sqrt().max(1e-12);
        for piece in &self.outer {
            for frac in [0.5, 0.25, 0.75] {
                let m = piece.curve.point(frac);
                let t = piece.curve.tangent(frac);
                let n = [-t[1], t[0]];
                for k in 1..8 {
                    let d = scale_len * 1e-2 / (1u32 << k) as f64;
                    let p = add(m, scale(n, d));
                    if self.contains(p) && self.boundary_distance(p) > 0.25 * d {
                        return Some(p);
                    }
                }
            }
        }
        None
    }

    /// Parameter intervals `(s0, s1)` of the line `o + s d` inside the region.
    pub fn line_intervals(&self, o: P2, d: P2) -> Vec<(f64, f64)> {
        self.line_intervals_with(o, d).into_iter().map(|(a, b, _, _)| (a, b)).collect()
    }

    /// The intervals of the line `o + s d` inside the region, each with the
    /// index (in `pieces()` order) of the boundary piece crossed at its
    /// start and at its end.
    pub fn line_intervals_with(&self, o: P2, d: P2) -> Vec<(f64, f64, usize, usize)> {
        let mut ss: Vec<(f64, usize)> = Vec::new();
        for (qi, q) in self.pieces().enumerate() {
            for (s, _) in q.curve.line_hits(o, d) {
                ss.push((s, qi));
            }
        }
        if ss.is_empty() {
            return Vec::new();
        }
        ss.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let dn = norm(d).max(1e-300);
        let mut out: Vec<(f64, f64, usize, usize)> = Vec::new();
        for w in ss.windows(2) {
            if (w[1].0 - w[0].0) * dn < 1e-12 {
                continue;
            }
            let mid = add(o, scale(d, 0.5 * (w[0].0 + w[1].0)));
            if self.contains(mid) {
                if let Some(last) = out.last_mut() {
                    if (w[0].0 - last.1).abs() * dn < 1e-9 {
                        last.1 = w[1].0;
                        last.3 = w[1].1;
                        continue;
                    }
                }
                out.push((w[0].0, w[1].0, w[0].1, w[1].1));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Arrangement
// ---------------------------------------------------------------------------

struct HalfEdge {
    from: usize,
    to: usize,
    curve: Curve,
    piece: usize, // index into the split pieces
    twin: usize,
}

struct Split {
    pieces: Vec<Piece>,
    hes: Vec<HalfEdge>,
    /// Outgoing half-edges per vertex, sorted counter-clockwise.
    out: Vec<Vec<usize>>,
    /// One position per vertex (the first end point snapped to it).
    verts: Vec<P2>,
}

/// The curve from `from` to `to` that is `c` but for its ends, each
/// within the tolerance of where `c` had it. A line is the line between
/// them; an arc stays on its circle, and ends on it abreast of them.
fn with_ends(c: &Curve, from: P2, to: P2) -> Curve {
    match *c {
        Curve::Line { .. } => Curve::Line { a: from, b: to },
        Curve::Arc { c: centre, r, a0, sweep } => {
            if sweep.abs() >= std::f64::consts::TAU - 1e-9 || (dist(c.start(), from) == 0.0 && dist(c.end(), to) == 0.0) {
                return *c;
            }
            let near = |angle: f64, old: f64| old + (angle - old + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
            let b0 = near((from[1] - centre[1]).atan2(from[0] - centre[0]), a0);
            let b1 = near((to[1] - centre[1]).atan2(to[0] - centre[0]), a0 + sweep);
            Curve::Arc { c: centre, r, a0: b0, sweep: b1 - b0 }
        }
    }
}

fn angle_key(h: &HalfEdge) -> (f64, f64) {
    let t = h.curve.tangent(0.0);
    (t[1].atan2(t[0]), h.curve.curvature())
}

fn build_split(input: &[Piece], eps: f64) -> Split {
    // 1. Intersection parameters per piece.
    let n = input.len();
    let mut params: Vec<Vec<f64>> = vec![vec![0.0, 1.0]; n];
    let boxes: Vec<[f64; 4]> = input.iter().map(|p| p.curve.bbox()).collect();
    for i in 0..n {
        for j in i + 1..n {
            let a = boxes[i];
            let b = boxes[j];
            if a[2] + eps < b[0] || b[2] + eps < a[0] || a[3] + eps < b[1] || b[3] + eps < a[1] {
                continue;
            }
            for (t, u) in geom::intersect(&input[i].curve, &input[j].curve, eps) {
                params[i].push(t);
                params[j].push(u);
            }
        }
    }
    // A vertex within eps of a piece is on it. Two computations of one
    // curve (a sketch line and the edge of a surface fitted up to it, a
    // crease traced from either side) differ by their tolerances, run
    // beside each other and cross where chance has it; split at each
    // other's vertices they are the same edges twice, and one edge below.
    let mut points: Vec<P2> = Vec::new();
    for (i, p) in input.iter().enumerate() {
        for &t in &params[i] {
            points.push(p.curve.point(t));
        }
    }
    if !points.is_empty() {
        const N: usize = 64;
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        for q in &points {
            for k in 0..2 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
        }
        let width = [((hi[0] - lo[0]) / N as f64).max(1e-300), ((hi[1] - lo[1]) / N as f64).max(1e-300)];
        let cell = |v: f64, k: usize| (((v - lo[k]) / width[k]).floor().max(0.0) as usize).min(N - 1);
        let mut grid: Vec<Vec<usize>> = vec![Vec::new(); N * N];
        for (k, q) in points.iter().enumerate() {
            grid[cell(q[1], 1) * N + cell(q[0], 0)].push(k);
        }
        for i in 0..n {
            let b = boxes[i];
            for row in cell(b[1] - eps, 1)..=cell(b[3] + eps, 1) {
                for col in cell(b[0] - eps, 0)..=cell(b[2] + eps, 0) {
                    for &k in &grid[row * N + col] {
                        let q = points[k];
                        if q[0] < b[0] - eps || q[0] > b[2] + eps || q[1] < b[1] - eps || q[1] > b[3] + eps {
                            continue;
                        }
                        let (t, d) = input[i].curve.nearest(q);
                        if d <= eps {
                            params[i].push(t);
                        }
                    }
                }
            }
        }
    }
    // 2. Split.
    let mut pieces: Vec<Piece> = Vec::new();
    let mut ends: Vec<P2> = Vec::new();
    for (i, p) in input.iter().enumerate() {
        let len = p.curve.length();
        if len <= eps {
            continue;
        }
        let mut ts = params[i].clone();
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut cleaned: Vec<f64> = Vec::new();
        for t in ts {
            if let Some(&last) = cleaned.last() {
                if (t - last) * len <= eps {
                    continue;
                }
            }
            cleaned.push(t);
        }
        // Keep the end at 1 even when the last split lands within eps of it.
        if let Some(last) = cleaned.last_mut() {
            if *last < 1.0 {
                *last = 1.0;
            }
        }
        for w in cleaned.windows(2) {
            let c = p.curve.sub(w[0], w[1]);
            pieces.push(Piece { id: p.id.clone(), curve: c });
            ends.push(c.start());
            ends.push(c.end());
        }
    }
    // 3. Snap end points (union-find by distance).
    let m = ends.len();
    let mut parent: Vec<usize> = (0..m).collect();
    fn find(p: &mut Vec<usize>, i: usize) -> usize {
        let mut r = i;
        while p[r] != r {
            r = p[r];
        }
        let mut c = i;
        while p[c] != r {
            let nx = p[c];
            p[c] = r;
            c = nx;
        }
        r
    }
    // Bucket by a coarse grid to avoid the quadratic pass on big sketches.
    let cell = (4.0 * eps).max(1e-12);
    let mut buckets: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, e) in ends.iter().enumerate() {
        buckets.entry(((e[0] / cell).floor() as i64, (e[1] / cell).floor() as i64)).or_default().push(i);
    }
    for (i, e) in ends.iter().enumerate() {
        let cx = (e[0] / cell).floor() as i64;
        let cy = (e[1] / cell).floor() as i64;
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(list) = buckets.get(&(cx + dx, cy + dy)) {
                    for &j in list {
                        if j > i && dist(ends[i], ends[j]) <= eps {
                            let ri = find(&mut parent, i);
                            let rj = find(&mut parent, j);
                            if ri != rj {
                                parent[ri] = rj;
                            }
                        }
                    }
                }
            }
        }
    }
    let mut vid: HashMap<usize, usize> = HashMap::new();
    let mut vertex_of = vec![0usize; m];
    for i in 0..m {
        let r = find(&mut parent, i);
        let next = vid.len();
        let v = *vid.entry(r).or_insert(next);
        vertex_of[i] = v;
    }
    let nv = vid.len();
    let mut verts: Vec<Option<P2>> = vec![None; nv];
    for i in 0..m {
        verts[vertex_of[i]].get_or_insert(ends[i]);
    }
    let verts: Vec<P2> = verts.into_iter().map(|v| v.unwrap_or([0.0, 0.0])).collect();
    // 4. Half-edges. Sub-pieces joining the same two vertices along the same
    // curve (a sketch line drawn twice, two rectangles sharing a side) are
    // one edge; the duplicate is dropped so it cannot separate a face from
    // itself. Loops (u == v) are full circles and are always distinct.
    // (The middles are compared by distance: rounded to a grid, two
    // within a rounding error of each other can fall either side of a
    // grid line.)
    // An edge runs from its vertex to its vertex: what comes out of one
    // arrangement goes into the next, where ends that were one vertex
    // here have to be one again.
    for (k, p) in pieces.iter_mut().enumerate() {
        p.curve = with_ends(&p.curve, verts[vertex_of[2 * k]], verts[vertex_of[2 * k + 1]]);
    }
    let mut seen: HashMap<(usize, usize), Vec<P2>> = HashMap::new();
    let mut hes: Vec<HalfEdge> = Vec::with_capacity(2 * pieces.len());
    for (k, p) in pieces.iter().enumerate() {
        let u = vertex_of[2 * k];
        let v = vertex_of[2 * k + 1];
        if u != v {
            let (lo, hi) = if u < v { (u, v) } else { (v, u) };
            let mid = p.curve.point(0.5);
            let mids = seen.entry((lo, hi)).or_default();
            if mids.iter().any(|m| dist(*m, mid) <= 2.0 * eps) {
                continue;
            }
            mids.push(mid);
        } else if !matches!(p.curve, Curve::Arc { sweep, .. } if sweep.abs() >= std::f64::consts::TAU - 1e-9) {
            // Both ends at one vertex and not a circle: shorter than
            // points are apart.
            continue;
        }
        let f = hes.len();
        hes.push(HalfEdge { from: u, to: v, curve: p.curve, piece: k, twin: f + 1 });
        hes.push(HalfEdge { from: v, to: u, curve: p.curve.reversed(), piece: k, twin: f });
    }
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); nv];
    for (h, he) in hes.iter().enumerate() {
        out[he.from].push(h);
    }
    // Tangent junctions (fillet arcs meeting lines, arcs touching circles)
    // leave a vertex in the same direction up to the snapping error, which
    // for an arc of radius r is about eps / r in angle; there the curvature
    // orders them (the piece curving left is counter-clockwise of the one
    // curving right).
    let angle_tol = 1e-5f64.max(100.0 * eps);
    for list in out.iter_mut() {
        // Measure angles from the middle of the largest empty gap around the
        // vertex, so no tangent pair straddles the wrap of atan2.
        let mut angles: Vec<f64> = list.iter().map(|&h| angle_key(&hes[h]).0).collect();
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut base = 0.0;
        if angles.len() >= 2 {
            let mut best_gap = -1.0;
            for i in 0..angles.len() {
                let a = angles[i];
                let b = if i + 1 < angles.len() { angles[i + 1] } else { angles[0] + std::f64::consts::TAU };
                if b - a > best_gap {
                    best_gap = b - a;
                    base = 0.5 * (a + b);
                }
            }
        }
        let key = |h: usize| -> (f64, f64) {
            let (ang, curv) = angle_key(&hes[h]);
            ((ang - base).rem_euclid(std::f64::consts::TAU), curv)
        };
        list.sort_by(|&a, &b| {
            let ka = key(a);
            let kb = key(b);
            let d = ka.0 - kb.0;
            if d.abs() > angle_tol {
                d.partial_cmp(&0.0).unwrap()
            } else {
                ka.1.partial_cmp(&kb.1).unwrap()
            }
        });
    }
    Split { pieces, hes, out, verts }
}

struct Face {
    /// The loop as walked: a line that ends inside the face is in it
    /// out and back.
    pieces: Vec<Piece>,
    /// The loop without the edges it has on both sides: what bounds
    /// the face.
    around: Vec<Piece>,
    hes: Vec<usize>,
    area: f64,
}

fn walk_faces(split: &Split) -> Vec<Face> {
    let mut visited = vec![false; split.hes.len()];
    let mut faces = Vec::new();
    for start in 0..split.hes.len() {
        if visited[start] {
            continue;
        }
        let mut loop_hes = Vec::new();
        let mut h = start;
        loop {
            visited[h] = true;
            loop_hes.push(h);
            let he = &split.hes[h];
            let list = &split.out[he.to];
            let i = list.iter().position(|&x| x == he.twin).expect("twin listed at its origin");
            let next = list[(i + list.len() - 1) % list.len()];
            h = next;
            if h == start {
                break;
            }
            if visited[h] {
                // Should not happen in a consistent walk; bail out.
                break;
            }
        }
        let piece_of = |h: usize| Piece { id: split.pieces[split.hes[h].piece].id.clone(), curve: split.hes[h].curve };
        let pieces: Vec<Piece> = loop_hes.iter().map(|&h| piece_of(h)).collect();
        // A line that ends inside the face (a construction line left in
        // a sketch) is walked out and back, and bounds nothing.
        let mut around: Vec<usize> = Vec::with_capacity(loop_hes.len());
        for &h in &loop_hes {
            if around.last().is_some_and(|&last| split.hes[last].twin == h) {
                around.pop();
            } else {
                around.push(h);
            }
        }
        while around.len() > 1 && split.hes[around[around.len() - 1]].twin == around[0] {
            around.pop();
            around.remove(0);
        }
        let around: Vec<Piece> = around.into_iter().map(piece_of).collect();
        let area = loop_area(&pieces);
        faces.push(Face { pieces, around, hes: loop_hes, area });
    }
    faces
}

/// A point just inside the face side (left) of a loop's first piece.
fn face_probe(pieces: &[Piece], scale_len: f64) -> P2 {
    let p = &pieces[0];
    let m = p.curve.point(0.5);
    let t = p.curve.tangent(0.5);
    add(m, scale([-t[1], t[0]], scale_len))
}

fn assemble_regions(faces: Vec<Face>, eps: f64) -> Vec<Region> {
    let mut positives: Vec<Region> = Vec::new();
    let mut pos_index: Vec<usize> = Vec::new();
    for (i, f) in faces.iter().enumerate() {
        if f.area > eps * eps {
            // (The ids are of all that was walked, as they were before
            // such lines were left out of the boundary: regions are
            // looked up by them.)
            let ids = f.pieces.iter().map(|p| p.id.clone()).collect();
            positives.push(Region { outer: f.around.clone(), holes: Vec::new(), ids, area: f.area });
            pos_index.push(i);
        }
    }
    for f in faces.iter() {
        if f.area > -(eps * eps) {
            continue;
        }
        let probe = face_probe(&f.around, 4.0 * eps);
        let mut best: Option<(usize, f64)> = None;
        for (k, r) in positives.iter().enumerate() {
            if loop_winding(&r.outer, probe) > std::f64::consts::PI {
                if best.is_none_or(|(_, a)| r.area < a) {
                    best = Some((k, r.area));
                }
            }
        }
        if let Some((k, _)) = best {
            let r = &mut positives[k];
            r.area += f.area;
            for p in &f.pieces {
                r.ids.insert(p.id.clone());
            }
            r.holes.push(f.around.clone());
        }
    }
    positives
}

/// The bounded faces of the arrangement of `pieces`. `eps` is the distance
/// within which end points are considered the same vertex.
pub fn arrange(pieces: &[Piece], eps: f64) -> Vec<Region> {
    let split = build_split(pieces, eps);
    let faces = walk_faces(&split);
    assemble_regions(faces, eps)
}

/// One edge of an [`Arrangement`]: a piece between two vertices with the
/// bounded faces on its two sides.
#[derive(Clone, Debug)]
pub struct ArrEdge {
    /// The id of the input piece it is part of.
    pub id: String,
    /// From `from` to `to`.
    pub curve: Curve,
    pub from: usize,
    pub to: usize,
    /// Indices into [`Arrangement::faces`]; `None` is the unbounded face.
    pub left: Option<usize>,
    pub right: Option<usize>,
}

/// The arrangement of a set of pieces with its adjacency: what
/// [`arrange`] walks, kept.
#[derive(Clone, Debug)]
pub struct Arrangement {
    pub vertices: Vec<P2>,
    pub edges: Vec<ArrEdge>,
    pub faces: Vec<Region>,
}

/// The arrangement of `pieces` (see [`arrange`] for `eps`). Pieces that
/// coincide are one edge, under the id of the first. The faces are as
/// they were walked: a line that ends inside a face is in its boundary,
/// out and back (in [`arrange`]'s regions it is not), so that a point
/// far from a face's boundary is far from every edge.
pub fn arrangement(pieces: &[Piece], eps: f64) -> Arrangement {
    let split = build_split(pieces, eps);
    let loops = walk_faces(&split);
    // Bounded faces are the positive loops; a negative loop is the outside
    // of a connected group of edges, a hole of the smallest face around it.
    let mut face_of_loop: Vec<Option<usize>> = vec![None; loops.len()];
    let mut faces: Vec<Region> = Vec::new();
    for (i, f) in loops.iter().enumerate() {
        if f.area > eps * eps {
            let ids = f.pieces.iter().map(|p| p.id.clone()).collect();
            face_of_loop[i] = Some(faces.len());
            faces.push(Region { outer: f.pieces.clone(), holes: Vec::new(), ids, area: f.area });
        }
    }
    for (i, f) in loops.iter().enumerate() {
        if face_of_loop[i].is_some() {
            continue;
        }
        // (A loop around no area is lines that end in the face they are
        // in: its edges have that face on both sides.)
        let first = Piece { id: String::new(), curve: split.hes[f.hes[0]].curve };
        let probe = face_probe(&[first], 4.0 * eps);
        let mut best: Option<(usize, f64)> = None;
        for (k, r) in faces.iter().enumerate() {
            if loop_winding(&r.outer, probe) > std::f64::consts::PI && best.is_none_or(|(_, a)| loop_area(&r.outer) < a) {
                best = Some((k, loop_area(&r.outer)));
            }
        }
        if let Some((k, _)) = best {
            face_of_loop[i] = Some(k);
            let r = &mut faces[k];
            for p in &f.pieces {
                r.ids.insert(p.id.clone());
            }
            r.area += f.area;
            r.holes.push(f.pieces.clone());
        }
    }
    let mut face_of_he: Vec<Option<usize>> = vec![None; split.hes.len()];
    for (i, f) in loops.iter().enumerate() {
        for &h in &f.hes {
            face_of_he[h] = face_of_loop[i];
        }
    }
    let edges = (0..split.hes.len() / 2)
        .map(|e| {
            let he = &split.hes[2 * e];
            ArrEdge {
                id: split.pieces[he.piece].id.clone(),
                curve: he.curve,
                from: he.from,
                to: he.to,
                left: face_of_he[2 * e],
                right: face_of_he[2 * e + 1],
            }
        })
        .collect();
    Arrangement { vertices: split.verts, edges, faces }
}

// ---------------------------------------------------------------------------
// Booleans
// ---------------------------------------------------------------------------

fn any_contains(regions: &[Region], p: P2) -> bool {
    regions.iter().any(|r| r.contains(p))
}

fn boolean(a: &[Region], b: &[Region], keep: impl Fn(bool, bool) -> bool, eps: f64) -> Vec<Region> {
    let mut pieces: Vec<Piece> = Vec::new();
    for r in a.iter().chain(b.iter()) {
        pieces.extend(r.pieces().cloned());
    }
    let split = build_split(&pieces, eps);
    let faces = walk_faces(&split);
    // Classify each half-edge's face.
    let mut face_of_he = vec![usize::MAX; split.hes.len()];
    let mut kept = vec![false; faces.len()];
    for (fi, f) in faces.iter().enumerate() {
        for &h in &f.hes {
            face_of_he[h] = fi;
        }
        if f.area <= eps * eps {
            continue;
        }
        let probe_region = Region { outer: f.pieces.clone(), holes: Vec::new(), ids: BTreeSet::new(), area: f.area };
        let Some(p) = probe_region.interior_point() else { continue };
        kept[fi] = keep(any_contains(a, p), any_contains(b, p));
    }
    // Holes of kept faces are negative loops whose probe lies in a kept
    // positive face; a boundary piece survives when exactly one side is kept.
    // Decide per half-edge: its own face kept (positive faces directly; a
    // negative loop takes the status of the positive face enclosing it).
    let mut neg_status = vec![false; faces.len()];
    for (fi, f) in faces.iter().enumerate() {
        if f.area < -(eps * eps) {
            let probe = face_probe(&f.pieces, 4.0 * eps);
            let mut best: Option<(usize, f64)> = None;
            for (k, g) in faces.iter().enumerate() {
                if g.area > eps * eps && loop_winding(&g.pieces, probe) > std::f64::consts::PI {
                    if best.is_none_or(|(_, ar)| g.area < ar) {
                        best = Some((k, g.area));
                    }
                }
            }
            neg_status[fi] = best.map(|(k, _)| kept[k]).unwrap_or(false);
        }
    }
    let status = |h: usize| -> bool {
        let fi = face_of_he[h];
        if fi == usize::MAX {
            return false;
        }
        if faces[fi].area > eps * eps { kept[fi] } else { neg_status[fi] }
    };
    let mut boundary: Vec<Piece> = Vec::new();
    for (h, he) in split.hes.iter().enumerate() {
        if h % 2 == 1 {
            continue;
        }
        let s1 = status(h);
        let s2 = status(he.twin);
        if s1 != s2 {
            boundary.push(Piece { id: split.pieces[he.piece].id.clone(), curve: he.curve });
        }
    }
    // The arrangement of the surviving boundary also yields the faces
    // inside holes; keep only faces the predicate accepts.
    arrange(&boundary, eps)
        .into_iter()
        .filter(|r| r.interior_point().is_some_and(|p| keep(any_contains(a, p), any_contains(b, p))))
        .collect()
}

pub fn union(a: &[Region], b: &[Region], eps: f64) -> Vec<Region> {
    boolean(a, b, |x, y| x || y, eps)
}

pub fn difference(a: &[Region], b: &[Region], eps: f64) -> Vec<Region> {
    boolean(a, b, |x, y| x && !y, eps)
}

pub fn intersection(a: &[Region], b: &[Region], eps: f64) -> Vec<Region> {
    boolean(a, b, |x, y| x && y, eps)
}

/// Flatten an ellipse (centre, semi-axes, major-axis direction) into `n`
/// line pieces tagged `id`.
pub fn ellipse_pieces(id: &str, c: P2, major: f64, minor: f64, dir: P2, n: usize) -> Vec<Piece> {
    let d = scale(dir, 1.0 / norm(dir).max(1e-300));
    let e = [-d[1], d[0]];
    let pt = |k: usize| {
        let th = std::f64::consts::TAU * k as f64 / n as f64;
        add(c, add(scale(d, major * th.cos()), scale(e, minor * th.sin())))
    };
    (0..n).map(|k| Piece { id: id.to_string(), curve: Curve::Line { a: pt(k), b: pt(k + 1) } }).collect()
}

/// Round a corner of a loop: replace the joint of consecutive pieces `i` and
/// `i + 1` (which must be lines) by a tangent arc of radius `r`. Returns
/// `false` when the corner cannot take the arc.
pub fn fillet_corner(pieces: &mut Vec<Piece>, i: usize, r: f64, id: &str) -> bool {
    let n = pieces.len();
    let j = (i + 1) % n;
    let (Curve::Line { a: a0, b: b0 }, Curve::Line { a: a1, b: b1 }) = (pieces[i].curve, pieces[j].curve) else {
        return false;
    };
    let d0 = sub(b0, a0);
    let d1 = sub(b1, a1);
    let l0 = norm(d0);
    let l1 = norm(d1);
    if l0 == 0.0 || l1 == 0.0 {
        return false;
    }
    let u0 = scale(d0, 1.0 / l0);
    let u1 = scale(d1, 1.0 / l1);
    let turn = cross(u0, u1).atan2(dot(u0, u1)); // signed turning angle at the corner
    if turn.abs() < 1e-9 || turn.abs() > std::f64::consts::PI - 1e-9 {
        return false;
    }
    let tlen = r * (0.5 * turn.abs()).tan();
    if tlen > l0 + 1e-12 || tlen > l1 + 1e-12 {
        return false;
    }
    let corner = b0;
    let p0 = sub(corner, scale(u0, tlen));
    let p1 = add(corner, scale(u1, tlen));
    let ccw = turn > 0.0;
    let nrm = if ccw { [-u0[1], u0[0]] } else { [u0[1], -u0[0]] };
    let centre = add(p0, scale(nrm, r));
    pieces[i].curve = Curve::Line { a: a0, b: p0 };
    pieces[j].curve = Curve::Line { a: p1, b: b1 };
    let arc = Piece { id: id.to_string(), curve: Curve::arc_from_ends(centre, p0, p1, ccw) };
    pieces.insert(i + 1, arc);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn arrangement_knows_both_sides() {
        // Two squares sharing a side, a circle inside the left one.
        let mut pieces = rect("a", 0.0, 0.0, 1.0, 1.0);
        pieces.extend(rect("b", 1.0, 0.0, 2.0, 1.0));
        pieces.push(Piece { id: "c".into(), curve: Curve::circle([0.5, 0.5], 0.25) });
        let arr = arrangement(&pieces, 1e-9);
        assert_eq!(arr.faces.len(), 3);
        assert_eq!(arr.edges.len(), 8, "the shared side is one edge");
        let face_at = |p: P2| arr.faces.iter().position(|r| r.contains(p)).unwrap();
        let (left, right, disc) = (face_at([0.1, 0.1]), face_at([1.5, 0.5]), face_at([0.5, 0.5]));
        let shared: Vec<&ArrEdge> = arr.edges.iter().filter(|e| (e.curve.point(0.5)[0] - 1.0).abs() < 1e-12).collect();
        assert_eq!(shared.len(), 1);
        let mut sides = [shared[0].left, shared[0].right];
        sides.sort();
        let mut expect = [Some(left), Some(right)];
        expect.sort();
        assert_eq!(sides, expect);
        let ring = arr.edges.iter().find(|e| e.id == "c").unwrap();
        assert_eq!(ring.from, ring.to);
        let mut sides = [ring.left, ring.right];
        sides.sort();
        let mut expect = [Some(left), Some(disc)];
        expect.sort();
        assert_eq!(sides, expect);
        // Every outer edge has the unbounded face on one side.
        assert_eq!(arr.edges.iter().filter(|e| e.left.is_none() || e.right.is_none()).count(), 6);
        assert!((arr.faces[left].area - (1.0 - PI * 0.0625)).abs() < 1e-12);
        for e in &arr.edges {
            assert!(dist(arr.vertices[e.from], e.curve.start()) < 1e-9);
            assert!(dist(arr.vertices[e.to], e.curve.end()) < 1e-9);
        }
    }

    fn line(id: &str, a: P2, b: P2) -> Piece {
        Piece { id: id.to_string(), curve: Curve::Line { a, b } }
    }

    fn rect(prefix: &str, x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<Piece> {
        vec![
            line(&format!("{prefix}.bottom"), [x0, y0], [x1, y0]),
            line(&format!("{prefix}.right"), [x1, y0], [x1, y1]),
            line(&format!("{prefix}.top"), [x1, y1], [x0, y1]),
            line(&format!("{prefix}.left"), [x0, y1], [x0, y0]),
        ]
    }

    #[test]
    fn rectangle_with_circle_hole() {
        let mut pieces = rect("R", 0.0, 0.0, 4.0, 3.0);
        pieces.push(Piece { id: "C".into(), curve: Curve::circle([2.0, 1.5], 1.0) });
        let regs = arrange(&pieces, 1e-9);
        assert_eq!(regs.len(), 2);
        let disc = regs.iter().find(|r| r.holes.is_empty()).unwrap();
        let ring = regs.iter().find(|r| !r.holes.is_empty()).unwrap();
        assert!((disc.area - PI).abs() < 1e-9);
        assert!((ring.area - (12.0 - PI)).abs() < 1e-9);
        assert_eq!(ring.ids.len(), 5);
        assert!(ring.contains([0.5, 0.5]) && !ring.contains([2.0, 1.5]));
        assert!(disc.contains([2.0, 1.5]));
        assert!((ring.signed_distance([0.5, 0.5]) - 0.5).abs() < 1e-12);
        assert!((ring.signed_distance([2.0, 1.5]) + 1.0).abs() < 1e-12);
        let iv = ring.line_intervals([-1.0, 1.5], [1.0, 0.0]);
        assert_eq!(iv.len(), 2);
        assert!((iv[0].0 - 1.0).abs() < 1e-9 && (iv[0].1 - 2.0).abs() < 1e-9);
        assert!((iv[1].0 - 4.0).abs() < 1e-9 && (iv[1].1 - 5.0).abs() < 1e-9);
    }

    #[test]
    fn crossing_lines_make_four_regions() {
        let mut pieces = rect("R", 0.0, 0.0, 2.0, 2.0);
        pieces.push(line("H", [-0.5, 1.0], [2.5, 1.0]));
        pieces.push(line("V", [1.0, -0.5], [1.0, 2.5]));
        let regs = arrange(&pieces, 1e-9);
        assert_eq!(regs.len(), 4);
        for r in &regs {
            assert!((r.area - 1.0).abs() < 1e-9);
            assert_eq!(r.ids.len(), 4, "{:?}", r.ids);
        }
    }

    #[test]
    fn rounded_rectangle_is_one_region() {
        // Lines and tangent quarter arcs meeting at shared vertices.
        let r = 0.5;
        let pieces = vec![
            line("b", [r, 0.0], [3.0 - r, 0.0]),
            Piece { id: "c1".into(), curve: Curve::arc_from_ends([3.0 - r, r], [3.0 - r, 0.0], [3.0, r], true) },
            line("r", [3.0, r], [3.0, 2.0 - r]),
            Piece { id: "c2".into(), curve: Curve::arc_from_ends([3.0 - r, 2.0 - r], [3.0, 2.0 - r], [3.0 - r, 2.0], true) },
            line("t", [3.0 - r, 2.0], [r, 2.0]),
            Piece { id: "c3".into(), curve: Curve::arc_from_ends([r, 2.0 - r], [r, 2.0], [0.0, 2.0 - r], true) },
            line("l", [0.0, 2.0 - r], [0.0, r]),
            Piece { id: "c4".into(), curve: Curve::arc_from_ends([r, r], [0.0, r], [r, 0.0], true) },
        ];
        let regs = arrange(&pieces, 1e-9);
        assert_eq!(regs.len(), 1);
        let expect = 6.0 - (4.0 - PI) * r * r;
        assert!((regs[0].area - expect).abs() < 1e-9, "{}", regs[0].area);
        assert!(regs[0].holes.is_empty());
        // Corner rounding of a plain rectangle gives the same shape.
        let mut sq = rect("S", 0.0, 0.0, 3.0, 2.0);
        for k in (0..4).rev() {
            assert!(fillet_corner(&mut sq, k, r, "f"));
        }
        let regs2 = arrange(&sq, 1e-9);
        assert_eq!(regs2.len(), 1);
        assert!((regs2[0].area - expect).abs() < 1e-9);
    }

    #[test]
    fn two_rectangles_sharing_a_side() {
        let mut pieces = rect("A", 0.0, 0.0, 1.0, 1.0);
        pieces.extend(rect("B", 1.0, 0.0, 2.0, 1.0));
        let regs = arrange(&pieces, 1e-9);
        assert_eq!(regs.len(), 2);
        for r in &regs {
            assert!((r.area - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn booleans_of_overlapping_rectangles() {
        let a = arrange(&rect("A", 0.0, 0.0, 2.0, 2.0), 1e-9);
        let b = arrange(&rect("B", 1.0, 1.0, 3.0, 3.0), 1e-9);
        let u = union(&a, &b, 1e-9);
        assert_eq!(u.len(), 1);
        assert!((u[0].area - 7.0).abs() < 1e-9);
        assert_eq!(u[0].outer.len(), 8);
        let d = difference(&a, &b, 1e-9);
        assert_eq!(d.len(), 1);
        assert!((d[0].area - 3.0).abs() < 1e-9);
        let i = intersection(&a, &b, 1e-9);
        assert_eq!(i.len(), 1);
        assert!((i[0].area - 1.0).abs() < 1e-9);
        // A hole punched through the middle.
        let c = arrange(&rect("C", 0.5, 0.5, 1.5, 1.5), 1e-9);
        let d2 = difference(&a, &c, 1e-9);
        assert_eq!(d2.len(), 1);
        assert_eq!(d2[0].holes.len(), 1);
        assert!((d2[0].area - 3.0).abs() < 1e-9);
        // Disjoint union keeps two regions.
        let e = arrange(&rect("E", 5.0, 5.0, 6.0, 6.0), 1e-9);
        assert_eq!(union(&a, &e, 1e-9).len(), 2);
    }
}




#[cfg(test)]
mod corner_tests {
    use super::*;

    fn rect(prefix: &str, x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<Piece> {
        let l = |id: String, a: P2, b: P2| Piece { id, curve: Curve::Line { a, b } };
        vec![
            l(format!("{prefix}.b"), [x0, y0], [x1, y0]),
            l(format!("{prefix}.r"), [x1, y0], [x1, y1]),
            l(format!("{prefix}.t"), [x1, y1], [x0, y1]),
            l(format!("{prefix}.l"), [x0, y1], [x0, y0]),
        ]
    }

    #[test]
    fn corner_fillet_after_hole_differences() {
        let mut base = arrange(&rect("B", 0.0, 0.0, 0.6858, 0.6858), 1e-7);
        for k in 0..16 {
            let x = 0.05 + 0.03 * k as f64;
            let hole = arrange(&rect(&format!("H{k}"), x, 0.2, x + 0.005, 0.25), 1e-7);
            base = difference(&base, &hole, 1e-7);
        }
        assert_eq!(base.len(), 1);
        let area0 = base[0].area;
        assert!((area0 - (0.6858 * 0.6858 - 16.0 * 0.005 * 0.05)).abs() < 1e-9);
        // Round every outer corner with r = 12.7 mm.
        let reg = &mut base[0];
        let mut k = 0;
        let mut rounded = 0;
        while k < reg.outer.len() && rounded < 8 {
            if fillet_corner(&mut reg.outer, k, 0.0127, "f") {
                rounded += 1;
                k += 2;
            } else {
                k += 1;
            }
        }
        assert_eq!(rounded, 4);
        // Re-arrange to recompute the area: four corners lose (1 - pi/4) r^2 each.
        let pieces: Vec<Piece> = reg.pieces().cloned().collect();
        let again = arrange(&pieces, 1e-7);
        let outer = again.iter().max_by(|a, b| a.area.partial_cmp(&b.area).unwrap()).unwrap();
        let expect = area0 - 4.0 * (1.0 - std::f64::consts::FRAC_PI_4) * 0.0127 * 0.0127;
        assert!((outer.area - expect).abs() < 1e-9, "{} vs {expect}", outer.area);
        assert!(outer.contains([0.3, 0.1]));
        assert!(outer.contains([0.0005, 0.3]));
        assert!(!outer.contains([0.001, 0.001]));
        assert_eq!(outer.holes.len(), 16);
    }
}




#[cfg(test)]
mod dangling {
    use super::*;

    #[test]
    fn a_line_that_ends_inside_a_face_bounds_nothing() {
        let l = |id: &str, a: P2, b: P2| Piece { id: id.into(), curve: Curve::Line { a, b } };
        // A square, a line from the middle of its top side half way
        // down, and a line inside it that touches nothing.
        let pieces = vec![
            l("b", [0.0, 0.0], [1.0, 0.0]),
            l("r", [1.0, 0.0], [1.0, 1.0]),
            l("t", [1.0, 1.0], [0.0, 1.0]),
            l("l", [0.0, 1.0], [0.0, 0.0]),
            l("spur", [0.5, 1.0], [0.5, 0.5]),
            l("loose", [0.2, 0.2], [0.3, 0.2]),
        ];
        let regions = arrange(&pieces, 1e-9);
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        assert!((r.area - 1.0).abs() < 1e-12);
        assert!(r.holes.is_empty(), "{:?}", r.holes);
        // The top side in two parts, the others whole.
        assert_eq!(r.outer.len(), 5, "{:?}", r.outer);
        assert!(r.outer.iter().all(|p| p.id != "spur"));
        let around: f64 = r.outer.iter().map(|p| p.curve.length()).sum();
        assert!((around - 4.0).abs() < 1e-12);
        // The arrangement has the lines as edges, the face on both
        // sides.
        let arr = arrangement(&pieces, 1e-9);
        assert_eq!(arr.faces.len(), 1);
        for id in ["spur", "loose"] {
            let e = arr.edges.iter().find(|e| e.id == id).unwrap();
            assert_eq!((e.left, e.right), (Some(0), Some(0)), "{id}");
        }
    }
}

#[cfg(test)]
mod twice {
    use super::*;

    fn l(id: &str, a: P2, b: P2) -> Piece {
        Piece { id: id.into(), curve: Curve::Line { a, b } }
    }

    #[test]
    fn a_second_drawing_of_a_side() {
        // A square, and its top side drawn again a little off and cut in
        // three, overhanging at one end.
        let mut pieces = vec![l("b", [0.0, 0.0], [1.0, 0.0]), l("r", [1.0, 0.0], [1.0, 1.0]), l("t", [1.0, 1.0], [0.0, 1.0]), l("l", [0.0, 1.0], [0.0, 0.0])];
        pieces.push(l("again", [-0.5, 1.0 + 3e-7], [0.3, 1.0 - 2e-7]));
        pieces.push(l("again", [0.3, 1.0 - 2e-7], [0.7, 1.0 + 4e-7]));
        pieces.push(l("again", [0.7, 1.0 + 4e-7], [1.0, 1.0]));
        let arr = arrangement(&pieces, 1e-6);
        assert_eq!(arr.faces.len(), 1);
        assert!((arr.faces[0].area - 1.0).abs() < 1e-9);
        // The side is the first drawing's, in the parts the second cut
        // it into (at its two joints, and where it crosses between
        // them); of the second there is the overhang.
        let of = |id: &str| arr.edges.iter().filter(|e| e.id == id).count();
        assert_eq!((of("t"), of("again")), (4, 1), "{:?}", arr.edges);
        let regions = arrange(&pieces, 1e-6);
        assert_eq!(regions.len(), 1);
    }

    #[test]
    fn a_crossing_at_a_small_angle() {
        // A line across a square's side at a thousandth of a radian,
        // cut short within the tolerance of the side on its way in and
        // on its way out: it meets the side at both ends.
        let mut pieces = vec![l("b", [0.0, 0.0], [1.0, 0.0]), l("r", [1.0, 0.0], [1.0, 1.0]), l("t", [1.0, 1.0], [0.0, 1.0]), l("l", [0.0, 1.0], [0.0, 0.0])];
        pieces.push(l("m", [0.0, 0.5], [1.0, 0.5]));
        pieces.push(l("in", [0.2, 0.0], [0.5 - 5e-4, 0.5 - 5e-7]));
        pieces.push(l("out", [0.5 + 5e-4, 0.5 + 5e-7], [0.8, 1.0]));
        let arr = arrangement(&pieces, 1e-6);
        assert_eq!(arr.faces.len(), 4, "{:?}", arr.edges);
    }

    #[test]
    fn arcs_of_one_circle() {
        // A quarter of a circle and, about a centre a little off, the
        // eighth before its end and the quarter after: one curve.
        let first = Piece { id: "a".into(), curve: Curve::arc_from_ends([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], true) };
        let c = [2e-7, -1e-7];
        let second = Piece { id: "b".into(), curve: Curve::arc_from_ends(c, [c[0] + 0.5f64.sqrt(), c[1] + 0.5f64.sqrt()], [c[0] - 1.0, c[1]], true) };
        let pieces = vec![first, second, l("x", [-1.0, 0.0], [1.0, 0.0])];
        let arr = arrangement(&pieces, 1e-6);
        assert_eq!(arr.faces.len(), 1, "{:?}", arr.edges);
        assert!((arr.faces[0].area - std::f64::consts::FRAC_PI_2).abs() < 1e-5);
    }
}

#[cfg(test)]
mod strips {
    use super::*;

    #[test]
    fn coincident_pieces_with_middles_a_rounding_error_apart() {
        // Three strips side by side, their ends drawn twice: once per
        // strip and once across all three. The middle of the third
        // strip's end is 0.24209375, half way between two multiples of
        // the tolerance.
        let l = |id: &str, a: P2, b: P2| Piece { id: id.into(), curve: Curve::Line { a, b } };
        let pieces = vec![
            l("s0", [0.17780000000000057, 8.4826], [0.2016125000000003, 8.4826]),
            l("s1", [0.2016125, 8.4826], [0.2301875, 8.4826]),
            l("s2", [0.23018749999999985, 8.4826], [0.2539999999999998, 8.4826]),
            l("s3", [0.254, 8.4826], [0.1778, 8.4826]),
            l("e0", [0.17780000000000057, 0.3302], [0.2016125000000003, 0.3302]),
            l("e1", [0.2016125, 0.3302], [0.2301875, 0.3302]),
            l("e2", [0.23018749999999985, 0.3302], [0.2539999999999998, 0.3302]),
            l("e3", [0.254, 0.3302], [0.1778, 0.3302]),
            l("p0", [0.17780000000000057, 8.4826], [0.17780000000000057, 0.3302]),
            l("p1", [0.2016125, 8.4826], [0.2016125, 0.3302]),
            l("p2", [0.23018749999999985, 8.4826], [0.23018749999999985, 0.3302]),
            l("p3", [0.254, 8.4826], [0.254, 0.3302]),
        ];
        let arr = arrangement(&pieces, 1e-7);
        assert_eq!(arr.faces.len(), 3);
        assert_eq!(arr.edges.len(), 10);
        assert_eq!(arrange(&pieces, 1e-7).len(), 3);
    }
}
