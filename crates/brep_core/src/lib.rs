//! Boundary-representation solids as a sampleable payload: the data
//! contract between `step_import_operator` (which builds it) and
//! `brep_model_template` (which classifies points against it at sample
//! time).
//!
//! Like `trimesh_model_core`, the operator does all the work up front —
//! trim-loop projection, BVH construction, and serialization happen at
//! import time in [`payload::build_payload`] — and the generated model is
//! stateless: [`payload::PayloadView`] only reads. The IR ([`ir`]) and the
//! payload are internal to this crate and its two consumers; they are
//! deliberately NOT an ABI value type, so the surface vocabulary can grow
//! with importer coverage without touching the operator protocol.
//!
//! # Representation
//!
//! A [`ir::BRepModel`] is a set of solids — each a collection of trimmed
//! faces — plus placed instances (solid index + rigid transform + label).
//! Surfaces are exact analytic geometry (plane, cylinder, cone, sphere,
//! torus, extrusion of a polyline profile, NURBS); the only toleranced
//! data are the trim loops, which importers flatten into UV polylines.
//! Trim error is tangential along the exact surface, far below any
//! downstream meshing resolution.
//!
//! [`topo`] is the other representation of the same solids: shared
//! vertices and edges with exact curves, which is what a constructing
//! kernel builds and a STEP writer needs. It lowers into the IR.
//!
//! # Inside/outside semantics
//!
//! Occupancy is the union over instances. Per solid, classification is
//! crossing parity along a fixed skewed ray (the `trimesh_model_core`
//! approach), where a crossing is a ray–surface intersection whose UV
//! point lies inside the face's trim region (even-odd over the loops —
//! deliberately orientation-blind, so STEP sense flags never matter).
//! Kernels flag *suspect* casts — tangent hits, hits near a trim
//! boundary, near-pole/apex hits, unconverged Newton — and the query
//! re-casts along fallback directions before accepting a suspect parity.
//!
//! # Payload layout (little-endian; all sections 8-aligned)
//!
//! ```text
//! Header (72 bytes):
//!    0  magic          u32   "BRP1" (0x3150_5242)
//!    4  payload_len    u32   total byte length, header included
//!    8  instance_count u32
//!   12  solid_count    u32
//!   16  world bounds   6xf64 [min_x, max_x, min_y, max_y, min_z, max_z]
//!   64  instances_off  u32
//!   68  solids_off     u32   solid index: solid_count x u32 blob offsets
//! Instance record (152 bytes):
//!    0  solid          u32
//!    4  (pad)          u32
//!    8  world_aabb     6xf64
//!   56  world_to_local 12xf64 row-major 3x4 (rigid, precomputed inverse)
//! Solid blob:
//!    0  face_count   u32
//!    4  node_count   u32
//!    8  t_eps        f64  on-surface / crossing-count threshold
//!   16  eps_boundary f64  suspicion (re-cast) tolerance
//!   24  BVH nodes, node_count x 32 bytes:
//!         0  aabb_min 3xf32   (conservative over the trimmed face)
//!        12  aabb_max 3xf32
//!        24  a u32   internal: left child; leaf: 0x8000_0000 | first slot
//!        28  b u32   internal: right child; leaf: slot count
//!    .  slot table, face_count x u32: BVH leaf slots -> face offsets
//!       (from solid blob start), padded to 8
//!    .  face records
//! Face record:
//!    0  surface_type u32  0 plane / 1 cylinder / 2 cone / 3 sphere /
//!                         4 torus / 5 extrusion / 6 nurbs / 7 mesh
//!    4  color        u32  0 = unstyled, else 0xFFrrggbb (8-bit sRGB)
//!    8  uv_eps       2xf64  UV distance per axis equivalent to the 3D
//!                           suspicion tolerance at this face's scale
//!                           (0 for mesh faces, which have no UV space)
//!   24  u_period     f64    0 = aperiodic
//!   32  v_period     f64    0 = aperiodic
//!   40  trims_off    u32    from face start
//!   44  (pad)        u32
//!   48  surface data (type-specific, see `payload::write_surface`):
//!       plane      frame 12xf64
//!       cylinder   frame, radius
//!       cone       frame, ref_radius, tan_half_angle
//!       sphere     frame, radius
//!       torus      frame, major, minor
//!       extrusion  frame, count u32 + pad, profile count x 2xf64
//!                  (profile in frame XY, extruded along frame Z; u is
//!                  the polyline parameter — segment index + fraction —
//!                  and v the extrusion coordinate)
//!       nurbs      degree_u/v u32, nctrl_u/v u32, nknot_u/v u32,
//!                  seed_nu/nv u32, knots_u, knots_v,
//!                  ctrl (nctrl_u*nctrl_v x 4xf64 xyzw, v-fastest),
//!                  seed boxes seed_nu*seed_nv x 32 bytes
//!                  (aabb 6xf32, uv center 2xf32)
//!       mesh       vert_count u32, tri_count u32, seg_count u32,
//!                  node_count u32, band_max f64, seg_node_count u32 +
//!                  pad, triangle BVH nodes (node_count x 32, same node
//!                  layout as the solid BVH), boundary-segment BVH
//!                  nodes (seg_node_count x 32), verts (vert_count x
//!                  3xf64), tris (tri_count x 3xu32), boundary segs
//!                  (seg_count x (2xu32 + f32 band)). Triangles and
//!                  segments are in BVH leaf-contiguous order, so
//!                  leaves index the arrays directly. A segment's band
//!                  is the seam suspicion radius around it (the
//!                  chordal crack against neighboring exact faces);
//!                  `band_max` is their maximum. A mesh face's trim
//!                  blob is empty.
//! Trim blob:
//!    0  loop_count u32
//!    4  (pad)      u32
//!    8  uv_aabb    4xf64 [min_u, max_u, min_v, max_v] (unwrapped)
//!   40  loop offset table, loop_count x u32 (from trim blob start),
//!       padded to 8
//!    .  loops: count u32, (pad) u32, then count x 2xf64 uv points.
//!       Loops are closed implicitly (last connects to first) and stored
//!       *unwrapped*: on periodic surfaces consecutive points continue
//!       past the seam rather than jumping back into the base period.
//!       A loop that runs once around a periodic axis (the rim of a
//!       seamless cylinder: exporters built on Parasolid write no seam
//!       edge) therefore ends a period from where it began; it closes
//!       onto the next alias of its first point, not back across the
//!       face.
//! ```
//!
//! A frame is 12 f64: origin, then orthonormal basis rows x, y, z.
//! `world = origin + a*x + b*y + c*z` maps surface-local to solid-local
//! coordinates.

pub mod bvh;
pub mod ir;
pub mod math;
pub mod nurbs;
pub mod payload;
pub mod project;
pub mod surface;
pub mod topo;
pub mod trim;

pub const MAGIC: u32 = 0x3150_5242; // "BRP1"

/// Parity ray directions: primary matches `trimesh_model_core`'s skew
/// trick; fallbacks lean along different axes so a tangency or trim graze
/// along one direction is broken by the next.
pub const RAY_DIRS: [[f64; 3]; 3] = [
    [1.0, 1.618_033_988_7e-4, 2.718_281_828_4e-4],
    [-2.236_067_977e-4, 1.0, 1.414_213_562_3e-4],
    [3.141_592_653_5e-4, -2.718_281_828_4e-4, -1.0],
];
