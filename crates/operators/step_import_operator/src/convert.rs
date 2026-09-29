//! STEP entities -> `brep_core` IR: solids from manifold BReps (faces
//! lowered to exact surfaces, edge curves flattened and projected into
//! UV trim loops) and instances from the AP214 assembly graph.

use crate::entities::{Ctx, Curve, StepEdge, StepSurface, length_unit_scale, plane_angle_scale};
use crate::p21::{Arg, DataSection};
use brep_core::ir::{BRepModel, Face, Instance, Solid, Surface};
use brep_core::math::{Affine, Frame, Vec3, cross, dot, norm, normalize, sub};
use brep_core::nurbs::CurveData;
use brep_core::project;
use core::f64::consts::TAU;
use std::collections::HashMap;

/// Import tuning; `chord_tol` is the 3D flattening tolerance (metres)
/// for trim polylines and lowered extrusion profiles.
pub struct Options {
    pub chord_tol: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options { chord_tol: 5e-6 }
    }
}

/// Convert a parsed STEP file into a BRep model, walking the assembly
/// graph from its roots. Fails loudly: any face of any referenced solid
/// that cannot be converted fails the import with the entity id (a
/// silently missing face would flip parity for its whole solid).
pub fn build_model(data: &DataSection, opts: &Options) -> Result<BRepModel, String> {
    let scale = length_unit_scale(data)?;
    let angle_scale = plane_angle_scale(data)?;
    let ctx = Ctx {
        data,
        scale,
        angle_scale,
    };
    let asm = AssemblyGraph::build(data)?;
    let colors = crate::style::collect_colors(data);

    let mut model = BRepModel::default();
    let mut solid_index: HashMap<u64, usize> = HashMap::new();

    for root in &asm.roots {
        let label = asm.pd_name.get(root).cloned().unwrap_or_default();
        walk(
            &ctx,
            &asm,
            *root,
            Affine::IDENTITY,
            &label,
            0,
            opts,
            &colors,
            &mut model,
            &mut solid_index,
        )?;
    }
    if model.instances.is_empty() {
        return Err("no solid geometry found in the file".into());
    }
    Ok(model)
}

/// Keep only instances whose label passes the include/exclude lists
/// (case-sensitive exact match; empty include list means "everything"),
/// then drop unreferenced solids.
pub fn filter_instances(model: &mut BRepModel, include: &[String], exclude: &[String]) {
    model.instances.retain(|inst| {
        let inc = include.is_empty() || include.contains(&inst.label);
        inc && !exclude.contains(&inst.label)
    });
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut solids = Vec::new();
    for inst in &mut model.instances {
        let new = *remap.entry(inst.solid).or_insert_with(|| {
            solids.push(model.solids[inst.solid].clone());
            solids.len() - 1
        });
        inst.solid = new;
    }
    model.solids = solids;
}

// -------------------------------------------------------------------
// Assembly graph
// -------------------------------------------------------------------

struct Nauo {
    child_pd: u64,
    /// The NAUO's name — KiCad puts the reference designator here.
    label: String,
    /// Child-to-parent placement.
    transform: Affine,
}

struct AssemblyGraph {
    /// PRODUCT_DEFINITION -> its shape representation.
    pd_rep: HashMap<u64, u64>,
    /// Child links per parent PD.
    children: HashMap<u64, Vec<usize>>,
    nauos: Vec<Nauo>,
    /// PDs that are no NAUO's child.
    roots: Vec<u64>,
    /// PD -> product name (fallback label).
    pd_name: HashMap<u64, String>,
    /// Untransformed SHAPE_REPRESENTATION_RELATIONSHIP links, both
    /// directions. Exporters (Onshape among them) hang a product's
    /// geometry representations off its SDR representation this way;
    /// assembly placements use the *transformed* relationship, which is
    /// a complex instance and deliberately not collected here.
    rep_links: HashMap<u64, Vec<u64>>,
}

impl AssemblyGraph {
    fn build(data: &DataSection) -> Result<AssemblyGraph, String> {
        // Angle-preserving accessors only; no unit scaling is needed for
        // the graph itself, but transforms carry translations, so use a
        // scaled context.
        let scale = length_unit_scale(data)?;
        let ctx = Ctx {
            data,
            scale,
            angle_scale: plane_angle_scale(data)?,
        };

        let mut pd_rep = HashMap::new();
        let mut pd_name = HashMap::new();
        let mut nauo_ids = Vec::new();
        let mut cdsr_ids = Vec::new();
        let mut rep_links: HashMap<u64, Vec<u64>> = HashMap::new();
        // Sorted iteration: instance/solid order (and therefore payload
        // bytes) must not depend on hash order — downstream caching and
        // baked projects want reproducible output.
        let mut ids: Vec<u64> = data.entities.keys().copied().collect();
        ids.sort_unstable();
        for &id in &ids {
            let e = &data.entities[&id];
            if let Some(r) = e.simple() {
                match r.name.as_str() {
                    "SHAPE_DEFINITION_REPRESENTATION" => {
                        // (definition: represented_definition, used_representation).
                        // The definition select also admits plain
                        // PROPERTY_DEFINITIONs (Onshape hybrid exports
                        // attach 'HYBRID_SOURCE_ID' reps this way); only
                        // a PDS of a product definition names a shape.
                        let pds = data.deref(arg_at(r, 0, id)?)?;
                        let Some(pds_r) = pds
                            .simple()
                            .filter(|r| r.name == "PRODUCT_DEFINITION_SHAPE")
                        else {
                            continue;
                        };
                        let def = arg_at(pds_r, 2, id)?
                            .as_ref_id()
                            .ok_or_else(|| format!("#{id}: PDS definition not a reference"))?;
                        // Only PDS of product definitions matter here
                        // (placement PDSes point at NAUOs).
                        if data.get(def)?.is("PRODUCT_DEFINITION") {
                            let rep = arg_at(r, 1, id)?
                                .as_ref_id()
                                .ok_or_else(|| format!("#{id}: SDR rep not a reference"))?;
                            pd_rep.insert(def, rep);
                        }
                    }
                    "NEXT_ASSEMBLY_USAGE_OCCURRENCE" => nauo_ids.push(id),
                    "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION" => cdsr_ids.push(id),
                    "SHAPE_REPRESENTATION_RELATIONSHIP" => {
                        // (name, description, rep_1, rep_2)
                        let rep_1 = arg_at(r, 2, id)?
                            .as_ref_id()
                            .ok_or_else(|| format!("#{id}: SRR rep_1 not a reference"))?;
                        let rep_2 = arg_at(r, 3, id)?
                            .as_ref_id()
                            .ok_or_else(|| format!("#{id}: SRR rep_2 not a reference"))?;
                        rep_links.entry(rep_1).or_default().push(rep_2);
                        rep_links.entry(rep_2).or_default().push(rep_1);
                    }
                    "PRODUCT_DEFINITION" => {
                        // formation -> product -> name.
                        let name = (|| {
                            let formation = data.deref(r.args.get(2)?).ok()?;
                            let product = data.deref(formation.simple()?.args.get(2)?).ok()?;
                            let pr = product.simple()?;
                            let id_field = pr.args.first()?.as_str()?;
                            let name_field = pr.args.get(1).and_then(Arg::as_str);
                            Some(if id_field.is_empty() {
                                name_field.unwrap_or_default().to_string()
                            } else {
                                id_field.to_string()
                            })
                        })()
                        .unwrap_or_default();
                        pd_name.insert(id, name);
                    }
                    _ => {}
                }
            }
        }

        // CDSR: NAUO id -> (rep_1, rep_2, transform item pair).
        let mut nauo_cdsr: HashMap<u64, (u64, u64, u64, u64)> = HashMap::new();
        for id in cdsr_ids {
            let r = data.get(id)?.simple().ok_or("CDSR must be simple")?.clone();
            let rr = data.deref(arg_at(&r, 0, id)?)?;
            let rel = rr
                .record("REPRESENTATION_RELATIONSHIP")
                .ok_or_else(|| format!("#{id}: no REPRESENTATION_RELATIONSHIP"))?;
            let rep_1 = arg_at(rel, 2, id)?
                .as_ref_id()
                .ok_or_else(|| format!("#{id}: rep_1 not a reference"))?;
            let rep_2 = arg_at(rel, 3, id)?
                .as_ref_id()
                .ok_or_else(|| format!("#{id}: rep_2 not a reference"))?;
            let idt_id = rr
                .record("REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION")
                .and_then(|t| t.args.first())
                .and_then(Arg::as_ref_id)
                .ok_or_else(|| format!("#{id}: no transformation"))?;
            let idt = data.get(idt_id)?.simple().ok_or("IDT must be simple")?;
            let axis_1 = arg_at(idt, 2, idt_id)?
                .as_ref_id()
                .ok_or_else(|| format!("#{idt_id}: IDT item 1 not a reference"))?;
            let axis_2 = arg_at(idt, 3, idt_id)?
                .as_ref_id()
                .ok_or_else(|| format!("#{idt_id}: IDT item 2 not a reference"))?;
            // The PDS this CDSR describes points at the NAUO.
            let pds = data.deref(arg_at(&r, 1, id)?)?;
            let nauo = pds
                .simple()
                .filter(|p| p.name == "PRODUCT_DEFINITION_SHAPE")
                .and_then(|p| p.args.get(2))
                .and_then(Arg::as_ref_id)
                .ok_or_else(|| format!("#{id}: CDSR without a placement PDS"))?;
            nauo_cdsr.insert(nauo, (rep_1, rep_2, axis_1, axis_2));
        }

        let mut nauos = Vec::new();
        let mut children: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut child_pds: Vec<u64> = Vec::new();
        for id in nauo_ids {
            let e = data.get(id)?;
            let r = e.simple().ok_or("NAUO must be simple")?;
            let parent_pd = arg_at(r, 3, id)?
                .as_ref_id()
                .ok_or_else(|| format!("#{id}: NAUO relating not a reference"))?;
            let child_pd = arg_at(r, 4, id)?
                .as_ref_id()
                .ok_or_else(|| format!("#{id}: NAUO related not a reference"))?;
            let label = r
                .args
                .get(1)
                .and_then(Arg::as_str)
                .unwrap_or_default()
                .to_string();
            let (rep_1, rep_2, axis_1, axis_2) = *nauo_cdsr.get(&id).ok_or_else(|| {
                format!("#{id}: assembly occurrence has no placement (missing CDSR)")
            })?;
            // Child-to-parent: the child-side axis maps onto the
            // parent-side axis. Which IDT item is child-side follows
            // from which representation is the child's — and a rep pair
            // matching neither side would make any assignment a silent
            // guess that inverts every placement, so it fails loudly.
            let child_rep = pd_rep.get(&child_pd).copied();
            let (child_axis, parent_axis) = if child_rep == Some(rep_1) {
                (axis_1, axis_2)
            } else if child_rep == Some(rep_2) {
                (axis_2, axis_1)
            } else {
                return Err(format!(
                    "#{id}: cannot orient assembly transform: the child \
                     product's representation matches neither relation side"
                ));
            };
            let fc = frame_affine(&ctx.axis2(child_axis)?);
            let fp = frame_affine(&ctx.axis2(parent_axis)?);
            let transform = fp.compose(&fc.rigid_inverse()?);
            nauos.push(Nauo {
                child_pd,
                label,
                transform,
            });
            children.entry(parent_pd).or_default().push(nauos.len() - 1);
            child_pds.push(child_pd);
        }

        let mut roots: Vec<u64> = pd_rep
            .keys()
            .copied()
            .filter(|pd| !child_pds.contains(pd))
            .collect();
        roots.sort_unstable();
        if roots.is_empty() && !pd_rep.is_empty() {
            return Err("assembly graph has a cycle (no root product)".into());
        }
        Ok(AssemblyGraph {
            pd_rep,
            children,
            nauos,
            roots,
            pd_name,
            rep_links,
        })
    }

    /// `root` plus every representation transitively reachable over
    /// untransformed SRR links, in encounter order (deterministic:
    /// links were collected over sorted entity ids).
    fn linked_reps(&self, root: u64) -> Vec<u64> {
        let mut out = vec![root];
        let mut i = 0;
        while i < out.len() {
            if let Some(links) = self.rep_links.get(&out[i]) {
                for &l in links {
                    if !out.contains(&l) {
                        out.push(l);
                    }
                }
            }
            i += 1;
        }
        out
    }
}

/// `record.args[idx]` with a Result instead of a panic on malformed
/// entities.
fn arg_at(r: &crate::p21::Record, idx: usize, id: u64) -> Result<&Arg, String> {
    r.args
        .get(idx)
        .ok_or_else(|| format!("#{id} {}: missing argument {idx}", r.name))
}

/// Frame -> local-to-world affine.
fn frame_affine(f: &Frame) -> Affine {
    Affine([
        f.x[0],
        f.y[0],
        f.z[0],
        f.origin[0], //
        f.x[1],
        f.y[1],
        f.z[1],
        f.origin[1], //
        f.x[2],
        f.y[2],
        f.z[2],
        f.origin[2],
    ])
}

#[allow(clippy::too_many_arguments)]
fn walk(
    ctx: &Ctx,
    asm: &AssemblyGraph,
    pd: u64,
    transform: Affine,
    label: &str,
    depth: usize,
    opts: &Options,
    colors: &HashMap<u64, [f32; 3]>,
    model: &mut BRepModel,
    solid_index: &mut HashMap<u64, usize>,
) -> Result<(), String> {
    if depth > 64 {
        return Err(format!(
            "assembly nesting deeper than 64 at product #{pd} — cycle in the file?"
        ));
    }
    if let Some(&rep) = asm.pd_rep.get(&pd) {
        // Geometry of this product: items of its representation plus of
        // every representation attached over untransformed SRR links
        // (Onshape parks the shape reps there; the SDR rep holds only a
        // placement). MANIFOLD_SOLID_BREPs are bodies of their own;
        // surface-model shells and tessellated shells together form one
        // composite body (parity over the union of faces — correct for
        // disjoint closed shells and for hybrid shells glued along
        // shared boundary rings).
        let mut msbs: Vec<u64> = Vec::new();
        let mut shell_models: Vec<u64> = Vec::new();
        let mut tessellated: Vec<u64> = Vec::new();
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for rep_id in asm.linked_reps(rep) {
            let rep_entity = ctx.data.get(rep_id)?;
            let items = rep_entity
                .records
                .iter()
                .find_map(|r| r.args.get(1).and_then(Arg::as_list))
                .unwrap_or(&[]);
            for item in items {
                let Some(item_id) = item.as_ref_id() else {
                    continue;
                };
                if !seen.insert(item_id) {
                    continue;
                }
                let e = ctx.data.get(item_id)?;
                if e.is("MANIFOLD_SOLID_BREP") {
                    msbs.push(item_id);
                } else if e.is("SHELL_BASED_SURFACE_MODEL") {
                    shell_models.push(item_id);
                } else if e.is("TESSELLATED_SHELL") {
                    tessellated.push(item_id);
                }
            }
        }
        // With exact solids present, a tessellated rep is a redundant
        // preview of the same bodies — importing it too would cancel
        // their parity. Without them it is real geometry (the mesh part
        // of a hybrid body, or a mesh-only file).
        if !msbs.is_empty() {
            tessellated.clear();
        }
        for item_id in msbs {
            let solid = match solid_index.get(&item_id) {
                Some(&idx) => idx,
                None => {
                    let converted = convert_solid(ctx, item_id, opts, colors)
                        .map_err(|e| format!("solid #{item_id}: {e}"))?;
                    model.solids.push(converted);
                    let idx = model.solids.len() - 1;
                    solid_index.insert(item_id, idx);
                    idx
                }
            };
            model.instances.push(Instance {
                solid,
                local_to_world: transform,
                label: label.to_string(),
            });
        }
        if !shell_models.is_empty() || !tessellated.is_empty() {
            // The composite is keyed by the product's SDR rep: shared
            // sub-products reuse their converted body.
            let solid = match solid_index.get(&rep) {
                Some(&idx) => idx,
                None => {
                    let converted =
                        convert_composite(ctx, &shell_models, &tessellated, opts, colors)?;
                    model.solids.push(converted);
                    let idx = model.solids.len() - 1;
                    solid_index.insert(rep, idx);
                    idx
                }
            };
            model.instances.push(Instance {
                solid,
                local_to_world: transform,
                label: label.to_string(),
            });
        }
    }
    if let Some(kids) = asm.children.get(&pd) {
        for &k in kids {
            let nauo = &asm.nauos[k];
            let child_transform = transform.compose(&nauo.transform);
            // The occurrence name is the reference designator ("J7") in
            // KiCad exports; OCCT-generated names ("=>[0:1:1:3]") and
            // empty names inherit the enclosing label.
            let child_label = if !nauo.label.is_empty() && !nauo.label.starts_with("=>") {
                nauo.label.as_str()
            } else {
                label
            };
            walk(
                ctx,
                asm,
                nauo.child_pd,
                child_transform,
                child_label,
                depth + 1,
                opts,
                colors,
                model,
                solid_index,
            )?;
        }
    }
    Ok(())
}

// -------------------------------------------------------------------
// Solid and face conversion
// -------------------------------------------------------------------

fn convert_solid(
    ctx: &Ctx,
    msb_id: u64,
    opts: &Options,
    colors: &HashMap<u64, [f32; 3]>,
) -> Result<Solid, String> {
    let body_color = colors.get(&msb_id).copied();
    let mut faces = Vec::new();
    for face_id in ctx.solid_faces(msb_id)? {
        let mut face =
            convert_face(ctx, face_id, opts).map_err(|e| format!("face #{face_id}: {e}"))?;
        // Face styling overrides the body style.
        face.color = colors.get(&face_id).copied().or(body_color);
        faces.push(face);
    }
    Ok(Solid { faces })
}

/// One body from a product's surface-model shells plus tessellated
/// shells. Face styling falls back to the owning shell model's style
/// (Onshape styles the SHELL_BASED_SURFACE_MODEL / TESSELLATED_SHELL
/// entity, where OCCT solids style the MANIFOLD_SOLID_BREP).
fn convert_composite(
    ctx: &Ctx,
    shell_models: &[u64],
    tessellated: &[u64],
    opts: &Options,
    colors: &HashMap<u64, [f32; 3]>,
) -> Result<Solid, String> {
    let mut faces = Vec::new();
    for &sm_id in shell_models {
        let body_color = colors.get(&sm_id).copied();
        for face_id in ctx.surface_model_faces(sm_id)? {
            let mut face = convert_face(ctx, face_id, opts)
                .map_err(|e| format!("surface model #{sm_id}, face #{face_id}: {e}"))?;
            face.color = colors.get(&face_id).copied().or(body_color);
            faces.push(face);
        }
    }
    for &ts_id in tessellated {
        let shell_color = colors.get(&ts_id).copied();
        for item_id in ctx.tessellated_shell_items(ts_id)? {
            let mesh = ctx
                .triangulated_face(item_id)
                .map_err(|e| format!("tessellated shell #{ts_id}, face #{item_id}: {e}"))?;
            faces.push(Face {
                surface: Surface::Mesh(brep_core::ir::MeshSurface {
                    verts: mesh.verts,
                    tris: mesh.tris,
                }),
                trims: vec![],
                color: colors.get(&item_id).copied().or(shell_color),
            });
        }
    }
    Ok(Solid { faces })
}

fn convert_face(ctx: &Ctx, face_id: u64, opts: &Options) -> Result<Face, String> {
    let (surface_id, loops) = ctx.advanced_face(face_id)?;
    let step_surface = ctx
        .surface(surface_id)
        .map_err(|e| format!("surface: {e}"))?;
    let (surface, flipped) = lower_surface(step_surface, opts)?;
    let (same_sense, as_listed) = ctx.face_senses(face_id)?;

    let proj_tol = opts.chord_tol * 4.0 + 1e-6;
    let mut trims = Vec::new();
    for (li, lp) in loops.iter().enumerate() {
        let pts = loop_points(ctx, lp, opts).map_err(|e| format!("loop {li}: {e}"))?;
        let mut uv: Vec<[f64; 2]> = Vec::with_capacity(pts.len());
        let mut hint = None;
        for p in &pts {
            let q = project::project_point(&surface, *p, hint, proj_tol)
                .map_err(|e| format!("loop {li}: {e}"))?;
            uv.push(q);
            hint = Some(q);
        }
        project::verify_loop(&surface, &pts, &uv, proj_tol * 2.0)
            .map_err(|e| format!("loop {li}: {e}"))?;
        trims.push(uv);
    }
    // Loops around a periodic surface, and loops unwrapped from the far
    // side of its seam, take their place from the side the face is on.
    let reversed: Vec<bool> = as_listed.iter().map(|listed| !listed).collect();
    project::settle_periodic(&surface, same_sense != flipped, &mut trims, &reversed)?;
    Ok(Face {
        surface,
        trims,
        color: None,
    })
}

/// Lower a STEP surface to the exact `brep_core` vocabulary. Extrusions
/// of lines and axis-aligned circles become planes and cylinders; other
/// profiles flatten to a polyline prism at the chord tolerance — the
/// only approximation in the pipeline. The flag says the lowered
/// surface's normal is the opposite of the STEP surface's.
fn lower_surface(s: StepSurface, opts: &Options) -> Result<(Surface, bool), String> {
    let mut flipped = false;
    let lowered = match s {
        StepSurface::Plane(frame) => Surface::Plane { frame },
        StepSurface::Cylinder(frame, radius) => Surface::Cylinder { frame, radius },
        StepSurface::Cone(frame, radius, half_angle) => Surface::Cone {
            frame,
            radius,
            half_angle,
        },
        StepSurface::Sphere(frame, radius) => Surface::Sphere { frame, radius },
        StepSurface::Torus(frame, major, minor) => Surface::Torus {
            frame,
            major,
            minor,
        },
        StepSurface::Nurbs(n) => Surface::Nurbs(n),
        StepSurface::Extrusion(profile, dir) => {
            let dir_n = normalize(dir);
            match *profile {
                Curve::Line {
                    point,
                    dir: line_dir,
                } => {
                    let normal = cross(normalize(line_dir), dir_n);
                    if norm(normal) < 1e-9 {
                        return Err("extrusion of a line along itself".into());
                    }
                    Surface::Plane {
                        frame: Frame::from_axis_ref(point, normal, line_dir),
                    }
                }
                Curve::Circle { frame, radius } if dot(frame.z, dir_n).abs() > 1.0 - 1e-9 => {
                    // Swept against the circle's axis, the STEP surface
                    // faces the axis.
                    flipped = dot(frame.z, dir_n) < 0.0;
                    Surface::Cylinder { frame, radius }
                }
                other => {
                    // General profile: flatten in 3D, project onto the
                    // plane perpendicular to the extrusion direction.
                    let pts3 = match &other {
                        Curve::Circle { frame, radius } => {
                            project::flatten_circle(frame, *radius, 0.0, TAU, opts.chord_tol)
                        }
                        Curve::Ellipse { frame, a, b } => {
                            project::flatten_ellipse(frame, *a, *b, 0.0, TAU, opts.chord_tol)
                        }
                        Curve::Bspline(c) => {
                            let dom = c.domain();
                            project::flatten_bspline(c, dom[0], dom[1], opts.chord_tol)
                        }
                        Curve::Line { .. } => unreachable!(),
                    };
                    let origin = pts3[0];
                    let first_seg = sub(*pts3.last().unwrap(), pts3[0]);
                    let seg = if norm(first_seg) > 1e-9 {
                        first_seg
                    } else {
                        sub(pts3[pts3.len() / 2], pts3[0])
                    };
                    let frame = Frame::from_axis_ref(origin, dir_n, seg);
                    let profile: Vec<[f64; 2]> = pts3
                        .iter()
                        .map(|p| {
                            let l = frame.to_local(*p);
                            [l[0], l[1]]
                        })
                        .collect();
                    Surface::ExtrusionPolyline { frame, profile }
                }
            }
        }
    };
    Ok((lowered, flipped))
}

/// Both edge vertices must actually lie on the curve for a sub-arc trim
/// to be meaningful; a distant vertex means the curve association is
/// wrong and would silently produce a bogus trim.
fn check_vertex_on_curve(curve_id: u64, d0: f64, d1: f64, tol: f64) -> Result<(), String> {
    if d0 > tol || d1 > tol {
        return Err(format!(
            "curve #{curve_id}: edge vertex off the curve by {:.3e}",
            d0.max(d1)
        ));
    }
    Ok(())
}

/// Flatten one loop of oriented edges into a chained, closed 3D
/// polyline (the closing point is omitted).
fn loop_points(ctx: &Ctx, edges: &[StepEdge], opts: &Options) -> Result<Vec<Vec3>, String> {
    let tol = opts.chord_tol;
    let join_tol = (tol * 10.0).max(1e-6);
    let mut out: Vec<Vec3> = Vec::new();
    for (i, e) in edges.iter().enumerate() {
        let pts = edge_points(ctx, e, tol).map_err(|err| format!("edge {i}: {err}"))?;
        if let Some(last) = out.last()
            && norm(sub(*last, pts[0])) > join_tol
        {
            return Err(format!(
                "edge {i} does not chain: gap {:.3e}",
                norm(sub(*last, pts[0]))
            ));
        }
        let skip = usize::from(!out.is_empty());
        out.extend(&pts[skip..]);
    }
    if out.len() > 1 {
        let gap = norm(sub(*out.last().unwrap(), out[0]));
        if gap > join_tol {
            return Err(format!("loop does not close: gap {gap:.3e}"));
        }
        out.pop();
    }
    if out.len() < 3 {
        return Err("degenerate loop with fewer than 3 points".into());
    }
    Ok(out)
}

/// Flatten the arc of a circle/ellipse `(a cos t, b sin t)` in `frame`
/// between the edge's vertices, from its start vertex to its end vertex.
/// Coincident vertices mean the edge covers the whole closed curve.
fn conic_arc_points(frame: &Frame, a: f64, b: f64, e: &StepEdge, tol: f64) -> Vec<Vec3> {
    let t_of = |p: Vec3| {
        let l = frame.to_local(p);
        (l[1] / b).atan2(l[0] / a)
    };
    let t_start = t_of(e.start);
    let full = norm(sub(e.start, e.end)) < tol * 1e-3 + 1e-9;
    // Sweep CCW in the curve frame when the curve runs forward along
    // the traversal, CW otherwise.
    let (t0, t1) = if full {
        (t_start, t_start + TAU)
    } else {
        let mut t_end = t_of(e.end);
        if e.forward {
            if t_end <= t_start {
                t_end += TAU;
            }
            (t_start, t_end)
        } else {
            if t_end >= t_start {
                t_end -= TAU;
            }
            (t_start, t_end)
        }
    };
    let (lo, hi, reversed) = if full && !e.forward {
        (t_start - TAU, t_start, true)
    } else if t1 >= t0 {
        (t0, t1, false)
    } else {
        (t1, t0, true)
    };
    let mut pts = project::flatten_ellipse(frame, a, b, lo, hi, tol);
    if reversed {
        pts.reverse();
    }
    pts
}

/// Flatten one oriented edge into 3D points from its start vertex to
/// its end vertex.
fn edge_points(ctx: &Ctx, e: &StepEdge, tol: f64) -> Result<Vec<Vec3>, String> {
    let curve = ctx.curve(e.curve_id)?;
    let mut pts = match curve {
        Curve::Line { .. } => vec![e.start, e.end],
        Curve::Circle { frame, radius } => conic_arc_points(&frame, radius, radius, e, tol),
        Curve::Ellipse { frame, a, b } => conic_arc_points(&frame, a, b, e, tol),
        Curve::Bspline(c) => {
            let dom = c.domain();
            let closed = {
                let (p0, _) = brep_core::nurbs::curve_eval(&c, dom[0]);
                let (p1, _) = brep_core::nurbs::curve_eval(&c, dom[1]);
                norm(sub(p0, p1)) < tol * 1e-3 + 1e-9
            };
            if norm(sub(e.start, e.end)) < tol * 1e-3 + 1e-9 {
                // Degenerate vertex pair: the edge covers the whole
                // (closed) curve, like a full circle.
                let mut pts = project::flatten_bspline(&c, dom[0], dom[1], tol);
                if !e.forward {
                    pts.reverse();
                }
                pts
            } else {
                // Vertices may trim the curve to a sub-arc: exporters
                // like Onshape share one full curve between edges where
                // OCCT reparameterizes the curve to the edge bounds.
                // Locate both vertices and flatten the range between
                // them, in curve orientation (start -> end when the
                // curve runs forward along the traversal).
                let join_tol = (tol * 10.0).max(1e-6);
                let (t_from, t_to, reverse) = if e.forward {
                    let (t0, d0) = brep_core::nurbs::curve_closest_param(&c, e.start);
                    let (t1, d1) = brep_core::nurbs::curve_closest_param(&c, e.end);
                    check_vertex_on_curve(e.curve_id, d0, d1, join_tol)?;
                    (t0, t1, false)
                } else {
                    let (t0, d0) = brep_core::nurbs::curve_closest_param(&c, e.end);
                    let (t1, d1) = brep_core::nurbs::curve_closest_param(&c, e.start);
                    check_vertex_on_curve(e.curve_id, d0, d1, join_tol)?;
                    (t0, t1, true)
                };
                let mut pts = if t_from < t_to {
                    project::flatten_bspline(&c, t_from, t_to, tol)
                } else if closed {
                    // The sub-arc wraps the closed curve's seam.
                    let mut head = project::flatten_bspline(&c, t_from, dom[1], tol);
                    let tail = project::flatten_bspline(&c, dom[0], t_to, tol);
                    head.extend_from_slice(&tail[1..]);
                    head
                } else {
                    return Err(format!(
                        "curve #{}: edge vertices out of order along an open curve \
                         (t {t_from:.6} > t {t_to:.6})",
                        e.curve_id
                    ));
                };
                if reverse {
                    pts.reverse();
                }
                pts
            }
        }
    };
    // Snap the flattened ends onto the topological vertices so chaining
    // and closure tests see exact joins.
    let join_tol = (tol * 10.0).max(1e-6);
    let d_start = norm(sub(pts[0], e.start));
    let d_end = norm(sub(*pts.last().unwrap(), e.end));
    if d_start > join_tol || d_end > join_tol {
        return Err(format!(
            "flattened curve #{} misses its vertices by {d_start:.3e}/{d_end:.3e}",
            e.curve_id
        ));
    }
    let n = pts.len();
    pts[0] = e.start;
    pts[n - 1] = e.end;
    Ok(pts)
}
