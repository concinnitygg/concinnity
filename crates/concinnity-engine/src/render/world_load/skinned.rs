//! A world's skinned meshes at load: each mesh's decoded geometry, the shared
//! skinned buffers and per-slot draw records built from it (each mesh followed
//! by its hidden pre-reserved instance copies), and the posed template entity
//! per mesh that animation drives and a runtime spawn clones.

use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use concinnity_core::math::SaturatingFrom;
use hashbrown::HashMap;

use crate::animation::character_shape;
use crate::animation::skeleton::Skeleton;
use crate::animation::skinned_mesh_map::{
    SkinnedMeshMorphNames, SkinnedMeshNameIndex, SkinnedMeshSkinIndex,
};
use crate::components::{
    CharacterCapsule, CharacterRig, GlobalTransform, SkeletonJoint, SkinnedMesh, Transform,
    build_skeleton_from_joint_defs,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{MaterialHandle, PickIndex, PipelineContext, SkinnedMeshHandle};
use crate::geometry::payload_joints_to_defs;
use crate::gfx::mesh_payload::{self, PayloadMorphs, SkinnedVertex};
use crate::gfx::render_types::{MAX_JOINTS, SkinnedDrawObject, SkinnedIndex};
use crate::render::draw_list::append_lod_slices;
use crate::render::frame::extract::PickCandidate;
use crate::render::material_entry::{MaterialEntry, resolve_material_slots};
use crate::resource::SkinnedMeshTable;

/// One skinned mesh's decoded geometry, in the order the cook assigned handles
/// (the table index is the `SkinnedMeshHandle` keying the animation
/// correlation web).
pub struct SkinnedGeometry {
    /// The mesh's handle.
    pub handle: SkinnedMeshHandle,
    /// The mesh's interned name.
    pub name_id: AssetId,
    /// The baked mesh: placement, material, capsule and spawn reserve.
    pub mesh: SkinnedMesh,
    /// Its vertices.
    pub vertices: Vec<SkinnedVertex>,
    /// Its LOD0 indices.
    pub indices: Vec<u16>,
    /// Its bind-pose joints.
    pub joint_defs: Vec<SkeletonJoint>,
    /// Its morph targets.
    pub morphs: PayloadMorphs,
    /// Its LOD alternates: switch distance and indices.
    pub lod_alternates: Vec<(f32, Vec<u16>)>,
}

/// Decode every `SkinnedMesh` in the world's table, with the blob each payload
/// came from, and publish the name, skin and morph-name indices animation reads
/// at init. The first mesh that fails to decode fails the load.
///
/// # Errors
///
/// Returns an error naming the first mesh whose record or payload fails to
/// decode.
pub fn decode_skinned(
    ctx: &mut PipelineContext<'_>,
) -> Result<(Vec<SkinnedGeometry>, Vec<u32>), String> {
    let table = ctx
        .resource::<SkinnedMeshTable>()
        .cloned()
        .unwrap_or_default();
    let mut geometry = Vec::with_capacity(table.0.len());
    let mut blob_indices = Vec::with_capacity(table.0.len());
    let mut name_index: HashMap<AssetId, SkinnedMeshHandle> = HashMap::new();
    for (index, entry) in table.0.iter().enumerate() {
        let handle = SkinnedMeshHandle::new(u32::saturating_from(index));
        let (name_id, mesh): (u32, SkinnedMesh) = postcard::from_bytes(&entry.data_bytes)
            .map_err(|e| format!("SkinnedMesh handle {index} baked data failed to decode: {e}"))?;
        let name_id = AssetId(name_id);
        name_index.insert(name_id, handle);
        let locator = entry
            .payload
            .clone()
            .ok_or_else(|| format!("SkinnedMesh handle {index} has no compiled payload"))?;
        blob_indices.push(locator.blob_index);
        let bytes = ctx
            .read_payload(&locator)
            .map_err(|e| format!("failed to read SkinnedMesh handle {index} payload: {e}"))?;
        let payload = mesh_payload::deserialize_skinned_with_lods(bytes)
            .map_err(|e| format!("malformed SkinnedMesh payload: {e}"))?;
        geometry.push(SkinnedGeometry {
            handle,
            name_id,
            mesh,
            vertices: payload.vertices,
            indices: payload.indices,
            joint_defs: payload_joints_to_defs(payload.joints),
            morphs: payload.morphs,
            lod_alternates: payload.lods,
        });
    }
    ctx.insert_resource(SkinnedMeshNameIndex(name_index));
    ctx.insert_resource(SkinnedMeshSkinIndex(
        geometry.iter().map(|g| g.mesh.skin_index).collect(),
    ));
    ctx.insert_resource(SkinnedMeshMorphNames(
        geometry.iter().map(|g| g.morphs.names.clone()).collect(),
    ));
    Ok((geometry, blob_indices))
}

/// One skinned mesh's template: what its posed entity and rig are built from.
/// `template_index` is recorded rather than inferred from position because the
/// pre-reserved instance copies interleave the draw records (template, copies,
/// template, ...).
pub struct SkinnedTemplate {
    /// The mesh's handle.
    pub handle: SkinnedMeshHandle,
    /// The mesh's interned name.
    pub name_id: AssetId,
    /// The skinned slot of its visible draw record.
    pub template_index: SkinnedIndex,
    /// Its bind-pose skeleton.
    pub skeleton: Skeleton,
    /// Its morph target names.
    pub morph_names: Vec<String>,
    /// Its model matrix.
    pub model: [[f32; 4]; 4],
    /// Its character capsule, when it bakes one.
    pub capsule: Option<CharacterCapsule>,
    /// Its authored placement.
    pub transform: Transform,
    /// Its bind-pose bounds, minimum and maximum corner.
    pub local_bounds: ([f32; 3], [f32; 3]),
}

/// The shared skinned buffers, the per-slot draw records and morph targets,
/// the templates, and the `(template, copy)` pairs seeding the skinned
/// instance pool.
pub struct SkinnedAssembly {
    /// The shared skinned vertex buffer.
    pub vertices: Vec<SkinnedVertex>,
    /// Absolute indices into `vertices`, so u32 rather than the per-mesh u16
    /// the payload carries.
    pub indices: Vec<u32>,
    /// One draw record per skinned slot.
    pub draw_objects: Vec<SkinnedDrawObject>,
    /// Per-slot morph targets; copies share their template's.
    pub morphs: Vec<Option<Arc<PayloadMorphs>>>,
    /// One template per mesh.
    pub templates: Vec<SkinnedTemplate>,
    /// Each hidden copy beside the template it copies.
    pub pool_reservations: Vec<(SkinnedIndex, SkinnedIndex)>,
}

// Bind-pose (object-space) bounds over `vertices`; zero for an empty mesh.
fn bind_pose_bounds(vertices: &[SkinnedVertex]) -> ([f32; 3], [f32; 3]) {
    if vertices.is_empty() {
        return ([0.0; 3], [0.0; 3]);
    }
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for v in vertices {
        for a in 0..3 {
            lo[a] = lo[a].min(v.pos[a]);
            hi[a] = hi[a].max(v.pos[a]);
        }
    }
    (lo, hi)
}

impl SkinnedAssembly {
    // Append one slot's region of `g`'s geometry, visible or a hidden copy.
    fn push_slot(
        &mut self,
        g: &SkinnedGeometry,
        entry: &MaterialEntry,
        joint_count: usize,
        bounds: ([f32; 3], [f32; 3]),
        morphs: Option<&Arc<PayloadMorphs>>,
        visible: bool,
    ) -> SkinnedIndex {
        let base = u32::saturating_from(self.vertices.len());
        let index_offset = self.indices.len();
        self.vertices.extend_from_slice(&g.vertices);
        self.indices
            .extend(g.indices.iter().map(|i| u32::from(*i) + base));
        // LOD alternates share this slot's vertex region, rebased onto the
        // same `base` as LOD0.
        let lod_alternates = append_lod_slices(&mut self.indices, &g.lod_alternates, base);
        let slot = SkinnedIndex::from_usize(self.draw_objects.len());
        self.morphs.push(morphs.cloned());
        self.draw_objects.push(SkinnedDrawObject {
            vertex_base: base,
            vertex_count: g.vertices.len(),
            index_offset,
            index_count: g.indices.len(),
            model: g.mesh.model_matrix(),
            texture_slot: entry.albedo_slot,
            normal_map_slot: entry.normal_map_slot,
            material: entry.uniforms,
            visible,
            joint_count,
            local_bb_min: bounds.0,
            local_bb_max: bounds.1,
            lod_alternates,
        });
        slot
    }
}

/// Build the shared skinned buffers and draw records from `geometry`, each
/// mesh followed by `max_instances` hidden bind-pose copies for runtime
/// spawns. Each copy owns its own vertex region: the GPU skin fold writes the
/// deformed buffer by global vertex index, so two live instances sharing one
/// would clobber each other's pose. A mesh naming an unknown material fails
/// the load.
///
/// # Errors
///
/// Returns an error naming a mesh that references an unknown material.
pub fn assemble_skinned(
    geometry: &[SkinnedGeometry],
    material_map: &HashMap<MaterialHandle, MaterialEntry>,
) -> Result<SkinnedAssembly, String> {
    let mut out = SkinnedAssembly {
        vertices: Vec::new(),
        indices: Vec::new(),
        draw_objects: Vec::new(),
        morphs: Vec::new(),
        templates: Vec::with_capacity(geometry.len()),
        pool_reservations: Vec::new(),
    };
    for g in geometry {
        let entry = resolve_material_slots(g.mesh.material, material_map).map_err(|mat| {
            format!(
                "SkinnedMesh '{}' references unknown material {}",
                g.name_id,
                mat.index()
            )
        })?;
        let skeleton = build_skeleton_from_joint_defs(&g.joint_defs);
        let joint_count = skeleton.len().min(MAX_JOINTS);
        let bounds = bind_pose_bounds(&g.vertices);
        let morphs = (!g.morphs.is_empty()).then(|| Arc::new(g.morphs.clone()));
        let template = out.push_slot(g, &entry, joint_count, bounds, morphs.as_ref(), true);
        for _ in 0..g.mesh.max_instances {
            let copy = out.push_slot(g, &entry, joint_count, bounds, morphs.as_ref(), false);
            out.pool_reservations.push((template, copy));
        }
        out.templates.push(SkinnedTemplate {
            handle: g.handle,
            name_id: g.name_id,
            template_index: template,
            skeleton,
            morph_names: g.morphs.names.clone(),
            model: g.mesh.model_matrix(),
            capsule: g.mesh.capsule.clone(),
            transform: Transform {
                position: g.mesh.position,
                rotation_deg: g.mesh.rotation_deg,
                scale: g.mesh.scale,
            },
            local_bounds: bounds,
        });
    }
    Ok(out)
}

/// Spawn one posed template entity per skinned mesh, identified by its mesh's
/// id so a runtime `SpawnRequest` can clone it, with a character rig when it
/// declares a capsule. With a `PickIndex` published, each template is also
/// placed by a Transform and joins `picks`.
pub fn spawn_skinned_templates(
    ctx: &mut PipelineContext<'_>,
    templates: Vec<SkinnedTemplate>,
    picks: &mut Vec<PickCandidate>,
) {
    let shapes = character_shape::collect(ctx);
    let want_pick = ctx.resource::<PickIndex>().is_some();
    for SkinnedTemplate {
        handle,
        name_id,
        template_index,
        skeleton,
        morph_names,
        model,
        capsule,
        transform,
        local_bounds,
    } in templates
    {
        let layers = shapes
            .get(&handle)
            .map(|shape| character_shape::resolve(shape, &skeleton, &morph_names));
        let capsule = capsule.map(|c| match &layers {
            Some(l) => character_shape::proportioned_capsule(&c, &skeleton, &l.proportions),
            None => (c.half_height, c.radius),
        });
        let entity = ctx.components.spawn();
        ctx.insert(
            entity,
            character_shape::seed_pose(handle, template_index, skeleton, layers),
        );
        if want_pick {
            ctx.insert(entity, transform);
            ctx.insert(entity, GlobalTransform(model));
            picks.push(PickCandidate {
                asset_id: name_id,
                entity,
                local_min: local_bounds.0,
                local_max: local_bounds.1,
            });
        }
        ctx.identify(entity, name_id);
        // The physics creates the kinematic capsule from the rig, and the
        // render transform follows it.
        if let Some((half_height, radius)) = capsule {
            ctx.push(CharacterRig::new(
                handle,
                template_index,
                model,
                half_height.max(0.05),
                radius.max(0.05),
            ));
        }
    }
}

#[cfg(test)]
mod tests;
