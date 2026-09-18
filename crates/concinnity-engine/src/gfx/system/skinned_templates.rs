// Skinned mesh templates: the shared geometry upload, then one posed entity
// per mesh for AnimationSystem to drive and a runtime spawn to clone.

use std::sync::Arc;

use concinnity_core::animation::skeleton;
use concinnity_core::components::{CharacterCapsule, CharacterRig, GlobalTransform, Transform};
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{PickIndex, PipelineContext, SkinnedMeshHandle};
use concinnity_core::gfx::mesh_payload::{PayloadMorphs, SkinnedVertex};
use concinnity_core::gfx::render_types::SkinnedDrawObject;
use concinnity_core::render::backend::RenderBackend;
use concinnity_core::render::error::RenderResult;

use super::{GraphicsSystem, PickCandidate, character_shape};

// One skinned mesh's skeleton bookkeeping, driving the SkeletonPose +
// CharacterRig publish after the geometry upload. `template_index` is recorded
// explicitly rather than inferred from position because pre-reserved instance
// copies interleave the draw-object list (template, copies, template, ...).
pub(super) struct SkinnedSkeletonEntry {
    pub(super) handle: SkinnedMeshHandle,
    pub(super) name_id: AssetId,
    pub(super) template_index: usize,
    pub(super) skeleton: skeleton::Skeleton,
    pub(super) morph_names: Vec<String>,
    pub(super) model: [[f32; 4]; 4],
    pub(super) capsule: Option<CharacterCapsule>,
    // The authored placement and local bounds, for the editor's pick index.
    pub(super) transform: Transform,
    pub(super) local_bounds: ([f32; 3], [f32; 3]),
}

// The shared skinned buffers and per-slot records the backend uploads once.
pub(super) struct SkinnedUpload {
    pub(super) vertices: Vec<SkinnedVertex>,
    pub(super) indices: Vec<u32>,
    pub(super) draw_objects: Vec<SkinnedDrawObject>,
    pub(super) morphs: Vec<Option<Arc<PayloadMorphs>>>,
}

// Upload the skinned geometry, then the morph targets when any slot has some.
// The hidden instance copies among the draw objects seed the skinned pool.
fn upload_skinned_geometry(
    backend: &mut dyn RenderBackend,
    upload: SkinnedUpload,
) -> RenderResult<()> {
    backend.upload_skinned(&upload.vertices, &upload.indices, upload.draw_objects)?;
    if upload.morphs.iter().any(|m| m.is_some()) {
        backend.upload_skinned_morphs(upload.morphs)?;
    }
    Ok(())
}

impl GraphicsSystem {
    // Upload the skinned geometry and spawn one template per skinned mesh. The
    // poses are published whatever the backend, so the system graph is identical.
    pub(super) fn install_skinned_templates(
        &mut self,
        ctx: &mut PipelineContext,
        upload: SkinnedUpload,
        skeletons: Vec<SkinnedSkeletonEntry>,
    ) -> RenderResult<()> {
        if skeletons.is_empty() {
            return Ok(());
        }
        if let Some(backend) = self.backend.as_deref_mut() {
            upload_skinned_geometry(backend, upload)?;
        }
        self.spawn_skinned_templates(ctx, skeletons);
        Ok(())
    }

    // Spawn one posed template entity per skinned mesh, registered by name so a
    // runtime SpawnRequest can clone it, with a character rig when it declares
    // a capsule.
    fn spawn_skinned_templates(
        &mut self,
        ctx: &mut PipelineContext,
        skeletons: Vec<SkinnedSkeletonEntry>,
    ) {
        let skinned_count = skeletons.len();
        let shapes = character_shape::collect(ctx);
        let want_pick = ctx.resource::<PickIndex>().is_some();
        for SkinnedSkeletonEntry {
            handle,
            name_id,
            template_index,
            skeleton,
            morph_names,
            model,
            capsule,
            transform,
            local_bounds,
        } in skeletons
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
            // Under the editor the template is pickable and movable like a
            // prop: its Transform drives the per-frame skinned model push and
            // its bounds join the pick index.
            if want_pick {
                ctx.insert(entity, transform);
                ctx.insert(entity, GlobalTransform(model));
                self.pick_candidates.push(PickCandidate {
                    asset_id: name_id,
                    entity,
                    local_min: local_bounds.0,
                    local_max: local_bounds.1,
                });
            }
            // A runtime SpawnRequest resolves the template by its mesh's id,
            // the same way the static spawn path resolves a placement.
            ctx.identify(entity, name_id);
            // PhysicsSystem (init runs later this tick) creates the kinematic
            // capsule from the rig, and the render transform follows it.
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
        tracing::info!("GraphicsSystem: {} skinned mesh(es) ready", skinned_count);
    }
}
