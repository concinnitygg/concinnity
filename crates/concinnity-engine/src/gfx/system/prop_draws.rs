// The static draw list: every prop and instanced prop baked into draw objects,
// with each prop entity given its draw slots and init world matrix.

use std::collections::{HashMap, HashSet};

use concinnity_core::components::{
    GlobalTransform, InstancedProp, PropInstance, RenderHandle, SubMeshRef,
};
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{Entity, MaterialHandle, PickIndex, PipelineContext};
use concinnity_core::memory::InlineVec;
use concinnity_core::transform::propagation;

use super::{GraphicsSystem, PickCandidate};
use crate::gfx::draw_list::{self, DrawListData, LoadedMesh, RoomGeometry};
use crate::gfx::material_entry::MaterialEntry;

// The decoded geometry and materials the draw list resolves props against.
pub(super) struct PropDrawInputs<'a> {
    pub(super) model_map: &'a HashMap<AssetId, Vec<SubMeshRef>>,
    pub(super) mesh_geometry: &'a [LoadedMesh],
    pub(super) room_geometry: &'a [RoomGeometry],
    pub(super) texture_count: usize,
    pub(super) material_map: &'a HashMap<MaterialHandle, MaterialEntry>,
    pub(super) always_resident_meshes: &'a HashSet<usize>,
}

impl GraphicsSystem {
    // Build the draw list, attach a RenderHandle and GlobalTransform to every
    // prop entity, and record the sky props and (under a PickIndex) the pick
    // candidates. Drains InstancedProp: every instance becomes a baked draw.
    pub(super) fn assemble_prop_draws(
        &mut self,
        ctx: &mut PipelineContext,
        inputs: PropDrawInputs<'_>,
    ) -> Option<DrawListData> {
        // Drained before taking prop references because a drain shifts the column.
        let instanced_props = ctx.drain::<InstancedProp>();

        // The PropInstance marker enumerates the props in Prop order. Transform
        // does not: a SkyRotation pivot carries one without drawing anything.
        let prop_entities: Vec<Entity> = ctx
            .query_with_entity::<PropInstance>()
            .map(|(entity, _)| entity)
            .collect();

        // `items` and `world_mats` are column-aligned with `prop_entities`.
        let resolved = propagation::resolve_world_matrices(ctx);
        let entity_name: HashMap<Entity, AssetId> = ctx
            .resource::<concinnity_core::ecs::EntityByName>()
            .map(|n| n.0.iter().map(|(&id, &e)| (e, id)).collect())
            .unwrap_or_default();
        let mut items = Vec::with_capacity(prop_entities.len());
        let mut world_mats = Vec::with_capacity(prop_entities.len());
        for &entity in &prop_entities {
            let asset_id = entity_name.get(&entity).copied().unwrap_or_default();
            items.push(draw_list::decomposed_renderable_item(ctx, entity, asset_id));
            world_mats.push(
                resolved
                    .get(&entity)
                    .copied()
                    .unwrap_or(draw_list::IDENTITY4),
            );
        }

        // The props that draw the sky, so the frame step can keep them centered
        // on the camera.
        self.sky_props = prop_entities
            .iter()
            .zip(&items)
            .filter(|(_, item)| {
                item.mesh
                    .is_some_and(|mesh| inputs.always_resident_meshes.contains(&mesh.index()))
            })
            .map(|(&entity, _)| entity)
            .collect();

        let data = draw_list::build_draw_list(draw_list::DrawListInputs {
            items: &items,
            instanced_props: &instanced_props,
            world_mats: &world_mats,
            model_map: inputs.model_map,
            mesh_geometry: inputs.mesh_geometry,
            room_geometry: inputs.room_geometry,
            texture_count: inputs.texture_count,
            material_map: inputs.material_map,
            always_resident_meshes: inputs.always_resident_meshes,
        })?;

        // A PickIndex resource is the editor's opt-in (a shipped runtime never
        // has one): capture each prop's pick candidate so the frame step can
        // refresh the index from the live transforms.
        self.pick_candidates.clear();
        let want_pick = ctx.resource::<PickIndex>().is_some();
        for (i, &entity) in prop_entities.iter().enumerate() {
            let draws: InlineVec<u32> = data.prop_draw_indices[i]
                .iter()
                .map(|&slot| slot as u32)
                .collect();
            ctx.insert(entity, RenderHandle { draws });
            ctx.insert(entity, GlobalTransform(world_mats[i]));
            if want_pick {
                let (local_min, local_max) = data.prop_local_bounds[i];
                self.pick_candidates.push(PickCandidate {
                    asset_id: items[i].asset_id,
                    entity,
                    local_min,
                    local_max,
                });
            }
        }
        Some(data)
    }
}
