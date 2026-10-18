//! The static draw list: every prop and instanced prop baked into draw
//! objects, with each prop entity given its draw slots and init world matrix.

use alloc::vec::Vec;
use hashbrown::HashMap;

use crate::components::{GlobalTransform, InstancedProp, PropInstance, RenderHandle, SubMeshRef};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{Entity, MaterialHandle, PickIndex, PipelineContext};
use crate::gfx::render_types::DrawIndex;
use crate::memory::InlineVec;
use crate::render::draw_list::{self, DrawListData, LoadedMesh, RoomGeometry};
use crate::render::extract::PickCandidate;
use crate::render::material_entry::MaterialEntry;
use crate::render::terrain::LoadedTerrain;
use crate::transform::propagation;

/// The decoded geometry and materials the draw list resolves props against.
pub struct PropDrawInputs<'a> {
    /// Each model's sub-meshes, by asset id.
    pub model_map: &'a HashMap<AssetId, Vec<SubMeshRef>>,
    /// The decoded meshes.
    pub mesh_geometry: &'a [LoadedMesh],
    /// The decoded room geometry.
    pub room_geometry: &'a [RoomGeometry],
    /// The decoded terrains.
    pub terrains: &'a [LoadedTerrain],
    /// Texture slots in the pool.
    pub texture_count: usize,
    /// Each material's resolved entry.
    pub material_map: &'a HashMap<MaterialHandle, MaterialEntry>,
}

/// Build the draw list, attach a RenderHandle and GlobalTransform to every
/// prop entity, and (under a PickIndex, the editor's opt-in) refill
/// `pick_candidates`. Drains InstancedProp: every instance becomes a baked draw.
pub fn assemble_prop_draws(
    ctx: &mut PipelineContext,
    inputs: PropDrawInputs<'_>,
    pick_candidates: &mut Vec<PickCandidate>,
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
    let mut items = Vec::with_capacity(prop_entities.len());
    let mut world_mats = Vec::with_capacity(prop_entities.len());
    for &entity in &prop_entities {
        items.push(draw_list::decomposed_renderable_item(ctx, entity));
        world_mats.push(
            resolved
                .get(&entity)
                .copied()
                .unwrap_or(crate::transform::IDENTITY),
        );
    }

    let data = draw_list::build_draw_list(draw_list::DrawListInputs {
        items: &items,
        instanced_props: &instanced_props,
        world_mats: &world_mats,
        model_map: inputs.model_map,
        mesh_geometry: inputs.mesh_geometry,
        room_geometry: inputs.room_geometry,
        terrains: inputs.terrains,
        texture_count: inputs.texture_count,
        material_map: inputs.material_map,
    })?;

    // A shipped runtime never has a PickIndex: capture each prop's pick
    // candidate only for the editor, so the frame step can refresh the index
    // from the live transforms.
    pick_candidates.clear();
    let want_pick = ctx.resource::<PickIndex>().is_some();
    for (i, &entity) in prop_entities.iter().enumerate() {
        let draws: InlineVec<DrawIndex> = data.prop_draw_indices[i].iter().copied().collect();
        ctx.insert(entity, RenderHandle { draws });
        ctx.insert(entity, GlobalTransform(world_mats[i]));
        if want_pick && let Some(asset_id) = items[i].asset_id {
            let (local_min, local_max) = data.prop_local_bounds[i];
            pick_candidates.push(PickCandidate {
                asset_id,
                entity,
                local_min,
                local_max,
            });
        }
    }
    Some(data)
}
