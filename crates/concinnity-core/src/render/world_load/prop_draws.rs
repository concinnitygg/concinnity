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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{InstanceTransform, Prop};
    use crate::ecs::{MeshHandle, World};
    use crate::gfx::mesh_payload::Vertex;
    use crate::render::extract::FrameExtractor;
    use crate::render::snapshot::RenderSnapshot;
    use crate::test_support::vertex;
    use alloc::vec;

    const PROP: AssetId = AssetId(4);

    // A unit quad over x/z in [0, 1] at y = 0.
    fn quad() -> LoadedMesh {
        let at = |x, z| Vertex {
            pos: [x, 0.0, z],
            ..vertex()
        };
        LoadedMesh {
            vertices: vec![at(0.0, 0.0), at(1.0, 0.0), at(1.0, 1.0), at(0.0, 1.0)],
            indices: vec![0, 1, 2, 0, 2, 3],
            lod_alternates: Vec::new(),
            bounds: None,
            counts: None,
        }
    }

    // A world with one PROP placing the quad at (1, 2, 3), decomposed the way
    // the world's start does before graphics init.
    fn placed_quad() -> World {
        let mut world = World::new();
        world.push_identified(
            PROP,
            Prop {
                mesh: Some(MeshHandle::new(0)),
                position: [1.0, 2.0, 3.0],
                ..Default::default()
            },
        );
        crate::ecs::decompose::run(&mut world.context());
        world
    }

    fn assemble(world: &mut World, picks: &mut Vec<PickCandidate>) -> DrawListData {
        let meshes = [quad()];
        assemble_prop_draws(
            &mut world.context(),
            PropDrawInputs {
                model_map: &HashMap::new(),
                mesh_geometry: &meshes,
                room_geometry: &[],
                terrains: &[],
                texture_count: 0,
                material_map: &HashMap::new(),
            },
            picks,
        )
        .expect("assembles")
    }

    // An InstancedProp bakes its instances into one cluster rather than a draw
    // object each, and is drained: it has no per-frame update path.
    #[test]
    fn an_instanced_prop_bakes_into_one_cluster_and_is_drained() {
        let mut world = placed_quad();
        world.push(InstancedProp {
            mesh: Some(MeshHandle::new(0)),
            instances: (0..3)
                .map(|i| InstanceTransform {
                    position: [i as f32 * 2.0, 0.0, 0.0],
                    rotation_deg: [0.0; 3],
                    scale: [1.0; 3],
                })
                .collect(),
            ..Default::default()
        });
        let data = assemble(&mut world, &mut Vec::new());
        assert_eq!(data.instanced_clusters.len(), 1, "one mesh, one cluster");
        assert_eq!(data.draw_objects.len(), 1, "only the prop draws on its own");
        assert_eq!(world.context().query::<InstancedProp>().count(), 0);
    }

    // Under the editor's PickIndex each prop becomes a pick candidate, and the
    // frame publishes it with its world-space bounds; without one nothing is
    // captured.
    #[test]
    fn props_become_pick_candidates_only_under_a_pick_index() {
        let mut world = placed_quad();
        world.insert_resource(PickIndex::default());
        let mut fx = FrameExtractor::default();
        assemble(&mut world, fx.pick_candidates_mut());
        assert_eq!(fx.pick_candidates().len(), 1, "one candidate per prop");
        fx.extract(&mut world.context(), &mut RenderSnapshot::default(), 0);
        let index = world.resource::<PickIndex>().unwrap();
        assert_eq!(index.entries.len(), 1);
        let entry = &index.entries[0];
        assert_eq!(entry.asset_id, PROP);
        assert_eq!(entry.bb_min, [1.0, 2.0, 3.0]);
        assert_eq!(entry.bb_max, [2.0, 2.0, 4.0]);

        let mut world = placed_quad();
        let mut picks = Vec::new();
        assemble(&mut world, &mut picks);
        assert!(picks.is_empty());
    }
}
