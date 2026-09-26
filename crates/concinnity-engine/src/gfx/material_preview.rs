//! Live edits of a loaded Material's Shader parameters. The parameters live in
//! the renderer's material parameter table, one row per material, rather than in
//! any draw, so a change rewrites that one row: the call is recorded into the
//! frame's op queue, where submission replays it before the next draw, and every
//! draw of the material reads the new values from then on. The world's decoded
//! `MaterialTable` is kept in step, so a later decode of the material agrees
//! with what is drawn.

use concinnity_core::components::Material;
use concinnity_core::ecs::World;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::gfx::render_types::MATERIAL_PARAM_COUNT;
use concinnity_core::render::material_params;
use concinnity_core::resource::MaterialTable;

use crate::ecs::ActiveRenderQueues;
use crate::gfx::draw_preview;

/// Whether the running world has a renderer to send a parameter change to.
pub fn is_available(world: &World) -> bool {
    world
        .resource::<ActiveRenderQueues>()
        .is_some_and(|slot| slot.0.is_some())
}

/// Set the `params` of the `Material` asset interned as `name`. `false` when
/// the world has no renderer or loaded no material under that name (only a dev
/// session records material identities), which leaves the world untouched.
pub fn apply_params(world: &mut World, name: AssetId, params: [f32; MATERIAL_PARAM_COUNT]) -> bool {
    let Some(handle) = draw_preview::material_handle(world, name) else {
        return false;
    };
    if !is_available(world) {
        return false;
    }
    if let Some(entry) = world
        .resource_mut::<MaterialTable>()
        .and_then(|table| table.0.get_mut(handle.index()))
        && let Ok(mut mat) = postcard::from_bytes::<Material>(&entry.data_bytes)
    {
        mat.params = params;
        if let Ok(bytes) = postcard::to_allocvec(&mat) {
            entry.data_bytes = bytes;
        }
    }
    let row = material_params::row_of(Some(handle));
    draw_preview::with_ops(world, |ops| {
        ops.record(move |backend| backend.set_material_params(row, params));
    })
    .is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::RenderQueues;
    use crate::gfx::mock_backend::{Call, MockBackend, MockState, recording_backend};
    use crate::resource::MaterialNames;
    use concinnity_core::render::ops::RenderOps;
    use concinnity_core::resource::ResourceEntry;
    use std::sync::{Arc, Mutex};

    const PARAMS: [f32; MATERIAL_PARAM_COUNT] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // Two loaded materials: names 10 and 20 at handles 0 and 1.
    fn world() -> World {
        let mut world = World::new();
        let entry = |roughness| ResourceEntry {
            payload: None,
            data_bytes: postcard::to_allocvec(&Material {
                roughness,
                ..Default::default()
            })
            .expect("serializes"),
        };
        world.insert_resource(MaterialTable(vec![entry(0.25), entry(0.75)]));
        world.insert_resource(MaterialNames(vec![10, 20]));
        world
    }

    fn with_renderer(mut world: World) -> World {
        world.insert_resource(ActiveRenderQueues(Some(RenderQueues {
            ops: RenderOps::default(),
            slots: crate::gfx::render_slots::RenderSlots::new(0, true, &[]),
        })));
        world
    }

    fn replay(world: &mut World) -> Vec<Call> {
        let (calls, mut backend): (Arc<Mutex<MockState>>, MockBackend) = recording_backend();
        let mut queues = world
            .resource_mut::<ActiveRenderQueues>()
            .and_then(|slot| slot.0.take())
            .expect("the queue is parked again");
        queues.ops.replay(&mut backend);
        calls.lock().unwrap().calls.clone()
    }

    fn decoded(world: &World, handle: usize) -> Material {
        let table = world.resource::<MaterialTable>().expect("table");
        postcard::from_bytes(&table.0[handle].data_bytes).expect("decodes")
    }

    #[test]
    fn a_change_rewrites_the_materials_row() {
        let mut world = with_renderer(world());
        assert!(apply_params(&mut world, AssetId(20), PARAMS));
        assert_eq!(
            replay(&mut world),
            vec![Call::SetMaterialParams {
                row: 2,
                params: PARAMS
            }]
        );
    }

    // The decoded table follows the edit and nothing else in the material moves.
    #[test]
    fn the_decoded_table_follows_the_change() {
        let mut world = with_renderer(world());
        assert!(apply_params(&mut world, AssetId(10), PARAMS));
        let mat = decoded(&world, 0);
        assert_eq!(mat.params, PARAMS);
        assert_eq!(mat.roughness, 0.25);
        assert_eq!(decoded(&world, 1).params, [0.0; MATERIAL_PARAM_COUNT]);
    }

    #[test]
    fn a_world_without_a_renderer_takes_nothing() {
        let mut world = world();
        assert!(!is_available(&world));
        assert!(!apply_params(&mut world, AssetId(10), PARAMS));
        assert_eq!(decoded(&world, 0).params, [0.0; MATERIAL_PARAM_COUNT]);
    }

    #[test]
    fn a_material_the_world_never_loaded_takes_nothing() {
        let mut world = with_renderer(world());
        assert!(!apply_params(&mut world, AssetId(99), PARAMS));
        assert!(replay(&mut world).is_empty());
    }
}
