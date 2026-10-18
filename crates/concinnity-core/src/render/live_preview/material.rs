//! Live edits of a loaded Material's Shader parameters. The parameters live in
//! the renderer's material parameter table, one row per material, rather than in
//! any draw, so a change rewrites that one row and every draw of the material
//! reads the new values from then on. The world's decoded `MaterialTable` is
//! kept in step, so a later decode of the material agrees with what is drawn.

use crate::components::Material;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;
use crate::gfx::render_types::MATERIAL_PARAM_COUNT;
use crate::render::material_params;
use crate::resource::MaterialTable;

use super::draw::material_handle;
use super::{has_queue, with_ops};

/// Whether the running world has a renderer to send a parameter change to.
pub fn is_available(world: &World) -> bool {
    has_queue(world)
}

/// Set the `params` of the `Material` asset interned as `name`. `false` when
/// the world has no renderer or loaded no material under that name (only a dev
/// session records material identities), which leaves the world untouched.
pub fn apply_params(world: &mut World, name: AssetId, params: [f32; MATERIAL_PARAM_COUNT]) -> bool {
    let Some(handle) = material_handle(world, name) else {
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
    with_ops(world, |ops| {
        ops.record(move |backend| {
            if let Some(edit) = backend.live_edit() {
                edit.set_material_params(row, params);
            }
        });
    })
    .is_some()
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::super::test_support::Fixture;
    use super::*;
    use crate::render::backend::recording::Call;
    use crate::resource::{MaterialNames, ResourceEntry};

    const PARAMS: [f32; MATERIAL_PARAM_COUNT] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // Two loaded materials: names 10 and 20 at handles 0 and 1.
    fn world() -> World {
        let mut world = World::new();
        let entry = |roughness| ResourceEntry {
            payload: None,
            baked: Vec::new(),
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

    fn decoded(world: &World, handle: usize) -> Material {
        let table = world.resource::<MaterialTable>().expect("table");
        postcard::from_bytes(&table.0[handle].data_bytes).expect("decodes")
    }

    #[test]
    fn a_change_rewrites_the_materials_row() {
        let mut f = Fixture::new(world());
        assert!(apply_params(&mut f.world, AssetId(20), PARAMS));
        assert_eq!(
            f.replay(),
            vec![Call::SetMaterialParams {
                row: 2,
                params: PARAMS
            }]
        );
    }

    // The decoded table follows the edit and nothing else in the material moves.
    #[test]
    fn the_decoded_table_follows_the_change() {
        let mut f = Fixture::new(world());
        assert!(apply_params(&mut f.world, AssetId(10), PARAMS));
        let mat = decoded(&f.world, 0);
        assert_eq!(mat.params, PARAMS);
        assert_eq!(mat.roughness, 0.25);
        assert_eq!(decoded(&f.world, 1).params, [0.0; MATERIAL_PARAM_COUNT]);
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
        let mut f = Fixture::new(world());
        assert!(!apply_params(&mut f.world, AssetId(99), PARAMS));
        assert!(f.replay().is_empty());
    }
}
