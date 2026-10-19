use super::*;
use crate::blob::SceneGroup;
use crate::ecs::World;
use alloc::vec;

const A: AssetId = AssetId(1);
const B: AssetId = AssetId(2);

fn content<'a>(
    texture_slots: Option<usize>,
    mesh_draws: Option<&'a [DrawIndex]>,
    deferred_shaders: &'a [(u32, AssetId)],
) -> StreamedContent<'a> {
    StreamedContent {
        texture_slots,
        mesh_draws,
        shaders_stream: !deferred_shaders.is_empty(),
        deferred_shaders,
    }
}

fn members(residency: &SceneResidency) -> Vec<(u8, u32)> {
    let mut all: Vec<(u8, u32)> = residency.all_members().collect();
    all.sort_unstable();
    all
}

#[test]
fn nothing_streaming_needs_no_residency() {
    let mut world = World::new();
    assert!(scene_residency(&world.context(), &[A], content(None, None, &[])).is_none());
}

// Textures come from the baked groups (in range of the pool only), meshes
// from the streamed draws' scene members, shaders from the deferred buckets.
#[test]
fn each_pool_contributes_its_scene_members() {
    let mut world = World::new();
    world.insert_resource(BlobSceneGroups(vec![SceneGroup {
        scene: A,
        resources: vec![(ResourceKind::Texture, 1), (ResourceKind::Texture, 9)],
        defs: Vec::new(),
    }]));
    let entity = world.context().components.spawn();
    world.context().insert(entity, SceneMember(B));
    world.context().insert(
        entity,
        RenderHandle {
            draws: vec![DrawIndex(5)].into(),
        },
    );
    let draws = [DrawIndex(3), DrawIndex(5)];
    let residency = scene_residency(
        &world.context(),
        &[A, B],
        content(Some(4), Some(&draws), &[(2, A)]),
    )
    .expect("content streams");
    assert_eq!(
        members(&residency),
        [(CHANNEL_TEXTURE, 1), (CHANNEL_MESH, 1), (CHANNEL_SHADER, 2)]
    );
}
