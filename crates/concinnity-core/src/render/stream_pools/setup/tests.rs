use super::*;
use crate::bake::texture::TextureImage;
use crate::math::SaturatingFrom;
use alloc::vec;

const MIB: u64 = 1024 * 1024;

#[test]
fn a_world_streams_one_chunk_source_or_none() {
    let vw = VoxelWorld::default();
    let planet = Planet::default();
    assert!(matches!(
        ChunkSource::choose(Some(vw.clone()), None),
        ChunkSource::VoxelWorld(_)
    ));
    assert!(matches!(
        ChunkSource::choose(None, Some(planet.clone())),
        ChunkSource::Planet(_)
    ));
    assert!(matches!(ChunkSource::choose(None, None), ChunkSource::None));
    assert!(matches!(
        ChunkSource::choose(Some(vw), Some(planet)),
        ChunkSource::None
    ));
}

#[test]
fn texture_slots_count_locators_on_disk_and_payloads_in_memory() {
    assert_eq!(streamed_texture_slots(true, 0, 3), Some(3));
    assert_eq!(streamed_texture_slots(false, 5, 0), Some(5));
    assert_eq!(streamed_texture_slots(false, 0, 7), None);
}

// An authored budget wins; without one a GPU that reports nothing leaves the
// pool count-only.
#[test]
fn pool_byte_budgets_take_the_override_then_the_device() {
    let config = StreamingConfig {
        texture_budget_mb: 64,
        ..Default::default()
    };
    let textures = StreamPool::<TextureImage>::for_textures(&config, vec![vec![]; 2], 0);
    assert_eq!(textures.byte_budget(), Some(64 * MIB));
    assert_eq!(textures.len(), 2);

    let meshes = StreamPool::<DecodedMeshItem>::for_meshes(&config, vec![], 0);
    assert_eq!(meshes.byte_budget(), None);
    let device = 8 * 1024 * MIB;
    let meshes = StreamPool::<DecodedMeshItem>::for_meshes(&config, vec![], device);
    assert_eq!(
        meshes.byte_budget(),
        budget::derive_byte_budget(0, device, MESH_VRAM_FRACTION)
    );
}

type DecodedMeshItem = super::super::DecodedMesh;

#[test]
fn a_voxel_world_plan_sizes_its_window_headroom() {
    let vw = VoxelWorld::default();
    let plan = VoxelStream::plan(&vw, &HashMap::new(), &HashMap::new(), 0);
    let expected = budget::chunk_headroom(ChunkWindowShape {
        chunk_blocks: vw.chunk_blocks(),
        near_radius: u32::saturating_from(vw.view_radius()),
        far_radius: u32::saturating_from(vw.impostor_radius()),
        impostor_step: vw.impostor_step(),
    });
    assert_eq!(plan.headroom, expected);
    assert_eq!(plan.stream.chunk_w, vw.chunk_world_size().0);
    assert_eq!(plan.stream.origin_chunk, ChunkCoord::new(0, 0));
    assert!(plan.stream.draws.is_empty());
    assert_eq!(plan.stream.pool.byte_budget(), None);
}

#[test]
fn a_planet_plan_reserves_the_resident_tile_headroom() {
    let planet = Planet::default();
    let plan = PlanetStream::plan(&planet, None, &HashMap::new());
    assert_eq!(plan.headroom, budget::planet_tile_headroom());
    assert_eq!(plan.frame.shape, planet.shape());
}
