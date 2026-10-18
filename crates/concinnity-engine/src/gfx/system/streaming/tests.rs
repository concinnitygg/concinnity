use concinnity_core::ecs::Ref;
use concinnity_core::gfx::render_types::{MaterialUniforms, NO_NORMAL_MAP_SLOT};
use concinnity_core::render::backend::recording::{Call, SharedRecording, recording_backend};

use super::*;

const MIB: u64 = 1024 * 1024;

// A GraphicsSystem carrying the recording backend, as init has it while the
// setup routines wire the pools onto it. The mock reports the UNKNOWN GPU
// profile (0 bytes of memory), so a pool with no explicit override lands on
// its count-only policy.
fn system_with_backend() -> (SharedRecording, GraphicsSystem) {
    let (recorded, backend) = recording_backend();
    let mut gs = GraphicsSystem::new(None);
    gs.backend = Some(Box::new(backend));
    (recorded, gs)
}

// The payload bytes are never decoded at setup (the streamer's worker does
// that later), so any bytes stand in for a compiled texture.
fn texture_payloads(n: usize) -> Vec<Vec<u8>> {
    (0..n).map(|i| vec![i as u8; 8]).collect()
}

fn centers(n: usize) -> Vec<Vec<[f32; 3]>> {
    (0..n).map(|i| vec![[i as f32, 0.0, 0.0]]).collect()
}

fn mesh_payloads(n: usize) -> Vec<DecodedMesh> {
    (0..n)
        .map(|_| DecodedMesh {
            vertices: Vec::new(),
            indices: Vec::new(),
        })
        .collect()
}

fn mesh_setup(payloads: usize, draws: Vec<DrawIndex>) -> MeshStreamSetup {
    MeshStreamSetup {
        payloads: mesh_payloads(payloads),
        centers: centers(payloads),
        draw_indices: draws,
        disk_backed: false,
        seed_region: None,
        deferred_payloads: Default::default(),
    }
}

// A streamable world's texture slots are all evicted to placeholders at
// setup; the streamer then brings them back nearest-first.
#[test]
fn texture_setup_evicts_every_slot_and_builds_the_pool() {
    let (recorded, mut gs) = system_with_backend();
    gs.setup_texture_streaming(
        Some(StreamingConfig::default()),
        texture_payloads(2),
        &[],
        false,
        centers(2),
    );

    assert_eq!(gs.stream_pools.texture.as_ref().map(|s| s.len()), Some(2));
    assert!(gs.stream_workers.texture.is_some());
    let s = recorded.lock();
    assert!(s.saw(&Call::EvictTextureSlot(0)));
    assert!(s.saw(&Call::EvictTextureSlot(1)));
}

// The three guards that leave a world unstreamed: no config declared, no
// streamable slot, and no backend to wire the pool onto.
#[test]
fn texture_setup_is_skipped_without_a_config_slots_or_backend() {
    let (recorded, mut gs) = system_with_backend();
    gs.setup_texture_streaming(None, texture_payloads(2), &[], false, centers(2));
    assert!(
        gs.stream_pools.texture.is_none(),
        "no StreamingConfig declared"
    );

    gs.setup_texture_streaming(
        Some(StreamingConfig::default()),
        Vec::new(),
        &[],
        false,
        Vec::new(),
    );
    assert!(gs.stream_pools.texture.is_none(), "no slot to stream");
    assert!(recorded.lock().calls.is_empty());

    let mut headless = GraphicsSystem::new(None);
    headless.setup_texture_streaming(
        Some(StreamingConfig::default()),
        texture_payloads(1),
        &[],
        false,
        centers(1),
    );
    assert!(headless.stream_pools.texture.is_none(), "no backend");
}

// An explicit texture budget wins over the VRAM-derived fraction; with no
// override the mock's unreporting GPU leaves the pool count-only.
#[test]
fn texture_setup_honors_an_explicit_byte_budget() {
    let (_recorded, mut gs) = system_with_backend();
    gs.setup_texture_streaming(
        Some(StreamingConfig {
            texture_budget_mb: 8,
            ..Default::default()
        }),
        texture_payloads(1),
        &[],
        false,
        centers(1),
    );
    assert_eq!(
        gs.stream_pools.texture.as_ref().unwrap().byte_budget(),
        Some(8 * MIB)
    );

    let (_r2, mut count_only) = system_with_backend();
    count_only.setup_texture_streaming(
        Some(StreamingConfig::default()),
        texture_payloads(1),
        &[],
        false,
        centers(1),
    );
    let pool = count_only.stream_pools.texture.as_ref().unwrap();
    assert_eq!(pool.byte_budget(), None);
}

// A disk-backed world whose blob cannot be resolved gets no payload source,
// so its textures stay as built rather than evicted to placeholders that
// nothing would ever stream back.
#[test]
fn an_unresolvable_disk_source_leaves_the_textures_alone() {
    let (recorded, mut gs) = system_with_backend();
    let locators = vec![PayloadLocator {
        blob_index: 7,
        offset: 0,
        len: 8,
    }];
    gs.setup_texture_streaming(
        Some(StreamingConfig::default()),
        Vec::new(),
        &locators,
        true,
        centers(1),
    );
    assert!(gs.stream_pools.texture.is_none());
    assert!(!recorded.lock().saw(&Call::EvictTextureSlot(0)));
}

// With a shrinkable seed reserved, the streamed geometry was never baked in:
// the sub-allocators are seeded with the headroom block, and evicting here
// would free the placeholder region out from under the first resident draw.
#[test]
fn mesh_setup_seeds_the_reserved_region_instead_of_evicting() {
    let (recorded, mut gs) = system_with_backend();
    gs.setup_mesh_streaming(
        Some(StreamingConfig::default()),
        MeshStreamSetup {
            seed_region: Some(mesh_seed::MeshSeedRegion {
                vtx_offset: 0,
                vtx_bytes: 1024,
                idx_offset: 0,
                idx_bytes: 512,
            }),
            ..mesh_setup(2, vec![DrawIndex(4), DrawIndex(7)])
        },
    );

    let s = recorded.lock();
    assert!(s.saw(&Call::SeedMeshStreaming));
    assert!(
        !s.calls.iter().any(|c| matches!(c, Call::EvictMesh(_))),
        "evicting would corrupt the placeholder region"
    );
}

// Without a seed region each streamed mesh's build-time region is freed into
// the sub-allocators instead, and the id to draw map is retained.
#[test]
fn mesh_setup_frees_each_build_time_region_without_a_seed() {
    let (recorded, mut gs) = system_with_backend();
    gs.setup_mesh_streaming(
        Some(StreamingConfig::default()),
        mesh_setup(2, vec![DrawIndex(4), DrawIndex(7)]),
    );

    assert_eq!(gs.stream_pools.mesh.as_ref().map(|s| s.len()), Some(2));
    assert_eq!(gs.stream_pools.mesh_draws, vec![DrawIndex(4), DrawIndex(7)]);
    let s = recorded.lock();
    assert!(s.saw(&Call::EvictMesh(DrawIndex(4))));
    assert!(s.saw(&Call::EvictMesh(DrawIndex(7))));
    assert!(!s.saw(&Call::SeedMeshStreaming));
}

#[test]
fn mesh_setup_is_skipped_without_a_config_payloads_or_backend() {
    let (recorded, mut gs) = system_with_backend();
    gs.setup_mesh_streaming(None, mesh_setup(1, vec![DrawIndex(0)]));
    assert!(
        gs.stream_pools.mesh.is_none(),
        "no StreamingConfig declared"
    );

    gs.setup_mesh_streaming(Some(StreamingConfig::default()), mesh_setup(0, Vec::new()));
    assert!(gs.stream_pools.mesh.is_none(), "no mesh to stream");
    assert!(recorded.lock().calls.is_empty());

    let mut headless = GraphicsSystem::new(None);
    headless.setup_mesh_streaming(
        Some(StreamingConfig::default()),
        mesh_setup(1, vec![DrawIndex(0)]),
    );
    assert!(headless.stream_pools.mesh.is_none(), "no backend");
}

#[test]
fn mesh_setup_honors_an_explicit_byte_budget() {
    let (_recorded, mut gs) = system_with_backend();
    gs.setup_mesh_streaming(
        Some(StreamingConfig {
            mesh_budget_mb: 16,
            ..Default::default()
        }),
        mesh_setup(1, vec![DrawIndex(0)]),
    );
    assert_eq!(
        gs.stream_pools.mesh.as_ref().unwrap().byte_budget(),
        Some(16 * MIB)
    );
}

fn solid_block() -> BlockType {
    BlockType {
        solid: true,
        uv_min: [0.0, 0.0],
        uv_max: [1.0, 1.0],
        ..Default::default()
    }
}

// A VoxelWorld resolves its shared material to texture-pool slots + scalars
// and stands the chunk pool up on the backend at the world's chunk size.
#[test]
fn voxel_setup_resolves_the_material_and_builds_the_chunk_pool() {
    let (recorded, mut gs) = system_with_backend();
    let air = AssetId(1);
    let ground = AssetId(2);
    let handle = MaterialHandle::new(0);
    let mut material = MaterialUniforms::DEFAULT;
    material.roughness = 0.25;

    let block_types = HashMap::from([
        (
            air,
            BlockType {
                solid: false,
                ..Default::default()
            },
        ),
        (ground, solid_block()),
    ]);
    let material_map = HashMap::from([(
        handle,
        MaterialEntry {
            albedo_slot: 5,
            normal_map_slot: 6,
            uniforms: material,
            shader_bucket: 0,
        },
    )]);

    gs.setup_voxel_world_streaming(
        Some(VoxelWorld {
            chunk_blocks: [8, 16, 8],
            block_size: 2.0,
            view_radius: 1,
            palette: vec![Ref::new(air), Ref::new(ground)],
            material: Some(handle),
            ..Default::default()
        }),
        &block_types,
        &material_map,
    );

    let cs = gs.stream_pools.chunks.as_ref().expect("chunk pool built");
    assert_eq!(cs.material.texture_slot, 5);
    assert_eq!(cs.material.normal_map_slot, 6);
    assert_eq!(cs.material.material.roughness, 0.25);
    // 8 blocks of 2 world units on each of X / Z.
    assert_eq!((cs.chunk_w, cs.chunk_d), (16.0, 16.0));
    assert!(recorded.lock().saw(&Call::SetupChunkStreaming));
    assert!(gs.stream_workers.chunk.is_some());
}

// A VoxelWorld naming no material renders its chunks with the engine
// defaults rather than failing the world.
#[test]
fn voxel_setup_without_a_material_falls_back_to_the_defaults() {
    let (_recorded, mut gs) = system_with_backend();
    gs.setup_voxel_world_streaming(
        Some(VoxelWorld {
            view_radius: 1,
            ..Default::default()
        }),
        &HashMap::new(),
        &HashMap::new(),
    );

    let cs = gs.stream_pools.chunks.as_ref().expect("chunk pool built");
    assert_eq!(cs.material.texture_slot, 0);
    assert_eq!(cs.material.normal_map_slot, NO_NORMAL_MAP_SLOT);
    assert_eq!(cs.origin_chunk, chunk_coord::ChunkCoord::new(0, 0));
}

// A palette naming an id that is not a BlockType degrades that entry to air:
// the world still streams rather than failing over one bad name.
#[test]
fn an_unknown_palette_entry_degrades_to_air() {
    let (_recorded, mut gs) = system_with_backend();
    let known = AssetId(1);
    gs.setup_voxel_world_streaming(
        Some(VoxelWorld {
            view_radius: 1,
            palette: vec![Ref::new(AssetId(99)), Ref::new(known)],
            ..Default::default()
        }),
        &HashMap::from([(known, solid_block())]),
        &HashMap::new(),
    );
    assert!(gs.stream_pools.chunks.is_some(), "the world still streams");
}

#[test]
fn voxel_setup_is_skipped_without_a_voxel_world_or_backend() {
    let (recorded, mut gs) = system_with_backend();
    gs.setup_voxel_world_streaming(None, &HashMap::new(), &HashMap::new());
    assert!(gs.stream_pools.chunks.is_none(), "no VoxelWorld declared");
    assert!(recorded.lock().calls.is_empty());

    let mut headless = GraphicsSystem::new(None);
    headless.setup_voxel_world_streaming(
        Some(VoxelWorld::default()),
        &HashMap::new(),
        &HashMap::new(),
    );
    assert!(headless.stream_pools.chunks.is_none(), "no backend");
}

// Chunk streaming has no user override, so an unreporting GPU leaves the
// pool on its pure radius-only policy.
#[test]
fn chunk_streaming_is_count_only_when_the_gpu_reports_no_memory() {
    let (_recorded, mut gs) = system_with_backend();
    gs.setup_voxel_world_streaming(
        Some(VoxelWorld {
            view_radius: 1,
            ..Default::default()
        }),
        &HashMap::new(),
        &HashMap::new(),
    );
    assert_eq!(
        gs.stream_pools.chunks.as_ref().unwrap().pool.byte_budget(),
        None
    );
}
