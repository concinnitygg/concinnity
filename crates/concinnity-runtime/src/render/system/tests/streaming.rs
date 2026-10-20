use super::*;
use concinnity_engine::render::streaming::pools::{StreamingRuntime, StreamingStats};

// The streaming pools graphics init publishes, for the streaming assertions.
fn streaming_stats(world: &World) -> StreamingStats {
    StreamingRuntime::stats_in(world).expect("streaming runtime published at init")
}

#[test]
fn streaming_init_evicts_streamable_slots() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(StreamingConfig::default());
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    // The pools were handed off to the parked streaming runtime, so the
    // system's own scratch fields are cleared.
    assert!(gs.stream_pools.texture.is_none());
    assert!(gs.stream_pools.mesh.is_none());
    let stats = streaming_stats(&world);
    assert!(stats.texture.is_some(), "texture pool streaming");
    assert!(stats.mesh.is_some(), "mesh pool streaming");

    let s = lock(&state);
    // The albedo pool slot is evicted to a placeholder at init; the mesh
    // keeps its build-time region (cap covers the whole set) but is evicted
    // so the streamer brings it back nearest-first.
    assert!(s.saw(&Call::EvictTextureSlot(0)));
    assert!(s.saw(&Call::EvictMesh(DrawIndex(0))));
}

#[test]
fn texture_streaming_uploads_evicted_slots() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(StreamingConfig::default());
    let mut world = b.build();
    let mut gs = init_graphics(&mut world, hooks);

    // The streamer decodes on a worker thread; step until the completed
    // upload lands.
    concinnity_testing::eventually("the texture to stream back in", || {
        assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
        lock(&state)
            .calls
            .iter()
            .any(|c| {
                matches!(
                    c,
                    Call::UpdateTextureSlot {
                        slot: 0,
                        w: 2,
                        h: 2
                    }
                )
            })
            .then_some(())
    });
    let (resident, pending, unloaded) = streaming_stats(&world).texture.unwrap();
    assert_eq!((resident, pending, unloaded), (1, 0, 0));
}

#[test]
fn mesh_streaming_reuploads_evicted_geometry() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(StreamingConfig::default());
    let mut world = b.build();
    let mut gs = init_graphics(&mut world, hooks);

    concinnity_testing::eventually("the mesh to stream back in", || {
        assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
        lock(&state)
            .calls
            .iter()
            .any(|c| {
                matches!(
                    c,
                    Call::UploadMesh {
                        draw_idx: DrawIndex(0),
                        vertices: 4,
                        indices: 6,
                    }
                )
            })
            .then_some(())
    });
    let (resident, pending, unloaded) = streaming_stats(&world).mesh.unwrap();
    assert_eq!((resident, pending, unloaded), (1, 0, 0));
}

// A VoxelWorld rebases the draw's view + camera onto the chunk render origin:
// StreamingSystem publishes the camera-relative pair and GraphicsSystem submits
// it. Exercises the view-rebase timing across the StreamingSystem/GraphicsSystem
// split (a non-voxel world leaves the absolute view untouched -- see
// `camera_state_reaches_the_backend_each_frame`). No chunk needs to be resident:
// the rebase is a function of the camera position alone.
#[test]
fn voxel_world_rebases_the_draw_view_onto_the_chunk_origin() {
    use concinnity_engine::components::VoxelWorld;
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    // Default chunk is 16x16 world units; a small view radius keeps the chunk
    // dispatch light (no chunk is awaited).
    b.push(VoxelWorld {
        seed: 1,
        view_radius: 1,
        ..Default::default()
    });
    let mut world = b.build();
    let mut gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);
    // The chunk pool was handed to StreamingSystem's parked state.
    assert!(
        streaming_stats(&world).chunk.is_some(),
        "chunk pool streaming"
    );

    // Place the camera two chunks east and three chunks north of the origin
    // (floor(40/16)=2, floor(-40/16)=-3), so the rebase is non-trivial.
    let cam_pos = [40.0, 5.0, -40.0];
    let view = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-40.0, -5.0, 40.0, 1.0],
    ];
    {
        let mut ctx = world.context();
        let cam = ctx.query_mut::<Camera3D>().next().unwrap();
        cam.position = cam_pos;
        cam.view_matrix = view;
    }
    step(&mut gs, &mut world);

    // Render origin = chunk (2, -3) -> world (32, -48).
    let origin = [32.0, 0.0, -48.0];
    let expected_cam = [8.0, 5.0, 8.0];
    let expected_view = chunk_coord::camera_relative_view(view, cam_pos, origin);

    let s = lock(&state);
    match s.last_draw_frame() {
        Some(Call::DrawFrame { cam_pos: cp, .. }) => {
            assert_eq!(
                cp, expected_cam,
                "draw camera rebased onto the chunk origin"
            );
            assert_ne!(cp, cam_pos, "the rebase actually moved the camera");
        }
        other => panic!("expected a DrawFrame, got {other:?}"),
    }
    assert!(
        s.saw(&Call::UpdateView(expected_view)),
        "the camera-relative view reaches update_view, not the absolute one"
    );
    assert!(
        !s.saw(&Call::UpdateView(view)),
        "the absolute view is not what the backend received"
    );
}

// The hot-reload seams concinnity-dev drives through. The library never reads
// them, so they are exercised here: init parks the pushed fog beside the backend,
// while the source catalogs and the texture-name map are captured only under
// `cn debug` (this world is a plain build, so neither is parked).
#[test]
fn hot_reload_seams_park_the_fog_beside_the_backend() {
    use concinnity_engine::components::VolumetricFog;

    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(VolumetricFog {
        enabled: true,
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    assert!(
        world
            .resource::<crate::live_edit::hot_reload_sources::HotReloadSources>()
            .is_none(),
        "a plain build captures no source catalog"
    );
    let resources = world.context().resources;
    let (backend, fog, slots) = resources.get_disjoint_mut::<
        concinnity_engine::ecs::ActiveRenderBackend,
        concinnity_engine::render::environment::volumetric_fog::PushedFogSettings,
        concinnity_engine::render::world_load::source_maps::TextureNameSlots,
    >();
    assert!(
        fog.is_some_and(|fog| fog.0.is_some()),
        "the reload pass dedupes against the fog init pushed to the backend"
    );
    assert!(
        slots.is_none(),
        "the texture-name map is captured only under cn debug"
    );
    backend
        .and_then(|slot| slot.0.as_deref_mut())
        .expect("init parks the backend")
        .wait_idle();
    assert!(
        lock(&state).saw(&Call::WaitIdle),
        "the parked backend is the one init built"
    );
}

// Shrinkable seed VRAM: by default the whole streamed mesh set is baked into the
// shared buffers, so streaming reuses space but never shrinks GPU memory. When
// the residency cap is smaller than the streamed set, the resident geometry is
// compacted and a smaller seed headroom -- sized to the cap-many largest meshes
// -- is reserved for the streamed ones, so the GPU buffers are born small.
#[test]
fn a_residency_cap_below_the_streamed_set_reserves_a_smaller_seed() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    // Three streamable quads against a cap of one, so the planner has a set to
    // shrink (the cap plus its margin must still fall short of the set).
    for i in 0..2 {
        b.push_textured_quad(
            AssetId(10 + i),
            AssetId(20 + i),
            AssetId(30 + i),
            AssetId(860 + i),
        );
    }
    b.push(StreamingConfig {
        mesh_cap: 1,
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    let s = lock(&state);
    assert!(
        s.saw(&Call::SeedMeshStreaming),
        "the backend's sub-allocator is seeded with the smaller headroom block"
    );
    let init = s.init.as_ref().unwrap();
    assert_eq!(init.draw_objects.len(), 3);
    // The compaction reserves headroom for the two most-resident meshes rather
    // than baking all three quads (4 vertices each) in.
    assert!(
        init.vertex_count < 12,
        "the seed buffer is born smaller than the whole streamed set: {}",
        init.vertex_count
    );
}
