use alloc::string::ToString;
use alloc::vec::Vec;
use alloc::{format, vec};

use super::*;
use crate::gfx::mesh_payload::Vertex;
use crate::render::backend::recording::{Call, RecordingBackend, recording_backend};
use crate::render::frame::ops::ReplayOutcome;
use crate::render::slots::render_slots::RenderSlots;
use crate::render::streaming::pools::{ChunkRequest, LoadPool};

const IDENTITY4: [[f32; 4]; 4] = crate::transform::IDENTITY;

// Upper bound on frames a test drives waiting for a state; the bound turns a
// regression into a failure instead of a hang.
const MAX_FRAMES: usize = 64;

fn tri() -> DecodedMesh {
    let v = Vertex {
        pos: [0.0; 3],
        normal: [0.0, 1.0, 0.0],
        tangent: [1.0, 0.0, 0.0],
        color: [1.0; 3],
        uv: [0.0; 2],
    };
    DecodedMesh {
        vertices: vec![v; 3],
        indices: vec![0, 1, 2],
    }
}

fn translation_view(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [x, y, z, 1.0],
    ]
}

// A world with no deferred shader buckets.
struct NoShaderPump;

impl ShaderPump for NoShaderPump {
    fn set_blocked(&mut self, _bucket: u32, _blocked: bool) {}

    fn pump(&mut self, _ops: &mut RenderOps, _on_resident: &mut dyn FnMut(u32, bool)) {}
}

fn empty_state() -> StreamingState {
    StreamingState::new(StreamingPools::default(), 2)
}

// Texture + mesh pools of two items each: item 0 on the camera's origin and
// item 1 far out on +X. `resident_cap` chooses whether both fit (8) or the
// second must displace the first (1). Mesh stream ids map to draw slots 10 /
// 11 so an upload's routing shows in the call log.
fn pooled_state(resident_cap: usize) -> StreamingState {
    let centers = vec![vec![[0.0, 0.0, 0.0]], vec![[100.0, 0.0, 0.0]]];
    StreamingState::new(
        StreamingPools {
            texture: Some(StreamPool::new(centers.clone(), 4, resident_cap)),
            mesh: Some(StreamPool::new(centers, 4, resident_cap)),
            mesh_draws: vec![DrawIndex(10), DrawIndex(11)],
            ..Default::default()
        },
        2,
    )
}

// A chunk stream over 16x16 chunks at the given near / far radii.
fn chunk_stream(near: i32, far: i32) -> ChunkStream {
    ChunkStream {
        pool: ChunkPool::new(near, far, 64, 16.0, 16.0),
        draws: BTreeMap::new(),
        chunk_w: 16.0,
        chunk_d: 16.0,
        origin_chunk: ChunkCoord::new(0, 0),
        material: TileMaterial::from_entry(None),
    }
}

// What a worker answers for each request: the payload, or a failing chunk
// generation (so a chunk stays tracked but never uploads).
#[derive(Clone, Copy)]
struct Workers {
    chunks_fail: bool,
}

const WORKING: Workers = Workers { chunks_fail: false };

// Answer every request the last drive queued, as workers that finish before the
// next frame.
fn serve(state: &mut StreamingState, workers: Workers) {
    if let Some(pool) = state.texture_mut() {
        let ids: Vec<usize> = pool.mailbox().take_requests().collect();
        for id in ids {
            let image = TextureImage::rgba8(1, 1, vec![1, 2, 3, 4]);
            pool.mailbox().deliver((id, Ok(image)));
        }
    }
    if let Some(pool) = state.mesh_mut() {
        let ids: Vec<usize> = pool.mailbox().take_requests().collect();
        for id in ids {
            pool.mailbox().deliver((id, Ok(tri())));
        }
    }
    if let Some(pool) = state.chunk_pool_mut() {
        let requests: Vec<ChunkRequest> = pool.mailbox().take_requests().collect();
        for (coord, _) in requests {
            let load = if workers.chunks_fail {
                Err("test source".to_string())
            } else {
                Ok(tri())
            };
            pool.mailbox().deliver((coord, load));
        }
    }
}

fn queues() -> RenderQueues {
    RenderQueues {
        ops: RenderOps::default(),
        slots: RenderSlots::new(0, true, &[]),
    }
}

fn view(cam: [f32; 3], view: [[f32; 4]; 4], hidden: bool) -> DriveView<'static> {
    DriveView {
        cam_pos: cam,
        view,
        world_hidden: hidden,
        scene_pins: None,
        planet: None,
    }
}

// One frame: drive, replay the recorded ops onto the backend, roll back what
// failed, and let the workers answer the frame's requests.
fn frame_with(
    state: &mut StreamingState,
    backend: &mut RecordingBackend,
    queues: &mut RenderQueues,
    drive: &DriveView<'_>,
    workers: Workers,
) -> (([[f32; 4]; 4], [f32; 3]), ReplayOutcome) {
    let out = state.drive(queues, drive, None::<&mut NoShaderPump>);
    let mut ops = RenderOps::default();
    queues.ops.drain_into(&mut ops);
    let outcome = ops.replay(backend);
    serve(state, workers);
    (out, outcome)
}

fn drive_once(
    state: &mut StreamingState,
    backend: &mut RecordingBackend,
    drive: &DriveView<'_>,
) -> ([[f32; 4]; 4], [f32; 3]) {
    let mut q = queues();
    let (out, outcome) = frame_with(state, backend, &mut q, drive, WORKING);
    state.apply_op_failures(&outcome.failures, &mut q.slots);
    out
}

fn drive_until(
    state: &mut StreamingState,
    backend: &mut RecordingBackend,
    workers: Workers,
    done: impl Fn(&StreamingState) -> bool,
) {
    let mut q = queues();
    for _ in 0..MAX_FRAMES {
        let (_, outcome) = frame_with(
            state,
            backend,
            &mut q,
            &view([0.0; 3], IDENTITY4, false),
            workers,
        );
        state.apply_op_failures(&outcome.failures, &mut q.slots);
        if done(state) {
            return;
        }
    }
    panic!("streaming never reached the expected state");
}

// Drive until a replay reports a failure `wanted` accepts, applying every
// earlier replay's failures. Returns that replay's outcome unapplied.
fn replay_until_failure(
    state: &mut StreamingState,
    backend: &mut RecordingBackend,
    q: &mut RenderQueues,
    wanted: impl Fn(&OpFailure) -> bool,
) -> ReplayOutcome {
    for _ in 0..MAX_FRAMES {
        let (_, outcome) = frame_with(
            state,
            backend,
            q,
            &view([0.0; 3], IDENTITY4, false),
            WORKING,
        );
        if outcome.failures.iter().any(&wanted) {
            return outcome;
        }
        state.apply_op_failures(&outcome.failures, &mut q.slots);
    }
    panic!("no matching op failure after {MAX_FRAMES} frames");
}

fn texture(s: &StreamingState) -> &StreamPool<TextureImage> {
    s.pools.texture.as_ref().unwrap()
}

fn mesh(s: &StreamingState) -> &StreamPool<DecodedMesh> {
    s.pools.mesh.as_ref().unwrap()
}

#[test]
fn sample_pressure_engages_and_publishes() {
    let mut s = empty_state();
    // RSS at 92% of a 1000-byte budget: stage 1 engages.
    let p = s.sample_pressure(Some(920), 1000).expect("published");
    assert_eq!(s.pressure_stage, StreamPressureStage::Gate);
    assert!(p.under_pressure);
    assert_eq!(p.rss_bytes, 920);
    assert_eq!(p.budget_bytes, 1000);
}

#[test]
fn sample_pressure_escalates_when_rss_keeps_rising() {
    let mut s = empty_state();
    s.sample_pressure(Some(910), 1000);
    assert_eq!(s.pressure_stage, StreamPressureStage::Gate);
    // Still above engage and climbing: escalate to eviction.
    s.sample_pressure(Some(925), 1000);
    assert_eq!(s.pressure_stage, StreamPressureStage::Evict);
    assert!(s.pressure_factor < 1.0);
}

#[test]
fn sample_pressure_releases_with_hysteresis() {
    let mut s = empty_state();
    s.sample_pressure(Some(970), 1000);
    assert_eq!(s.pressure_stage, StreamPressureStage::Evict);
    // In the hysteresis band (85%): still latched.
    s.sample_pressure(Some(850), 1000);
    assert_eq!(s.pressure_stage, StreamPressureStage::Evict);
    // Below the release mark: the valve releases and restores the baseline.
    let p = s.sample_pressure(Some(700), 1000).expect("published");
    assert_eq!(s.pressure_stage, StreamPressureStage::None);
    assert_eq!(s.pressure_factor, 1.0);
    assert!(!p.under_pressure);
}

#[test]
fn sample_pressure_is_inert_without_rss() {
    let mut s = empty_state();
    s.sample_pressure(Some(970), 1000);
    let stage_before = s.pressure_stage;
    assert!(s.sample_pressure(None, 1000).is_none());
    assert_eq!(s.pressure_stage, stage_before);
}

#[test]
fn the_pressure_sample_runs_on_its_cadence() {
    let mut s = empty_state();
    assert!(s.pressure_due());
    let (_, mut backend) = recording_backend();
    drive_once(&mut s, &mut backend, &view([0.0; 3], IDENTITY4, false));
    assert!(!s.pressure_due());
}

// The translation column is the integer chunk delta scaled by chunk size; the
// basis stays identity.
#[test]
fn chunk_model_matrix_offsets_by_chunk_delta() {
    let m = chunk_model_matrix(ChunkCoord::new(2, -3), ChunkCoord::new(0, 0), 16.0, 10.0);
    assert_eq!(m[3], [32.0, 0.0, -30.0, 1.0]);
    assert_eq!(m[0], [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(m[1], [0.0, 1.0, 0.0, 0.0]);
    assert_eq!(m[2], [0.0, 0.0, 1.0, 0.0]);
}

#[test]
fn chunk_model_matrix_origin_chunk_is_untranslated() {
    let c = ChunkCoord::new(5, 7);
    let m = chunk_model_matrix(c, c, 16.0, 16.0);
    assert_eq!(m[3], [0.0, 0.0, 0.0, 1.0]);
}

// Deep pressure scales every pool's budget off the baseline captured at
// setup, and releasing restores each one exactly.
#[test]
fn deep_pressure_scales_each_pool_budget_and_release_restores_the_baseline() {
    let centers = vec![vec![[0.0; 3]]];
    let mut tex = StreamPool::new(centers.clone(), 4, 8);
    tex.set_byte_budget(Some(4000));
    let mut msh = StreamPool::new(centers, 4, 8);
    msh.set_byte_budget(Some(2000));
    let mut chunks = chunk_stream(0, 0);
    chunks.pool.set_byte_budget(Some(1000));
    let mut state = StreamingState::new(
        StreamingPools {
            texture: Some(tex),
            mesh: Some(msh),
            chunks: Some(chunks),
            ..Default::default()
        },
        2,
    );

    state.sample_pressure(Some(970), 1000);
    assert_eq!(state.pressure_stage, StreamPressureStage::Evict);
    let factor = state.pressure_factor;
    assert!(factor < 1.0);
    assert_eq!(
        texture(&state).byte_budget(),
        Some(pressure::scale_budget(4000, factor))
    );
    assert_eq!(
        mesh(&state).byte_budget(),
        Some(pressure::scale_budget(2000, factor))
    );
    assert_eq!(
        state.pools.chunks.as_ref().unwrap().pool.byte_budget(),
        Some(pressure::scale_budget(1000, factor))
    );

    state.sample_pressure(Some(100), 1000);
    assert_eq!(state.pressure_stage, StreamPressureStage::None);
    assert_eq!(texture(&state).byte_budget(), Some(4000));
    assert_eq!(mesh(&state).byte_budget(), Some(2000));
    assert_eq!(
        state.pools.chunks.as_ref().unwrap().pool.byte_budget(),
        Some(1000)
    );
}

// A pool with no captured baseline runs count-only: the valve must not invent
// a budget for it.
#[test]
fn count_only_pools_gain_no_byte_budget_under_pressure() {
    let mut state = pooled_state(8);
    state.pools.chunks = Some(chunk_stream(0, 0));
    state.sample_pressure(Some(970), 1000);
    assert_eq!(state.pressure_stage, StreamPressureStage::Evict);
    assert_eq!(texture(&state).byte_budget(), None);
    assert_eq!(mesh(&state).byte_budget(), None);
    assert_eq!(
        state.pools.chunks.as_ref().unwrap().pool.byte_budget(),
        None
    );
}

#[test]
fn streaming_stats_reports_every_active_pool() {
    let mut state = pooled_state(8);
    state.pools.chunks = Some(chunk_stream(0, 0));
    state
        .pools
        .texture
        .as_mut()
        .unwrap()
        .set_byte_budget(Some(4000));
    let stats = state.streaming_stats();
    assert_eq!(stats.texture, Some((0, 0, 2)));
    assert_eq!(stats.mesh, Some((0, 0, 2)));
    assert_eq!(stats.chunk, Some((0, 0)));
    assert_eq!(stats.texture_bytes, Some((0, 4000)));
    // A count-only pool reports a zero budget rather than dropping the row.
    assert_eq!(stats.mesh_bytes, Some((0, 0)));
    assert_eq!(stats.chunk_bytes, Some((0, 0)));
}

#[test]
fn streaming_stats_reports_nothing_without_pools() {
    let stats = empty_state().streaming_stats();
    assert!(stats.texture.is_none());
    assert!(stats.mesh.is_none());
    assert!(stats.chunk.is_none());
    assert!(stats.texture_bytes.is_none());
    assert!(stats.mesh_bytes.is_none());
    assert!(stats.chunk_bytes.is_none());
}

// The pools have no Debug of their own, so the state's reports which are
// active rather than dumping them.
#[test]
fn debug_reports_the_active_pools_and_the_frame_clock() {
    let mut state = pooled_state(8);
    state.frame_count = 7;
    state.pressure_stage = StreamPressureStage::Gate;
    let s = format!("{state:?}");
    assert!(s.contains("frame_count: 7"), "{s}");
    assert!(s.contains("texture: true"), "{s}");
    assert!(s.contains("mesh: true"), "{s}");
    assert!(s.contains("chunk: false"), "{s}");
    assert!(s.contains("pressure: Gate"), "{s}");
}

#[test]
fn drive_advances_the_frame_clock() {
    let (_, mut backend) = recording_backend();
    let mut state = empty_state();
    drive_once(&mut state, &mut backend, &view([0.0; 3], IDENTITY4, false));
    drive_once(&mut state, &mut backend, &view([0.0; 3], IDENTITY4, false));
    assert_eq!(state.frame_count(), 2);
}

// Behind an opaque menu the world is not drawn, so no pool requests a load.
#[test]
fn a_hidden_world_requests_no_loads() {
    let (_, mut backend) = recording_backend();
    let mut state = pooled_state(8);
    drive_once(&mut state, &mut backend, &view([0.0; 3], IDENTITY4, true));
    assert_eq!(texture(&state).stats(), (0, 0, 2));
    assert_eq!(mesh(&state).stats(), (0, 0, 2));
}

#[test]
fn a_visible_world_requests_texture_and_mesh_loads() {
    let (_, mut backend) = recording_backend();
    let mut state = pooled_state(8);
    let mut q = queues();
    state.drive(
        &mut q,
        &view([0.0; 3], IDENTITY4, false),
        None::<&mut NoShaderPump>,
    );
    assert!(state.texture_mut().unwrap().mailbox().has_requests());
    assert!(state.mesh_mut().unwrap().mailbox().has_requests());
    // A request moves an item off Unloaded the frame it is planned.
    assert!(texture(&state).stats().2 < 2);
    let _ = q.ops.replay(&mut backend);
}

// A texture upload refused for device memory feeds the valve and rolls the
// slot back so the planner retries it.
#[test]
fn an_out_of_memory_texture_upload_raises_pressure_and_rolls_back() {
    let (recorded, mut backend) = recording_backend();
    recorded.lock().fail_texture_upload =
        Some(RenderError::OutOfDeviceMemory("texture heap".into()));
    let mut state = pooled_state(8);
    let mut q = queues();
    let outcome = replay_until_failure(&mut state, &mut backend, &mut q, |f| {
        matches!(f, OpFailure::TextureUpload { .. })
    });
    assert!(outcome.memory_pressure);
    state.apply_op_failures(&outcome.failures, &mut q.slots);
    assert_eq!(texture(&state).stats().0, 0);
}

#[test]
fn a_non_memory_texture_upload_failure_rolls_back_without_pressure() {
    let (recorded, mut backend) = recording_backend();
    recorded.lock().fail_texture_upload = Some(RenderError::Other("bad image".into()));
    let mut state = pooled_state(8);
    let mut q = queues();
    let outcome = replay_until_failure(&mut state, &mut backend, &mut q, |f| {
        matches!(f, OpFailure::TextureUpload { .. })
    });
    assert!(!outcome.memory_pressure);
    state.apply_op_failures(&outcome.failures, &mut q.slots);
    assert_eq!(texture(&state).stats().0, 0);
}

// A failed eviction leaves the texture on the GPU, so the slot returns to
// resident in the pool and in scene residency, and its bytes count again.
#[test]
fn an_out_of_memory_texture_evict_raises_pressure_and_restores_residency() {
    let (recorded, mut backend) = recording_backend();
    let mut state = pooled_state(8);
    let scene = AssetId(80);
    state.pools.scene_residency = Some(SceneResidency::new(vec![(
        scene,
        vec![(CHANNEL_TEXTURE, 1)],
    )]));
    drive_until(&mut state, &mut backend, WORKING, |s| {
        texture(s).stats().0 == 2
    });
    let resident_bytes = texture(&state).resident_bytes();
    let progress = |s: &StreamingState| s.pools.scene_residency.as_ref().unwrap().progress(scene);
    assert_eq!(progress(&state), Some(1.0));

    recorded.lock().fail_texture_evict = Some(RenderError::OutOfDeviceMemory("evict".into()));
    state.pools.texture.as_mut().unwrap().set_blocked(1, true);
    let mut q = queues();
    let outcome = replay_until_failure(&mut state, &mut backend, &mut q, |f| {
        matches!(f, OpFailure::TextureEvict { .. })
    });
    assert!(outcome.memory_pressure);
    assert_eq!(outcome.failures, vec![OpFailure::TextureEvict { slot: 1 }]);
    assert_eq!(progress(&state), Some(0.0));

    state.apply_op_failures(&outcome.failures, &mut q.slots);
    assert_eq!(texture(&state).stats().0, 2);
    assert_eq!(texture(&state).resident_bytes(), resident_bytes);
    assert_eq!(progress(&state), Some(1.0));
}

// Only the pinned scene's members stream; switching the pin set drains the old
// scene and loads the new one.
#[test]
fn scene_residency_streams_only_the_pinned_scene_and_swaps_on_switch() {
    use crate::render::streaming::scene_residency::SceneLoadState;

    let (_, mut backend) = recording_backend();
    let mut state = pooled_state(8);
    let scene_a = AssetId(70);
    let scene_b = AssetId(71);
    let residency = SceneResidency::new(vec![
        (scene_a, vec![(CHANNEL_TEXTURE, 0), (CHANNEL_MESH, 0)]),
        (scene_b, vec![(CHANNEL_TEXTURE, 1), (CHANNEL_MESH, 1)]),
    ]);
    state.pools.scene_residency = Some(residency);
    // Init pins the start scene: every owned member blocked, then A's freed.
    state.pin_start_scene(scene_a, None::<&mut NoShaderPump>);

    let drive_pinned =
        |state: &mut StreamingState, backend: &mut RecordingBackend, pins: &[AssetId]| {
            drive_once(
                state,
                backend,
                &DriveView {
                    scene_pins: Some(pins),
                    ..view([0.0; 3], IDENTITY4, false)
                },
            );
        };
    let pins_a = [scene_a];
    for _ in 0..MAX_FRAMES {
        drive_pinned(&mut state, &mut backend, &pins_a);
        if state.scene_residency().unwrap().state(scene_a) == Some(SceneLoadState::Resident) {
            break;
        }
    }
    let r = state.scene_residency().unwrap();
    assert_eq!(r.state(scene_a), Some(SceneLoadState::Resident));
    assert_eq!(r.state(scene_b), Some(SceneLoadState::Unloaded));
    assert_eq!(texture(&state).stats().0, 1, "only A's texture is resident");

    let pins_b = [scene_b];
    for _ in 0..MAX_FRAMES {
        drive_pinned(&mut state, &mut backend, &pins_b);
        let r = state.scene_residency().unwrap();
        if r.state(scene_b) == Some(SceneLoadState::Resident)
            && r.state(scene_a) == Some(SceneLoadState::Unloaded)
        {
            break;
        }
    }
    let r = state.scene_residency().unwrap();
    assert_eq!(r.state(scene_b), Some(SceneLoadState::Resident));
    assert_eq!(r.progress(scene_b), Some(1.0));
    assert_eq!(r.state(scene_a), Some(SceneLoadState::Unloaded));
    assert_eq!(texture(&state).stats().0, 1);
    assert_eq!(mesh(&state).stats().0, 1);
}

// Stage 1 of the RAM valve holds residency where it is: no new requests.
#[test]
fn the_gate_stage_freezes_new_loads() {
    let (_, mut backend) = recording_backend();
    let mut state = pooled_state(8);
    state.pools.chunks = Some(chunk_stream(0, 0));
    state.pressure_stage = StreamPressureStage::Gate;
    drive_once(&mut state, &mut backend, &view([0.0; 3], IDENTITY4, false));
    assert_eq!(texture(&state).stats(), (0, 0, 2));
    assert_eq!(mesh(&state).stats(), (0, 0, 2));
    assert_eq!(state.pools.chunks.as_ref().unwrap().pool.stats(), (0, 0));
}

// Stage 2 keeps planning under the reduced budget, so the planner can still
// shed residents.
#[test]
fn the_evict_stage_keeps_planning() {
    let (_, mut backend) = recording_backend();
    let mut state = pooled_state(8);
    state.pressure_stage = StreamPressureStage::Evict;
    drive_once(&mut state, &mut backend, &view([0.0; 3], IDENTITY4, false));
    assert!(texture(&state).stats().2 < 2);
    assert!(mesh(&state).stats().2 < 2);
}

// Finished loads route to the backend: a texture by pool slot, a mesh through
// its stream-id to draw-slot map.
#[test]
fn completed_loads_are_uploaded_to_the_backend() {
    let (recorded, mut backend) = recording_backend();
    let mut state = pooled_state(8);
    drive_until(&mut state, &mut backend, WORKING, |s| {
        texture(s).stats().0 == 2 && mesh(s).stats().0 == 2
    });
    let s = recorded.lock();
    for slot in [0, 1] {
        assert!(s.saw(&Call::UpdateTextureSlot { slot, w: 1, h: 1 }));
    }
    for draw_idx in [DrawIndex(10), DrawIndex(11)] {
        assert!(s.saw(&Call::UploadMesh {
            draw_idx,
            vertices: 3,
            indices: 3,
        }));
    }
}

// A streamed mesh with no draw slot has nowhere to upload to; the drive
// swallows it rather than mis-routing the geometry onto another draw.
#[test]
fn a_streamed_mesh_without_a_draw_slot_uploads_nothing() {
    let (recorded, mut backend) = recording_backend();
    let mut state = pooled_state(8);
    state.pools.mesh_draws.clear();
    drive_until(&mut state, &mut backend, WORKING, |s| {
        mesh(s).stats().0 == 2
    });
    assert!(
        !recorded
            .lock()
            .calls
            .iter()
            .any(|c| matches!(c, Call::UploadMesh { .. }))
    );
}

// Over the resident cap, the pools shed the item the camera has left behind.
#[test]
fn moving_the_camera_evicts_the_now_distant_item_over_the_cap() {
    let (recorded, mut backend) = recording_backend();
    let mut state = pooled_state(1);
    drive_until(&mut state, &mut backend, WORKING, |s| {
        texture(s).stats().0 == 1 && mesh(s).stats().0 == 1
    });
    recorded.lock().calls.clear();
    drive_once(
        &mut state,
        &mut backend,
        &view([100.0, 0.0, 0.0], IDENTITY4, false),
    );
    let s = recorded.lock();
    assert!(s.saw(&Call::EvictTextureSlot(0)), "{:?}", s.calls);
    assert!(s.saw(&Call::EvictMesh(DrawIndex(10))), "{:?}", s.calls);
}

#[test]
fn without_chunk_streaming_the_view_and_camera_stay_absolute() {
    let (_, mut backend) = recording_backend();
    let mut state = empty_state();
    let v = translation_view(-40.0, -5.0, 40.0);
    let cam = [40.0, 5.0, -40.0];
    assert_eq!(
        drive_once(&mut state, &mut backend, &view(cam, v, false)),
        (v, cam)
    );
}

// An unbounded world renders from small coordinates: the view and the camera
// are rebased onto the camera's own chunk.
#[test]
fn chunk_streaming_rebases_the_view_onto_the_camera_chunk() {
    let (_, mut backend) = recording_backend();
    let mut state = empty_state();
    state.pools.chunks = Some(chunk_stream(0, 0));
    // 16-unit chunks: floor(40/16) = 2, floor(-40/16) = -3.
    let cam = [40.0, 5.0, -40.0];
    let v = translation_view(-40.0, -5.0, 40.0);
    let (out_view, out_cam) = drive_once(&mut state, &mut backend, &view(cam, v, false));
    assert_eq!(out_cam, [8.0, 5.0, 8.0]);
    assert_eq!(
        out_view,
        chunk_coord::camera_relative_view(v, cam, [32.0, 0.0, -48.0])
    );
    assert_ne!(out_view, v, "the rebase actually rewrote the view");
    assert_eq!(
        state.pools.chunks.as_ref().unwrap().origin_chunk,
        ChunkCoord::new(2, -3)
    );
}

// Crossing a chunk boundary moves the render origin and re-places every
// resident chunk against it; staying put does not re-push identical models.
#[test]
fn crossing_into_a_new_chunk_rebases_resident_chunk_models() {
    let (recorded, mut backend) = recording_backend();
    let mut state = empty_state();
    let mut cs = chunk_stream(1, 1);
    cs.draws.insert(ChunkCoord::new(0, 0), DrawIndex(3));
    cs.draws.insert(ChunkCoord::new(1, 0), DrawIndex(4));
    state.pools.chunks = Some(cs);
    let failing = Workers { chunks_fail: true };
    let mut q = queues();
    frame_with(
        &mut state,
        &mut backend,
        &mut q,
        &view([20.0, 0.0, 0.0], IDENTITY4, false),
        failing,
    );
    {
        let s = recorded.lock();
        assert!(s.saw(&Call::SetChunkModel(DrawIndex(3))));
        assert!(s.saw(&Call::SetChunkModel(DrawIndex(4))));
    }
    assert_eq!(
        state.pools.chunks.as_ref().unwrap().origin_chunk,
        ChunkCoord::new(1, 0)
    );
    recorded.lock().calls.clear();
    frame_with(
        &mut state,
        &mut backend,
        &mut q,
        &view([21.0, 0.0, 0.0], IDENTITY4, false),
        failing,
    );
    assert!(
        !recorded
            .lock()
            .calls
            .iter()
            .any(|c| matches!(c, Call::SetChunkModel(_)))
    );
}

// A chunk that leaves the view window releases its draw slot on the backend
// and stops being tracked as resident.
#[test]
fn chunks_leaving_the_view_window_are_removed_from_the_backend() {
    let (recorded, mut backend) = recording_backend();
    let mut state = empty_state();
    state.pools.chunks = Some(chunk_stream(0, 0));
    drive_until(&mut state, &mut backend, WORKING, |s| {
        s.pools.chunks.as_ref().unwrap().pool.stats().0 == 1
    });
    let draw = state.pools.chunks.as_ref().unwrap().draws[&ChunkCoord::new(0, 0)];
    let mut q = queues();
    frame_with(
        &mut state,
        &mut backend,
        &mut q,
        &view([800.0, 0.0, 0.0], IDENTITY4, false),
        WORKING,
    );
    assert!(recorded.lock().saw(&Call::RemoveChunkMesh(draw)));
    assert!(
        !state
            .pools
            .chunks
            .as_ref()
            .unwrap()
            .draws
            .contains_key(&ChunkCoord::new(0, 0))
    );
}

// A generated chunk is added to the backend and its draw slot recorded, so a
// later rebase or eviction can find it.
#[test]
fn a_generated_chunk_is_added_and_its_draw_slot_tracked() {
    let (recorded, mut backend) = recording_backend();
    let mut state = empty_state();
    state.pools.chunks = Some(chunk_stream(0, 0));
    drive_until(&mut state, &mut backend, WORKING, |s| {
        s.pools.chunks.as_ref().unwrap().pool.stats().0 == 1
    });
    let cs = state.pools.chunks.as_ref().unwrap();
    assert_eq!(cs.draws.get(&ChunkCoord::new(0, 0)), Some(&DrawIndex(0)));
    assert!(recorded.lock().saw(&Call::AddChunkMesh));
}

// A shader pump hears each bucket its scene pins or unpins, and the residency
// it reports lands on the scene.
#[test]
fn the_shader_pump_follows_the_scene_pins() {
    struct Pump {
        blocked: Vec<(u32, bool)>,
    }
    impl ShaderPump for Pump {
        fn set_blocked(&mut self, bucket: u32, blocked: bool) {
            self.blocked.push((bucket, blocked));
        }
        fn pump(&mut self, _ops: &mut RenderOps, on_resident: &mut dyn FnMut(u32, bool)) {
            if self.blocked.last() == Some(&(1, false)) {
                on_resident(1, true);
            }
        }
    }
    let scene = AssetId(81);
    let mut state = StreamingState::new(
        StreamingPools {
            scene_residency: Some(SceneResidency::new(vec![(
                scene,
                vec![(CHANNEL_SHADER, 1)],
            )])),
            ..Default::default()
        },
        2,
    );
    let mut pump = Pump {
        blocked: Vec::new(),
    };
    let pins = [scene];
    state.drive(
        &mut queues(),
        &DriveView {
            scene_pins: Some(&pins),
            ..view([0.0; 3], IDENTITY4, false)
        },
        Some(&mut pump),
    );
    assert_eq!(pump.blocked, vec![(1, false)]);
    assert!(!state.scene_residency().unwrap().any_loading());
}

// A load that panicked or failed to decode is requested once and never again,
// and the scene owning it still finishes loading; a refused send is asked for
// again.
#[test]
fn a_failed_load_is_requested_once_and_its_scene_still_loads() {
    use crate::render::streaming::scene_residency::SceneLoadState;

    let (_, mut backend) = recording_backend();
    let scene = AssetId(90);
    let centers: Vec<Vec<[f32; 3]>> = (0..4).map(|i| vec![[i as f32, 0.0, 0.0]]).collect();
    let mut state = StreamingState::new(
        StreamingPools {
            texture: Some(StreamPool::new(centers, 4, 8)),
            scene_residency: Some(SceneResidency::new(vec![(
                scene,
                vec![
                    (CHANNEL_TEXTURE, 0),
                    (CHANNEL_TEXTURE, 1),
                    (CHANNEL_TEXTURE, 3),
                ],
            )])),
            ..Default::default()
        },
        2,
    );
    state.pin_start_scene(scene, None::<&mut NoShaderPump>);
    let pins = [scene];
    let mut asked = Vec::new();
    let mut refused_once = false;
    for _ in 0..8 {
        let mut q = queues();
        state.drive(
            &mut q,
            &DriveView {
                scene_pins: Some(&pins),
                ..view([0.0; 3], IDENTITY4, false)
            },
            None::<&mut NoShaderPump>,
        );
        let mut ops = RenderOps::default();
        q.ops.drain_into(&mut ops);
        ops.replay(&mut backend);
        let pool = state.texture_mut().unwrap();
        let ids: Vec<usize> = pool.mailbox().take_requests().collect();
        for id in ids {
            asked.push(id);
            match id {
                0 => pool.load_failed(0),
                1 => pool.mailbox().deliver((1, Err("undecodable".to_string()))),
                2 if !refused_once => {
                    refused_once = true;
                    pool.request_refused(2);
                }
                _ => {
                    let image = TextureImage::rgba8(1, 1, vec![1, 2, 3, 4]);
                    pool.mailbox().deliver((id, Ok(image)));
                }
            }
        }
    }
    assert_eq!(asked.iter().filter(|&&id| id == 0).count(), 1, "{asked:?}");
    assert_eq!(asked.iter().filter(|&&id| id == 1).count(), 1, "{asked:?}");
    assert_eq!(asked.iter().filter(|&&id| id == 2).count(), 2, "{asked:?}");
    let residency = state.scene_residency().unwrap();
    assert_eq!(residency.state(scene), Some(SceneLoadState::Resident));
    assert_eq!(
        texture(&state).stats().0,
        2,
        "the failed slots are not resident"
    );
    assert_eq!(residency.progress(scene), Some(1.0));
}

#[test]
fn pending_sums_every_streaming_pool_and_is_none_without_one() {
    assert_eq!(StreamingStats::default().pending(), None);
    let stats = StreamingStats {
        texture: Some((4, 2, 1)),
        planet: Some((30, 5)),
        ..StreamingStats::default()
    };
    assert_eq!(stats.pending(), Some(7));
}
