use super::*;
use crate::gfx::streaming::texture::{self, PayloadSource};
use concinnity_core::ecs::World;
use concinnity_core::render::backend::recording::{Call, recording_backend};
use concinnity_core::render::ops::RenderOps;
use concinnity_core::render::queues::RenderQueues;
use concinnity_core::render::render_slots::RenderSlots;
use concinnity_core::render::scene_residency::{CHANNEL_SHADER, SceneResidency};
use concinnity_core::render::stream_pools::{StreamPool, StreamingPools, StreamingPressure};
use std::sync::Arc;

// Upper bound on frames a test drives while a load thread works. The threads
// decode on their own, so a test that waits on one yields rather than sleeps;
// the bound turns a regression into a failure instead of a hang.
const MAX_DRIVE_SPINS: usize = 100_000;

fn translation_view(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [x, y, z, 1.0],
    ]
}

fn empty_runtime() -> StreamingRuntime {
    StreamingRuntime::new(
        StreamingState::new(StreamingPools::default(), 2),
        StreamWorkers::default(),
        None,
    )
}

// A source yielding a fixed 1x1 texture for any id.
struct ConstTexture;
impl PayloadSource for ConstTexture {
    fn fetch(&self, _id: usize) -> Result<TextureImage, String> {
        Ok(TextureImage::rgba8(1, 1, vec![1, 2, 3, 4]))
    }
}

// A two-slot texture pool on a real load thread.
fn texture_runtime() -> StreamingRuntime {
    let centers = vec![vec![[0.0, 0.0, 0.0]], vec![[100.0, 0.0, 0.0]]];
    StreamingRuntime::new(
        StreamingState::new(
            StreamingPools {
                texture: Some(StreamPool::new(centers, 4, 8)),
                ..Default::default()
            },
            2,
        ),
        StreamWorkers {
            texture: Some(texture::spawn_worker(Arc::new(ConstTexture)).expect("thread")),
            ..Default::default()
        },
        None,
    )
}

fn queues() -> RenderQueues {
    RenderQueues {
        ops: RenderOps::default(),
        slots: RenderSlots::new(0, true, &[]),
    }
}

fn view(pins: Option<&[AssetId]>) -> DriveView<'_> {
    DriveView {
        cam_pos: [0.0; 3],
        view: IDENTITY4,
        world_hidden: false,
        scene_pins: pins,
        planet: None,
    }
}

// Requests cross to the load thread after the drive and come back through the
// next frames' deliveries, uploading through the recorded ops.
#[test]
fn the_texture_pool_loads_through_its_thread() {
    let (recorded, mut backend) = recording_backend();
    let mut runtime = texture_runtime();
    for _ in 0..MAX_DRIVE_SPINS {
        let mut q = queues();
        runtime.drive(&mut q, view(None));
        q.ops.replay(&mut backend);
        if runtime.state.streaming_stats().texture == Some((2, 0, 0)) {
            break;
        }
        std::thread::yield_now();
    }
    assert_eq!(runtime.state.streaming_stats().texture, Some((2, 0, 0)));
    let s = recorded.lock();
    for slot in [0, 1] {
        assert!(s.saw(&Call::UpdateTextureSlot { slot, w: 1, h: 1 }));
    }
}

// A pinned scene owning a shader bucket keeps loading until the bucket's
// off-thread build comes back and its install is recorded.
#[test]
fn a_pinned_scene_loads_once_its_shader_bucket_is_installed() {
    use crate::gfx::streaming::shader::{DeferredBucket, ShaderPayloadSource};
    use concinnity_core::components::ShaderPrograms;

    let cooked = ShaderPrograms {
        name: "cooked".into(),
        ..Default::default()
    };
    let scene = AssetId(81);
    let mut runtime = StreamingRuntime::new(
        StreamingState::new(
            StreamingPools {
                scene_residency: Some(SceneResidency::new(vec![(
                    scene,
                    vec![(CHANNEL_SHADER, 1)],
                )])),
                ..Default::default()
            },
            2,
        ),
        StreamWorkers::default(),
        Some(ShaderWarmup::new(
            vec![DeferredBucket {
                bucket: 1,
                source: ShaderPayloadSource::Bytes(cooked.encode().unwrap()),
            }],
            None,
        )),
    );
    let (recorded, mut backend) = recording_backend();
    let pins = [scene];
    let mut drive = |runtime: &mut StreamingRuntime| {
        let mut q = queues();
        runtime.drive(&mut q, view(Some(&pins)));
        q.ops.replay(&mut backend);
    };
    drive(&mut runtime);
    let loading = |r: &StreamingRuntime| r.state.scene_residency().unwrap().any_loading();
    assert!(
        loading(&runtime),
        "the dispatch alone does not load the scene"
    );
    for _ in 0..MAX_DRIVE_SPINS {
        if !loading(&runtime) {
            break;
        }
        drive(&mut runtime);
        std::thread::yield_now();
    }
    assert!(!loading(&runtime));
    assert_eq!(
        recorded.lock().calls,
        [Call::InstallWorldShader {
            bucket: 1,
            name: "cooked".into(),
            prepared: None,
        }]
    );
}

// The drift tracker rides the valve's sample: it reports nothing until the
// session settles, and the two terms it then reports account for exactly the
// RSS movement between them.
#[test]
fn sample_drift_reports_once_settled_and_splits_the_whole_rss_movement() {
    const RSS: u64 = 2 * 1024 * 1024 * 1024;
    const BUDGET: u64 = 4 * RSS;
    let mut r = empty_runtime();
    assert_eq!(r.sample_drift(None, BUDGET), None, "no RSS, no drift");
    let d = (0..16)
        .find_map(|_| r.sample_drift(Some(RSS), BUDGET))
        .expect("a steady run settles the baseline");
    assert_eq!(d.verdict, crate::app::mem_drift::DriftVerdict::Settled);
    // RSS did not move, so whatever the heap did, the outside term is its
    // exact complement.
    assert_eq!(d.heap_growth_bytes + d.outside_heap_growth_bytes, 0);
}

// Drift is reported, never acted on: the valve is decided by its own sample.
#[test]
fn sample_drift_never_moves_the_valve() {
    const RSS: u64 = 2 * 1024 * 1024 * 1024;
    let mut r = empty_runtime();
    for _ in 0..4 {
        r.sample_drift(Some(RSS), 4 * RSS);
    }
    assert!(!format!("{:?}", r.state).contains("Gate"));
    assert!(!format!("{:?}", r.state).contains("Evict"));
}

// A world with a camera the step will read the absolute view + position from.
fn world_with_camera(position: [f32; 3], view_matrix: [[f32; 4]; 4]) -> World {
    let mut w = World::new();
    w.push(Camera3D {
        position,
        view_matrix,
        ..Camera3D::bake(Default::default())
    });
    w
}

// Publish the op queue + slot allocator the step takes and reparks.
fn park_render_queues(w: &mut World) {
    w.insert_resource(ActiveRenderQueues::new(queues()));
}

fn step_streaming(w: &mut World) -> StepResult {
    StreamingSystem::new().step(&mut w.context())
}

fn camera_view(w: &World) -> CameraRelativeView {
    *w.resource::<CameraRelativeView>()
        .expect("camera-relative view published")
}

fn parked(w: &World) -> &StreamingRuntime {
    w.resource::<StreamingRuntime>()
        .expect("runtime parked again")
}

#[test]
fn a_step_without_parked_state_publishes_no_view() {
    let mut w = World::new();
    assert_eq!(step_streaming(&mut w), StepResult::Continue);
    assert!(w.resource::<CameraRelativeView>().is_none());
}

// Init succeeded but the backend is gone: the draw still needs a view, so the
// absolute camera is published and the state stays parked.
#[test]
fn a_step_without_a_backend_still_publishes_the_absolute_view() {
    let view = translation_view(-1.0, -2.0, -3.0);
    let mut w = world_with_camera([1.0, 2.0, 3.0], view);
    w.insert_resource(empty_runtime());
    assert_eq!(step_streaming(&mut w), StepResult::Continue);
    assert_eq!(camera_view(&w).cam_pos, [1.0, 2.0, 3.0]);
    assert_eq!(camera_view(&w).view, view);
    assert_eq!(parked(&w).state.frame_count(), 0, "no frame was driven");
}

#[test]
fn a_step_without_a_camera_publishes_the_identity_view() {
    let mut w = World::new();
    w.insert_resource(empty_runtime());
    park_render_queues(&mut w);
    step_streaming(&mut w);
    assert_eq!(camera_view(&w).view, IDENTITY4);
    assert_eq!(camera_view(&w).cam_pos, [0.0; 3]);
}

// The op queue + slot allocator are taken for the step and parked again for
// the systems that follow, and the frame clock ticks once per step.
#[test]
fn a_step_drives_the_pools_and_reparks_the_queues() {
    let mut w = world_with_camera([0.0; 3], IDENTITY4);
    w.insert_resource(texture_runtime());
    park_render_queues(&mut w);
    step_streaming(&mut w);
    assert!(
        w.resource::<ActiveRenderQueues>()
            .is_some_and(|slot| slot.0.is_some())
    );
    assert_eq!(parked(&w).state.frame_count(), 1);
    let (_, _, unloaded) = parked(&w).state.streaming_stats().texture.unwrap();
    assert!(unloaded < 2, "the pools were driven");
}

// The overlay's flag is peeked, not taken: streaming pauses behind an opaque
// menu, and GraphicsSystem still finds the frame later this tick.
#[test]
fn an_opaque_overlay_suspends_streaming_without_consuming_the_frame() {
    let mut w = world_with_camera([0.0; 3], IDENTITY4);
    w.insert_resource(texture_runtime());
    park_render_queues(&mut w);
    w.insert_resource(OverlayFrame {
        world_hidden: true,
        ..Default::default()
    });
    step_streaming(&mut w);
    assert_eq!(parked(&w).state.streaming_stats().texture, Some((0, 0, 2)));
    assert!(w.resource::<OverlayFrame>().is_some());
}

// RSS is a syscall, so the valve only samples on its throttled cadence.
#[test]
fn ram_pressure_is_sampled_only_on_the_throttled_cadence() {
    let mut w = world_with_camera([0.0; 3], IDENTITY4);
    w.insert_resource(empty_runtime());
    park_render_queues(&mut w);
    // A 1 MiB ceiling is under any real process RSS, so a sample that ran
    // would certainly engage the valve.
    step_streaming(&mut w);
    w.insert_resource(concinnity_core::budget::MemoryBudget::compute(None, 1));
    step_streaming(&mut w);
    assert!(w.resource::<StreamingPressure>().is_none());
}

// No published ceiling means no valve: nothing is reported.
#[test]
fn ram_pressure_is_not_sampled_without_a_memory_budget() {
    let mut w = World::new();
    w.insert_resource(empty_runtime());
    step_streaming(&mut w);
    assert!(w.resource::<StreamingPressure>().is_none());
}

#[test]
fn an_rss_sample_over_the_memory_budget_publishes_engaged_pressure() {
    let mut w = World::new();
    w.insert_resource(empty_runtime());
    w.insert_resource(concinnity_core::budget::MemoryBudget::compute(None, 1));
    step_streaming(&mut w);
    // The valve is inert (and publishes nothing) where RSS cannot be read.
    match crate::app::sysmem::process_resident_bytes() {
        Some(_) => {
            let p = w.resource::<StreamingPressure>().expect("published");
            assert!(p.under_pressure);
            assert_eq!(p.budget_bytes, 1024 * 1024);
        }
        None => assert!(w.resource::<StreamingPressure>().is_none()),
    }
}
