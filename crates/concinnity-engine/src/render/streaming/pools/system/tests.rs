use super::*;
use crate::bake::texture::TextureImage;
use crate::camera::camera3d::Camera3DArgs;
use crate::render::frame::ops::RenderOps;
use crate::render::slots::render_slots::RenderSlots;
use crate::render::streaming::pools::{LoadPool, StreamPool, StreamingPools, StreamingPressure};
use alloc::vec;

// Answers every texture request with a 1x1 texture on the next delivery, as a
// load thread would a frame later.
#[derive(Default)]
struct Instant {
    answered: Vec<usize>,
}

impl StreamLoaders for Instant {
    fn deliver(&mut self, state: &mut StreamingState) {
        if let Some(pool) = state.texture_mut() {
            for id in self.answered.drain(..) {
                let texture = TextureImage::rgba8(1, 1, vec![1, 2, 3, 4]);
                pool.mailbox().deliver((id, Ok(texture)));
            }
        }
    }

    fn send(&mut self, state: &mut StreamingState) {
        if let Some(pool) = state.texture_mut() {
            while let Some(id) = pool.mailbox().pop_request() {
                self.answered.push(id);
            }
        }
    }
}

fn empty_runtime() -> StreamingRuntime {
    StreamingRuntime::new(
        StreamingState::new(StreamingPools::default(), 2),
        Box::new(Instant::default()),
    )
}

// A two-slot texture pool, one item at the origin and one far out.
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
        Box::new(Instant::default()),
    )
}

fn queues() -> RenderQueues {
    RenderQueues {
        ops: RenderOps::default(),
        slots: RenderSlots::new(0, true, &[]),
    }
}

fn world_with_camera(position: [f32; 3], view_matrix: [[f32; 4]; 4]) -> World {
    let mut w = World::new();
    w.push(Camera3D {
        view_matrix,
        position,
        ..Camera3D::bake(Camera3DArgs::default())
    });
    w
}

fn step(w: &mut World) -> StepResult {
    StreamingSystem::default().step(&mut w.context())
}

fn camera_view(w: &World) -> CameraRelativeView {
    *w.resource::<CameraRelativeView>()
        .expect("camera-relative view published")
}

fn runtime(w: &World) -> &StreamingRuntime {
    w.resource::<StreamingRuntime>().expect("runtime published")
}

fn resident_bytes() -> Option<u64> {
    Some(1 << 40)
}

fn translation(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
    let mut m = IDENTITY;
    m[3] = [x, y, z, 1.0];
    m
}

// Requests go out after the drive and come back through the next frame's
// delivery.
#[test]
fn loads_cross_to_the_loaders_and_back() {
    let mut runtime = texture_runtime();
    for _ in 0..3 {
        runtime.drive(
            &mut queues(),
            DriveView {
                cam_pos: [0.0; 3],
                view: IDENTITY,
                world_hidden: false,
                scene_pins: None,
                planet: None,
            },
        );
    }
    assert_eq!(runtime.state.streaming_stats().texture, Some((2, 0, 0)));
}

#[test]
fn a_step_without_a_runtime_publishes_no_view() {
    let mut w = World::new();
    assert_eq!(step(&mut w), StepResult::Continue);
    assert!(w.resource::<CameraRelativeView>().is_none());
    assert!(StreamingRuntime::stats_in(&w).is_none());
    w.insert_resource(texture_runtime());
    assert_eq!(
        StreamingRuntime::stats_in(&w).and_then(|s| s.texture),
        Some((0, 0, 2))
    );
}

// The runtime is there but its recording surfaces are not: the draw still
// needs a view, so the absolute camera is published.
#[test]
fn a_step_without_queues_still_publishes_the_absolute_view() {
    let view = translation(-1.0, -2.0, -3.0);
    let mut w = world_with_camera([1.0, 2.0, 3.0], view);
    w.insert_resource(empty_runtime());
    step(&mut w);
    assert_eq!(camera_view(&w).cam_pos, [1.0, 2.0, 3.0]);
    assert_eq!(camera_view(&w).view, view);
    assert_eq!(runtime(&w).state.frame_count(), 0, "no frame was driven");
}

#[test]
fn a_step_without_a_camera_publishes_the_identity_view() {
    let mut w = World::new();
    w.insert_resource(empty_runtime());
    w.insert_resource(ActiveRenderQueues::new(queues()));
    step(&mut w);
    assert_eq!(camera_view(&w).view, IDENTITY);
    assert_eq!(camera_view(&w).cam_pos, [0.0; 3]);
}

// The recording surfaces are taken for the step and published again for the
// systems that follow, and the frame clock ticks once per step.
#[test]
fn a_step_drives_the_pools_and_returns_the_queues() {
    let mut w = world_with_camera([0.0; 3], IDENTITY);
    w.insert_resource(texture_runtime());
    w.insert_resource(ActiveRenderQueues::new(queues()));
    step(&mut w);
    assert!(
        w.resource::<ActiveRenderQueues>()
            .is_some_and(|slot| slot.0.is_some())
    );
    assert_eq!(runtime(&w).state.frame_count(), 1);
    let (_, _, unloaded) = runtime(&w).state.streaming_stats().texture.unwrap();
    assert!(unloaded < 2, "the pools were driven");
}

// The overlay's flag is peeked, not taken: streaming pauses behind an opaque
// menu, and the frame submit still finds the overlay frame later this tick.
#[test]
fn an_opaque_overlay_suspends_streaming_without_consuming_the_frame() {
    let mut w = world_with_camera([0.0; 3], IDENTITY);
    w.insert_resource(texture_runtime());
    w.insert_resource(ActiveRenderQueues::new(queues()));
    w.insert_resource(OverlayFrame {
        world_hidden: true,
        ..Default::default()
    });
    step(&mut w);
    assert_eq!(runtime(&w).state.streaming_stats().texture, Some((0, 0, 2)));
    assert!(w.resource::<OverlayFrame>().is_some());
}

// Reading RSS costs a call into the OS, so the valve samples only on its
// throttled cadence: the first frame samples, the second does not.
#[test]
fn ram_pressure_is_sampled_only_on_the_throttled_cadence() {
    let mut w = World::new();
    w.insert_resource(empty_runtime());
    w.insert_resource(ResidentMemory(resident_bytes));
    w.insert_resource(ActiveRenderQueues::new(queues()));
    step(&mut w);
    w.insert_resource(MemoryBudget::compute(None, 1));
    step(&mut w);
    assert!(w.resource::<StreamingPressure>().is_none());
}

#[test]
fn ram_pressure_is_not_sampled_without_a_memory_budget() {
    let mut w = World::new();
    w.insert_resource(empty_runtime());
    w.insert_resource(ResidentMemory(resident_bytes));
    step(&mut w);
    assert!(w.resource::<StreamingPressure>().is_none());
}

#[test]
fn an_rss_sample_over_the_memory_budget_publishes_engaged_pressure() {
    let mut w = World::new();
    w.insert_resource(empty_runtime());
    w.insert_resource(MemoryBudget::compute(None, 1));
    step(&mut w);
    assert!(
        w.resource::<StreamingPressure>().is_none(),
        "no resident-set reading, no valve"
    );

    let mut w = World::new();
    w.insert_resource(empty_runtime());
    w.insert_resource(MemoryBudget::compute(None, 1));
    w.insert_resource(ResidentMemory(resident_bytes));
    step(&mut w);
    let p = w.resource::<StreamingPressure>().expect("published");
    assert!(p.under_pressure);
    assert_eq!(p.budget_bytes, 1024 * 1024);
}

// The drift tracker rides the valve's sample: it reports nothing until the
// session settles, and its two terms account for exactly the RSS movement.
#[test]
fn sample_drift_reports_once_settled_and_splits_the_whole_rss_movement() {
    const RSS: u64 = 2 * 1024 * 1024 * 1024;
    const BUDGET: u64 = 4 * RSS;
    let mut r = empty_runtime();
    assert_eq!(r.sample_drift(None, BUDGET, 0), None, "no RSS, no drift");
    let d = (0..16)
        .find_map(|i| r.sample_drift(Some(RSS), BUDGET, i * 1_000_000))
        .expect("a steady run settles the baseline");
    assert_eq!(d.verdict, DriftVerdict::Settled);
    assert_eq!(d.heap_growth_bytes + d.outside_heap_growth_bytes, 0);
}
