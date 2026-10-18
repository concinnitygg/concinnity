use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use super::*;
use crate::components::DirectionalLight;
use crate::render::backend::recording::{Call, recording_backend};
use crate::render::error::DeviceLostReason;
use crate::render::lights::DirectionalLightSet;

fn no_clock() -> u64 {
    0
}

fn pushed_lights(directional: Option<DirectionalLightSet>) -> Vec<Call> {
    let (state, mut backend) = recording_backend();
    let mut snap = RenderSnapshot::default();
    snap.frame.directional = directional;
    submit(
        &mut FramePolicy::default(),
        &mut snap,
        &mut backend,
        no_clock,
    );
    let calls = state.lock().calls.clone();
    calls
        .into_iter()
        .filter(|c| matches!(c, Call::UpdateDirectionalLights(_)))
        .collect()
}

fn draw_failing_with(e: RenderError) -> SubmitOutcome {
    let (state, mut backend) = recording_backend();
    state.lock().fail_draw = Some(e);
    submit(
        &mut FramePolicy::default(),
        &mut RenderSnapshot::default(),
        &mut backend,
        no_clock,
    )
}

// The one clock this module's timing test advances: each replayed op and the
// draw move it on by a known span.
static FAKE_US: AtomicU64 = AtomicU64::new(0);

fn fake_clock() -> u64 {
    FAKE_US.load(Ordering::Relaxed)
}

// The submission's time spans come from the host clock: the replay is its
// own span, and the frame's GPU wait is taken out of the whole.
#[test]
fn the_time_spans_come_from_the_clock_with_the_gpu_wait_taken_out() {
    let (state, mut backend) = recording_backend();
    {
        let mut s = state.lock();
        s.on_draw = Some(Box::new(|| {
            FAKE_US.fetch_add(5_000, Ordering::Relaxed);
        }));
        s.render_stats.gpu_wait_us = 3_000;
    }
    let mut snap = RenderSnapshot::default();
    snap.ops.record(|_| {
        FAKE_US.fetch_add(2_000, Ordering::Relaxed);
    });
    let stats = submit(
        &mut FramePolicy::default(),
        &mut snap,
        &mut backend,
        fake_clock,
    )
    .render_stats
    .expect("a drawn frame");
    assert_eq!(stats.replay_us, 2_000);
    assert_eq!(stats.render_cpu_us, 7_000 - 3_000);
}

#[test]
fn a_wait_longer_than_the_submission_reads_as_no_cpu_work_rather_than_wrapping() {
    let (state, mut backend) = recording_backend();
    state.lock().render_stats.gpu_wait_us = u32::MAX;
    let stats = submit(
        &mut FramePolicy::default(),
        &mut RenderSnapshot::default(),
        &mut backend,
        no_clock,
    )
    .render_stats
    .expect("a drawn frame");
    assert_eq!(stats.render_cpu_us, 0);
}

#[test]
fn a_draw_out_of_device_memory_raises_memory_pressure() {
    let outcome = draw_failing_with(RenderError::OutOfDeviceMemory("draw".into()));
    assert!(outcome.replay.memory_pressure);
}

#[test]
fn a_draw_failing_for_another_reason_raises_no_memory_pressure() {
    let outcome = draw_failing_with(RenderError::Other("draw".into()));
    assert!(!outcome.replay.memory_pressure);
}

// A lost device stops without draining and names what was lost.
#[test]
fn a_lost_device_stops_with_its_detail() {
    let outcome = draw_failing_with(RenderError::DeviceLost {
        reason: DeviceLostReason::Removed,
        detail: "gone".to_string(),
    });
    assert_eq!(outcome.result, StepResult::Stop);
    assert!(
        outcome
            .device_lost
            .as_deref()
            .is_some_and(|d| d.contains("gone"))
    );
}

// The capture each drawn frame asked the backend for, after a draw that
// fails with `fail` when one is given.
fn captured(
    capture: Option<&str>,
    fail: Option<RenderError>,
) -> Vec<Option<alloc::string::String>> {
    let (state, mut backend) = recording_backend();
    state.lock().fail_draw = fail;
    let mut snap = RenderSnapshot {
        capture: capture.map(str::to_string),
        ..Default::default()
    };
    submit(
        &mut FramePolicy::default(),
        &mut snap,
        &mut backend,
        no_clock,
    );
    assert!(snap.capture.is_none(), "a capture is taken once");
    let calls = state.lock().calls.clone();
    assert!(
        !calls.iter().any(|c| matches!(c, Call::Screenshot(_))),
        "a frame capture never takes the blocking screenshot"
    );
    calls
        .into_iter()
        .filter_map(|c| match c {
            Call::DrawFrame { capture, .. } => Some(capture),
            _ => None,
        })
        .collect()
}

#[test]
fn a_frame_carrying_a_capture_asks_its_own_draw_for_it() {
    assert_eq!(
        captured(Some("shot.png"), None),
        [Some("shot.png".to_string())]
    );
}

#[test]
fn a_frame_without_a_capture_asks_for_none() {
    assert_eq!(captured(None, None), [None]);
}

#[test]
fn a_dropped_frame_does_not_carry_its_capture_to_the_next() {
    let (state, mut backend) = recording_backend();
    state.lock().fail_draw = Some(RenderError::Other("draw".into()));
    let mut snap = RenderSnapshot {
        capture: Some("shot.png".to_string()),
        ..Default::default()
    };
    let mut policy = FramePolicy::default();
    let outcome = submit(&mut policy, &mut snap, &mut backend, no_clock);
    assert!(outcome.skipped);
    state.lock().fail_draw = None;
    submit(&mut policy, &mut snap, &mut backend, no_clock);
    let last = state.lock().last_draw_frame();
    assert!(matches!(last, Some(Call::DrawFrame { capture: None, .. })));
}

#[test]
fn a_frame_carrying_no_lights_leaves_the_backend_set_alone() {
    assert!(pushed_lights(None).is_empty());
}

#[test]
fn a_frame_carrying_lights_installs_them_before_the_draw() {
    let sun = DirectionalLight {
        direction: [0.0, 1.0, 0.0],
        color: [1.0, 0.5, 0.25],
        intensity: 2.0,
    };
    let pushed = pushed_lights(Some(DirectionalLightSet::collect(core::iter::once(sun))));
    assert_eq!(
        pushed,
        vec![Call::UpdateDirectionalLights(vec![(
            sun.direction,
            sun.color,
            sun.intensity
        )])]
    );
}
