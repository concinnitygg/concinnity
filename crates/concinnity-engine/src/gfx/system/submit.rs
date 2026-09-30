// GraphicsSystem frame submission: replay one extracted RenderSnapshot onto
// the render backend. Deliberately takes no PipelineContext, so the draw
// path cannot read live world state; everything it consumes arrives through
// the snapshot and everything it produces leaves through SubmitOutcome.

use concinnity_core::ecs::StepResult;
use concinnity_core::profile::RenderStats;
use concinnity_core::render::backend::{FrameParams, RenderBackend};
use concinnity_core::render::error;
use concinnity_core::render::ops::ReplayOutcome;
use concinnity_core::render::snapshot::{RenderSnapshot, SceneOp};

use std::time::Instant;

use super::frame_policy::{FrameAction, FramePolicy};

// What one frame's submission produced, applied to the world by `run_step`
// after the backend is parked again.
pub(crate) struct SubmitOutcome {
    pub result: StepResult,
    // The backend's stats for a drawn (or skipped) frame; `None` when the
    // step stopped before reaching the draw.
    pub(crate) render_stats: Option<RenderStats>,
    // What the snapshot's op replay produced (failures to roll back), plus
    // memory pressure from an upload or the draw.
    pub replay: ReplayOutcome,
    // The stop was a device loss: the queue can never signal, so no caller
    // may wait_idle on the way out.
    pub(crate) device_lost: bool,
}

impl SubmitOutcome {
    fn stop() -> Self {
        Self {
            result: StepResult::Stop,
            render_stats: None,
            replay: ReplayOutcome::default(),
            device_lost: false,
        }
    }
}

pub(crate) fn submit(
    policy: &mut FramePolicy,
    snap: &mut RenderSnapshot,
    backend: &mut dyn RenderBackend,
) -> SubmitOutcome {
    let started = Instant::now();
    // Replay the tick's recorded backend effects first, in record order:
    // spawn slot ops, settings appliers, and streaming uploads all landed
    // before the draw when they ran in-place, and still do here.
    let mut replay = snap.ops.replay(backend);

    backend.set_ui_cursor_hidden(snap.ui.cursor_hidden);
    if let Some(on) = snap.ui.menu_mode {
        backend.set_menu_mode(on);
    }
    if let Some(capture) = snap.ui.camera_capture {
        backend.set_camera_capture(capture);
    }

    if backend.window_closed() {
        tracing::info!("GraphicsSystem: window closed");
        backend.wait_idle();
        return SubmitOutcome {
            replay,
            ..SubmitOutcome::stop()
        };
    }

    if !snap.models.is_empty() {
        backend.update_models(&snap.models);
    }

    for (skinned_index, joints) in snap.poses.iter() {
        backend.update_skinned_pose(skinned_index, joints);
    }
    for (skinned_index, weights) in snap.morphs.iter() {
        backend.update_morph_weights(skinned_index, weights);
    }
    if !snap.skinned_models.is_empty() {
        backend.update_skinned_models(&snap.skinned_models);
    }

    // Scene fade / visibility effects recorded during extraction, replayed in
    // record order (a mid-fade scene switch fades then swaps visibility).
    for op in &snap.scene_ops {
        match *op {
            SceneOp::SetFade(fade) => backend.set_fade(fade),
            SceneOp::Visibility { draw_idx, visible } => {
                backend.update_visibility(draw_idx, visible)
            }
        }
    }

    // The frame's directional lights, when the sky turned this frame. Carried
    // on the snapshot rather than recorded as an op so a turning sky allocates
    // nothing per frame; the backends early-out on an unchanged set.
    if let Some(set) = snap.frame.directional {
        backend.update_directional_lights(set.as_slice());
    }

    // On Metal, pump_ns_events runs inside draw_frame, so update_view is
    // called first so any key/mouse events that arrived since the last tick
    // are in InputState before InputSystem's take_input() (scheduled right
    // after this system) snapshots and clears it.
    backend.update_view(snap.frame.view);
    match backend.draw_frame(FrameParams {
        elapsed: snap.frame.elapsed,
        fov_y_radians: snap.frame.fov_y_radians,
        near: snap.frame.near,
        far: snap.frame.far,
        cam_pos: snap.frame.cam_pos,
        text_calls: &snap.text_calls,
        lines: &snap.lines,
        world_hidden: snap.frame.world_hidden,
        view_mode: snap.frame.view_mode,
        show: snap.frame.show,
        sky_rot: snap.frame.sky_rot,
    }) {
        Ok(()) => policy.frame_succeeded(),
        Err(e) => {
            replay.memory_pressure |= matches!(e, error::RenderError::OutOfDeviceMemory(_));
            match policy.on_frame_error(&e) {
                FrameAction::SkipFrame => {}
                FrameAction::Shutdown => {
                    backend.wait_idle();
                    return SubmitOutcome {
                        replay,
                        ..SubmitOutcome::stop()
                    };
                }
                // The device no longer services work; waiting for it to idle
                // would block on a queue that can never signal, so stop
                // without draining.
                FrameAction::ShutdownDeviceLost => {
                    tracing::error!("GraphicsSystem: device lost, stopping: {}", e);
                    crate::crash::report_device_lost(&e.to_string());
                    return SubmitOutcome {
                        replay,
                        device_lost: true,
                        ..SubmitOutcome::stop()
                    };
                }
            }
        }
    }

    let mut render_stats = backend.render_stats();
    render_stats.render_cpu_us = micros_since(started).saturating_sub(render_stats.gpu_wait_us);
    SubmitOutcome {
        result: StepResult::Continue,
        render_stats: Some(render_stats),
        replay,
        device_lost: false,
    }
}

/// Wall microseconds since `start`, saturating rather than wrapping.
pub(crate) fn micros_since(start: Instant) -> u32 {
    start.elapsed().as_micros().min(u32::MAX as u128) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::mock_backend::{Call, recording_backend};
    use concinnity_core::components::DirectionalLight;
    use concinnity_core::render::lights::DirectionalLightSet;

    fn pushed_lights(directional: Option<DirectionalLightSet>) -> Vec<Call> {
        let (state, mut backend) = recording_backend();
        let mut snap = RenderSnapshot::default();
        snap.frame.directional = directional;
        submit(&mut FramePolicy::default(), &mut snap, &mut backend);
        let calls = state.lock().unwrap().calls.clone();
        calls
            .into_iter()
            .filter(|c| matches!(c, Call::UpdateDirectionalLights(_)))
            .collect()
    }

    fn draw_failing_with(e: error::RenderError) -> SubmitOutcome {
        let (state, mut backend) = recording_backend();
        state.lock().unwrap().fail_draw = Some(e);
        submit(
            &mut FramePolicy::default(),
            &mut RenderSnapshot::default(),
            &mut backend,
        )
    }

    // Submit one frame to a backend whose draw takes `draw` and reports
    // `gpu_wait_us`, returning the frame's render stats and the wall time the
    // whole call took.
    fn submit_timed(draw: std::time::Duration, gpu_wait_us: u32) -> (RenderStats, u32) {
        let (state, mut backend) = recording_backend();
        {
            let mut s = state.lock().unwrap();
            s.draw_duration = draw;
            s.render_stats.gpu_wait_us = gpu_wait_us;
        }
        let started = Instant::now();
        let outcome = submit(
            &mut FramePolicy::default(),
            &mut RenderSnapshot::default(),
            &mut backend,
        );
        let wall = micros_since(started);
        (outcome.render_stats.expect("a drawn frame"), wall)
    }

    #[test]
    fn the_render_cpu_time_covers_the_whole_submission() {
        let (stats, wall) = submit_timed(std::time::Duration::from_millis(2), 0);
        assert!(stats.render_cpu_us >= 2_000, "{} us", stats.render_cpu_us);
        assert!(
            stats.render_cpu_us <= wall,
            "{} > {wall}",
            stats.render_cpu_us
        );
    }

    #[test]
    fn the_render_cpu_time_has_the_same_frames_gpu_wait_taken_out() {
        // A draw that spent 3 ms blocked on the GPU did no CPU work in them.
        let (stats, wall) = submit_timed(std::time::Duration::from_millis(3), 3_000);
        assert_eq!(stats.gpu_wait_us, 3_000);
        assert!(
            stats.render_cpu_us <= wall - 3_000,
            "{} us of {wall}",
            stats.render_cpu_us
        );
    }

    #[test]
    fn a_wait_longer_than_the_submission_reads_as_no_cpu_work_rather_than_wrapping() {
        let (stats, _) = submit_timed(std::time::Duration::ZERO, u32::MAX);
        assert_eq!(stats.render_cpu_us, 0);
    }

    #[test]
    fn a_draw_out_of_device_memory_raises_memory_pressure() {
        let outcome = draw_failing_with(error::RenderError::OutOfDeviceMemory("draw".into()));
        assert!(outcome.replay.memory_pressure);
    }

    #[test]
    fn a_draw_failing_for_another_reason_raises_no_memory_pressure() {
        let outcome = draw_failing_with(error::RenderError::Other("draw".into()));
        assert!(!outcome.replay.memory_pressure);
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
}
