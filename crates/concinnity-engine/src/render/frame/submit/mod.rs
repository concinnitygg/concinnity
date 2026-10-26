//! Frame submission: replay one extracted [`RenderSnapshot`] onto the render
//! backend. Deliberately takes no `PipelineContext`, so the draw path cannot
//! read live world state; everything it consumes arrives through the snapshot
//! and everything it produces leaves through [`SubmitOutcome`].

use alloc::string::{String, ToString};
use concinnity_core::math::SaturatingFrom;

use crate::ecs::StepResult;
use crate::profile::RenderStats;
use crate::render::backend::error::RenderError;
use crate::render::backend::{FrameParams, RenderBackend, SkinnedFrame};
use crate::render::frame::ops::ReplayOutcome;
use crate::render::frame::snapshot::{RenderSnapshot, SceneOp};

mod go_timing;
mod pace;
mod policy;

pub use go_timing::GoTiming;
pub use pace::{FramePace, PaceWait, frame_cap};
pub use policy::FramePolicy;

use policy::FrameAction;

/// What one frame's submission produced, applied to the world after the
/// backend is parked again.
pub struct SubmitOutcome {
    /// Whether the run goes on.
    pub result: StepResult,
    /// The backend's stats for a drawn (or skipped) frame; `None` when the
    /// step stopped before reaching the draw.
    pub render_stats: Option<RenderStats>,
    /// What the snapshot's op replay produced (failures to roll back), plus
    /// memory pressure from an upload or the draw.
    pub replay: ReplayOutcome,
    /// The stop was a device loss, with its detail: the queue can never
    /// signal, so no caller may `wait_idle` on the way out.
    pub device_lost: Option<String>,
    /// The frame policy dropped the draw, so nothing this snapshot carried
    /// reached the GPU.
    pub skipped: bool,
}

impl SubmitOutcome {
    fn stop(replay: ReplayOutcome) -> Self {
        Self {
            result: StepResult::Stop,
            render_stats: None,
            replay,
            device_lost: None,
            skipped: false,
        }
    }
}

/// Replay `snap` onto `backend` and draw it, recovering from a failed draw
/// through `policy`. `now_us` is the host's monotonic clock in microseconds,
/// which times the replay and the submission.
pub fn submit(
    policy: &mut FramePolicy,
    snap: &mut RenderSnapshot,
    backend: &mut dyn RenderBackend,
    now_us: fn() -> u64,
) -> SubmitOutcome {
    let started = now_us();
    let since = |start: u64| u32::saturating_from(now_us().saturating_sub(start));
    // Replay the tick's recorded backend effects first, in record order:
    // spawn slot ops, settings appliers, and streaming uploads all landed
    // before the draw when they ran in-place, and still do here.
    let mut replay = snap.ops.replay(backend);
    let replay_us = since(started);

    backend.set_ui_cursor_hidden(snap.ui.cursor_hidden);
    if let Some(on) = snap.ui.menu_mode {
        backend.set_menu_mode(on);
    }
    if let Some(capture) = snap.ui.camera_capture {
        backend.set_camera_capture(capture);
    }

    if backend.window_closed() {
        tracing::info!("window closed");
        backend.wait_idle();
        return SubmitOutcome::stop(replay);
    }

    if !snap.models.is_empty() {
        backend.update_models(&snap.models);
    }

    backend.update_skinned(SkinnedFrame {
        poses: &snap.poses,
        morphs: &snap.morphs,
        models: &snap.skinned_models,
    });

    // Scene fade / visibility effects recorded during extraction, replayed in
    // record order (a mid-fade scene switch fades then swaps visibility).
    for op in &snap.scene_ops {
        match *op {
            SceneOp::SetFade(fade) => backend.set_fade(fade),
            SceneOp::Visibility { draw_idx, visible } => {
                backend.update_visibility(draw_idx, visible);
            }
        }
    }

    // The frame's directional lights, when the sky turned this frame. Carried
    // on the snapshot rather than recorded as an op so a turning sky allocates
    // nothing per frame; the backends early-out on an unchanged set.
    if let Some(set) = snap.frame.directional {
        backend.update_directional_lights(set.as_slice());
    }

    // On Metal, the OS event pump runs inside draw_frame, so update_view is
    // called first so any key/mouse events that arrived since the last tick
    // are in the input state before the sample taken after the draw.
    backend.update_view(snap.frame.view);
    let mut skipped = false;
    match backend.draw_frame(FrameParams {
        elapsed: snap.frame.elapsed,
        water_time: snap.frame.water_time,
        fov_y_radians: snap.frame.fov_y_radians,
        near: snap.frame.near,
        view_distance: snap.frame.view_distance,
        cam_pos: snap.frame.cam_pos,
        text_calls: if snap.hide_ui { &[] } else { &snap.text_calls },
        lines: &snap.lines,
        world_hidden: snap.frame.world_hidden,
        view_mode: snap.frame.view_mode,
        show: snap.frame.show,
        sky_rot: snap.frame.sky_rot,
        history_reset: snap.frame.history_reset.any(),
        rebase: snap.frame.rebase,
        grass_benders: &snap.grass_benders,
        water_stirs: &snap.water_stirs,
        particle_bursts: &snap.particle_bursts,
        atmosphere: snap.frame.atmosphere,
        daylight: &snap.frame.daylight,
        sunlight: snap.frame.sunlight,
        input_sampled_us: snap.frame.input_sampled_us,
        capture: snap.capture.as_deref(),
    }) {
        Ok(()) => {
            policy.frame_succeeded();
            // Logged once the reset reaches the GPU: a dropped frame's is
            // handed to the next frame and logged there.
            if snap.frame.history_reset.any() {
                tracing::info!("temporal history reset: {}", snap.frame.history_reset);
            }
        }
        Err(e) => {
            replay.memory_pressure |= matches!(e, RenderError::OutOfDeviceMemory(_));
            match policy.on_frame_error(&e) {
                FrameAction::SkipFrame => skipped = true,
                FrameAction::Shutdown => {
                    backend.wait_idle();
                    return SubmitOutcome::stop(replay);
                }
                // The device no longer services work; waiting for it to idle
                // would block on a queue that can never signal, so stop
                // without draining.
                FrameAction::ShutdownDeviceLost => {
                    tracing::error!("device lost, stopping: {}", e);
                    return SubmitOutcome {
                        device_lost: Some(e.to_string()),
                        ..SubmitOutcome::stop(replay)
                    };
                }
            }
        }
    }

    let mut render_stats = backend.render_stats();
    render_stats.render_cpu_us = since(started)
        .saturating_sub(render_stats.gpu_wait_us)
        .saturating_add(snap.extract_us);
    render_stats.extract_us = snap.extract_us;
    render_stats.replay_us = replay_us;
    // The backend saves a capture from the frame it rode; a dropped frame
    // presented nothing to save.
    if let Some(path) = snap.capture.take()
        && skipped
    {
        tracing::warn!("frame capture {path} skipped: the frame was dropped");
    }
    SubmitOutcome {
        result: StepResult::Continue,
        render_stats: Some(render_stats),
        replay,
        device_lost: None,
        skipped,
    }
}

#[cfg(test)]
mod tests;
