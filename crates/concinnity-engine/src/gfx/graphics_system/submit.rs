// GraphicsSystem frame submission: replay one extracted RenderSnapshot onto
// the render backend. Deliberately takes no PipelineContext, so the draw
// path cannot read live world state; everything it consumes arrives through
// the snapshot and everything it produces leaves through SubmitOutcome.

use super::frame_policy::{FrameAction, FramePolicy};
use crate::ecs::StepResult;
use crate::gfx::backend::{FrameParams, RenderBackend};
use crate::gfx::ops::ReplayOutcome;
use crate::gfx::profile::RenderStats;
use crate::gfx::snapshot::{RenderSnapshot, SceneOp};

// What one frame's submission produced, applied to the world by `run_step`
// after the backend is parked again.
pub(crate) struct SubmitOutcome {
    pub result: StepResult,
    // The backend's stats for a drawn (or skipped) frame; `None` when the
    // step stopped before reaching the draw.
    pub render_stats: Option<RenderStats>,
    // A device-memory failure occurred; the caller records it where the
    // streaming valve can observe it.
    pub memory_pressure: bool,
    // What the snapshot's op replay produced (failures to roll back,
    // memory pressure from an upload).
    pub replay: ReplayOutcome,
    // The stop was a device loss: the queue can never signal, so no caller
    // may wait_idle on the way out.
    pub device_lost: bool,
}

impl SubmitOutcome {
    fn stop() -> Self {
        Self {
            result: StepResult::Stop,
            render_stats: None,
            memory_pressure: false,
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
    // Replay the tick's recorded backend effects first, in record order:
    // spawn slot ops, settings appliers, and streaming uploads all landed
    // before the draw when they ran in-place, and still do here.
    let replay = snap.ops.replay(backend);

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

    // On Metal, pump_ns_events runs inside draw_frame, so update_view is
    // called first so any key/mouse events that arrived since the last tick
    // are in InputState before InputSystem's take_input() (scheduled right
    // after this system) snapshots and clears it.
    backend.update_view(snap.frame.view);
    let mut memory_pressure = false;
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
    }) {
        Ok(()) => policy.frame_succeeded(),
        Err(e) => {
            memory_pressure = matches!(e, crate::gfx::error::RenderError::OutOfDeviceMemory(_));
            match policy.on_frame_error(&e) {
                FrameAction::SkipFrame => {}
                FrameAction::Shutdown => {
                    backend.wait_idle();
                    return SubmitOutcome {
                        memory_pressure,
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
                        memory_pressure,
                        replay,
                        device_lost: true,
                        ..SubmitOutcome::stop()
                    };
                }
            }
        }
    }

    SubmitOutcome {
        result: StepResult::Continue,
        render_stats: Some(backend.render_stats()),
        memory_pressure,
        replay,
        device_lost: false,
    }
}
