//! The order every backend's frame runs in: the view snapped, the history
//! reset or carried, the frame acquired, planned, executed and closed.

use concinnity_core::math::SaturatingFrom;

use crate::profile::FrameSpans;
use crate::render::backend::FrameParams;
use crate::render::backend::error::RenderResult;
use crate::render::probe::radiance::with_sky_cube_scale;
use crate::render::scene_state::SceneState;

use super::{CameraInputs, FramePlan};

/// A backend's half of the frame, in the order [`drive_frame`] calls it.
pub trait FrameDevice {
    /// What [`Self::begin`] acquired: the frame's slot, its target and what
    /// it records into.
    type Frame;

    /// The scene the frame shows.
    fn frame_scene(&mut self) -> &mut SceneState;

    /// Whether the world draws an atmosphere, whose sky cube is stored scaled.
    fn draws_atmosphere(&self) -> bool;

    /// Work owed ahead of the frame: rebuilds requested since the last one,
    /// and the world's content carried into a rebased frame.
    ///
    /// # Errors
    ///
    /// Returns an error if a rebuild or the rebased content fails to apply.
    fn prepare(&mut self, params: &FrameParams<'_>) -> RenderResult<()>;

    /// Drop the temporal history this backend keeps itself: its
    /// accumulation rings and its upscaler's.
    fn reset_history(&mut self);

    /// Wait for the frame's slot and acquire its target, or `None` to skip a
    /// frame nothing can be drawn into.
    ///
    /// # Errors
    ///
    /// Returns an error if waiting for the slot or acquiring the target fails.
    fn begin(&mut self, params: &FrameParams<'_>) -> RenderResult<Option<Self::Frame>>;

    /// Plan the frame `frame` draws.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame cannot be planned.
    fn plan(&mut self, frame: &Self::Frame, params: &FrameParams<'_>) -> RenderResult<FramePlan>;

    /// Record, submit and present `plan` into `frame`.
    ///
    /// # Errors
    ///
    /// Returns an error if recording, submitting, or presenting the frame
    /// fails.
    fn execute(
        &mut self,
        frame: Self::Frame,
        plan: &FramePlan,
        params: &FrameParams<'_>,
    ) -> RenderResult<()>;

    /// Close an executed frame: what this backend's accumulations wrote is
    /// next frame's history.
    fn finish(&mut self, plan: &FramePlan);
}

/// The camera `params` describes, viewed through `scene`.
pub fn camera_inputs(scene: &SceneState, params: &FrameParams<'_>) -> CameraInputs {
    CameraInputs {
        view: scene.view.matrix,
        position: params.cam_pos,
        fov_y_radians: params.fov_y_radians,
        near: params.near,
        view_distance: params.view_distance,
    }
}

/// Run one frame on `device`, timing its preparation and plan into the
/// scene's [`SceneState::spans`] by `now_us`, a monotonic clock in
/// microseconds.
///
/// # Errors
///
/// Returns an error if any of the device's steps fails.
pub fn drive_frame<D: FrameDevice>(
    device: &mut D,
    params: FrameParams<'_>,
    now_us: fn() -> u64,
) -> RenderResult<()> {
    let since = |start: u64| u32::saturating_from(now_us().saturating_sub(start));
    let atmosphere = device.draws_atmosphere();
    let view = &mut device.frame_scene().view;
    view.mode = params.view_mode;
    view.show = params.show;
    view.near = params.near;
    view.view_distance = params.view_distance;
    view.sky_rot = with_sky_cube_scale(params.sky_rot, atmosphere);
    view.sunlight = params.sunlight;
    view.daylight = *params.daylight;
    device.frame_scene().spans.set(FrameSpans::default());

    let prepared = now_us();
    device.prepare(&params)?;
    let prepare_us = since(prepared);
    let spans = &device.frame_scene().spans;
    spans.set(FrameSpans {
        prepare_us,
        ..spans.get()
    });
    if params.history_reset {
        device.frame_scene().planner.reset_history();
        device.reset_history();
    } else if let Some(rebase) = &params.rebase {
        let scene = device.frame_scene();
        scene.planner.rebase(rebase);
        scene.model_history.get_mut().forget();
    }

    let Some(frame) = device.begin(&params)? else {
        return Ok(());
    };
    let planned = now_us();
    let plan = device.plan(&frame, &params)?;
    let plan_us = since(planned);
    let spans = &device.frame_scene().spans;
    spans.set(FrameSpans {
        plan_us,
        ..spans.get()
    });
    device.execute(frame, &plan, &params)?;
    device.finish(&plan);
    device.frame_scene().planner.finish(plan);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::render_types::LightUniforms;
    use crate::gfx::view_modes::{ShowFlags, ViewMode};
    use crate::render::frame::plan::{EffectInputs, PlanInputs, SurfaceInputs};
    use crate::render::light::state::LightState;
    use crate::render::render_graph::{FrameContent, ResidentFeatures};
    use crate::render::scene_state::DrawList;
    use alloc::vec::Vec;
    use concinnity_core::math::narrow_f32;

    #[derive(Debug, PartialEq)]
    enum Step {
        Prepare,
        Reset,
        Begin,
        Plan,
        Execute,
        Finish,
    }

    struct Recorder {
        scene: SceneState,
        lights: LightState,
        steps: Vec<Step>,
        acquire: bool,
    }

    impl Recorder {
        fn new(acquire: bool) -> Self {
            Self {
                scene: SceneState::new(DrawList::unreserved(Vec::new(), 0), [0.0; 4]),
                lights: LightState::new(LightUniforms::DEFAULT, &[], 1),
                steps: Vec::new(),
                acquire,
            }
        }
    }

    impl FrameDevice for Recorder {
        type Frame = ();

        fn frame_scene(&mut self) -> &mut SceneState {
            &mut self.scene
        }

        fn draws_atmosphere(&self) -> bool {
            false
        }

        fn prepare(&mut self, _: &FrameParams<'_>) -> RenderResult<()> {
            self.steps.push(Step::Prepare);
            Ok(())
        }

        fn reset_history(&mut self) {
            self.steps.push(Step::Reset);
        }

        fn begin(&mut self, _: &FrameParams<'_>) -> RenderResult<Option<()>> {
            self.steps.push(Step::Begin);
            Ok(self.acquire.then_some(()))
        }

        fn plan(&mut self, (): &(), params: &FrameParams<'_>) -> RenderResult<FramePlan> {
            self.steps.push(Step::Plan);
            let inputs = PlanInputs {
                camera: camera_inputs(&self.scene, params),
                elapsed: narrow_f32(params.elapsed),
                render_size: [8, 8],
                resident: ResidentFeatures {
                    hdr_width: 8,
                    hdr_height: 8,
                    hdr_sample_count: 1,
                    ..ResidentFeatures::default()
                },
                content: FrameContent::default(),
                upscaler_jitter: None,
                lights: &self.lights,
                probes: &[],
                fog: None,
                pre_exposure: 1.0,
                surface: SurfaceInputs::default(),
                effects: EffectInputs::default(),
            };
            self.scene.planner.plan(&self.scene.view, &inputs)
        }

        fn execute(&mut self, (): (), _: &FramePlan, _: &FrameParams<'_>) -> RenderResult<()> {
            self.steps.push(Step::Execute);
            Ok(())
        }

        fn finish(&mut self, _: &FramePlan) {
            self.steps.push(Step::Finish);
        }
    }

    fn params(history_reset: bool) -> FrameParams<'static> {
        FrameParams {
            elapsed: 1.0,
            water_time: 1.0,
            fov_y_radians: 1.0,
            near: 0.25,
            view_distance: Some(90.0),
            cam_pos: [0.0; 3],
            text_calls: &[],
            lines: &[],
            world_hidden: false,
            view_mode: ViewMode::Unlit,
            show: ShowFlags::default(),
            sky_rot: crate::sky::SkyOrientation::IDENTITY_ROWS,
            history_reset,
            rebase: None,
            grass_benders: &[],
            water_stirs: &[],
            particle_bursts: &[],
            atmosphere: None,
            daylight: &crate::render::uniforms::DaylightBlock::OFF,
            sunlight: 3.0,
            prop_probe: None,
            input_sampled_us: None,
            capture: None,
        }
    }

    // A clock that moves on a millisecond each time it is read. Only one test
    // reads it, so the spans it measures are exact.
    fn ticking_clock() -> u64 {
        static NOW: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
        NOW.fetch_add(1_000, core::sync::atomic::Ordering::Relaxed)
    }

    #[test]
    fn a_frame_runs_every_step_in_order_and_snaps_the_view() {
        let mut device = Recorder::new(true);
        drive_frame(&mut device, params(true), ticking_clock).unwrap();
        assert_eq!(
            device.steps,
            [
                Step::Prepare,
                Step::Reset,
                Step::Begin,
                Step::Plan,
                Step::Execute,
                Step::Finish
            ]
        );
        assert_eq!(device.scene.view.mode, ViewMode::Unlit);
        assert_eq!(device.scene.view.near, 0.25);
        assert_eq!(device.scene.view.view_distance, Some(90.0));
        assert_eq!(device.scene.view.sunlight, 3.0);
        assert_eq!(device.scene.spans.get().prepare_us, 1_000);
        assert_eq!(device.scene.spans.get().plan_us, 1_000);
    }

    #[test]
    fn a_frame_without_a_target_stops_after_begin() {
        let mut device = Recorder::new(false);
        drive_frame(&mut device, params(false), || 0).unwrap();
        assert_eq!(device.steps, [Step::Prepare, Step::Begin]);
    }
}
