//! The frame every backend executes, planned once in core: the graph that
//! runs, the camera it runs through, and every CPU-built block a pass binds.
//!
//! A backend reports what it has built ([`ResidentFeatures`]) and what the
//! world holds this frame ([`FrameContent`]); [`FramePlanner::plan`] turns them
//! into a [`FramePlan`], and [`FramePlanner::finish`] closes the frame once the
//! backend has executed it.

mod camera;
mod drive;
mod shadow;
mod temporal;

pub use camera::{CameraInputs, FrameCamera, Jitter};
use concinnity_core::math::SaturatingFrom;
pub use drive::{FrameDevice, camera_inputs, drive_frame};
pub use shadow::ShadowPlan;
pub use temporal::TemporalState;

use crate::gfx::render_types::{
    ClusterCamera, ClusterParams, FogFroxelParams, FogParams, RtParams, SsaoParams, SsgiParams,
    SsrParams,
};
use crate::gfx::view_modes::ViewMode;
use crate::planet::Rebase;
use crate::render::backend::error::{RenderError, RenderResult};
use crate::render::backend::init::ShadowCadence;
use crate::render::environment::atmosphere::AtmosphereView;
use crate::render::environment::volumetric_fog::FogSettings;
use crate::render::light::lights::clustering_active;
use crate::render::light::state::LightState;
use crate::render::post::reflection_path::ReflectionPath;
use crate::render::post::rt_reflections::{RtParamsInputs, RtReflectionSettings};
use crate::render::post::ssao::settings::SsaoSettings;
use crate::render::post::ssgi::settings::{SsgiFrame, SsgiSettings};
use crate::render::post::ssr::settings::{SsrParamsInputs, SsrSettings};
use crate::render::render_graph::{
    self, CompiledGraph, FeatureGates, FrameContent, FrameFacts, FrameGraphInputs, GraphCache,
    PlannedFacts, ResidentFeatures,
};
use crate::render::scene_state::SceneView;
use crate::render::shadow::schedule::CascadeCamera;
use crate::render::temporal::model_history::HistoryMode;
use crate::render::temporal::reactive_mask::ReactiveMaskPlan;
use crate::render::temporal::view_history::ViewFrame;
use crate::render::uniforms::probe::ProbeUniforms;
use crate::render::uniforms::{GBufferView, ViewUniforms};

/// What the surfaces a frame shades sample, as the backend has bound them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SurfaceInputs {
    /// Mip levels of the bound environment prefilter cube; zero for none.
    pub prefilter_mip_count: f32,
    /// 1.0 when the bound irradiance cube holds the environment's diffuse light.
    pub irradiance: f32,
    /// The surfaces take their indirect bounce from the screen-space trace.
    pub screen_bounce: bool,
}

/// The screen-space effects' live settings, each `None` while the effect is
/// not built.
#[derive(Clone, Copy, Debug, Default)]
pub struct EffectInputs {
    /// Ambient occlusion.
    pub ssao: Option<SsaoSettings>,
    /// Screen-space reflections.
    pub ssr: Option<SsrSettings>,
    /// Screen-space global illumination, with where its accumulation stands.
    pub ssgi: Option<(SsgiSettings, SsgiFrame)>,
    /// Ray-traced reflections.
    pub rt: Option<RtReflectionSettings>,
    /// The factor that rescales last frame's accumulations to this frame's
    /// pre-exposure.
    pub history_scale: f32,
}

/// The screen-space effects' blocks, built once for every pass that binds
/// them.
#[derive(Clone, Copy, Debug, Default)]
pub struct EffectBlocks {
    /// Ambient occlusion.
    pub ssao: Option<SsaoParams>,
    /// Screen-space reflections.
    pub ssr: Option<SsrParams>,
    /// Screen-space global illumination.
    pub ssgi: Option<SsgiParams>,
    /// Ray-traced reflections, while an acceleration structure is live.
    pub rt: Option<RtParams>,
}

/// What the backend hands the planner for one frame.
pub struct PlanInputs<'a> {
    /// The camera.
    pub camera: CameraInputs,
    /// Seconds since the world started.
    pub elapsed: f32,
    /// The render target's size in pixels.
    pub render_size: [u32; 2],
    /// What the backend has built.
    pub resident: ResidentFeatures,
    /// What the world holds this frame.
    pub content: FrameContent,
    /// The running upscaler's own jitter at [`FramePlanner::jitter_phase`],
    /// when it prescribes one.
    pub upscaler_jitter: Option<[f32; 2]>,
    /// The live light block.
    pub lights: &'a LightState,
    /// The baked reflection probes the clusters bin.
    pub probes: &'a [ProbeUniforms],
    /// The live fog settings.
    pub fog: Option<FogSettings>,
    /// The scale the HDR chain is written at this frame.
    pub pre_exposure: f32,
    /// What the surfaces sample.
    pub surface: SurfaceInputs,
    /// The screen-space effects' settings.
    pub effects: EffectInputs,
}

/// The fog's two blocks, laid along the un-jittered view.
#[derive(Clone, Copy, Debug)]
pub struct FogBlocks {
    /// The volume integration and the fragment's reconstruction.
    pub params: FogParams,
    /// The froxel grid.
    pub froxel: FogFroxelParams,
}

/// One frame, as every backend executes it.
pub struct FramePlan {
    /// The compiled graph.
    pub graph: CompiledGraph,
    /// The inputs `graph` was built from.
    pub inputs: FrameGraphInputs,
    /// The camera.
    pub camera: FrameCamera,
    /// The camera's world position.
    pub cam_pos: [f32; 3],
    /// Seconds since the world started.
    pub elapsed: f32,
    /// A pass reprojects through the pre-pass's motion this frame.
    pub motion_read: bool,
    /// The main view block.
    pub view: ViewUniforms,
    /// The main camera's cluster grid.
    pub cluster: ClusterParams,
    /// Local lights or probes are binned this frame.
    pub clustered: bool,
    /// The fog's blocks, when fog contributes.
    pub fog: Option<FogBlocks>,
    /// The G-buffer pre-pass's view block.
    pub gbuffer_view: GBufferView,
    /// The view the atmosphere draws the sky and haze from.
    pub atmosphere: AtmosphereView,
    /// The cull records to pack and how they treat the model history, or
    /// `None` on a frame the cull does not run.
    pub records: Option<HistoryMode>,
    /// How the reactive mask is written and read.
    pub reactive: ReactiveMaskPlan,
    /// The screen-space effects' blocks.
    pub effects: EffectBlocks,
    pyramid: bool,
}

/// The policy state a frame is planned from, carried from frame to frame.
pub struct FramePlanner {
    /// The runtime-switched passes the settings and the last apply leave on.
    pub gates: FeatureGates,
    /// The shadow schedules.
    pub shadow: ShadowPlan,
    temporal: TemporalState,
    graph: GraphCache,
}

impl Default for FramePlanner {
    /// Shadows scheduled as the default graphics settings schedule them.
    fn default() -> Self {
        Self::new(crate::settings::ResolvedGraphics::default().shadow_cadence())
    }
}

impl FramePlanner {
    /// A planner with nothing gated on, scheduling shadows by `cadence`.
    pub fn new(cadence: ShadowCadence) -> Self {
        Self {
            gates: FeatureGates::default(),
            shadow: ShadowPlan::new(cadence),
            temporal: TemporalState::default(),
            graph: GraphCache::default(),
        }
    }

    /// The phase this frame's jitter is taken at.
    pub fn jitter_phase(&self) -> u32 {
        self.temporal.jitter_phase()
    }

    /// The temporal state.
    pub fn temporal(&self) -> &TemporalState {
        &self.temporal
    }

    /// Drop the occlusion history, for a pyramid that no longer matches the view.
    pub fn invalidate_occlusion(&mut self) {
        self.temporal.invalidate_occlusion();
    }

    /// Drop every history for a frame that does not continue the last.
    pub fn reset_history(&mut self) {
        self.temporal.reset();
    }

    /// Carry every history into the frame the world moved to.
    pub fn rebase(&mut self, rebase: &Rebase) {
        self.temporal.rebase(rebase);
    }

    /// Plan the frame `view` shows from `inputs`.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame graph does not compile.
    pub fn plan(&mut self, view: &SceneView, inputs: &PlanInputs<'_>) -> RenderResult<FramePlan> {
        let r = &inputs.resident;
        let c = &inputs.content;
        let [width, height] = inputs.render_size;
        let cam = &inputs.camera;
        let sun = inputs.lights.sun();

        let jitter = match (r.upscaler.is_some(), inputs.upscaler_jitter) {
            (true, Some(offset)) => Jitter::Upscaler(offset),
            (true, None) => Jitter::Halton,
            (false, _) if r.taa => Jitter::Halton,
            (false, _) => Jitter::None,
        };
        let camera = FrameCamera::new(cam, inputs.render_size, jitter, self.jitter_phase());

        self.shadow.refresh(
            r.shadow.then_some(r.shadow_map_size),
            sun.direction,
            CascadeCamera {
                view: cam.view,
                position: cam.position,
                fov_y_rad: cam.fov_y_radians,
                aspect: camera.aspect,
                near: cam.near,
                view_distance: cam.view_distance,
            },
            c.shadowed_spot_count,
        );

        let num_local_lights = inputs.lights.uniforms().num_local_lights;
        let clustered = clustering_active(num_local_lights, inputs.probes.len());
        let facts = FrameFacts::derive(
            r,
            c,
            &PlannedFacts {
                cascade_mask: self.shadow.cascade_mask,
                clustered,
            },
        );
        let seeded = render_graph::frame_graph_inputs(&self.gates, &facts);
        let graph_inputs = render_graph::apply_view(&seeded, view.mode, view.show);
        let graph = self
            .graph
            .take(&graph_inputs)
            .map_err(|e| RenderError::Other(alloc::format!("frame graph: {e}")))?;

        let motion_read = r.taa || r.upscaler.is_some() || c.ssgi_motion;
        let cur = ViewFrame {
            vp: camera.cur_vp,
            elapsed: inputs.elapsed,
            cam_pos: cam.position,
        };
        let viewport = [width as f32, height as f32];
        let reflections = r.reflection_composite
            && ReflectionPath::new(self.gates.ssr, self.gates.rt_pass, c.bvh_live).resolves();
        let records = (graph_inputs.bindless_cull_enabled && !c.world_hidden).then_some(
            if motion_read && r.model_history {
                HistoryMode::Track
            } else {
                HistoryMode::Stale
            },
        );
        Ok(FramePlan {
            view: ViewUniforms {
                vp: camera.vp,
                view: cam.view,
                elapsed: inputs.elapsed,
                reflections_enabled: f32::from(u8::from(reflections)),
                cam_pos: cam.position,
                prefilter_mip_count: inputs.surface.prefilter_mip_count,
                shade_mode: f32::from(u8::from(view.mode == ViewMode::Unlit)),
                ambient_occlusion: 1.0,
                sky_rot: view.sky_rot,
                irradiance: inputs.surface.irradiance,
                pre_exposure: inputs.pre_exposure,
                screen_bounce: f32::from(u8::from(inputs.surface.screen_bounce)),
                stand_ins: 0.0,
                daylight: view.daylight,
                reflection: 0.0,
                _pad: [0.0; 3],
                tile_morph: view.tile_morph_view(),
            },
            cluster: ClusterParams::for_camera(
                &ClusterCamera {
                    view: cam.view,
                    proj: camera.proj,
                    position: cam.position,
                    near: cam.near,
                    range: inputs.lights.cluster_reach().range(
                        cam.position,
                        cam.near,
                        inputs.probes,
                        cam.view_distance,
                    ),
                    width,
                    height,
                },
                num_local_lights,
                u32::saturating_from(inputs.probes.len()),
            ),
            clustered,
            fog: inputs
                .fog
                .filter(FogSettings::contributes)
                .map(|fog| FogBlocks {
                    params: fog.params(
                        camera.inv_cur_vp,
                        cam.position,
                        sun.direction,
                        sun.color,
                        viewport,
                        inputs.pre_exposure,
                    ),
                    froxel: fog.froxel_params(cam.view, cam.near),
                }),
            gbuffer_view: GBufferView::new(
                camera.vp,
                cam.view,
                cur,
                self.temporal.prev_view_or(cur),
                self.temporal.prev_model_carry(),
                motion_read,
            ),
            atmosphere: AtmosphereView {
                inv_vp: camera.inv_cur_vp,
                depth_inv_vp: camera.inv_vp,
                cam_pos: cam.position,
                viewport,
                sky_rot: view.sky_rot,
                pre_exposure: inputs.pre_exposure,
            },
            records,
            reactive: ReactiveMaskPlan::of(&graph_inputs),
            effects: effect_blocks(inputs, &camera, view.sky_rot, c.bvh_live),
            graph,
            inputs: graph_inputs,
            camera,
            cam_pos: cam.position,
            elapsed: inputs.elapsed,
            motion_read,
            pyramid: r.hiz,
        })
    }

    /// Close an executed frame: keep its graph for the next one and advance
    /// the temporal state past it.
    pub fn finish(&mut self, plan: FramePlan) {
        let cur = ViewFrame {
            vp: plan.camera.cur_vp,
            elapsed: plan.elapsed,
            cam_pos: plan.cam_pos,
        };
        self.temporal.advance(cur, plan.motion_read, plan.pyramid);
        self.graph.put(plan.inputs, plan.graph);
    }
}

// The effects' blocks for `camera`, the ray-traced one only over a live BVH.
fn effect_blocks(
    inputs: &PlanInputs<'_>,
    camera: &FrameCamera,
    sky_rot: [[f32; 4]; 3],
    bvh_live: bool,
) -> EffectBlocks {
    let cam = &inputs.camera;
    let fx = &inputs.effects;
    let sun = inputs.lights.sun();
    let prefilter_mip_count = inputs.surface.prefilter_mip_count;
    EffectBlocks {
        ssao: fx
            .ssao
            .map(|s| s.params(cam.fov_y_radians, camera.aspect, inputs.render_size[1])),
        ssr: fx.ssr.map(|s| {
            s.params(SsrParamsInputs {
                fov_y_radians: cam.fov_y_radians,
                aspect: camera.aspect,
                view: cam.view,
                cam_pos: cam.position,
                prefilter_mip_count,
                sky_rot,
                pre_exposure: inputs.pre_exposure,
            })
        }),
        ssgi: fx
            .ssgi
            .map(|(s, frame)| s.params(cam.fov_y_radians, camera.aspect, frame, fx.history_scale)),
        rt: fx.rt.filter(|_| bvh_live).map(|s| {
            s.params(RtParamsInputs {
                fov_y_radians: cam.fov_y_radians,
                aspect: camera.aspect,
                view: cam.view,
                cam_pos: cam.position,
                sun_dir: sun.direction,
                sun_color: sun.color,
                prefilter_mip_count,
                sky_rot,
                pre_exposure: inputs.pre_exposure,
            })
        }),
    }
}

#[cfg(test)]
mod tests;
