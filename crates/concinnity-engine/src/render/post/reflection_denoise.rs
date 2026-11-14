//! The ray-traced reflection's denoiser, written once for every backend.
//!
//! The trace draws one direction per texel per frame from each surface's
//! glossy lobe, and marches a texel's sky only on its turn, so a single
//! frame's reflection is noisy and partly unfinished. The denoiser accumulates
//! it over frames at the trace resolution, in two draws ahead of the reflection
//! composite's blur:
//!
//! 1. the surface each texel stands for, kept for the next frame to tell one
//!    surface from another where depth alone cannot, with how many frames it
//!    has been seen for;
//! 2. the accumulation: last frame's, reprojected and kept where last frame saw
//!    the same surface, clamped to this frame's neighborhood and blended with
//!    this frame's reflection by the frames it holds, carrying this frame's
//!    composite weight.
//!
//! The composite then blends the accumulation in place of the frame's own
//! reflection.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::gfx::camera::{camera_to_world, view_ray_scale};
use crate::gfx::render_types::ReflectionDenoiseParams;
use crate::render::backend::error::{RenderError, RenderResult};
use crate::render::render_graph::{
    ClearValue, PixelFormat, TextureDesc, TextureSize, TextureUsage,
};
use crate::transform::{Mat4, mat4_mul};

use super::device::{
    PostBind, PostBlend, PostDraw, PostExtent, PostLoadOp, PostPassDevice, PostSampler,
    PostTargetState, PostTiming,
};
use super::history::HistoryRing;
use super::program::PostProgram;

/// The camera this frame's reflection was traced from, and the one the
/// accumulation was last drawn through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenoiseCamera {
    /// World-to-view matrix, column-major.
    pub view: Mat4,
    /// World-space camera position.
    pub cam_pos: [f32; 3],
    /// Vertical field of view in radians.
    pub fov_y_radians: f32,
    /// Viewport aspect ratio, width over height.
    pub aspect: f32,
    /// The previous frame's unjittered view-projection, carried into this
    /// frame's world.
    pub prev_view_proj: Mat4,
    /// What last frame's accumulation is multiplied by to read at this frame's
    /// pre-exposure.
    pub history_scale: f32,
}

/// The denoiser's constants for `camera`, with the history marked invalid;
/// the denoiser marks it valid once its accumulation holds a frame.
pub fn denoise_params(camera: &DenoiseCamera) -> ReflectionDenoiseParams {
    let inv_view = camera_to_world(camera.view, camera.cam_pos);
    let (tan_half_fov_y, aspect) = view_ray_scale(camera.fov_y_radians, camera.aspect);
    let row = |i: usize| [inv_view[0][i], inv_view[1][i], inv_view[2][i], 0.0];
    ReflectionDenoiseParams {
        reproject: mat4_mul(camera.prev_view_proj, inv_view),
        view_to_world: [row(0), row(1), row(2)],
        tan_half_fov_y,
        aspect,
        history_valid: 0.0,
        history_scale: camera.history_scale,
    }
}

/// What the denoiser reads this frame beyond its own accumulation.
pub struct DenoiseInputs<'t, D: PostPassDevice + ?Sized + 't> {
    /// This frame's reflection: radiance in `.rgb`, composite weight in `.a`.
    pub reflection: D::TextureRef<'t>,
    /// The trace's hits: the reflected surface's distance (`.r`) and how the
    /// texel's look past the authored reach went (`.g`).
    pub hits: D::TextureRef<'t>,
    /// The G-buffer's view normal (`.rgb`) and linear depth (`.a`).
    pub normal_depth: D::TextureRef<'t>,
    /// The G-buffer's surface roughness.
    pub roughness: D::TextureRef<'t>,
    /// The motion vectors.
    pub velocity: D::TextureRef<'t>,
    /// This frame's constants (see [`denoise_params`]).
    pub params: &'t ReflectionDenoiseParams,
}

/// The two draws' pipelines.
pub struct DenoisePipelines<Pipeline> {
    /// The surface kept for the next frame.
    pub surface: Pipeline,
    /// The accumulation.
    pub temporal: Pipeline,
}

/// Build both pipelines.
///
/// # Errors
///
/// Returns an error if a pipeline fails to build.
pub fn build_pipelines<D: PostPassDevice>(
    device: &D,
) -> RenderResult<DenoisePipelines<D::Pipeline>> {
    let format = PixelFormat::Rgba16Float;
    Ok(DenoisePipelines {
        surface: device.create_pipeline(
            PostProgram::ReflectionSurface,
            format,
            PostBlend::Replace,
        )?,
        temporal: device.create_pipeline(
            PostProgram::ReflectionTemporal,
            format,
            PostBlend::Replace,
        )?,
    })
}

const HISTORY_LABEL: &str = "reflection_history";
const SURFACE_LABEL: &str = "reflection_surface";

// A target at the render resolution divided by `scale` on each axis, the
// trace's own size.
fn trace_desc(scale: u32) -> TextureDesc {
    let size = TextureSize::DrawableScaled(1.0 / scale.max(1) as f32);
    TextureDesc {
        width: size,
        height: size,
        depth: 1,
        format: PixelFormat::Rgba16Float,
        sample_count: 1,
        array_layers: 1,
        mip_levels: 1,
        usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
        clear: ClearValue::Color([0.0, 0.0, 0.0, 0.0]),
    }
}

// The accumulation and kept surface, one of each per ring slot.
struct DenoiseTargets<Target> {
    history: Vec<Target>,
    surface: Vec<Target>,
}

/// The accumulation and kept surface, one of each per ring slot, at the trace
/// resolution, and the ring that decides which slot a frame writes.
pub struct ReflectionDenoiser<Target> {
    targets: DenoiseTargets<Target>,
    ring: HistoryRing,
    // Set when this frame's accumulation is encoded, so `advance` only keeps a
    // slot that was written.
    written: AtomicBool,
    trace_scale: u32,
}

impl<Target> ReflectionDenoiser<Target> {
    /// Create the targets for a render resolution of `extent` and a trace
    /// divided by `trace_scale`.
    ///
    /// # Errors
    ///
    /// Returns an error if a target cannot be created.
    pub fn new<D>(device: &D, trace_scale: u32, extent: PostExtent) -> RenderResult<Self>
    where
        D: PostPassDevice<Target = Target>,
    {
        let ring = HistoryRing::ping_pong();
        let trace_scale = trace_scale.max(1);
        let targets = create_targets(device, trace_scale, extent, ring.slots())?;
        Ok(Self {
            targets,
            ring,
            written: AtomicBool::new(false),
            trace_scale,
        })
    }

    /// The per-axis divisor the targets are sized by.
    pub fn trace_scale(&self) -> u32 {
        self.trace_scale
    }

    /// Recreate the targets for a new render resolution and forget the
    /// accumulation. Keeps the old targets if creation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if a target cannot be created.
    pub fn resize<D>(&mut self, device: &D, extent: PostExtent) -> RenderResult<()>
    where
        D: PostPassDevice<Target = Target>,
    {
        self.targets = create_targets(device, self.trace_scale, extent, self.ring.slots())?;
        self.ring.invalidate();
        Ok(())
    }

    /// Step to the next frame: what this frame wrote becomes next frame's
    /// history. A frame that never encoded the accumulation forgets it.
    pub fn advance(&mut self) {
        if self.written.swap(false, Ordering::Relaxed) {
            self.ring.advance();
        } else {
            self.ring.invalidate();
        }
    }

    /// Forget the accumulation, so the next frame starts over.
    pub fn reset_history(&mut self) {
        self.ring.invalidate();
    }

    /// Encode both draws, the first opening the reflection's timing span, and
    /// return the accumulation this frame wrote.
    ///
    /// # Errors
    ///
    /// Returns an error if a ring slot is missing or the device fails to encode
    /// a draw.
    pub fn encode<'t, D>(
        &'t self,
        device: &D,
        rec: &D::Recorder,
        pipelines: &'t DenoisePipelines<D::Pipeline>,
        inputs: &DenoiseInputs<'t, D>,
        timing: PostTiming,
    ) -> RenderResult<D::TextureRef<'t>>
    where
        D: PostPassDevice<Target = Target> + 't,
    {
        let write = self.ring.write();
        let read = self.ring.history();
        let targets = &self.targets;
        let history_out = slot(&targets.history, write, HISTORY_LABEL)?;
        let surface_out = slot(&targets.surface, write, SURFACE_LABEL)?;
        let history_in = slot(&targets.history, read, HISTORY_LABEL)?;
        let surface_in = slot(&targets.surface, read, SURFACE_LABEL)?;
        let params = ReflectionDenoiseParams {
            history_valid: if self.ring.valid() { 1.0 } else { 0.0 },
            ..*inputs.params
        };
        let constants = bytemuck::bytes_of(&params);
        device.encode(
            rec,
            &PostDraw {
                target: device.target_attachment(surface_out),
                state: PostTargetState::Pass,
                load: PostLoadOp::DontCare,
                timing,
                pipeline: &pipelines.surface,
                binds: &[
                    linear::<D>(inputs.normal_depth),
                    linear::<D>(inputs.velocity),
                    linear::<D>(device.target_ref(surface_in)),
                ],
                constants,
                label: "reflection surface",
            },
        )?;
        let binds = [
            linear::<D>(inputs.reflection),
            linear::<D>(inputs.hits),
            linear::<D>(inputs.normal_depth),
            linear::<D>(inputs.velocity),
            linear::<D>(device.target_ref(history_in)),
            linear::<D>(device.target_ref(surface_in)),
            linear::<D>(inputs.roughness),
        ];
        device.encode(
            rec,
            &PostDraw {
                target: device.target_attachment(history_out),
                state: PostTargetState::Pass,
                load: PostLoadOp::DontCare,
                timing: PostTiming::None,
                pipeline: &pipelines.temporal,
                binds: &binds,
                constants,
                label: "reflection temporal",
            },
        )?;
        self.written.store(true, Ordering::Relaxed);
        Ok(device.target_ref(history_out))
    }
}

// The accumulation and surface targets, `slots` of each.
fn create_targets<D: PostPassDevice>(
    device: &D,
    trace_scale: u32,
    extent: PostExtent,
    slots: usize,
) -> RenderResult<DenoiseTargets<D::Target>> {
    let desc = trace_desc(trace_scale);
    let mut targets = DenoiseTargets {
        history: Vec::with_capacity(slots),
        surface: Vec::with_capacity(slots),
    };
    for _ in 0..slots {
        targets
            .history
            .push(device.create_target(HISTORY_LABEL, &desc, extent)?);
        targets
            .surface
            .push(device.create_target(SURFACE_LABEL, &desc, extent)?);
    }
    Ok(targets)
}

// Ring slot `index` of `targets`.
fn slot<'a, Target>(targets: &'a [Target], index: usize, label: &str) -> RenderResult<&'a Target> {
    targets.get(index).ok_or_else(|| {
        RenderError::Other(alloc::format!("{label}: no target in ring slot {index}"))
    })
}

fn linear<'t, D: PostPassDevice + ?Sized + 't>(texture: D::TextureRef<'t>) -> PostBind<'t, D> {
    PostBind {
        texture,
        sampler: PostSampler::LinearClamp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::IDENTITY;

    fn camera(prev_view_proj: Mat4) -> DenoiseCamera {
        DenoiseCamera {
            view: IDENTITY,
            cam_pos: [1.0, 2.0, 3.0],
            fov_y_radians: core::f32::consts::FRAC_PI_2,
            aspect: 2.0,
            prev_view_proj,
            history_scale: 0.5,
        }
    }

    #[test]
    fn a_still_camera_reprojects_through_its_own_view() {
        // Last frame drawn through this frame's own view: a view-space point
        // lands back on itself.
        let mut prev_view = IDENTITY;
        prev_view[3] = [-1.0, -2.0, -3.0, 1.0];
        let p = denoise_params(&camera(prev_view));
        for (col, expected) in p.reproject.iter().zip(IDENTITY.iter()) {
            for (a, b) in col.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn the_params_carry_the_camera_and_start_without_history() {
        let p = denoise_params(&camera(IDENTITY));
        assert_eq!(p.history_valid, 0.0);
        assert_eq!(p.history_scale, 0.5);
        assert!((p.tan_half_fov_y - 1.0).abs() < 1e-6);
        assert_eq!(p.aspect, 2.0);
        // An unrotated camera's view-to-world rows are the identity's.
        assert_eq!(p.view_to_world[0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(p.view_to_world[1], [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(p.view_to_world[2], [0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn the_targets_take_the_trace_extent() {
        use crate::render::post::device::resolve_extent;
        use crate::render::post::rt_reflections::RtReflectionSettings;
        for divisor in [1, 2, 4] {
            let settings = RtReflectionSettings {
                divisor,
                ..RtReflectionSettings::resolve(1.0, 40.0)
            };
            for (w, h) in [(1280, 720), (1921, 1081), (3, 1), (3200, 1800)] {
                let e = resolve_extent(
                    &trace_desc(divisor),
                    PostExtent {
                        width: w,
                        height: h,
                    },
                );
                assert_eq!(
                    (e.width, e.height),
                    settings.trace_extent(w, h),
                    "{w}x{h} at 1/{divisor}"
                );
            }
        }
    }
}
