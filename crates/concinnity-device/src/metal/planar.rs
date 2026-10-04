//! Planar reflection for flat reflectors (water surfaces + glass panes). The
//! scene is rendered a second time from the camera reflected across each
//! reflector plane (mirror view + oblique near-plane clip so geometry behind the
//! plane never leaks in) into a dedicated target; the reflective surface then
//! samples that target projectively for a sharp, scene-correct reflection instead
//! of the blurry box-projected probe cube.
//!
//! One mirror render per DISTINCT plane, budgeted by `MAX_PLANAR_PLANES`:
//! near-coplanar reflectors share one render (one wall of windows = one plane),
//! and reflectors past the budget fall back to the probe cube (logged at init,
//! see `metal/init`). The layout is the pure, unit-tested
//! `planar_reflection::PlanarReflectors`.
//!
//! A reflector reads its mirror only at its own screen pixels, so each frame's
//! `PlanarFramePlan` crops every mirror render to the rectangle its reflectors
//! cover and skips a plane whose reflectors are all off screen. Each rendered
//! plane gets a DEDICATED mirror cull against its reflected-camera frustum,
//! narrowed to that rectangle, so geometry visible only in the reflection
//! (behind or beside the main camera) is captured and geometry that cannot reach
//! the rectangle is not drawn. The GPU cull kernel re-runs into that plane's own
//! mirror ICB (`encode_mirror_cull`), which the face render executes.

#![deny(unsafe_op_in_unsafe_fn)]

use super::error::allocation_failed;
use concinnity_core::gfx::frustum::Frustum;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::planar_reflection::{self, PlanarReflectors};
use concinnity_core::transform::mat4_inverse;
use concinnity_core::transform::mat4_mul;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLPixelFormat, MTLTexture, MTLTextureType, MTLTextureUsage};

use super::context::MtlContext;
use super::cull::MirrorCull;
use super::descriptors::TextureDesc;
use super::draw::main::{FacePass, FaceTargets, GpuFrameBuffers, MainPassCamera};
use super::pass_timing::{PassId, PassTimer};

// Clip the reflection a hair toward the kept (camera) side of the plane so
// geometry exactly on the surface is not lost to near-plane precision.
const PLANAR_CLIP_BIAS: f32 = 0.02;

// Texels a mirror's crop is grown by on every side, covering the bilinear
// footprint of the reflector's lookup.
const PLANAR_CROP_MARGIN: u32 = 2;

// The engine capacity ceiling for distinct reflection planes (water + glass): the
// count the mirror-target set + mirror ICB slots below are sized to. Single-sourced
// from `gfx::planar_reflection` so the three backends stay in lockstep by
// construction. The per-frame budget passed to `assign_planar_slots` at init can be
// lower under a quality preset / GPU tier (fewer full-res mirror passes on weaker
// hardware), never higher; reflectors past the active budget fall back to the
// box-projected probe cube.
pub(in crate::metal) const MAX_PLANAR_PLANES: usize = planar_reflection::MAX_PLANAR_PLANES;

// Planar reflection render targets for one plane, at the mirror resolution.
// MSAA color + depth (rendered into, then resolved) plus a single-sample resolve
// the reflective shader samples. The mirror pass reuses the main pipelines, so it
// carries their sample count: at one sample there is no `msaa_color` and the pass
// draws straight into `resolve`.
pub(in crate::metal) struct PlanarReflectionTargets {
    pub(in crate::metal) msaa_color: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    pub(in crate::metal) depth: Retained<ProtocolObject<dyn MTLTexture>>,
    pub(in crate::metal) resolve: Retained<ProtocolObject<dyn MTLTexture>>,
}

// The world's planar reflection layout and one set of targets per mirror plane.
// A water surface or glass pane samples the resolve of the slot it was assigned
// at init. The plane geometry is recomputed (oriented toward the camera) per
// frame, but the planes, the slots and the reflector bounds are fixed at init.
// The targets are reallocated on resize.
pub(in crate::metal) struct PlanarReflectionSet {
    pub(in crate::metal) targets: Vec<PlanarReflectionTargets>,
    pub(in crate::metal) layout: PlanarReflectors,
    width: u32,
    height: u32,
    sample_count: u32,
}

// Build one plane's targets at `width`x`height`. Color + depth match
// the main pipeline's attachment formats + sample count so `encode_main_into_face`
// binds the standard pipelines; the resolve is shader-readable.
fn create_planar_targets(
    device: &ProtocolObject<dyn MTLDevice>,
    width: u32,
    height: u32,
    sample_count: u32,
) -> RenderResult<PlanarReflectionTargets> {
    let multisampled = sample_count > 1;
    let color = if multisampled {
        let desc = TextureDesc {
            kind: MTLTextureType::Type2DMultisample,
            format: MTLPixelFormat::RGBA16Float,
            width: width as usize,
            height: height as usize,
            sample_count: sample_count as usize,
            usage: MTLTextureUsage::RenderTarget,
            ..Default::default()
        }
        .build();
        Some(
            device
                .newTextureWithDescriptor(&desc)
                .ok_or_else(|| allocation_failed("planar MSAA color target"))?,
        )
    } else {
        None
    };
    let depth = {
        let desc = TextureDesc {
            kind: if multisampled {
                MTLTextureType::Type2DMultisample
            } else {
                MTLTextureType::Type2D
            },
            format: MTLPixelFormat::Depth32Float,
            width: width as usize,
            height: height as usize,
            sample_count: sample_count as usize,
            usage: MTLTextureUsage::RenderTarget,
            ..Default::default()
        }
        .build();
        device
            .newTextureWithDescriptor(&desc)
            .ok_or_else(|| allocation_failed("planar depth target"))?
    };
    let resolve = {
        let desc = TextureDesc {
            format: MTLPixelFormat::RGBA16Float,
            width: width as usize,
            height: height as usize,
            usage: MTLTextureUsage(MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::RenderTarget.0),
            ..Default::default()
        }
        .build();
        device
            .newTextureWithDescriptor(&desc)
            .ok_or_else(|| allocation_failed("planar resolve target"))?
    };
    Ok(PlanarReflectionTargets {
        msaa_color: color,
        depth,
        resolve,
    })
}

impl PlanarReflectionSet {
    // One set of targets per plane in `layout`, sized from the `render_w` x
    // `render_h` render resolution by the layout's mirror resolution.
    pub(in crate::metal) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        layout: PlanarReflectors,
        (render_w, render_h): (u32, u32),
        sample_count: u32,
    ) -> RenderResult<Self> {
        let (width, height) = layout.target_size(render_w, render_h);
        let targets = create_targets(device, layout.planes().len(), (width, height), sample_count)?;
        Ok(Self {
            targets,
            layout,
            width,
            height,
            sample_count,
        })
    }

    // Reallocate the targets for a new render resolution. The layout carries over.
    pub(in crate::metal) fn resize(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        (render_w, render_h): (u32, u32),
    ) -> RenderResult<()> {
        let (width, height) = self.layout.target_size(render_w, render_h);
        self.targets = create_targets(
            device,
            self.layout.planes().len(),
            (width, height),
            self.sample_count,
        )?;
        self.width = width;
        self.height = height;
        Ok(())
    }
}

fn create_targets(
    device: &ProtocolObject<dyn MTLDevice>,
    count: usize,
    (width, height): (u32, u32),
    sample_count: u32,
) -> RenderResult<Vec<PlanarReflectionTargets>> {
    (0..count)
        .map(|_| create_planar_targets(device, width, height, sample_count))
        .collect()
}

impl MtlContext {
    // Whether the transparent pass samples planar mirrors at all: the world has a
    // mirror set, and a visible water surface holds a slot or the per-pixel trace
    // is not live. Seeds the graph's `PlanarReflection` node.
    pub(in crate::metal) fn planar_mirrors_needed(&self) -> bool {
        planar_reflection::planar_pass_needed(
            self.planar_reflection
                .as_ref()
                .is_some_and(|s| !s.targets.is_empty()),
            self.water_planar_slot_live(),
            self.rt_transparent_active(),
        )
    }

    // Render the scene reflected across every plane the frame's plan keeps into
    // that plane's target, cropped to the plan's rectangle and reusing this
    // frame's bindless buffers. A plane the plan skips renders nothing, and its
    // reflectors take the probe path (see `collect_*_transparent_draws`). Each
    // plane is oriented toward the camera so the oblique near-plane clip keeps
    // the camera's side (a no-op for water above the surface; flips a glass
    // pane's normal when viewed from its back). The pass's timing span opens on
    // the first mirror cull and closes on the last face render.
    pub(in crate::metal) fn encode_planar_reflections(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        params: &super::graph_exec::GraphFrameParams,
    ) -> RenderResult<()> {
        let Some(set) = self.planar_reflection.as_ref() else {
            return Ok(());
        };
        let crops =
            params
                .planar
                .crops(set.targets.len(), set.width, set.height, PLANAR_CROP_MARGIN);
        let crops = crops.as_slice();

        // Recover the (jittered) projection from this frame's view-projection so
        // the mirror render shares the main camera's projection + jitter, keeping
        // the reflection aligned with the reflective fragment's screen-space sample.
        let proj = mat4_mul(params.vp, mat4_inverse(self.state.view.matrix));
        // The span opens and closes on the face renders, which always encode; a
        // mirror cull is skipped on a frame with no records, and a span opened
        // on one would drop the whole pass from the frame's timings.
        for (i, &(slot, crop)) in crops.iter().enumerate() {
            let plane = set.layout.planes()[slot];
            let oriented = planar_reflection::orient_plane_toward(plane, params.cam_pos);
            let m = planar_reflection::planar_matrices(
                self.state.view.matrix,
                proj,
                params.cam_pos,
                oriented,
                PLANAR_CLIP_BIAS,
            );

            // Re-cull against this plane's reflected-camera frustum, narrowed to
            // the crop, so geometry visible only in the reflection (behind or
            // beside the main camera) is captured and geometry that cannot reach
            // the crop is not. The reflected view-proj carries the oblique
            // near-plane clip, so the frustum also rejects geometry behind the
            // reflector. A frame with no cull records has no mirror to fill, and
            // the face render then draws nothing.
            let mirror_frustum =
                Frustum::from_camera(crop.crop_view_projection(m.view_proj, set.width, set.height));
            let icb_override = match (params.object_buffer, params.draw_args_buffer) {
                (Some(object_buffer), Some(draw_args_buffer)) => {
                    self.encode_mirror_cull(
                        cmd_buf,
                        MirrorCull {
                            object_buffer,
                            draw_args_buffer,
                            frustum: &mirror_frustum,
                            eye: m.eye,
                            slot,
                            timer: PassTimer::None,
                        },
                    )?;
                    self.cull.mirror_slots.get(slot).map(|s| s.icb.as_ref())
                }
                _ => None,
            };

            let targets = &set.targets[slot];
            self.encode_main_into_face(
                cmd_buf,
                FaceTargets {
                    color_msaa: targets.msaa_color.as_deref(),
                    depth: &targets.depth,
                    resolve: &targets.resolve,
                    resolve_slice: 0,
                },
                MainPassCamera {
                    elapsed: params.elapsed,
                    vp: m.view_proj,
                    view: m.view,
                    cam_pos: m.eye,
                },
                GpuFrameBuffers {
                    object_buffer: params.object_buffer,
                    material_params: params.material_params,
                    bindless_tex_args: params.bindless_tex_args,
                    deformed_skinned: params.deformed_skinned,
                    counts: self.draw_record_counts(),
                },
                FacePass {
                    icb_override,
                    scissor: Some(crop),
                    timer: PassTimer::span(PassId::PlanarReflection, i, crops.len()),
                },
            )?;
        }
        Ok(())
    }
}
