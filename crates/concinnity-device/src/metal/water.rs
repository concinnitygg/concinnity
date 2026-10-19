//! Water is a producer for the engine's transparent pass (`PassId::Transparent`:
//! after `SsrResolve`, before `TaaResolve` / Upscale). It contributes one
//! `TransparentDraw` per `WaterSurface`; the shared `encode_transparent`
//! encoder owns the render pass, the scene snapshot, and back-to-front sorting.
//!
//! For each surface the vertex shader displaces a flat tessellated quad by a sum
//! of Gerstner waves; the fragment shader composites:
//!   * Refraction: sample the pre-transparent scene snapshot at a
//!     normal-perturbed screen UV.
//!   * Tint: shallow to deep color mix by water-column thickness derived from
//!     the difference between the main depth and the water surface depth.
//!   * Foam: a soft mask where the seabed is just below the surface.
//!   * Reflection: the sharp planar reflection where the surface has one, else
//!     the box-projected reflection-probe set, else the IBL prefilter cubemap,
//!     else a hand-tuned sky gradient.
//!   * Fresnel: Schlick-power mix of refraction-tinted vs. reflected color.
//!
//! Output blends with `SRC_ALPHA` / `ONE_MINUS_SRC_ALPHA` into `scene_pre_taa`.
//!
//! The shaders are the shared `shaders/water.hlsl`, the single source all three
//! backends compile; the pipeline state matches the glass panes exactly, because
//! the same transparent encoder feeds both.
//!
//! Refraction samples `targets.hdr.transparent_scene_copy` (the snapshot the
//! transparent encoder blits from the current scene-pre-taa before drawing) so
//! water renders correctly whether or not SSR produced a distinct scene texture
//! (with SSR off, scene-pre-taa aliases `hdr_resolve`, and sampling it directly
//! would be reading the attachment being written).

#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::gfx::mesh_payload::Vertex;
use concinnity_core::math::SaturatingFrom;
use concinnity_core::planet::Rebase;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::planar_reflection::PlanarFramePlan;
use concinnity_core::render::transparent;
use concinnity_core::render::uniforms::{TransparentView, WaterParams};
use concinnity_core::render::water::WaterBody;
use concinnity_core::render::water::frame::WaterMotion;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLRenderPipelineState, MTLResourceOptions};

use super::builtin_shaders;
use super::context::MtlContext;
use super::error::allocation_failed;
use super::glass::build_transparent_pipeline_stages;
use super::transparent::{TransparentDraw, bytes_of};

// Per-surface GPU state: a static tessellated grid VB + IB, and the motion its
// params take each frame.
pub(in crate::metal) struct WaterSurfaceRecord {
    pub(in crate::metal) vertex_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(in crate::metal) index_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(in crate::metal) index_count: u32,
    pub(in crate::metal) motion: WaterMotion,
    pub(in crate::metal) visible: bool,
    // World-space center, for the back-to-front camera-distance sort.
    pub(in crate::metal) center: [f32; 3],
    // Planar reflection slot this surface samples (index into the
    // `PlanarReflectionSet`). `None` when the world has no planar set or this
    // surface's plane overflowed the budget; the shader then keeps the probe/sky
    // path. Assigned at init by `assign_planar_slots`.
    pub(in crate::metal) planar_slot: Option<usize>,
}

// Build a [`WaterSurfaceRecord`] for one resolved `WaterSurface`: its grid
// (`WaterBody::mesh`) uploads once; per-frame uniforms (the view, and the
// params advanced to the frame's time and camera) come in through the encoder.
pub(in crate::metal) fn build_water_surface_record(
    device: &ProtocolObject<dyn MTLDevice>,
    body: &WaterBody,
) -> RenderResult<WaterSurfaceRecord> {
    let surface = &body.surface;
    // A grid the core geometry refuses (too many vertices for a u16 index) is a
    // world-authoring limit, not a device failure.
    let (packed, idxs) = body.mesh().map_err(RenderError::Other)?;
    let vb_bytes = packed.len() * size_of::<Vertex>();
    let ib_bytes = idxs.len() * size_of::<u16>();

    // SAFETY: the pointer and length describe the live `packed` allocation, and Metal copies those
    // bytes into the new buffer before the call returns.
    let vb = unsafe {
        let ptr = std::ptr::NonNull::new(packed.as_ptr() as *mut _).ok_or_else(|| {
            RenderError::Other("water vertex buffer: source pointer is null".into())
        })?;
        device
            .newBufferWithBytes_length_options(ptr, vb_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| allocation_failed("the water vertex buffer"))?
    };
    // SAFETY: the pointer and length describe the live `idxs` allocation, and Metal copies those
    // bytes into the new buffer before the call returns.
    let ib = unsafe {
        let ptr = std::ptr::NonNull::new(idxs.as_ptr() as *mut _).ok_or_else(|| {
            RenderError::Other("water index buffer: source pointer is null".into())
        })?;
        device
            .newBufferWithBytes_length_options(ptr, ib_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| allocation_failed("the water index buffer"))?
    };

    Ok(WaterSurfaceRecord {
        vertex_buffer: vb,
        index_buffer: ib,
        index_count: u32::saturating_from(idxs.len()),
        motion: WaterMotion::new(body, false),
        visible: surface.visible,
        center: body.sort_center(),
        // Patched after `assign_planar_slots` runs over all reflectors in init.
        planar_slot: None,
    })
}

// The pipelines a planet sea draws with, the same three as a rectangle's
// from the sea's own programs.
pub(in crate::metal) struct SeaPipelines {
    pub(in crate::metal) pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub(in crate::metal) pipeline_rt: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub(in crate::metal) pipeline_rt_textured:
        Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
}

impl SeaPipelines {
    // The sea's pipelines, the traced pair only where the device can `trace`.
    pub(in crate::metal) fn build(
        device: &ProtocolObject<dyn MTLDevice>,
        hot_reload: bool,
        trace: bool,
    ) -> RenderResult<Self> {
        let vert = &builtin_shaders::WATER_SEA_VERT;
        let build = |frag| build_water_pipeline_with(device, hot_reload, vert, frag);
        let (pipeline_rt, pipeline_rt_textured) = if trace {
            (
                Some(build(&builtin_shaders::WATER_SEA_FRAG_RT)?),
                Some(build(&builtin_shaders::WATER_SEA_FRAG_RT_TEXTURED)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            pipeline: build(&builtin_shaders::WATER_SEA_FRAG)?,
            pipeline_rt,
            pipeline_rt_textured,
        })
    }
}

impl MtlContext {
    // True when a visible water surface holds a planar slot, so the mirror
    // re-render has a consumer this frame even while the trace is live. Water
    // takes the mirror over its own trace (see `water.hlsl`), so this is what
    // the planar gate reads; glass is deliberately not counted.
    pub(in crate::metal) fn water_planar_slot_live(&self) -> bool {
        self.water
            .surfaces
            .iter()
            .any(|s| s.visible && s.planar_slot.is_some())
    }

    // Carry the water surfaces into the frame the world moved to.
    pub(in crate::metal) fn rebase_water(&mut self, rebase: &Rebase) {
        for surface in &mut self.water.surfaces {
            surface.motion.rebase(rebase);
        }
    }

    // True when a visible surface samples a simulated ocean, so `WaterSim` has
    // a reader this frame.
    pub(in crate::metal) fn water_sim_live(&self) -> bool {
        self.water
            .ocean
            .as_ref()
            .is_some_and(super::water_sim::OceanSim::simulates)
            && self
                .water
                .surfaces
                .iter()
                .any(|s| s.visible && s.motion.body().ocean.is_some())
    }

    // Contribute one [`TransparentDraw`] per visible water surface to the
    // transparent pass. The shared `encode_transparent` encoder owns the render
    // pass, the scene snapshot, back-to-front sorting, and the shared reflection
    // bindings (prefilter cube + probe cubes + probe set + cube sampler). Each
    // draw binds the snapshot (refraction source) at texture(0) and the resolved
    // main depth at texture(1). Sampling the snapshot rather than `hdr_resolve` is
    // what lets water render with SSR off.
    pub(in crate::metal) fn collect_water_transparent_draws(
        &self,
        view: &TransparentView,
        // Seconds since the world started, which the waves and ripples are
        // advanced to.
        time: f64,
        bindless: bool,
        mirrors: &PlanarFramePlan,
        out: &mut Vec<TransparentDraw>,
    ) {
        // Pipeline selection (matched by `encode_transparent`'s binding):
        //   RT on + bindless world  -> textured RT trace (bindless albedo)
        //   RT on                   -> flat RT trace (per-object tint)
        //   RT off                  -> box-projected probe cube / sky prefilter
        // `rt.accel` live means RT is on; `bindless` means the texture pool
        // exists. Falls back through to the probe pipeline. Either RT pipeline
        // still takes the planar mirror over its own trace where the surface has
        // a slot, so this picks the fragment, not the reflection source.
        let rt_on = self.rt.accel.is_some();
        let pick = |base, rt, rt_textured| match (rt_on && bindless, rt_textured, rt_on, rt) {
            (true, Some(p), _, _) | (_, _, true, Some(p)) => Some(p),
            _ => base,
        };
        let Some(rectangle) = pick(
            self.water.pipeline.as_ref(),
            self.water.pipeline_rt.as_ref(),
            self.water.pipeline_rt_textured.as_ref(),
        ) else {
            return;
        };
        let sea = self.water.sea.as_ref().and_then(|s| {
            pick(
                Some(&s.pipeline),
                s.pipeline_rt.as_ref(),
                s.pipeline_rt_textured.as_ref(),
            )
        });
        let cam = view.camera_pos;
        let planar_set = self.planar_reflection.as_ref();
        for surface in &self.water.surfaces {
            if !surface.visible {
                continue;
            }
            let pipeline = match (surface.motion.body().sea.is_some(), sea) {
                (false, _) => rectangle,
                (true, Some(p)) => p,
                (true, None) => continue,
            };
            // Everything but the motion and the planar flag below is
            // asset-side-static.
            let mut params = surface.motion.params_at(time, [cam[0], cam[1], cam[2]]);
            let mut fragment_textures = vec![
                // The refraction snapshot (texture 0) + resolved main depth
                // (texture 1). The IBL prefilter cube (texture 2), the probe cube
                // argument buffer, cube sampler (sampler 1) and probe set are
                // bound globally by `encode_transparent` (shared with glass).
                (0, self.targets.hdr.transparent_scene_copy.clone()),
                (1, self.targets.hdr.depth_resolve.clone()),
            ];
            // Select the sharp planar reflection when this frame rendered the
            // surface's slot; bind that slot's resolve at the planar slot. Both
            // fragments honor the flag, so this outranks the trace as well.
            // Otherwise the shader keeps the trace / probe / sky path.
            if mirrors.samples_mirror(surface.planar_slot)
                && let Some(targets) = surface
                    .planar_slot
                    .and_then(|s| planar_set.and_then(|set| set.targets.get(s)))
            {
                params.planar = WaterParams::planar_lane(surface.motion.params().roughness, true);
                fragment_textures.push((
                    super::transparent::GLASS_PLANAR_TEXTURE_INDEX,
                    targets.resolve.clone(),
                ));
            }
            let c = surface.center;
            let sort_distance = transparent::sort_distance(c, [cam[0], cam[1], cam[2]]);
            out.push(TransparentDraw {
                pipeline: pipeline.clone(),
                reflection_pipeline: None,
                vertex_buffer: surface.vertex_buffer.clone(),
                index_buffer: surface.index_buffer.clone(),
                index_count: surface.index_count,
                index_type: objc2_metal::MTLIndexType::UInt16,
                index_offset_bytes: 0,
                base_vertex: 0,
                params: bytes_of(&params),
                fragment_textures,
                fragment_samplers: vec![(0, self.composite.sampler.clone())],
                sort_distance,
            });
        }
    }
}

// Build the water render pipeline. Standard 5-attribute vertex layout at
// buffer(1); the same descriptor the glass panes and the main pass use, so any
// `ProceduralMesh::water_grid` mesh can bind directly. Output target is
// `scene_pre_taa` (RGBA16Float single-sample); SRC_ALPHA blend writes the
// transparent water on top of whatever the SsrResolve pass produced.
pub(super) fn build_water_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    build_water_pipeline_with(
        device,
        hot_reload,
        &builtin_shaders::WATER_VERT,
        &builtin_shaders::WATER_FRAG,
    )
}

// Build the ray-traced water pipeline: the same vertex layout + blend, but the
// `water_rt_fragment` variant traces a sharp reflection ray against the scene
// acceleration structure for surfaces with no mirror plane, instead of sampling
// a probe cube (one with a plane samples it either way). Built only on
// RT-capable devices (its metallib carries a real ray query); selected per-frame
// only while `self.rt.accel` is live, the probe pipeline otherwise. This is the
// FLAT variant (per-object material tint as albedo).
pub(super) fn build_water_pipeline_rt(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    build_water_pipeline_with(
        device,
        hot_reload,
        &builtin_shaders::WATER_VERT,
        &builtin_shaders::WATER_FRAG_RT,
    )
}

// Build the textured ray-traced water pipeline: the same trace as the flat RT
// variant, but the reflected hit's albedo / normal / emissive are sampled from
// the bindless texture pool (buffer 10) instead of a flat per-object tint.
// Selected over the flat variant only in a bindless world.
pub(super) fn build_water_pipeline_rt_textured(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    build_water_pipeline_with(
        device,
        hot_reload,
        &builtin_shaders::WATER_VERT,
        &builtin_shaders::WATER_FRAG_RT_TEXTURED,
    )
}

// The water pipelines, whose stages come from the single-source `water.hlsl`.
// Each fragment variant declares only the resources it binds, so each is its own
// metallib while the vertex is compiled once for all of them.
fn build_water_pipeline_with(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
    vertex: &builtin_shaders::ShaderProgram,
    fragment: &builtin_shaders::ShaderProgram,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vert_fn = builtin_shaders::entry_function(device, vertex, hot_reload)?;
    let frag_fn = builtin_shaders::entry_function(device, fragment, hot_reload)?;
    build_transparent_pipeline_stages(device, &vert_fn, &frag_fn)
}
