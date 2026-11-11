//! Per-frame encoder for the volumetric-fog pass. Runs after the main HDR
//! pass (and after the decal pass, so fog sits on top of decals just like it
//! does for any other resolved scene color) and before SSR / TAA, so the
//! reflections and history reproject through the integrated fog color and
//! transmittance.
//!
//! The pass is a single fullscreen triangle: the fragment shader samples the
//! main pass's MSAA depth attachment, reconstructs each pixel's world-space
//! surface point via the inverse VP, ray-marches a sun-lit homogeneous
//! medium with exponential height falloff, and writes `(scattered_rgb, 1 -
//! transmittance)` so the pipeline's `over` blend yields
//! `scene * T + scattered` automatically.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_engine::gfx::render_types::{FogFroxelParams, FogParams};
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::render_graph::{FOG_FROXEL_X, FOG_FROXEL_Y, FOG_FROXEL_Z};
use concinnity_engine::render::shader::programs::abi::fog::FROXEL_GROUP;
use concinnity_engine::render::shader::programs::abi::fog::metal::{draw, froxel};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLCommandBuffer as _, MTLComputePipelineState, MTLDevice as _, MTLLoadAction, MTLPixelFormat,
    MTLPrimitiveType, MTLRenderCommandEncoder as _, MTLRenderPassDescriptor,
    MTLRenderPipelineState, MTLStoreAction, MTLTexture, MTLTextureType, MTLTextureUsage,
};

use super::builtin_shaders::{FOG_FRAG, FOG_FROXEL, compute_pipeline};
use super::context::MtlContext;
use super::descriptors::TextureDesc;
use super::dispatch::dispatch_threads;
use super::encode::{ComputeEncode, RenderEncode};
use super::error::allocation_failed;
use super::post::fullscreen::{
    FullscreenBlend, build_fullscreen_pipeline, set_fragment_sampler_range,
};
use super::scoped_encoder::ScopedEncoder;

// The volumetric-fog passes: the fullscreen ray-march pipeline, and the
// froxel-volume compute pipeline + its 3D output volume. All `Some` only when
// the world declares a `VolumetricFog` with `enabled = true`; `None` skips the
// pass entirely. The live settings are the scene's (`SceneState::fog`).
pub(crate) struct FogState {
    pub pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub froxel_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub froxel_volume: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
}

impl MtlContext {
    // Encode the volumetric-fog pass. Caller has already ended the main HDR
    // pass (and the decal pass, if any), so `targets.hdr.depth` (MSAA) holds
    // the scene depth and `targets.hdr.hdr_resolve` holds the resolved
    // scene + decals color. The pass alpha-blends a single lit ray-march
    // over `hdr_resolve`.
    // pub(in crate::metal) so the render-graph executor in
    // metal/graph_exec.rs can dispatch this pass from a CompiledGraph.
    pub(in crate::metal) fn encode_fog(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        params: &FogParams,
        froxel_params: &FogFroxelParams,
    ) -> RenderResult<u32> {
        let Some(pipeline) = &self.fog.pipeline else {
            return Ok(0);
        };
        let Some(volume) = &self.fog.froxel_volume else {
            return Ok(0);
        };

        let pass_desc = MTLRenderPassDescriptor::new();
        // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
        // declares.
        unsafe {
            let ca = pass_desc.colorAttachments().objectAtIndexedSubscript(0);
            ca.setTexture(Some(self.targets.hdr.hdr_resolve.as_ref()));
            ca.setLoadAction(MTLLoadAction::Load);
            ca.setStoreAction(MTLStoreAction::Store);
        }

        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_render(&pass_desc, super::pass_timing::PassId::Fog);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&pass_desc)
                .ok_or_else(|| RenderError::Other("failed to get fog render encoder".into()))?,
            ns_string!("volumetric fog"),
        );
        enc.set_pipeline(pipeline);

        enc.set_fragment_value(params, draw::FOG.index);
        enc.set_fragment_value(froxel_params, draw::FROXEL.index);
        // Read the single-sample `depth_resolve` (post-
        // Main depth + any raymarched surface depth) so fog
        // attenuates raymarched surfaces by their true distance. It is
        // fetched by pixel coordinate and never sampled, so it takes no
        // sampler slot; the volume is trilinearly filtered and takes
        // sampler(0). `composite.sampler` is the linear clamp-to-edge state the
        // shader used to declare inline as a constexpr sampler.
        enc.set_fragment_texture(
            self.targets.hdr.depth_resolve.as_ref(),
            draw::SCENE_DEPTH.index,
        );
        enc.set_fragment_texture(volume.as_ref(), draw::VOLUME.index);
        set_fragment_sampler_range(&enc, &self.composite.sampler, draw::VOLUME_SAMPLER.index, 1);
        // Fullscreen triangle: 3 vertices, no vertex buffer.
        // SAFETY: the vertex shader generates all three vertices, so the draw reads no bound
        // vertex buffer.
        unsafe {
            enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }

        Ok(1)
    }

    // Encode the volumetric-fog froxel-volume compute pass. One thread per
    // (x, y) tile of the 3D volume; the kernel walks the Z slices from front
    // to back, accumulating per-slab scatter + transmittance with a CSM
    // shadow tap per slice. Writes `(scattered_rgb, 1 - transmittance)` into
    // the slice of `fog_froxel_volume`. Caller must dispatch this before
    // `encode_fog`, which samples the same volume.
    pub(in crate::metal) fn encode_fog_froxel(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        params: &FogParams,
        froxel_params: &FogFroxelParams,
    ) -> RenderResult<u32> {
        let Some(pipeline) = &self.fog.froxel_pipeline else {
            return Ok(0);
        };
        let Some(volume) = &self.fog.froxel_volume else {
            return Ok(0);
        };

        let cmd_buf_dyn: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer> = cmd_buf;
        let desc = objc2_metal::MTLComputePassDescriptor::new();
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_compute(&desc, super::pass_timing::PassId::FogFroxel);
        }
        let enc = ScopedEncoder::new(
            cmd_buf_dyn
                .computeCommandEncoderWithDescriptor(&desc)
                .ok_or_else(|| {
                    RenderError::Other("failed to get fog froxel compute encoder".into())
                })?,
            ns_string!("fog froxel volume"),
        );
        enc.set_pipeline(pipeline);

        enc.set_value(params, froxel::FOG.index);
        enc.set_value(froxel_params, froxel::FROXEL.index);
        // ShadowUniforms at buffer(2) so the kernel can pick a CSM cascade per
        // froxel.
        enc.set_value(&self.state.planner.shadow.uniforms, froxel::SHADOW.index);
        enc.set_texture(self.shadow.map.as_ref(), froxel::SHADOW_MAP.index);
        enc.set_texture(self.shadow.far_texture(), froxel::FAR_SHADOW_MAP.index);
        enc.set_texture(volume.as_ref(), froxel::VOLUME.index);
        // The per-slab CSM tap's comparison sampler, which the shader used
        // to declare inline as a constexpr sampler. `shadow.sampler` is the
        // same linear / clamp-to-edge / less-equal state the main pass taps
        // the cascades with.
        enc.set_sampler(self.shadow.sampler.as_ref(), froxel::SHADOW_SAMPLER.index);

        // One thread per (x, y) tile; the kernel walks Z internally.
        dispatch_threads(&enc, [FOG_FROXEL_X, FOG_FROXEL_Y, 1], FROXEL_GROUP);
        Ok(0)
    }
}

// Build the volumetric-fog pipeline: a fullscreen triangle that samples the
// scene depth, maps each pixel into the froxel volume the compute kernel
// filled, and composites the result over the resolved HDR target with a
// standard `over` alpha blend. Takes the shared `fullscreen_vertex`, which is
// what settles the triangle winding the hand-written Metal vertex wound the
// other way from every other backend.
pub(super) fn build_fog_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    // `(scattered, 1 - T)` over `scene` -> `scene * T + scattered`.
    build_fullscreen_pipeline(
        device,
        &FOG_FRAG,
        MTLPixelFormat::RGBA16Float,
        FullscreenBlend::PremultipliedOver,
        hot_reload,
    )
}

// Build the volumetric-fog froxel-volume compute pipeline from the same
// single-source file the fragment above compiles: the two halves share
// `FogParams` and `FogFroxelParams`, so they move as one unit.
pub(super) fn build_fog_froxel_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    compute_pipeline(device, &FOG_FROXEL, hot_reload)
}

// Allocate the 3D `RGBA16Float` volume the froxel kernel writes and the
// fog fragment shader samples. Dimensions live in
// [`concinnity_engine::render::render_graph::FOG_FROXEL_X`] / `Y` / `Z`.
pub(super) fn build_fog_froxel_volume(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
) -> RenderResult<Retained<ProtocolObject<dyn MTLTexture>>> {
    let desc = TextureDesc {
        kind: MTLTextureType::Type3D,
        format: MTLPixelFormat::RGBA16Float,
        width: FOG_FROXEL_X as usize,
        height: FOG_FROXEL_Y as usize,
        depth: FOG_FROXEL_Z as usize,
        usage: MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
        ..Default::default()
    }
    .build();
    device
        .newTextureWithDescriptor(&desc)
        .ok_or_else(|| allocation_failed("the fog froxel volume texture"))
}
