//! DirectX's share of screen-space global illumination, which is its settings
//! and where the pass reads and writes this frame. Every stage -- the pipelines, the depth
//! pyramids, the accumulation and all the draws -- is written once in
//! `concinnity_engine::render::post::ssgi` and reaches D3D12 through
//! `DxPostDevice`.

use concinnity_engine::gfx::render_types::SsgiParams;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::post::device::PostExtent;
use concinnity_engine::render::post::ssgi::settings::{SsgiFrame, SsgiSettings};
use concinnity_engine::render::post::ssgi::{
    SsgiPass, SsgiPipelines, SsgiResolveInputs, SsgiTraceInputs,
};
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::context::DxContext;
use crate::directx::descriptor_slot::SrvSlot;
use crate::directx::post::post_device::{DxPostDevice, PostPipeline, PostTarget};

// SSGI resources held by `DxContext` when `IndirectLighting.technique`
// is `ssgi`.
pub(in crate::directx) struct SsgiResources {
    // Resolved authored tunables; turned into a per-frame `SsgiParams` block.
    pub(in crate::directx) settings: SsgiSettings,
    pass: SsgiPass<PostPipeline, PostTarget>,
}

impl SsgiResources {
    // Build every pipeline and target for a render resolution of `width` x
    // `height`.
    pub(in crate::directx) fn new(
        device: &DxPostDevice<'_>,
        width: u32,
        height: u32,
        settings: SsgiSettings,
    ) -> RenderResult<Self> {
        Ok(Self {
            settings,
            pass: SsgiPass::new(device, settings.gi_scale, PostExtent { width, height })?,
        })
    }

    // Recreate the targets at a new render resolution. Every draw reads its
    // descriptors per frame, so the slots they land on are free to move.
    pub(in crate::directx) fn resize_to(
        &mut self,
        device: &DxPostDevice<'_>,
        width: u32,
        height: u32,
    ) -> RenderResult<()> {
        self.pass.resize(device, PostExtent { width, height })
    }

    // Swap in freshly built pipelines. Driven by shader hot reload; the caller
    // has already idled the device.
    pub(in crate::directx) fn swap_pipelines(&mut self, pipelines: SsgiPipelines<PostPipeline>) {
        self.pass.swap_pipelines(pipelines);
    }

    // Step the accumulation ring once the frame is recorded.
    pub(in crate::directx) fn advance(&mut self) {
        self.pass.advance();
    }

    // Forget the accumulation for a frame that does not continue the last.
    pub(in crate::directx) fn reset_history(&mut self) {
        self.pass.reset_history();
    }
}

impl DxContext {
    // Whether the main pass's screen-indirect table carries SSGI's bounce:
    // whenever SSGI is built, which is when the graph runs its resolve.
    pub(in crate::directx) fn screen_bounce(&self) -> bool {
        self.ssgi.is_some()
    }

    // The table the main pass samples the screen-space indirect light through:
    // SSGI's resolved visibility and bounce while it is built, else the
    // ambient occlusion (or its white fallback).
    pub(in crate::directx) fn screen_indirect_srv_gpu(&self) -> SrvSlot {
        match &self.ssgi {
            Some(ssgi) => ssgi.pass.indirect().srv_gpu(),
            None => self.ssao_ao_srv_gpu(),
        }
    }

    // SSGI's resolve output, for the graph to move between the resolve's write
    // and the main pass's read.
    pub(in crate::directx) fn ssgi_indirect_resource(&self) -> Option<&ID3D12Resource> {
        self.ssgi
            .as_ref()
            .map(|ssgi| ssgi.pass.indirect().resource())
    }

    // Where the accumulation stands this frame.
    pub(in crate::directx) fn ssgi_frame(&self) -> Option<SsgiFrame> {
        self.ssgi.as_ref().map(|s| s.pass.frame())
    }

    // Encode the resolve ahead of the main pass: last frame's accumulation
    // reprojected onto this frame's G-buffer, folded with the occlusion, into
    // the output the graph has put in RENDER_TARGET.
    pub(in crate::directx) fn encode_ssgi_resolve(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        params: Option<&SsgiParams>,
    ) {
        let (Some(ssgi), Some(gbuffer), Some(params)) = (&self.ssgi, &self.gbuffer, params) else {
            return;
        };
        if let Err(e) = ssgi.pass.encode_resolve(
            &self.post_device(frame_idx),
            cmd,
            &SsgiResolveInputs {
                normal_depth: gbuffer.normal_depth_srv_gpu,
                velocity: gbuffer.velocity_srv_gpu,
                occlusion: self.ssao_ao_srv_gpu(),
            },
            params,
        ) {
            tracing::error!("SSGI resolve: {e}");
        }
    }

    // Encode SSGI after the main pass: the depth pyramid, the age, the trace
    // over the G-buffer and the accumulation the next frame's resolve reads.
    // The graph has put the scene it samples in the shader-resource state.
    pub(in crate::directx) fn encode_ssgi(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        params: Option<&SsgiParams>,
    ) {
        // With no G-buffer there is nothing to trace against, so skip rather
        // than read a stale descriptor.
        let (Some(ssgi), Some(gbuffer), Some(params)) = (&self.ssgi, &self.gbuffer, params) else {
            return;
        };
        if let Err(e) = ssgi.pass.encode(
            &self.post_device(frame_idx),
            cmd,
            &SsgiTraceInputs {
                scene: self.targets.hdr.srv_gpu,
                normal_depth: gbuffer.normal_depth_srv_gpu,
                velocity: gbuffer.velocity_srv_gpu,
                reflectance: gbuffer.roughness_srv_gpu,
            },
            params,
        ) {
            tracing::error!("SSGI: {e}");
        }
    }
}
