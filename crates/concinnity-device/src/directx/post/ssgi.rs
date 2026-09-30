//! DirectX's share of screen-space global illumination, which is its settings,
//! where the pass reads and writes this frame, and the one resource state the
//! graph cannot express for it. Every stage -- the pipelines, the depth
//! pyramids, the accumulation and all the draws -- is written once in
//! `concinnity_core::render::post::ssgi` and reaches D3D12 through
//! `DxPostDevice`.

use concinnity_core::render::error::RenderResult;
use concinnity_core::render::post::device::PostExtent;
use concinnity_core::render::post::ssgi::settings::SsgiSettings;
use concinnity_core::render::post::ssgi::{SsgiPass, SsgiPipelines, SsgiTraceInputs};
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::context::DxContext;
use crate::directx::post::post_device::{DxPostDevice, PostPipeline, PostTarget};
use crate::directx::texture::transition_barrier;

// SSGI resources held by `DxContext` when `PostProcessConfig.indirect_lighting`
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
        device: &DxPostDevice,
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
        device: &DxPostDevice,
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
}

impl DxContext {
    // Encode SSGI: the depth pyramid, the trace over the G-buffer, the
    // accumulation, and the composite into the scene spine (`hdr_resolve`, or
    // `hdr_color` with MSAA off). Runs on the hdr_resolve read-modify-write
    // chain after the main pass.
    pub(in crate::directx) fn encode_ssgi(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        fov_y_radians: f32,
        aspect: f32,
    ) {
        let Some(ssgi) = &self.ssgi else { return };
        // With no G-buffer there is nothing to trace against, so skip rather
        // than read a stale descriptor.
        let Some(gbuffer) = &self.gbuffer else { return };
        let params = ssgi
            .settings
            .params(fov_y_radians, aspect, ssgi.pass.frame());
        let device = self.post_device(frame_idx);
        let normal_depth = gbuffer.normal_depth_srv_gpu;
        if let Err(e) = ssgi
            .pass
            .encode_pyramid(&device, cmd, normal_depth, &params)
        {
            tracing::error!("SSGI: {e}");
            return;
        }

        // The trace samples the scene spine while the composite blends into it,
        // so this node reads and writes one resource. The graph models that as a
        // single write and leaves the spine in RENDER_TARGET; sampling it needs
        // the shader-resource state, so borrow it for the trace and hand it
        // back. Finer than one-state-per-resource, hence inline.
        let borrow = |from, to| {
            // SAFETY: the command list is in the recording state, and the
            // resource the barrier names is live for the call.
            unsafe {
                cmd.ResourceBarrier(&[transition_barrier(self.hdr_scene_target(), from, to)]);
            }
        };
        borrow(
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        let traced = ssgi.pass.encode_trace(
            &device,
            cmd,
            SsgiTraceInputs {
                scene: self.targets.hdr.srv_gpu,
                normal_depth,
                velocity: gbuffer.velocity_srv_gpu,
            },
            &params,
        );
        borrow(
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        let result = traced.and_then(|()| {
            ssgi.pass.encode_composite(
                &device,
                cmd,
                self.hdr_scene_attachment(),
                normal_depth,
                &params,
            )
        });
        if let Err(e) = result {
            tracing::error!("SSGI: {e}");
        }
    }
}
