// src/directx/post/ssgi.rs
//
// DirectX's share of screen-space global illumination, which is its settings,
// where the pass reads and writes this frame, and the one resource state the
// graph cannot express for it. The gather and composite -- their pipelines, the
// reduced gather target and both draws -- are written once in
// `concinnity_core::render::post::ssgi` and reach D3D12 through `DxPostDevice`.

use concinnity_core::gfx::ssgi::SsgiSettings;
use concinnity_core::render::post::device::PostExtent;
use concinnity_core::render::post::ssgi::{SsgiPass, SsgiPipelines};
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
    // Build both pipelines and the gather target for a render resolution of
    // `width` x `height`.
    pub(in crate::directx) fn new(
        device: &DxPostDevice,
        width: u32,
        height: u32,
        settings: SsgiSettings,
    ) -> Result<Self, String> {
        Ok(Self {
            settings,
            pass: SsgiPass::new(device, settings.gi_scale, PostExtent { width, height })?,
        })
    }

    // Recreate the gather target at a new render resolution. The composite reads
    // its descriptor per frame, so the slot it lands on is free to move.
    pub(in crate::directx) fn resize_to(
        &mut self,
        device: &DxPostDevice,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        self.pass.resize(device, PostExtent { width, height })
    }

    // Swap in freshly built pipelines. Driven by shader hot reload; the caller
    // has already idled the device.
    pub(in crate::directx) fn swap_pipelines(&mut self, pipelines: SsgiPipelines<PostPipeline>) {
        self.pass.swap_pipelines(pipelines);
    }
}

impl DxContext {
    // Encode the SSGI gather + composite: hemisphere rays marched over the
    // G-buffer into the reduced gather target, then blurred and added into the
    // scene spine (`hdr_resolve`, or `hdr_color` with MSAA off). Runs on the
    // hdr_resolve read-modify-write chain after the main pass.
    pub(in crate::directx) fn encode_ssgi(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        fov_y_radians: f32,
        aspect: f32,
    ) {
        let Some(ssgi) = &self.ssgi else { return };
        // With no G-buffer there is nothing to gather against, so skip rather
        // than read a stale descriptor.
        let Some(gbuffer) = &self.gbuffer else { return };
        let params = ssgi.settings.params(fov_y_radians, aspect);
        let device = self.post_device(frame_idx);

        // The gather samples the scene spine while the composite blends into it,
        // so this node reads and writes one resource. The graph models that as a
        // single write and leaves the spine in RENDER_TARGET; sampling it needs
        // the shader-resource state, so borrow it for the gather and hand it
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
        let gathered = ssgi.pass.encode_gather(
            &device,
            cmd,
            self.hdr.srv_gpu,
            gbuffer.normal_depth_srv_gpu,
            &params,
        );
        borrow(
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        let result = gathered.and_then(|()| {
            ssgi.pass.encode_composite(
                &device,
                cmd,
                self.hdr_scene_attachment(),
                gbuffer.normal_depth_srv_gpu,
                &params,
            )
        });
        if let Err(e) = result {
            tracing::error!("SSGI: {e}");
        }
    }
}
