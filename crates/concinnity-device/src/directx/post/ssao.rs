//! DirectX's share of SSAO (GTAO): the settings, the white fallback the forward
//! pass binds while it is off, and the view of the pool's `ao_output` the blur
//! writes. The kernel and blur -- their pipelines, the raw occlusion between
//! them and both draws -- are written once in
//! `concinnity_core::render::post::ssao` and reach D3D12 through
//! `DxPostDevice`. The view normal + linear depth they read come from the
//! unified G-buffer pre-pass.

use concinnity_core::render::error::RenderResult;
use concinnity_core::render::post::device::PostExtent;
use concinnity_core::render::post::ssao::settings::SsaoSettings;
use concinnity_core::render::post::ssao::{OCCLUSION_FORMAT, SsaoInputs, SsaoPass, SsaoPipelines};
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::allocator::PooledTexture;
use crate::directx::context::DxContext;
use crate::directx::descriptor_slot::SrvSlot;
use crate::directx::post::post_device::{DxPostDevice, PooledTarget, PostPipeline, PostTarget};

// SSAO (GTAO). `resources` is `Some` only when `PostProcessConfig.ssao` is set;
// otherwise the kernel and blur are skipped and the main pass samples the 1x1
// `white` fallback (always present, so the main-pass root signature's AO SRV
// slot always points at a valid descriptor) through `white_srv_gpu` for a
// pass-through ambient term.
pub(in crate::directx) struct SsaoState {
    pub resources: Option<SsaoResources>,
    #[expect(
        dead_code,
        reason = "held to keep the fallback texture resident; the pass binds white_srv_gpu"
    )]
    pub white: PooledTexture,
    pub white_srv_gpu: SrvSlot,
}

// SSAO resources held by `DxContext` when `PostProcessConfig.ssao` is on.
pub(in crate::directx) struct SsaoResources {
    // Resolved authored tunables; turned into a per-frame `SsaoParams` push.
    pub(in crate::directx) settings: SsaoSettings,
    pass: SsaoPass<PostPipeline, PostTarget>,
    // The pool's `ao_output`, which the blur writes and the main pass samples.
    output: PooledTarget,
}

impl SsaoResources {
    // Build the kernel and blur at render resolution `extent`, over the pool's
    // `ao_output`.
    pub(in crate::directx) fn new(
        device: &DxPostDevice,
        settings: SsaoSettings,
        extent: PostExtent,
        ao_output: &ID3D12Resource,
    ) -> RenderResult<Self> {
        Ok(Self {
            settings,
            pass: SsaoPass::new(device, extent)?,
            output: device.pooled_target(ao_output, OCCLUSION_FORMAT, extent)?,
        })
    }

    // Recreate the raw occlusion for a new render resolution and view the
    // rebuilt pool's `ao_output`. The caller has already idled the device.
    pub(in crate::directx) fn resize(
        &mut self,
        device: &DxPostDevice,
        extent: PostExtent,
        ao_output: &ID3D12Resource,
    ) -> RenderResult<()> {
        self.pass.resize(device, extent)?;
        self.repoint_output(device, extent, ao_output)
    }

    // View the pool's `ao_output` after a rebuild relocated it.
    pub(in crate::directx) fn repoint_output(
        &mut self,
        device: &DxPostDevice,
        extent: PostExtent,
        ao_output: &ID3D12Resource,
    ) -> RenderResult<()> {
        self.output = device.pooled_target(ao_output, OCCLUSION_FORMAT, extent)?;
        Ok(())
    }

    pub(in crate::directx) fn swap_pipelines(&mut self, pipelines: SsaoPipelines<PostPipeline>) {
        self.pass.swap_pipelines(pipelines);
    }
}

impl DxContext {
    // GPU descriptor handle of the AO SRV the main pass should sample.
    // Returns the blurred SSAO output when SSAO is on, otherwise the 1x1
    // white fallback so the ambient multiplier is a constant 1.0.
    pub(in crate::directx) fn ssao_ao_srv_gpu(&self) -> SrvSlot {
        match &self.ssao.resources {
            Some(s) => s.output.srv_gpu(),
            None => self.ssao.white_srv_gpu,
        }
    }

    // Encode the GTAO kernel and the depth-aware blur over the unified
    // G-buffer pre-pass into `ao_output`, which the graph has put in
    // RENDER_TARGET for this node and moves back for the main pass. No-op when
    // SSAO is disabled or the G-buffer is absent.
    pub(in crate::directx) fn encode_ssao(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        fov_y_radians: f32,
        aspect: f32,
    ) -> RenderResult<()> {
        let (Some(ssao), Some(gbuffer)) = (&self.ssao.resources, &self.gbuffer) else {
            return Ok(());
        };
        ssao.pass.encode(
            &self.post_device(frame_idx),
            cmd,
            SsaoInputs {
                normal_depth: gbuffer.normal_depth_srv_gpu,
                output: ssao.output.attachment(),
            },
            &ssao.settings.params(fov_y_radians, aspect),
        )
    }
}
