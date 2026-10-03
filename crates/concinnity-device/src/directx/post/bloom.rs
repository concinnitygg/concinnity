//! DirectX's share of bloom: the view of the pool's top octave the chain writes
//! through, and the one state change the chain owns. The chain itself -- its
//! pipelines, the octaves below the top, and every draw -- is written once in
//! `concinnity_core::render::post::bloom` and reaches D3D12 through
//! `DxPostDevice`.

use concinnity_core::render::error::RenderResult;
use concinnity_core::render::post::bloom::{BloomPass, BloomPipelines, top_extent};
use concinnity_core::render::post::device::PostExtent;
use concinnity_core::render::render_graph::PixelFormat;
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::context::DxContext;
use crate::directx::descriptor_slot::SrvSlot;
use crate::directx::post::post_device::{DxPostDevice, PooledTarget, PostPipeline, PostTarget};
use crate::directx::texture::transition_barrier;

// The chain and the pool's `bloom_top` it writes, viewed through the post block.
pub(in crate::directx) struct BloomResources {
    pass: BloomPass<PostPipeline, PostTarget>,
    top: PooledTarget,
}

impl BloomResources {
    // Build the chain for an output of `output`, over the pool's `top`.
    pub(in crate::directx) fn new(
        device: &DxPostDevice,
        output: PostExtent,
        top: &ID3D12Resource,
    ) -> RenderResult<Self> {
        Ok(Self {
            pass: BloomPass::new(device, output)?,
            top: view_top(device, output, top)?,
        })
    }

    // Recreate the octaves for a new output and view the rebuilt pool's `top`.
    // The caller has already idled the device.
    pub(in crate::directx) fn resize(
        &mut self,
        device: &DxPostDevice,
        output: PostExtent,
        top: &ID3D12Resource,
    ) -> RenderResult<()> {
        self.pass.resize(device, output)?;
        self.repoint_top(device, output, top)
    }

    // View the pool's `top` after a rebuild relocated it.
    pub(in crate::directx) fn repoint_top(
        &mut self,
        device: &DxPostDevice,
        output: PostExtent,
        top: &ID3D12Resource,
    ) -> RenderResult<()> {
        self.top = view_top(device, output, top)?;
        Ok(())
    }

    // The glow the composite samples.
    pub(in crate::directx) fn top_srv_gpu(&self) -> SrvSlot {
        self.top.srv_gpu()
    }

    pub(in crate::directx) fn swap_pipelines(&mut self, pipelines: BloomPipelines<PostPipeline>) {
        self.pass.swap_pipelines(pipelines);
    }
}

fn view_top(
    device: &DxPostDevice,
    output: PostExtent,
    top: &ID3D12Resource,
) -> RenderResult<PooledTarget> {
    device.pooled_target(top, PixelFormat::Rgba16Float, top_extent(output))
}

impl DxContext {
    // Encode the chain over `scene_srv` (post-TAA when TAA is on, the HDR scene
    // otherwise). On return `bloom_top` holds the glow the composite samples.
    // Called only when `post_process.bloom_intensity > 0`.
    pub(in crate::directx) fn encode_bloom(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        scene_srv: SrvSlot,
    ) -> RenderResult<()> {
        let Some(bloom) = &self.bloom else {
            return Ok(());
        };
        let device = self.post_device(frame_idx);
        let (pass, bloom_top) = (&bloom.pass, &bloom.top);
        // `bloom_top` arrives in RENDER_TARGET, the graph's state for this node's
        // write, and must leave in it. Between the prefilter and the last
        // upsample the downsample chain samples it, which is the one state
        // change this node owns.
        pass.encode_prefilter(
            &device,
            cmd,
            scene_srv,
            bloom_top.attachment(),
            &self.post_process,
        )?;
        let sampled = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        let target = D3D12_RESOURCE_STATE_RENDER_TARGET;
        // SAFETY: the command list is in the recording state and the pooled
        // resource is live for the frame it records.
        unsafe {
            cmd.ResourceBarrier(&[transition_barrier(bloom_top.resource(), target, sampled)])
        };
        pass.encode_chain(&device, cmd, bloom_top.srv_gpu())?;
        // SAFETY: as above.
        unsafe {
            cmd.ResourceBarrier(&[transition_barrier(bloom_top.resource(), sampled, target)])
        };
        pass.encode_last_upsample(&device, cmd, bloom_top.attachment())
    }
}
