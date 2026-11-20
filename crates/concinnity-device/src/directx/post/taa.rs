//! DirectX's share of temporal anti-aliasing, which is the jitter counter and
//! where the resolve's inputs come from this frame. The resolve itself --
//! its pipeline, its ping-pong accumulation targets, the history-validity gate
//! and the draw -- is written once in `concinnity_engine::render::post::taa` and
//! reaches D3D12 through `DxPostDevice`.

use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::post::device::PostExtent;
use concinnity_engine::render::post::taa::{TaaInputs, TaaPass, TaaRing};
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::context::DxContext;
use crate::directx::post::post_device::{DxPostDevice, PostPipeline, PostTarget};

// The shared temporal resolve, holding D3D12's own pipeline and target types.
pub(in crate::directx) type DxTaaPass = TaaPass<PostPipeline, PostTarget>;

// Temporal-anti-aliasing resources: the shared resolve.
pub(in crate::directx) struct TaaResources {
    pub(in crate::directx) pass: DxTaaPass,
}

impl TaaResources {
    // Build the resolve at `width` x `height` render resolution.
    pub(in crate::directx) fn new(
        device: &DxPostDevice<'_>,
        width: u32,
        height: u32,
    ) -> RenderResult<Self> {
        Ok(Self {
            pass: TaaPass::new(device, TaaRing::ping_pong(), PostExtent { width, height })?,
        })
    }

    // The accumulation slot this frame writes; the other is the history it
    // samples. Bloom and the composite read this slot's target afterwards.
    pub(in crate::directx) fn output_index(&self) -> usize {
        self.pass.ring().write()
    }

    // The accumulation target this frame writes.
    pub(in crate::directx) fn output(&self) -> &PostTarget {
        self.pass.target(self.output_index())
    }

    // Rebuild the accumulation targets in their slots. History is invalid
    // across a resize, so the next frame starts fresh.
    pub(in crate::directx) fn resize_to(
        &mut self,
        device: &DxPostDevice<'_>,
        width: u32,
        height: u32,
    ) -> RenderResult<()> {
        self.pass.resize(device, PostExtent { width, height })?;
        Ok(())
    }
}

impl DxContext {
    // Encode the TAA resolve: reproject history through the motion buffer, clip
    // to the neighborhood, and blend into this frame's ping-pong slot.
    pub(in crate::directx) fn encode_taa(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        reactive_written: bool,
    ) {
        let Some(taa) = &self.taa else { return };
        let Some(gbuffer) = &self.gbuffer else { return };
        let device = self.post_device(frame_idx);
        if let Err(e) = taa.pass.encode(
            &device,
            cmd,
            taa.output_index(),
            &TaaInputs {
                scene: self.scene_srv_for_post(),
                velocity: gbuffer.velocity_srv_gpu,
                normal_depth: gbuffer.normal_depth_srv_gpu,
                reactive: reactive_written.then(|| self.targets.reactive_mask.srv_gpu()),
                history_scale: self.pre_exposure.history_scale(),
            },
        ) {
            tracing::error!("TAA resolve: {e}");
        }
    }
}
