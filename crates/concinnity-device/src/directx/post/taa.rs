// src/directx/post/taa.rs
//
// DirectX's share of temporal anti-aliasing, which is the jitter counter and
// where the resolve's two inputs come from this frame. The resolve itself --
// its pipeline, its ping-pong accumulation targets, the history-validity gate
// and the draw -- is written once in `concinnity_core::render::post::taa` and
// reaches D3D12 through `DxPostDevice`.

use std::cell::Cell;

use windows::Win32::Graphics::Direct3D12::*;

use concinnity_core::render::post::device::PostExtent;
use concinnity_core::render::post::taa::{TaaInputs, TaaPass, TaaRing};

use crate::directx::context::DxContext;
use crate::directx::post::post_device::{DxPostDevice, PostPipeline, PostTarget};

// The shared temporal resolve, holding D3D12's own pipeline and target types.
pub(in crate::directx) type DxTaaPass = TaaPass<PostPipeline, PostTarget>;

// Temporal-anti-aliasing resources: the shared resolve plus the frame counter
// driving the Halton jitter sequence. The counter is here rather than in the
// resolve because it also drives the temporal upscalers, which run in the
// resolve's place; `Cell` because `record_frame` reads it through `&self`.
pub(in crate::directx) struct TaaResources {
    pub(in crate::directx) pass: DxTaaPass,
    pub(in crate::directx) frame: Cell<u32>,
}

impl TaaResources {
    // Build the resolve at `width` x `height` render resolution.
    pub(in crate::directx) fn new(
        device: &DxPostDevice,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        Ok(Self {
            pass: TaaPass::new(device, TaaRing::ping_pong(), PostExtent { width, height })?,
            frame: Cell::new(0),
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

    // Rebuild the accumulation targets at a new resolution. Their descriptors
    // come back on the same heap slots (the shared post block is rewound first),
    // so the live composite and bloom-prefilter bindings keep working without a
    // re-bind. History is unreliable across a resize -- the reprojection
    // coordinates were generated at the old resolution -- so the resolve treats
    // the next frame as the first, and the jitter sequence restarts with it.
    pub(in crate::directx) fn resize_to(
        &mut self,
        device: &DxPostDevice,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        device.descriptors.rewind();
        self.pass.resize(device, PostExtent { width, height })?;
        self.frame.set(0);
        Ok(())
    }
}

impl DxContext {
    // Encode the TAA history-resolve pass: a fullscreen triangle that reprojects
    // the accumulated history through the G-buffer's motion buffer, clips it to
    // the current frame's neighbourhood, and blends. Writes this frame's
    // ping-pong slot, reading the other as history. Called only when `self.taa`
    // is `Some`, after the unified G-buffer pre-pass.
    pub(in crate::directx) fn encode_taa(&self, cmd: &ID3D12GraphicsCommandList) {
        let Some(taa) = &self.taa else { return };
        let Some(gbuffer) = &self.gbuffer else { return };
        let device = self.post_device();
        if let Err(e) = taa.pass.encode(
            &device,
            cmd,
            taa.output_index(),
            TaaInputs {
                scene: self.scene_srv_for_post(),
                velocity: gbuffer.velocity_srv_gpu,
            },
        ) {
            tracing::error!("TAA resolve: {e}");
        }
    }
}
