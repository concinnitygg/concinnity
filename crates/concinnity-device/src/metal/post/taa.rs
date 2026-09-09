// src/metal/post/taa.rs
//
// Metal's share of temporal anti-aliasing, which is the toggle and the jitter
// counter. The resolve itself -- its pipeline, its ping-pong accumulation
// targets, the history-validity gate and the draw -- is written once in
// `concinnity_core::render::post::taa` and reaches Metal through
// `MtlPostDevice`.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLRenderPipelineState, MTLTexture};

use concinnity_core::render::post::device::PostExtent;
use concinnity_core::render::post::taa::{TaaInputs, TaaPass, TaaRing};

use crate::metal::context::MtlContext;
use crate::metal::post::post_device::MtlPostDevice;

// The shared temporal resolve, holding Metal's own pipeline and target handles.
pub(crate) type MtlTaaPass = TaaPass<
    Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    Retained<ProtocolObject<dyn MTLTexture>>,
>;

// Temporal-anti-aliasing state: whether the effect runs, the shared resolve when
// it does, and the frame counter driving the Halton projection jitter. The
// jitter counter is here rather than in the shared pass because it also drives
// the MetalFX upscaler, which runs in the resolve's place.
pub(crate) struct TaaState {
    // Toggle resolved from `PostProcessConfig.aa_mode`; false skips the TAA pass
    // and the projection jitter entirely.
    pub enabled: bool,
    // The resolve. `Some` only when TAA is on (and not bypassed by the
    // upscaler).
    pub pass: Option<MtlTaaPass>,
    // Frame counter driving the Halton projection-jitter sequence.
    pub frame: u32,
}

impl TaaState {
    // The accumulation target this frame writes: what bloom and the composite
    // sample as the scene once the resolve has run.
    pub(crate) fn output(&self) -> Option<&Retained<ProtocolObject<dyn MTLTexture>>> {
        let pass = self.pass.as_ref()?;
        Some(pass.target(pass.ring().write()))
    }
}

// Build the resolve at `width` x `height` render resolution.
pub(crate) fn build_taa_pass(
    device: &MtlPostDevice,
    width: u32,
    height: u32,
) -> Result<MtlTaaPass, String> {
    TaaPass::new(device, TaaRing::ping_pong(), PostExtent { width, height })
}

impl MtlContext {
    // The post-pass device over this context: the Metal device, the linear
    // clamp-to-edge sampler every screen-space source is read through, and the
    // GPU-timing resources.
    pub(in crate::metal) fn post_device(&self) -> MtlPostDevice<'_> {
        MtlPostDevice {
            device: &self.device,
            sampler: &self.post_sampler,
            timing: self.diagnostics.pass_timing.as_ref(),
            hot_reload: self.hot_reload.enabled,
        }
    }

    // Encode the TAA resolve pass: one fullscreen-triangle draw that blends
    // `scene_input` (the reflection composite's output, or `hdr_resolve` when no
    // reflection path is live) with the reprojected history. Runs between SSR
    // and bloom; its output is both the scene colour the later passes consume
    // and next frame's history.
    pub(in crate::metal) fn encode_taa(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        scene_input: &ProtocolObject<dyn objc2_metal::MTLTexture>,
    ) -> Result<u32, String> {
        let pass = self
            .taa
            .pass
            .as_ref()
            .ok_or("TAA enabled but the resolve is missing")?;
        // Pool-owned, so it is fetched at encode time: a pool rebuild repacks
        // every slot, and a cached handle would point at another resource.
        let velocity = self
            .gbuffer_velocity()
            .ok_or("TAA enabled but the pooled G-buffer velocity is missing")?;
        let device = self.post_device();
        pass.encode(
            &device,
            cmd_buf,
            pass.ring().write(),
            TaaInputs {
                scene: scene_input,
                velocity,
            },
        )?;
        Ok(0)
    }
}
