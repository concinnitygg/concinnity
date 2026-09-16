//! The composite pass state: the post-process pipeline and its sampler.

use concinnity_core::render::error::{RenderError, RenderResult};
use objc2_metal::{
    MTLDevice as _, MTLSamplerAddressMode, MTLSamplerDescriptor, MTLSamplerMinMagFilter,
};

use super::InitGpu;
use crate::metal::context::CompositeState;
use crate::metal::pipeline::build_post_pipeline;

// The composite pass samples the resolved HDR target with a linear-clamp
// filter and writes either ACES-tonemapped + gamma + FXAA-filtered output (SDR
// drawable) or linear extended-range values (HDR drawable) into the swapchain.
pub(super) fn build_composite(gpu: &InitGpu<'_>) -> RenderResult<CompositeState> {
    let device = &*gpu.hw.device;
    let pipeline = build_post_pipeline(device, gpu.hw.swap_pixel_format, gpu.hot_reload)?;
    let sampler = {
        let desc = MTLSamplerDescriptor::new();
        desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
        desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
        desc.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
        desc.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
        // Clamp the R axis too -- the same sampler trilinearly filters the
        // 3D color LUT in the composite pass.
        desc.setRAddressMode(MTLSamplerAddressMode::ClampToEdge);
        device
            .newSamplerStateWithDescriptor(&desc)
            .ok_or_else(|| RenderError::Other("failed to create post sampler state".into()))?
    };
    Ok(CompositeState { pipeline, sampler })
}
