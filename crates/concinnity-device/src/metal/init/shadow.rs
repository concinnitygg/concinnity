//! Cascade and spot shadow states: the shadow maps, the compare sampler, and
//! the static spot projections.

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types::{self, NUM_SHADOW_CASCADES, SpotShadowData};
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::backend::init::ShadowParams;
use concinnity_engine::render::shadow::spot;
use objc2_metal::{
    MTLDevice as _, MTLResourceOptions, MTLSamplerAddressMode, MTLSamplerDescriptor,
    MTLSamplerMinMagFilter,
};

use super::InitGpu;
use crate::metal::context::{ShadowState, SpotShadowState, bytes_of_slice};
use crate::metal::depth::SHADOW_SAMPLE_COMPARE;
use crate::metal::texture::{create_shadow_map_array, create_shadow_map_fallback};

pub(super) fn build_shadow(gpu: &InitGpu<'_>, shadows: &ShadowParams) -> RenderResult<ShadowState> {
    let device = &*gpu.hw.device;

    // The cascade array map exists only when the map size is > 0. The fallback
    // 1x1 shadow map (all depth = 1.0 = max = lit) is always bound so fragment
    // shaders can safely sample texture(2) as a depth array.
    let enabled = shadows.map_size > 0;
    let (map, map_size) = if enabled {
        // Depth32Float 2D array, NUM_SHADOW_CASCADES layers, GPU-private.
        let shadow_tex = create_shadow_map_array(
            device,
            shadows.map_size,
            u32::saturating_from(NUM_SHADOW_CASCADES),
        )?;
        (shadow_tex, shadows.map_size)
    } else {
        // 1x1 fallback depth array holding the clear value, which every
        // reference passes against: fully lit.
        (create_shadow_map_fallback(device)?, 1)
    };
    // compare sampler for PCF: always created so texture(2) / sampler(1) are
    // always bound; returns 1.0 (lit) where the reference is no farther from
    // the light than the stored depth.
    let sampler = {
        let desc = MTLSamplerDescriptor::new();
        desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
        desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
        desc.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
        desc.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
        desc.setCompareFunction(SHADOW_SAMPLE_COMPARE);
        // Rides the engine sampler block alongside the pool sampler.
        desc.setSupportArgumentBuffers(true);
        device
            .newSamplerStateWithDescriptor(&desc)
            .ok_or_else(|| RenderError::Other("failed to create shadow sampler state".into()))?
    };

    Ok(ShadowState {
        enabled,
        map,
        map_size,
        sampler,
    })
}

// Spot shadow resources: one array slice per shadow-casting spot plus the
// per-slice projections. Local lights are static, so both are built once
// here and never rebuilt. A scene with no casting spot gets the 1x1
// fallback array (depth 1.0 = lit) and a placeholder buffer, since Metal
// rejects zero-length buffers and the fragment binding must stay valid.
pub(super) fn build_spot_shadow(
    gpu: &InitGpu<'_>,
    spot_shadows: &[SpotShadowData],
    shadow_map_size: u32,
) -> RenderResult<SpotShadowState> {
    let device = &*gpu.hw.device;
    let count = u32::saturating_from(spot_shadows.len());
    let map = if spot_shadows.is_empty() {
        create_shadow_map_fallback(device)?
    } else {
        create_shadow_map_array(
            device,
            render_types::spot_shadow_slice_size(shadow_map_size),
            count,
        )?
    };
    let buffer = if spot_shadows.is_empty() {
        gpu.hw.allocator.alloc_buffer(
            size_of::<SpotShadowData>(),
            MTLResourceOptions::StorageModeShared,
        )
    } else {
        gpu.hw.allocator.alloc_buffer_with_bytes(
            bytes_of_slice(spot_shadows),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .map_err(|e| e.context("spot-shadow buffer"))?;
    Ok(SpotShadowState {
        map,
        buffer,
        count,
        frusta: spot_shadows.iter().map(spot::slice_frustum).collect(),
        render_mask: 0,
    })
}
