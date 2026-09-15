//! Cascade and spot shadow states: the shadow maps, the cascade pipeline, the
//! compare sampler, and the static spot projections.

use concinnity_core::gfx::render_types::{
    self, LightUniforms, NUM_SHADOW_CASCADES, SpotShadowData,
};
use concinnity_core::render::backend_init::ShadowParams;
use concinnity_core::render::csm;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::lights;
use objc2_metal::{
    MTLCompareFunction, MTLDevice as _, MTLResourceOptions, MTLSamplerAddressMode,
    MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLVertexDescriptor,
};

use super::{InitGpu, pipelines};
use crate::metal::context::{ShadowState, SpotShadowState, bytes_of_slice};
use crate::metal::texture::{create_shadow_map_array, create_shadow_map_fallback};

pub(super) fn build_shadow(
    gpu: &InitGpu<'_>,
    vert_desc: &MTLVertexDescriptor,
    shadows: &ShadowParams,
    light_uniforms: &LightUniforms,
) -> RenderResult<ShadowState> {
    let device = &*gpu.hw.device;

    // shadow pipeline + array map: created only when the map size is > 0.
    // The fallback 1x1 shadow map (all depth = 1.0 = max = lit) is always
    // bound so fragment shaders can safely sample texture(2) as a depth array.
    let (pipeline_state, map, uniforms, map_size) = if shadows.map_size > 0 {
        let shadow_ps = pipelines::build_shadow_pipeline(device, vert_desc, gpu.hot_reload)?;
        // Depth32Float 2D array, NUM_SHADOW_CASCADES layers, GPU-private.
        let shadow_tex =
            create_shadow_map_array(device, shadows.map_size, NUM_SHADOW_CASCADES as u32)?;
        (
            Some(shadow_ps),
            shadow_tex,
            csm::empty_shadow_uniforms(),
            shadows.map_size,
        )
    } else {
        // 1x1 fallback depth array (value 1.0 = fully lit).
        let shadow_tex = create_shadow_map_fallback(device)?;
        (None, shadow_tex, csm::empty_shadow_uniforms(), 1)
    };

    // compare sampler for PCF: always created so texture(2) / sampler(1) are
    // always bound; LessEqual returns 1.0 (lit) when reference <= stored depth.
    let sampler = {
        let desc = MTLSamplerDescriptor::new();
        desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
        desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
        desc.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
        desc.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
        desc.setCompareFunction(MTLCompareFunction::LessEqual);
        // Rides the engine sampler block alongside the pool sampler.
        desc.setSupportArgumentBuffers(true);
        device
            .newSamplerStateWithDescriptor(&desc)
            .ok_or("failed to create shadow sampler state")?
    };

    // Cache the first directional light's direction; per-frame CSM updates
    // use it. `update_directional_lights` re-caches it when the sun changes.
    let light_dir = lights::sun_direction(light_uniforms);

    Ok(ShadowState {
        pipeline_state,
        map,
        map_size,
        update: shadows.update,
        distance: shadows.distance,
        cascades: shadows.cascades,
        scheduler: Default::default(),
        render_mask: 0,
        sampler,
        uniforms,
        light_dir,
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
    let count = spot_shadows.len() as u32;
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
            std::mem::size_of::<SpotShadowData>(),
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
        scheduler: Default::default(),
        render_mask: 0,
    })
}
