//! The GPU-driven shadow views, cascades, spot slices and the far view alike:
//! the frustum-only decision kernel and the depth-only bindless shadow
//! pipelines, one per depth format.

use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::backend::init::ShadowMapSizes;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLComputePipelineState, MTLPixelFormat, MTLRenderPipelineState, MTLVertexDescriptor,
};

use super::bindless::BindlessPass;
use crate::metal::cull::build_shadow_cull_pipeline;
use crate::metal::init::InitGpu;
use crate::metal::init::pipelines::build_shadow_bindless_pipeline;

pub(super) struct ShadowCull {
    pub(super) pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(super) bindless_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // The same draws into the far shadow's Depth16Unorm map.
    pub(super) far_bindless_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
}

// Built only for a scene world with shadows enabled; a UI-only or shadowless
// world leaves both `None` and renders no shadow views. The cascade and spot
// ICBs + their argument buffers are allocated lazily by
// `ensure_shadow_icb_capacity`, one region per view times `cull_count` once
// geometry is known.
pub(super) fn build_shadow_cull(
    gpu: &InitGpu<'_>,
    bindless: &BindlessPass,
    vert_desc: &MTLVertexDescriptor,
    maps: ShadowMapSizes,
) -> RenderResult<ShadowCull> {
    let device = &*gpu.hw.device;
    if maps.cascades == 0 || !bindless.active {
        return Ok(ShadowCull {
            pipeline: None,
            bindless_pipeline: None,
            far_bindless_pipeline: None,
        });
    }
    let far_bindless_pipeline = if maps.far > 0 {
        Some(build_shadow_bindless_pipeline(
            device,
            vert_desc,
            MTLPixelFormat::Depth16Unorm,
            gpu.hot_reload,
        )?)
    } else {
        None
    };
    Ok(ShadowCull {
        pipeline: Some(build_shadow_cull_pipeline(device, gpu.hot_reload)?),
        bindless_pipeline: Some(build_shadow_bindless_pipeline(
            device,
            vert_desc,
            MTLPixelFormat::Depth32Float,
            gpu.hot_reload,
        )?),
        far_bindless_pipeline,
    })
}
