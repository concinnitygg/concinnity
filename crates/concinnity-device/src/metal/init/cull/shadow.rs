//! The GPU-driven cascade shadow: the frustum-only decision kernel and the
//! depth-only bindless shadow pipeline.

use concinnity_core::render::error::RenderResult;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputePipelineState, MTLRenderPipelineState, MTLVertexDescriptor};

use super::bindless::BindlessPass;
use crate::metal::cull::build_shadow_cull_pipeline;
use crate::metal::init::InitGpu;
use crate::metal::init::pipelines::build_shadow_bindless_pipeline;

pub(super) struct ShadowCull {
    pub(super) pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(super) bindless_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
}

// Built only for a scene world with shadows enabled; a UI-only or shadowless
// world leaves both `None` and renders no cascades. The shadow ICB + its
// argument buffer are allocated lazily by `ensure_shadow_icb_capacity` (sized
// to NUM_SHADOW_CASCADES * cull_count once geometry is known).
pub(super) fn build_shadow_cull(
    gpu: &InitGpu<'_>,
    bindless: &BindlessPass,
    vert_desc: &MTLVertexDescriptor,
    shadow_enabled: bool,
) -> RenderResult<ShadowCull> {
    let device = &*gpu.hw.device;
    let (pipeline, bindless_pipeline) = if shadow_enabled && bindless.active {
        let sc = build_shadow_cull_pipeline(device, gpu.hot_reload)?;
        let sb = build_shadow_bindless_pipeline(device, vert_desc, gpu.hot_reload)?;
        (Some(sc), Some(sb))
    } else {
        (None, None)
    };
    Ok(ShadowCull {
        pipeline,
        bindless_pipeline,
    })
}
