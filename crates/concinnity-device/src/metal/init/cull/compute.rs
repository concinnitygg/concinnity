//! The compute cull: the early pass and late decision kernels, the kernels
//! that pack their draws into ICB lists, and the Hi-Z depth pyramid the
//! occlusion test reads.

use concinnity_engine::render::backend::error::RenderResult;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLArgumentEncoder, MTLComputePipelineState};

use super::bindless::BindlessPass;
use crate::metal::cull::build_cull_pipeline;
use crate::metal::hiz::HiZResources;
use crate::metal::init::{Features, InitGpu};

pub(super) struct ComputeCull {
    pub(super) pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(super) pipeline_late: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(super) scan_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(super) scatter_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(super) icb_arg_encoder: Option<Retained<ProtocolObject<dyn MTLArgumentEncoder>>>,
    pub(super) hiz: Option<HiZResources>,
}

pub(super) fn build_compute_cull(
    gpu: &InitGpu<'_>,
    bindless: &BindlessPass,
    features: &Features,
) -> RenderResult<ComputeCull> {
    if bindless.main_pipeline.is_none() {
        return Ok(ComputeCull {
            pipeline: None,
            pipeline_late: None,
            scan_pipeline: None,
            scatter_pipeline: None,
            icb_arg_encoder: None,
            hiz: None,
        });
    }
    let device = &*gpu.hw.device;

    // The GPU-driven cull's pipelines, the late pass included: built whenever the
    // bindless path is active (cheap: one extra compute pipeline) and only
    // used when `occlusion_two_pass` is on at runtime.
    let cull = build_cull_pipeline(device, gpu.hot_reload)?;

    // Hi-Z depth pyramid for GPU-driven occlusion culling, sized to the render
    // (depth) resolution; `resize_targets_if_needed` rebuilds it on a window
    // resize. The cull kernel projects each AABB through the previous frame's
    // depth pyramid and culls fully-occluded objects.
    let hiz = HiZResources::new(
        device,
        features.render.0,
        features.render.1,
        gpu.hot_reload,
        features.hdr_samples,
    )?;

    Ok(ComputeCull {
        pipeline: Some(cull.decide),
        pipeline_late: Some(cull.decide_late),
        scan_pipeline: Some(cull.scan),
        scatter_pipeline: Some(cull.scatter),
        icb_arg_encoder: Some(cull.icb_arg_encoder),
        hiz: Some(hiz),
    })
}
