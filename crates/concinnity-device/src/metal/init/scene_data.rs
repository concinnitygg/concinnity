//! Per-scene lighting data the GPU derives each frame: the clustered light
//! binning state.

use concinnity_core::render::error::RenderResult;

use super::InitGpu;
use crate::metal::light_cull::{
    LightCullState, build_cluster_light_buffer, build_light_cull_pipeline,
};

// Clustered-lighting resources: the per-cluster list buffer (always allocated
// so the forward pass has a valid fragment buffer(12) binding) and the binning
// compute pipeline. Built whatever the world declares, since reflection probes
// are placed after init.
pub(super) fn build_light_cull(gpu: &InitGpu<'_>) -> RenderResult<LightCullState> {
    let device = &*gpu.hw.device;
    Ok(LightCullState {
        pipeline: build_light_cull_pipeline(device, gpu.hot_reload)?,
        cluster_buffer: build_cluster_light_buffer(device)?,
    })
}
