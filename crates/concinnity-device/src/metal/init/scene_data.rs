//! Per-scene lighting data the GPU derives each frame: the clustered light
//! binning state.

use concinnity_core::gfx::render_types::GpuLight;
use concinnity_core::render::error::RenderResult;

use super::InitGpu;
use crate::metal::light_cull::{
    LightCullState, build_cluster_light_buffer, build_light_cull_pipeline,
};

// Clustered-lighting resources: the per-cluster light-index buffer
// (always allocated so the forward pass has a valid fragment buffer(12)
// binding) and the binning compute pipeline (built only when the world
// has local lights to bin).
pub(super) fn build_light_cull(
    gpu: &InitGpu<'_>,
    local_lights: &[GpuLight],
) -> RenderResult<LightCullState> {
    let device = &*gpu.hw.device;
    let cluster_buffer = build_cluster_light_buffer(device)?;
    let pipeline = if local_lights.is_empty() {
        None
    } else {
        Some(build_light_cull_pipeline(device, gpu.hot_reload)?)
    };
    Ok(LightCullState {
        pipeline,
        cluster_buffer,
    })
}
