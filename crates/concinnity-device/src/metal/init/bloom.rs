//! The bloom chain's prefilter, downsample and upsample pipelines. Its mip
//! targets are built with the scene targets.

use concinnity_core::render::error::RenderResult;

use super::InitGpu;
use crate::metal::post::{BloomPipelines, build_bloom_pipelines};

// Built for any world with a 3D scene (only its uniforms vary at runtime).
// Without one there is nothing to threshold, and the graph never inserts the
// Bloom pass.
pub(super) fn build_bloom(gpu: &InitGpu<'_>, scene: bool) -> RenderResult<Option<BloomPipelines>> {
    Ok(if scene {
        Some(build_bloom_pipelines(&gpu.hw.device, gpu.hot_reload)?)
    } else {
        None
    })
}
