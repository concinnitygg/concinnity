//! Bloom: the shared chain and the view of the pool's `bloom_top` it writes, at
//! the drawable size.

use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::post::device::PostExtent;

use crate::directx::context::DxTargets;
use crate::directx::post::bloom::BloomResources;
use crate::directx::post::post_device::DxPostDevice;

pub(super) fn build_bloom(
    device: &DxPostDevice,
    targets: &DxTargets,
    output: (u32, u32),
) -> RenderResult<BloomResources> {
    let top = targets
        .transient_pool
        .resource_for("bloom_top")
        .ok_or_else(|| RenderError::Other("transient pool missing bloom_top".into()))?;
    BloomResources::new(
        device,
        PostExtent {
            width: output.0,
            height: output.1,
        },
        top,
    )
}
