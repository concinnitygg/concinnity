//! The bloom chain: its pipelines and the octaves below the pool's top one.

use concinnity_core::render::error::RenderResult;

use crate::metal::post::post_device::MtlPostDevice;
use crate::metal::post::{MtlBloomPass, build_bloom_pass};

// Built for any world with a 3D scene (only its uniforms vary at runtime), at
// the output resolution so the glow stays on the panel's pixel grid. Without a
// scene there is nothing to threshold, and the graph never inserts the Bloom
// pass.
pub(super) fn build_bloom(
    device: &MtlPostDevice,
    scene: bool,
    output: (u32, u32),
) -> RenderResult<Option<MtlBloomPass>> {
    Ok(if scene {
        Some(build_bloom_pass(device, output.0, output.1)?)
    } else {
        None
    })
}
