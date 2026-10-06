//! Render-resolution scene targets: the depth-stencil states, the HDR scene
//! target, and the render graph's transient texture pool.

use concinnity_core::render::error::RenderResult;
use concinnity_core::render::render_graph::{PoolGates, plan_pool_slots};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use super::{Features, InitGpu};
use crate::metal::context::MtlTargets;
use crate::metal::depth::Depth;
use crate::metal::texture::create_hdr_targets;
use crate::metal::transient_pool::TransientTexturePool;

pub(super) fn build_targets(gpu: &InitGpu<'_>, features: &Features) -> RenderResult<MtlTargets> {
    let device = &*gpu.hw.device;
    let (render_w, render_h) = features.render;

    let depth_state = Depth::write().state(device)?;
    let depth_state_inclusive = Depth::write_inclusive().state(device)?;
    let depth_state_read_only = Depth::read_only().state(device)?;
    let hdr = create_hdr_targets(device, render_w, render_h, features.hdr_samples)?;
    let transient_pool = build_transient_pool(
        device,
        features.ssao_enabled,
        features.gbuffer_enabled,
        features.render,
        features.output,
    )?;

    Ok(MtlTargets {
        hdr,
        output: features.output,
        transient_pool,
        depth_state,
        depth_state_inclusive,
        depth_state_read_only,
        geometry_less: !features.scene,
    })
}

// Render-graph transient texture pool: `ao_output` (SSAO's blurred
// occlusion), `bloom_top` (the bloom chain's top octave, half output
// resolution), and the
// G-buffer pre-pass's normal+depth / roughness / velocity channels. The
// planner packs whichever of them have disjoint lifetimes onto shared heap
// slots. The bloom chain, the composite and the pre-pass read their targets
// back out of it by label.
pub(in crate::metal) fn build_transient_pool(
    device: &ProtocolObject<dyn MTLDevice>,
    ssao_enabled: bool,
    gbuffer_enabled: bool,
    render: (u32, u32),
    output: (u32, u32),
) -> RenderResult<TransientTexturePool> {
    TransientTexturePool::build(
        device,
        &plan_pool_slots(
            PoolGates {
                ssao: ssao_enabled,
                gbuffer: gbuffer_enabled,
            },
            render,
            output,
        )?,
    )
}
