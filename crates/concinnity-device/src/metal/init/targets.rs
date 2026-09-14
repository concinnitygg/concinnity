//! Render-resolution scene targets: the depth-stencil states, the HDR scene
//! target, the render graph's transient texture pool, and the bloom mip chain
//! whose top mip lives in that pool.

use concinnity_core::render::error::RenderResult;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use super::{Features, InitGpu, pipelines};
use crate::metal::context::MtlTargets;
use crate::metal::post::create_bloom_targets;
use crate::metal::texture::create_hdr_targets;
use crate::metal::transient_pool::{TransientTexturePool, transient_slots};

pub(super) fn build_targets(gpu: &InitGpu<'_>, features: &Features) -> RenderResult<MtlTargets> {
    let device = &*gpu.hw.device;
    let (render_w, render_h) = features.render;
    let (output_w, output_h) = features.output;

    let depth_state = pipelines::make_depth_state(device)?;
    let depth_state_read_only = pipelines::make_depth_state_read_only(device)?;
    let hdr = create_hdr_targets(device, render_w, render_h, features.hdr_samples)?;
    let transient_pool = build_transient_pool(
        device,
        features.ssao_enabled,
        features.gbuffer_enabled,
        features.render,
        features.output,
    )?;

    // Bloom samples whatever scene_color the post stack hands it: that's at
    // output (drawable) resolution when MetalFX upscaling is on, native
    // resolution otherwise. Sized off the output extent so bloom stays crisp at
    // the panel's pixel grid. The targets exist for every world because the
    // composite pass binds the top mip unconditionally (1x1 scene-less). Mip 0
    // is the pool's `bloom_top`, which the pool always manages.
    let bloom = create_bloom_targets(device, output_w, output_h, transient_pool.bloom_top()?)?;

    Ok(MtlTargets {
        hdr,
        bloom,
        transient_pool,
        depth_state,
        depth_state_read_only,
        geometry_less: !features.scene,
    })
}

// Render-graph transient texture pool: `ao_output` (SSAO's blurred
// occlusion), `bloom_top` (bloom mip 0, half output resolution), and the
// G-buffer pre-pass's normal+depth / roughness / velocity channels. The
// planner packs whichever of them have disjoint lifetimes onto shared heap
// slots. Built before the bloom chain and the pre-pass, both of which read
// their targets back out of it by label.
pub(in crate::metal) fn build_transient_pool(
    device: &ProtocolObject<dyn MTLDevice>,
    ssao_enabled: bool,
    gbuffer_enabled: bool,
    render: (u32, u32),
    output: (u32, u32),
) -> RenderResult<TransientTexturePool> {
    TransientTexturePool::build(
        device,
        &transient_slots(ssao_enabled, gbuffer_enabled, render, output)?,
    )
}
