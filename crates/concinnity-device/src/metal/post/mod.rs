// src/metal/post/mod.rs
//
// Screen-space and post-process passes for the Metal frame encoder. Each
// effect lives in its own file with its pipeline builder(s), target
// allocator(s), and per-frame encoder(s) co-located:
//
//   gbuffer.rs unified normal+depth / roughness / velocity G-buffer pre-pass
//   ssao.rs   GTAO depth+normal pre-pass + horizon-search kernel + blur
//   ssr.rs    reflection targets + composite, and the inputs to the shared resolve
//   ssgi.rs   the inputs to the shared gather + composite
//   taa.rs    the TAA toggle + jitter counter over the shared resolve
//   bloom.rs  prefilter + downsample/upsample mip chain
//
// `post_device.rs` is the shared fullscreen post-pass seam's Metal half.
//
// Pipeline builders / targets that any other module reaches are re-exported
// here so call sites have a single `crate::metal::post::*` import.

pub(super) mod bloom;
pub(super) mod fullscreen;
pub(super) mod gbuffer;
pub(super) mod post_device;
pub(super) mod rt_reflections;
pub(super) mod ssao;
pub(super) mod ssgi;
pub(super) mod ssr;
pub(super) mod taa;
pub(super) mod upscale;

pub(super) use bloom::{BloomPipelines, BloomTargets, build_bloom_pipelines, create_bloom_targets};
pub(super) use gbuffer::{GBufferState, build_gbuffer_bindless_pipeline, create_gbuffer_targets};
pub(super) use rt_reflections::build_rt_reflection_pipeline;
pub(super) use ssao::{SsaoState, build_ssao_pipeline, create_ssao_targets};
pub(super) use ssgi::SsgiState;
pub(super) use ssr::{
    SsrState, build_reflection_blur_pipeline, build_reflection_composite_pipeline,
    create_ssr_targets,
};
pub(super) use taa::{TaaState, build_taa_pass};
pub(super) use upscale::{MetalFXUpscaler, UpscaleState, temporal_scaler_supported};
