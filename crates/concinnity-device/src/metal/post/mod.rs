//! Screen-space and post-process passes for the Metal frame encoder. Each
//! effect lives in its own file with its pipeline builder(s), target
//! allocator(s), and per-frame encoder(s) co-located:
//!
//!   gbuffer.rs              unified normal+depth / roughness / velocity G-buffer pre-pass
//!   ssao.rs                 the SSAO settings + white fallback over the shared kernel + blur
//!   ssr.rs                  the reflection target, and the inputs to the shared resolve
//!   reflection_composite.rs the inputs to the shared roughness blur + composite
//!   ssgi.rs                 the inputs to the shared trace + composite
//!   taa.rs                  the TAA toggle + jitter counter over the shared resolve
//!   bloom.rs                the inputs to the shared bloom chain
//!
//! `post_device.rs` is the shared fullscreen post-pass seam's Metal half.
//!
//! Pipeline builders / targets that any other module reaches are re-exported
//! here so call sites have a single `crate::metal::post::*` import.

pub(super) mod bloom;
pub(super) mod fullscreen;
pub(super) mod gbuffer;
pub(super) mod post_device;
pub(super) mod reflection_composite;
pub(super) mod rt_reflections;
pub(super) mod ssao;
pub(super) mod ssgi;
pub(super) mod ssr;
pub(super) mod taa;
pub(super) mod upscale;

pub(super) use bloom::{MtlBloomPass, build_bloom_pass};
pub(super) use gbuffer::{GBufferState, build_gbuffer_prepass_pipeline, create_gbuffer_targets};
pub(super) use reflection_composite::build_reflection_composite;
pub(super) use rt_reflections::build_rt_reflection_pipeline;
pub(super) use ssao::SsaoState;
pub(super) use ssgi::SsgiState;
pub(super) use ssr::{SsrState, create_reflection_target};
pub(super) use taa::{TaaState, build_taa_pass};
pub(super) use upscale::{MetalFXUpscaler, UpscaleState, temporal_scaler_supported};
