//! Post-process effects for the D3D12 backend, each owning pipeline + targets +
//! per-frame encoder co-located in one file:
//!
//!   bloom.rs                the pooled top octave + state change of the shared bloom chain
//!   gbuffer.rs              unified normal+depth / roughness / velocity MRT pre-pass
//!   taa.rs                  the jitter counter + inputs of the shared TAA resolve
//!   ssao.rs                 the settings + pooled output of the shared SSAO kernel and blur
//!   ssr.rs                  the reflection target + inputs of the shared SSR resolve
//!   reflection_composite.rs the inputs of the shared roughness blur + composite
//!   ssgi.rs                 the settings + inputs of the shared SSGI trace and composite
//!
//! Mirrors src/metal/post/ (same per-effect file shape).

pub(in crate::directx) mod bloom;
pub(in crate::directx) mod gbuffer;
mod gbuffer_sky;
pub(in crate::directx) mod reflection_composite;
pub(in crate::directx) mod rt_reflections;
pub(in crate::directx) mod ssao;
pub(in crate::directx) mod ssgi;
pub(in crate::directx) mod ssr;
pub(in crate::directx) mod taa;

pub(in crate::directx) mod descriptors;
pub(in crate::directx) mod post_device;
pub(in crate::directx) mod upscale;
