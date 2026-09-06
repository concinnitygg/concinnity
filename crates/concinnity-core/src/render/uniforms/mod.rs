// src/uniforms/
//
// The `#[repr(C)]` blocks the CPU uploads into a single-source `.slang` shader,
// declared once for every backend.
//
// These used to be declared per backend, in `metal/uniforms.rs`,
// `vulkan/uniforms.rs` and `directx/uniforms.rs`, from the days when each
// backend had its own shader source and its own idea of the layout. The
// single-source migration collapsed the shader side to one declaration, and
// `shader_layout` in concinnity-device then proved the CPU sides byte-identical
// on all three targets -- so a second and third copy could only ever drift.
//
// What stays per backend is what is genuinely per backend: a block only one
// host binds (Metal's `ModelUniforms`), or one whose shader is still
// hand-written per backend (the cull kernel, the per-draw morph kernel, Metal's
// water).
//
// Binding slots are not part of these declarations: the same block lands at a
// different index on each backend, so where it binds belongs to the backend
// that binds it.

pub mod bindless;
pub mod geometry;
pub mod post;
pub mod probe;
pub mod raymarch;
pub mod transparent;
pub mod view;

pub use bindless::BINDLESS_POOL_SIZE;
pub use geometry::{DecalParams, DecalView, GpuParticle, LineView, ParticleView, SkinParams};
pub use post::{AutoExposureParams, HizParams, TaaParams};
pub use probe::{MAX_PROBES, ProbePrefilterParams, ProbeSet, ProbeUniforms};
pub use raymarch::{RaymarchShadowCascade, RaymarchView, RaymarchVolumeUniforms};
pub use transparent::{
    GlassMeshParams, GlassParams, TransparentView, WATER_MAX_WAVES, WaterParams, WaterWaveGpu,
};
pub use view::{GBufferView, ViewUniforms};
