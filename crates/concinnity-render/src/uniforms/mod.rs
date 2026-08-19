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
// host binds (Vulkan's combined `GbModelPush`, Metal's `ModelUniforms`), or one
// whose shader is still hand-written per backend (the cull kernel, the skinning
// and morph kernels, the raymarch templates, Metal's water and glass_mesh_rt).
//
// Binding slots are not part of these declarations: the same block lands at a
// different index on each backend, so where it binds belongs to the backend
// that binds it. The one exception is called out on `ViewUniforms`, whose Metal
// slot and field names are a published contract for world-authored shaders.

pub mod geometry;
pub mod post;
pub mod probe;
pub mod transparent;
pub mod view;

pub use geometry::{DecalParams, DecalView, GpuParticle, LineView, ParticleView};
pub use post::{AutoExposureParams, HizParams, TaaParams};
pub use probe::{MAX_PROBES, ProbeSet, ProbeUniforms};
pub use transparent::{GlassParams, TransparentView};
pub use view::{GBufferModel, GBufferView, ViewUniforms};
