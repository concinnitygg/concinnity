//! The `#[repr(C)]` blocks the CPU uploads into a single-source `.slang` shader,
//! declared once for every backend.
//!
//! The `metal`, `directx` and `vulkan` children are the exception: a block only
//! one backend binds (Metal's `ModelUniforms`), or one whose shader is still
//! hand-written per backend (the cull kernel, the per-draw morph kernel, Metal's
//! water).
//!
//! Binding slots are not part of these declarations: the same block lands at a
//! different index on each backend, so where it binds belongs to the backend
//! that binds it.

pub mod bindless;
pub mod directx;
pub mod geometry;
pub mod metal;
pub mod post;
pub mod probe;
pub mod raymarch;
pub mod transparent;
pub mod view;
pub mod vulkan;

pub use bindless::BINDLESS_POOL_SIZE;
pub use geometry::{
    DecalParams, DecalView, GpuParticle, LineView, ModelHistoryParams, ParticleView, SkinParams,
};
pub use post::{AutoExposureParams, HizParams, HizSpdParams, TaaParams};
pub use probe::{MAX_PROBES, ProbePrefilterParams, ProbeSet, ProbeUniforms};
pub use raymarch::{RaymarchShadowCascade, RaymarchView, RaymarchVolumeUniforms};
pub use transparent::{
    GlassMeshParams, GlassParams, TransparentView, WATER_MAX_WAVES, WaterParams, WaterWaveGpu,
};
pub use view::{GBufferView, ViewUniforms};
