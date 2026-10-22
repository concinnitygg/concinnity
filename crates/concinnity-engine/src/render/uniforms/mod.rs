//! The `#[repr(C)]` blocks the CPU uploads into a single-source shader,
//! declared once for every backend.
//!
//! The `metal`, `directx` and `vulkan` children are the exception: a block only
//! one backend binds, or one whose shader is still hand-written per backend
//! (the cull kernel, the per-draw morph kernel, Metal's water).
//!
//! Binding slots are not part of these declarations: the same block lands at a
//! different index on each backend, so where it binds belongs to the backend
//! that binds it.

pub mod bindless;
pub mod canopy;
pub mod daylight;
pub mod directx;
pub mod geometry;
pub mod grass;
pub mod ground;
pub mod metal;
pub mod ocean;
pub mod post;
pub mod probe;
pub mod raymarch;
pub mod shore;
pub mod transparent;
pub mod view;
pub mod vulkan;

pub use bindless::BINDLESS_POOL_SIZE;
pub use daylight::{DaylightBlock, DaylightBody, DaylightView};
pub use geometry::{
    DecalParams, DecalView, GpuParticle, LineView, ModelHistoryParams, ParticleView, SkinParams,
};
pub use post::{AutoExposureParams, ExposureMeasurement, HizParams, HizSpdParams, TaaParams};
pub use probe::{
    DEFAULT_PROBE_RECORD_CAPACITY, ProbePrefilterParams, ProbeSet, ProbeUniforms,
    grown_probe_capacity,
};
pub use raymarch::{RaymarchShadowCascade, RaymarchView, RaymarchVolumeUniforms};
pub use shore::WaterShoreGpu;
pub use transparent::{
    GlassMeshParams, GlassParams, TransparentView, WATER_MAX_WAVES, WaterOceanGpu, WaterParams,
    WaterRippleGpu, WaterSeaFaceGpu, WaterSeaGpu, WaterWaveGpu,
};
pub use view::{FaceCamera, GBufferView, PassCamera, ViewUniforms};
