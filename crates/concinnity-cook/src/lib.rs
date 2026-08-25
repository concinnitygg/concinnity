//! concinnity-cook: the asset compile pipeline, extracted from concinnity-cpu
//! so the runtime foundation no longer carries the build-only dependencies
//! (fbxcel, fontdue, shaderc, sha2, kira). This crate turns world.jsonl + source
//! files into the binary blobs the runtime reads; it depends on concinnity-cpu
//! and core has no edge back into it.
//!
//! Bridge: the vocabulary and compute modules below are re-exported crate-wide
//! so code moved here keeps resolving its `crate::{assets,ecs,gfx,result}`
//! paths. The payload *decoders* and shared payload types live in
//! `concinnity_cpu::build`; this crate's modules call back into them.
//! The source importers parse artist-supplied files, so a panic here is a crash
//! on a malformed asset rather than a bug. Invariants that genuinely cannot fail
//! use `expect` with the invariant named; tests unwrap freely.
#![warn(clippy::unwrap_used)]
#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "tests unwrap freely; the crate-wide warn covers non-test code"
    )
)]

pub(crate) use concinnity_core::{assets, result};
pub(crate) use concinnity_cpu::gfx;

// The vocabulary's ECS surface, with the build-time name interner shadowing its
// `asset_id`: the interner keeps a per-thread table, so it lives in
// concinnity-cpu and re-exports the vocabulary's `AssetId` / `AssetRef`.
pub(crate) mod ecs {
    pub(crate) use concinnity_core::ecs::*;
    pub(crate) use concinnity_cpu::ecs::asset_id;
}
// The source-asset lookup lives in concinnity-store, re-exported so cook code
// keeps naming it under `crate::source`.
pub(crate) use concinnity_store::source;

// Build-host API, re-exported deliberately: a host driving the pipeline (the
// CLI, an example harness) works against cook alone, the way a runtime host
// works against concinnity-engine. `paths` is the state tree cook builds into
// (anchoring it, locating `data/`); `platform` is the backend platform whose
// shader payloads cook compiles and caches.
pub use concinnity_core::platform;
pub use concinnity_store::paths;

// The world front half -- the authored model, the type vocabulary
// (`ComponentType` / `ResourceAssetType`), the typed spec vocabulary, and the
// pure semantic checks -- lives in concinnity-world; the registry is bound here
// so cook code keeps resolving `crate::registry`. cook composes its
// compile-backed checks on top (crate::check) and owns expansion (crate::world).
// Consumers reach the rest of the authoring surface through concinnity-world
// directly.
pub(crate) use concinnity_world::registry;

pub mod asset;
pub mod asset_api;
pub(crate) mod asset_impls;
pub(crate) mod audio_clip;
// Source-image format decoders (Targa, DDS, and the BCn block decompressors DDS
// needs). Build-only: they turn an authored `.tga` / `.dds` into RGBA the
// texture encoder packs; the runtime plays the compiled RGBA payload.
pub(crate) mod bcn;
pub mod blob;
pub mod cache;
pub mod character;
pub(crate) mod character_shape;
pub mod check;
pub mod color_lut;
pub mod cubemap;
pub(crate) mod dds;
pub mod environment_map;
pub mod fbx;
/// Referenced-file assets: paths and their compiled payloads.
pub mod file;
// Stat-based identity behind the cook's read memos, and the settle window that
// keeps a same-tick equal-length rewrite from being served stale.
mod file_stamp;
pub mod font;
/// Build-time mesh generators + payload compilers. The runtime-side mesh helpers
/// they share (tangents, the voxel mesher, chunk streaming) stay in
/// `concinnity_cpu::geometry`; this module re-exports what cook code names.
pub mod geometry;
pub mod glb;
pub mod gltf;
pub mod gltf_source;
/// Build-time HDR source primitives (Radiance decode, equirect->cube, cube
/// payload format) shared by the CubemapTexture + EnvironmentMap compilers.
pub mod hdr;
pub mod import;
/// KTX2 container decode: BCn block passthrough + Basis Universal (ETC1S / UASTC)
/// transcode into the tagged compressed texture payload. Build-only.
pub mod ktx2;
pub mod mesh_compile;
pub mod mesh_reimport;
/// Recognises a `.glb` that packages an environment image as a sphere you stand
/// inside, so it imports as an EnvironmentMap instead of scene geometry.
pub mod panorama;
pub(crate) mod physics_budget;
pub mod pipeline;
pub mod resource_handles;
pub(crate) mod root_motion;
pub(crate) mod scene_partition;
pub mod shader;
pub(crate) mod spawn_population;
pub mod texture;
pub mod tga;
pub mod thumbnail;
pub(crate) mod wavefront;
/// world.jsonl parsing, validation, and macro expansion.
pub mod world;

// Public build API: the entry points the CLI, the editor FFI, and the infra
// server call. The runtime-side decode + world parse API stays in
// concinnity-cpu.
pub use pipeline::TextureSourceInfo;
pub use pipeline::{
    BuildProgress, PipelineResult, build_compiled, build_compiled_with_progress, build_from_path,
    build_pipeline_from_str, validate_asset, validate_world_jsonl, write_build_outputs,
};
pub use world::prepare_world;
