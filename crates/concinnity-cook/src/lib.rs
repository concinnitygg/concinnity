//! concinnity-cook: the build side, kept out of the runtime foundation so that
//! foundation carries none of the build-only dependencies (fbxcel, shaderc,
//! sha2, kira). This crate turns world.jsonl + source files into the binary
//! blobs the runtime reads; it depends on concinnity-core and core has no edge
//! back into it.
//!
//! The stages read in order: `authoring` is the authored model and the type
//! vocabulary it is written in, `build_only` expands the types that never reach
//! a blob, `check` validates the expanded world, and the compile path turns
//! what is left into payloads. That path is three groups plus the orchestration
//! that sequences them: `import` reads the artist-supplied source files,
//! `codec` decodes the container and image formats they carry, `compile` turns
//! an asset's args and sources into its payload, and `pipeline` / `blob` /
//! `cache` at the root drive the whole run.
//!
//! Bridge: the vocabulary and compute modules below are re-exported crate-wide
//! so code moved here keeps resolving its `crate::{components,ecs,gfx,result}`
//! paths. `crate::components` is the runtime half only; the authoring-only
//! types this crate expands away are named from
//! `crate::authoring::registry::build_only` where they are used, so a use site
//! says which half it works on. The payload *decoders* and shared payload types
//! live in `concinnity_core::bake`; this crate's modules call back into them.
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

pub(crate) use concinnity_core::gfx;
pub(crate) use concinnity_core::{components, result};

// The vocabulary's ECS surface, with the build-time name interner shadowing its
// `asset_id`: the interner keeps a per-thread table, so it lives in
// `concinnity_host::thread` and re-exports the vocabulary's `AssetId` /
// `AssetRef`.
pub(crate) mod ecs {
    pub(crate) use concinnity_core::ecs::*;
    pub(crate) use concinnity_host::thread::asset_id;
}
// The source-asset lookup lives in `concinnity_host::store`, re-exported so
// cook code keeps naming it under `crate::source`. It resolves against a
// directory its caller supplies; every build threads that root down from its
// entry point rather than reading one for itself.
pub(crate) use concinnity_host::store::source;

// Build-host API, re-exported deliberately: a host driving the pipeline (the
// CLI, an example harness) works against cook alone, the way a runtime host
// works against concinnity-engine. `paths` is the state tree cook builds into
// (anchoring it, locating `data/`, and naming the `assets/` a host passes as a
// build's search root); `platform` is the shader-platform vocabulary a host
// names the backend it cooks for with.
pub use concinnity_core::platform;
pub use concinnity_host::store::paths;

// The authoring type vocabulary, bound at the root so the compile path keeps
// resolving `crate::registry` without naming the namespace it belongs to.
pub(crate) use authoring::registry;

pub(crate) mod asset;
pub mod asset_api;
pub(crate) mod asset_impls;
/// The authored world model: world.jsonl parsing and I/O, the type vocabulary,
/// the asset cross-reference metadata, the typed spec vocabulary, and the
/// build-only args schemas. Everything the cook reads before it compiles
/// anything.
pub mod authoring;
pub mod blob;
/// The build-time expansion passes: one module per build-only asset type, plus
/// preset loading and the build front-half orchestrator.
pub mod build_only;
pub mod cache;
pub mod check;
// Container and image format decoders, shared by the compilers that encode
// their output.
pub(crate) mod codec;
/// The compile path: per-asset payload compilers plus the whole-world passes
/// that run with them.
pub mod compile;
// Stat-based identity behind the cook's read memos, and the settle window that
// keeps a same-tick equal-length rewrite from being served stale.
mod file_stamp;
/// Source-file readers: the scene expansion and the container readers it
/// dispatches into.
pub mod import;
pub mod pipeline;
pub mod resource_handles;

// Public build API: the entry points the CLI, the editor FFI, and the infra
// server call. The runtime-side decode API stays in concinnity-core.
pub use build_only::prepare_world;
pub use pipeline::{
    BuildProgress, PipelineResult, build_compiled, build_compiled_with_progress, build_from_path,
    build_pipeline_from_str, validate_asset, validate_world_jsonl, write_build_outputs,
};
