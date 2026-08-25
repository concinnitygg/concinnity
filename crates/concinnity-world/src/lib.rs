//! concinnity-world: the build-side world front half.
//!
//! Owns the authored world model (world.jsonl parsing and I/O, structural
//! validation), the authoring type vocabulary (`RegisteredType`, instantiated
//! from the shared registry list in concinnity-core), the asset
//! cross-reference metadata, semantic validation
//! (`check`), the typed authoring vocabulary (`spec`), and the world templates
//! built from it (`template`). Everything here operates on the authored input;
//! the shipped runtime plays compiled blobs and never links this crate.
//!
//! Deliberately excluded: build-time expansion (SceneImport, prefabs, injection
//! passes), importers/encoders, payload compilation, and blob writing -- all of
//! that is concinnity-cook, which sits on top of this crate and composes its
//! compile-backed per-asset checks through `check::check_world_with`.
//!
//! The authored medium is JSON (an asset's `args` is its public JSON schema), so
//! serde_json is intrinsic to this crate's job; typed struct-first authoring is
//! served by `spec`'s `AssetSpec` builders, converted to world entries by
//! `spec::json`.

// Bridge: re-export the modules the moved code names under crate::* so its
// `crate::<module>` import paths resolve unchanged.
pub(crate) use concinnity_core::{components, platform, result};

// The vocabulary's ECS surface, with the build-time name interner shadowing its
// `asset_id`: the interner keeps a per-thread table, so it lives in
// concinnity-cpu and re-exports the vocabulary's `AssetId` / `AssetRef`.
pub(crate) mod ecs {
    pub(crate) use concinnity_core::ecs::*;
    pub(crate) use concinnity_cpu::ecs::asset_id;
}
pub(crate) use concinnity_store::paths;

pub mod check;
pub mod refs;
pub mod registry;
pub mod resource_type;
pub mod source_args;
pub mod spec;
pub mod template;
pub mod validate;
pub mod world;
