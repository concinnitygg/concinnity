// concinnity-asset
//
// The authored-data schema for the engine's assets: the plain structs, enums,
// and serde `Default`s a world.jsonl declares, plus the identity (`AssetId`) and
// typed reference (`AssetRef<T>`) primitives they are built from.
//
// This crate holds DATA ONLY. All behavior -- the ECS `Component` impls,
// validation, companion expansion, and the cook-time name -> id resolution --
// lives in concinnity-core / concinnity-cook. The crate is `#![no_std]` (using
// only `core` + `alloc`) with serde as its single dependency, so it can never
// pull in engine logic and is consumable from doc tooling and external authoring
// tools alike.

#![no_std]

extern crate alloc;
#[cfg(test)]
extern crate std;

mod id;
mod reference;

pub use id::AssetId;
pub use reference::{AssetRef, de_opt_asset_ref};
