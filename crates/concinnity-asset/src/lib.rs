// concinnity-asset
//
// The authored-data schema for the engine's assets: the plain structs, enums,
// and serde `Default`s a world.jsonl declares, plus the identity (`AssetId`) and
// typed reference (`AssetRef<T>`) primitives they are built from.
//
// This crate holds DATA ONLY. All behavior -- the ECS `Component` impls,
// validation, companion expansion, and the name -> id interner the resolver seam
// points at -- lives in concinnity-core / concinnity-cook. The crate is
// `#![no_std]` (using only `core` + `alloc`) with serde as its single
// dependency, so it can never pull in engine logic and is consumable from doc
// tooling and external authoring tools alike.

#![no_std]

extern crate alloc;
#[cfg(test)]
extern crate std;

mod id;
mod reference;
mod resolver;

pub use id::{AssetId, de_opt_asset_ref};
pub use reference::{AssetRef, de_opt_asset_ref_typed};
pub use resolver::{ResolveFn, set_name_resolver};
