//! Asset identity: names declared in world.jsonl are interned to an `AssetId` in
//! declaration order, and the blob and the runtime carry only the integer, so
//! every cross-reference lookup is an integer compare.
//!
//! The identity (`AssetId`) and typed reference (`AssetRef`) types themselves
//! live in the dependency-light concinnity-asset schema crate and are re-exported
//! here under the historical `crate::ecs::asset_id` paths. The interner those
//! types resolve a name through keeps a per-thread table and so belongs to the
//! std-linked crate above; concinnity-host owns it and installs it into the
//! schema crate's resolver seam. At runtime references are already integers, so
//! the seam is never consulted.

pub use concinnity_asset::{AssetId, AssetRef, de_opt_asset_ref, de_opt_asset_ref_typed};
