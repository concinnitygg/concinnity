//! The authored world model: the input side of the cook.
//!
//! Owns world.jsonl parsing and I/O plus structural validation
//! (`world`), the authoring type vocabulary (`registry`, whose
//! `RegisteredType` is instantiated from the shared registry list in
//! concinnity-core), the asset cross-reference metadata (`refs`,
//! `resource_type`), the typed authoring vocabulary (`spec`) and the world
//! templates built from it (`template`), and the build-only args schemas
//! (`schema`). Everything here operates on the authored input; the shipped
//! runtime plays compiled blobs and never links this crate.
//!
//! The authored medium is JSON (an asset's `args` is its public JSON schema),
//! so serde_json is intrinsic to this module's job; typed struct-first
//! authoring is served by `spec`'s `AssetSpec` builders, converted to world
//! entries by `spec::json`.
//!
//! Semantic validation of an authored world lives one level up, in
//! `crate::check`, alongside the compile-backed checks that need this crate's
//! compilers.

pub mod refs;
pub mod registry;
pub(crate) mod resource_type;
pub(crate) mod schema;
pub(crate) mod source_args;
pub mod spec;
pub mod template;
pub mod validate;
pub mod world;
