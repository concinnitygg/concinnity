// concinnity-world: the build-side world front half.
//
// Owns the authored world model (world.jsonl parsing and I/O, structural
// validation), the authoring type vocabulary (`ComponentType` and
// `ResourceAssetType`, both instantiated from the shared registry lists in
// concinnity-core), the asset cross-reference metadata, semantic validation
// (`check`), and the typed-spec bridge (`template_spec`). Everything here
// operates on the authored input; the shipped runtime plays compiled blobs and
// never links this crate.
//
// Deliberately excluded: build-time expansion (SceneImport, prefabs, injection
// passes), importers/encoders, payload compilation, and blob writing -- all of
// that is concinnity-cook, which sits on top of this crate and composes its
// compile-backed per-asset checks through `check::check_world_with`.
//
// The authored medium is JSON (an asset's `args` is its public JSON schema),
// so serde_json is intrinsic to this crate's job; typed struct-first authoring
// is served by concinnity-templates' `AssetSpec` builders, converted to world
// entries at this crate's edge (`template_spec`).

// Bridge: re-export the core modules the moved code names under crate::* so
// its `crate::<module>` import paths resolve unchanged.
#[allow(unused_imports)]
pub(crate) use concinnity_core::{assets, build, ecs, paths, result};

pub mod check;
pub mod registry;
pub mod resource_type;
pub mod source_args;
pub mod template_spec;
pub mod validate;
pub mod world;
