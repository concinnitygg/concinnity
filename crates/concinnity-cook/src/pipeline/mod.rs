//! Compile stage of the build pipeline. The world is loaded, expanded, and
//! validated upstream by crate::build_only::prepare_world; this module takes the
//! resulting WorldJsonlAsset list and:
//! - Resolves each asset to a BlobAssetDef via asset_api::create_asset_def()
//! - Compiles payloads for assets that need compilation
//! - Packs all payloads into blobs using PayloadPacker (fills locators)
//! - Sorts: components first, then systems in declared order
//!
//! The submodules follow that order: `desugar` rewrites source-backed assets
//! into inline data, `scene_refs` bakes the naming-convention references, `pack`
//! compiles and packs the payloads through `dispatch`, and `entry` is the
//! sequence the public entry points drive. `validate` is the compile-free half,
//! for callers that only want the checks.

mod desugar;
mod dispatch;
mod entry;
mod pack;
mod result;
mod scene_refs;
mod validate;

#[cfg(test)]
mod fixtures;

pub use entry::{
    BuildProgress, build_compiled, build_compiled_with_progress, build_from_path,
    build_pipeline_from_str, write_blobs_to, write_build_outputs,
};
pub use result::{MeshSourceInfo, PipelineResult, TextureSourceInfo};
pub use validate::{validate_asset, validate_world_jsonl};

// The mesh kinds' declarable type names. Both are resource assets (no
// `Component` impl, so no `::NAME` const); the desugar passes and the cache
// probe match on these.
const MESH_TYPE: &str = "Mesh";
const SKINNED_MESH_TYPE: &str = "SkinnedMesh";

// Collapse a list of validation errors into a single io::Error. The messages
// are newline-joined so an upstream caller (e.g. the infra agentic loop) sees
// every problem from one call.
fn errors_to_io(errors: Vec<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, errors.join("\n"))
}
