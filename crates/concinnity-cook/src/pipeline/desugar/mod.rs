//! Rewrites source-backed assets into the inline data the compile pass expects:
//! a `.glb` or `.fbx` reference becomes vertices, a skeleton, or animation
//! tracks in the asset's own args. Runs before anything else reads them.

mod animation;
mod fbx;
mod gltf;

#[cfg(test)]
mod fixtures;

use crate::authoring::world::WorldJsonlAsset;

pub(in crate::pipeline) use animation::{desugar_animation_imports, desugar_root_motion};
pub(in crate::pipeline) use fbx::{desugar_fbx_meshes, desugar_fbx_skinned_meshes};
pub(in crate::pipeline) use gltf::{desugar_gltf_meshes, desugar_gltf_skinned_meshes};

// Which skinned mesh of the asset's source file it selects; absent means the
// file's first.
fn skin_index_arg(asset: &WorldJsonlAsset) -> u32 {
    asset
        .args
        .get("skin_index")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32
}
