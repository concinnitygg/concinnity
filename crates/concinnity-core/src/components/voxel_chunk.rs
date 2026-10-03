// Voxel-chunk schema.

use crate::components::BlockType;
use crate::ecs::PayloadLocator;
use crate::ecs::Ref;
use alloc::vec::Vec;

/// A voxel grid that compiles into a single mesh.
///
/// A dense grid of blocks compiled into a single mesh at build time. Use one
/// chunk per region of a voxel/Minecraft-style world; reference it from a
/// [Prop](#prop)'s `mesh` field. Hidden faces between two solid blocks are
/// dropped, so a fully filled chunk contributes zero triangles to its interior.
///
/// The palette must contain at least one entry whose [BlockType](#blocktype) has
/// `solid: false` (typically named `air`); cells whose palette entry is
/// non-solid emit no faces. Faces are only emitted between a solid block and
/// either an empty neighbor or the outside of the chunk.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct VoxelChunk {
    /// [BlockType](#blocktype) asset names. `blocks[i]` is an index into this list.
    pub palette: Vec<Ref<BlockType>>,
    /// Chunk dimensions `[dx, dy, dz]` in blocks.
    pub dim: [u32; 3],
    /// World units per block edge.
    #[asset(default = 1.0)]
    pub block_size: f32,
    /// Flat block array, length `dx*dy*dz`. Index = `x + y*dx + z*dx*dy`.
    pub blocks: Vec<u32>,
    /// Number of level-of-detail versions to generate, including the original.
    /// `1` (the default) generates none.
    #[asset(default = 1)]
    pub lod_levels: u32,
    /// Camera distances at which to switch to each lower-detail version; empty
    /// lets the build choose defaults.
    pub lod_distances: Vec<f32>,
    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}
