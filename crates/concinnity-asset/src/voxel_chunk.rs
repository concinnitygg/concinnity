// Voxel-chunk schema.

use crate::{AssetId, PayloadLocator};
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
/// either an empty neighbour or the outside of the chunk.
///
/// ```jsonl
/// {"name":"air","type":"BlockType","args":{"solid":false}}
/// {"name":"stone","type":"BlockType","args":{"uv_min":[0,0],"uv_max":[1,1]}}
/// {"name":"my_chunk","type":"VoxelChunk","args":{
///   "palette":["air","stone"],
///   "dim":[2,1,1],
///   "blocks":[1,1]
/// }}
/// {"name":"chunk_prop","type":"Prop","args":{"mesh":"my_chunk","material":"mat_stone"}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct VoxelChunk {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// [BlockType](#blocktype) asset names. `blocks[i]` is an index into this list.
    pub palette: Vec<AssetId>,
    /// Chunk dimensions `[dx, dy, dz]` in blocks.
    pub dim: [u32; 3],
    /// World units per block edge.
    pub block_size: f32,
    /// Flat block array, length `dx*dy*dz`. Index = `x + y*dx + z*dx*dy`.
    pub blocks: Vec<u32>,
    /// Number of level-of-detail versions to generate, including the original.
    /// `1` (the default) generates none.
    pub lod_levels: u32,
    /// Camera distances at which to switch to each lower-detail version; empty
    /// lets the build choose defaults.
    #[serde(default)]
    pub lod_distances: Vec<f32>,
    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

impl Default for VoxelChunk {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            palette: Vec::new(),
            dim: [0, 0, 0],
            block_size: 1.0,
            blocks: Vec::new(),
            lod_levels: 1,
            lod_distances: Vec::new(),
            locator: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_chunk_is_empty_with_metre_sized_blocks() {
        let c = VoxelChunk::default();
        assert_eq!(c.dim, [0, 0, 0]);
        assert_eq!(c.block_size, 1.0);
        assert!(c.blocks.is_empty());
        assert!(c.palette.is_empty());
        assert_eq!(c.lod_levels, 1);
        assert!(c.lod_distances.is_empty());
        assert!(c.locator.is_none());
    }

    #[test]
    fn an_authored_chunk_parses_and_round_trips_through_postcard() {
        crate::test_support::install_resolvers();
        let c: VoxelChunk = serde_json::from_str(
            r#"{"palette":["air","stone"],"dim":[2,1,2],"block_size":0.5,
                "blocks":[0,1,1,0],"lod_levels":2,"lod_distances":[16]}"#,
        )
        .unwrap();
        assert_eq!(c.palette, [AssetId(3), AssetId(5)]);
        // The block list indexes the palette, one entry per cell in `dim`.
        assert_eq!(c.blocks.len() as u32, c.dim[0] * c.dim[1] * c.dim[2]);

        let bytes = postcard::to_allocvec(&c).unwrap();
        let back: VoxelChunk = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.dim, [2, 1, 2]);
        assert_eq!(back.block_size, 0.5);
        assert_eq!(back.blocks, [0, 1, 1, 0]);
        assert_eq!(back.lod_levels, 2);
        assert_eq!(back.lod_distances, [16.0]);
        assert_eq!(back.asset_id, AssetId::default());
    }
}
