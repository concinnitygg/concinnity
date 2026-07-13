// Infinite procedurally generated voxel world schema.

use crate::{AssetId, MaterialHandle, de_opt_material_handle};
use alloc::vec::Vec;

/// An infinite, procedurally generated voxel world.
///
/// Where a [VoxelChunk](#voxelchunk) is one authored chunk compiled to a fixed
/// mesh at build time, a `VoxelWorld` describes an *unbounded* world: chunks are
/// generated on demand from `seed` as the camera moves and streamed in and out
/// around it. The grid is infinite on X/Z and a single chunk tall on Y.
/// Declaring one opts the world into chunk streaming; with no `VoxelWorld`
/// present nothing changes.
///
/// The `palette` lists [BlockType](#blocktype) assets; the generator uses index
/// 0 as air, index 1 as the surface block, and index 2 (when present) as the
/// subsurface block. `material` supplies the textures and lighting shared by
/// every chunk.
///
/// ```jsonl
/// {"name":"air","type":"BlockType","args":{"solid":false}}
/// {"name":"grass","type":"BlockType","args":{"uv_min":[0,0],"uv_max":[1,1]}}
/// {"name":"stone","type":"BlockType","args":{"uv_min":[0,0],"uv_max":[1,1]}}
/// {"name":"overworld","type":"VoxelWorld","args":{
///   "seed":42,"view_radius":6,"palette":["air","grass","stone"],"material":"mat_ground"
/// }}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct VoxelWorld {
    /// Deterministic terrain seed. The same seed always generates the same
    /// world, so a chunk regenerates identically each time it streams back in.
    pub seed: u64,
    /// Blocks per chunk `[dx, dy, dz]`. Y is the world's fixed vertical extent.
    pub chunk_blocks: [u32; 3],
    /// World units per block edge.
    pub block_size: f32,
    /// Chunk radius streamed around the camera at full voxel detail.
    pub view_radius: u32,
    /// Outer chunk radius streamed as cheap coarse impostors. Chunks farther
    /// than `view_radius` but within `impostor_radius` render as a low-detail
    /// surface mesh instead of full voxel geometry. `0` (the default) or any
    /// value `<= view_radius` disables impostors.
    pub impostor_radius: u32,
    /// Coarse-grid step (in blocks) for distant-chunk impostors: the surface is
    /// sampled every `impostor_step` blocks. Higher = cheaper and coarser.
    pub impostor_step: u32,
    /// Maximum number of chunks generated and loaded per frame.
    pub load_budget: u32,
    /// [BlockType](#blocktype) asset names. Index 0 is air; 1 is the surface
    /// block; 2, when present, is the subsurface block.
    pub palette: Vec<AssetId>,
    /// [Material](#material) shared by every chunk: textures and lighting.
    #[serde(deserialize_with = "de_opt_material_handle")]
    pub material: Option<MaterialHandle>,
}

impl Default for VoxelWorld {
    fn default() -> Self {
        Self {
            seed: 0,
            chunk_blocks: [16, 24, 16],
            block_size: 1.0,
            view_radius: 5,
            impostor_radius: 0,
            impostor_step: 4,
            load_budget: 3,
            palette: Vec::new(),
            material: None,
        }
    }
}

// These accessors feed the Metal chunk-streaming path for now
// (Vulkan / DirectX catch-up is a follow-up).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
impl VoxelWorld {
    /// Blocks per chunk, each axis floored at 1 so a chunk is never degenerate.
    pub fn chunk_blocks(&self) -> [u32; 3] {
        [
            self.chunk_blocks[0].max(1),
            self.chunk_blocks[1].max(1),
            self.chunk_blocks[2].max(1),
        ]
    }

    /// World units per block edge, floored at a small positive value.
    pub fn block_size(&self) -> f32 {
        self.block_size.max(0.01)
    }

    /// World-space `(x, z)` size of one chunk.
    pub fn chunk_world_size(&self) -> (f32, f32) {
        let b = self.chunk_blocks();
        let s = self.block_size();
        (b[0] as f32 * s, b[2] as f32 * s)
    }

    /// View radius in chunks, floored at 0 and capped so a typo cannot ask for
    /// a multi-thousand-chunk window.
    pub fn view_radius(&self) -> i32 {
        (self.view_radius as i32).clamp(0, 32)
    }

    /// Effective impostor (far) radius in chunks. Capped well above the
    /// full-detail cap since impostors are cheap, and floored at `view_radius`
    /// (a smaller value disables impostors, there is no far band to fill).
    pub fn impostor_radius(&self) -> i32 {
        (self.impostor_radius as i32)
            .clamp(0, 96)
            .max(self.view_radius())
    }

    /// Coarse-grid step in blocks for distant impostors, floored at 1 and
    /// capped so a typo cannot collapse the whole surface to a single quad on a
    /// huge chunk (still valid, just degenerate).
    pub fn impostor_step(&self) -> u32 {
        self.impostor_step.clamp(1, 64)
    }

    /// Whether the distant-impostor far band is active: an impostor radius
    /// strictly beyond the full-detail radius.
    pub fn impostors_enabled(&self) -> bool {
        self.impostor_radius() > self.view_radius()
    }

    /// Per-frame chunk load budget as a `usize`, floored at 1 so a stray 0
    /// cannot wedge streaming permanently.
    pub fn load_budget(&self) -> usize {
        (self.load_budget as usize).max(1)
    }
}
