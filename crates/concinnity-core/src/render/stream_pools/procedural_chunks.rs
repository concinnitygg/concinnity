//! The shipped chunk source: deterministic procedural terrain. Generation is a
//! pure function of the seed and the coordinate, so a chunk that streams out
//! and back in is regenerated identically, and no RAM or disk copy is kept.

use crate::math::SaturatingFrom;
use alloc::string::String;
use alloc::vec::Vec;

use super::decoded_mesh::DecodedMesh;
use crate::geometry::{
    ChunkBlockType, ChunkGenerator, build_chunk_impostor_mesh, build_chunk_mesh,
};
use crate::gfx::chunk_coord::ChunkCoord;
use crate::render::chunk_window::ChunkDetail;

/// A [`ChunkGenerator`] over a resolved block palette, meshing full chunks and
/// their distant impostors.
pub struct ProceduralChunks {
    generator: ChunkGenerator,
    palette: Vec<ChunkBlockType>,
    chunk_blocks: [u32; 3],
    block_size: f32,
    // Coarse-grid step (in blocks) for the distant-impostor surface.
    impostor_step: u32,
    // The surface block's atlas UVs, so impostors texture like the full chunks.
    surface_block: ChunkBlockType,
}

impl ProceduralChunks {
    /// A source for a world with the given seed, chunk dimensions, block size,
    /// resolved `BlockType` palette, and distant-impostor coarse step.
    pub fn new(
        seed: u64,
        chunk_blocks: [u32; 3],
        block_size: f32,
        palette: Vec<ChunkBlockType>,
        impostor_step: u32,
    ) -> Self {
        let generator =
            ChunkGenerator::new(seed, chunk_blocks, u32::saturating_from(palette.len()));
        // The surface block is the one the generator paints the top of each
        // column with; a degenerate single-entry palette falls back to its
        // first entry.
        let surface_idx = generator.surface_palette_index() as usize;
        let surface_block = palette
            .get(surface_idx)
            .or_else(|| palette.first())
            .copied()
            .unwrap_or(ChunkBlockType {
                solid: true,
                uv_top: [0.0, 0.0, 1.0, 1.0],
                uv_bottom: [0.0, 0.0, 1.0, 1.0],
                uv_side: [0.0, 0.0, 1.0, 1.0],
            });
        Self {
            generator,
            palette,
            chunk_blocks,
            block_size,
            impostor_step: impostor_step.max(1),
            surface_block,
        }
    }

    /// The geometry for chunk `coord` at `detail`: full voxel geometry
    /// (`ChunkDetail::Near`) or a coarse distant impostor
    /// (`ChunkDetail::Far`).
    pub fn generate(&self, coord: ChunkCoord, detail: ChunkDetail) -> Result<DecodedMesh, String> {
        let (vertices, indices) = match detail {
            ChunkDetail::Near => {
                let blocks = self.generator.generate(coord);
                build_chunk_mesh(self.chunk_blocks, self.block_size, &blocks, &self.palette)?
            }
            ChunkDetail::Far => {
                let heights = self.impostor_heights(coord);
                build_chunk_impostor_mesh(
                    self.chunk_blocks,
                    self.block_size,
                    self.impostor_step,
                    &heights,
                    self.surface_block.uv_top,
                    self.surface_block.uv_side,
                )?
            }
        };
        Ok(DecodedMesh { vertices, indices })
    }

    // The terrain surface height on the coarse impostor grid: corner (gx, gz)
    // at the world column its clamped local position maps to. Sampling by
    // world column keeps adjacent impostors watertight along their shared edge.
    fn impostor_heights(&self, coord: ChunkCoord) -> Vec<i32> {
        let step = self.impostor_step;
        let [dx, _dy, dz] = self.chunk_blocks;
        let nx = dx.div_ceil(step);
        let nz = dz.div_ceil(step);
        let base_x = coord.x * i32::saturating_from(dx);
        let base_z = coord.z * i32::saturating_from(dz);
        let mut heights = Vec::with_capacity(((nx + 1) * (nz + 1)) as usize);
        for gz in 0..=nz {
            let lz = i32::saturating_from((gz * step).min(dz));
            for gx in 0..=nx {
                let lx = i32::saturating_from((gx * step).min(dx));
                heights.push(
                    self.generator
                        .surface_height_world(base_x + lx, base_z + lz),
                );
            }
        }
        heights
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn a_chunk_and_its_cheaper_impostor_both_generate() {
        let palette = vec![
            ChunkBlockType {
                solid: false,
                uv_top: [0.0; 4],
                uv_bottom: [0.0; 4],
                uv_side: [0.0; 4],
            },
            ChunkBlockType {
                solid: true,
                uv_top: [0.0, 0.0, 1.0, 1.0],
                uv_bottom: [0.0, 0.0, 1.0, 1.0],
                uv_side: [0.0, 0.0, 1.0, 1.0],
            },
        ];
        let source = ProceduralChunks::new(42, [8, 16, 8], 1.0, palette, 4);
        let full = source
            .generate(ChunkCoord::new(0, 0), ChunkDetail::Near)
            .expect("full ok");
        assert!(!full.vertices.is_empty());
        assert!(!full.indices.is_empty());
        let impostor = source
            .generate(ChunkCoord::new(0, 0), ChunkDetail::Far)
            .expect("impostor ok");
        assert!(!impostor.vertices.is_empty());
        assert!(
            impostor.vertices.len() < full.vertices.len(),
            "impostor ({}) should be cheaper than full ({})",
            impostor.vertices.len(),
            full.vertices.len()
        );
    }
}
