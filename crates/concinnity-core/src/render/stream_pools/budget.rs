//! What the streaming pools may hold: each pool's resident-byte budget, and the
//! device headroom and draw records a voxel world's chunk window needs.

use crate::gfx::mesh_payload::Vertex;
use crate::render::chunk_window::{max_full_detail_chunks, max_resident_chunks};

/// Default share of the GPU's reported memory the texture pool may hold.
pub const TEXTURE_VRAM_FRACTION: f64 = 0.50;
/// Default share of the GPU's reported memory the mesh pool may hold.
pub const MESH_VRAM_FRACTION: f64 = 0.30;
/// Share of the GPU's reported memory a voxel world's chunks may hold. Chunk
/// streaming takes no user override, so this is always the budget.
pub const CHUNK_VRAM_FRACTION: f64 = 0.15;

// The most draw records a chunk window reserves, so a typo radius cannot
// demand gigabytes of record memory.
const MAX_CHUNK_RECORDS: u64 = 65536;

// The most device memory a chunk window's geometry headroom takes, on each of
// the vertex and index buffers.
const MAX_CHUNK_HEADROOM: u64 = 512 * 1024 * 1024;

/// A pool's resident-byte budget: a non-zero `override_mb` in mebibytes, or
/// else `fraction` of the GPU's reported memory. `None` (count-only
/// residency) when neither is known.
pub fn derive_byte_budget(override_mb: u32, gpu_memory_bytes: u64, fraction: f64) -> Option<u64> {
    if override_mb != 0 {
        return Some(u64::from(override_mb) * 1024 * 1024);
    }
    if gpu_memory_bytes == 0 {
        return None;
    }
    Some((gpu_memory_bytes as f64 * fraction) as u64)
}

/// The draw records a chunk window reaching `far_radius` reserves: one for
/// every chunk it can hold resident, capped.
pub fn chunk_reserve_count(far_radius: u32) -> usize {
    max_resident_chunks(far_radius).min(MAX_CHUNK_RECORDS) as usize
}

/// The shape of a voxel world's chunk window, which sizes its headroom.
#[derive(Debug, Clone, Copy)]
pub struct ChunkWindowShape {
    /// Blocks along a chunk's X, Y and Z.
    pub chunk_blocks: [u32; 3],
    /// Full-detail chunks reach this far.
    pub near_radius: u32,
    /// Impostors reach this far.
    pub far_radius: u32,
    /// The impostor's coarse cell, in blocks.
    pub impostor_step: u32,
}

/// The vertex and index bytes a chunk window's worst-case residency takes:
/// full-detail meshes over the near band, impostors over the rest, each
/// buffer capped.
pub fn chunk_headroom(shape: ChunkWindowShape) -> (usize, usize) {
    let near_chunks = max_full_detail_chunks(shape.near_radius);
    let far_chunks = max_resident_chunks(shape.far_radius).saturating_sub(near_chunks);
    let vertex = core::mem::size_of::<Vertex>() as u64;
    let index = core::mem::size_of::<u32>() as u64;
    let [bx, _, bz] = shape.chunk_blocks.map(u64::from);

    // A generous face count for rolling terrain; an over-budget chunk fails
    // its add rather than overrunning the buffer. Indices widen to u32.
    let faces = bx * bz * 4;
    let full_vtx = (faces * 4).min(u64::from(u16::MAX)) * vertex;
    let full_idx = faces * 6 * index;

    // One quad per coarse cell plus a perimeter skirt.
    let step = u64::from(shape.impostor_step.max(1));
    let (nx, nz) = (bx.div_ceil(step), bz.div_ceil(step));
    let quads = nx * nz + 2 * (nx + nz);
    let impostor_vtx = quads * 4 * vertex;
    let impostor_idx = quads * 6 * index;

    let vtx = (near_chunks * full_vtx + far_chunks * impostor_vtx).min(MAX_CHUNK_HEADROOM);
    let idx = (near_chunks * full_idx + far_chunks * impostor_idx).min(MAX_CHUNK_HEADROOM);
    (vtx as usize, idx as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;
    const VRAM: u64 = 8 * 1024 * MIB;

    #[test]
    fn an_override_wins_over_the_gpu_memory() {
        assert_eq!(
            derive_byte_budget(256, VRAM, TEXTURE_VRAM_FRACTION),
            Some(256 * MIB)
        );
        assert_eq!(
            derive_byte_budget(64, 0, MESH_VRAM_FRACTION),
            Some(64 * MIB)
        );
    }

    #[test]
    fn without_an_override_the_budget_is_a_share_of_the_gpu_memory() {
        assert_eq!(
            derive_byte_budget(0, VRAM, TEXTURE_VRAM_FRACTION),
            Some(VRAM / 2)
        );
        assert_eq!(
            derive_byte_budget(0, VRAM, CHUNK_VRAM_FRACTION),
            Some((VRAM as f64 * 0.15) as u64)
        );
        assert_eq!(derive_byte_budget(0, 0, MESH_VRAM_FRACTION), None);
    }

    // Every chunk the window can hold gets a record, until the cap.
    #[test]
    fn the_chunk_reserve_covers_the_resident_window() {
        for far in [0, 2, 5, 10] {
            assert_eq!(chunk_reserve_count(far) as u64, max_resident_chunks(far));
        }
        assert_eq!(chunk_reserve_count(500) as u64, MAX_CHUNK_RECORDS);
    }

    // Impostors cost less headroom than full chunks over the same window, and
    // a huge radius is capped.
    #[test]
    fn the_headroom_sizes_each_band_and_is_capped() {
        let shape = |near, far| ChunkWindowShape {
            chunk_blocks: [16, 64, 16],
            near_radius: near,
            far_radius: far,
            impostor_step: 4,
        };
        let full = chunk_headroom(shape(4, 4));
        let banded = chunk_headroom(shape(1, 4));
        assert!(banded.0 < full.0 && banded.1 < full.1);
        assert_eq!(
            chunk_headroom(shape(200, 200)),
            (MAX_CHUNK_HEADROOM as usize, MAX_CHUNK_HEADROOM as usize)
        );
    }
}
