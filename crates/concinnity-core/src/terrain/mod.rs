//! A terrain's cooked surface: the one square height grid its render mesh, its
//! physics collider and its grass placement are all read from, so the three
//! agree triangle for triangle.
//!
//! The grid's heights come from a generated noise field ([`noise_heights`]) or
//! a decoded heightmap image ([`heightmap_heights`]); the cook decodes images,
//! this crate only samples pixels. A grass layer's density mask rides beside
//! the grid as a [`DensityMask`]. [`payload`] is the codec the cook writes and
//! the runtime reads.

mod grid;
mod mask;
mod mesh;
pub mod payload;
mod source;

pub use grid::TerrainGrid;
pub use mask::{DensityMask, MAX_MASK_SIZE};
pub use mesh::{TERRAIN_CHUNK_CELLS, TerrainChunk, terrain_chunks, terrain_chunks_colored};
pub use source::{heightmap_heights, noise_heights};

/// Fewest grid cells along a terrain's side.
pub const MIN_TERRAIN_RESOLUTION: u32 = 4;

/// Most grid cells along a terrain's side.
pub const MAX_TERRAIN_RESOLUTION: u32 = 1024;

/// Smallest half-extent a terrain keeps, in meters.
pub const MIN_TERRAIN_EXTENT: f32 = 0.5;

/// The grid resolution an authored `resolution` builds at.
pub fn clamp_resolution(resolution: u32) -> u32 {
    resolution.clamp(MIN_TERRAIN_RESOLUTION, MAX_TERRAIN_RESOLUTION)
}
