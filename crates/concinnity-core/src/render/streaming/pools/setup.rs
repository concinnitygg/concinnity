//! What a world's graphics init stands up for streaming: which chunk content
//! it streams, how many texture slots stream, and each pool sized from the world's settings and the device's memory.
//! The host builds the payload sources and load threads and makes the device
//! calls.

use crate::math::SaturatingFrom;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use hashbrown::HashMap;

use super::budget::{
    self, CHUNK_VRAM_FRACTION, ChunkWindowShape, MESH_VRAM_FRACTION, TEXTURE_VRAM_FRACTION,
};
use super::{
    ChunkPool, ChunkStream, PlanetTiles, ProceduralChunks, StreamPool, StreamedItem, TileMaterial,
};
use crate::components::{BlockType, Planet, StreamingConfig, VoxelWorld};
use crate::ecs::MaterialHandle;
use crate::ecs::asset_id::AssetId;
use crate::geometry::ChunkBlockType;
use crate::gfx::chunk_coord::ChunkCoord;
use crate::planet::{LocalFrame, PlanetFrame};
use crate::render::environment::grass::CanopyRange;
use crate::render::material_entry::MaterialEntry;

/// The chunk content a world streams: a `VoxelWorld` or a Planet, never both.
#[derive(Debug, Clone)]
pub enum ChunkSource {
    /// No chunk streaming.
    None,
    /// An infinite voxel world's chunks.
    VoxelWorld(VoxelWorld),
    /// A planet's terrain tiles.
    Planet(Planet),
}

impl ChunkSource {
    /// The source a world declaring `voxel_world` and `planet` streams. A world
    /// declaring both streams neither.
    pub fn choose(voxel_world: Option<VoxelWorld>, planet: Option<Planet>) -> Self {
        match (voxel_world, planet) {
            (Some(_), Some(_)) => {
                tracing::error!(
                    "graphics: a world streams a VoxelWorld or a Planet, not both; neither is drawn"
                );
                Self::None
            }
            (Some(voxel_world), None) => Self::VoxelWorld(voxel_world),
            (None, Some(planet)) => Self::Planet(planet),
            (None, None) => Self::None,
        }
    }
}

/// How many texture slots stream: one per compiled texture, counted from the
/// payload locators when the payloads stay on disk. `None` with no texture.
pub fn streamed_texture_slots(
    disk_backed: bool,
    payloads: usize,
    locators: usize,
) -> Option<usize> {
    let slots = if disk_backed { locators } else { payloads };
    (slots > 0).then_some(slots)
}

impl<T: StreamedItem> StreamPool<T> {
    /// The texture pool `config` asks for, its byte budget the authored
    /// override or a share of `device_memory_bytes`.
    pub fn for_textures(
        config: &StreamingConfig,
        centers: Vec<Vec<[f32; 3]>>,
        device_memory_bytes: u64,
    ) -> Self {
        let mut pool = Self::new(centers, config.budget(), config.cap());
        pool.set_byte_budget(budget::derive_byte_budget(
            config.texture_budget_mb,
            device_memory_bytes,
            TEXTURE_VRAM_FRACTION,
        ));
        pool
    }

    /// The mesh pool `config` asks for, its byte budget the authored override
    /// or a share of `device_memory_bytes`.
    pub fn for_meshes(
        config: &StreamingConfig,
        centers: Vec<Vec<[f32; 3]>>,
        device_memory_bytes: u64,
    ) -> Self {
        let mut pool = Self::new(centers, config.mesh_budget(), config.mesh_cap());
        pool.set_byte_budget(budget::derive_byte_budget(
            config.mesh_budget_mb,
            device_memory_bytes,
            MESH_VRAM_FRACTION,
        ));
        pool
    }
}

/// A `VoxelWorld`'s chunk streaming: the pool, the generator its load thread
/// runs and the vertex / index headroom its chunk window needs.
pub struct VoxelStream {
    /// The chunk pool with its draw bookkeeping.
    pub stream: ChunkStream,
    /// The chunk generator.
    pub source: ProceduralChunks,
    /// Vertex and index bytes the shared buffers grow by.
    pub headroom: (usize, usize),
}

impl VoxelStream {
    /// The streaming `voxel_world` asks for. Each palette id is a `BlockType`
    /// (index 0 is air); a `VoxelWorld` takes no byte-budget override, so its
    /// budget is a share of `device_memory_bytes`.
    pub fn plan(
        voxel_world: &VoxelWorld,
        block_types: &HashMap<AssetId, BlockType>,
        material_map: &HashMap<MaterialHandle, MaterialEntry>,
        device_memory_bytes: u64,
    ) -> Self {
        let palette = ChunkBlockType::resolve_palette(&voxel_world.palette, block_types);
        let material = TileMaterial::from_entry(
            voxel_world
                .material
                .and_then(|id| material_map.get(&id).copied()),
        );
        let chunk_blocks = voxel_world.chunk_blocks();
        let (chunk_w, chunk_d) = voxel_world.chunk_world_size();
        let (near_radius, far_radius) = (voxel_world.view_radius(), voxel_world.impostor_radius());
        let impostor_step = voxel_world.impostor_step();
        let headroom = budget::chunk_headroom(ChunkWindowShape {
            chunk_blocks,
            near_radius: u32::saturating_from(near_radius),
            far_radius: u32::saturating_from(far_radius),
            impostor_step,
        });
        let source = ProceduralChunks::new(
            voxel_world.seed,
            chunk_blocks,
            voxel_world.block_size(),
            palette,
            impostor_step,
        );
        let mut pool = ChunkPool::new(
            near_radius,
            far_radius,
            voxel_world.load_budget(),
            chunk_w,
            chunk_d,
        );
        pool.set_byte_budget(budget::derive_byte_budget(
            0,
            device_memory_bytes,
            CHUNK_VRAM_FRACTION,
        ));
        Self {
            stream: ChunkStream {
                pool,
                draws: BTreeMap::default(),
                chunk_w,
                chunk_d,
                // Seeded at the world origin; the first drive rebases onto the
                // camera's actual chunk before any chunk is resident.
                origin_chunk: ChunkCoord::new(0, 0),
                material,
            },
            source,
            headroom,
        }
    }
}

/// A planet's terrain streaming: the frame its tiles are built in, what they
/// draw with and the vertex / index headroom for the most tiles a planet keeps
/// resident.
#[derive(Clone, Copy)]
pub struct PlanetStream {
    /// The planet's shape in its authored frame.
    pub frame: PlanetFrame,
    /// What every tile draws with.
    pub material: TileMaterial,
    /// Vertex and index bytes the shared buffers grow by.
    pub headroom: (usize, usize),
}

impl PlanetStream {
    /// The streaming `planet` asks for, its ground blending toward `far_field`
    /// past its grass.
    pub fn plan(
        planet: &Planet,
        far_field: Option<&CanopyRange>,
        material_map: &HashMap<MaterialHandle, MaterialEntry>,
    ) -> Self {
        Self {
            frame: PlanetFrame {
                shape: planet.shape(),
                frame: LocalFrame::AUTHORED,
            },
            material: TileMaterial::for_planet(
                planet
                    .material
                    .and_then(|id| material_map.get(&id).copied()),
                far_field,
            ),
            headroom: budget::planet_tile_headroom(),
        }
    }

    /// The tile pool.
    pub fn tiles(&self) -> PlanetTiles {
        PlanetTiles::new(&self.frame, self.material)
    }
}

#[cfg(test)]
mod tests;
