// GraphicsSystem asset-streaming setup: wires the texture, normal-map, mesh,
// voxel-world and planet streaming pools onto the backend, starts their load
// threads, and parks both for StreamingSystem.

use std::collections::BTreeSet;
use std::sync::Arc;

use hashbrown::HashMap;

use concinnity_core::components::{BlockType, Planet, StreamingConfig, VoxelWorld};
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{MaterialHandle, PayloadLocator, PipelineContext};
use concinnity_core::geometry::ChunkBlockType;
use concinnity_core::gfx::render_types::DrawIndex;
use concinnity_core::gfx::{chunk_coord, mesh_seed};
use concinnity_core::planet::{LocalFrame, MAX_RESIDENT_TILES, PlanetFrame};
use concinnity_core::render::draw_list::DeferredMeshSeed;
use concinnity_core::render::grass::CanopyRange;
use concinnity_core::render::material_entry::MaterialEntry;
use concinnity_core::render::scene_members::{self, StreamedContent};
use concinnity_core::render::scene_residency::SceneResidency;
use concinnity_core::render::stream_pools::budget::{
    self, CHUNK_VRAM_FRACTION, ChunkWindowShape, MESH_VRAM_FRACTION, TEXTURE_VRAM_FRACTION,
};
use concinnity_core::render::stream_pools::{
    ChunkPool, ChunkStream, DecodedMesh, DeferredMeshPayload, PlanetTiles, ProceduralChunks,
    StreamPool, StreamingRuntime, StreamingState, TileMaterial,
};
use concinnity_core::render::world_load::stream_seed::StreamSeed;

use super::GraphicsSystem;
use super::stream_sources::{build_texture_payload_source, disk_mesh_payload};
use crate::gfx::streaming::mesh::{self, MeshPayloadSource};
use crate::gfx::streaming::{planet as tile_workers, texture};
use crate::worker::Worker;

// The streaming seed plus the payloads and voxel content the pools stream from.
pub(super) struct StreamingSetup<'a> {
    pub(super) config: Option<StreamingConfig>,
    pub(super) seed: StreamSeed,
    pub(super) texture_payloads: Vec<Vec<u8>>,
    pub(super) texture_locators: &'a [PayloadLocator],
    // Texture slots only the cook reads, which never stream in.
    pub(super) cook_only_textures: &'a BTreeSet<usize>,
    pub(super) disk_backed: bool,
    pub(super) deferred_mesh_seeds: &'a HashMap<usize, DeferredMeshSeed>,
    pub(super) voxel_world: Option<VoxelWorld>,
    pub(super) planet: Option<Planet>,
    // The color the planet's ground blends toward past its grass.
    pub(super) planet_far_field: Option<CanopyRange>,
    pub(super) block_types: &'a HashMap<AssetId, BlockType>,
    pub(super) material_map: &'a HashMap<MaterialHandle, MaterialEntry>,
}

// The mesh-streaming inputs init assembles: per-stream geometry copies,
// scoring centers, draw mapping, the seed headroom, and the deferred payload
// refs the worker decodes on demand.
pub(super) struct MeshStreamSetup {
    pub(super) payloads: Vec<DecodedMesh>,
    pub(super) centers: Vec<Vec<[f32; 3]>>,
    pub(super) draw_indices: Vec<DrawIndex>,
    pub(super) disk_backed: bool,
    pub(super) seed_region: Option<mesh_seed::MeshSeedRegion>,
    pub(super) deferred_payloads: HashMap<usize, DeferredMeshPayload>,
}

// A pool's resident-byte budget, logged.
fn describe_budget(budget: Option<u64>) -> String {
    budget.map_or_else(
        || "count-only".to_string(),
        |b| format!("{} MB", b / (1024 * 1024)),
    )
}

// A load worker, or `None` with a warning when the OS refused its thread: that
// pool is then left unstreamed.
fn started<Req: crate::worker::Job, Res>(
    pool: &str,
    worker: std::io::Result<Worker<Req, Res>>,
) -> Option<Worker<Req, Res>> {
    worker
        .map_err(|e| tracing::warn!("GraphicsSystem: {pool} streaming left off: {e}"))
        .ok()
}

impl GraphicsSystem {
    // Wire the texture, mesh and voxel-world or planet streaming pools onto
    // the backend.
    pub(super) fn setup_streaming(&mut self, setup: StreamingSetup<'_>) {
        let StreamingSetup {
            config,
            seed,
            texture_payloads,
            texture_locators,
            cook_only_textures,
            disk_backed,
            deferred_mesh_seeds,
            voxel_world,
            planet,
            planet_far_field,
            block_types,
            material_map,
        } = setup;
        self.setup_texture_streaming(
            config.clone(),
            texture_payloads,
            texture_locators,
            disk_backed,
            seed.texture_centers,
        );
        if let Some(streamer) = self.stream_pools.texture.as_mut() {
            for &slot in cook_only_textures {
                streamer.exclude(slot);
            }
        }
        // Per-stream-id payload refs for the deferred meshes, so the worker
        // can decode them from the blob payload when their scene pins.
        let deferred_payloads = concinnity_core::render::stream_pools::deferred_mesh_payloads(
            deferred_mesh_seeds,
            &seed.draw_to_handle,
            &seed.mesh.draw_indices,
            disk_mesh_payload,
        );
        if !deferred_payloads.is_empty() {
            tracing::info!(
                "GraphicsSystem: deferred {} scene-owned mesh payload(s) past init",
                deferred_payloads.len()
            );
        }
        self.setup_mesh_streaming(
            config,
            MeshStreamSetup {
                payloads: seed.mesh.payloads,
                centers: seed.mesh.centers,
                draw_indices: seed.mesh.draw_indices,
                disk_backed,
                seed_region: seed.seed_region,
                deferred_payloads,
            },
        );
        match (voxel_world, planet) {
            (Some(_), Some(_)) => tracing::error!(
                "GraphicsSystem: a world streams a VoxelWorld or a Planet, not both; the Planet is not drawn"
            ),
            (None, Some(planet)) => {
                self.setup_planet_streaming(&planet, planet_far_field, material_map)
            }
            (voxel_world, None) => {
                self.setup_voxel_world_streaming(voxel_world, block_types, material_map)
            }
        }
    }

    // Scene residency over the streaming pools.
    fn build_scene_residency(&self, ctx: &PipelineContext) -> Option<SceneResidency> {
        let content = StreamedContent {
            texture_slots: self.stream_pools.texture.as_ref().map(|t| t.len()),
            mesh_draws: self
                .stream_pools
                .mesh
                .is_some()
                .then_some(self.stream_pools.mesh_draws.as_slice()),
            shaders_stream: self
                .stream_workers
                .shaders
                .as_ref()
                .is_some_and(|w| !w.is_empty()),
            deferred_shaders: &self.deferred_shader_scenes,
        };
        scene_members::scene_residency(ctx, &self.scene_flow.as_ref()?.scenes, content)
    }

    // Hand the streaming pools built at init, with their load threads, to
    // StreamingSystem, the start scene pinned so only it and the unowned set
    // stream. The state's frame clock starts at 0 in lockstep with this
    // system's own (both tick once per world step), so eviction retire-frames
    // match the draw's frame.
    pub(super) fn publish_streaming_state(
        &mut self,
        ctx: &mut PipelineContext,
        frames_in_flight: usize,
    ) {
        self.stream_pools.scene_residency = self.build_scene_residency(ctx);
        let mut state =
            StreamingState::new(std::mem::take(&mut self.stream_pools), frames_in_flight);
        let mut workers = std::mem::take(&mut self.stream_workers);
        if let Some(start) = self.scene_flow.as_ref().map(|flow| flow.current) {
            state.pin_start_scene(start, workers.shaders.as_mut());
        }
        ctx.insert_resource(StreamingRuntime::new(state, Box::new(workers)));
    }

    pub(super) fn setup_texture_streaming(
        &mut self,
        config: Option<StreamingConfig>,
        texture_payloads: Vec<Vec<u8>>,
        texture_locators: &[PayloadLocator],
        disk_backed: bool,
        texture_centers: Vec<Vec<[f32; 3]>>,
    ) {
        let Some(config) = config else { return };
        // When disk-backed the payloads were not retained, so the streamed
        // slot count comes from the locators instead.
        let slot_count = if disk_backed {
            texture_locators.len()
        } else {
            texture_payloads.len()
        };
        if slot_count == 0 {
            return;
        }
        let Some(backend) = self.backend.as_deref_mut() else {
            return;
        };
        let source =
            match build_texture_payload_source(texture_payloads, texture_locators, disk_backed) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("GraphicsSystem: texture streaming source: {}", e);
                    return;
                }
            };
        let Some(worker) = started("texture", texture::spawn_worker(source)) else {
            return;
        };
        // Each backend's update_texture_slot rewrites whichever descriptors,
        // argument-buffers, or per-cluster SRVs sample that slot.
        for slot in 0..slot_count {
            if let Err(e) = backend.evict_texture_slot(slot) {
                tracing::warn!("GraphicsSystem: texture evict slot {}: {}", slot, e);
            }
        }
        let mut streamer = StreamPool::new(texture_centers, config.budget(), config.cap());
        let byte_budget = budget::derive_byte_budget(
            config.texture_budget_mb,
            backend.gpu_profile().memory_budget_bytes,
            TEXTURE_VRAM_FRACTION,
        );
        streamer.set_byte_budget(byte_budget);
        tracing::info!(
            "GraphicsSystem: texture streaming enabled ({} textures, {} source, budget {}/frame, cap {}, byte budget {})",
            streamer.len(),
            if disk_backed { "disk" } else { "ram" },
            config.budget(),
            config.cap(),
            describe_budget(byte_budget),
        );
        self.stream_pools.texture = Some(streamer);
        self.stream_workers.texture = Some(worker);
    }

    // Stand up the mesh-geometry streaming pool when a StreamingConfig was
    // declared. The streamer brings each streamed draw's geometry resident over
    // the next frames, nearest first.
    //
    // The payload source depends on where the world came from: a disk-backed
    // `cn run` world writes the streamed geometry to a scratch file and
    // re-reads it from there (no persistent RAM copy), an in-memory `cn debug`
    // world keeps the geometry RAM-resident.
    pub(super) fn setup_mesh_streaming(
        &mut self,
        config: Option<StreamingConfig>,
        setup: MeshStreamSetup,
    ) {
        let MeshStreamSetup {
            payloads,
            centers,
            draw_indices,
            disk_backed,
            seed_region,
            deferred_payloads,
        } = setup;
        let Some(config) = config else { return };
        if payloads.is_empty() {
            return;
        }
        let Some(backend) = self.backend.as_deref_mut() else {
            return;
        };
        let source: Arc<dyn MeshPayloadSource> = if disk_backed {
            match mesh::write_mesh_scratch(mesh::default_scratch_path(), &payloads) {
                Ok(s) => Arc::new(s),
                Err(e) => {
                    tracing::error!("GraphicsSystem: mesh streaming scratch file: {}", e);
                    return;
                }
            }
        } else {
            Arc::new(mesh::MemMeshSource::new(payloads))
        };
        // Deferred meshes have no geometry copy in the base source; their
        // fetch decodes the blob payload instead.
        let source: Arc<dyn MeshPayloadSource> = if deferred_payloads.is_empty() {
            source
        } else {
            Arc::new(mesh::SceneDeferredMeshSource::new(
                source,
                deferred_payloads,
            ))
        };
        let Some(worker) = started("mesh", mesh::spawn_worker(source)) else {
            return;
        };
        // Init residency. With a shrinkable seed the streamed geometry was
        // never baked into the buffers (compaction marked each streamed draw
        // non-resident and reserved one headroom block), so the sub-allocators
        // are seeded with that block; evicting would free the placeholder
        // offset-0 region out from under the first resident draw. Without one,
        // each streamed mesh's build-time region is freed into the
        // sub-allocators, reusable at once since nothing has drawn yet.
        match seed_region {
            Some(r) => {
                backend.seed_mesh_streaming(r.vtx_offset, r.vtx_bytes, r.idx_offset, r.idx_bytes);
            }
            None => {
                for &draw_idx in &draw_indices {
                    if let Err(e) = backend.evict_mesh(draw_idx, 0) {
                        tracing::warn!("GraphicsSystem: mesh evict draw {}: {}", draw_idx, e);
                    }
                }
            }
        }
        let mut streamer = StreamPool::new(centers, config.mesh_budget(), config.mesh_cap());
        let byte_budget = budget::derive_byte_budget(
            config.mesh_budget_mb,
            backend.gpu_profile().memory_budget_bytes,
            MESH_VRAM_FRACTION,
        );
        streamer.set_byte_budget(byte_budget);
        tracing::info!(
            "GraphicsSystem: mesh streaming enabled ({} meshes, {} source, budget {}/frame, cap {}, byte budget {})",
            streamer.len(),
            if disk_backed { "disk" } else { "ram" },
            config.mesh_budget(),
            config.mesh_cap(),
            describe_budget(byte_budget),
        );
        self.stream_pools.mesh = Some(streamer);
        self.stream_workers.mesh = Some(worker);
        self.stream_pools.mesh_draws = draw_indices;
    }

    // Stand up an infinite voxel world's chunk streaming when a VoxelWorld was
    // declared: resolve the block palette and shared material, grow the GPU
    // buffers by the chunk window's headroom, and build the chunk pool and its
    // generation thread.
    pub(super) fn setup_voxel_world_streaming(
        &mut self,
        voxel_world: Option<VoxelWorld>,
        block_types: &HashMap<AssetId, BlockType>,
        material_map: &HashMap<MaterialHandle, MaterialEntry>,
    ) {
        let Some(vw) = voxel_world else { return };

        // Each palette id is a BlockType; index 0 is air.
        let palette = ChunkBlockType::resolve_palette(&vw.palette, block_types);
        let material =
            TileMaterial::from_entry(vw.material.and_then(|id| material_map.get(&id).copied()));

        let chunk_blocks = vw.chunk_blocks();
        let (chunk_w, chunk_d) = vw.chunk_world_size();
        let (near_radius, far_radius) = (vw.view_radius(), vw.impostor_radius());
        let impostor_step = vw.impostor_step();
        let (chunk_vtx_bytes, chunk_idx_bytes) = budget::chunk_headroom(ChunkWindowShape {
            chunk_blocks,
            near_radius: near_radius as u32,
            far_radius: far_radius as u32,
            impostor_step,
        });

        let Some(backend) = self.backend.as_deref_mut() else {
            return;
        };
        let source = ProceduralChunks::new(
            vw.seed,
            chunk_blocks,
            vw.block_size(),
            palette,
            impostor_step,
        );
        let Some(worker) = started(
            "VoxelWorld chunk",
            Worker::spawn("cn-chunk-stream", 1, move |(coord, detail)| {
                (coord, source.generate(coord, detail))
            }),
        ) else {
            return;
        };
        if let Err(e) = backend.setup_chunk_streaming(chunk_vtx_bytes, chunk_idx_bytes) {
            tracing::error!("GraphicsSystem: VoxelWorld chunk streaming: {}", e);
            return;
        }

        let mut streamer =
            ChunkPool::new(near_radius, far_radius, vw.load_budget(), chunk_w, chunk_d);
        // A GPU that reports nothing leaves the pool on its radius-only policy.
        let byte_budget = budget::derive_byte_budget(
            0,
            backend.gpu_profile().memory_budget_bytes,
            CHUNK_VRAM_FRACTION,
        );
        streamer.set_byte_budget(byte_budget);
        tracing::info!(
            "GraphicsSystem: VoxelWorld streaming enabled (seed {}, {}x{}x{} blocks, near-radius {}, impostor-radius {} (step {}), budget {}/frame, byte budget {}, {} KiB chunk headroom)",
            vw.seed,
            chunk_blocks[0],
            chunk_blocks[1],
            chunk_blocks[2],
            near_radius,
            if vw.impostors_enabled() {
                far_radius
            } else {
                0
            },
            impostor_step,
            vw.load_budget(),
            describe_budget(byte_budget),
            (chunk_vtx_bytes + chunk_idx_bytes) / 1024,
        );
        self.stream_workers.chunk = Some(worker);
        self.stream_pools.chunks = Some(ChunkStream {
            pool: streamer,
            draws: Default::default(),
            chunk_w,
            chunk_d,
            // Seeded at the world origin; the first drive rebases onto the
            // camera's actual chunk before any chunk is resident, so the seed
            // never places geometry.
            origin_chunk: chunk_coord::ChunkCoord::new(0, 0),
            material,
        });
    }

    // Stand up a planet's terrain streaming: grow the GPU buffers by a chunk
    // headroom holding the most tiles a planet keeps resident, resolve the
    // planet's material, and start the tile workers.
    pub(super) fn setup_planet_streaming(
        &mut self,
        planet: &Planet,
        far_field: Option<CanopyRange>,
        material_map: &HashMap<MaterialHandle, MaterialEntry>,
    ) {
        let material = TileMaterial::for_planet(
            planet
                .material
                .and_then(|id| material_map.get(&id).copied()),
            far_field.as_ref(),
        );
        let (vertex_bytes, index_bytes) = budget::planet_tile_headroom();
        let Some(backend) = self.backend.as_deref_mut() else {
            return;
        };
        let frame = PlanetFrame {
            shape: planet.shape(),
            frame: LocalFrame::AUTHORED,
        };
        let Some(worker) = started("Planet terrain", tile_workers::spawn_workers(frame.shape))
        else {
            return;
        };
        if let Err(e) = backend.setup_chunk_streaming(vertex_bytes, index_bytes) {
            tracing::error!("GraphicsSystem: Planet terrain streaming: {e}");
            return;
        }
        tracing::info!(
            "GraphicsSystem: Planet terrain streaming enabled (radius {} m, up to {} tiles, {} KiB headroom)",
            planet.radius,
            MAX_RESIDENT_TILES,
            (vertex_bytes + index_bytes) / 1024,
        );
        self.stream_workers.planet = Some(worker);
        self.stream_pools.planet = Some(PlanetTiles::new(&frame, material));
    }
}

#[cfg(test)]
mod tests;
