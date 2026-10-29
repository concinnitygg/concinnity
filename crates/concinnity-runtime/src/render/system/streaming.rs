// GraphicsSystem asset-streaming setup: wires the texture, normal-map, mesh,
// voxel-world and planet streaming pools onto the backend, starts their load
// threads, and parks both for StreamingSystem.

use std::collections::BTreeSet;
use std::sync::Arc;

use hashbrown::HashMap;

use concinnity_engine::components::{BlockType, Planet, StreamingConfig, VoxelWorld};
use concinnity_engine::ecs::asset_id::AssetId;
use concinnity_engine::ecs::{MaterialHandle, PayloadLocator, PipelineContext};
use concinnity_engine::gfx::render_types::DrawIndex;
use concinnity_engine::planet::MAX_RESIDENT_TILES;
use concinnity_engine::render::draw_list::DeferredMeshSeed;
use concinnity_engine::render::environment::grass::CanopyRange;
use concinnity_engine::render::material_entry::MaterialEntry;
use concinnity_engine::render::streaming::mesh_seed;
use concinnity_engine::render::streaming::pools::{
    ChunkSource, DecodedMesh, DeferredMeshPayload, PlanetStream, StreamPool, StreamingRuntime,
    StreamingState, VoxelStream, streamed_texture_slots,
};
use concinnity_engine::render::streaming::scene_members::{self, StreamedContent};
use concinnity_engine::render::streaming::scene_residency::SceneResidency;
use concinnity_engine::render::world_load::stream_seed::StreamSeed;

use super::GraphicsSystem;
use crate::render::streaming::mesh::{self, MeshPayloadSource};
use crate::render::streaming::sources::{build_texture_payload_source, disk_mesh_payload};
use crate::render::streaming::{planet as tile_workers, texture};
use concinnity_host::thread::worker::Worker;

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
fn started<Req, Res>(
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
        let deferred_payloads = concinnity_engine::render::streaming::pools::deferred_mesh_payloads(
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
        match ChunkSource::choose(voxel_world, planet) {
            ChunkSource::None => {}
            ChunkSource::VoxelWorld(vw) => {
                self.setup_voxel_world_streaming(&vw, block_types, material_map);
            }
            ChunkSource::Planet(planet) => {
                self.setup_planet_streaming(&planet, planet_far_field, material_map);
            }
        }
    }

    // Scene residency over the streaming pools.
    fn build_scene_residency(&self, ctx: &PipelineContext<'_>) -> Option<SceneResidency> {
        let content = StreamedContent {
            texture_slots: self.stream_pools.texture.as_ref().map(StreamPool::len),
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
            deferred_shaders: self.shader_deferral.scenes(),
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
        ctx: &mut PipelineContext<'_>,
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
        let Some(slot_count) =
            streamed_texture_slots(disk_backed, texture_payloads.len(), texture_locators.len())
        else {
            return;
        };
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
        let streamer = StreamPool::for_textures(
            &config,
            texture_centers,
            backend.gpu_profile().memory_budget_bytes,
        );
        tracing::info!(
            "GraphicsSystem: texture streaming enabled ({} textures, {} source, budget {}/frame, cap {}, byte budget {})",
            streamer.len(),
            if disk_backed { "disk" } else { "ram" },
            config.budget(),
            config.cap(),
            describe_budget(streamer.byte_budget()),
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
    // re-reads it from there (no persistent RAM copy), an in-memory `cn run -f`
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
        // Init residency. A shrinkable seed means the streamed geometry was
        // never baked in: seed the sub-allocators with its headroom block
        // (evicting would free the placeholder region under the first resident
        // draw). Otherwise free each streamed draw's build-time region.
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
        let streamer =
            StreamPool::for_meshes(&config, centers, backend.gpu_profile().memory_budget_bytes);
        tracing::info!(
            "GraphicsSystem: mesh streaming enabled ({} meshes, {} source, budget {}/frame, cap {}, byte budget {})",
            streamer.len(),
            if disk_backed { "disk" } else { "ram" },
            config.mesh_budget(),
            config.mesh_cap(),
            describe_budget(streamer.byte_budget()),
        );
        self.stream_pools.mesh = Some(streamer);
        self.stream_workers.mesh = Some(worker);
        self.stream_pools.mesh_draws = draw_indices;
    }

    // Stand up an infinite voxel world's chunk streaming: grow the GPU buffers
    // by the chunk window's headroom and start the chunk pool's generation
    // thread.
    pub(super) fn setup_voxel_world_streaming(
        &mut self,
        vw: &VoxelWorld,
        block_types: &HashMap<AssetId, BlockType>,
        material_map: &HashMap<MaterialHandle, MaterialEntry>,
    ) {
        let Some(backend) = self.backend.as_deref_mut() else {
            return;
        };
        let VoxelStream {
            stream,
            source,
            headroom: (chunk_vtx_bytes, chunk_idx_bytes),
        } = VoxelStream::plan(
            vw,
            block_types,
            material_map,
            backend.gpu_profile().memory_budget_bytes,
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
        let chunk_blocks = vw.chunk_blocks();
        tracing::info!(
            "GraphicsSystem: VoxelWorld streaming enabled (seed {}, {}x{}x{} blocks, near-radius {}, impostor-radius {} (step {}), budget {}/frame, byte budget {}, {} KiB chunk headroom)",
            vw.seed,
            chunk_blocks[0],
            chunk_blocks[1],
            chunk_blocks[2],
            vw.view_radius(),
            if vw.impostors_enabled() {
                vw.impostor_radius()
            } else {
                0
            },
            vw.impostor_step(),
            vw.load_budget(),
            describe_budget(stream.pool.byte_budget()),
            (chunk_vtx_bytes + chunk_idx_bytes) / 1024,
        );
        self.stream_workers.chunk = Some(worker);
        self.stream_pools.chunks = Some(stream);
    }

    // Stand up a planet's terrain streaming: grow the GPU buffers by the
    // planet's tile headroom and start the tile workers.
    pub(super) fn setup_planet_streaming(
        &mut self,
        planet: &Planet,
        far_field: Option<CanopyRange>,
        material_map: &HashMap<MaterialHandle, MaterialEntry>,
    ) {
        let plan = PlanetStream::plan(planet, far_field.as_ref(), material_map);
        let (vertex_bytes, index_bytes) = plan.headroom;
        let Some(backend) = self.backend.as_deref_mut() else {
            return;
        };
        let Some(worker) = started("Planet terrain", tile_workers::spawn_workers()) else {
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
        self.stream_pools.planet = Some(plan.tiles());
    }
}

#[cfg(test)]
mod tests;
