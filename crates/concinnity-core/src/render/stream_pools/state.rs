//! Every streaming pool a world runs, driven once per frame: scene pins, the
//! texture, mesh and voxel-chunk pools, the RAM back-off valve and the
//! device-memory report. GPU effects are recorded into the frame's op queue
//! with owned payloads and replayed at submission; an op the backend refuses
//! comes back a frame later as an [`OpFailure`] and is rolled back by
//! [`StreamingState::apply_op_failures`].

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::accounting::{self, PoolReport};
use super::chunk_pool::ChunkPool;
use super::decoded_mesh::DecodedMesh;
use super::item_pool::StreamPool;
use super::planet_tiles::{PlanetTiles, TileCamera, TileMaterial};
use super::pressure::{self, StreamPressureStage};
use super::stats_log::PoolHeartbeats;
use crate::bake::texture::TextureImage;
use crate::ecs::asset_id::AssetId;
use crate::gfx::chunk_coord::{self, ChunkCoord};
use crate::gfx::render_types::DrawIndex;
use crate::memory::MemTag;
use crate::planet::PlanetFrame;
use crate::render::backend::ChunkMesh;
use crate::render::error::RenderError;
use crate::render::ops::{OpFailure, RenderOps};
use crate::render::queues::RenderQueues;
use crate::render::render_slots::RenderSlots;
use crate::render::scene_residency::{
    CHANNEL_MESH, CHANNEL_SHADER, CHANNEL_TEXTURE, Member, SceneResidency,
};

// Throttled RSS sampling cadence for the RAM back-off valve, ~2x/second near
// 60 fps: RSS is a syscall, so it is read off the frame clock rather than every
// frame.
const PRESSURE_SAMPLE_INTERVAL: u64 = 30;

/// The shader buckets a scene owns, built off the frame thread as their scene
/// pins. The host implements it over its build threads.
pub trait ShaderPump {
    /// Block or unblock `bucket` as its scene unpins or pins.
    fn set_blocked(&mut self, bucket: u32, blocked: bool);

    /// Record this frame's shader work, telling `on_resident` each bucket
    /// whose residency changed.
    fn pump(&mut self, ops: &mut RenderOps, on_resident: &mut dyn FnMut(u32, bool));
}

/// An infinite `VoxelWorld`'s streaming: the chunk pool, the resident
/// chunk-to-draw map, and what every chunk draws with.
pub struct ChunkStream {
    /// The chunk pool.
    pub pool: ChunkPool,
    /// Each resident chunk's draw slot.
    pub draws: BTreeMap<ChunkCoord, DrawIndex>,
    /// One chunk's world-space width.
    pub chunk_w: f32,
    /// One chunk's world-space depth.
    pub chunk_d: f32,
    /// The chunk every resident chunk's model matrix is placed relative to,
    /// following the camera's chunk.
    pub origin_chunk: ChunkCoord,
    /// What every chunk draws with.
    pub material: TileMaterial,
}

/// The pools a world's graphics load builds.
#[derive(Default)]
pub struct StreamingPools {
    /// The shared albedo + normal-map texture pool, when a `StreamingConfig`
    /// is declared.
    pub texture: Option<StreamPool<TextureImage>>,
    /// The mesh-geometry pool, under the same condition.
    pub mesh: Option<StreamPool<DecodedMesh>>,
    /// Each streamed mesh's draw slot, by stream id.
    pub mesh_draws: Vec<DrawIndex>,
    /// Voxel-world chunks, when a `VoxelWorld` is declared.
    pub chunks: Option<ChunkStream>,
    /// A planet's terrain tiles, when a `Planet` is declared.
    pub planet: Option<PlanetTiles>,
    /// Scene-pinned residency over the pools, when the world declares scenes.
    pub scene_residency: Option<SceneResidency>,
}

impl StreamingPools {
    // Block or unblock scene-residency `member` on whatever streams it.
    fn set_member_blocked<S: ShaderPump + ?Sized>(
        &mut self,
        shaders: Option<&mut S>,
        (channel, id): Member,
        blocked: bool,
    ) {
        match channel {
            CHANNEL_TEXTURE => {
                if let Some(p) = &mut self.texture {
                    p.set_blocked(id as usize, blocked);
                }
            }
            CHANNEL_MESH => {
                if let Some(p) = &mut self.mesh {
                    p.set_blocked(id as usize, blocked);
                }
            }
            CHANNEL_SHADER => {
                if let Some(s) = shaders {
                    s.set_blocked(id, blocked);
                }
            }
            _ => {}
        }
    }
}

// Record scene-residency member `(channel, id)` failing for good, so its scene
// loads without it.
fn note_failed(residency: &mut Option<SceneResidency>, channel: u8, id: usize) {
    if let Some(residency) = residency {
        residency.note_failed((channel, id as u32));
    }
}

// Record scene-residency member `(channel, id)` arriving or leaving.
fn note_resident(residency: &mut Option<SceneResidency>, channel: u8, id: u32, resident: bool) {
    if let Some(residency) = residency {
        residency.note_resident((channel, id), resident);
    }
}

/// `(resident, pending, ...)` counts for each streaming pool, or `None` when
/// that pool is not streaming.
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    /// `(resident, pending, unloaded)` texture counts when streaming.
    pub texture: Option<(usize, usize, usize)>,
    /// `(resident, pending, unloaded)` mesh counts when streaming.
    pub mesh: Option<(usize, usize, usize)>,
    /// `(resident, pending)` chunk counts when a `VoxelWorld` is streaming.
    pub chunk: Option<(usize, usize)>,
    /// `(resident, pending)` terrain tile counts when a `Planet` is streaming.
    pub planet: Option<(usize, usize)>,
    /// `(resident_bytes, byte_budget)` for the texture pool when streaming;
    /// `byte_budget` is 0 when the pool runs count-only.
    pub texture_bytes: Option<(u64, u64)>,
    /// `(resident_bytes, byte_budget)` for the mesh pool when streaming.
    pub mesh_bytes: Option<(u64, u64)>,
    /// `(resident_bytes, byte_budget)` for the chunk pool when a `VoxelWorld` is
    /// streaming; `byte_budget` is 0 when the GPU reported no memory figure.
    pub chunk_bytes: Option<(u64, u64)>,
}

/// Live process-RAM pressure on streaming, published on each throttled sample
/// when a `MemoryBudget` is present. `under_pressure` is true whenever the
/// back-off valve is engaged (gating loads or evicting).
#[derive(Debug, Clone, Copy)]
pub struct StreamingPressure {
    /// Process resident-set size at the sample.
    pub rss_bytes: u64,
    /// The published memory budget.
    pub budget_bytes: u64,
    /// Whether the back-off valve is engaged.
    pub under_pressure: bool,
}

/// A planet's frame and the camera its tiles are chosen for.
#[derive(Debug, Clone, Copy)]
pub struct PlanetView {
    /// The frame the planet was last published in.
    pub frame: PlanetFrame,
    /// The camera, and how large the ground looks from it.
    pub camera: TileCamera,
}

/// What a frame's streaming drive sees.
#[derive(Debug, Clone, Copy)]
pub struct DriveView<'a> {
    /// The camera position in absolute world space.
    pub cam_pos: [f32; 3],
    /// The camera's absolute view matrix.
    pub view: [[f32; 4]; 4],
    /// The world is hidden behind an opaque menu.
    pub world_hidden: bool,
    /// The scenes pinned resident this frame.
    pub scene_pins: Option<&'a [AssetId]>,
    /// The planet whose tiles stream, when one does.
    pub planet: Option<PlanetView>,
}

/// The streaming pools and the frame clock they run on. The clock ticks once
/// per drive, in lockstep with the renderer's (both start at 0 and tick once
/// per world step), so eviction retire-frames and LRU scores use the frame
/// number the draw does.
pub struct StreamingState {
    pools: StreamingPools,
    frame_count: u64,
    // Frames the backend keeps in flight: an eviction's freed region cannot be
    // reused until the command buffers that drew it retire.
    frames_in_flight: usize,
    // Each pool's resident-byte budget at setup; the valve scales the live
    // budget below it under deep pressure and restores it exactly on release.
    texture_baseline: Option<u64>,
    mesh_baseline: Option<u64>,
    chunk_baseline: Option<u64>,
    pressure_stage: StreamPressureStage,
    pressure_factor: f64,
    last_sampled_rss: Option<u64>,
    heartbeats: PoolHeartbeats,
}

impl core::fmt::Debug for StreamingState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StreamingState")
            .field("frame_count", &self.frame_count)
            .field("texture", &self.pools.texture.is_some())
            .field("mesh", &self.pools.mesh.is_some())
            .field("chunk", &self.pools.chunks.is_some())
            .field("pressure", &self.pressure_stage)
            .finish_non_exhaustive()
    }
}

impl StreamingState {
    /// The pools init built, each pool's current byte budget captured as the
    /// valve's baseline.
    pub fn new(pools: StreamingPools, frames_in_flight: usize) -> Self {
        Self {
            texture_baseline: pools.texture.as_ref().and_then(StreamPool::byte_budget),
            mesh_baseline: pools.mesh.as_ref().and_then(StreamPool::byte_budget),
            chunk_baseline: pools.chunks.as_ref().and_then(|c| c.pool.byte_budget()),
            pools,
            frame_count: 0,
            frames_in_flight,
            pressure_stage: StreamPressureStage::None,
            pressure_factor: 1.0,
            last_sampled_rss: None,
            heartbeats: PoolHeartbeats::default(),
        }
    }

    /// Block every scene-owned member, then pin `start`, so only it and the
    /// unowned set stream.
    pub fn pin_start_scene<S: ShaderPump + ?Sized>(
        &mut self,
        start: AssetId,
        mut shaders: Option<&mut S>,
    ) {
        let Some(residency) = &self.pools.scene_residency else {
            return;
        };
        let members: Vec<Member> = residency.all_members().collect();
        for member in members {
            self.pools
                .set_member_blocked(shaders.as_deref_mut(), member, true);
        }
        self.sync_scene_pins(Some(&[start]), shaders);
    }

    /// The texture pool, when streaming.
    pub fn texture_mut(&mut self) -> Option<&mut StreamPool<TextureImage>> {
        self.pools.texture.as_mut()
    }

    /// The mesh pool, when streaming.
    pub fn mesh_mut(&mut self) -> Option<&mut StreamPool<DecodedMesh>> {
        self.pools.mesh.as_mut()
    }

    /// The chunk pool, when a `VoxelWorld` streams.
    pub fn chunk_pool_mut(&mut self) -> Option<&mut ChunkPool> {
        self.pools.chunks.as_mut().map(|c| &mut c.pool)
    }

    /// The planet's tiles, when a `Planet` streams.
    pub fn planet_mut(&mut self) -> Option<&mut PlanetTiles> {
        self.pools.planet.as_mut()
    }

    /// Scene-pinned residency, when the world declares scenes.
    pub fn scene_residency(&self) -> Option<&SceneResidency> {
        self.pools.scene_residency.as_ref()
    }

    /// Frames driven so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Whether this frame takes the throttled RAM sample.
    pub fn pressure_due(&self) -> bool {
        self.frame_count.is_multiple_of(PRESSURE_SAMPLE_INTERVAL)
    }

    /// Roll back the recorded ops that failed at the previous frame's replay:
    /// a refused mesh or texture upload returns to `Unloaded` (retried once
    /// freed space reclaims), a failed texture eviction returns to resident
    /// (the texture is still on the GPU), and a failed chunk or tile add drops
    /// its tracking and frees its draw slot.
    pub fn apply_op_failures(&mut self, failures: &[OpFailure], slots: &mut RenderSlots) {
        let pools = &mut self.pools;
        for &failure in failures {
            match failure {
                OpFailure::MeshUpload { stream_id } => {
                    if let Some(pool) = &mut pools.mesh {
                        pool.note_upload_failed(stream_id);
                    }
                    note_resident(
                        &mut pools.scene_residency,
                        CHANNEL_MESH,
                        stream_id as u32,
                        false,
                    );
                }
                OpFailure::TextureUpload { slot } => {
                    if let Some(pool) = &mut pools.texture {
                        pool.note_upload_failed(slot);
                    }
                    note_resident(
                        &mut pools.scene_residency,
                        CHANNEL_TEXTURE,
                        slot as u32,
                        false,
                    );
                }
                OpFailure::TextureEvict { slot } => {
                    if let Some(pool) = &mut pools.texture {
                        pool.note_evict_failed(slot, self.frame_count);
                    }
                    note_resident(
                        &mut pools.scene_residency,
                        CHANNEL_TEXTURE,
                        slot as u32,
                        true,
                    );
                }
                OpFailure::PlanetTileAdd { tile } => {
                    if let Some(tiles) = &mut pools.planet {
                        tiles.roll_back(tile, slots);
                    }
                }
                OpFailure::ChunkAdd { coord } => {
                    if let Some(cs) = &mut pools.chunks
                        && let Some(draw_idx) = cs.draws.remove(&coord)
                    {
                        slots.free_draw(draw_idx);
                        tracing::warn!("chunk add ({},{}) rolled back", coord.x, coord.z);
                    }
                }
            }
        }
    }

    // Show, hide, drop and request a planet's tiles for its camera, in the
    // frame the planet was last published in.
    fn step_planet(&mut self, planet: &PlanetView, queues: &mut RenderQueues) {
        let retire_frame = self.retire_frame();
        if let Some(tiles) = &mut self.pools.planet {
            tiles.step(
                &planet.frame,
                planet.camera,
                &mut queues.ops,
                &mut queues.slots,
                self.frame_count,
                retire_frame,
            );
        }
    }

    fn retire_frame(&self) -> u64 {
        self.frame_count + self.frames_in_flight as u64
    }

    /// Score, request and apply this frame's streaming for every active pool,
    /// then return the view and camera position the draw should use (absolute
    /// unless a `VoxelWorld` rebases them). Advances the frame clock.
    pub fn drive<S: ShaderPump + ?Sized>(
        &mut self,
        queues: &mut RenderQueues,
        view: DriveView<'_>,
        shaders: Option<&mut S>,
    ) -> ([[f32; 4]; 4], [f32; 3]) {
        if let Some(planet) = &view.planet {
            self.step_planet(planet, queues);
        }
        // Stage 1 of the RAM valve freezes new load requests: the pools keep
        // their residency but stop growing. Stage 2 keeps planning (under a
        // reduced byte budget) so the planner can evict.
        let loads_frozen = self.pressure_stage.freezes_loads();
        let mut shaders = shaders;
        self.sync_scene_pins(view.scene_pins, shaders.as_deref_mut());

        // Start building (or release) the shader buckets whose scene pin
        // changed, and install the builds that finished.
        if let Some(pump) = shaders {
            let residency = &mut self.pools.scene_residency;
            pump.pump(&mut queues.ops, &mut |bucket, resident| {
                note_resident(residency, CHANNEL_SHADER, bucket, resident);
            });
        }

        // A pinned scene mid-load keeps the pools streaming even while the
        // world is hidden, so a loading screen's opaque backdrop does not pause
        // the very load whose progress it shows.
        let world_hidden = view.world_hidden
            && !self
                .pools
                .scene_residency
                .as_ref()
                .is_some_and(SceneResidency::any_loading);

        if !world_hidden {
            self.drive_textures(&mut queues.ops, view.cam_pos, loads_frozen);
            self.drive_meshes(&mut queues.ops, view.cam_pos, loads_frozen);
        }
        let out = self.drive_chunks(queues, &view, loads_frozen);
        self.frame_count += 1;
        out
    }

    // Sync scene pins onto the pools: members of a scene leaving the pin set
    // are blocked (never load, evict next plan), members of a scene entering it
    // unblock and stream in through the normal planning path.
    fn sync_scene_pins<S: ShaderPump + ?Sized>(
        &mut self,
        pins: Option<&[AssetId]>,
        mut shaders: Option<&mut S>,
    ) {
        let (Some(residency), Some(pins)) = (self.pools.scene_residency.as_mut(), pins) else {
            return;
        };
        let changes = residency.sync_pins(pins);
        for (members, blocked) in [(&changes.blocked, true), (&changes.unblocked, false)] {
            for &member in members {
                self.pools
                    .set_member_blocked(shaders.as_deref_mut(), member, blocked);
            }
        }
    }

    // Re-score every texture slot by camera distance, request this frame's
    // loads, then apply finished uploads and evictions. Each backend's
    // update_texture_slot rewrites whatever samples the slot, so it takes
    // effect on this same frame.
    fn drive_textures(&mut self, ops: &mut RenderOps, cam_pos: [f32; 3], loads_frozen: bool) {
        let Some(pool) = &mut self.pools.texture else {
            return;
        };
        let residency = &mut self.pools.scene_residency;
        let frame = self.frame_count;
        pool.update_scores(cam_pos, frame);
        if !loads_frozen {
            for slot in pool.plan() {
                ops.record_with(move |backend, out| {
                    if let Err(e) = backend.evict_texture_slot(slot) {
                        tracing::warn!("texture evict slot {}: {}", slot, e);
                        out.memory_pressure |= matches!(e, RenderError::OutOfDeviceMemory(_));
                        out.failures.push(OpFailure::TextureEvict { slot });
                    }
                });
                note_resident(residency, CHANNEL_TEXTURE, slot as u32, false);
            }
        }
        pool.drain_completed(frame, |slot, image| {
            // Marked resident on handoff; a refused upload comes back as an op
            // failure and is rolled back next tick.
            ops.record_with(move |backend, out| {
                if let Err(e) = backend.update_texture_slot(slot, &image) {
                    tracing::debug!("texture upload slot {} deferred: {}", slot, e);
                    out.memory_pressure |= matches!(e, RenderError::OutOfDeviceMemory(_));
                    out.failures.push(OpFailure::TextureUpload { slot });
                }
            });
            note_resident(residency, CHANNEL_TEXTURE, slot as u32, true);
        });
        pool.drain_failed(|slot| note_failed(residency, CHANNEL_TEXTURE, slot));
        if let Some((resident, pending, unloaded)) =
            self.heartbeats.texture.sample(frame, || pool.stats())
        {
            tracing::info!(
                "texture streaming: {} resident, {} pending, {} unloaded",
                resident,
                pending,
                unloaded
            );
        }
    }

    // Re-score each streamed mesh by camera distance, request this frame's
    // loads, then apply finished geometry uploads and evictions.
    fn drive_meshes(&mut self, ops: &mut RenderOps, cam_pos: [f32; 3], loads_frozen: bool) {
        // A runtime eviction's freed space must not be reused until the
        // in-flight command buffers that drew it retire.
        let retire_frame = self.retire_frame();
        let Some(pool) = &mut self.pools.mesh else {
            return;
        };
        let (draws, residency) = (&self.pools.mesh_draws, &mut self.pools.scene_residency);
        let frame = self.frame_count;
        pool.update_scores(cam_pos, frame);
        if !loads_frozen {
            for stream_id in pool.plan() {
                if let Some(&draw_idx) = draws.get(stream_id) {
                    ops.record(move |backend| {
                        if let Err(e) = backend.evict_mesh(draw_idx, retire_frame) {
                            tracing::warn!("mesh evict draw {}: {}", draw_idx, e);
                        }
                    });
                }
                note_resident(residency, CHANNEL_MESH, stream_id as u32, false);
            }
        }
        pool.drain_completed(frame, |stream_id, mesh| {
            // Marked resident on handoff; a transient seed-full refusal comes
            // back as an op failure and is rolled back next tick.
            if let Some(&draw_idx) = draws.get(stream_id) {
                let DecodedMesh { vertices, indices } = mesh;
                ops.record_with(move |backend, out| {
                    if let Err(e) = backend.upload_mesh(draw_idx, &vertices, &indices, frame) {
                        tracing::debug!("mesh upload draw {} deferred: {}", draw_idx, e);
                        out.memory_pressure |= matches!(e, RenderError::OutOfDeviceMemory(_));
                        out.failures.push(OpFailure::MeshUpload { stream_id });
                    }
                });
            }
            note_resident(residency, CHANNEL_MESH, stream_id as u32, true);
        });
        pool.drain_failed(|stream_id| note_failed(residency, CHANNEL_MESH, stream_id));
        if let Some((resident, pending, unloaded)) =
            self.heartbeats.mesh.sample(frame, || pool.stats())
        {
            tracing::info!(
                "mesh streaming: {} resident, {} pending, {} unloaded",
                resident,
                pending,
                unloaded
            );
        }
    }

    // Generate and upload the chunks entering the camera's view window and
    // remove those that have left it. Chunk geometry is placed relative to a
    // render origin that follows the camera's chunk, and the view and camera
    // handed to the backend are rebased onto the same origin, so an unbounded
    // world renders from small coordinates. Both stay absolute when no
    // VoxelWorld streams.
    fn drive_chunks(
        &mut self,
        queues: &mut RenderQueues,
        view: &DriveView<'_>,
        loads_frozen: bool,
    ) -> ([[f32; 4]; 4], [f32; 3]) {
        let retire_frame = self.retire_frame();
        let frame = self.frame_count;
        let Some(cs) = &mut self.pools.chunks else {
            return (view.view, view.cam_pos);
        };
        let (ops, slots) = (&mut queues.ops, &mut queues.slots);
        let camera_chunk = cs.pool.camera_chunk(view.cam_pos);
        // Stage 1 of the valve freezes chunk residency like the texture and
        // mesh gate: skipping the plan stops both new generation and window
        // eviction while in-flight loads still apply below.
        if !loads_frozen {
            for coord in cs.pool.plan(camera_chunk) {
                if let Some(draw_idx) = cs.draws.remove(&coord) {
                    slots.free_draw(draw_idx);
                    ops.record(move |backend| {
                        if let Err(e) = backend.remove_chunk_mesh(draw_idx, retire_frame) {
                            tracing::warn!("chunk remove ({},{}): {}", coord.x, coord.z, e);
                        }
                    });
                }
            }
        }
        // The camera crossed into a new chunk: move the render origin to it and
        // rebase every resident chunk's model matrix. The rebase is exact, so a
        // stationary chunk shows no TAA velocity across the shift.
        if camera_chunk != cs.origin_chunk {
            for (&coord, &draw_idx) in &cs.draws {
                let model = chunk_model_matrix(coord, camera_chunk, cs.chunk_w, cs.chunk_d);
                ops.record(move |backend| {
                    if let Err(e) = backend.set_chunk_model(draw_idx, model) {
                        tracing::warn!("chunk rebase ({},{}): {}", coord.x, coord.z, e);
                    }
                });
            }
            cs.origin_chunk = camera_chunk;
        }
        let (chunk_w, chunk_d) = (cs.chunk_w, cs.chunk_d);
        let material = cs.material;
        let draws = &mut cs.draws;
        cs.pool.drain_completed(|coord, verts, idxs| {
            if verts.is_empty() || idxs.is_empty() {
                tracing::warn!("chunk add ({},{}): empty chunk geometry", coord.x, coord.z);
                return;
            }
            let model = chunk_model_matrix(coord, camera_chunk, chunk_w, chunk_d);
            let dst = slots.allocate_draw();
            draws.insert(coord, dst.slot());
            // A failed add comes back as an op failure; the rollback drops the
            // tracking and frees the slot.
            ops.record_with(move |backend, out| {
                let mesh = ChunkMesh {
                    verts: &verts,
                    idxs: &idxs,
                    model,
                    texture_slot: material.texture_slot,
                    normal_map_slot: material.normal_map_slot,
                    material: material.material,
                    frame,
                    planet_tile: None,
                    bounds: None,
                };
                if let Err(e) = backend.add_chunk_mesh(mesh, dst) {
                    tracing::warn!("chunk add ({},{}): {}", coord.x, coord.z, e);
                    out.memory_pressure |= matches!(e, RenderError::OutOfDeviceMemory(_));
                    out.failures.push(OpFailure::ChunkAdd { coord });
                }
            });
        });
        // Rebase the view + camera onto the render origin so the origin-relative
        // chunk geometry transforms exactly.
        let (ox, oz) = camera_chunk.origin_world(cs.chunk_w, cs.chunk_d);
        let final_view = chunk_coord::camera_relative_view(view.view, view.cam_pos, [ox, 0.0, oz]);
        let final_cam_pos = [view.cam_pos[0] - ox, view.cam_pos[1], view.cam_pos[2] - oz];
        if let Some((resident, pending, near, far)) = self.heartbeats.chunk.sample(frame, || {
            let (resident, pending) = cs.pool.stats();
            let (near, far) = cs.pool.detail_counts();
            (resident, pending, near, far)
        }) {
            tracing::info!(
                "chunk streaming: {} resident ({} full, {} impostor), {} pending",
                resident,
                near,
                far,
                pending
            );
        }
        (final_view, final_cam_pos)
    }

    /// Re-evaluate the RAM back-off valve from a fresh RSS sample and apply its
    /// decision to the pools' byte budgets. Returns the reading to publish, or
    /// `None` when RSS is unavailable (the valve stays inert). `budget` is the
    /// `MemoryBudget` ceiling.
    pub fn sample_pressure(&mut self, rss: Option<u64>, budget: u64) -> Option<StreamingPressure> {
        let rss = rss?;
        let rising = self.last_sampled_rss.is_some_and(|prev| rss > prev);
        let prev_stage = self.pressure_stage;
        let decision =
            pressure::step_pressure(rss, budget, rising, prev_stage, self.pressure_factor);

        // Re-apply the byte budgets only while evicting (the factor may have
        // tightened) or on the way out of eviction (restore the baseline
        // exactly), so a world never under pressure keeps its baselines.
        use StreamPressureStage::Evict;
        match (prev_stage, decision.stage) {
            (_, Evict) => self.apply_byte_factor(decision.budget_factor),
            (Evict, _) => self.apply_byte_factor(1.0),
            _ => {}
        }

        self.pressure_stage = decision.stage;
        self.pressure_factor = decision.budget_factor;
        self.last_sampled_rss = Some(rss);
        Some(StreamingPressure {
            rss_bytes: rss,
            budget_bytes: budget,
            under_pressure: decision.stage != StreamPressureStage::None,
        })
    }

    // Scale each pool's byte budget to `factor` of its baseline. Pools with no
    // baseline (count-only) have nothing to reduce.
    fn apply_byte_factor(&mut self, factor: f64) {
        if let (Some(pool), Some(baseline)) = (self.pools.texture.as_mut(), self.texture_baseline) {
            pool.set_byte_budget(Some(pressure::scale_budget(baseline, factor)));
        }
        if let (Some(pool), Some(baseline)) = (self.pools.mesh.as_mut(), self.mesh_baseline) {
            pool.set_byte_budget(Some(pressure::scale_budget(baseline, factor)));
        }
        // The chunk window answers a reduced budget by shrinking its effective
        // view radius, dropping the far impostor band first.
        if let (Some(cs), Some(baseline)) = (self.pools.chunks.as_mut(), self.chunk_baseline) {
            cs.pool
                .set_byte_budget(Some(pressure::scale_budget(baseline, factor)));
        }
    }

    /// Publish each streaming pool's device footprint into the shared memory
    /// ledger. `transient_pool_bytes` is the render graph's transient pool,
    /// which no other path accounts for; it rides under `Textures` beside the
    /// texture pool, on its usage and its budget both.
    pub fn publish_footprint(&self, transient_pool_bytes: u64) {
        accounting::publish(
            crate::memory::ledger(),
            self.pool_reports(transient_pool_bytes),
        );
    }

    fn pool_reports(&self, transient_pool_bytes: u64) -> impl Iterator<Item = PoolReport> {
        let textures = accounting::textures_report(
            self.pools
                .texture
                .as_ref()
                .map(|p| (p.resident_bytes(), p.byte_budget())),
            transient_pool_bytes,
        );
        [
            self.pools
                .mesh
                .as_ref()
                .map(|p| (MemTag::Meshes, p.resident_bytes(), p.byte_budget())),
            self.pools.chunks.as_ref().map(|cs| {
                (
                    MemTag::Chunks,
                    cs.pool.resident_bytes(),
                    cs.pool.byte_budget(),
                )
            }),
        ]
        .into_iter()
        .flatten()
        .map(|(tag, resident_bytes, byte_budget)| PoolReport {
            tag,
            resident_bytes,
            byte_budget,
        })
        .chain(textures)
    }

    /// Counts for each active streaming pool.
    pub fn streaming_stats(&self) -> StreamingStats {
        StreamingStats {
            texture: self.pools.texture.as_ref().map(StreamPool::stats),
            mesh: self.pools.mesh.as_ref().map(StreamPool::stats),
            chunk: self.pools.chunks.as_ref().map(|cs| cs.pool.stats()),
            planet: self.pools.planet.as_ref().map(PlanetTiles::stats),
            texture_bytes: self
                .pools
                .texture
                .as_ref()
                .map(|p| (p.resident_bytes(), p.byte_budget().unwrap_or(0))),
            mesh_bytes: self
                .pools
                .mesh
                .as_ref()
                .map(|p| (p.resident_bytes(), p.byte_budget().unwrap_or(0))),
            chunk_bytes: self
                .pools
                .chunks
                .as_ref()
                .map(|cs| (cs.pool.resident_bytes(), cs.pool.byte_budget().unwrap_or(0))),
        }
    }
}

/// The model matrix that places chunk `coord`'s origin-local geometry relative
/// to the render origin `origin`, so the on-GPU transform stays exact and small
/// however far the world origin is.
pub fn chunk_model_matrix(
    coord: ChunkCoord,
    origin: ChunkCoord,
    chunk_w: f32,
    chunk_d: f32,
) -> [[f32; 4]; 4] {
    let dx = (coord.x - origin.x) as f32 * chunk_w;
    let dz = (coord.z - origin.z) as f32 * chunk_d;
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [dx, 0.0, dz, 1.0],
    ]
}

#[cfg(test)]
mod tests;
