// src/gfx/streaming_system/mod.rs
//
// StreamingSystem: drives the asset-streaming pools (albedo/normal texture,
// mesh geometry, and infinite voxel-world chunks) against the world's parked
// render backend, and publishes the camera-relative view the draw consumes.
//
// Scheduled immediately before GraphicsSystem, so a chunk world's view rebase
// (see `CameraRelativeView`) is ready for this same frame's submit, and any
// texture / mesh upload lands before the draw. GraphicsSystem's init builds the
// streamers (world content + backend support) and parks them here as the
// `StreamingState` resource; each step takes it and puts it back, so the state
// and the `PipelineContext` are never borrowed together (the same handoff the
// backend, settings, and overlay states use).
//
// The streamers themselves (the OS-coupled worker threads + channels) live in
// `crate::gfx::streaming::{texture, mesh, chunk}`; this module only
// scores, dispatches, and applies their results each frame.

use crate::assets::Camera3D;
use crate::ecs::{ActiveRenderBackend, PipelineContext, StepResult, System};
use crate::gfx::backend::{ChunkMesh, RenderBackend};
use crate::gfx::overlay::OverlayFrame;

pub(crate) mod pressure;

const IDENTITY4: [[f32; 4]; 4] = crate::gfx::draw_list::IDENTITY4;

// Throttled RSS sampling cadence for the process-RAM back-off valve. RSS is a
// syscall, so the valve re-evaluates ~2x/second (every 30 frames near 60 fps)
// off the frame clock rather than every frame.
const PRESSURE_SAMPLE_INTERVAL: u64 = 30;

// The camera-relative view + position GraphicsSystem hands to `update_view` /
// `draw_frame`. Published every frame by StreamingSystem: the world's absolute
// view + camera position when no `VoxelWorld` is streaming, or both rebased
// onto the chunk render origin when one is (so an unbounded world renders from
// small coordinates without large-coordinate jitter). GraphicsSystem falls back
// to the absolute `Camera3D` values if this resource is absent (a unit test
// driving GraphicsSystem without StreamingSystem).
#[derive(Debug, Clone, Copy)]
pub struct CameraRelativeView {
    pub view: [[f32; 4]; 4],
    pub cam_pos: [f32; 3],
}

// Runtime state for streaming an infinite `VoxelWorld`: the chunk streamer,
// the resident chunk-to-draw-index map, and the per-chunk render parameters
// (chunk size for the camera-to-chunk mapping and model placement, plus the
// shared material every chunk draws with).
// Most fields are read only by the Metal chunk-streaming drive; on non-macOS
// builds the struct is still constructed but those fields go unread.
#[cfg_attr(not(backend_metal), allow(dead_code))]
pub(crate) struct ChunkStreamState {
    pub(crate) streamer: crate::gfx::streaming::chunk::ChunkStreamer,
    // Maps a resident chunk's coordinate to its `DrawObject` index.
    pub(crate) draws: std::collections::BTreeMap<crate::gfx::chunk_coord::ChunkCoord, usize>,
    pub(crate) chunk_w: f32,
    pub(crate) chunk_d: f32,
    // Render origin for camera-relative rendering: the chunk every resident
    // chunk's model matrix is currently placed relative to. It follows the
    // camera's chunk; when it changes the resident chunks are rebased onto the
    // new origin.
    pub(crate) origin_chunk: crate::gfx::chunk_coord::ChunkCoord,
    pub(crate) texture_slot: usize,
    pub(crate) normal_map_slot: usize,
    pub(crate) material: crate::gfx::render_types::MaterialUniforms,
}

// `(resident, pending, unloaded)` counts for each streaming pool, or `None`
// when that pool is not streaming. Read by the debug server's `streaming`
// command for headless verification. Only the `cn debug` binary consumes it,
// so it reads as dead code in a plain library build.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    pub texture: Option<(usize, usize, usize)>,
    pub mesh: Option<(usize, usize, usize)>,
    // `(resident, pending)` chunk counts when a `VoxelWorld` is streaming.
    pub chunk: Option<(usize, usize)>,
    // `(resident_bytes, byte_budget)` for the texture pool when streaming;
    // `byte_budget` is 0 when the pool runs count-only (no byte budget).
    pub texture_bytes: Option<(u64, u64)>,
    // `(resident_bytes, byte_budget)` for the mesh pool when streaming.
    pub mesh_bytes: Option<(u64, u64)>,
    // `(resident_bytes, byte_budget)` for the chunk pool when a VoxelWorld is
    // streaming; `byte_budget` is 0 when the GPU reported no memory figure.
    pub chunk_bytes: Option<(u64, u64)>,
}

// Live process-RAM pressure on streaming, published by StreamingSystem on each
// throttled sample when a `MemoryBudget` is present. `under_pressure` is true
// whenever the back-off valve is engaged (gating loads or evicting). Read by the
// debug server's `streaming` command for headless verification; harmless (and
// unread) in a plain `cn run`. Absent entirely when no `MemoryBudget` is
// published or RSS cannot be queried, in which case the valve is inert.
#[derive(Debug, Clone, Copy)]
pub struct StreamingPressure {
    pub rss_bytes: u64,
    pub budget_bytes: u64,
    pub under_pressure: bool,
}

// The streaming pools GraphicsSystem's init builds and hands off. Held as a
// parked resource; StreamingSystem takes it each step, drives the pools, and
// puts it back. `frame_count` is this system's own frame clock, incremented
// once per step; it stays in lockstep with GraphicsSystem's (both start at 0
// and tick once per world step), so eviction retire-frames and the LRU scores
// use the same frame number the draw does.
pub(crate) struct StreamingState {
    // Shared albedo + normal-map texture pool streamer. `Some` only when a
    // `StreamingConfig` was declared and the backend supports it (Metal).
    pub(crate) texture_streamer: Option<crate::gfx::streaming::texture::TextureStreamer>,
    // Mesh-geometry streamer. `Some` under the same conditions as above.
    pub(crate) mesh_streamer: Option<crate::gfx::streaming::mesh::MeshStreamer>,
    // Maps a streamed mesh's id to its DrawObject index, so completed loads and
    // evictions are applied to the right draw. Empty when not streaming.
    pub(crate) mesh_stream_draw_indices: Vec<usize>,
    // Infinite voxel-world chunk streaming. `Some` only when a `VoxelWorld` was
    // declared and the backend supports it (Metal).
    pub(crate) chunk_stream: Option<ChunkStreamState>,
    // This system's frame clock (see the struct doc).
    pub(crate) frame_count: u64,
    // Frames the backend keeps in flight: an eviction's freed region cannot be
    // reused until the command buffers that drew it retire, at
    // `frame_count + frames_in_flight`.
    pub(crate) frames_in_flight: usize,
    // Baseline (derived at setup) resident-byte budget for each pool; `None`
    // when the pool runs count-only (chunks: when no VoxelWorld streams or the
    // GPU reports no memory). The RAM back-off valve reduces the live budget
    // below this under deep pressure and restores it exactly on release.
    pub(crate) texture_baseline_budget: Option<u64>,
    pub(crate) mesh_baseline_budget: Option<u64>,
    pub(crate) chunk_baseline_budget: Option<u64>,
    // Process-RAM back-off valve state (see `pressure`), re-evaluated on the
    // throttled RSS sample. `pressure_factor` is the byte-budget scale currently
    // applied to the pools (1.0 = baseline); `last_sampled_rss` feeds the
    // "still rising" escalation from stage 1 to stage 2.
    pub(crate) pressure_stage: pressure::StreamPressureStage,
    pub(crate) pressure_factor: f64,
    pub(crate) last_sampled_rss: Option<u64>,
}

impl std::fmt::Debug for StreamingState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingState")
            .field("frame_count", &self.frame_count)
            .field("texture", &self.texture_streamer.is_some())
            .field("mesh", &self.mesh_streamer.is_some())
            .field("chunk", &self.chunk_stream.is_some())
            .field("pressure", &self.pressure_stage)
            .finish()
    }
}

#[derive(Debug, Default)]
pub struct StreamingSystem;

impl StreamingSystem {
    pub fn new() -> Self {
        Self
    }
}

impl System for StreamingSystem {
    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // No parked state (graphics init has not succeeded): nothing to drive,
        // and GraphicsSystem is not drawing either, so no view to publish.
        let Some(mut state) = ctx.resources.remove::<StreamingState>() else {
            return StepResult::Continue;
        };

        // Throttled process-RAM back-off sample (~2x/sec). Reads the world's
        // `MemoryBudget` ceiling and live RSS; when RSS nears the ceiling the
        // valve engages (stage 1 gates new loads, stage 2 shrinks residency).
        // Skipped entirely when no `MemoryBudget` is published or RSS is
        // unavailable, leaving streaming on its byte-budget policy unchanged.
        if state.frame_count.is_multiple_of(PRESSURE_SAMPLE_INTERVAL) {
            let budget = ctx
                .resource::<crate::app::budget::MemoryBudget>()
                .map(|b| b.budget_bytes);
            if let Some(budget) = budget {
                let rss = crate::app::sysmem::process_resident_bytes();
                if let Some(pressure) = state.sample_pressure(rss, budget) {
                    ctx.insert_resource(pressure);
                }
            }
        }

        // The camera the draw will use (written by the camera controller last
        // tick) is the absolute fallback when no chunk streaming rebases it.
        let (view_matrix, cam_pos) = ctx
            .query::<Camera3D>()
            .next()
            .map(|c| (c.view_matrix, c.position))
            .unwrap_or((IDENTITY4, [0.0; 3]));
        // Peek (not remove) the overlay's world-hidden flag: OverlaySystem
        // published it first this tick and GraphicsSystem removes it later.
        // Streaming pauses behind an opaque menu (the world is not drawn).
        let world_hidden = ctx
            .resource::<OverlayFrame>()
            .map(|o| o.world_hidden)
            .unwrap_or(false);

        let Some(mut backend) = ActiveRenderBackend::take(ctx.resources) else {
            // Init succeeded but the backend was taken (should not happen in the
            // schedule): publish the absolute view so the draw is still driven.
            ctx.insert_resource(CameraRelativeView {
                view: view_matrix,
                cam_pos,
            });
            ctx.resources.insert(state);
            return StepResult::Continue;
        };

        let (view, cam_pos) = state.drive(backend.as_mut(), cam_pos, view_matrix, world_hidden);

        ActiveRenderBackend::put(ctx.resources, backend);
        ctx.insert_resource(CameraRelativeView { view, cam_pos });
        ctx.resources.insert(state);
        StepResult::Continue
    }
}

impl StreamingState {
    // Score, dispatch, and apply this frame's streaming for every active pool,
    // then return the camera-relative view + position the draw should use
    // (absolute unless a `VoxelWorld` rebases them). Advances the frame clock.
    fn drive(
        &mut self,
        backend: &mut dyn RenderBackend,
        cam_pos: [f32; 3],
        view_matrix: [[f32; 4]; 4],
        world_hidden: bool,
    ) -> ([[f32; 4]; 4], [f32; 3]) {
        // Stage 1 of the RAM back-off valve freezes new load dispatch: the pools
        // keep their current residency but stop growing. Stage 2 keeps
        // dispatching (under a reduced byte budget) so the planner can evict.
        let loads_frozen = self.pressure_stage.freezes_loads();

        // Drive albedo-texture streaming: re-score every slot by camera
        // distance, dispatch this frame's background loads within budget, then
        // apply completed uploads + evictions. Each backend's
        // update_texture_slot rewrites whichever descriptors / argument-buffers
        // sample that slot so it takes effect on this same draw_frame.
        if !world_hidden && let Some(streamer) = &mut self.texture_streamer {
            streamer.update_scores(cam_pos, self.frame_count);
            if !loads_frozen {
                for slot in streamer.plan_and_dispatch() {
                    if let Err(e) = backend.evict_texture_slot(slot) {
                        tracing::warn!("StreamingSystem: texture evict slot {}: {}", slot, e);
                    }
                }
            }
            streamer.drain_completed(self.frame_count, |slot, image| {
                if let Err(e) = backend.update_texture_slot(slot, image) {
                    tracing::warn!("StreamingSystem: texture upload slot {}: {}", slot, e);
                }
            });
            // Surface streaming progress periodically so a headless run can
            // confirm textures are coming resident.
            if self.frame_count.is_multiple_of(120) {
                let (resident, pending, unloaded) = streamer.stats();
                tracing::info!(
                    "StreamingSystem: texture streaming -- {} resident, {} pending, {} unloaded",
                    resident,
                    pending,
                    unloaded
                );
            }
        }

        // Drive mesh-geometry streaming: re-score each streamed mesh by camera
        // distance, dispatch this frame's background loads, then apply completed
        // geometry uploads + evictions. A mesh is skipped in every pass until
        // its geometry region is resident.
        if !world_hidden && let Some(streamer) = &mut self.mesh_streamer {
            streamer.update_scores(cam_pos, self.frame_count);
            if !loads_frozen {
                // A runtime eviction's freed space must not be reused until the
                // in-flight command buffers that drew it retire.
                let retire_frame = self.frame_count + self.frames_in_flight as u64;
                for stream_id in streamer.plan_and_dispatch() {
                    if let Some(&draw_idx) = self.mesh_stream_draw_indices.get(stream_id)
                        && let Err(e) = backend.evict_mesh(draw_idx, retire_frame)
                    {
                        tracing::warn!("StreamingSystem: mesh evict draw {}: {}", draw_idx, e);
                    }
                }
            }
            let draw_indices = &self.mesh_stream_draw_indices;
            let frame = self.frame_count;
            streamer.drain_completed(self.frame_count, |stream_id, verts, idxs| {
                match draw_indices.get(stream_id) {
                    // Return the upload result so the streamer can roll a
                    // transient seed-full miss back to Unloaded and retry it once
                    // freed regions reclaim, rather than marking the mesh
                    // resident with no GPU geometry.
                    Some(&draw_idx) => backend.upload_mesh(draw_idx, verts, idxs, frame),
                    None => Ok(()),
                }
            });
            if self.frame_count.is_multiple_of(120) {
                let (resident, pending, unloaded) = streamer.stats();
                tracing::info!(
                    "StreamingSystem: mesh streaming -- {} resident, {} pending, {} unloaded",
                    resident,
                    pending,
                    unloaded
                );
            }
        }

        // Drive infinite-world chunk streaming: generate + upload the chunks
        // entering the camera's view window and remove those that have left it.
        // None unless a VoxelWorld was declared.
        //
        // Camera-relative rendering: chunk geometry is placed relative to a
        // render origin that follows the camera's chunk, and the view + camera
        // position handed to the backend are rebased onto the same origin. The
        // world transform is unchanged -- it is just evaluated from small
        // coordinates, so an unbounded world renders without large-coordinate
        // jitter. The view + camera stay absolute when no VoxelWorld is
        // streaming, leaving a non-voxel world byte-for-byte unchanged.
        let mut final_view = view_matrix;
        let mut final_cam_pos = cam_pos;
        if let Some(cs) = &mut self.chunk_stream {
            let camera_chunk = cs.streamer.camera_chunk(cam_pos);
            let retire_frame = self.frame_count + self.frames_in_flight as u64;
            // Stage 1 of the RAM valve freezes chunk residency, mirroring the
            // texture/mesh gate: skipping plan_and_dispatch stops both new
            // generation and window eviction, so residency holds steady while
            // the drain below keeps applying in-flight loads. Stage 2 keeps
            // planning under a reduced byte budget, so the window clamp evicts.
            if !loads_frozen {
                for coord in cs.streamer.plan_and_dispatch(camera_chunk) {
                    if let Some(draw_idx) = cs.draws.remove(&coord)
                        && let Err(e) = backend.remove_chunk_mesh(draw_idx, retire_frame)
                    {
                        tracing::warn!(
                            "StreamingSystem: chunk remove ({},{}): {}",
                            coord.x,
                            coord.z,
                            e
                        );
                    }
                }
            }
            // The camera crossed into a new chunk: move the render origin to it
            // and rebase every resident chunk's model matrix. `prev_draw_models`
            // is deliberately left alone -- the rebase is exact, so a stationary
            // chunk shows zero TAA velocity across the shift.
            if camera_chunk != cs.origin_chunk {
                for (&coord, &draw_idx) in &cs.draws {
                    let model = chunk_model_matrix(coord, camera_chunk, cs.chunk_w, cs.chunk_d);
                    if let Err(e) = backend.set_chunk_model(draw_idx, model) {
                        tracing::warn!(
                            "StreamingSystem: chunk rebase ({},{}): {}",
                            coord.x,
                            coord.z,
                            e
                        );
                    }
                }
                cs.origin_chunk = camera_chunk;
            }
            let frame = self.frame_count;
            let (chunk_w, chunk_d) = (cs.chunk_w, cs.chunk_d);
            let (tex, nm, mat) = (cs.texture_slot, cs.normal_map_slot, cs.material);
            let mut added: Vec<(crate::gfx::chunk_coord::ChunkCoord, usize)> = Vec::new();
            cs.streamer.drain_completed(|coord, verts, idxs| {
                let model = chunk_model_matrix(coord, camera_chunk, chunk_w, chunk_d);
                match backend.add_chunk_mesh(ChunkMesh {
                    verts,
                    idxs,
                    model,
                    texture_slot: tex,
                    normal_map_slot: nm,
                    material: mat,
                    frame,
                }) {
                    Ok(draw_idx) => added.push((coord, draw_idx)),
                    Err(e) => tracing::warn!(
                        "StreamingSystem: chunk add ({},{}): {}",
                        coord.x,
                        coord.z,
                        e
                    ),
                }
            });
            for (coord, draw_idx) in added {
                cs.draws.insert(coord, draw_idx);
            }
            // Rebase the view + camera onto the render origin so the
            // origin-relative chunk geometry above transforms exactly.
            let (ox, oz) = camera_chunk.origin_world(cs.chunk_w, cs.chunk_d);
            let origin = [ox, 0.0, oz];
            final_view =
                crate::gfx::chunk_coord::camera_relative_view(view_matrix, cam_pos, origin);
            final_cam_pos = [cam_pos[0] - ox, cam_pos[1], cam_pos[2] - oz];
            if self.frame_count.is_multiple_of(120) {
                let (resident, pending) = cs.streamer.stats();
                let (near, far) = cs.streamer.detail_counts();
                tracing::info!(
                    "StreamingSystem: chunk streaming -- {} resident ({} full, {} impostor), {} pending",
                    resident,
                    near,
                    far,
                    pending
                );
            }
        }

        self.frame_count += 1;
        (final_view, final_cam_pos)
    }

    // Re-evaluate the process-RAM back-off valve from a fresh RSS sample and
    // apply its decision to the texture + mesh pools. Returns the pressure
    // reading to publish, or `None` when RSS is unavailable (the valve stays
    // inert and nothing is published). `budget` is the `MemoryBudget` ceiling.
    fn sample_pressure(&mut self, rss: Option<u64>, budget: u64) -> Option<StreamingPressure> {
        let rss = rss?;
        let rising = self.last_sampled_rss.is_some_and(|prev| rss > prev);
        let prev_stage = self.pressure_stage;
        let decision =
            pressure::step_pressure(rss, budget, rising, prev_stage, self.pressure_factor);

        // Re-apply the pool byte budgets only when the reduced-budget state
        // actually needs it: while evicting (the factor may have tightened) or
        // on the transition out of eviction (restore the baseline exactly).
        // Staying at None/Gate leaves the budgets at their baseline untouched,
        // so a world never under pressure behaves exactly as before.
        use pressure::StreamPressureStage::Evict;
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
            under_pressure: decision.stage != pressure::StreamPressureStage::None,
        })
    }

    // Scale each pool's byte budget to `factor` of its captured baseline. Pools
    // with no baseline (count-only) are left alone; there is nothing to reduce.
    fn apply_byte_factor(&mut self, factor: f64) {
        if let (Some(streamer), Some(baseline)) =
            (self.texture_streamer.as_mut(), self.texture_baseline_budget)
        {
            streamer.set_byte_budget(Some(pressure::scale_budget(baseline, factor)));
        }
        if let (Some(streamer), Some(baseline)) =
            (self.mesh_streamer.as_mut(), self.mesh_baseline_budget)
        {
            streamer.set_byte_budget(Some(pressure::scale_budget(baseline, factor)));
        }
        // The chunk pool's byte clamp responds to a reduced budget by shrinking
        // its effective view radius, dropping the far impostor band first.
        if let (Some(cs), Some(baseline)) = (self.chunk_stream.as_mut(), self.chunk_baseline_budget)
        {
            cs.streamer
                .set_byte_budget(Some(pressure::scale_budget(baseline, factor)));
        }
    }

    // `(resident, pending, unloaded)` counts for each active streaming pool.
    // Consumed only by the `cn debug` binary's `streaming` command, so it reads
    // as dead code in a plain library build.
    #[allow(dead_code)]
    pub(crate) fn streaming_stats(&self) -> StreamingStats {
        StreamingStats {
            texture: self.texture_streamer.as_ref().map(|s| s.stats()),
            mesh: self.mesh_streamer.as_ref().map(|s| s.stats()),
            chunk: self.chunk_stream.as_ref().map(|cs| cs.streamer.stats()),
            texture_bytes: self
                .texture_streamer
                .as_ref()
                .map(|s| (s.resident_bytes(), s.byte_budget().unwrap_or(0))),
            mesh_bytes: self
                .mesh_streamer
                .as_ref()
                .map(|s| (s.resident_bytes(), s.byte_budget().unwrap_or(0))),
            chunk_bytes: self.chunk_stream.as_ref().map(|cs| {
                (
                    cs.streamer.resident_bytes(),
                    cs.streamer.byte_budget().unwrap_or(0),
                )
            }),
        }
    }
}

// Model matrix that places chunk `coord`'s origin-local geometry relative to
// the render origin `origin`, so the on-GPU transform stays exact and small
// regardless of how far the world origin is. The matching view matrix is
// rebased onto the same origin by `camera_relative_view`, which keeps an
// unbounded world's precision intact.
pub(crate) fn chunk_model_matrix(
    coord: crate::gfx::chunk_coord::ChunkCoord,
    origin: crate::gfx::chunk_coord::ChunkCoord,
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
mod tests {
    use super::*;
    use crate::blob::BlobData;
    use crate::ecs::{ActiveRenderBackend, ComponentStorage, Resources};
    use crate::gfx::chunk_coord::ChunkCoord;
    use crate::gfx::chunk_window::ChunkDetail;
    use crate::gfx::mesh_payload::Vertex;
    use crate::gfx::mock_backend::{Call, MockBackend, recording_backend};
    use crate::gfx::profile::FrameProfile;
    use crate::gfx::streaming::chunk::{ChunkSource, ChunkStreamer};
    use crate::gfx::streaming::mesh::{DecodedMesh, MeshPayloadSource, MeshStreamer};
    use crate::gfx::streaming::texture::{DecodedTexture, PayloadSource, TextureStreamer};
    use pressure::StreamPressureStage;
    use std::sync::Arc;

    // Upper bound on `drive_until` iterations. The pools decode on their own
    // worker threads, so a test that waits on one yields rather than sleeps;
    // the bound turns a regression into a failure instead of a hang.
    const MAX_DRIVE_SPINS: usize = 100_000;

    fn vtx() -> Vertex {
        Vertex {
            pos: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            color: [1.0; 3],
            uv: [0.0; 2],
        }
    }

    fn tri() -> DecodedMesh {
        DecodedMesh {
            vertices: vec![vtx(), vtx(), vtx()],
            indices: vec![0, 1, 2],
        }
    }

    // Sources yielding a fixed payload for any id, so the streamer workers
    // complete without the build pipeline.
    struct ConstTexture;
    impl PayloadSource for ConstTexture {
        fn fetch(&self, _id: usize) -> Result<DecodedTexture, String> {
            Ok(DecodedTexture {
                image: crate::build::texture::TextureImage::rgba8(1, 1, vec![1, 2, 3, 4]),
            })
        }
    }

    struct ConstMesh;
    impl MeshPayloadSource for ConstMesh {
        fn fetch(&self, _id: usize) -> Result<DecodedMesh, String> {
            Ok(tri())
        }
    }

    struct ConstChunk;
    impl ChunkSource for ConstChunk {
        fn generate(
            &self,
            _coord: ChunkCoord,
            _detail: ChunkDetail,
        ) -> Result<DecodedMesh, String> {
            Ok(tri())
        }
    }

    // A source whose generation always fails, so a chunk is tracked by the
    // window but never uploaded: it keeps the resident-draw map under the
    // test's control rather than the worker's timing.
    struct FailingChunk;
    impl ChunkSource for FailingChunk {
        fn generate(
            &self,
            _coord: ChunkCoord,
            _detail: ChunkDetail,
        ) -> Result<DecodedMesh, String> {
            Err("test source".to_string())
        }
    }

    // A view matrix that is a pure translation: enough to tell a rebased view
    // apart from the absolute one it was derived from.
    fn translation_view(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [x, y, z, 1.0],
        ]
    }

    // A bare StreamingState with no pools: enough to exercise the RAM valve's
    // sampling + stage machine without standing up the streamer worker threads.
    // `apply_byte_factor` is a no-op with no baselines, so the transitions run
    // exactly as they would with pools attached.
    fn empty_state() -> StreamingState {
        StreamingState {
            texture_streamer: None,
            mesh_streamer: None,
            mesh_stream_draw_indices: Vec::new(),
            chunk_stream: None,
            frame_count: 0,
            frames_in_flight: 2,
            texture_baseline_budget: None,
            mesh_baseline_budget: None,
            chunk_baseline_budget: None,
            pressure_stage: StreamPressureStage::None,
            pressure_factor: 1.0,
            last_sampled_rss: None,
        }
    }

    // Texture + mesh pools of two items each: item 0 sits on the camera's
    // origin and item 1 far out on +X, so a camera at either end orders the two
    // unambiguously. `resident_cap` chooses whether both fit at once (8) or the
    // second must displace the first (1). Mesh stream ids map to draw slots
    // 10 / 11 so an upload's routing is visible in the call log.
    fn pooled_state(resident_cap: usize) -> StreamingState {
        let centers = vec![vec![[0.0, 0.0, 0.0]], vec![[100.0, 0.0, 0.0]]];
        let mut state = empty_state();
        state.texture_streamer = Some(TextureStreamer::new(
            Arc::new(ConstTexture),
            centers.clone(),
            4,
            resident_cap,
        ));
        state.mesh_streamer = Some(MeshStreamer::new(
            Arc::new(ConstMesh),
            centers,
            4,
            resident_cap,
        ));
        state.mesh_stream_draw_indices = vec![10, 11];
        state
    }

    // A chunk pool over 16x16 chunks at the given near / far radii.
    fn chunk_state(source: Arc<dyn ChunkSource>, near: i32, far: i32) -> ChunkStreamState {
        ChunkStreamState {
            streamer: ChunkStreamer::new(source, near, far, 64, 16.0, 16.0),
            draws: std::collections::BTreeMap::new(),
            chunk_w: 16.0,
            chunk_d: 16.0,
            origin_chunk: ChunkCoord::new(0, 0),
            texture_slot: 0,
            normal_map_slot: crate::gfx::render_types::NO_NORMAL_MAP_SLOT,
            material: crate::gfx::render_types::MaterialUniforms::DEFAULT,
        }
    }

    // Drive frames until `done` holds. Yields (never sleeps) between frames
    // while the pools' workers decode; panics rather than hanging if the state
    // is never reached.
    fn drive_until(
        state: &mut StreamingState,
        backend: &mut MockBackend,
        cam: [f32; 3],
        done: impl Fn(&StreamingState) -> bool,
    ) {
        for _ in 0..MAX_DRIVE_SPINS {
            state.drive(backend, cam, IDENTITY4, false);
            if done(state) {
                return;
            }
            std::thread::yield_now();
        }
        panic!("streaming never reached the expected state");
    }

    // Owns the storage a PipelineContext borrows from, for the `step` tests.
    struct StepWorld {
        components: ComponentStorage,
        blob: BlobData,
        profile: FrameProfile,
        resources: Resources,
    }

    impl StepWorld {
        fn new() -> Self {
            Self {
                components: ComponentStorage::default(),
                blob: BlobData::empty(),
                profile: FrameProfile::default(),
                resources: Resources::new(),
            }
        }

        // Place a camera the step will read the absolute view + position from.
        fn with_camera(mut self, position: [f32; 3], view_matrix: [[f32; 4]; 4]) -> Self {
            self.components.push_typed(Camera3D {
                position,
                view_matrix,
                ..Camera3D::bake(Default::default())
            });
            self
        }

        fn park_backend(&mut self) {
            let (_recorded, backend) = recording_backend();
            self.resources
                .insert(ActiveRenderBackend(Some(Box::new(backend))));
        }

        fn step(&mut self) -> StepResult {
            let mut ctx = PipelineContext {
                components: &mut self.components,
                blob: &mut self.blob,
                profile: &mut self.profile,
                resources: &mut self.resources,
            };
            StreamingSystem::new().step(&mut ctx)
        }

        fn view(&self) -> CameraRelativeView {
            *self
                .resources
                .get::<CameraRelativeView>()
                .expect("camera-relative view published")
        }

        fn parked_state(&self) -> &StreamingState {
            self.resources
                .get::<StreamingState>()
                .expect("state parked again")
        }
    }

    #[test]
    fn sample_pressure_engages_and_publishes() {
        let mut s = empty_state();
        // RSS at 92% of a 1000-byte budget: stage 1 engages.
        let p = s.sample_pressure(Some(920), 1000).expect("published");
        assert_eq!(s.pressure_stage, StreamPressureStage::Gate);
        assert!(p.under_pressure);
        assert_eq!(p.rss_bytes, 920);
        assert_eq!(p.budget_bytes, 1000);
    }

    #[test]
    fn sample_pressure_escalates_when_rss_keeps_rising() {
        let mut s = empty_state();
        s.sample_pressure(Some(910), 1000);
        assert_eq!(s.pressure_stage, StreamPressureStage::Gate);
        // Still above engage and climbing: escalate to eviction.
        s.sample_pressure(Some(925), 1000);
        assert_eq!(s.pressure_stage, StreamPressureStage::Evict);
        assert!(s.pressure_factor < 1.0);
    }

    #[test]
    fn sample_pressure_releases_with_hysteresis() {
        let mut s = empty_state();
        s.sample_pressure(Some(970), 1000); // straight to evict
        assert_eq!(s.pressure_stage, StreamPressureStage::Evict);
        // In the hysteresis band (85%): still latched.
        s.sample_pressure(Some(850), 1000);
        assert_eq!(s.pressure_stage, StreamPressureStage::Evict);
        // Below the release mark: valve releases and restores the baseline.
        let p = s.sample_pressure(Some(700), 1000).expect("published");
        assert_eq!(s.pressure_stage, StreamPressureStage::None);
        assert_eq!(s.pressure_factor, 1.0);
        assert!(!p.under_pressure);
    }

    #[test]
    fn sample_pressure_is_inert_without_rss() {
        let mut s = empty_state();
        s.sample_pressure(Some(970), 1000);
        let stage_before = s.pressure_stage;
        // A failed RSS query publishes nothing and leaves the stage untouched.
        assert!(s.sample_pressure(None, 1000).is_none());
        assert_eq!(s.pressure_stage, stage_before);
    }

    // The translation column is the integer chunk delta scaled by chunk size;
    // the basis stays identity.
    #[test]
    fn chunk_model_matrix_offsets_by_chunk_delta() {
        let m = chunk_model_matrix(ChunkCoord::new(2, -3), ChunkCoord::new(0, 0), 16.0, 10.0);
        assert_eq!(m[3], [32.0, 0.0, -30.0, 1.0]);
        assert_eq!(m[0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(m[1], [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(m[2], [0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn chunk_model_matrix_origin_chunk_is_untranslated() {
        let c = ChunkCoord::new(5, 7);
        let m = chunk_model_matrix(c, c, 16.0, 16.0);
        assert_eq!(m[3], [0.0, 0.0, 0.0, 1.0]);
    }

    // Deep pressure scales every pool's budget off its own captured baseline,
    // and releasing restores each one exactly (not merely approximately).
    #[test]
    fn deep_pressure_scales_each_pool_budget_and_release_restores_the_baseline() {
        let mut state = pooled_state(8);
        state.chunk_stream = Some(chunk_state(Arc::new(ConstChunk), 0, 0));
        state.texture_baseline_budget = Some(4000);
        state.mesh_baseline_budget = Some(2000);
        state.chunk_baseline_budget = Some(1000);
        state.apply_byte_factor(1.0);

        state.sample_pressure(Some(970), 1000);
        assert_eq!(state.pressure_stage, StreamPressureStage::Evict);
        let factor = state.pressure_factor;
        assert!(factor < 1.0);
        let tex = state.texture_streamer.as_ref().unwrap().byte_budget();
        let mesh = state.mesh_streamer.as_ref().unwrap().byte_budget();
        let chunk = state.chunk_stream.as_ref().unwrap().streamer.byte_budget();
        assert_eq!(tex, Some(pressure::scale_budget(4000, factor)));
        assert_eq!(mesh, Some(pressure::scale_budget(2000, factor)));
        assert_eq!(chunk, Some(pressure::scale_budget(1000, factor)));

        state.sample_pressure(Some(100), 1000);
        assert_eq!(state.pressure_stage, StreamPressureStage::None);
        assert_eq!(
            state.texture_streamer.as_ref().unwrap().byte_budget(),
            Some(4000)
        );
        assert_eq!(
            state.mesh_streamer.as_ref().unwrap().byte_budget(),
            Some(2000)
        );
        assert_eq!(
            state.chunk_stream.as_ref().unwrap().streamer.byte_budget(),
            Some(1000)
        );
    }

    // A pool with no captured baseline runs count-only: the valve has nothing
    // to scale, so it must not invent a budget for it.
    #[test]
    fn count_only_pools_gain_no_byte_budget_under_pressure() {
        let mut state = pooled_state(8);
        state.chunk_stream = Some(chunk_state(Arc::new(ConstChunk), 0, 0));

        state.sample_pressure(Some(970), 1000);
        assert_eq!(state.pressure_stage, StreamPressureStage::Evict);
        assert_eq!(state.texture_streamer.as_ref().unwrap().byte_budget(), None);
        assert_eq!(state.mesh_streamer.as_ref().unwrap().byte_budget(), None);
        assert_eq!(
            state.chunk_stream.as_ref().unwrap().streamer.byte_budget(),
            None
        );
    }

    #[test]
    fn streaming_stats_reports_every_active_pool() {
        let mut state = pooled_state(8);
        state.chunk_stream = Some(chunk_state(Arc::new(ConstChunk), 0, 0));
        state
            .texture_streamer
            .as_mut()
            .unwrap()
            .set_byte_budget(Some(4000));

        let stats = state.streaming_stats();
        assert_eq!(stats.texture, Some((0, 0, 2)));
        assert_eq!(stats.mesh, Some((0, 0, 2)));
        assert_eq!(stats.chunk, Some((0, 0)));
        assert_eq!(stats.texture_bytes, Some((0, 4000)));
        // A count-only pool reports a zero budget rather than dropping the row.
        assert_eq!(stats.mesh_bytes, Some((0, 0)));
        assert_eq!(stats.chunk_bytes, Some((0, 0)));
    }

    #[test]
    fn streaming_stats_reports_nothing_without_pools() {
        let stats = empty_state().streaming_stats();
        assert!(stats.texture.is_none());
        assert!(stats.mesh.is_none());
        assert!(stats.chunk.is_none());
        assert!(stats.texture_bytes.is_none());
        assert!(stats.mesh_bytes.is_none());
        assert!(stats.chunk_bytes.is_none());
    }

    // The pools have no Debug of their own, so StreamingState's is hand-written
    // to report which are active rather than trying to dump them.
    #[test]
    fn debug_reports_the_active_pools_and_the_frame_clock() {
        let mut state = pooled_state(8);
        state.frame_count = 7;
        state.pressure_stage = StreamPressureStage::Gate;

        let s = format!("{state:?}");
        assert!(s.contains("frame_count: 7"), "{s}");
        assert!(s.contains("texture: true"), "{s}");
        assert!(s.contains("mesh: true"), "{s}");
        assert!(s.contains("chunk: false"), "{s}");
        assert!(s.contains("pressure: Gate"), "{s}");
    }

    #[test]
    fn drive_advances_the_frame_clock() {
        let (_recorded, mut backend) = recording_backend();
        let mut state = empty_state();
        state.drive(&mut backend, [0.0; 3], IDENTITY4, false);
        state.drive(&mut backend, [0.0; 3], IDENTITY4, false);
        assert_eq!(state.frame_count, 2);
    }

    // Behind an opaque menu the world is not drawn, so no pool dispatches: both
    // stay fully unloaded rather than paying for loads nothing will show.
    #[test]
    fn a_hidden_world_dispatches_no_loads() {
        let (_recorded, mut backend) = recording_backend();
        let mut state = pooled_state(8);
        state.drive(&mut backend, [0.0; 3], IDENTITY4, true);
        assert_eq!(state.texture_streamer.as_ref().unwrap().stats(), (0, 0, 2));
        assert_eq!(state.mesh_streamer.as_ref().unwrap().stats(), (0, 0, 2));
    }

    #[test]
    fn a_visible_world_dispatches_texture_and_mesh_loads() {
        let (_recorded, mut backend) = recording_backend();
        let mut state = pooled_state(8);
        state.drive(&mut backend, [0.0; 3], IDENTITY4, false);
        // Dispatch moves an item off Unloaded the same frame it is planned.
        assert!(state.texture_streamer.as_ref().unwrap().stats().2 < 2);
        assert!(state.mesh_streamer.as_ref().unwrap().stats().2 < 2);
    }

    // Stage 1 of the RAM valve holds residency where it is: no new dispatch.
    #[test]
    fn the_gate_stage_freezes_new_loads() {
        let (_recorded, mut backend) = recording_backend();
        let mut state = pooled_state(8);
        state.chunk_stream = Some(chunk_state(Arc::new(ConstChunk), 0, 0));
        state.pressure_stage = StreamPressureStage::Gate;

        state.drive(&mut backend, [0.0; 3], IDENTITY4, false);
        assert_eq!(state.texture_streamer.as_ref().unwrap().stats(), (0, 0, 2));
        assert_eq!(state.mesh_streamer.as_ref().unwrap().stats(), (0, 0, 2));
        assert_eq!(
            state.chunk_stream.as_ref().unwrap().streamer.stats(),
            (0, 0)
        );
    }

    // Stage 2 keeps planning under the reduced budget, so the planner can still
    // shed residents; freezing it instead would strand the pools over budget.
    #[test]
    fn the_evict_stage_keeps_planning() {
        let (_recorded, mut backend) = recording_backend();
        let mut state = pooled_state(8);
        state.pressure_stage = StreamPressureStage::Evict;
        state.drive(&mut backend, [0.0; 3], IDENTITY4, false);
        assert!(state.texture_streamer.as_ref().unwrap().stats().2 < 2);
        assert!(state.mesh_streamer.as_ref().unwrap().stats().2 < 2);
    }

    // Completed loads route to the backend: a texture by pool slot, a mesh
    // through its stream-id -> draw-slot map.
    #[test]
    fn completed_loads_are_uploaded_to_the_backend() {
        let (recorded, mut backend) = recording_backend();
        let mut state = pooled_state(8);
        drive_until(&mut state, &mut backend, [0.0; 3], |s| {
            s.texture_streamer.as_ref().unwrap().stats().0 == 2
                && s.mesh_streamer.as_ref().unwrap().stats().0 == 2
        });

        let s = recorded.lock().unwrap();
        assert!(s.saw(&Call::UpdateTextureSlot {
            slot: 0,
            w: 1,
            h: 1
        }));
        assert!(s.saw(&Call::UpdateTextureSlot {
            slot: 1,
            w: 1,
            h: 1
        }));
        assert!(s.saw(&Call::UploadMesh {
            draw_idx: 10,
            vertices: 3,
            indices: 3,
        }));
        assert!(s.saw(&Call::UploadMesh {
            draw_idx: 11,
            vertices: 3,
            indices: 3,
        }));
    }

    // A streamed mesh with no draw slot has nowhere to upload to; the drive
    // must swallow it rather than mis-routing the geometry onto another draw.
    #[test]
    fn a_streamed_mesh_without_a_draw_slot_uploads_nothing() {
        let (recorded, mut backend) = recording_backend();
        let mut state = pooled_state(8);
        state.mesh_stream_draw_indices.clear();
        drive_until(&mut state, &mut backend, [0.0; 3], |s| {
            s.mesh_streamer.as_ref().unwrap().stats().0 == 2
        });
        assert!(
            !recorded
                .lock()
                .unwrap()
                .calls
                .iter()
                .any(|c| matches!(c, Call::UploadMesh { .. }))
        );
    }

    // Over the resident cap, the pools shed the item the camera has left
    // behind so the nearer one can take its place.
    #[test]
    fn moving_the_camera_evicts_the_now_distant_item_over_the_cap() {
        let (recorded, mut backend) = recording_backend();
        // Cap of 1: only one texture / mesh may be resident at a time.
        let mut state = pooled_state(1);
        drive_until(&mut state, &mut backend, [0.0; 3], |s| {
            s.texture_streamer.as_ref().unwrap().stats().0 == 1
                && s.mesh_streamer.as_ref().unwrap().stats().0 == 1
        });
        recorded.lock().unwrap().calls.clear();

        // Item 1's center is now the near one, so item 0 is displaced.
        state.drive(&mut backend, [100.0, 0.0, 0.0], IDENTITY4, false);
        let s = recorded.lock().unwrap();
        assert!(s.saw(&Call::EvictTextureSlot(0)), "{:?}", s.calls);
        assert!(s.saw(&Call::EvictMesh(10)), "{:?}", s.calls);
    }

    #[test]
    fn without_chunk_streaming_the_view_and_camera_stay_absolute() {
        let (_recorded, mut backend) = recording_backend();
        let mut state = empty_state();
        let view = translation_view(-40.0, -5.0, 40.0);
        let cam = [40.0, 5.0, -40.0];
        assert_eq!(state.drive(&mut backend, cam, view, false), (view, cam));
    }

    // An unbounded world renders from small coordinates: both the view and the
    // camera are rebased onto the camera's own chunk.
    #[test]
    fn chunk_streaming_rebases_the_view_onto_the_camera_chunk() {
        let (_recorded, mut backend) = recording_backend();
        let mut state = empty_state();
        state.chunk_stream = Some(chunk_state(Arc::new(FailingChunk), 0, 0));
        // 16-unit chunks: floor(40/16) = 2, floor(-40/16) = -3.
        let cam = [40.0, 5.0, -40.0];
        let view = translation_view(-40.0, -5.0, 40.0);
        let (out_view, out_cam) = state.drive(&mut backend, cam, view, false);

        let origin = [32.0, 0.0, -48.0];
        assert_eq!(out_cam, [8.0, 5.0, 8.0]);
        assert_eq!(
            out_view,
            crate::gfx::chunk_coord::camera_relative_view(view, cam, origin)
        );
        assert_ne!(out_view, view, "the rebase actually rewrote the view");
        assert_eq!(
            state.chunk_stream.as_ref().unwrap().origin_chunk,
            ChunkCoord::new(2, -3)
        );
    }

    // Crossing a chunk boundary moves the render origin and re-places every
    // resident chunk against it; staying put must not re-push identical models.
    #[test]
    fn crossing_into_a_new_chunk_rebases_resident_chunk_models() {
        let (recorded, mut backend) = recording_backend();
        let mut state = empty_state();
        let mut cs = chunk_state(Arc::new(FailingChunk), 1, 1);
        // Stand in for two chunks already uploaded at draw slots 3 and 4.
        cs.draws.insert(ChunkCoord::new(0, 0), 3);
        cs.draws.insert(ChunkCoord::new(1, 0), 4);
        state.chunk_stream = Some(cs);

        state.drive(&mut backend, [20.0, 0.0, 0.0], IDENTITY4, false);
        {
            let s = recorded.lock().unwrap();
            assert!(s.saw(&Call::SetChunkModel(3)));
            assert!(s.saw(&Call::SetChunkModel(4)));
        }
        assert_eq!(
            state.chunk_stream.as_ref().unwrap().origin_chunk,
            ChunkCoord::new(1, 0)
        );

        recorded.lock().unwrap().calls.clear();
        state.drive(&mut backend, [21.0, 0.0, 0.0], IDENTITY4, false);
        assert!(
            !recorded
                .lock()
                .unwrap()
                .calls
                .iter()
                .any(|c| matches!(c, Call::SetChunkModel(_)))
        );
    }

    // A chunk that leaves the view window releases its draw slot on the
    // backend and stops being tracked as resident.
    #[test]
    fn chunks_leaving_the_view_window_are_removed_from_the_backend() {
        let (recorded, mut backend) = recording_backend();
        let mut state = empty_state();
        let mut cs = chunk_state(Arc::new(FailingChunk), 0, 0);
        // Stand in for chunk (0, 0) already uploaded at draw slot 9.
        cs.draws.insert(ChunkCoord::new(0, 0), 9);
        state.chunk_stream = Some(cs);

        // Frame 1 at the origin puts (0, 0) in the window.
        state.drive(&mut backend, [0.0; 3], IDENTITY4, false);
        // Frame 2 far east: (0, 0) is past the evict band.
        state.drive(&mut backend, [800.0, 0.0, 0.0], IDENTITY4, false);

        assert!(recorded.lock().unwrap().saw(&Call::RemoveChunkMesh(9)));
        assert!(
            !state
                .chunk_stream
                .as_ref()
                .unwrap()
                .draws
                .contains_key(&ChunkCoord::new(0, 0))
        );
    }

    // A generated chunk is added to the backend and its returned draw slot
    // recorded, so a later rebase or eviction can find it.
    #[test]
    fn a_generated_chunk_is_added_and_its_draw_slot_tracked() {
        let (recorded, mut backend) = recording_backend();
        let mut state = empty_state();
        state.chunk_stream = Some(chunk_state(Arc::new(ConstChunk), 0, 0));
        drive_until(&mut state, &mut backend, [0.0; 3], |s| {
            s.chunk_stream.as_ref().unwrap().streamer.stats().0 == 1
        });

        let cs = state.chunk_stream.as_ref().unwrap();
        assert_eq!(cs.draws.get(&ChunkCoord::new(0, 0)), Some(&0));
        assert!(recorded.lock().unwrap().saw(&Call::AddChunkMesh));
    }

    #[test]
    fn a_step_without_parked_state_publishes_no_view() {
        let mut w = StepWorld::new();
        assert_eq!(w.step(), StepResult::Continue);
        assert!(w.resources.get::<CameraRelativeView>().is_none());
    }

    // Init succeeded but the backend is gone: the draw still needs a view, so
    // the absolute camera is published and the state stays parked.
    #[test]
    fn a_step_without_a_backend_still_publishes_the_absolute_view() {
        let view = translation_view(-1.0, -2.0, -3.0);
        let mut w = StepWorld::new().with_camera([1.0, 2.0, 3.0], view);
        w.resources.insert(empty_state());

        assert_eq!(w.step(), StepResult::Continue);
        assert_eq!(w.view().cam_pos, [1.0, 2.0, 3.0]);
        assert_eq!(w.view().view, view);
        assert_eq!(w.parked_state().frame_count, 0, "no frame was driven");
    }

    #[test]
    fn a_step_without_a_camera_publishes_the_identity_view() {
        let mut w = StepWorld::new();
        w.resources.insert(empty_state());
        w.park_backend();

        w.step();
        assert_eq!(w.view().view, IDENTITY4);
        assert_eq!(w.view().cam_pos, [0.0; 3]);
    }

    // The backend is taken for the step and parked again for the systems that
    // follow, and the frame clock ticks once per step.
    #[test]
    fn a_step_drives_the_backend_and_parks_it_again() {
        let mut w = StepWorld::new().with_camera([0.0; 3], IDENTITY4);
        w.resources.insert(pooled_state(8));
        w.park_backend();

        w.step();
        assert!(
            w.resources
                .get::<ActiveRenderBackend>()
                .is_some_and(|slot| slot.0.is_some())
        );
        assert_eq!(w.parked_state().frame_count, 1);
        assert!(
            w.parked_state()
                .texture_streamer
                .as_ref()
                .unwrap()
                .stats()
                .2
                < 2,
            "the pools were driven"
        );
    }

    // The overlay's flag is peeked, not taken: streaming pauses behind an
    // opaque menu, and GraphicsSystem still finds the frame later this tick.
    #[test]
    fn an_opaque_overlay_suspends_streaming_without_consuming_the_frame() {
        let mut w = StepWorld::new().with_camera([0.0; 3], IDENTITY4);
        w.resources.insert(pooled_state(8));
        w.park_backend();
        w.resources.insert(OverlayFrame {
            world_hidden: true,
            ..Default::default()
        });

        w.step();
        assert_eq!(
            w.parked_state().texture_streamer.as_ref().unwrap().stats(),
            (0, 0, 2)
        );
        assert!(w.resources.get::<OverlayFrame>().is_some());
    }

    // RSS is a syscall, so the valve only samples on its throttled cadence.
    #[test]
    fn ram_pressure_is_sampled_only_on_the_throttled_cadence() {
        let mut w = StepWorld::new();
        let mut state = empty_state();
        state.frame_count = 1;
        w.resources.insert(state);
        // A 1 MiB ceiling is under any real process RSS, so a sample that ran
        // would certainly engage the valve.
        w.resources
            .insert(crate::app::budget::MemoryBudget::compute(None, 1));

        w.step();
        assert!(w.resources.get::<StreamingPressure>().is_none());
    }

    // No published ceiling means no valve: streaming stays on its byte-budget
    // policy and nothing is reported.
    #[test]
    fn ram_pressure_is_not_sampled_without_a_memory_budget() {
        let mut w = StepWorld::new();
        w.resources.insert(empty_state());
        w.step();
        assert!(w.resources.get::<StreamingPressure>().is_none());
    }

    #[test]
    fn an_rss_sample_over_the_memory_budget_publishes_engaged_pressure() {
        let mut w = StepWorld::new();
        w.resources.insert(empty_state());
        w.resources
            .insert(crate::app::budget::MemoryBudget::compute(None, 1));

        w.step();
        // The valve is inert (and publishes nothing) where RSS cannot be read.
        match crate::app::sysmem::process_resident_bytes() {
            Some(_) => {
                let p = w.resources.get::<StreamingPressure>().expect("published");
                assert!(p.under_pressure);
                assert_eq!(p.budget_bytes, 1024 * 1024);
                assert_ne!(w.parked_state().pressure_stage, StreamPressureStage::None);
            }
            None => assert!(w.resources.get::<StreamingPressure>().is_none()),
        }
    }
}
