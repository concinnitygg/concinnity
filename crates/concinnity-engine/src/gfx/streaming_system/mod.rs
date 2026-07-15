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
    // when the pool runs count-only. The RAM back-off valve reduces the live
    // budget below this under deep pressure and restores it exactly on release.
    pub(crate) texture_baseline_budget: Option<u64>,
    pub(crate) mesh_baseline_budget: Option<u64>,
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
            streamer.drain_completed(self.frame_count, |slot, w, h, px| {
                if let Err(e) = backend.update_texture_slot(slot, w, h, px) {
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
    use crate::gfx::chunk_coord::ChunkCoord;
    use pressure::StreamPressureStage;

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
            pressure_stage: StreamPressureStage::None,
            pressure_factor: 1.0,
            last_sampled_rss: None,
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
}
