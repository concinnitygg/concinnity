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

const IDENTITY4: [[f32; 4]; 4] = crate::gfx::draw_list::IDENTITY4;

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
}

impl std::fmt::Debug for StreamingState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingState")
            .field("frame_count", &self.frame_count)
            .field("texture", &self.texture_streamer.is_some())
            .field("mesh", &self.mesh_streamer.is_some())
            .field("chunk", &self.chunk_stream.is_some())
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
        // Drive albedo-texture streaming: re-score every slot by camera
        // distance, dispatch this frame's background loads within budget, then
        // apply completed uploads + evictions. Each backend's
        // update_texture_slot rewrites whichever descriptors / argument-buffers
        // sample that slot so it takes effect on this same draw_frame.
        if !world_hidden && let Some(streamer) = &mut self.texture_streamer {
            streamer.update_scores(cam_pos, self.frame_count);
            for slot in streamer.plan_and_dispatch() {
                if let Err(e) = backend.evict_texture_slot(slot) {
                    tracing::warn!("StreamingSystem: texture evict slot {}: {}", slot, e);
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

    // `(resident, pending, unloaded)` counts for each active streaming pool.
    // Consumed only by the `cn debug` binary's `streaming` command, so it reads
    // as dead code in a plain library build.
    #[allow(dead_code)]
    pub(crate) fn streaming_stats(&self) -> StreamingStats {
        StreamingStats {
            texture: self.texture_streamer.as_ref().map(|s| s.stats()),
            mesh: self.mesh_streamer.as_ref().map(|s| s.stats()),
            chunk: self.chunk_stream.as_ref().map(|cs| cs.streamer.stats()),
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
