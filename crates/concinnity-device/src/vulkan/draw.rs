// Frame recording for the Vulkan backend.
// Encodes the shadow pass, main scene pass, and post stack into a single
// command buffer. Called from VkContext::draw_frame each tick. Every
// pass dispatches through the render-graph executor; see
// `vulkan/graph_exec.rs` and `vulkan/composite.rs`.

use ash::vk;
use concinnity_core::gfx::transform::IDENTITY;
use concinnity_core::gfx::transform::mat4_inverse;

use crate::gfx::render_graph::{FrameGraphInputs, build_frame_graph};
use crate::gfx::render_types::{LightUniforms, LineVertex, ShadowUniforms, TextDrawCall};

use super::context::VkContext;
use super::graph_exec::GraphFrameParams;
use concinnity_core::gfx::projection::perspective_rh;
use concinnity_core::gfx::transform::mat4_mul;

// `ViewUniforms` (the std140 main-pass `ViewBlock` UBO) is a GPU-free layout
// struct that lives in `core::render`; re-export it so
// `crate::vulkan::draw::ViewUniforms` is unchanged for the passes that fill it.
pub(in crate::vulkan) use concinnity_core::render::uniforms::ViewUniforms;

// One term of the Halton low-discrepancy sequence, drives the sub-pixel
// projection jitter so successive TAA frames sample slightly different
// positions. Mirrors `halton` in metal/draw.rs.
fn halton(mut index: u32, base: u32) -> f32 {
    let mut result = 0.0_f32;
    let mut f = 1.0_f32;
    while index > 0 {
        f /= base as f32;
        result += f * (index % base) as f32;
        index /= base;
    }
    result
}

// Where `record_frame` records this frame's GPU work: the outer "end" command
// buffer, the acquired swapchain image, and the frame-in-flight slot the
// per-frame resources (query pool block, view UBO, cull buffers) index into.
#[derive(Clone, Copy)]
pub(super) struct RecordFrameTargets {
    pub cmd: vk::CommandBuffer,
    pub image_index: u32,
    pub frame_idx: usize,
}

// Per-frame camera / view state plus the overlay text drawn with it. Drives the
// projection, cascade selection, sub-pixel jitter, and view UBO for this frame.
pub(super) struct RecordFrameView<'a> {
    pub elapsed: f32,
    pub fov_y_radians: f32,
    pub near: f32,
    pub far: f32,
    pub cam_pos: [f32; 3],
    pub text_calls: &'a [TextDrawCall],
    // This frame's expanded line ribbons. Empty whenever nothing published
    // lines, which also drops the pass from the graph.
    pub lines: &'a [LineVertex],
}

impl VkContext {
    // Rebuild this frame's `GpuObjectData` storage buffer for the bindless
    // static pass: one 144-byte record per build-time `DrawObject`, indexed
    // by object id. Streamed `VoxelWorld` chunks (past `draw.n_objects`) are
    // skipped: they render through the legacy pipeline. The pool indices
    // address the shared handle-indexed texture pool: albedo = `texture_slot`,
    // normal = the normal map's own handle (or the flat-normal fallback slot
    // for a normal-less draw). Rebuilt every frame so `update_model` /
    // `update_visibility` edits are reflected; a no-op when bindless is off.
    fn build_object_buffer(&self, frame_idx: usize) {
        let Some(buf) = self.cull.object_buffers.get(frame_idx) else {
            return;
        };
        self.build_object_records_into(buf);
    }

    // Write the bindless `GpuObjectData` records (static + streamed-chunk +
    // skinned-tail) into `buf`. Factored out of `build_object_buffer` so the
    // reflection-probe capture can build the same records into its own bake-owned
    // buffer (the instance tail is left untouched, so a bake buffer must be zeroed
    // first -- a zero record is a disabled draw the cull kernel skips, which is how
    // the probe omits instanced geometry in V1).
    pub(in crate::vulkan) fn build_object_records_into(
        &self,
        buf: &super::allocator::PooledBuffer,
    ) {
        use crate::gfx::render_types::{
            GpuObjectData, albedo_pool_index, normal_pool_index, pack_object_record,
            pack_skinned_record,
        };
        let texture_count = self.textures.len() as u32;
        let stride = std::mem::size_of::<GpuObjectData>();
        for (i, obj) in self
            .draw
            .objects
            .iter()
            .take(self.draw.n_objects)
            .enumerate()
        {
            let albedo = albedo_pool_index(obj.texture_slot, texture_count);
            let normal = normal_pool_index(obj.normal_map_slot, texture_count);
            let rec = pack_object_record(obj, albedo, normal);
            buf.write_val(i * stride, &rec);
        }

        // Streamed chunks: one record each in the reserved region at
        // `[chunk_record_base() + k]`, packed like a static object (chunk geometry
        // already lives in the shared VB/IB with the chunk's `base_vertex`, so they
        // ride the static + instance prefix indirect draw). Per-chunk flat-pool
        // texture indices give per-chunk materials. A non-resident / unused slot's
        // stale record here is never read -- `build_draw_args_buffer` disables it,
        // and the cull kernel skips `objects[i]` for a disabled record.
        let chunk_base = self.chunk_record_base();
        self.for_each_chunk_record(|k, obj| {
            let albedo = albedo_pool_index(obj.texture_slot, texture_count);
            let normal = normal_pool_index(obj.normal_map_slot, texture_count);
            let rec = pack_object_record(obj, albedo, normal);
            buf.write_val((chunk_base + k) * stride, &rec);
        });

        // Skinned objects: one record each in the reserved tail at
        // `[skinned_record_base(), cull_count())`. `model = obj.model` (applied
        // after the per-frame skin deform), flat-pool texture indices like a static
        // object, and a padded bind-pose AABB so the cull kernel can frustum/Hi-Z
        // test them. Drawn by the main pass's 2nd indirect draw. `take(n_skinned)`
        // no-ops when the fold is inactive.
        let skinned_base = self.skinned_record_base();
        for (k, obj) in self
            .skinned
            .draw_objects
            .iter()
            .take(self.draw.n_skinned)
            .enumerate()
        {
            let albedo = albedo_pool_index(obj.texture_slot, texture_count);
            let normal = normal_pool_index(obj.normal_map_slot, texture_count);
            let rec = pack_skinned_record(obj, albedo, normal);
            buf.write_val((skinned_base + k) * stride, &rec);
        }
    }

    // Rebuild this frame's `GpuDrawArgs` storage buffer for the GPU-cull
    // compute kernel: one 16-byte record per build-time `DrawObject`, carrying
    // the indexed-draw arguments the kernel encodes plus the per-frame
    // cull-decision bits (`update_visibility` / streaming residency). Streamed
    // chunks (past `draw.n_objects`) are skipped; a no-op when bindless is off.
    // The per-object `(index_offset, index_count)` is the active LOD slice
    // picked by camera distance, so the bindless main pass renders the
    // chosen LOD with no shader-side change. Mirrors `directx/cull.rs`.
    fn build_draw_args_buffer(&self, frame_idx: usize, cam_pos: [f32; 3]) {
        let Some(buf) = self.cull.draw_args_buffers.get(frame_idx) else {
            return;
        };
        self.build_draw_args_records_into(buf, cam_pos);
    }

    // Write the GPU-cull `GpuDrawArgs` records (static + streamed-chunk +
    // skinned-tail, the per-object active-LOD slice picked by distance from
    // `cam_pos`) into `buf`. Factored out of `build_draw_args_buffer` so the
    // reflection-probe capture can build the same args into its own bake-owned
    // buffer against the probe eye. The instance tail is left untouched (a zeroed
    // bake buffer keeps it disabled = skipped).
    pub(in crate::vulkan) fn build_draw_args_records_into(
        &self,
        buf: &super::allocator::PooledBuffer,
        cam_pos: [f32; 3],
    ) {
        use crate::gfx::render_types::{GpuDrawArgs, draw_args_bucket_bits, draw_args_flags};
        let stride = std::mem::size_of::<GpuDrawArgs>();
        // A see-through glass mesh (Layer 2) is disabled in the opaque pass when
        // the RT path is live: it draws in the transparent pass instead. Clearing
        // ENABLED makes the cull kernel reset its command to a no-op (the same
        // path invisible / non-resident objects take), so it neither draws opaque
        // nor occludes the refraction snapshot. The object keeps its slot, so
        // every parallel cull / object-buffer / prev-model index stays intact.
        let mesh_glass_active = self.mesh_glass_active();
        for (i, obj) in self
            .draw
            .objects
            .iter()
            .take(self.draw.n_objects)
            .enumerate()
        {
            // Per-frame active LOD pick. Objects with no alternates fall
            // straight through to LOD0.
            let d = crate::gfx::lod::camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            let opaque_visible =
                obj.visible && !(mesh_glass_active && obj.material.see_through != 0);
            let rec = GpuDrawArgs {
                index_count: index_count as u32,
                index_offset: index_offset as u32,
                base_vertex: obj.base_vertex as u32,
                // The record's shader bucket rides the upper flag bits so the
                // cull kernel can route its command into that bucket's region.
                flags: draw_args_flags(opaque_visible, obj.resident, obj.cullable())
                    | draw_args_bucket_bits(obj.shader_bucket),
            };
            buf.write_val(i * stride, &rec);
        }

        // Streamed chunks: one draw-arg each in the reserved region at
        // `[chunk_record_base() + k]`. Chunk geometry lives in the shared VB/IB, so
        // the args carry the chunk's own `base_vertex` + index slice and the chunk
        // rides the static + instance prefix indirect draw. Chunks are non-cullable
        // (NaN AABB), so a resident chunk draws unconditionally; a freed slot's
        // `resident` clear disables it. The unused reserve tail is disabled.
        let chunk_base = self.chunk_record_base();
        let n_resident_chunks = self.for_each_chunk_record(|k, obj| {
            // Chunks have no LOD alternates; `active_lod(0.0)` returns the base slice
            // (and avoids a NaN camera distance from the chunk's NaN AABB).
            let (index_offset, index_count) = obj.active_lod(0.0);
            let rec = GpuDrawArgs {
                index_count: index_count as u32,
                index_offset: index_offset as u32,
                base_vertex: obj.base_vertex as u32,
                flags: draw_args_flags(obj.visible, obj.resident, obj.cullable()),
            };
            buf.write_val((chunk_base + k) * stride, &rec);
        });
        // Disable the unused chunk reserve tail so vacated / never-used slots draw
        // nothing (the cull kernel skips `objects[i]` for an ENABLED-clear record).
        let disabled = GpuDrawArgs {
            index_count: 0,
            index_offset: 0,
            base_vertex: 0,
            flags: 0,
        };
        for k in n_resident_chunks..self.draw.n_chunk {
            buf.write_val((chunk_base + k) * stride, &disabled);
        }

        // Skinned objects: one record each in the reserved tail. The main pass's
        // 2nd indirect draw binds the per-frame deformed VB + the skinned IB,
        // so `base_vertex = 0` (the deformed buffer mirrors global skinned indexing)
        // and the active-LOD slice is the element offset into the skinned IB.
        // Skinned objects carry a finite padded bind-pose AABB (`pack_skinned_record`),
        // so they are cullable + resident. `take(n_skinned)` no-ops when inactive.
        let skinned_base = self.skinned_record_base();
        for (k, obj) in self
            .skinned
            .draw_objects
            .iter()
            .take(self.draw.n_skinned)
            .enumerate()
        {
            let d = crate::gfx::lod::skinned_camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            let rec = GpuDrawArgs {
                index_count: index_count as u32,
                index_offset: index_offset as u32,
                base_vertex: 0,
                flags: draw_args_flags(obj.visible, true, true),
            };
            buf.write_val((skinned_base + k) * stride, &rec);
        }
    }

    pub(super) fn record_frame(
        &mut self,
        targets: RecordFrameTargets,
        view: RecordFrameView<'_>,
        world_hidden: bool,
    ) -> Result<Vec<vk::CommandBuffer>, String> {
        let RecordFrameTargets {
            cmd,
            image_index,
            frame_idx,
        } = targets;
        let RecordFrameView {
            elapsed,
            fov_y_radians,
            near,
            far,
            cam_pos,
            text_calls,
            lines,
        } = view;
        let device = self.device.clone();
        let device = &device;
        // The scene rasterises at render resolution (== swapchain extent unless
        // upscaling). Cascade / projection aspect, the HDR graph dims, and the
        // sub-pixel jitter all derive from this, not the display extent.
        let extent = self.render_extent;

        // Profiler-overlay timestamp pair. The pool slot for this frame is
        // reset (the matching `get_query_pool_results` already ran at the top
        // of `draw_frame`, after the fence wait that gated the previous trip's
        // writes), then the start tick is recorded as the first cmd-buffer
        // op. The matching end tick is written just before
        // `end_command_buffer` returns control. Mirrors the DirectX
        // EndQuery(TIMESTAMP) + ResolveQueryData pattern.
        // Outer "start" command buffer: the leading timestamp (the frame's
        // first GPU op), recorded into its own buffer so it can be submitted
        // before the per-pass buffers. The query-pool reset must precede every
        // pass, hence it lives here at the head of the batch.
        let start_cmd = self.commands.start_command_buffers[frame_idx];
        // SAFETY: `cmd` belongs to this frame slot, whose fence was already waited on, so it is not
        // in flight; reset then begin puts it in the recording state, which is what the subsequent
        // recording requires.
        unsafe {
            device
                .reset_command_buffer(start_cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("reset start cmd buf: {e}"))?;
            device
                .begin_command_buffer(
                    start_cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| format!("begin start cmd buf: {e}"))?;
        }
        if let Some(pool) = self.timestamp_query_pool {
            // Reset this frame's whole timestamp block (whole-frame pair + every
            // per-pass pair) here, before any per-pass buffer writes into it.
            // `start_cmd` is submitted first, so the reset precedes every write in
            // queue order. The whole-frame start goes in the block's first slot;
            // each pass writes its own pair (see graph_exec); the whole-frame end
            // is the block's second slot, written in the end buffer below.
            let block_base = super::pass_timing::frame_block_base(frame_idx);
            let (wf_start, _) = super::pass_timing::whole_frame_pair(frame_idx);
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                device.cmd_reset_query_pool(
                    start_cmd,
                    pool,
                    block_base,
                    super::pass_timing::SLOTS_PER_FRAME as u32,
                );
                device.cmd_write_timestamp(
                    start_cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    pool,
                    wf_start,
                );
            }
        }
        // Hardware ray-traced reflections: rebuild the TLAS for moved props (when
        // the dynamic mode + dirty gate call for it) onto the start buffer, which
        // is submitted before every per-pass trace, then re-point this frame's RT
        // descriptor set at the live TLAS + geometry table. A no-op when RT is
        // off. Recorded here (not on a per-pass worker) because it needs
        // `&mut self`; the start buffer carries it within the frame's submit batch.
        self.rt_dynamic_update(start_cmd, frame_idx);
        // SAFETY: `cmd` is in the recording state, which is what `end_command_buffer` requires.
        unsafe {
            device
                .end_command_buffer(start_cmd)
                .map_err(|e| format!("end start cmd buf: {e}"))?;
        }

        // Recompute cascade VPs + splits from the current camera + light, and
        // push the result to the shadow UBO so both passes see the same data.
        let cascade_aspect = if extent.height == 0 {
            1.0
        } else {
            extent.width as f32 / extent.height as f32
        };
        if self.shadow.pipeline.is_some() {
            let fresh =
                crate::gfx::csm::compute_shadow_uniforms(crate::gfx::csm::ShadowUniformInputs {
                    view: self.view.matrix,
                    cam_pos,
                    fov_y_rad: fov_y_radians,
                    aspect: cascade_aspect,
                    near,
                    shadow_distance: (self.shadow.distance as f32).min(far),
                    light_dir_to_source: self.shadow.light_dir,
                    shadow_map_size: self.shadow.map_size,
                    active_cascades: self.shadow.cascades,
                });
            // Advance the cascade schedule and refresh only this frame's
            // cascades' light VPs; skipped cascades keep the VP + depth their
            // slice was last rendered with, so the Main pass samples each cascade
            // consistently. Splits depend only on the camera range (not which
            // cascades render), so always refresh. encode_shadow_pass
            // re-rasterizes only the masked slices.
            let update = self.shadow.update;
            let mask = self
                .shadow
                .scheduler
                .next_mask(update, self.shadow.cascades);
            self.shadow.render_mask = mask;
            self.shadow.uniforms.cascade_splits = fresh.cascade_splits;
            self.shadow.uniforms.active_cascades = fresh.active_cascades;
            for i in 0..crate::gfx::render_types::NUM_SHADOW_CASCADES {
                if mask & (1u32 << i) != 0 {
                    self.shadow.uniforms.light_vps[i] = fresh.light_vps[i];
                }
            }
            upload_shadow_uniforms(&self.shadow.ubos[frame_idx], &self.shadow.uniforms);
        }

        // Spot shadow refresh schedule. Prime-then-round-robin over the slices,
        // so N shadowed spots cost one extra depth render per frame rather than
        // N. No uniform refresh: the projections are static and were baked at
        // init. A no-op (mask stays 0) when the world has no shadowed spot.
        self.spot_shadow.advance(matches!(
            self.shadow.update,
            crate::components::ShadowUpdate::EveryFrame
        ));

        // Push this frame's skinning matrices into the per-frame joint buffers
        // before the skinned shadow + main passes read them. No-op when no
        // SkinnedMesh is declared.
        self.upload_joint_matrices(frame_idx);
        // Push this frame's morph weights into the per-frame weight buffers the
        // skin fold reads. No-op when no SkinnedMesh carries morph targets.
        self.upload_morph_weights(frame_idx);

        // Auto-exposure: step the EMA from a previous frame's GPU
        // measurement before any pipeline reads `post_process.exposure`.
        // The fence wait at the top of `draw_frame` already gated the
        // GPU work that wrote this slot's readback, so the value is
        // committed. No-op when auto-exposure is disabled.
        self.update_auto_exposure(elapsed, frame_idx);

        // Line resources: built on the first frame that publishes lines (and
        // this slot's vertex buffer grown to fit them), so the graph gate below
        // can see them live this same frame and a world that never draws a line
        // never compiles them. Safe here: the frame fence at the top of
        // `draw_frame` retired everything that read this slot last trip.
        self.ensure_line_pipeline(frame_idx, lines);

        //  Per-frame seed inputs for the shared backend-agnostic frame
        //  builder ([gfx/render_graph/frame.rs](../../gfx/render_graph/frame.rs)).
        //  Decals landed 2026-05-24; Fog followed; AutoExposure landed
        //  2026-05-25; Particles landed 2026-05-25. The flags track
        //  whether each pipeline is built: the encoders skip cheaply
        //  when there is nothing live to draw.
        let seed_inputs = FrameGraphInputs {
            shadow_enabled: self.shadow.pipeline.is_some(),
            shadow_map_size: self.shadow.map_size,
            hdr_width: extent.width,
            hdr_height: extent.height,
            hdr_sample_count: self.msaa_samples.as_raw(),
            bindless_cull_enabled: self.cull.cull_pipeline.is_some() && self.cull_count() > 0,
            auto_exposure_enabled: self.auto_exposure.resources.is_some(),
            bloom_enabled: self.post_process.bloom_intensity > 0.0,
            // Velocity (motion vectors) runs for TAA *or* temporal upscaling
            // (FSR consumes them); TAA resources are forced built under
            // upscaling, so `taa.is_some()` already implies it, but spell out
            // the upscale case too.
            velocity_enabled: self.taa.is_some() || self.upscale.is_some(),
            // TAA resolve and Upscale are mutually exclusive (both do temporal
            // accumulation and share the graph slot). Drop TAA when upscaling.
            taa_enabled: self.taa.is_some() && self.upscale.is_none(),
            // The SSR *resolve* (reflection compositing). `self.ssr` may exist
            // for a SSGI-only build (it owns the shared pre-pass G-buffer), so
            // gate the resolve node on the dedicated flag, not `ssr.is_some()`.
            ssr_enabled: self.ssr_resolve_active,
            particles_enabled: self.particle.resources.is_some()
                && self.particle.records.iter().any(|p| p.is_some()),
            // Gated on both the resources (built at init when the world declared
            // a VolumetricFog) and the live settings, so runtime
            // `update_fog_settings(None)` drops the FogFroxel + Fog passes from
            // the graph entirely. Mirrors Metal's `pipeline && settings` gate.
            fog_enabled: self.fog.resources.is_some() && self.fog.settings.is_some(),
            decals_enabled: self.decal.resources.is_some()
                && self.decal.records.iter().any(|d| d.is_some()),
            // The SSR pre-pass G-buffer is shared with SSGI, so it runs whenever
            // `self.ssr` exists (built for SSR resolve *or* SSGI).
            ssr_prepass_enabled: self.ssr.is_some(),
            ssao_enabled: self.ssao.is_some(),
            // Gated on the resources (built at init when at least one `.glsl`
            // SdfVolume survived the filter) AND a currently-visible volume, so
            // an all-hidden world drops the pass from the graph.
            raymarch_enabled: self.raymarch.as_ref().is_some_and(|r| r.any_visible()),
            // Temporal upscaling (FSR via FidelityFX). `Some` only when the
            // world opted in AND the FFX VK runtime + context built; the shared
            // builder then runs `Upscale` in the `TaaResolve` slot, reading the
            // post-SSR scene + velocity and writing the swapchain-res scene the
            // bloom + composite stack samples.
            upscale_enabled: self.upscale.is_some(),
            // Transparent / translucent pass: on when the world declared a
            // visible `GlassPanel` or `WaterSurface`. The shared builder then
            // seeds the Transparent node and the executor draws every record
            // back-to-front over the post-SSR scene.
            transparent_enabled: self.transparent.as_ref().is_some_and(|t| t.any_visible())
                || self.mesh_glass_visible(),
            // Two-pass Hi-Z occlusion: inserts HizBuild -> Cull2 -> Main2 after
            // Main when the world requested `occlusion_two_pass` and the bindless
            // GPU-cull path + phase-2 resources are live. `two_pass_occlusion_active`
            // is the single gate the executor's phase-2 arms + the phase-1
            // render-pass selection share, so the graph shape matches what the
            // executor dispatches. The graph builder further ANDs this with
            // `bindless_cull_enabled`, which is already implied here.
            two_pass_occlusion_enabled: self.two_pass_occlusion_active(),
            // The terminal Hi-Z build. Present whenever the GPU-cull path built a
            // pyramid: the frame ends by reducing its final depth into it for the
            // next frame's phase-1 occlusion test.
            hiz_build_enabled: self.cull.hiz.is_some(),
            // Screen-space global illumination. `Some` only when the world
            // selected `indirect_lighting: ssgi`; the graph then inserts the
            // `Ssgi` node on the hdr_resolve RMW chain (which forces the SSR
            // pre-pass on, since `self.ssr` is built for SSGI too).
            ssgi_enabled: self.ssgi.is_some(),
            // Hardware ray-traced reflections (`VK_KHR_ray_query`). On only when
            // the world requested it, the GPU exposed the ray-query extensions,
            // and the acceleration structure built; the shared builder then emits
            // `RtReflections` in the `SsrResolve` slot (RT takes precedence; SSR
            // is the non-RT-GPU fallback).
            rt_reflections_enabled: self.rt_reflections_active(),
            // Unified geometry pre-pass: one `GBufferPrepass` node rasterises the
            // normal+depth / roughness / velocity MRT every screen-space consumer
            // (SSR / SSAO / SSGI / TAA / FSR) reads, replacing the separate SSR /
            // SSAO / velocity pre-passes. On exactly when the merged buffer was
            // built (any of those consumers is live); the shared builder then
            // emits the single node and skips `SsrPrepass` / `Velocity`. Mirrors
            // DirectX's `unified_gbuffer_prepass: self.gbuffer.is_some()`.
            unified_gbuffer_prepass: self.gbuffer.is_some(),
            // An opaque menu backdrop hides the scene: the shared builder masks
            // every world pass off, collapsing to Main (a bare clear, fed the
            // empty scene below) -> Composite (presents the overlay).
            world_hidden,
            // Clustered light binning. The compute pipeline is built only when
            // the world has local lights to bin, so this also gates the
            // `LightCull` graph node; otherwise the forward pass brute-forces.
            clustered_lighting_enabled: self.light_cull.pipeline.is_some(),
            // Zero drops the SpotShadow node and its imported array from the
            // graph entirely, which is the common case (no shadow-casting spot).
            shadowed_spot_count: self.spot_shadow.count(),
            spot_shadow_slice_size: self.spot_shadow.slice_size,
            // Lines run only on the frames a system published them (the
            // `cn editor` axes), and only once their resources are live: the
            // build is lazy, so a shipped runtime never compiles them.
            lines_enabled: !lines.is_empty() && self.lines.resources.is_some(),
            // Set by the view-mode mask below (occlusion view only).
            composite_reads_ao: false,
        };
        // The viewport's view mode + show flags mask the seeded inputs (the
        // per-frame counterpart of the init-time trims); Lit with every flag
        // set is the identity, so a shipped runtime is unaffected.
        let seed_inputs =
            crate::gfx::render_graph::apply_view(&seed_inputs, self.view.mode, self.view.show);

        //  Camera projection + per-frame view state. Computed before the main
        //  render pass begins so the GPU-cull compute dispatch (which Vulkan
        //  forbids inside a render pass) can read this frame's frustum.
        let aspect = if extent.height == 0 {
            1.0
        } else {
            extent.width as f32 / extent.height as f32
        };
        let proj = perspective_rh(fov_y_radians, aspect, near, far);
        // Un-jittered camera VP, fed to the velocity pre-pass so the stored
        // motion vector is free of the sub-pixel projection jitter.
        let cur_vp = mat4_mul(proj, self.view.matrix);
        // When TAA is on, offset the projection by a sub-pixel Halton jitter so
        // the accumulation has fresh sample positions each frame. The jitter is
        // a pure NDC x/y shift (depth is unaffected): `proj[2][0/1]` are the
        // z-coefficients of clip x/y, so subtracting the jitter there shifts
        // post-divide NDC by exactly the jitter amount (clip.w == -view_z).
        // Mirrors the jitter in metal/draw.rs.
        let render_proj = if let Some(up) = self.upscale.as_ref() {
            // The active backend prescribes the jitter sequence (render-pixel
            // units): FSR queries its FFX-tuned offsets, DLSS / XeSS use the
            // shared Halton-2/3. The same offset feeds the dispatch via
            // `set_jitter`. Phase index = the TAA frame counter, which advances
            // every frame here (TAA resources are forced built under upscaling).
            let phase = self.taa.as_ref().map(|t| t.taa_frame).unwrap_or(0);
            let [jx_px, jy_px] = up.jitter_offset(phase);
            up.set_jitter([jx_px, jy_px]);
            let mut p = proj;
            p[2][0] -= jx_px * 2.0 / extent.width.max(1) as f32;
            p[2][1] -= jy_px * 2.0 / extent.height.max(1) as f32;
            p
        } else if let Some(taa_frame) = self.taa.as_ref().map(|t| t.taa_frame) {
            let idx = taa_frame % 8 + 1;
            let jx = (halton(idx, 2) - 0.5) * 2.0 / extent.width.max(1) as f32;
            let jy = (halton(idx, 3) - 0.5) * 2.0 / extent.height.max(1) as f32;
            let mut p = proj;
            p[2][0] -= jx;
            p[2][1] -= jy;
            p
        } else {
            proj
        };
        let vp_mat = mat4_mul(render_proj, self.view.matrix);

        // Clustered light-binning params (main camera). The compute pass reads
        // these to build each cluster's world-space AABB (un-jittered inverse VP
        // + camera forward, matching the fog froxel convention) and the forward
        // pass reads the grid dims / depth range / screen size to place a
        // fragment. `use_clusters` is set only when the world has local lights;
        // otherwise the forward pass iterates them all. The planar / probe global
        // sets bind the static `use_clusters = 0` copy instead.
        let clustered = self.light_cull.pipeline.is_some();
        let cluster_params = crate::gfx::render_types::ClusterParams {
            inv_view_proj: mat4_inverse(mat4_mul(proj, self.view.matrix)),
            cam_pos,
            z_near: near.max(1e-3),
            view_forward: [
                -self.view.matrix[0][2],
                -self.view.matrix[1][2],
                -self.view.matrix[2][2],
            ],
            z_far: far,
            grid_x: crate::gfx::render_types::CLUSTER_GRID_X,
            grid_y: crate::gfx::render_types::CLUSTER_GRID_Y,
            grid_z: crate::gfx::render_types::CLUSTER_GRID_Z,
            num_lights: self.uniforms.light_uniforms.num_local_lights.max(0) as u32,
            screen_w: extent.width as f32,
            screen_h: extent.height as f32,
            use_clusters: u32::from(clustered),
            _pad: 0,
        };
        self.write_cluster_params(frame_idx, &cluster_params);

        // Update view UBO for this frame.
        let view_uni = ViewUniforms {
            vp: vp_mat,
            view: self.view.matrix,
            elapsed,
            // Hand glossy dielectric specular to the SSR / RT resolve when its
            // composite owns the scene image this frame (the composite is present
            // iff a resolve is active), else the forward shader keeps it all.
            reflections_enabled: if self.reflection_composite.is_some() {
                1.0
            } else {
                0.0
            },
            cam_pos: [cam_pos[0], cam_pos[1], cam_pos[2]],
            prefilter_mip_count: self.prefilter_mip_count as f32,
            shade_mode: self.shade_mode(),
            _end_pad: 0.0,
            sky_rot: self.view.sky_rot,
        };
        self.uniforms.view_ubo_buffers[frame_idx].write_val(0, &view_uni);

        // Light uniforms for this frame. Written only into slots a live edit
        // (directional set / ambient scale) has re-armed, so a steady world
        // writes nothing after the ring has caught up.
        if self.uniforms.light_dirty.take(frame_idx) {
            upload_light_uniforms(
                &self.uniforms.light_ubo_buffers[frame_idx],
                &self.uniforms.light_uniforms,
            );
        }

        // Reflection-probe set (global set 0 binding 7): EMPTY (count 0 = sky
        // reflection) until a probe bakes, so the forward shader keeps the sky
        // path. Uploaded every frame so a later install is picked up immediately.
        self.uniforms.probe_set_ubo_buffers[frame_idx].write_val(0, &self.probe.set);

        let frustum = crate::gfx::frustum::Frustum::from_view_projection(vp_mat);

        // Compute-cull host-side prep: rebuild this frame's
        // `GpuObjectData` + `GpuDrawArgs` storage buffers with the
        // latest per-object state. These are mapped-memory writes and
        // run unconditionally on the CPU side so the cull compute pass
        // (dispatched by the graph executor below) sees fresh data.
        // The compute dispatch itself runs through the graph as
        // `PassId::Cull`; the toposort orders Cull → Main via the
        // `draw_args` buffer RAW edge declared on Main.
        // Skipped while the world is hidden behind an opaque menu: the masked
        // graph drops the Cull pass and Main runs as a bare clear, so this
        // per-object buffer prep would feed nothing.
        if !world_hidden && seed_inputs.bindless_cull_enabled {
            self.build_object_buffer(frame_idx);
            self.build_draw_args_buffer(frame_idx, cam_pos);
        }

        // CPU visibility list (BVH-culled cullables + draw.always fallback).
        // Computed before the main render pass so the SSAO pre-pass below can
        // walk the same set, and so velocity / TAA later can reuse it without
        // a second BVH walk. `mem::take` swaps out the persistent scratch
        // buffer so its heap allocation is reused across frames; it's put
        // back below before we return Ok (error path loses capacity, fine
        // since record_frame errors are exceptional). Left empty when the world
        // is hidden so the Main pass draws nothing behind the menu.
        let mut visible = std::mem::take(&mut self.draw.visible_scratch);
        visible.clear();
        if !world_hidden {
            self.draw
                .bvh
                .query(&frustum, cam_pos, |idx| visible.push(idx));
            visible.sort_unstable();
            visible.extend_from_slice(&self.draw.always);
        }

        //  Single merged frame graph dispatched in one
        //  `execute_graph` call. The toposort orders Cull → Main via
        //  the `draw_args` buffer RAW edge, SsaoBlur / Shadow → Main
        //  via their texture RAW edges, and the post-stack chain
        //  (SsrResolve → TaaResolve → Bloom → Composite) via the
        //  scene_color version chain. Velocity pins before TaaResolve
        //  via the `velocity` texture read. Shadow's encoder owns
        //  its DEPTH → SHADER_READ post-loop transition; Cull's owns
        //  its SHADER_WRITE → INDIRECT_COMMAND_READ memory barrier;
        //  each render-pass attachment pins the per-pass layouts.
        //  Deriving `vkCmdPipelineBarrier` from `pass.barriers_before`
        //  (so encoders can shed their inline barriers) is a
        //  follow-up. The pre-graph dispatch site sits at the natural
        //  Main location: after the per-frame view/projection +
        //  visible-set compute (Vulkan forbids compute inside a
        //  render pass, so Cull must come before any render-pass
        //  dispatch from the executor).
        // Reuse the cached compiled graph when this frame's inputs match the ones
        // it was built from (the common case: graph topology changes only when a
        // feature toggles or a target resizes). Taken out of the cache so the later
        // `&mut self` execute_graph does not conflict with a borrow of it; put back
        // after execution. A mismatch (or a cold cache) rebuilds.
        let graph = match self.draw.graph_cache.take() {
            Some((cached_inputs, cached_graph)) if cached_inputs == seed_inputs => cached_graph,
            _ => build_frame_graph(&seed_inputs).map_err(|e| format!("frame graph: {e}"))?,
        };
        let params = GraphFrameParams {
            cmd,
            image_index,
            frame_idx,
            text_calls,
            lines,
            world_hidden,
            visible: &visible,
            frustum: &frustum,
            cam_pos,
            vp_mat,
            cur_vp,
            fov_y_radians,
            aspect,
            elapsed,
            near,
            far,
        };
        // Each non-composite pass is recorded into its own command buffer
        // (returned here in graph order); Composite + the post-graph work below
        // record into `cmd` (the outer "end" buffer). The whole frame is
        // submitted as `[start, ...pass_bufs, end]` by `draw_frame`.
        let pass_bufs = self.execute_graph(&graph, &params)?;
        // Cache the compiled graph under this frame's inputs so the next frame with
        // matching inputs skips the rebuild.
        self.draw.graph_cache = Some((seed_inputs, graph));

        // The Hi-Z reduction that feeds next frame's cull is the graph's terminal
        // `HizFinal` pass, so it has already been recorded above; `hiz_valid` only
        // tracks whether a pyramid at the current resolution now exists.

        // The cascade slices rest sampled (SHADER_READ_ONLY_OPTIMAL) between
        // frames; next frame's Shadow producer barrier (graph-driven) performs
        // the SHADER_READ_ONLY -> DEPTH_STENCIL_ATTACHMENT reset over every
        // cascade layer, so no inline end-of-frame restore is needed here.

        // Advance the TAA jitter sequence (`taa_frame > 0` also validates the
        // history for next frame). The motion-vector temporal state lives on the
        // unified G-buffer (advanced below); TAA only consumes its velocity view.
        if let Some(taa) = &mut self.taa {
            taa.taa_frame = taa.taa_frame.wrapping_add(1);
        }

        // Advance the unified G-buffer's velocity-channel temporal state in
        // lockstep with TAA's: this frame's un-jittered VP becomes next frame's
        // `prev_vp`, and every object transform is snapshotted so the next
        // GBufferPrepass can diff against it. Owned by `GbufferResources` so the
        // motion vector works for any consumer (TAA or FSR), exactly mirroring
        // the TAA advance above. Mirrors DirectX's `prev_view_proj`/`prev_models`
        // bookkeeping in `record_frame`.
        if let Some(gb) = &mut self.gbuffer {
            gb.prev_view_proj = cur_vp;
            gb.prev_models.resize(self.draw.objects.len(), IDENTITY);
            for (prev, obj) in gb.prev_models.iter_mut().zip(self.draw.objects.iter()) {
                *prev = obj.model;
            }
        }

        // Advance Hi-Z temporal state: this frame's un-jittered VP becomes next
        // frame's occlusion-test projection, and the pyramid the graph's
        // `HizFinal` pass just wrote is now valid for next frame's cull (kept
        // independent of TAA, which may be off while Hi-Z is on).
        if self.cull.hiz.is_some() {
            self.cull.hiz_prev_view_proj = cur_vp;
            self.cull.hiz_valid = true;
        }

        self.draw.visible_scratch = visible;

        // Drain the parallel-safe draw-call accumulator (bumped by every pass
        // encoder, including those fanned onto rayon workers) into this frame's
        // `frame_stats` for the profiler overlay. All recording is done by here.
        let mut stats = self.frame_stats.get();
        stats.draw_calls = self
            .draw_calls_accum
            .load(std::sync::atomic::Ordering::Relaxed);
        self.frame_stats.set(stats);

        // End-of-frame timestamp for the profiler overlay. Pairs with the
        // TOP_OF_PIPE write near the top of the function (the block's first pair).
        if let Some(pool) = self.timestamp_query_pool {
            let (_, wf_end) = super::pass_timing::whole_frame_pair(frame_idx);
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    pool,
                    wf_end,
                );
            }
        }

        // The per-frame submission order, excluding the outer "end" buffer
        // (`cmd`) the caller appends: the `start` buffer (leading timestamp)
        // followed by the per-pass buffers in graph order. Submission order is
        // GPU order on the single graphics queue, so every encoder's inline
        // barrier synchronises against the prior pass across buffer boundaries.
        let mut submit = Vec::with_capacity(pass_bufs.len() + 1);
        submit.push(start_cmd);
        submit.extend(pass_bufs);
        Ok(submit)
    }
}

// Helper: write ShadowUniforms into one persistently-mapped slot of the
// per-frame-in-flight ring. The slot belongs to a frame whose fence the caller
// already waited.
pub(super) fn upload_shadow_uniforms(ubo: &super::allocator::PooledBuffer, su: &ShadowUniforms) {
    ubo.write_val(0, su);
}

// Helper: write LightUniforms into one persistently-mapped slot of the
// per-frame-in-flight ring. The slot belongs to a frame whose fence the caller
// already waited.
pub(super) fn upload_light_uniforms(
    light_ubo: &super::allocator::PooledBuffer,
    lu: &LightUniforms,
) {
    light_ubo.write_val(0, lu);
}

// Helper: one-shot upload of the per-scene local lights into the static SSBO.
// An empty scene leaves the 1-element placeholder buffer untouched
// (num_local_lights == 0 keeps the shader from reading it). Mirrors the single
// upload path of upload_light_uniforms; the SSBO is never rewritten per-frame.
pub(super) fn upload_static_records<T: Copy>(
    buffer: &super::allocator::PooledBuffer,
    records: &[T],
) {
    if records.is_empty() {
        return;
    }
    buffer.write_slice(0, records);
}
