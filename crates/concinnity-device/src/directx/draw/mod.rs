//! `DxContext::record_frame` -- the per-frame orchestration. The per-pass GPU
//! encoders live in sibling files (shadow / main / composite) plus the
//! post-process effects in `directx/post/` (bloom / TAA / SSAO):
//!
//!   shadow.rs              cascaded shadow map (depth-only, per cascade)
//!   main.rs                SSAO pre-pass + main HDR pass (bindless indirect,
//!                          skinned tail, phase-2 re-issue) + HDR resolve barriers
//!   composite.rs           ACES tonemap + composite + text overlay
//!   ../post/{bloom,taa,ssao}.rs    pipeline + targets + encoder, co-located

use concinnity_core::gfx::frustum::Frustum;
use concinnity_core::gfx::render_types::{
    ClusterCamera, ClusterParams, LightUniforms, LineVertex, ShadowUniforms, TextDrawCall,
};
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::lights;
use concinnity_core::render::pass_timing;
use concinnity_core::render::render_graph::build_frame_graph;
use windows::Win32::Graphics::Direct3D12::*;

use super::com;
use super::context::DxContext;
use super::graph_exec::GraphFrameParams;
use crate::directx::error::map_hresult;
use stages::FrameProjection;

mod composite;
pub(in crate::directx) mod main;
pub(in crate::directx) mod shadow;
pub(in crate::directx) mod spot_shadow;
mod stages;

// `ViewUniforms` (the main-pass `ViewBlock` cbuffer) is a GPU-free layout struct
// that lives in `core::render`; re-export it so
// `crate::directx::draw::ViewUniforms` is unchanged for the passes that fill it.
pub(in crate::directx) use concinnity_core::render::uniforms::ViewUniforms;

// The command list + back-buffer target this frame records into.
#[derive(Clone, Copy)]
pub(super) struct RecordFrameTargets<'a> {
    // Outer "end" command list carrying Composite + restore barriers.
    pub cmd: &'a ID3D12GraphicsCommandList,
    // Swapchain back-buffer resource.
    pub back_buffer: &'a ID3D12Resource,
    // CPU descriptor handle for the back-buffer RTV.
    pub back_buffer_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    // Frame-in-flight slot for per-frame resource indexing.
    pub frame_idx: usize,
}

// Per-frame camera / view state plus the overlay text drawn with it.
#[derive(Clone, Copy)]
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

// Scene render resolution (every scene pass) plus output resolution (composite).
#[derive(Clone, Copy)]
pub(super) struct RecordFrameResolution {
    // Off-screen scene render resolution; drives every scene pass + sub-pixel jitter.
    pub width: u32,
    pub height: u32,
    // Drawable (swapchain) resolution; only the Composite pass uses it.
    pub output_width: u32,
    pub output_height: u32,
}

impl DxContext {
    // Resets this slot's START list, pre-inits its timestamps, records the RT
    // acceleration-structure update, and closes it.
    pub(in crate::directx) fn record_frame_start(
        &mut self,
        frame: usize,
    ) -> RenderResult<ID3D12GraphicsCommandList> {
        // 1. Reset + record the START cmd list (timestamp pre-init only),
        //    then close it immediately. The `wait_frame_slot` fence wait
        //    gated this slot's previous submission, so it's
        //    safe to reset.
        // SAFETY: the fence for this frame slot was already waited on, so no submission still
        // references what is being reset.
        unsafe { self.commands.command_allocators[frame].Reset() }
            .map_err(|e| map_hresult(e.code(), "start allocator reset"))?;
        // Owned clone (COM refcount bump) so the per-frame RT acceleration-
        // structure update below can take `&mut self` without holding a borrow
        // of `self.commands.command_lists`.
        let start_cmd = self.commands.command_lists[frame].clone();
        // SAFETY: the fence for this frame slot was already waited on, so no submission still
        // references what is being reset.
        unsafe { start_cmd.Reset(&self.commands.command_allocators[frame], None) }
            .map_err(|e| map_hresult(e.code(), "start cmd reset"))?;

        // Timestamp the start of this frame's GPU work + pre-initialize
        // every per-pass slot in this frame's block. The end-of-frame
        // `ResolveQueryData` covers the whole block, and the D3D12 debug
        // layer flags any slot in the resolved range that never had
        // `EndQuery` called on it; without the pre-init the graph
        // would spam those errors for every feature the world opted out
        // of. The executor's per-pass start/end calls overwrite the
        // slots of active passes with real timestamps; inactive slots
        // keep this frame-start timestamp for both start and end, so
        // `ts_end > ts_start` evaluates false on readback and they
        // cleanly report 0 µs.
        if let Some(heap) = self.timestamps.query_heap.as_ref() {
            let (start_slot, _) = pass_timing::whole_frame_pair(frame);
            let block_base = (frame * pass_timing::SLOTS_PER_FRAME) as u32;
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                start_cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, start_slot);
                // Pre-init each per-pass (start, end) pair as **end then
                // start** so inactive passes wind up with `ts_start >
                // ts_end` and the readback's `ts_end > ts_start` check
                // returns false (clean 0 µs reading).
                let pass_count = pass_timing::SLOTS_PER_FRAME / 2 - 1;
                for pass_idx in 0..pass_count as u32 {
                    let pair_start = block_base + 2 + 2 * pass_idx;
                    let pair_end = pair_start + 1;
                    start_cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, pair_end);
                    start_cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, pair_start);
                }
            }
        }

        // Per-frame hardware-RT acceleration-structure update: when a
        // participating prop moved, rebuild the TLAS + geometry table onto the
        // start cmd list (submitted before every per-pass trace on the serial
        // DIRECT queue, so the rebuild is ordered before this frame's reflection
        // trace reads it). A no-op when RT reflections are off or the BVH is
        // static this frame.
        self.rt_dynamic_update(&start_cmd, frame);

        // SAFETY: the command list is live and in the recording state, which is what `Close`
        // requires.
        unsafe { start_cmd.Close() }.map_err(|e| map_hresult(e.code(), "start cmd close"))?;
        Ok(start_cmd)
    }

    // Drive a single frame through the render graph. `end_cmd` is the
    // outer "end" cmd list (composite + final timestamp + resolve +
    // per-frame restore barriers); the executor encodes the Composite
    // pass onto it inline, and the post-graph restore barriers below
    // also append onto it. Returns the per-pass cmd lists the executor
    // recorded (in topological pass order) so the caller can submit
    // them between the "start" outer cmd list (timestamp pre-init,
    // closed by the caller before record_frame) and the "end" outer
    // cmd list.
    pub(super) fn record_frame(
        &self,
        targets: RecordFrameTargets<'_>,
        view: RecordFrameView<'_>,
        resolution: RecordFrameResolution,
        world_hidden: bool,
    ) -> RenderResult<Vec<ID3D12GraphicsCommandList>> {
        let RecordFrameTargets {
            cmd: end_cmd,
            back_buffer,
            back_buffer_rtv,
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
        let RecordFrameResolution {
            width,
            height,
            output_width,
            output_height,
        } = resolution;
        // Render-target aspect, shared by the main projection below and the
        // cull frustum.
        let aspect = if height == 0 {
            1.0
        } else {
            width as f32 / height as f32
        };

        // Cascaded-shadow UBO upload. `draw_frame` already advanced the cascade
        // schedule and merged this frame's cascades into `self.shadow.uniforms`
        // (skipped cascades keep the VP their slice was last rendered with, so
        // the Main pass samples each cascade consistently); upload the carried
        // set to this frame's shadow UBO (persistent mapping). It is the empty
        // (identity VP / infinite split) set when shadows are disabled.
        // SAFETY: the destination is the persistent mapping of an UPLOAD-heap constant buffer that
        // init sized for this payload, and the source is a separate live value, so the ranges
        // cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                &self.shadow.uniforms as *const ShadowUniforms as *const u8,
                self.uniforms.shadow_ubo_ptrs[frame_idx],
                std::mem::size_of::<ShadowUniforms>(),
            );
        }
        let shadow_ubo_gva = com::gpu_va(&self.uniforms.shadow_ubo_resources[frame_idx]);

        // Light uniforms for this frame. Written only into slots a live edit
        // (directional set / ambient scale) has re-armed, so a steady world
        // writes nothing after the ring has caught up.
        if self.uniforms.take_light_dirty(frame_idx) {
            upload_light_uniforms(
                self.uniforms.light_ubo_ptrs[frame_idx],
                &self.uniforms.light_uniforms,
            );
        }

        // Push this frame's skinning matrices into the per-frame joint buffers
        // before the skinned shadow + main passes read them. No-op when no
        // SkinnedMesh is declared.
        self.upload_joint_matrices(frame_idx);
        // Push this frame's morph weights into the per-frame weight buffers the
        // skin fold reads. No-op when no SkinnedMesh carries morph targets.
        self.upload_morph_weights(frame_idx);

        // GPU-driven cull gating; matches the inner check in
        // encode_main_pass's bindless branch. When on, the host-side
        // per-frame object buffer rebuild runs inline here (mapped-memory
        // CPU work, mirrors Vulkan's pattern) and the pre-graph picks up
        // a `PassId::Cull` node that writes the indirect command buffer
        // ahead of Main.
        let bindless_cull_enabled = self.cull.main_bindless_pso.is_some() && self.cull_count() > 0;
        // Skipped while the world is hidden behind an opaque menu: the masked
        // graph drops the Cull pass and Main runs as a bare clear, so this
        // per-object buffer rebuild would feed nothing.
        if !world_hidden && bindless_cull_enabled {
            self.build_object_buffer(frame_idx);
        }

        // Per-cluster LOD bucketing + instance-buffer upload. Has to
        // happen BEFORE `execute_graph` because SSAO / SSR / TAA-velocity
        // pre-passes (which run earlier than main in the graph) read the
        // same per-frame upload buffer; with LOD bucketing the byte
        // layout depends on `cam_pos`, so they need the **current**
        // frame's data, not previous-frame leftovers. No-op when no
        // instanced cluster declared LOD alternates (every cluster
        // collapses to a single LOD0 bucket containing all instances,
        // same byte order as an unbucketed upload).
        if !world_hidden && !self.instanced.clusters.is_empty() {
            self.build_instance_upload(cam_pos);
        }

        // Clustered binning runs while a local light or a baked probe is live,
        // the same test `ClusterParams::for_camera` sets `use_clusters` by, so
        // no reader sees a list the skipped `LightCull` node did not write.
        let clustered = lights::clustering_active(
            self.uniforms.light_uniforms.num_local_lights,
            self.probe.book.count(),
        );
        let seed_inputs = self.frame_graph_inputs(
            width,
            height,
            bindless_cull_enabled,
            clustered,
            lines,
            world_hidden,
        );
        let FrameProjection {
            proj,
            cur_vp,
            vp_mat,
        } = self.frame_projection(fov_y_radians, aspect, near, far, width, height);

        // Clustered light-binning params (main camera). The compute pass reads
        // these to build each cluster's world-space AABB (un-jittered inverse VP
        // + camera forward, matching the fog froxel convention) and the forward
        // pass reads the grid dims / depth range / screen size to place a
        // fragment. `use_clusters` is set only while a local light or a baked
        // probe is live; otherwise every reader iterates them all (zero
        // iterations) rather than reading a list the skipped binning pass never
        // wrote. Slot 1 of the same buffer holds the `use_clusters = 0`
        // copy the planar / probe re-renders bind (written once at init).
        let cluster_params = ClusterParams::for_camera(
            &ClusterCamera {
                view: self.view.matrix,
                proj,
                position: cam_pos,
                near,
                far,
                width,
                height,
            },
            self.uniforms.light_uniforms.num_local_lights,
            self.probe.book.count() as u32,
        );
        self.write_cluster_params(frame_idx, &cluster_params);

        // Upload this frame's view UBO.
        // Fade the forward probe specular only when a resolve will actually
        // composite its reflection over this scene. `reflection_resolve_active`
        // alone matches the resolve gating, but require the composite target too
        // so the fade can never zero a reflection that was never re-added.
        let reflections_enabled =
            if self.reflection_composite.is_some() && self.reflection_resolve_active() {
                1.0
            } else {
                0.0
            };
        let view_uni = ViewUniforms {
            vp: vp_mat,
            view: self.view.matrix,
            elapsed,
            reflections_enabled,
            cam_pos: [cam_pos[0], cam_pos[1], cam_pos[2]],
            prefilter_mip_count: self.scene.env_map.prefilter_mip_count as f32,
            shade_mode: self.shade_mode(),
            _end_pad: 0.0,
            sky_rot: self.view.sky_rot,
        };
        // SAFETY: the destination is the persistent mapping of an UPLOAD-heap constant buffer that
        // init sized for this payload, and the source is a separate live value, so the ranges
        // cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                &view_uni as *const ViewUniforms as *const u8,
                self.uniforms.view_ubo_ptrs[frame_idx],
                std::mem::size_of::<ViewUniforms>(),
            );
        }

        let frustum = Frustum::from_view_projection(vp_mat);

        let (view_gva, light_gva, local_lights_gva) = (
            com::gpu_va(&self.uniforms.view_ubo_resources[frame_idx]),
            com::gpu_va(&self.uniforms.light_ubo_resources[frame_idx]),
            com::gpu_va(&self.uniforms.local_light_buffer),
        );

        // Scene source for the bloom prefilter + the composite. Priority:
        //   1. FSR3 upscaler output (when temporal upscaling is on; the
        //      graph excludes TaaResolve in that case so the TAA history
        //      slots are never written; sampling them would yield black).
        //   2. TAA history output (when TAA is on and upscale is off).
        //   3. SSR resolve / raw HDR fallback via `scene_srv_for_post`.
        // The handle value is stable across the TAA dispatch:
        // `taa.output_index()` is `frame % 2`, and the frame counter is
        // only bumped *after* Composite, so reading the index before the
        // executor runs GBufferPrepass + TaaResolve gives the same pointer the
        // encoders write into / sample from.
        let scene_srv = if self.upscale.backend.is_some() {
            self.scene_srv_for_post()
        } else {
            match &self.taa {
                Some(taa) => taa.output().srv_gpu(),
                None => self.scene_srv_for_post(),
            }
        };

        // Single graph dispatch: every render-stack pass plus the Composite
        // presenter routed through one `execute_graph` call. The graph
        // shape lives in the shared
        // [gfx/render_graph/frame.rs::build_frame_graph](../../gfx/render_graph/frame.rs);
        // see [directx/graph_exec.rs](../graph_exec.rs) for the DirectX
        // executor that routes each `PassId` to its `encode_*` method.
        // Reuse the cached compiled graph when this frame's inputs match the ones
        // it was built from (the common case: graph topology changes only when a
        // feature toggles or a target resizes). `take`n out of the cache (the
        // `borrow_mut` guard drops at the end of this statement) so the owned graph
        // no longer borrows `self`; a mismatch (or a cold cache) rebuilds.
        let cached_graph = self.draw.graph_cache.borrow_mut().take();
        let frame_graph = match cached_graph {
            Some((cached_inputs, cached)) if cached_inputs == seed_inputs => cached,
            _ => build_frame_graph(&seed_inputs)
                .map_err(|e| RenderError::Other(format!("frame-graph compile: {e}")))?,
        };
        let frame_params = GraphFrameParams {
            cmd: end_cmd,
            frame_idx,
            back_buffer,
            back_buffer_rtv,
            text_calls,
            lines,
            world_hidden,
            scene_srv,
            width,
            height,
            output_width,
            output_height,
            cam_pos,
            shadow_ubo_gva,
            view_gva,
            light_gva,
            local_lights_gva,
            vp_mat,
            cur_vp,
            frustum: &frustum,
            fov_y_radians,
            aspect,
            elapsed,
            near,
            far,
        };
        let pass_cmd_lists = self.execute_graph(&frame_graph, &frame_params)?;
        // Cache the compiled graph under this frame's inputs so the next frame with
        // matching inputs skips the rebuild.
        *self.draw.graph_cache.borrow_mut() = Some((seed_inputs, frame_graph));

        self.advance_temporal_state(cur_vp);

        Ok(pass_cmd_lists)
    }
}

// Write LightUniforms into one persistently-mapped slot of the per-frame light
// CBV ring. The slot belongs to a frame whose fence the caller already waited.
pub(super) fn upload_light_uniforms(slot: *mut u8, lu: &LightUniforms) {
    // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and the source
    // is a separate live value, so the ranges cannot overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(
            lu as *const LightUniforms as *const u8,
            slot,
            std::mem::size_of::<LightUniforms>(),
        );
    }
}

// Root parameter indices for the spot shadow binds, which differ per root
// signature because each grew a different number of earlier parameters.
#[derive(Clone, Copy)]
pub(in crate::directx) struct LocalLightParams {
    // Root SRV carrying the `SpotShadowData` buffer.
    pub spot_buffer: u32,
    // Descriptor table carrying the spot shadow depth array SRV.
    pub spot_table: u32,
    // Root SRV carrying the `AreaLightData` buffer.
    pub area_buffer: u32,
    // Descriptor table carrying the two LTC lookup tables.
    pub ltc_table: u32,
}

impl LocalLightParams {
    // The GPU-driven main root signature.
    pub(crate) const BINDLESS: Self = Self {
        spot_buffer: 15,
        spot_table: 16,
        area_buffer: 17,
        ltc_table: 18,
    };
}

// Root parameters the bindless main pass binds the reflection-probe set at:
// the cube array table (t7), the ProbeSet constant buffer (b4) and the records
// (a root SRV at t8).
const MAIN_PROBE_CUBES_PARAM: u32 = 10;
const MAIN_PROBE_SET_PARAM: u32 = 11;
const MAIN_PROBE_RECORDS_PARAM: u32 = 19;

impl DxContext {
    // Bind a probe set on the bindless main root signature: the cube array
    // table, the ProbeSet constant buffer at `set_cbv` and the records at
    // `records`.
    pub(super) fn bind_main_probe_set(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        set_cbv: u64,
        records: u64,
    ) {
        use crate::directx::descriptor_slot::DescriptorTables as _;
        // SAFETY: the command list is in the recording state with the SRV heap
        // bound, and the table slot and both buffers are live for the frame the
        // list records.
        unsafe {
            cmd.set_graphics_srv_table(MAIN_PROBE_CUBES_PARAM, self.probe_cube_table_gpu());
            cmd.SetGraphicsRootConstantBufferView(MAIN_PROBE_SET_PARAM, set_cbv);
            cmd.SetGraphicsRootShaderResourceView(MAIN_PROBE_RECORDS_PARAM, records);
        }
    }
}

// One-shot upload of a per-scene record list into its static storage buffer.
// Sibling of `upload_light_uniforms`: same Map / copy / Unmap path, run once at
// init for buffers that are never rewritten per frame (the local-light list and
// the spot shadow projections). `label` names the buffer in the error.
pub(super) fn upload_static_records<T: Copy>(
    buffer: &ID3D12Resource,
    records: &[T],
    label: &str,
) -> RenderResult<()> {
    let bytes = std::mem::size_of_val(records);
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { buffer.Map(0, None, Some(&mut ptr)) }
        .map_err(|e| map_hresult(e.code(), &format!("map {label} buffer")))?;
    // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and the source
    // is a separate allocation, so the ranges cannot overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(records.as_ptr() as *const u8, ptr as *mut u8, bytes);
        buffer.Unmap(0, None);
    }
    Ok(())
}
