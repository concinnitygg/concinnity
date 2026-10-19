//! `DxContext::record_frame` -- the per-frame orchestration. The per-pass GPU
//! encoders live in sibling files (shadow / main / composite) plus the
//! post-process effects in `directx/post/` (bloom / TAA / SSAO):
//!
//!   shadow.rs              cascaded shadow map (depth-only, per cascade)
//!   main.rs                SSAO pre-pass + main HDR pass (bindless indirect,
//!                          skinned tail, phase-2 re-issue) + HDR resolve barriers
//!   composite.rs           ACES tonemap + composite + text overlay
//!   ../post/{bloom,taa,ssao}.rs    pipeline + targets + encoder, co-located

use concinnity_core::gfx::render_types::{LightUniforms, LineVertex, ShadowUniforms, TextDrawCall};
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::frame_plan::FramePlan;
use concinnity_core::render::pass_timing;
use concinnity_core::render::render_graph::PassId;
use windows::Win32::Graphics::Direct3D12::*;

use super::com;
use super::context::DxContext;
use super::graph_exec::GraphFrameParams;
use crate::directx::error::map_hresult;
pub(super) use stages::PlanArgs;

pub(in crate::directx) mod composite;
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
    // Seconds since the world started, narrowed to f32 where a pass needs no
    // more; the water's motion takes it whole.
    pub elapsed: f64,
    pub fov_y_radians: f32,
    pub near: f32,
    pub cam_pos: [f32; 3],
    pub text_calls: &'a [TextDrawCall],
    // This frame's expanded line ribbons. Empty whenever nothing published
    // lines, which also drops the pass from the graph.
    pub lines: &'a [LineVertex],
    // What may be trampling the grass this frame.
    pub grass_benders: &'a [concinnity_core::render::grass::GrassBender],
    pub particle: Option<concinnity_core::render::particles::ParticleFrame>,
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

// What `record_frame` recorded: the per-pass cmd lists, in topological pass
// order, and whether they carry the model-history prime it took from the
// tracker, which the caller hands back if the frame then fails to submit.
pub(super) struct RecordedFrame {
    pub pass_cmd_lists: Vec<ID3D12GraphicsCommandList>,
    pub primed_model_history: bool,
}

impl DxContext {
    // Resets this slot's START list, pre-inits its timestamps, records the RT
    // acceleration-structure update, and closes it.
    pub(in crate::directx) fn record_frame_start(
        &mut self,
        frame: usize,
        world_hidden: bool,
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
        // static this frame. Skipped while the world is hidden, which traces
        // nothing; a topology change stays owed to the next frame that draws it.
        if !world_hidden {
            self.rt_dynamic_update(&start_cmd, frame);
        }

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
    // recorded (`RecordedFrame`) so the caller can submit them between the
    // "start" outer cmd list (timestamp pre-init, closed by the caller before
    // record_frame) and the "end" outer cmd list.
    pub(super) fn record_frame(
        &self,
        targets: RecordFrameTargets<'_>,
        view: RecordFrameView<'_>,
        resolution: RecordFrameResolution,
        plan: &FramePlan,
        world_hidden: bool,
    ) -> RenderResult<RecordedFrame> {
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
            cam_pos,
            text_calls,
            lines,
            grass_benders,
            particle,
        } = view;
        let RecordFrameResolution {
            width,
            height,
            output_width,
            output_height,
        } = resolution;
        // Cascaded-shadow UBO upload: the plan's carried cascade set (skipped
        // cascades keep the VP their slice was last rendered with, so the Main
        // pass samples each cascade consistently), into this frame's shadow UBO
        // (persistent mapping). It is the empty (identity VP / infinite split)
        // set when shadows are disabled.
        // SAFETY: the destination is the persistent mapping of an UPLOAD-heap constant buffer that
        // init sized for this payload, and the source is a separate live value, so the ranges
        // cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&raw const self.state.planner.shadow.uniforms).cast::<u8>(),
                self.uniforms.shadow_ubo_ptrs[frame_idx],
                size_of::<ShadowUniforms>(),
            );
        }
        let shadow_ubo_gva = com::gpu_va(&self.uniforms.shadow_ubo_resources[frame_idx]);

        // Light uniforms for this frame. Written only into slots a live edit
        // (directional set / ambient scale) has re-armed, so a steady world
        // writes nothing after the ring has caught up.
        if self.uniforms.lights.take_dirty(frame_idx) {
            upload_light_uniforms(
                self.uniforms.light_ubo_ptrs[frame_idx],
                self.uniforms.lights.uniforms(),
            );
        }
        self.grass_lod.set(self.grass.as_ref().map(|g| {
            g.field
                .lod_view(concinnity_core::render::grass::lod::focal_length(
                    height,
                    fov_y_radians,
                ))
        }));
        self.upload_canopy(frame_idx);

        // Push this frame's skinning matrices into the per-frame joint buffers
        // before the skin fold reads them. No-op when no SkinnedMesh is
        // declared.
        self.upload_joint_matrices(frame_idx);
        // Push this frame's morph weights into the per-frame weight buffers the
        // skin fold reads. No-op when no SkinnedMesh carries morph targets.
        self.upload_morph_weights(frame_idx);

        // Compute-cull host-side prep: rebuild this frame's object + draw-args
        // buffers (mapped-memory CPU work) on a frame whose plan runs the cull,
        // so the graph's `PassId::Cull` node writes the indirect command buffer
        // ahead of Main from fresh data.
        if let Some(history) = plan.records {
            let packing = self.build_record_buffers(frame_idx, cam_pos, history);
            let mut stats = self.diagnostics.frame_stats.get();
            stats.packing_fan_out = packing;
            self.diagnostics.frame_stats.set(stats);
        }

        // The plan's blocks into this frame's ring slots.
        self.prepare_atmosphere(frame_idx, plan.atmosphere);
        self.write_cluster_params(frame_idx, &plan.cluster);
        // SAFETY: the destination is the persistent mapping of an UPLOAD-heap constant buffer that
        // init sized for this payload, and the source is a separate live value, so the ranges
        // cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&raw const plan.view).cast::<u8>(),
                self.uniforms.view_ubo_ptrs[frame_idx],
                size_of::<ViewUniforms>(),
            );
        }

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
        // presenter routed through one `execute_graph` call, over the graph the
        // plan compiled from the shared builder; see
        // [directx/graph_exec.rs](../graph_exec.rs) for the DirectX executor
        // that routes each `PassId` to its `encode_*` method.
        let frame_graph = &plan.graph;
        // The line ring is reserved and filled here rather than by the Lines
        // pass, since growing it allocates and no fanned-out pass may.
        let lines = if frame_graph.pass(PassId::Lines).is_some() {
            self.upload_lines(frame_idx, lines)?
        } else {
            None
        };
        // A rebuild's prime request is spent only on a frame whose G-buffer
        // pre-pass runs the history snapshot; any other frame leaves it pending.
        // Taken after every other fallible step but the graph itself.
        let prime_model_history = frame_graph.pass(PassId::GBufferPrepass).is_some()
            && self.state.model_history.borrow_mut().take_prime();
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
            vp_mat: plan.camera.vp,
            cur_vp: plan.camera.cur_vp,
            frustum: &plan.camera.frustum,
            fov_y_radians,
            elapsed,
            near,
            prime_model_history,
            grass_benders,
            particle,
            planar: self
                .planar_reflection
                .as_ref()
                .map(|set| set.frame_plan(plan.camera.vp))
                .unwrap_or_default(),
            cluster_params: plan.cluster,
            fog: plan.fog,
            gbuffer_view: plan.gbuffer_view,
            effects: plan.effects,
            reactive: plan.reactive,
        };
        let mut pass_cmd_lists = self.pass_lists.take();
        pass_cmd_lists.clear();
        if let Err(e) = self.execute_graph(frame_graph, &frame_params, &mut pass_cmd_lists) {
            if prime_model_history {
                self.state.model_history.borrow_mut().request_prime();
            }
            return Err(e);
        }
        Ok(RecordedFrame {
            pass_cmd_lists,
            primed_model_history: prime_model_history,
        })
    }
}

// Write LightUniforms into one persistently-mapped slot of the per-frame light
// CBV ring. The slot belongs to a frame whose fence the caller already waited.
pub(super) fn upload_light_uniforms(slot: *mut u8, lu: &LightUniforms) {
    // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and the source
    // is a separate live value, so the ranges cannot overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(
            std::ptr::from_ref::<LightUniforms>(lu).cast::<u8>(),
            slot,
            size_of::<LightUniforms>(),
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

// The bindless main root parameter the shared vertex stream binds at (t9),
// where a planet tile's morph reads its neighbors.
pub(in crate::directx) const MAIN_MESH_VERTICES_PARAM: u32 = 25;

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
    let bytes = size_of_val(records);
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { buffer.Map(0, None, Some(&raw mut ptr)) }
        .map_err(|e| map_hresult(e.code(), &format!("map {label} buffer")))?;
    // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and the source
    // is a separate allocation, so the ranges cannot overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(records.as_ptr().cast::<u8>(), ptr.cast::<u8>(), bytes);
        buffer.Unmap(0, None);
    }
    Ok(())
}
