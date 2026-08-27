// src/directx/graph_exec.rs
//
// DirectX-side executor for the render graph. `DxContext::execute_graph`
// walks the `CompiledGraph` produced by the shared
// [`gfx::render_graph::build_frame_graph`](../gfx/render_graph/frame.rs)
// and dispatches each pass to its `encode_*` method. Mirrors the Metal
// + Vulkan executors; every backend now drives the same builder.
//
// **Per-pass command lists.** Each non-composite pass records into its
// own `ID3D12GraphicsCommandList` (drawn from the `pass_cmd_lists` pool
// on `DxContext`). The fan-out runs on `jobs::pool()` via `rayon::scope`
// so workers encode in parallel; each worker resets its assigned
// allocator + cmd list, brackets the encode with start/end TIMESTAMP
// queries, encodes the pass, and closes the cmd list. The main thread
// then submits every closed cmd list in topological pass order via
// `ExecuteCommandLists`. The Composite pass keeps using the outer
// "end" cmd list that `draw_frame` owns (so the final timestamp +
// `ResolveQueryData` ride the same submission). Mirrors
// `metal/graph_exec.rs`.
//
// Per-pass `barriers_before` is consumed for every resource the barrier registry
// resolves: `emit_graph_barriers` translates their graph state transitions into
// `D3D12_RESOURCE_BARRIER` transitions at the start of each pass's own command
// list, and `emit_graph_restores` returns any that the frame left off their
// resting state at the end of the outer "end" list. Every other resource still
// owns its transitions inline in its encoder; `barrier_audit.rs` classifies each
// remaining site.
//
// The registry decides two things per resource: which D3D12 resource backs it,
// and what state it rests in between frames. Its class -- what a `Write` means --
// comes from the usage the graph declares, so this executor and the Vulkan one
// cannot disagree about it. Resting cannot: `shadow_map` and `hdr_depth` are both
// depth targets, and the first rests sampled while the second rests DEPTH_WRITE.
//
// Bundled passes:
//   * `PassId::SsaoBlur` dispatches the bundled `encode_ssao` (which
//     internally encodes the SSAO pre-pass + GTAO kernel + depth-aware
//     blur). `PassId::SsaoPrepass` / `PassId::SsaoKernel` stay
//     timing-only and the executor rejects them as graph nodes.
//   * `PassId::ParticlesDraw` dispatches the bundled `encode_particles`
//     (compute sim + render draw). `PassId::ParticlesSim` stays
//     timing-only and the executor rejects it as a graph node.

use concinnity_core::gfx::transform::mat4_inverse;
use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;

use crate::gfx::render_graph::{
    CompiledGraph, CompiledPass, GraphResourceClass, PassId, final_states,
};
use crate::gfx::render_types::{LineVertex, TextDrawCall};

use super::barrier_translate::{DxBarrier, d3d12_barrier, d3d12_restore};
use super::context::DxContext;
use super::parallel_encoder::{ParallelCtxRef, SendableCmdList, pool_index};
use super::texture::{aliasing_barrier, transition_barrier, uav_barrier};

// One resolved barrier target: the D3D12 resources a graph resource backs, its
// class, and its resting state (created / cross-frame-restored). Built once per
// frame by `build_barrier_registry`; the resources are refcount clones, read only
// to record transitions into a worker's command list.
//
// `resources` is a list because a graph resource may stand for several GPU
// objects that are always in the same state, transitioned in one
// `ResourceBarrier`. Nothing uses that today -- the G-buffer pre-pass's
// attachments are separate graph resources, since their consumers differ -- but
// the shape is what keeps one timeline per object available when a future
// resource genuinely needs it.
struct DxBarrierTarget {
    resources: Vec<ID3D12Resource>,
    class: GraphResourceClass,
    resting: D3D12_RESOURCE_STATES,
}

// `ResourceId`-indexed table of barrier targets for the migrated graph resources
// (`None` for every resource the executor doesn't graph-drive). A resource is
// graph-driven iff it has a `Some` entry, so this table is the single source of
// truth that replaced the old label allowlist + per-label resolver. Built on the
// main thread by `build_barrier_registry`, where the only field-naming of the
// migrated resources lives (so it is what re-cuts when those fields move into
// sub-structs); the parallel emit path stays field-agnostic.
struct DxBarrierRegistry(Vec<Option<DxBarrierTarget>>);

// SAFETY: same read-only contract as `ParallelCtxRef` / `SendableCmdList` (see
// `parallel_encoder.rs`). The registry holds refcount clones of D3D12 resource
// handles that workers only read, to record `ResourceBarrier` calls into their
// own command lists; every worker joins before the borrow that built the
// registry ends. D3D12 device-derived objects are thread-safe for shared read
// per Microsoft's free-threading rules.
unsafe impl Sync for DxBarrierRegistry {}

// Per-pass aliasing barriers, indexed by topological pass position: the pooled
// transients this pass first-writes that reclaim a shared heap region from an
// earlier transient. Built once per frame on the main thread; the resources are
// refcount clones workers only read.
struct DxAliasBarriers(Vec<Vec<ID3D12Resource>>);

// SAFETY: same read-only contract as `DxBarrierRegistry` above.
unsafe impl Sync for DxAliasBarriers {}

// Emit the aliasing barriers for a pass: for each pooled transient that reclaims
// a shared heap region here, announce the reuse, then re-initialize the resource
// so its first write is legal. The aliasing barrier leaves the memory's contents
// undefined and D3D12 rejects a placed render target's use until a
// Clear/Discard/Copy initializes it, so Discard each (in RENDER_TARGET, then
// back to its resting PIXEL_SHADER_RESOURCE state) before the producing pass's
// own resting -> RENDER_TARGET transition runs. The pass then fully overwrites
// it. Both managed transients rest sampled; a future non-sampled aliased member
// would need its resting state threaded through here.
fn emit_alias_barriers(cmd: &ID3D12GraphicsCommandList, resources: &[ID3D12Resource]) {
    const RESTING: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    for res in resources {
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.ResourceBarrier(&[aliasing_barrier(res)]);
            cmd.ResourceBarrier(&[transition_barrier(
                res,
                RESTING,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            )]);
            cmd.DiscardResource(res, None);
            cmd.ResourceBarrier(&[transition_barrier(
                res,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                RESTING,
            )]);
        }
    }
}

// Emit the native transitions for the migrated graph resources from a pass's
// `barriers_before`, resolved through the registry. Called at the start of each
// pass's own command list, before the pass encodes, so the transition lands in
// the same submission slot the prior inline barrier used to. A resource with no
// registry entry is skipped and keeps its inline barriers. Takes no `DxContext`:
// the field-to-resource mapping was already resolved into the registry, so this
// parallel path is field-agnostic.
fn emit_graph_barriers(
    cmd: &ID3D12GraphicsCommandList,
    registry: &DxBarrierRegistry,
    pass: &CompiledPass,
) {
    for op in &pass.barriers_before {
        let Some(Some(target)) = registry.0.get(op.resource_index()) else {
            continue;
        };
        let Some(barrier) = d3d12_barrier(
            target.class,
            target.resting,
            op.source_state(),
            op.to_state(),
            op.read_stages(),
        ) else {
            continue;
        };
        let native: Vec<D3D12_RESOURCE_BARRIER> = target
            .resources
            .iter()
            .map(|r| match barrier {
                DxBarrier::Transition(before, after) => transition_barrier(r, before, after),
                DxBarrier::Uav => uav_barrier(r),
            })
            .collect();
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.ResourceBarrier(&native);
        }
    }
}

// Everything a pass owes its command list before its body: the aliasing barriers
// for any pooled transient it first-writes (which must precede the resting ->
// RENDER_TARGET transition below), then its graph-derived transitions. Vulkan
// emits the two halves in the same order, for the same reason.
//
// One function because the two recording paths are otherwise asymmetric --
// Composite records into the outer "end" list on the main thread while every
// other pass fans out to a worker -- and that asymmetry is exactly how Composite
// came to record no graph barriers at all, silently dropping any a driven
// resource declared there.
fn emit_pass_prologue(
    cmd: &ID3D12GraphicsCommandList,
    registry: &DxBarrierRegistry,
    alias: &DxAliasBarriers,
    idx: usize,
    pass: &CompiledPass,
) {
    emit_alias_barriers(cmd, &alias.0[idx]);
    emit_graph_barriers(cmd, registry, pass);
}

// Return every driven resource the frame left off its resting state, so the next
// frame's first transition for it names the state the resource is really in (the
// debug layer rejects a mismatch). Recorded last into the outer "end" command
// list, which executes after every pass list. A frame that ends every resource at
// rest emits nothing.
fn emit_graph_restores(
    cmd: &ID3D12GraphicsCommandList,
    registry: &DxBarrierRegistry,
    graph: &CompiledGraph,
) {
    for (idx, (state, stages)) in final_states(graph).into_iter().enumerate() {
        let Some(Some(target)) = registry.0.get(idx) else {
            continue;
        };
        let Some((before, after)) = d3d12_restore(target.class, target.resting, state, stages)
        else {
            continue;
        };
        let native: Vec<D3D12_RESOURCE_BARRIER> = target
            .resources
            .iter()
            .map(|r| transition_barrier(r, before, after))
            .collect();
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.ResourceBarrier(&native);
        }
    }
}

// Check the graph's barrier coverage and cross-frame state contract for every
// resource this executor drives, on the frame's real compiled graph. Two
// invariants, both cheap enough to run per frame under `debug_assertions`:
//
//   * every declared read / write of a driven resource is preceded by a
//     transition putting it in the matching state, in the consuming stage;
//   * a driven resource is back in the resting state its registry entry declares
//     once the frame's restores have run, so the next frame's first transition
//     (whose `Undefined` source resolves to that resting state) names the state
//     the resource is really in. The debug layer rejects a mismatch. This is the
//     check the restore pass exists to satisfy; it fires if a resource ends in a
//     state no restore can express.
//
// This is where a registry entry that claims a resource the graph does not fully
// cover shows up; the headless sweep in `render_graph::validate` covers the
// deriver itself. Mirrors the Vulkan executor's `debug_assert_graph_drives`.
#[cfg(debug_assertions)]
fn debug_assert_graph_drives(graph: &CompiledGraph, registry: &DxBarrierRegistry) {
    use super::barrier_translate::d3d12_state;
    use crate::gfx::render_graph::{ResourceState, barrier_coverage_gaps_for_driven};

    let driven: Vec<bool> = registry.0.iter().map(|t| t.is_some()).collect();
    let gaps = barrier_coverage_gaps_for_driven(graph, &driven);
    assert!(
        gaps.is_empty(),
        "render graph (directx): uncovered accesses on graph-driven resources: {}",
        gaps.iter()
            .map(|g| g.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );

    for (idx, (state, stages)) in final_states(graph).into_iter().enumerate() {
        let Some(Some(target)) = registry.0.get(idx) else {
            continue;
        };
        if state == ResourceState::Undefined {
            continue;
        }
        let restored = match d3d12_restore(target.class, target.resting, state, stages) {
            Some((_, after)) => after,
            None => d3d12_state(target.class, state, stages),
        };
        assert_eq!(
            restored.0, target.resting.0,
            "render graph (directx): {} rests in {:?} but the frame leaves it in {:?}",
            graph.resources[idx].label, target.resting, restored,
        );
    }
}

// The back-buffer render target the composite pass writes: the resource (for its
// RENDER_TARGET <-> PRESENT transitions) plus its RTV handle.
#[derive(Clone, Copy)]
pub(in crate::directx) struct CompositeRenderTarget<'a> {
    pub back_buffer: &'a ID3D12Resource,
    pub back_buffer_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
}

// Output (drawable) resolution the composite pass writes at. Under temporal
// upscaling this differs from the scene render dimensions.
#[derive(Clone, Copy)]
pub(in crate::directx) struct CompositeResolution {
    pub width: u32,
    pub height: u32,
}

// Camera + viewport state for the bindless main pass.
#[derive(Clone, Copy)]
pub(in crate::directx) struct MainPassCamera<'a> {
    // Render-target dimensions in pixels.
    pub width: u32,
    pub height: u32,
    // Camera frustum for per-cluster culling and LOD selection.
    pub frustum: &'a crate::gfx::frustum::Frustum,
    // Camera world position.
    pub cam_pos: [f32; 3],
}

// GPU virtual addresses of this frame's view / light / shadow constant buffers.
#[derive(Clone, Copy)]
pub(in crate::directx) struct FrameGpuBuffers {
    pub view_gva: u64,
    pub light_gva: u64,
    // GVA of the static per-scene GpuLight storage buffer (root SRV).
    pub local_lights_gva: u64,
    pub shadow_ubo_gva: u64,
}

// Per-frame params the executor threads into each pass's `encode_*`
// method. Mirrors the Vulkan executor's `GraphFrameParams`.
pub(in crate::directx) struct GraphFrameParams<'a> {
    pub cmd: &'a ID3D12GraphicsCommandList,
    pub frame_idx: usize,
    pub back_buffer: &'a ID3D12Resource,
    pub back_buffer_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub text_calls: &'a [TextDrawCall],
    // This frame's expanded line ribbons, consumed by the Lines pass. Empty
    // whenever nothing published lines, in which case the graph carries no
    // Lines node either.
    pub lines: &'a [LineVertex],
    // An opaque menu backdrop hides the scene: the Main pass clears its target
    // and skips every draw (the masked graph drops all other world passes), so
    // nothing of the world renders behind the menu.
    pub world_hidden: bool,
    // Scene SRV the composite shader samples: TAA history when TAA is
    // on, SSR output when SSR is on and TAA is off, raw HDR scene SRV
    // otherwise. Computed in `record_frame` once before the dispatch;
    // stable across the whole graph because `taa.frame` only ticks
    // after Composite, so reading `taa.output_index()` upfront points
    // at the same TAA history slot the TaaResolve encoder will write
    // into and the Composite encoder samples.
    pub scene_srv: D3D12_GPU_DESCRIPTOR_HANDLE,
    // Off-screen scene render resolution. Every scene pass (Shadow, Main,
    // SSAO, SSR, Velocity, Fog, Raymarch, Decals, Particles) rasterises at
    // this size, and the sub-pixel jitter is converted to NDC against it.
    // Equals `output_*` when temporal upscaling is off.
    pub width: u32,
    pub height: u32,
    // Drawable (swapchain) resolution. Only the Composite + text pass uses
    // these; it renders the fullscreen tonemap triangle into the
    // output-sized back buffer and sets the text overlay's window-dim
    // uniform from them. Bloom samples its own output-sized mip extents, so
    // it needs no dim param. Equals `width`/`height` when upscaling is off.
    pub output_width: u32,
    pub output_height: u32,
    // Camera world-space position. Shadow uses it for CSM cascade
    // distance bookkeeping inside `encode_shadow_pass`; Main uses it
    // for per-cluster distance culling and the SSAO bundle's pre-pass;
    // future migrating passes (SSAO standalone, SSR pre-pass,
    // Velocity) will share it.
    pub cam_pos: [f32; 3],
    // GPU virtual address of this frame's `ShadowUniforms` constant
    // buffer (the cached cascade VPs + light direction). Consumed by
    // Shadow and Main.
    pub shadow_ubo_gva: u64,
    // GPU virtual address of this frame's `ViewUniforms` constant
    // buffer. Consumed by Main.
    pub view_gva: u64,
    // GPU virtual address of the shared `LightUniforms` constant
    // buffer. Consumed by Main (and any future pass that lights the
    // scene).
    pub light_gva: u64,
    // GPU virtual address of the static per-scene `GpuLight` storage
    // buffer. Consumed by Main's bindless + legacy sub-passes as a root SRV.
    pub local_lights_gva: u64,
    // Jittered camera view-projection matrix (sub-pixel Halton jitter
    // applied when TAA is on). Consumed by Main and Velocity (the
    // jittered VP path); when SSR-prepass migrates it shares this.
    pub vp_mat: [[f32; 4]; 4],
    // Un-jittered camera view-projection matrix. Velocity uses it
    // alongside `vp_mat` (jittered) and the prior frame's `prev_vp`
    // stored on the context, so the stored motion vector is free of
    // sub-pixel jitter.
    pub cur_vp: [[f32; 4]; 4],
    // Camera frustum derived from `vp_mat`. Consumed by Main's
    // per-cluster culling and the bundled SSAO pre-pass.
    pub frustum: &'a crate::gfx::frustum::Frustum,
    // Vertical FOV in radians. Consumed by Main's SSAO pre-pass
    // (depth-reconstruction geometry) and SsrResolve's ray-march
    // projection.
    pub fov_y_radians: f32,
    // Camera aspect ratio (width / height). Consumed by SsrResolve
    // for the ray-march projection.
    pub aspect: f32,
    // Seconds since the engine started. Consumed by ParticlesDraw's
    // bundled compute sim (delta-time computed against the last
    // per-emitter elapsed snapshot stored on `DxContext`).
    pub elapsed: f32,
    // Camera near-plane in view units. Consumed by `FogFroxel` to map the
    // front edge of the froxel volume onto view-space depth, and by
    // `Upscale` (FSR3 dispatch's `cameraNear`).
    pub near: f32,
    // Camera far-plane in view units. Consumed by `Upscale` (FSR3
    // dispatch's `cameraFar`).
    pub far: f32,
    // BVH-culled visible-object indices (sorted, with `draw.always`
    // appended). Consumed by Main's bindless + legacy + instanced
    // sub-passes.
    pub visible: &'a [u32],
}

impl DxContext {
    // Walk a compiled render graph and dispatch each non-composite pass
    // to its own freshly-reset per-pass `ID3D12GraphicsCommandList`,
    // fanning the encode work across rayon workers. Composite stays on
    // the outer "end" cmd list (`params.cmd`) the caller provides; the
    // final timestamp + `ResolveQueryData` ride the same submission.
    // Returns the closed per-pass cmd lists in topological pass order
    // (excluding composite); the caller submits them via
    // `ExecuteCommandLists` between the "start" outer cmd list (which
    // holds the timestamp pre-init) and the "end" outer cmd list (which
    // holds composite + post).
    //
    // `&self` mirrors every DirectX `encode_*` method; per-frame mutable
    // state lives behind `RwLock` / `Cell` / `AtomicU32` so the encoders
    // stay sound under the parallel fan-out (see
    // [`super::parallel_encoder`] for the Send/Sync contract).
    pub(in crate::directx) fn execute_graph(
        &self,
        graph: &CompiledGraph,
        params: &GraphFrameParams<'_>,
    ) -> Result<Vec<ID3D12GraphicsCommandList>, String> {
        // Find Composite's slot (if any) so we can skip it in the
        // worker fan-out and run it inline on the main thread instead.
        let composite_idx = graph.passes.iter().position(|p| p.id == PassId::Composite);

        // Slot per graph pass: each worker stashes its closed cmd list
        // here on success, indexed by topological position. Main thread
        // collects them into the return Vec in order after the join.
        let worker_slots: Mutex<Vec<Option<SendableCmdList>>> =
            Mutex::new((0..graph.passes.len()).map(|_| None).collect());
        let first_error: Mutex<Option<String>> = Mutex::new(None);

        let ctx_ref = ParallelCtxRef::new(self);
        // Resolve every migrated resource's barrier target once, on the main
        // thread, then share the table read-only into the parallel pass workers.
        let registry = self.build_barrier_registry(graph, params.frame_idx);
        #[cfg(debug_assertions)]
        debug_assert_graph_drives(graph, &registry);
        #[cfg(debug_assertions)]
        crate::gfx::render_graph::assert_slot_aliasing_sound(
            graph,
            self.transient_pool.slot_labels(),
            "directx",
        );
        let registry_ref = &registry;
        // Likewise resolve the per-pass aliasing barriers (which pooled transients
        // reclaim a shared heap region) once, shared read-only into the workers.
        let alias_barriers = self.build_alias_barriers(graph);
        let alias_barriers_ref = &alias_barriers;
        let frame_idx = params.frame_idx;

        crate::jobs::pool().install(|| {
            rayon::scope(|scope| {
                for (idx, pass) in graph.passes.iter().enumerate() {
                    if Some(idx) == composite_idx {
                        continue;
                    }
                    let pass_id = pass.id;
                    let first_error_ref = &first_error;
                    let worker_slots_ref = &worker_slots;
                    scope.spawn(move |_| {
                        let ctx = ctx_ref.as_ctx();
                        let pool_idx = pool_index(frame_idx, pass_id);
                        let alloc = &ctx.commands.pass_allocators[pool_idx];
                        let cmd = &ctx.commands.pass_cmd_lists[pool_idx];

                        // Reset this pass's allocator + cmd list so we
                        // can record fresh into it. The previous frame's
                        // submission for this same (frame, pass) slot
                        // has already retired by the time we get here
                        // (the FRAMES-deep fence wait at the top of
                        // `draw_frame` gates the entire slot).
                        // SAFETY: the fence for this frame slot was already waited on, so no
                        // submission still references what is being reset.
                        if let Err(e) = unsafe { alloc.Reset() } {
                            let mut lock = first_error_ref.lock().unwrap();
                            if lock.is_none() {
                                *lock = Some(format!(
                                    "per-pass allocator reset ({}): {e}",
                                    pass_id.name()
                                ));
                            }
                            return;
                        }
                        // SAFETY: the fence for this frame slot was already waited on, so no
                        // submission still references what is being reset.
                        if let Err(e) = unsafe { cmd.Reset(alloc, None) } {
                            let mut lock = first_error_ref.lock().unwrap();
                            if lock.is_none() {
                                *lock = Some(format!(
                                    "per-pass cmd list reset ({}): {e}",
                                    pass_id.name()
                                ));
                            }
                            return;
                        }

                        // Per-pass GPU timing: bracket the encoder with
                        // start + end TIMESTAMP `EndQuery` calls into
                        // pre-allocated heap slots. The frame's whole
                        // block is resolved by the "end" outer cmd list
                        // at the end of the frame and read back at the
                        // top of the next frame. See
                        // [`super::pass_timing`] for the slot layout.
                        if let Some(heap) = ctx.timestamps.query_heap.as_ref() {
                            let (start_slot, _) = super::pass_timing::pass_pair(frame_idx, pass_id);
                            // SAFETY: the command list is in the recording state, and every
                            // resource, descriptor and slice these commands name is live for the
                            // call.
                            unsafe {
                                cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, start_slot);
                            }
                        }

                        emit_pass_prologue(cmd, registry_ref, alias_barriers_ref, idx, pass);

                        let encode_result = ctx.encode_pass_into(pass_id, cmd, params);

                        if let Some(heap) = ctx.timestamps.query_heap.as_ref() {
                            let (_, end_slot) = super::pass_timing::pass_pair(frame_idx, pass_id);
                            // SAFETY: the command list is in the recording state, and every
                            // resource, descriptor and slice these commands name is live for the
                            // call.
                            unsafe {
                                cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, end_slot);
                            }
                        }

                        // SAFETY: the command list is live and in the recording state, which is
                        // what `Close` requires.
                        if let Err(e) = unsafe { cmd.Close() } {
                            let mut lock = first_error_ref.lock().unwrap();
                            if lock.is_none() {
                                *lock = Some(format!(
                                    "per-pass cmd list close ({}): {e}",
                                    pass_id.name()
                                ));
                            }
                            return;
                        }

                        match encode_result {
                            Ok(()) => {
                                let mut lock = worker_slots_ref.lock().unwrap();
                                lock[idx] = Some(SendableCmdList(cmd.clone()));
                            }
                            Err(e) => {
                                let mut lock = first_error_ref.lock().unwrap();
                                if lock.is_none() {
                                    *lock = Some(e);
                                }
                            }
                        }
                    });
                }
            });
        });

        if let Some(err) = first_error.into_inner().unwrap_or(None) {
            return Err(err);
        }

        // Composite stays on the outer "end" cmd list (`params.cmd`) the
        // caller supplied. The final timestamp `EndQuery` +
        // `ResolveQueryData` are appended onto the same cmd list by
        // `draw_frame` after this returns, so composite + post-resolve
        // ride one submission.
        if let Some(idx) = composite_idx {
            if let Some(heap) = self.timestamps.query_heap.as_ref() {
                let (start_slot, _) = super::pass_timing::pass_pair(frame_idx, PassId::Composite);
                // SAFETY: the command list is in the recording state, and every resource,
                // descriptor and slice these commands name is live for the call.
                unsafe {
                    params
                        .cmd
                        .EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, start_slot);
                }
            }
            emit_pass_prologue(
                params.cmd,
                &registry,
                &alias_barriers,
                idx,
                &graph.passes[idx],
            );
            self.encode_composite_and_text(
                params.cmd,
                params.frame_idx,
                CompositeRenderTarget {
                    back_buffer: params.back_buffer,
                    back_buffer_rtv: params.back_buffer_rtv,
                },
                params.text_calls,
                params.scene_srv,
                // Composite runs at drawable resolution; it samples the
                // (output-sized) upscaler result / scene SRV through a
                // fullscreen triangle and writes the output-sized back
                // buffer. Under upscaling this differs from the scene
                // render dims in `params.width`/`height`.
                CompositeResolution {
                    width: params.output_width,
                    height: params.output_height,
                },
            )?;
            if let Some(heap) = self.timestamps.query_heap.as_ref() {
                let (_, end_slot) = super::pass_timing::pass_pair(frame_idx, PassId::Composite);
                // SAFETY: the command list is in the recording state, and every resource,
                // descriptor and slice these commands name is live for the call.
                unsafe {
                    params
                        .cmd
                        .EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, end_slot);
                }
            }
        }

        // Return every driven resource the frame left off its resting state.
        // Recorded last into the outer "end" list, which executes after every
        // pass list.
        emit_graph_restores(params.cmd, &registry, graph);

        // Collect every worker-encoded cmd list in topological pass
        // order. The empty slots (composite, plus any skipped no-op
        // pass that returned without stashing) drop out; workers only
        // stash on success.
        let slots = worker_slots
            .into_inner()
            .map_err(|_| "graph executor (directx): worker slot mutex poisoned".to_string())?;
        let mut ordered = Vec::with_capacity(graph.passes.len());
        for cb in slots.into_iter().flatten() {
            ordered.push(cb.0);
        }
        Ok(ordered)
    }

    // Resolve every migrated graph resource to its barrier target, indexed by
    // `ResourceId` (its position in `graph.resources`), so the parallel emit path
    // can look a target up by `BarrierOp::resource_index()`. This is the single
    // place that names the migrated resources' backing `DxContext` fields;
    // field-grouping re-cuts here, not in the executor. A resource the owning
    // feature disabled (or one never migrated) gets `None`, and the graph carries
    // no barrier for it either.
    fn build_barrier_registry(&self, graph: &CompiledGraph, frame_idx: usize) -> DxBarrierRegistry {
        DxBarrierRegistry(
            graph
                .resources
                .iter()
                .map(|res| {
                    let class = res.class()?;
                    let (resources, resting) =
                        self.barrier_objects_for_label(res.label, frame_idx)?;
                    Some(DxBarrierTarget {
                        resources,
                        class,
                        resting,
                    })
                })
                .collect(),
        )
    }

    // Resolve, per pass, the pooled transients that reclaim a shared heap region
    // when this pass first-writes them. A resource is aliased iff the pool gives
    // it a slot predecessor; its aliasing barrier lands before the pass at its
    // `lifetime.first`. Empty for every resource the pool does not alias, so the
    // table is empty whenever no slot is shared this frame (e.g. bloom off leaves
    // `ao_output` aliased but `bloom_top` absent from the graph; ssao off leaves
    // `bloom_top` un-aliased). Mirrors the Vulkan executor's `build_alias_barriers`.
    fn build_alias_barriers(&self, graph: &CompiledGraph) -> DxAliasBarriers {
        let mut table: Vec<Vec<ID3D12Resource>> = vec![Vec::new(); graph.passes.len()];
        for res in &graph.resources {
            if self.transient_pool.alias_predecessor(res.label).is_none() {
                continue;
            }
            if let Some(r) = self.transient_pool.resource_for(res.label) {
                let first = res.lifetime.first;
                if first < table.len() {
                    table[first].push(r.clone());
                }
            }
        }
        DxAliasBarriers(table)
    }

    // Map one graph resource label to the D3D12 resources backing it and their
    // resting state (the state they were created in and return to at the end of
    // every frame), so a first-use `Undefined` transition names the state the
    // resource is really in. The barrier class is NOT decided here: it follows
    // the usage the graph declares (`CompiledResource::class`), so this backend
    // and Vulkan cannot disagree about what a resource is. Resting state stays
    // per-resource because it is not derivable from usage: `shadow_map` and
    // `hdr_depth` are both depth targets, and one rests sampled while the other
    // rests as a depth attachment. `None` means the owning feature is inactive,
    // and the graph carries no node for it either.
    fn barrier_objects_for_label(
        &self,
        label: &str,
        frame_idx: usize,
    ) -> Option<(Vec<ID3D12Resource>, D3D12_RESOURCE_STATES)> {
        // A per-frame buffer resolves through the frame slot being recorded.
        let buffer = |slots: &[ID3D12Resource], resting| {
            slots.get(frame_idx).map(|r| (vec![r.clone()], resting))
        };
        // The common case: one graph resource, one GPU object.
        let one = |r: &ID3D12Resource, resting| (vec![r.clone()], resting);
        const SAMPLED: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        match label {
            // Indirect commands the cull kernel writes (UAV) and the main pass
            // consumes through `ExecuteIndirect`.
            "draw_args" => buffer(
                &self.cull.indirect_cmd_buffers,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
            // Phase-2 counterpart, written by `Cull2` and consumed by `Main2`.
            "draw_args2" => buffer(
                &self.cull.indirect_cmd_buffers_2,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
            // Phase 1 writes it and phase 2 reads it through the same root UAV,
            // so it never leaves `UNORDERED_ACCESS`.
            "cull_status" => buffer(
                &self.cull.cull_status_buffers,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
            // Per-cluster light index lists: the dispatch flips them to UAV and
            // back, so they rest sampled. One buffer, not per-frame.
            "cluster_light_list" => Some(one(&self.light_cull.cluster_buffer, SAMPLED)),
            "ao_output" => self
                .transient_pool
                .resource_for("ao_output")
                .map(|r| one(r, SAMPLED)),
            // The bloom chain's half-resolution top octave, and the only mip the
            // graph models: the Bloom node writes it, Composite samples it, and
            // the finer octaves in between never leave the node. Pooled, so it
            // resolves through the transient pool exactly like `ao_output`.
            "bloom_top" => self
                .transient_pool
                .resource_for("bloom_top")
                .map(|r| one(r, SAMPLED)),
            // The cascade array rests sampled: the Shadow producer barrier is the
            // real cross-frame reset for this frame's shadow loop and the Main
            // consumer returns it to sampled. Created sampled, so frame 0's
            // producer starts from the resource's real state.
            "shadow_map" => self
                .shadow
                .resource
                .as_ref()
                .filter(|_| !self.shadow.dsvs.is_empty())
                .map(|s| one(&s.resource, SAMPLED)),
            // The spot array rests sampled exactly like the cascades. Only
            // imported when the world has shadowed spots, so the `dsvs` filter
            // matches the graph gate.
            "spot_shadow_map" => self
                .spot_shadow
                .resource
                .as_ref()
                .filter(|_| !self.spot_shadow.dsvs.is_empty())
                .map(|s| one(&s.resource, SAMPLED)),
            // The froxel volume rests sampled: the FogFroxel producer opens it for
            // the compute write and the Fog consumer closes it for the sample.
            "fog_froxel_volume" => self
                .fog
                .resources
                .as_ref()
                .map(|f| one(&f.volume_resource, SAMPLED)),
            // Main depth rests as the depth attachment: one resource shared by
            // every frame in flight, created in DEPTH_WRITE, and the frame's
            // restore returns it there for the next main pass. This is the case
            // that keeps resting per-resource rather than per-class -- shadow_map
            // is the same class and rests sampled.
            "hdr_depth" => Some(one(&self.depth.resource, D3D12_RESOURCE_STATE_DEPTH_WRITE)),
            // The multisample colour attachment, which exists only when the
            // world is multisampled -- and so does the graph resource. It rests
            // in RENDER_TARGET and no pass ever samples it, so every derived
            // transition collapses to a no-op and the entry drives nothing. It
            // is registered anyway because that is what makes the MSAA resolve's
            // RENDER_TARGET <-> RESOLVE_SOURCE pair intra-pass rather than a
            // frame-path transition the graph left behind.
            "hdr_color" => self
                .hdr
                .resolve
                .is_some()
                .then(|| one(&self.hdr.color, D3D12_RESOURCE_STATE_RENDER_TARGET)),
            // The single-sample scene spine every decoration blends into. Which
            // object backs it, and where it rests, both follow MSAA: with MSAA
            // on it is the resolve target and rests sampled; with MSAA off there
            // is no resolve step and `hdr.color` *is* the spine, left in
            // RENDER_TARGET for the next frame's main pass. Its class is the
            // same either way.
            "hdr_resolve" => Some(match &self.hdr.resolve {
                Some(resolve) => one(resolve, SAMPLED),
                None => one(&self.hdr.color, D3D12_RESOURCE_STATE_RENDER_TARGET),
            }),
            // The scene-with-reflections the post stack consumes. Declared by
            // the graph exactly when a reflection resolve runs, which is the
            // same predicate that builds this target, so the entry resolves
            // whenever the resource exists.
            "scene_pre_taa" => self
                .reflection_composite
                .as_ref()
                .map(|rc| one(&rc.output, SAMPLED)),
            // The post-TAA scene. Two mutually exclusive writers back it, and
            // only one is driven: the TAA resolve writes this frame's ping-pong
            // history slot, which rests sampled like any other colour target,
            // while the temporal upscaler writes a compute output whose
            // between-frames state depends on whether a previous frame
            // dispatched (`output_is_psr`). The graph's resting model has no way
            // to say that, so under upscaling the entry stays `None` and
            // `post/upscale/fsr.rs` keeps owning its transitions.
            "scene_color" => self
                .taa
                .as_ref()
                .filter(|_| self.upscale.backend.is_none())
                .map(|taa| one(&taa.history[taa.output_index()], SAMPLED)),
            // The unified G-buffer pre-pass's colour targets, one entry each.
            // One draw writes all three, but their consumers differ -- the
            // reflection resolve reads normal+depth and roughness, the temporal
            // passes read velocity -- so they are separate graph resources with
            // separate lifetimes. All three rest sampled.
            "gbuffer_normal_depth" => self
                .gbuffer
                .as_ref()
                .map(|gb| one(&gb.normal_depth, SAMPLED)),
            "gbuffer_roughness" => self.gbuffer.as_ref().map(|gb| one(&gb.roughness, SAMPLED)),
            "gbuffer_velocity" => self.gbuffer.as_ref().map(|gb| one(&gb.velocity, SAMPLED)),
            // `gbuffer_depth` is deliberately unregistered: it is a depth
            // target rather than a colour one, and the only pass that moves it
            // is the upscaler, which borrows it inside its own dispatch.
            // The Hi-Z pyramid rests where the cull kernel samples it, which is a
            // compute stage, so it is the non-pixel shader-resource state rather
            // than the sampled default.
            "hiz_pyramid" => self
                .cull
                .hiz
                .as_ref()
                .map(|h| one(&h.texture, h.rest_state)),
            _ => None,
        }
    }

    // Build the per-frame `RaymarchView` cbuffer payload from the
    // graph executor's frame params. The matrix inputs match what the
    // Main pass rasterises with (the un-jittered VP), so raymarched
    // surfaces share their NDC depth space with rasterised geometry.
    fn build_raymarch_view(&self, params: &GraphFrameParams<'_>) -> super::raymarch::RaymarchView {
        let inv_vp = mat4_inverse(params.cur_vp);
        super::raymarch::RaymarchView {
            vp: params.cur_vp,
            inv_vp,
            cam_pos: params.cam_pos,
            _pad0: 0.0,
            viewport: [params.width as f32, params.height as f32],
            time: params.elapsed,
            prefilter_mip_count: self.env_map.prefilter_mip_count as f32,
        }
    }

    // Build the per-frame `TransparentView` cbuffer payload for the transparent pass.
    // Uses the jittered VP (`vp_mat`) the Main pass rasterised with, so the
    // glass quad's clip-space depth matches the stored main-depth the fragment
    // shader tests against. Mirrors `encode_decals`' use of `vp_mat`.
    fn build_transparent_view(
        &self,
        params: &GraphFrameParams<'_>,
    ) -> super::transparent::TransparentView {
        let inv_vp = mat4_inverse(params.vp_mat);
        super::transparent::TransparentView {
            vp: params.vp_mat,
            inv_vp,
            camera_pos: [params.cam_pos[0], params.cam_pos[1], params.cam_pos[2], 0.0],
            viewport: [params.width as f32, params.height as f32],
            time: params.elapsed,
            prefilter_mip_count: self.env_map.prefilter_mip_count as f32,
        }
    }

    // Per-pass dispatch, called from both the worker fan-out and the
    // main-thread composite arm. Each arm encodes onto the `cmd` it's
    // given (the worker's per-pass cmd list, or the outer "end" cmd
    // list for composite). Composite is **not** routed through this
    // method; the caller calls `encode_composite_and_text` directly
    // so the trailing timestamp + resolve land on the same cmd list.
    fn encode_pass_into(
        &self,
        pass_id: PassId,
        cmd: &ID3D12GraphicsCommandList,
        params: &GraphFrameParams<'_>,
    ) -> Result<(), String> {
        match pass_id {
            PassId::Cull => {
                self.encode_cull(cmd, params.frame_idx, params.frustum, params.cam_pos);
                // Pose the skinned objects' deformed-vertex buffer for this frame
                // (a no-op when no skinned mesh is folded in). Independent of the
                // cull; both feed Main, which the toposort orders after Cull.
                self.encode_skin(cmd, params.frame_idx);
            }
            PassId::SsaoBlur => {
                self.encode_ssao(cmd, params.fov_y_radians, params.aspect);
            }
            PassId::SsaoPrepass | PassId::SsaoKernel => {
                return Err(format!(
                    "graph executor (directx): pass {} is bundled inside SsaoBlur \
                     (encode_ssao encodes all three SSAO sub-passes); it \
                     should not appear as its own graph node",
                    pass_id.name()
                ));
            }
            PassId::ReflectionComposite => {
                // Metal-only inline pass; never scheduled on DirectX. Handled
                // here only to keep the dispatch match exhaustive.
                return Err(format!(
                    "graph executor (directx): pass {} is a Metal-only inline \
                     reflection composite and should not appear as a graph node",
                    pass_id.name()
                ));
            }
            PassId::LightCull => {
                // Bins the local lights into per-cluster index lists. The
                // builder emits this node only when the world has local lights
                // (matching `clustered_lighting_enabled`), and the RAW edge on
                // `cluster_light_list` pins it before Main, which reads the
                // same buffer.
                self.encode_light_cull(cmd, params.frame_idx)?;
            }
            PassId::SsrPrepass => {
                // Merged into GBufferPrepass on DX: the builder emits the
                // unified node (unified_gbuffer_prepass = true) and never this.
                return Err(format!(
                    "graph executor (directx): pass {} is merged into GBufferPrepass \
                     and should not appear in the frame graph",
                    pass_id.name()
                ));
            }
            PassId::Shadow => {
                // Build the raymarch view only when at least one volume
                // opted in to shadow casting; otherwise pass `None` so
                // the shadow encoder skips the SDF caster sub-pass with
                // zero overhead. The view stays consistent with the
                // matching `PassId::Raymarch` build later this frame:
                // same `cur_vp`, `cam_pos`, `elapsed`, viewport, and
                // prefilter mip count.
                let raymarch_view = self
                    .raymarch
                    .as_ref()
                    .filter(|rm| rm.any_shadow_casters())
                    .map(|_| self.build_raymarch_view(params));
                self.encode_shadow_pass(
                    cmd,
                    params.frame_idx,
                    params.shadow_ubo_gva,
                    params.cam_pos,
                    raymarch_view.as_ref(),
                );
            }
            PassId::SpotShadow => {
                // One depth-only render per scheduled spot slice. The builder
                // emits this node only when the world has shadow-casting spots,
                // and the RAW edge on `spot_shadow_map` pins it before Main.
                self.encode_spot_shadow_pass(cmd, params.frame_idx, params.cam_pos);
            }
            PassId::AutoExposure => {
                self.encode_auto_exposure(cmd, params.frame_idx);
            }
            PassId::Main => {
                self.encode_main_pass(
                    cmd,
                    params.frame_idx,
                    MainPassCamera {
                        width: params.width,
                        height: params.height,
                        frustum: params.frustum,
                        cam_pos: params.cam_pos,
                    },
                    FrameGpuBuffers {
                        view_gva: params.view_gva,
                        light_gva: params.light_gva,
                        local_lights_gva: params.local_lights_gva,
                        shadow_ubo_gva: params.shadow_ubo_gva,
                    },
                    params.visible,
                    params.world_hidden,
                );
            }
            PassId::Decals => {
                self.encode_decals(cmd, params.frame_idx, params.vp_mat, params.frustum);
            }
            PassId::Lines => {
                self.encode_lines(cmd, params.frame_idx, params.vp_mat, params.lines)?;
            }
            PassId::Fog => {
                self.encode_fog(cmd, params.frame_idx, params.vp_mat, params.cam_pos);
            }
            PassId::ParticlesDraw => {
                self.encode_particles(
                    cmd,
                    params.frame_idx,
                    params.elapsed,
                    params.vp_mat,
                    params.frustum,
                );
            }
            PassId::ParticlesSim => {
                return Err(format!(
                    "graph executor (directx): pass {} is bundled inside ParticlesDraw \
                     (encode_particles runs both compute sim and render); it \
                     should not appear as its own graph node",
                    pass_id.name()
                ));
            }
            PassId::SsrResolve => {
                self.encode_ssr_resolve(
                    cmd,
                    params.frame_idx,
                    params.fov_y_radians,
                    params.aspect,
                    params.cam_pos,
                );
            }
            PassId::Velocity => {
                // Merged into GBufferPrepass on DX: the builder emits the
                // unified node (unified_gbuffer_prepass = true) and never this.
                return Err(format!(
                    "graph executor (directx): pass {} is merged into GBufferPrepass \
                     and should not appear in the frame graph",
                    pass_id.name()
                ));
            }
            PassId::TaaResolve => {
                self.encode_taa(cmd);
            }
            PassId::Bloom => {
                self.encode_bloom(cmd, params.scene_srv);
            }
            PassId::Composite => {
                // Composite is run inline on the outer "end" cmd list
                // by `execute_graph` itself so it shares a submission
                // with the trailing timestamp + resolve. This arm is
                // unreachable through the worker fan-out; see the
                // method docstring.
                return Err(
                    "graph executor (directx): Composite must run on the outer cmd \
                     list: encode_pass_into is not the right entry point"
                        .into(),
                );
            }
            PassId::Raymarch => {
                let view = self.build_raymarch_view(params);
                self.encode_raymarch(cmd, params.frame_idx, &view)?;
            }
            PassId::FogFroxel => {
                self.encode_fog_froxel(
                    cmd,
                    params.frame_idx,
                    params.near,
                    params.vp_mat,
                    params.cam_pos,
                    params.shadow_ubo_gva,
                );
            }
            PassId::Upscale => {
                // FSR3 temporal upscaler. Driven by the shared graph
                // when `FrameGraphInputs::upscale_enabled` is on (see
                // `record_frame::seed_inputs`). The encoder dispatches
                // FFX against this pass's per-pass cmd list, reading
                // the post-SSR scene + velocity + main depth and
                // writing into the upscaler's output texture (which
                // bloom + composite then sample via `scene_srv_for_post`).
                self.encode_upscale(cmd, params)?;
            }
            PassId::Transparent => {
                // Generic translucent pass: draws the world's glass panes and
                // water surfaces back-to-front over the post-SSR scene. Gated by
                // `FrameGraphInputs::transparent_enabled`
                // (`DxContext::transparent_enabled`), so it only appears when the
                // world declared a visible `GlassPanel` or `WaterSurface`.
                //
                // Planar reflections run inline at the head of the pass (same cmd
                // list -> the per-plane mirror resolves are ready before the
                // transparent draws sample them). A no-op when the world has no
                // planar set. `planar_pass_needed` decides: a visible water
                // surface holding a slot always needs it (water takes the mirror
                // over the trace), and so does any reflector when the per-pixel
                // trace will not run. Gating on `rt_transparent_active` (not
                // `rt_reflections_active`) keeps planar alive when RT is live but
                // a producer's RT pipelines failed to build, so its probe/planar
                // fallback samples a freshly rendered resolve.
                if self.planar_pass_needed() {
                    self.encode_planar_reflections(cmd, params)?;
                }
                let view = self.build_transparent_view(params);
                self.encode_transparent(
                    cmd,
                    params.frame_idx,
                    &view,
                    params.fov_y_radians,
                    params.aspect,
                )?;
            }
            PassId::HizBuild | PassId::HizFinal => {
                // Two Hi-Z builds share one encoder. `HizBuild` rebuilds the
                // pyramid mid-frame from phase-1 depth so Cull2 re-tests the
                // phase-1 occluded objects against up-to-date depth; `HizFinal`
                // reduces the frame's final depth for the next frame's phase-1
                // cull. Both read the depth the graph has already transitioned.
                self.encode_hiz_build(cmd);
            }
            PassId::Cull2 => {
                self.encode_cull_phase2(cmd, params.frame_idx, params.frustum, params.cur_vp);
            }
            PassId::Main2 => {
                self.encode_main_pass_phase2(
                    cmd,
                    params.frame_idx,
                    params.width,
                    params.height,
                    FrameGpuBuffers {
                        view_gva: params.view_gva,
                        light_gva: params.light_gva,
                        local_lights_gva: params.local_lights_gva,
                        shadow_ubo_gva: params.shadow_ubo_gva,
                    },
                );
            }
            PassId::Ssgi => {
                self.encode_ssgi(cmd, params.frame_idx, params.fov_y_radians, params.aspect);
            }
            PassId::RtReflections => {
                // Hardware ray-traced reflections (DXR inline `RayQuery`). Traces
                // a reflection ray per glossy pixel against the scene TLAS and
                // composites into the RT output target, which `scene_srv_for_post`
                // then feeds the post stack. Occupies the SsrResolve slot; gated
                // by `FrameGraphInputs::rt_reflections_enabled`
                // (`DxContext::rt_reflections_active`). The per-frame TLAS update
                // already ran on the outer "start" cmd list before this trace.
                self.encode_rt_reflections(
                    cmd,
                    params.frame_idx,
                    params.fov_y_radians,
                    params.aspect,
                    params.cam_pos,
                );
            }
            PassId::GBufferPrepass => {
                // Unified geometry pre-pass: one jittered traversal writes
                // normal+depth, roughness, and motion for every screen-space
                // consumer (SSR / SSAO / SSGI / TAA / FSR). `params.vp_mat` is
                // the jittered VP (rasterisation, matching the main pass);
                // `params.cur_vp` is the un-jittered VP the shader uses with the
                // previous VP for the motion vector. The velocity channel
                // carries real motion only when a consumer reads it (TAA or
                // FSR active, i.e. `self.taa.is_some()`); otherwise cur == prev
                // and it stays a harmless zero.
                self.encode_gbuffer_prepass(
                    cmd,
                    params.frame_idx,
                    crate::directx::post::gbuffer::GbufferPrepassView {
                        jittered_vp: params.vp_mat,
                        cur_vp: params.cur_vp,
                        frustum: params.frustum,
                        cam_pos: params.cam_pos,
                    },
                    params.visible,
                    self.taa.is_some(),
                );
            }
        }
        Ok(())
    }
}
