//! DirectX-side executor for the render graph. `DxContext::execute_graph` walks
//! the `CompiledGraph` produced by the shared
//! `concinnity_engine::render::render_graph::build_frame_graph` and dispatches
//! each pass to its `encode_*` method. Every backend drives the same builder.
//!
//! **Per-pass command lists.** Each non-composite pass records into its
//! own `ID3D12GraphicsCommandList` (drawn from the `pass_cmd_lists` pool
//! on `DxContext`). The fan-out runs on `jobs::pool()` through the shared
//! `execute_passes`, so workers encode in parallel; each worker resets its assigned
//! allocator + cmd list, brackets the encode with start/end TIMESTAMP
//! queries, encodes the pass, and closes the cmd list. The main thread
//! then submits every closed cmd list in topological pass order via
//! `ExecuteCommandLists`. The Composite pass keeps using the outer
//! "end" cmd list that `draw_frame` owns (so the final timestamp +
//! `ResolveQueryData` ride the same submission). Mirrors
//! `metal/graph_exec.rs`.
//!
//! Both of a pass's barrier lists are consumed for every resource the registry
//! resolves: `emit_pass_prologue` translates their graph state transitions into
//! `D3D12_RESOURCE_BARRIER` transitions at the start of each pass's own command
//! list -- batched, so a pass costs one `ResourceBarrier` call -- and
//! `emit_graph_restores` returns any that the frame left off their resting state
//! at the end of the outer "end" list. Every other resource still owns its
//! transitions inline in its encoder; `audit/barrier.rs` classifies each
//! remaining site.
//!
//! The registry decides two things per resource: which D3D12 resource backs it,
//! and what state it rests in between frames. Its class -- what a `Write` means --
//! comes from the usage the graph declares, so this executor and the Vulkan one
//! cannot disagree about it. Resting cannot: `shadow_map` and `hdr_depth` are both
//! depth targets, and the first rests sampled while the second rests `DEPTH_WRITE`.
//!
//! Bundled passes:
//!   * `PassId::SsaoBlur` dispatches the bundled `encode_ssao` (which
//!     internally encodes the GTAO depth copy, kernel and depth-aware
//!     blur), timing its sub-passes as `GpuSpan::SsaoDepth` /
//!     `GpuSpan::SsaoKernel`.
//!
//! `PassId::ParticlesSim` and `PassId::ParticlesDraw` are two nodes with their
//! own command lists: the sim integrates every live emitter's pool and the draw
//! reads those pools in its vertex stage, with the transition between them
//! derived from the graph rather than emitted inline.

use concinnity_core::math::narrow_f32;
use concinnity_engine::profile::FrameSpans;
use concinnity_engine::render::backend::FrameParams;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::frame::plan::FramePlan;
use concinnity_engine::render::particles::ParticleFrame;
use concinnity_engine::render::planar_reflection::PlanarFramePlan;
use concinnity_engine::render::render_graph::model_history_pass;
use concinnity_engine::render::render_graph::{
    AliasRows, BarrierOp, BarrierTable, CompiledGraph, CompiledPass, FAR_SHADOW_MAP_RESTS,
    GraphResourceClass, PassId, PassRecorder, PassScratch, execute_passes,
};
use concinnity_engine::render::uniforms::{PassCamera, PassLighting};
use concinnity_engine::render::{backend::pass_timing, render_graph};
use concinnity_engine::transform::mat4_inverse;
use concinnity_host::thread::jobs;
use std::mem::ManuallyDrop;
use windows::Win32::Graphics::Direct3D12::*;

use super::barrier_translate::{DxBarrier, d3d12_barrier, d3d12_restore};
use super::context::DxContext;
use super::parallel_encoder::{ParallelCtxRef, SendableCmdList, pool_index};
use super::texture::{aliasing_barrier, transition_barrier, uav_barrier};
use crate::directx::descriptor_slot::SrvSlot;
use crate::directx::line::LineUpload;

// A D3D12 resource named without a reference count of its own, so the barrier
// tables can persist across frames with no `AddRef` / `Release` traffic. Every
// table holding one is cleared and refilled from live `DxContext` state before a
// frame reads it, so a pointer left from an earlier frame is never dereferenced.
struct ResourceRef(ManuallyDrop<ID3D12Resource>);

impl ResourceRef {
    fn new(resource: &ID3D12Resource) -> Self {
        // SAFETY: an interface is one non-null COM pointer, so this copies the
        // pointer with no refcount change, and `ManuallyDrop` never releases it.
        Self(unsafe { std::mem::transmute_copy(resource) })
    }
}

impl std::ops::Deref for ResourceRef {
    type Target = ID3D12Resource;

    fn deref(&self) -> &ID3D12Resource {
        &self.0
    }
}

// One resolved barrier target: the D3D12 resource a graph resource backs, its
// class, and its resting state (created / cross-frame-restored).
struct BarrierTarget {
    object: DxTargetObject,
    class: GraphResourceClass,
    resting: D3D12_RESOURCE_STATES,
}

// What a graph resource resolves to: one D3D12 resource, or several under one
// label as an arena range (`particle_pool`, one resource for every live
// emitter's pool). Each gets its own barrier.
enum DxTargetObject {
    One(ResourceRef),
    Set { first: usize, count: usize },
}

// `ResourceId`-indexed table of barrier targets for the graph-driven resources
// (`None` for every resource the executor doesn't graph-drive). Refilled on the
// main thread by `refill_graph_tables`, where the only field-naming of the
// graph-driven resources lives; the parallel emit path stays field-agnostic.
type BarrierRegistry = BarrierTable<BarrierTarget, ResourceRef>;

// The resources one target names: a single object, or the whole set.
fn target_objects<'r>(
    registry: &'r BarrierRegistry,
    target: &'r BarrierTarget,
) -> &'r [ResourceRef] {
    match &target.object {
        DxTargetObject::One(r) => std::slice::from_ref(r),
        DxTargetObject::Set { first, count } => registry.set(*first, *count),
    }
}

// One pooled transient a pass reclaims a shared heap region for, resolved once
// per frame on the main thread and filed under the pass that first-writes it.
struct DxAliasBarrier {
    // Its `ResourceId`, so the prologue can recognize the pass's own transition
    // for it and open that from the discard state instead of from rest.
    resource_index: usize,
    resource: ResourceRef,
    // Where it sits between frames: the state the discard opens from, and the
    // one it is put back in when the graph drives it no further.
    resting: D3D12_RESOURCE_STATES,
}

// The per-frame tables the executor builds, kept across frames on `DxContext`
// so their backing allocations are reused.
#[derive(Default)]
pub(in crate::directx) struct GraphScratch {
    registry: BarrierRegistry,
    alias: AliasRows<DxAliasBarrier>,
    #[cfg(debug_assertions)]
    checked: render_graph::CheckedGraphs,
    passes: PassScratch<SendableCmdList>,
}

// Records one graph pass into its own per-`(frame, pass)` command list on a
// job-pool worker.
struct DxPassRecorder<'a, 'p> {
    ctx: ParallelCtxRef<'a>,
    params: &'a GraphFrameParams<'p>,
    particle: Option<&'a ParticleFrame>,
    grass: Option<&'a super::grass::GrassFrame>,
    registry: &'a BarrierRegistry,
    alias: &'a AliasRows<DxAliasBarrier>,
}

impl PassRecorder for DxPassRecorder<'_, '_> {
    type Recorded = SendableCmdList;

    fn record(&self, index: usize, pass: &CompiledPass) -> RenderResult<SendableCmdList> {
        let ctx = self.ctx.as_ctx();
        let frame_idx = self.params.frame_idx;
        let pass_id = pass.id;
        let pool_idx = pool_index(frame_idx, pass_id);
        let alloc = &ctx.commands.pass_allocators[pool_idx];
        let cmd = &ctx.commands.pass_cmd_lists[pool_idx];
        // Reset this pass's allocator and list; the frame slot's fence already retired
        // the previous submission.
        // SAFETY: the fence for this frame slot was already waited on, so no submission still
        // references what is being reset.
        unsafe { alloc.Reset() }.map_err(|e| {
            super::error::map_hresult(
                e.code(),
                &format!("per-pass allocator reset ({})", pass_id.name()),
            )
        })?;
        // SAFETY: the fence for this frame slot was already waited on, so no submission still
        // references what is being reset.
        unsafe { cmd.Reset(alloc, None) }.map_err(|e| {
            super::error::map_hresult(
                e.code(),
                &format!("per-pass cmd list reset ({})", pass_id.name()),
            )
        })?;
        // Bracket the encoder with a timestamp pair; the end list resolves the block
        // (see [`pass_timing`]).
        if let Some(heap) = ctx.timestamps.query_heap.as_ref() {
            let (start_slot, _) = pass_timing::pass_pair(frame_idx, pass_id);
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe { cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, start_slot) };
        }
        emit_pass_prologue(cmd, self.registry, self.alias.for_pass(index), pass);
        let encoded = ctx.encode_pass_into(pass_id, cmd, self.params, self.particle, self.grass);
        if encoded.is_ok() {
            emit_pass_epilogue(cmd, self.registry, pass);
        }
        if let Some(heap) = ctx.timestamps.query_heap.as_ref() {
            let (_, end_slot) = pass_timing::pass_pair(frame_idx, pass_id);
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe { cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, end_slot) };
        }
        // SAFETY: the command list is live and in the recording state, which is what `Close`
        // requires.
        unsafe { cmd.Close() }.map_err(|e| {
            super::error::map_hresult(
                e.code(),
                &format!("per-pass cmd list close ({})", pass_id.name()),
            )
        })?;
        encoded.map(|()| SendableCmdList(cmd.clone()))
    }
}

// The state a placed resource must be in for `DiscardResource` to re-initialize
// it once an aliasing barrier has claimed its heap region.
const DISCARD_STATE: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_RENDER_TARGET;

// How many barriers one `ResourceBarrier` call carries. A pass declares far
// fewer, so this is a cap rather than a budget: a batch that fills flushes and
// keeps filling, which costs an extra call and stays correct.
const BARRIER_BATCH: usize = 16;

// Barriers gathered for one `ResourceBarrier` call, since D3D12 charges per
// call. Fixed capacity, so no allocation; the borrowed resource pointers tie it
// to the registry's lifetime.
struct BarrierBatch<'a> {
    cmd: &'a ID3D12GraphicsCommandList,
    entries: [D3D12_RESOURCE_BARRIER; BARRIER_BATCH],
    len: usize,
}

impl<'a> BarrierBatch<'a> {
    fn new(cmd: &'a ID3D12GraphicsCommandList) -> Self {
        Self {
            cmd,
            entries: std::array::from_fn(|_| D3D12_RESOURCE_BARRIER::default()),
            len: 0,
        }
    }

    fn push(&mut self, barrier: D3D12_RESOURCE_BARRIER) {
        if self.len == BARRIER_BATCH {
            self.flush();
        }
        self.entries[self.len] = barrier;
        self.len += 1;
    }

    fn flush(&mut self) {
        if self.len == 0 {
            return;
        }
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            self.cmd.ResourceBarrier(&self.entries[..self.len]);
        }
        self.len = 0;
    }
}

// Translate one of a pass's `barriers_before` and stage it. Unregistered
// resources and no-op transitions are skipped. `opened_at` overrides a first
// use's starting state, for a resource this pass just discarded.
fn stage_graph_barrier(
    batch: &mut BarrierBatch<'_>,
    registry: &BarrierRegistry,
    op: &BarrierOp,
    opened_at: Option<D3D12_RESOURCE_STATES>,
) {
    let Some(target) = registry.target(op.resource_index()) else {
        return;
    };
    let Some(barrier) = d3d12_barrier(
        target.class,
        opened_at.unwrap_or(target.resting),
        op.source_state(),
        op.to_state(),
        op.read_stages(),
    ) else {
        return;
    };
    for resource in target_objects(registry, target) {
        batch.push(match barrier {
            DxBarrier::Transition(before, after) => transition_barrier(resource, before, after),
            DxBarrier::Uav => uav_barrier(resource),
        });
    }
}

// One slot of a per-frame buffer ring, as a barrier target: the slot for the
// frame being recorded. A free function rather than a closure so the borrow it
// returns is tied to the ring, not to the resolver's own frame.
fn frame_slot(
    slots: &[ID3D12Resource],
    frame_idx: usize,
    resting: D3D12_RESOURCE_STATES,
) -> Option<(&ID3D12Resource, D3D12_RESOURCE_STATES)> {
    Some((slots.get(frame_idx)?, resting))
}

// Whether the pass's own graph transition takes a reclaimed resource on from the
// discard, so the prologue does not have to put it back at rest itself.
fn graph_takes_over(
    registry: &BarrierRegistry,
    pass: &CompiledPass,
    resource_index: usize,
) -> bool {
    registry.target(resource_index).is_some()
        && pass
            .barriers_before
            .iter()
            .any(|op| op.resource_index() == resource_index)
}

// What a pass owes its list before its body: aliasing barriers and a discard
// for any pooled transient it first writes, then its transitions. Shared by
// Composite and the worker-recorded passes.
fn emit_pass_prologue(
    cmd: &ID3D12GraphicsCommandList,
    registry: &BarrierRegistry,
    aliased: &[DxAliasBarrier],
    pass: &CompiledPass,
) {
    let mut batch = BarrierBatch::new(cmd);
    for a in aliased {
        batch.push(aliasing_barrier(&a.resource));
        if a.resting != DISCARD_STATE {
            batch.push(transition_barrier(&a.resource, a.resting, DISCARD_STATE));
        }
    }
    for op in &pass.barriers_before {
        if aliased
            .iter()
            .any(|a| a.resource_index == op.resource_index())
        {
            continue;
        }
        stage_graph_barrier(&mut batch, registry, op, None);
    }
    batch.flush();

    if aliased.is_empty() {
        return;
    }
    for a in aliased {
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.DiscardResource(&*a.resource, None);
        }
    }
    for op in &pass.barriers_before {
        if aliased
            .iter()
            .any(|a| a.resource_index == op.resource_index())
        {
            // The aliasing barrier lands at the resource's first use, so the
            // transition here opens from no prior state, and the discard is what
            // that resolves to instead of the resting state.
            debug_assert_eq!(
                op.source_state(),
                render_graph::ResourceState::Undefined,
                "a reclaimed transient is transitioned from a state the discard replaced"
            );
            stage_graph_barrier(&mut batch, registry, op, Some(DISCARD_STATE));
        }
    }
    // A reclaimed resource the graph drives no further would otherwise sit in the
    // discard state, which is not where the next frame expects to find it.
    for a in aliased {
        if a.resting != DISCARD_STATE && !graph_takes_over(registry, pass, a.resource_index) {
            batch.push(transition_barrier(&a.resource, DISCARD_STATE, a.resting));
        }
    }
    batch.flush();
}

// The transitions the graph records on the producing side of a read run split
// across the two queues, recorded right after the pass's work.
fn emit_pass_epilogue(
    cmd: &ID3D12GraphicsCommandList,
    registry: &BarrierRegistry,
    pass: &CompiledPass,
) {
    if pass.barriers_after.is_empty() {
        return;
    }
    let mut batch = BarrierBatch::new(cmd);
    for op in &pass.barriers_after {
        stage_graph_barrier(&mut batch, registry, op, None);
    }
    batch.flush();
}

// Return every driven resource the frame left off its resting state, in the
// "end" list, so the next frame's first transition names its real state.
fn emit_graph_restores(
    cmd: &ID3D12GraphicsCommandList,
    registry: &BarrierRegistry,
    graph: &CompiledGraph,
) {
    let mut batch = BarrierBatch::new(cmd);
    for (idx, &(state, stages)) in graph.final_states().iter().enumerate() {
        let Some(target) = registry.target(idx) else {
            continue;
        };
        let Some((before, after)) = d3d12_restore(target.class, target.resting, state, stages)
        else {
            continue;
        };
        for resource in target_objects(registry, target) {
            batch.push(transition_barrier(resource, before, after));
        }
    }
    batch.flush();
}

// Check barrier coverage and the cross-frame state contract for every driven
// resource, once per graph and driven set under `debug_assertions`: every
// access is preceded by a matching transition, and every driven resource ends
// the frame in its resting state.
#[cfg(debug_assertions)]
fn debug_assert_graph_drives(graph: &CompiledGraph, registry: &BarrierRegistry) {
    use super::barrier_translate::d3d12_state;
    use concinnity_engine::render::render_graph::ResourceState;

    let driven: Vec<bool> = registry.driven().collect();
    render_graph::assert_driven_barriers_sound(graph, &driven, "directx");

    for (idx, &(state, stages)) in graph.final_states().iter().enumerate() {
        let Some(target) = registry.target(idx) else {
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

// Render-target dimensions of the main pass, in pixels.
#[derive(Clone, Copy)]
pub(in crate::directx) struct MainPassExtent {
    pub width: u32,
    pub height: u32,
}

// GPU virtual addresses of this frame's view / light / shadow constant buffers.
#[derive(Clone, Copy)]
#[expect(
    clippy::struct_field_names,
    reason = "the shared suffix names what each field holds: a GPU virtual address"
)]
pub(in crate::directx) struct FrameGpuBuffers {
    pub view_gva: u64,
    pub light_gva: u64,
    // GVA of the static per-scene GpuLight storage buffer (root SRV).
    pub local_lights_gva: u64,
    pub shadow_ubo_gva: u64,
}

// Per-frame params the executor threads into each pass's `encode_*` method.
pub(in crate::directx) struct GraphFrameParams<'a> {
    // The frame as planned, and the host's inputs it was planned from.
    pub plan: &'a FramePlan,
    pub frame: &'a FrameParams<'a>,
    pub cmd: &'a ID3D12GraphicsCommandList,
    pub frame_idx: usize,
    pub back_buffer: &'a ID3D12Resource,
    pub back_buffer_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    // This frame's line ribbons, uploaded on the main thread before the fan-out
    // so the Lines pass never touches the upload ring or the allocator. `None`
    // when there is nothing to draw, including whenever the graph carries no
    // Lines node.
    pub lines: Option<LineUpload>,
    // The scene SRV the composite samples, computed once before the dispatch;
    // stable for the graph since `taa.frame` ticks only after Composite.
    pub scene_srv: SrvSlot,
    // Off-screen scene render resolution. Every scene pass (Shadow, Main,
    // SSAO, SSR, Velocity, Fog, Raymarch, Decals, Particles) rasterizes at
    // this size, and the sub-pixel jitter is converted to NDC against it.
    // Equals `output_*` when temporal upscaling is off.
    pub width: u32,
    pub height: u32,
    // The drawable resolution, used only by Composite and text. Equals
    // `width`/`height` without upscaling.
    pub output_width: u32,
    pub output_height: u32,
    // GPU virtual address of this frame's `ShadowUniforms` constant
    // buffer (the cached cascade VPs + light direction). Consumed by
    // Shadow and Main.
    pub shadow_ubo_gva: u64,
    // The prop probe's faces' `ShadowUniforms`: the near cascades only.
    pub probe_shadow_ubo_gva: u64,
    // GPU virtual address of this frame's `ViewUniforms` constant
    // buffer. Consumed by Main.
    pub view_gva: u64,
    // GPU virtual address of the shared `LightUniforms` constant
    // buffer. Consumed by Main (and any future pass that lights the
    // scene).
    pub light_gva: u64,
    // GPU virtual address of the static per-scene `GpuLight` storage
    // buffer. Consumed by Main's bindless pass as a root SRV.
    pub local_lights_gva: u64,
    // Whether this frame's model-history snapshot fills every ring slot rather
    // than only its own: the rebuild's prime request, taken from the tracker
    // before the fan-out. Consumed by the snapshot after the last G-buffer pass.
    pub prime_model_history: bool,
    // Which planar mirrors this frame renders, and the screen rectangle each
    // covers; computed once from the camera so the mirror pass and the transparent
    // pass that samples it agree.
    pub planar: PlanarFramePlan,
    // This frame's particle clock, advanced before recording; `None` when the
    // particle passes are inert.
    pub particle: Option<ParticleFrame>,
}

impl DxContext {
    // Record each pass of a compiled graph: every non-composite pass into its own
    // freshly reset list on a worker, Composite into the caller's "end" list.
    // Returns the closed pass lists in graph order. Per-frame mutable state lives
    // behind atomics and locks so `&self` encoders stay sound in parallel.
    pub(in crate::directx) fn execute_graph(
        &self,
        graph: &CompiledGraph,
        params: &GraphFrameParams<'_>,
        pass_lists: &mut Vec<ID3D12GraphicsCommandList>,
    ) -> RenderResult<()> {
        // Particle per-frame state (dt / frame index / per-emitter spawn runs
        // and their params-ring slots) is advanced here, once, before any pass
        // encodes, so the sim and draw halves record on separate workers against
        // one consistent frame. `None` when the pass is inert.
        let particle_ref = params.particle.as_ref();
        if let Some(frame) = particle_ref {
            self.write_particle_params(params.frame_idx, frame, params.plan.camera.vp);
        }
        // The grass block for the kernel and both draws, written once here and
        // only on a frame whose graph runs the kernel, so the draw-argument
        // slot it fills alternates frame to frame.
        let grass_started = std::time::Instant::now();
        let grass_frame = graph
            .pass(PassId::Grass)
            .is_some()
            .then(|| {
                self.prepare_grass_frame(super::grass::GrassRequest {
                    frame_idx: params.frame_idx,
                    cam_pos: params.plan.cam_pos,
                    vp: params.plan.camera.cur_vp,
                    elapsed: narrow_f32(params.frame.elapsed),
                    cast: graph.pass(PassId::GrassShadow).is_some(),
                    benders: params.frame.grass_benders,
                })
            })
            .flatten();
        self.state.spans.set(FrameSpans {
            grass_us: concinnity_host::clock::micros_since(grass_started),
            ..self.state.spans.get()
        });

        // Resolve every graph-driven resource's barrier target and every pass's
        // aliasing barriers once, on the main thread, then share the tables
        // read-only into the parallel pass workers.
        let mut scratch = self.graph_scratch.take().unwrap_or_default();
        self.refill_graph_tables(&mut scratch, graph, params.frame_idx);
        // This executor records one serial stream in `graph.passes` order, the
        // schedule's queues interleaved back into compiled order, which is legal
        // because the compiled order is a topological order for both queues.
        #[cfg(debug_assertions)]
        {
            let slot_labels = self.targets.transient_pool.slot_labels();
            let salt = scratch.registry.driven_key();
            if scratch.checked.first_sight(graph, slot_labels, salt) {
                debug_assert_graph_drives(graph, &scratch.registry);
                render_graph::assert_graph_sound(graph, slot_labels, "directx");
            }
        }

        let GraphScratch {
            registry,
            alias,
            passes,
            ..
        } = &mut scratch;
        let recorder = DxPassRecorder {
            ctx: ParallelCtxRef::new(self),
            params,
            particle: particle_ref,
            grass: grass_frame.as_ref(),
            registry,
            alias,
        };
        let mut executed = execute_passes(graph, &recorder, jobs::pool(), passes)?;
        let mut stats = self.diagnostics.frame_stats.get();
        stats.recording_fan_out = executed.timing;
        stats.pass_record_us = executed.record_us;
        self.diagnostics.frame_stats.set(stats);
        // Every worker-encoded cmd list, in topological pass order.
        pass_lists.extend(executed.take().map(|(_, list)| list.0));

        // Composite on the "end" list, where the final timestamp and resolve follow.
        let frame_idx = params.frame_idx;
        if let Some(idx) = executed.presenter {
            if let Some(heap) = self.timestamps.query_heap.as_ref() {
                let (start_slot, _) = pass_timing::pass_pair(frame_idx, PassId::Composite);
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
                registry,
                alias.for_pass(idx),
                &graph.passes[idx],
            );
            self.encode_composite_and_text(
                params.cmd,
                params.frame_idx,
                CompositeRenderTarget {
                    back_buffer: params.back_buffer,
                    back_buffer_rtv: params.back_buffer_rtv,
                },
                params.frame.text_calls,
                crate::directx::draw::composite::CompositeSources {
                    scene_srv: params.scene_srv,
                    reactive_valid: params.plan.reactive.readable,
                },
                // Composite runs at drawable resolution, which differs from the render dims
                // under upscaling.
                CompositeResolution {
                    width: params.output_width,
                    height: params.output_height,
                },
            )?;
            emit_pass_epilogue(params.cmd, registry, &graph.passes[idx]);
            if let Some(heap) = self.timestamps.query_heap.as_ref() {
                let (_, end_slot) = pass_timing::pass_pair(frame_idx, PassId::Composite);
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
        emit_graph_restores(params.cmd, registry, graph);
        // Hand the tables' allocations back for the next frame to refill. An
        // error path above skips this and the next frame allocates fresh.
        self.graph_scratch.set(Some(scratch));
        Ok(())
    }

    // Resolve every graph-driven resource to its barrier target and every aliased
    // transient to the pass that first writes it; the one place that names the
    // backing fields. A driven aliased resource is discarded from its registry
    // resting state; an undriven pooled member from sampled.
    fn refill_graph_tables(
        &self,
        scratch: &mut GraphScratch,
        graph: &CompiledGraph,
        frame_idx: usize,
    ) {
        let GraphScratch {
            registry, alias, ..
        } = scratch;
        registry.refill(graph, |res, arena| {
            let class = res.class()?;
            let (object, resting) = self.barrier_objects_for_label(res.label, frame_idx, arena)?;
            Some(BarrierTarget {
                object,
                class,
                resting,
            })
        });
        let pool = &self.targets.transient_pool;
        alias.refill(graph, |index, res| {
            pool.alias_predecessor(res.label)?;
            let resource = pool.resource_for(res.label)?;
            Some(DxAliasBarrier {
                resource_index: index,
                resource: ResourceRef::new(resource),
                resting: match registry.target(index) {
                    Some(target) => target.resting,
                    None => D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                },
            })
        });
    }

    // The backing resource and resting state of a graph label; `None` when the
    // feature is inactive.
    fn barrier_objects_for_label(
        &self,
        label: &str,
        frame_idx: usize,
        arena: &mut Vec<ResourceRef>,
    ) -> Option<(DxTargetObject, D3D12_RESOURCE_STATES)> {
        const SAMPLED: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        // Every live emitter's particle pool, one graph resource for the whole set,
        // resting in UNORDERED_ACCESS where the simulation writes it.
        // The grass kernels' blades, draw arguments and bend field, resting in
        // UNORDERED_ACCESS.
        if let Some(grass) = &self.grass {
            let resource = match label {
                "grass_blades" => Some(&grass.blades),
                "grass_args" => Some(&grass.args),
                "grass_shadow_blades" => Some(&grass.shadow_blades),
                "grass_shadow_args" => Some(&grass.shadow_args),
                "grass_bend" => Some(&grass.bend_field),
                _ => None,
            };
            if let Some(resource) = resource {
                return Some((
                    DxTargetObject::One(ResourceRef::new(resource)),
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                ));
            }
        }
        if label == "particle_pool" {
            let first = arena.len();
            arena.extend(
                self.particle
                    .gpu
                    .iter()
                    .map(|gpu| ResourceRef::new(&gpu.pool)),
            );
            let count = arena.len() - first;
            return (count > 0).then_some((
                DxTargetObject::Set { first, count },
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ));
        }
        let one = |r: Option<(&ID3D12Resource, D3D12_RESOURCE_STATES)>| {
            r.map(|(resource, resting)| (DxTargetObject::One(ResourceRef::new(resource)), resting))
        };
        one(match label {
            // Indirect commands the cull kernel writes (UAV) and the main pass
            // consumes through `ExecuteIndirect`.
            "draw_args" => frame_slot(
                &self.cull.indirect_cmd_buffers,
                frame_idx,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
            // Late counterpart, written by `CullLate` and consumed by `MainLate`.
            "draw_args_late" => frame_slot(
                &self.cull.indirect_cmd_buffers_late,
                frame_idx,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
            // The early pass writes it and the late pass reads it through the same root UAV,
            // so it never leaves `UNORDERED_ACCESS`.
            "cull_status" => frame_slot(
                &self.cull.cull_status_buffers,
                frame_idx,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
            // Per-cluster light lists and probe masks: the dispatch flips them to UAV and
            // back, so they rest sampled. One buffer, not per-frame.
            "cluster_lists" => Some((&self.light_cull.cluster_buffer, SAMPLED)),
            "ao_output" => self
                .targets
                .transient_pool
                .resource_for("ao_output")
                .map(|r| (r, SAMPLED)),
            // The bloom chain's half-resolution top octave, and the only mip the
            // graph models: the Bloom node writes it, Composite samples it, and
            // the finer octaves in between never leave the node. Pooled, so it
            // resolves through the transient pool exactly like `ao_output`.
            "bloom_top" => self
                .targets
                .transient_pool
                .resource_for("bloom_top")
                .map(|r| (r, SAMPLED)),
            // SSGI's resolve output: the pass owns it across frames, the resolve
            // writes it and the main pass samples it, so it rests sampled.
            "ssgi_indirect" => self.ssgi_indirect_resource().map(|r| (r, SAMPLED)),
            // The cascade array rests sampled: the Shadow producer barrier is the
            // real cross-frame reset for this frame's shadow loop and the Main
            // consumer returns it to sampled. Created sampled, so frame 0's
            // producer starts from the resource's real state.
            "shadow_map" => self
                .shadow
                .resource
                .as_ref()
                .filter(|_| !self.shadow.dsvs.is_empty())
                .map(|s| (&s.resource, SAMPLED)),
            // The spot array rests sampled exactly like the cascades. Only
            // imported when the world has shadowed spots, so the `dsvs` filter
            // matches the graph gate.
            "spot_shadow_map" => self
                .spot_shadow
                .resource
                .as_ref()
                .filter(|_| !self.spot_shadow.dsvs.is_empty())
                .map(|s| (&s.resource, SAMPLED)),
            // The far map rests readable in the stages the graph opens it in.
            "far_shadow_map" => self.shadow.far.as_ref().map(|far| {
                let resting = super::barrier_translate::resting_read_state(FAR_SHADOW_MAP_RESTS);
                (&far.map, resting)
            }),
            // The froxel volume rests sampled: the FogFroxel producer opens it for
            // the compute write and the Fog consumer closes it for the sample.
            "fog_froxel_volume" => self
                .fog
                .resources
                .as_ref()
                .map(|f| (&f.volume_resource, SAMPLED)),
            // Main depth rests as the depth attachment, unlike `shadow_map`.
            "hdr_depth" => Some((
                &self.targets.depth.resource,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
            )),
            // The MSAA color, present only with MSAA. Its transitions are no-ops, but
            // registering it keeps the MSAA resolve's transitions inside the pass.
            "hdr_color" => self
                .targets
                .hdr
                .resolve
                .is_some()
                .then_some((&self.targets.hdr.color, D3D12_RESOURCE_STATE_RENDER_TARGET)),
            // The single-sample scene spine: the resolve target, resting sampled, with
            // MSAA; `hdr.color`, resting in RENDER_TARGET, without.
            "hdr_resolve" => Some(match &self.targets.hdr.resolve {
                Some(resolve) => (resolve, SAMPLED),
                None => (&self.targets.hdr.color, D3D12_RESOURCE_STATE_RENDER_TARGET),
            }),
            // The scene-with-reflections the post stack consumes. Declared by
            // the graph exactly when a reflection resolve runs, which is the
            // same predicate that builds this target, so the entry resolves
            // whenever the resource exists.
            "scene_pre_taa" => self
                .reflection_composite
                .as_ref()
                .map(|rc| (rc.output().resource(), SAMPLED)),
            // The post-TAA scene, driven only when TAA writes it. The upscaler's output has
            // a between-frames state the graph cannot express, so under upscaling it stays
            // unregistered and `post/upscale/fsr.rs` owns its transitions.
            "scene_color" => self
                .taa
                .as_ref()
                .filter(|_| self.upscale.backend.is_none())
                .map(|taa| (&taa.output().resource, SAMPLED)),
            // The G-buffer's color targets, separate graph resources since their consumers
            // differ. All rest sampled.
            "gbuffer_normal_depth" => self.gbuffer.as_ref().map(|gb| (&gb.normal_depth, SAMPLED)),
            "gbuffer_roughness" => self.gbuffer.as_ref().map(|gb| (&gb.roughness, SAMPLED)),
            "gbuffer_velocity" => self.gbuffer.as_ref().map(|gb| (&gb.velocity, SAMPLED)),
            // The reactive mask: in the graph only on a frame that stores and
            // reads it, and sampled between frames.
            "reactive_mask" => Some((&self.targets.reactive_mask.resource, SAMPLED)),
            // `gbuffer_depth` is unregistered: only the upscaler moves it, inside its own
            // dispatch.
            // The Hi-Z pyramid rests in the compute-readable state the cull samples it in.
            "hiz_pyramid" => self.cull.hiz.as_ref().map(|h| (&h.texture, h.rest_state)),
            _ => None,
        })
    }

    // The per-frame `RaymarchView` payload. Its VP is the jittered one the Main
    // pass rasterizes with (`plan.camera.vp`), so raymarched surfaces share their NDC
    // depth space with rasterized geometry and take the same TAA jitter.
    fn build_raymarch_view(&self, params: &GraphFrameParams<'_>) -> super::raymarch::RaymarchView {
        super::raymarch::RaymarchView::new(&self.pass_camera(params, params.plan.camera.vp))
    }

    // The per-frame `TransparentView` payload. Its VP is the jittered one
    // (`plan.camera.vp`) the Main pass rasterized with, so the glass quad's clip-space
    // depth matches the stored main-depth the fragment shader tests against.
    fn build_transparent_view(
        &self,
        params: &GraphFrameParams<'_>,
    ) -> super::transparent::TransparentView {
        super::transparent::TransparentView::new(
            &self.pass_camera(params, params.plan.camera.vp),
            self.state.lights.uniforms(),
            &self.state.view.daylight,
        )
        .with_air(self.transparent_air(params.frame_idx).is_some())
    }

    // The camera a screen-space pass reconstructs from: `vp` at the frame's
    // render size, with this frame's IBL and sky rotation.
    fn pass_camera(&self, params: &GraphFrameParams<'_>, vp: [[f32; 4]; 4]) -> PassCamera {
        PassCamera::new(
            vp,
            mat4_inverse(vp),
            params.plan.cam_pos,
            [params.width as f32, params.height as f32],
            narrow_f32(params.frame.elapsed),
            self.pass_lighting(),
        )
    }

    // The frame's IBL and exposure, the same for every pass camera.
    pub(in crate::directx) fn pass_lighting(&self) -> PassLighting {
        PassLighting {
            prefilter_mip_count: self.scene.env_map.prefilter_mip_count as f32,
            irradiance: self.irradiance_lane(),
            sky_rot: self.state.view.sky_rot,
            pre_exposure: self.pre_exposure.value(),
        }
    }

    // Per-pass dispatch onto the given list. Composite is not routed here.
    fn encode_pass_into(
        &self,
        pass_id: PassId,
        cmd: &ID3D12GraphicsCommandList,
        params: &GraphFrameParams<'_>,
        particle_frame: Option<&ParticleFrame>,
        grass_frame: Option<&super::grass::GrassFrame>,
    ) -> RenderResult<()> {
        match pass_id {
            PassId::Cull => {
                self.encode_cull(
                    cmd,
                    params.frame_idx,
                    &params.plan.camera.frustum,
                    params.plan.cam_pos,
                );
                // Pose the skinned objects' deformed-vertex buffer for this frame
                // (a no-op when no skinned mesh is folded in). Independent of the
                // cull; both feed Main, which the toposort orders after Cull.
                self.encode_skin(cmd, params.frame_idx);
            }
            PassId::SsaoBlur => {
                self.encode_ssao(cmd, params.frame_idx, params.plan.effects.ssao.as_ref())?;
            }
            PassId::Grass => {
                if let Some(frame) = grass_frame {
                    self.encode_grass(cmd, frame);
                }
            }
            PassId::GrassBend => {
                if let Some(frame) = grass_frame {
                    self.encode_grass_bend(cmd, frame);
                }
            }
            PassId::GrassShadow => {
                if let Some(frame) = grass_frame {
                    self.encode_grass_shadow(cmd, frame);
                }
            }
            PassId::LightCull => {
                // Bins lights and probes per cluster; emitted only while either is live, and
                // ordered before Main by `cluster_lists`.
                self.encode_light_cull(
                    cmd,
                    params.frame_idx,
                    self.main_cluster_grid(params.frame_idx),
                );
            }
            PassId::Shadow => {
                // The raymarch view, only when a volume casts shadows; the same view the
                // `Raymarch` pass builds later this frame.
                let raymarch_view = self
                    .raymarch
                    .as_ref()
                    .filter(|rm| rm.any_shadow_casters())
                    .map(|_| self.build_raymarch_view(params));
                self.encode_shadow_pass(
                    cmd,
                    params.frame_idx,
                    params.shadow_ubo_gva,
                    raymarch_view.as_ref(),
                    grass_frame,
                );
            }
            PassId::SpotShadow => {
                // One depth-only render per scheduled spot slice. The builder
                // emits this node only when the world has shadow-casting spots,
                // and the RAW edge on `spot_shadow_map` pins it before Main.
                self.encode_spot_shadow_pass(cmd, params.frame_idx);
            }
            PassId::FarShadow => {
                // One depth-only render of the large casters into the far map,
                // on the frames its schedule redraws it.
                self.encode_far_shadow_pass(cmd, params.frame_idx);
            }
            PassId::AutoExposure => {
                self.encode_auto_exposure(cmd, params.frame_idx);
            }
            PassId::Main => {
                self.encode_main_pass(
                    cmd,
                    params.frame_idx,
                    MainPassExtent {
                        width: params.width,
                        height: params.height,
                    },
                    FrameGpuBuffers {
                        view_gva: params.view_gva,
                        light_gva: params.light_gva,
                        local_lights_gva: params.local_lights_gva,
                        shadow_ubo_gva: params.shadow_ubo_gva,
                    },
                    params.frame.world_hidden,
                    grass_frame,
                );
            }
            PassId::Decals => {
                self.encode_decals(
                    cmd,
                    params.frame_idx,
                    params.plan.camera.vp,
                    &params.plan.camera.frustum,
                );
            }
            PassId::Lines => {
                self.encode_lines(cmd, params.frame_idx, params.plan.camera.vp, params.lines);
            }
            PassId::Fog => {
                self.encode_fog(cmd, params.frame_idx);
            }
            PassId::Atmosphere => self.encode_atmosphere(cmd, params.frame_idx),
            PassId::WaterSim => {
                self.encode_water_sim(cmd, params.frame_idx, params.frame.water_time);
            }
            PassId::AerialPerspective => self.encode_aerial_perspective(cmd, params.frame_idx),
            PassId::Clouds => self.encode_atmosphere_clouds(cmd, params.frame_idx),
            PassId::CloudComposite => {
                self.encode_atmosphere_cloud_composite(cmd, params.frame_idx);
            }
            PassId::ParticlesSim => {
                // Integrates each live emitter's pool; the particle state was advanced before
                // the fan-out.
                if let Some(frame) = particle_frame {
                    self.encode_particles_sim(cmd, params.frame_idx, *frame);
                }
            }
            PassId::ParticlesDraw | PassId::ParticlesLateDraw => {
                let late = pass_id == PassId::ParticlesLateDraw;
                let reactive = if late {
                    params.plan.reactive.late_particles
                } else {
                    params.plan.reactive.particles
                };
                self.targets.reactive_mask.clear(cmd, reactive);
                if let Some(frame) = particle_frame {
                    self.encode_particles_draw(
                        cmd,
                        params.frame_idx,
                        frame,
                        &params.plan.camera.frustum,
                        super::particle::ParticleStage { reactive, late },
                    );
                }
            }
            PassId::SsrResolve => {
                self.encode_ssr_resolve(
                    cmd,
                    params.frame_idx,
                    params.plan.effects.ssr.as_ref(),
                    &params.plan.effects.reflection_env,
                );
            }
            PassId::TaaResolve => {
                self.encode_taa(cmd, params.frame_idx, params.plan.reactive.readable);
            }
            PassId::Bloom => {
                self.encode_bloom(cmd, params.frame_idx, params.scene_srv)?;
            }
            PassId::Composite => {
                // Composite is recorded by `execute_graph` itself, never through the fan-out.
                return Err(RenderError::Other(
                    "graph executor (directx): Composite must run on the outer cmd \
                     list: encode_pass_into is not the right entry point"
                        .to_string(),
                ));
            }
            PassId::Raymarch => {
                let view = self.build_raymarch_view(params);
                self.encode_raymarch(cmd, params.frame_idx, &view, &params.plan.camera.frustum)?;
            }
            PassId::FogFroxel => {
                let blocks = params.plan.fog.as_ref().ok_or_else(|| {
                    RenderError::Other(
                        "graph executor (directx): FogFroxel without fog blocks".into(),
                    )
                })?;
                self.encode_fog_froxel(cmd, params.frame_idx, blocks, params.shadow_ubo_gva);
            }
            PassId::Upscale => {
                // The temporal upscaler, reading the decorated scene, velocity and depth into
                // the output bloom and composite sample.
                self.encode_upscale(cmd, params)?;
            }
            PassId::PropProbe => self.encode_prop_probe(cmd, params)?,
            PassId::PlanarReflection => {
                // Mirror renders for the flat reflectors in view, each cropped to
                // the screen rectangle its reflectors cover (see `planar.rs`).
                // Recorded ahead of `Transparent` in the frame's submission, each
                // plane's resolve left shader-readable for it.
                self.encode_planar_reflections(cmd, params);
            }
            PassId::Transparent => {
                // Glass panes and water surfaces back-to-front over the post-SSR scene.
                self.targets
                    .reactive_mask
                    .clear(cmd, params.plan.reactive.transparent);
                let view = self.build_transparent_view(params);
                self.advance_transparent_water(
                    params.frame_idx,
                    params.frame.water_time,
                    params.plan.cam_pos,
                );
                self.encode_transparent(
                    cmd,
                    params.frame_idx,
                    &view,
                    params.plan.effects.rt.as_ref(),
                    params.plan.reactive.transparent,
                )?;
            }
            PassId::HizEarly | PassId::HizFinal => {
                // `HizEarly` rebuilds the pyramid from early depth for `CullLate`; `HizFinal`
                // reduces the final depth for next frame's early cull.
                self.encode_hiz_build(cmd);
            }
            PassId::ShadowReceivers => self.encode_shadow_receivers(cmd, params.frame_idx),
            PassId::ViewSamples => self.encode_view_samples(cmd, params.frame_idx),
            PassId::CullLate => {
                self.encode_cull_late(
                    cmd,
                    params.frame_idx,
                    &params.plan.camera.frustum,
                    params.plan.camera.cur_vp,
                );
            }
            PassId::MainLate => {
                self.encode_main_pass_late(
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
                self.encode_ssgi(cmd, params.frame_idx, params.plan.effects.ssgi.as_ref());
            }
            PassId::SsgiResolve => {
                self.encode_ssgi_resolve(cmd, params.frame_idx, params.plan.effects.ssgi.as_ref());
            }
            PassId::Disocclusion => self.encode_disocclusion(cmd, params.frame_idx)?,
            PassId::RtReflections => {
                // Hardware ray-traced reflections in the `SsrResolve` slot. The TLAS update
                // already ran on the "start" list.
                self.encode_rt_reflections(
                    cmd,
                    params.frame_idx,
                    params.plan.effects.rt.as_ref(),
                    &params.plan.effects.reflection_env,
                    params.plan.effects.reflection_denoise.as_ref(),
                );
            }
            PassId::GBufferPrepass | PassId::GBufferLate => {
                if pass_id == PassId::GBufferLate {
                    self.encode_gbuffer_late(cmd, params.frame_idx, &params.plan.gbuffer_view);
                } else {
                    // The unified geometry pre-pass: normal+depth, roughness and motion for
                    // every screen-space consumer, rasterized with the jittered VP. Motion is
                    // real only when a consumer reads it.
                    self.encode_gbuffer_prepass(
                        cmd,
                        params.frame_idx,
                        crate::directx::post::gbuffer::GbufferPrepassView {
                            jittered_vp: params.plan.camera.vp,
                            elapsed: narrow_f32(params.frame.elapsed),
                            cam_pos: params.plan.cam_pos,
                            frustum: &params.plan.camera.frustum,
                        },
                        crate::directx::post::gbuffer::GbufferPrepassFrame {
                            block: &params.plan.gbuffer_view,
                            grass: grass_frame,
                        },
                    );
                }
                if model_history_pass(&params.plan.graph) == Some(pass_id) {
                    self.encode_model_history(cmd, params.frame_idx, params.prime_model_history);
                }
            }
        }
        Ok(())
    }
}
