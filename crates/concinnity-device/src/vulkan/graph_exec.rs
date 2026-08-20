// src/vulkan/graph_exec.rs
//
// Vulkan-side executor for the render graph. `VkContext::execute_graph`
// walks the `CompiledGraph` produced by the shared
// [`gfx::render_graph::build_frame_graph`](../gfx/render_graph/frame.rs)
// and dispatches each pass to its `encode_*` method. Mirrors the Metal
// + DirectX executors: every backend now drives the same builder.
//
// The catch-all arm at the bottom returns a clear error if any not-yet-ported
// `PassId` slips into the compiled graph.
//
// Per-pass `barriers_before` is consumed for every resource the executor's
// barrier registry resolves: `emit_graph_barriers` translates their graph state
// transitions into explicit `vkCmdPipelineBarrier` calls at the start of each
// pass's command buffer, and `emit_graph_restores` returns any that the frame
// left off their resting layout at the end of the outer "end" buffer. A resource
// with no registry entry keeps whatever transitions its encoder or render pass
// owns; `barrier_audit.rs` classifies every one of those remaining sites.
//
// The registry decides two things per resource: which GPU object backs it, and
// what layout it rests in between frames. Its class -- what a `Write` means --
// comes from the usage the graph declares, so this executor and the DirectX one
// cannot disagree about it. Resting cannot: `shadow_map` and `hdr_depth` are both
// depth targets, and the first rests sampled (its staggered cascades keep the
// depth they were last rendered with) while the second discards.
//
// Bundled passes:
//   * `PassId::SsaoBlur` dispatches the bundled `encode_ssao` (GTAO
//     kernel + depth-aware blur over the unified pre-pass normal+depth).
//     `PassId::SsaoPrepass` / `PassId::SsaoKernel` stay timing-only and
//     the executor rejects them as graph nodes.

use ash::vk;

use crate::gfx::frustum::Frustum;
use crate::gfx::render_graph::{
    CompiledGraph, CompiledPass, GraphResourceClass, PassId, final_states,
};
use crate::gfx::render_types::{LineVertex, TextDrawCall};

use super::barrier_translate::{VkResting, vk_restore, vk_transition};
use super::context::VkContext;
use super::parallel_encoder::ParallelCtxRef;
use super::post::gbuffer::GbufferPrepassView;

// The GPU object a graph resource backs: an image with the subresource extent a
// barrier must cover (the cascade count for the CSM `shadow_map`, the mip count
// for the Hi-Z pyramid), or a buffer.
#[derive(Copy, Clone)]
enum VkTargetObject {
    Image {
        image: vk::Image,
        mip_levels: u32,
        layer_count: u32,
    },
    Buffer {
        buffer: vk::Buffer,
    },
}

// One resolved barrier target: the object a graph resource backs, its class, and
// the layout it sits in between frames. Built once per frame by
// `build_barrier_registry`. `vk::Image` / `vk::Buffer` are plain `Send + Sync`
// handles, so the registry shares into the parallel pass workers with no wrapper
// (unlike the DirectX side's COM handles).
struct VkBarrierTarget {
    object: VkTargetObject,
    class: GraphResourceClass,
    resting: VkResting,
}

// `ResourceId`-indexed table of barrier targets for the migrated graph resources
// (`None` for every resource the executor doesn't graph-drive). A resource is
// graph-driven iff it has a `Some` entry, so this table is the single source of
// truth that replaced the old label allowlist + per-label resolver. Built on the
// main thread by `build_barrier_registry`, where the only field-naming of the
// migrated resources lives; the parallel emit path stays field-agnostic.
struct VkBarrierRegistry(Vec<Option<VkBarrierTarget>>);

// Emit the explicit image-layout transitions for the migrated graph resources
// from a pass's `barriers_before`, resolved through the registry. Called at the
// start of each pass's own command buffer, before the pass encodes, so the
// transition lands ahead of the pass's render pass in the same submission. A
// resource with no registry entry is skipped and keeps its render-pass-driven
// transition; a transition whose layout does not change (e.g. the depth
// producer's no-op Undefined -> Write) is skipped too. Takes `&ash::Device`, not
// `&VkContext`: the field-to-image mapping was already resolved into the
// registry, so this parallel path is field-agnostic.
fn emit_graph_barriers(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    registry: &VkBarrierRegistry,
    pass: &CompiledPass,
) {
    for op in &pass.barriers_before {
        let Some(Some(target)) = registry.0.get(op.resource_index()) else {
            continue;
        };
        let Some(transition) = vk_transition(
            target.class,
            target.resting,
            op.source_state(),
            op.to_state(),
            op.read_stages(),
        ) else {
            continue;
        };
        emit_one(device, cmd, target, transition);
    }
}

// Record one resolved transition against a target's GPU object: a buffer memory
// barrier over the whole range, or an image memory barrier over every mip and
// array layer the target declares.
fn emit_one(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    target: &VkBarrierTarget,
    transition: (
        vk::ImageLayout,
        vk::ImageLayout,
        vk::AccessFlags,
        vk::AccessFlags,
        vk::PipelineStageFlags,
        vk::PipelineStageFlags,
    ),
) {
    let (old_layout, new_layout, src_access, dst_access, src_stage, dst_stage) = transition;
    match target.object {
        VkTargetObject::Buffer { buffer } => {
            // A buffer has no layout; the whole barrier is the access + stage
            // dependency, over the whole range.
            let barrier = vk::BufferMemoryBarrier::default()
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE)
                .src_access_mask(src_access)
                .dst_access_mask(dst_access);
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                device.cmd_pipeline_barrier(
                    cmd,
                    src_stage,
                    dst_stage,
                    vk::DependencyFlags::empty(),
                    &[],
                    std::slice::from_ref(&barrier),
                    &[],
                );
            }
        }
        VkTargetObject::Image {
            image,
            mip_levels,
            layer_count,
        } => {
            let aspect = match target.class {
                GraphResourceClass::DepthTarget => vk::ImageAspectFlags::DEPTH,
                _ => vk::ImageAspectFlags::COLOR,
            };
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(old_layout)
                .new_layout(new_layout)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count,
                })
                .src_access_mask(src_access)
                .dst_access_mask(dst_access);
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                device.cmd_pipeline_barrier(
                    cmd,
                    src_stage,
                    dst_stage,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    std::slice::from_ref(&barrier),
                );
            }
        }
    }
}

// Return every driven resource the frame left off its resting layout, so the next
// frame's first transition for it opens from the layout the image is really in.
// Recorded into the frame's outer "end" command buffer, which is submitted after
// every pass buffer, so these run last. A frame that ends every resource at rest
// (the common case: nothing needs one) emits nothing.
fn emit_graph_restores(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    registry: &VkBarrierRegistry,
    graph: &CompiledGraph,
) {
    for (idx, (state, stages)) in final_states(graph).into_iter().enumerate() {
        let Some(Some(target)) = registry.0.get(idx) else {
            continue;
        };
        let Some(transition) = vk_restore(target.class, target.resting, state, stages) else {
            continue;
        };
        emit_one(device, cmd, target, transition);
    }
}

// Emit the aliasing barriers for a pass: for each pooled transient this pass
// first-writes whose memory is reused from an earlier transient in the same slot
// (`images`), order that earlier resource's prior use before this write. The
// members are colour targets, so the dependency is the colour/fragment domain:
// the predecessor's last use is either a fragment-shader sample (e.g.
// `ao_output` read by Main) or a colour write, and this member's first use is a
// colour write (e.g. the bloom prefilter). `UNDEFINED -> COLOR_ATTACHMENT`
// discards the predecessor's contents in the shared memory (the member is fully
// rewritten before it is read). Per-resource stage derivation can refine this
// when a non-colour member is aliased.
fn emit_alias_barriers(device: &ash::Device, cmd: vk::CommandBuffer, images: &[vk::Image]) {
    for &image in images {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier),
            );
        }
    }
}

// Everything a pass owes its command buffer before its body: the aliasing
// barriers for any pooled transient it first-writes, then its graph-derived
// transitions.
//
// That order matches DirectX, and it is the order the two can be combined in: a
// resource that is both aliased and graph-driven needs its slot claimed before
// the derived transition that opens it for writing, since the aliasing barrier
// is what makes the memory legally its own. No resource is both today (here
// `ao_output` is driven but sits alone in its slot, while `bloom_top` aliases
// but has no registry entry), so keeping the orders identical across backends
// is what stops that from being discovered one backend at a time.
//
// One function because the two recording paths are otherwise asymmetric --
// Composite records into the outer "end" buffer on the main thread while every
// other pass fans out to a worker -- and that asymmetry is exactly how the
// DirectX executor came to record no graph barriers for Composite at all,
// silently dropping any a driven resource declared there.
fn emit_pass_prologue(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    registry: &VkBarrierRegistry,
    alias: &[vk::Image],
    pass: &CompiledPass,
) {
    emit_alias_barriers(device, cmd, alias);
    emit_graph_barriers(device, cmd, registry, pass);
}

// Check the graph's barrier coverage and cross-frame layout contract for every
// resource this executor drives, on the frame's real compiled graph. Two
// invariants, both cheap enough to run per frame under `debug_assertions`:
//
//   * every declared read / write of a driven resource is preceded by a
//     transition putting it in the matching state, in the consuming stage;
//   * a driven resource is back in its resting layout once the frame's restores
//     have run, so the next frame's producer barrier names a source layout the
//     image is really in. This is the check the restore pass exists to satisfy;
//     it fires if a resource ends in a state no restore can express.
//
// This is where a registry entry that claims a resource the graph does not fully
// cover shows up; the headless sweep in `render_graph::validate` covers the
// deriver itself.
#[cfg(debug_assertions)]
fn debug_assert_graph_drives(graph: &CompiledGraph, registry: &VkBarrierRegistry) {
    use super::barrier_translate::vk_state;
    use crate::gfx::render_graph::{ResourceState, barrier_coverage_gaps_for_driven};

    let driven: Vec<bool> = registry.0.iter().map(|t| t.is_some()).collect();
    let gaps = barrier_coverage_gaps_for_driven(graph, &driven);
    assert!(
        gaps.is_empty(),
        "render graph (vulkan): uncovered accesses on graph-driven resources: {}",
        gaps.iter()
            .map(|g| g.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );

    for (idx, (state, stages)) in final_states(graph).into_iter().enumerate() {
        let Some(Some(target)) = registry.0.get(idx) else {
            continue;
        };
        // Buffers have no layout, and a discard-resting resource's next first use
        // is legal from whatever layout the frame leaves.
        if target.class.is_buffer()
            || target.resting == VkResting::Discarded
            || state == ResourceState::Undefined
        {
            continue;
        }
        let restored = match vk_restore(target.class, target.resting, state, stages) {
            Some((_, new, ..)) => new,
            None => vk_state(target.class, state, stages).0,
        };
        assert_eq!(
            restored,
            target.resting.layout(),
            "render graph (vulkan): {} rests in {:?} but the frame leaves it in {:?}",
            graph.resources[idx].label,
            target.resting.layout(),
            restored,
        );
    }
}

// Per-frame params the executor threads into each pass's `encode_*`
// method. The set grows as more passes migrate; Composite needs the
// swapchain image index + text calls, Shadow needs neither (it reads
// per-frame state straight off `&self`), Main needs the BVH-culled
// visible set + frustum + camera position. New fields land here when
// a pass that needs them migrates.
pub(in crate::vulkan) struct GraphFrameParams<'a> {
    pub cmd: vk::CommandBuffer,
    pub image_index: u32,
    pub frame_idx: usize,
    pub text_calls: &'a [TextDrawCall],
    // This frame's expanded line ribbons, consumed by the Lines pass. Empty
    // whenever nothing published lines, in which case the graph carries no
    // Lines node either.
    pub lines: &'a [LineVertex],
    // An opaque menu backdrop hides the scene: the Main pass clears its target
    // and skips every draw (the masked graph drops all other world passes), so
    // nothing of the world renders behind the menu.
    pub world_hidden: bool,
    // CPU visibility list (BVH-culled cullables + always_draw fallback).
    // Consumed by Main's legacy + instanced fallback passes and the
    // unified G-buffer pre-pass.
    pub visible: &'a [u32],
    // Camera frustum used to cull instanced clusters during Main's
    // per-cluster draw loop, and by the G-buffer pre-pass for the same
    // reason.
    pub frustum: &'a Frustum,
    // Camera world-space position used for per-cluster distance-cull
    // during Main's instanced sub-pass and the G-buffer pre-pass.
    pub cam_pos: [f32; 3],
    // Jittered view-projection matrix (with TAA Halton jitter when
    // TAA is on). Consumed by the G-buffer pre-pass to rasterise the
    // normal+depth / roughness / velocity MRT.
    pub vp_mat: [[f32; 4]; 4],
    // Un-jittered current-frame view-projection matrix. The G-buffer
    // pre-pass uses it (alongside `vp_mat` for the jittered VP and the
    // prior frame's `prev_view_proj` stored on `GbufferResources`) so the
    // stored motion vector is free of sub-pixel jitter. `Default::default()`
    // for frames where the velocity channel isn't dispatched.
    pub cur_vp: [[f32; 4]; 4],
    // Vertical FOV in radians: SSAO needs it for the projection
    // reconstruction used by the GTAO horizon search; SSR resolve
    // uses it for the same.
    pub fov_y_radians: f32,
    // Camera aspect ratio (width / height); same SSAO + SSR use
    // as `fov_y_radians`.
    pub aspect: f32,
    // Frame-global elapsed seconds. The particle encoder needs it to
    // derive `dt` from the last-frame snapshot it stashed in a `Cell`;
    // the compute kernel multiplies dt against `spawn_rate` and the
    // integration step.
    pub elapsed: f32,
    // Camera near-plane in view units. The FogFroxel kernel needs it to map
    // each Z slab onto the linear-Z `[near, max_distance]` volume range.
    pub near: f32,
    // Camera far-plane in view units. The temporal-upscale dispatch (FSR;
    // DLSS / XeSS ignore it) needs the near + far + FOV to linearise depth for
    // its reprojection.
    pub far: f32,
}

impl VkContext {
    // Walk a compiled render graph and record each pass. Every non-composite
    // pass is recorded into its own per-`(frame, pass)` primary command buffer
    // (parallel command-buffer recording); Composite is recorded into the
    // frame's outer "end" buffer (`params.cmd`) on the main thread because it
    // writes the swapchain image + allocates transient text buffers. Returns
    // the per-pass buffers in graph (toposort) order; the caller submits
    // `[start, ...returned, end]` in one `vkQueueSubmit`, so submission order =
    // GPU order and every encoder's inline barrier still synchronises against
    // the prior pass across the command-buffer boundary. Any not-yet-migrated
    // `PassId` returns a clear error.
    pub(in crate::vulkan) fn execute_graph(
        &mut self,
        graph: &CompiledGraph,
        params: &GraphFrameParams<'_>,
    ) -> Result<Vec<vk::CommandBuffer>, String> {
        // Particle per-frame state (dt / frame index / per-emitter spawn
        // budgets) is advanced here on `&mut self` before any pass encodes, so
        // the `&self` `encode_particles` (which may run on a parallel-recording
        // worker) never mutates the particle `Cell`s. `None` when the pass is
        // inert. Mirrors Metal's `prepare_particle_pass` hoist.
        let particle_frame = self.prepare_particle_pass(params.elapsed);

        // Instanced clusters: recompute the per-cluster LOD-bucket partition
        // and upload the bucket-ordered instance matrices on `&mut self` before
        // the fan-out, so every instanced pass (Main + the unified G-buffer
        // pre-pass + Shadow) reads a consistent partition while recording on
        // worker threads. Inert when no clusters are declared.
        self.prepare_instanced_clusters(params.frame_idx, params.cam_pos);

        // Composite stays on the main thread (it writes the swapchain image
        // and allocates + drops transient text buffers through the RefCell
        // device allocator); every other pass fans onto a `jobs::pool()`
        // worker that records into its own `(frame, pass)` command buffer.
        let composite_idx = graph.passes.iter().position(|p| p.id == PassId::Composite);
        let frame_idx = params.frame_idx;
        let device = self.device.clone();

        // One output slot per graph pass index; each worker stores its finished
        // command buffer at its own index. Disjoint indices, but a `Mutex`
        // keeps the store sound + simple (it's once per pass). `first_error`
        // captures the first worker failure.
        let worker_slots: std::sync::Mutex<Vec<Option<vk::CommandBuffer>>> =
            std::sync::Mutex::new(vec![None; graph.passes.len()]);
        let first_error: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

        // Resolve every migrated resource's barrier target once, on the main
        // thread, then share the table read-only into the parallel pass workers.
        let registry = self.build_barrier_registry(graph, frame_idx);
        #[cfg(debug_assertions)]
        debug_assert_graph_drives(graph, &registry);
        #[cfg(debug_assertions)]
        crate::gfx::render_graph::assert_slot_aliasing_sound(
            graph,
            self.transient_pool.slot_labels(),
            "vulkan",
        );
        // Per-pass aliasing barriers for the pooled transients that share memory
        // this frame (e.g. `bloom_top` reusing `ao_output`'s slot). Empty when no
        // slot is shared.
        let alias_barriers = self.build_alias_barriers(graph, frame_idx);
        let ctx_ref = ParallelCtxRef::new(self);
        let particle_ref = particle_frame.as_ref();
        let device_ref = &device;
        let worker_slots_ref = &worker_slots;
        let first_error_ref = &first_error;
        let registry_ref = &registry;
        let alias_barriers_ref = &alias_barriers;

        crate::jobs::pool().install(|| {
            rayon::scope(|scope| {
                for (idx, pass) in graph.passes.iter().enumerate() {
                    if Some(idx) == composite_idx {
                        continue;
                    }
                    let pass_id = pass.id;
                    scope.spawn(move |_| {
                        let ctx = ctx_ref.as_ctx();
                        let pool_idx =
                            frame_idx * crate::gfx::render_graph::PASS_COUNT + pass_id as usize;
                        let buf = ctx.commands.pass_command_buffers[pool_idx];
                        let set_err = |msg: String| {
                            let mut lock = first_error_ref.lock().unwrap();
                            if lock.is_none() {
                                *lock = Some(msg);
                            }
                        };
                        // Reset + begin this pass's own buffer (its own pool, so
                        // no cross-worker pool contention), encode, end.
                        // SAFETY: `cmd` belongs to this frame slot, whose fence was already waited
                        // on, so it is not in flight; reset then begin puts it in the recording
                        // state, which is what the subsequent recording requires.
                        let begin = unsafe {
                            device_ref
                                .reset_command_buffer(buf, vk::CommandBufferResetFlags::empty())
                                .and_then(|()| {
                                    device_ref.begin_command_buffer(
                                        buf,
                                        &vk::CommandBufferBeginInfo::default()
                                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                                    )
                                })
                        };
                        if let Err(e) = begin {
                            set_err(format!("begin pass cmd buf ({}): {e}", pass_id.name()));
                            return;
                        }
                        // Per-pass GPU timing: bracket this pass's encode with a
                        // (start, end) timestamp pair in its own buffer. The block
                        // was reset in the start buffer (submitted first), so these
                        // writes are valid. A pass absent from a later frame's graph
                        // leaves its slots unwritten; the readback's
                        // `WITH_AVAILABILITY` reports those as 0.
                        if let Some(pool) = ctx.timestamp_query_pool {
                            let (ts_start, _) = super::pass_timing::pass_pair(frame_idx, pass_id);
                            // SAFETY: `cmd` is a command buffer in the recording state, and every
                            // handle and slice these commands name is live for the call.
                            unsafe {
                                device_ref.cmd_write_timestamp(
                                    buf,
                                    vk::PipelineStageFlags::TOP_OF_PIPE,
                                    pool,
                                    ts_start,
                                );
                            }
                        }
                        emit_pass_prologue(
                            device_ref,
                            buf,
                            registry_ref,
                            &alias_barriers_ref[idx],
                            pass,
                        );
                        if let Err(e) = ctx.encode_pass_into(pass_id, buf, params, particle_ref) {
                            set_err(e);
                            return;
                        }
                        if let Some(pool) = ctx.timestamp_query_pool {
                            let (_, ts_end) = super::pass_timing::pass_pair(frame_idx, pass_id);
                            // SAFETY: `cmd` is a command buffer in the recording state, and every
                            // handle and slice these commands name is live for the call.
                            unsafe {
                                device_ref.cmd_write_timestamp(
                                    buf,
                                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                                    pool,
                                    ts_end,
                                );
                            }
                        }
                        // SAFETY: `cmd` is in the recording state, which is what
                        // `end_command_buffer` requires.
                        if let Err(e) = unsafe { device_ref.end_command_buffer(buf) } {
                            set_err(format!("end pass cmd buf ({}): {e}", pass_id.name()));
                            return;
                        }
                        worker_slots_ref.lock().unwrap()[idx] = Some(buf);
                    });
                }
            });
        });

        if let Some(e) = first_error.into_inner().unwrap() {
            return Err(e);
        }

        // Composite on the main thread, into the outer "end" buffer. Bracket it
        // with its own per-pass timestamp pair (in the end buffer, which also
        // carries the whole-frame end timestamp written later in `record_frame`).
        if let Some(idx) = composite_idx {
            if let Some(pool) = self.timestamp_query_pool {
                let (ts_start, _) = super::pass_timing::pass_pair(frame_idx, PassId::Composite);
                // SAFETY: `cmd` is a command buffer in the recording state, and every handle and
                // slice these commands name is live for the call.
                unsafe {
                    self.device.cmd_write_timestamp(
                        params.cmd,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        pool,
                        ts_start,
                    );
                }
            }
            emit_pass_prologue(
                &self.device,
                params.cmd,
                &registry,
                &alias_barriers[idx],
                &graph.passes[idx],
            );
            self.encode_pass_into(
                PassId::Composite,
                params.cmd,
                params,
                particle_frame.as_ref(),
            )?;
            if let Some(pool) = self.timestamp_query_pool {
                let (_, ts_end) = super::pass_timing::pass_pair(frame_idx, PassId::Composite);
                // SAFETY: `cmd` is a command buffer in the recording state, and every handle and
                // slice these commands name is live for the call.
                unsafe {
                    self.device.cmd_write_timestamp(
                        params.cmd,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        pool,
                        ts_end,
                    );
                }
            }
        }

        // Return every driven resource the frame left off its resting layout.
        // Recorded last into the outer "end" buffer, which is submitted after
        // every pass buffer.
        emit_graph_restores(&self.device, params.cmd, &registry, graph);

        // Collect the per-pass buffers in ascending graph index = toposort
        // order (the `None` Composite slot is skipped). Never sort: the submit
        // array order must equal toposort order for the inline barriers to
        // synchronise correctly across buffer boundaries.
        let ordered: Vec<vk::CommandBuffer> = worker_slots
            .into_inner()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        Ok(ordered)
    }

    // Resolve every migrated graph resource to its barrier target, indexed by
    // `ResourceId` (its position in `graph.resources`), so the parallel emit path
    // can look a target up by `BarrierOp::resource_index()`. This is the single
    // place that names the migrated resources' backing `VkContext` fields;
    // field-grouping re-cuts here, not in the executor. A resource the owning
    // feature disabled (or one never migrated) gets `None`, and the graph carries
    // no barrier for it either.
    fn build_barrier_registry(&self, graph: &CompiledGraph, frame_idx: usize) -> VkBarrierRegistry {
        VkBarrierRegistry(
            graph
                .resources
                .iter()
                .map(|res| {
                    let class = res.class()?;
                    let (object, resting) = self.barrier_object_for_label(res.label, frame_idx)?;
                    Some(VkBarrierTarget {
                        object,
                        class,
                        resting,
                    })
                })
                .collect(),
        )
    }

    // Build the per-pass aliasing-barrier table for this frame: `table[i]` holds
    // the pooled images to alias-barrier at the start of graph pass `i`. A pooled
    // transient that reuses an earlier transient's memory (a slot predecessor in
    // the pool) needs the predecessor's prior use ordered before its first write;
    // the barrier lands before the pass that first writes it (`lifetime.first`).
    // Empty for every resource the pool does not alias (no predecessor), so the
    // table is empty whenever no slot is shared this frame.
    fn build_alias_barriers(&self, graph: &CompiledGraph, frame_idx: usize) -> Vec<Vec<vk::Image>> {
        let mut table = vec![Vec::new(); graph.passes.len()];
        for res in &graph.resources {
            if self.transient_pool.alias_predecessor(res.label).is_none() {
                continue;
            }
            if let Some(image) = self.transient_pool.image_for(res.label, frame_idx) {
                let first = res.lifetime.first;
                if first < table.len() {
                    table[first].push(image);
                }
            }
        }
        table
    }

    // Map one graph resource label to its backing GPU object and the layout it
    // sits in between frames. `frame_idx` selects the frame-in-flight copy for the
    // per-frame targets (`ao_output`, main depth) and the per-frame cull buffers.
    // The resource's barrier class is NOT decided here: it follows the usage the
    // graph declares (`CompiledResource::class`), so this backend and DirectX
    // cannot disagree about what a resource is. Resting does stay here, because it
    // is genuinely per-resource and per-backend: `shadow_map` and `hdr_depth` are
    // both depth targets and rest differently. `None` means the owning feature is
    // inactive, and the graph carries no node for it either.
    fn barrier_object_for_label(
        &self,
        label: &str,
        frame_idx: usize,
    ) -> Option<(VkTargetObject, VkResting)> {
        // A per-frame buffer resolves through its frame slot. Buffers have no
        // layout, so their resting is immaterial.
        let buffer = |slots: &[super::allocator::PooledBuffer]| {
            slots.get(frame_idx).map(|b| {
                (
                    VkTargetObject::Buffer { buffer: b.buffer() },
                    VkResting::Discarded,
                )
            })
        };
        let image = |image, mip_levels, layer_count, resting| {
            (
                VkTargetObject::Image {
                    image,
                    mip_levels,
                    layer_count,
                },
                resting,
            )
        };
        match label {
            // The indirect-draw commands the cull kernel writes and the main pass
            // consumes through `cmd_draw_indexed_indirect`.
            "draw_args" => buffer(&self.cull.indirect_buffers),
            // Phase-2 counterpart, written by `Cull2` and consumed by `Main2`.
            "draw_args2" => buffer(&self.cull.indirect_buffers2),
            // Per-object cull status: phase 1 writes it, phase 2 reads it.
            "cull_status" => buffer(&self.cull.cull_status_buffers),
            // Per-cluster light index lists: `LightCull` writes them, the main
            // pass's fragment shader reads them. One buffer, not per-frame.
            "cluster_light_list" => Some((
                VkTargetObject::Buffer {
                    buffer: self.light_cull.cluster_buffer.buffer(),
                },
                VkResting::Discarded,
            )),
            // A pooled transient: fully rewritten each frame, so its first use
            // discards whatever the pool's previous tenant left.
            "ao_output" => self
                .transient_pool
                .image_for("ao_output", frame_idx)
                .map(|i| image(i, 1, 1, VkResting::Discarded)),
            // The cascade array; one layer per cascade. Rests sampled: under the
            // hybrid schedule a skipped slice keeps the depth it was last rendered
            // with, so the producer must not discard.
            "shadow_map" if !self.shadow.framebuffers.is_empty() => Some(image(
                self.shadow.map.image,
                1,
                self.shadow.framebuffers.len() as u32,
                VkResting::Sampled,
            )),
            // The spot array; one layer per shadowed spot. Only imported when the
            // world has shadowed spots, so the framebuffer guard matches the gate.
            // Rests sampled for the same staggered-slice reason.
            "spot_shadow_map" if !self.spot_shadow.framebuffers.is_empty() => Some(image(
                self.spot_shadow.map.image,
                1,
                self.spot_shadow.framebuffers.len() as u32,
                VkResting::Sampled,
            )),
            // The volumetric-fog scatter volume: one array layer (the 3D volume).
            "fog_froxel_volume" => self
                .fog_resources
                .as_ref()
                .map(|f| image(f.volume.image(), 1, 1, VkResting::Sampled)),
            // Main depth, one image per frame in flight. Rests discarded: the main
            // pass clears it and its render pass declares an UNDEFINED initial
            // layout, so nothing survives the frame boundary.
            "hdr_depth" => self
                .depth_images
                .get(frame_idx)
                .map(|d| image(d.image, 1, 1, VkResting::Discarded)),
            // The Hi-Z pyramid, barriered over its whole mip chain. Rests sampled:
            // the *next* frame's cull reads what this frame's terminal build wrote.
            "hiz_pyramid" => self
                .cull
                .hiz
                .as_ref()
                .map(|h| image(h.pyramid.image(), h.mip_count, 1, VkResting::Sampled)),
            // Everything else the graph models is left to its encoder, and the
            // scene spine is the bulk of it: `hdr_color`, `hdr_resolve`,
            // `scene_pre_taa`, `scene_color`, `bloom_top` and the three
            // `gbuffer_*` channels. Each is a render pass attachment whose
            // initial / final layouts move it between the attachment and sampled
            // layouts, and whose external subpass dependencies order it against
            // the neighbouring passes -- a form the driver can fold into the
            // pass, which a standalone barrier is not. So the ops the graph
            // derives for them are deliberately dropped here rather than
            // pending. Adding an entry above means taking the layout and the
            // dependencies out of that resource's render pass in the same
            // change, or it is transitioned twice.
            _ => None,
        }
    }

    // Record a single render-graph pass into `cmd`. Shared by the (current)
    // serial driver and the parallel fan-out: takes `&self` so it can run on a
    // worker thread. `particle_frame` is the precomputed per-frame particle
    // state from `prepare_particle_pass` (the only pass needing pre-advanced
    // state).
    pub(in crate::vulkan) fn encode_pass_into(
        &self,
        pass_id: PassId,
        cmd: vk::CommandBuffer,
        params: &GraphFrameParams<'_>,
        particle_frame: Option<&(f32, u32, Vec<u32>)>,
    ) -> Result<(), String> {
        match pass_id {
            PassId::Cull => {
                self.encode_cull(cmd, params.frame_idx, params.frustum, params.cam_pos);
                // Pose the skinned objects' deformed-vertex buffer for this frame
                // (a no-op when no skinned mesh is folded in). Independent of the
                // cull; both feed Main, which the toposort orders after Cull.
                self.encode_skin(cmd, params.frame_idx);
            }
            PassId::LightCull => {
                // Bins the local lights into per-cluster index lists. The builder
                // emits this node only when the world has local lights (matching
                // `clustered_lighting_enabled`), and the RAW edge on
                // `cluster_light_list` pins it before Main, which reads the same
                // buffer.
                self.encode_light_cull(cmd, params.frame_idx);
            }
            PassId::SsaoBlur => {
                // The single graph node for the bundled `encode_ssao`
                // dispatch: encodes the GTAO kernel + depth-aware blur over the
                // unified pre-pass's normal+depth. The SsaoPrepass / SsaoKernel
                // PassIds stay timing-only (rejected as graph nodes below) like
                // Metal's same pattern.
                self.encode_ssao(cmd, params.frame_idx, params.fov_y_radians, params.aspect);
            }
            PassId::SsaoPrepass | PassId::SsaoKernel => {
                return Err(format!(
                    "graph executor (vulkan): pass {} is bundled inside SsaoBlur \
                     (encode_ssao encodes the SSAO kernel + blur sub-passes); it \
                     should not appear as its own graph node",
                    pass_id.name()
                ));
            }
            PassId::ReflectionComposite => {
                // Metal-only inline pass; never scheduled on Vulkan. Handled here
                // only to keep the dispatch match exhaustive.
                return Err(format!(
                    "graph executor (vulkan): pass {} is a Metal-only inline \
                     reflection composite and should not appear as a graph node",
                    pass_id.name()
                ));
            }
            PassId::SsrPrepass => {
                // Merged into GBufferPrepass on Vulkan: the builder emits the
                // unified node (unified_gbuffer_prepass = true) and never this.
                return Err(format!(
                    "graph executor (vulkan): pass {} is merged into GBufferPrepass \
                     and should not appear in the frame graph",
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
            PassId::Ssgi => {
                self.encode_ssgi(cmd, params.frame_idx, params.fov_y_radians, params.aspect);
            }
            PassId::RtReflections => {
                // Hardware ray-traced reflections (inline `rayQueryEXT`). Traces a
                // reflection ray per glossy pixel against the scene TLAS and
                // composites into the RT output target, which then feeds the post
                // stack. Occupies the `SsrResolve` slot; gated by
                // `FrameGraphInputs::rt_reflections_enabled`
                // (`VkContext::rt_reflections_active`). The per-frame TLAS update +
                // descriptor re-point already ran on the outer "start" buffer.
                self.encode_rt_reflections(
                    cmd,
                    params.frame_idx,
                    params.fov_y_radians,
                    params.aspect,
                    params.cam_pos,
                );
            }
            PassId::Velocity => {
                // Merged into GBufferPrepass on Vulkan: the builder emits the
                // unified node (unified_gbuffer_prepass = true) and never this.
                return Err(format!(
                    "graph executor (vulkan): pass {} is merged into GBufferPrepass \
                     and should not appear in the frame graph",
                    pass_id.name()
                ));
            }
            PassId::TaaResolve => {
                self.encode_taa(cmd, params.frame_idx);
            }
            PassId::Upscale => {
                self.encode_upscale(cmd, params)?;
            }
            PassId::Bloom => {
                self.encode_bloom(cmd, params.frame_idx);
            }
            PassId::Shadow => {
                self.encode_shadow_pass(cmd, params.frame_idx, params.cam_pos, params.elapsed);
            }
            PassId::SpotShadow => {
                // One depth-only render per scheduled spot slice. The builder
                // emits this node only when the world has shadow-casting spots,
                // and the RAW edge on `spot_shadow_map` pins it before Main.
                self.encode_spot_shadow_pass(cmd, params.frame_idx, params.cam_pos);
            }
            PassId::Main => {
                self.encode_main_pass(
                    cmd,
                    params.frame_idx,
                    params.visible,
                    params.frustum,
                    params.cam_pos,
                    params.world_hidden,
                );
            }
            PassId::Composite => {
                self.encode_composite_and_text(
                    cmd,
                    params.image_index,
                    params.frame_idx,
                    params.text_calls,
                )?;
            }
            PassId::Decals => {
                self.encode_decals(cmd, params.frame_idx, params.vp_mat, params.frustum);
            }
            PassId::Lines => {
                self.encode_lines(cmd, params.frame_idx, params.vp_mat, params.lines);
            }
            PassId::FogFroxel => {
                // Populate the screen-aligned 3D scatter/transmittance volume
                // the `Fog` render pass samples. The shared graph seeds this
                // before `Fog` (RAW edge on the froxel volume handle).
                self.encode_fog_froxel(
                    cmd,
                    params.frame_idx,
                    params.near,
                    params.vp_mat,
                    params.cam_pos,
                );
            }
            PassId::Fog => {
                self.encode_fog(cmd, params.frame_idx, params.vp_mat, params.cam_pos);
            }
            PassId::AutoExposure => {
                self.encode_auto_exposure(cmd, params.frame_idx);
            }
            PassId::ParticlesDraw => {
                if let Some(frame) = particle_frame {
                    self.encode_particles(
                        cmd,
                        params.frame_idx,
                        frame,
                        params.vp_mat,
                        params.frustum,
                    );
                }
            }
            PassId::Raymarch => {
                // Composite each visible SDF volume into the scene. Uses the
                // jittered VP (the matrix the main pass rasterised depth with)
                // so the reprojected hit depth shares the scene's depth space.
                let view = self.build_raymarch_view(params.vp_mat, params.cam_pos, params.elapsed);
                self.encode_raymarch(cmd, params.frame_idx, &view)?;
            }
            PassId::Transparent => {
                // Generic translucent pass: draws the world's glass panels
                // back-to-front over the post-SSR scene. Gated by
                // `FrameGraphInputs::transparent_enabled` (set from
                // `glass.any_visible()`), so it only appears when the world
                // declared visible `GlassPanel`s. Uses the jittered VP (the
                // matrix the main pass rasterised depth with) so the glass
                // quad's clip-space depth matches the stored main-depth the
                // fragment shader tests against. Water is a separate
                // (Metal-only) producer not ported here.
                // Planar reflections run inline at the head of the pass (same cmd
                // buffer -> each plane's mirror target is ready before the glass
                // draws sample it). A no-op when the world has no planar set.
                // Skipped when the per-pixel RT glass trace is live: it supersedes
                // planar (sharp + off-screen-correct), so the mirror re-render would
                // be wasted. Gating on `rt_glass_active` (not `rt_reflections_active`)
                // keeps planar alive when RT is live but the glass RT pipelines
                // failed to build, so the glass probe / planar fallback samples a
                // freshly rendered resolve. Mirrors DirectX.
                if !self.rt_glass_active() {
                    self.encode_planar_reflections(
                        cmd,
                        params.frame_idx,
                        params.vp_mat,
                        params.cam_pos,
                        params.elapsed,
                    )?;
                }
                let view =
                    self.build_transparent_view(params.vp_mat, params.cam_pos, params.elapsed);
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
                // pyramid mid-frame from phase-1 depth so `Cull2` re-tests the
                // phase-1 occluded objects against up-to-date depth; `HizFinal`
                // reduces the frame's final depth for the next frame's phase-1
                // cull. Both read the depth the graph has already transitioned.
                self.encode_hiz_build(cmd, params.frame_idx);
            }
            PassId::Cull2 => {
                self.encode_cull_phase2(
                    cmd,
                    params.frame_idx,
                    params.frustum,
                    params.cam_pos,
                    params.cur_vp,
                );
            }
            PassId::Main2 => {
                self.encode_main_pass_phase2(cmd, params.frame_idx);
            }
            PassId::GBufferPrepass => {
                // Unified geometry pre-pass: one jittered traversal writes
                // normal+depth, roughness, and motion for every screen-space
                // consumer (SSR / SSAO / SSGI / TAA / FSR), replacing the
                // separate SSR / SSAO / velocity pre-passes. `params.vp_mat` is
                // the jittered VP (rasterisation, matching the main pass);
                // `params.cur_vp` is the un-jittered VP the shader uses with the
                // previous VP for the motion vector. The velocity channel carries
                // real motion only when a consumer reads it (TAA or FSR active);
                // otherwise cur == prev and it stays a harmless zero. The merged
                // buffer is built whenever any of these consumers is on, so a
                // missing `self.gbuffer` here means the builder emitted this node
                // with no merged buffer present, a programming error.
                let gb = self.gbuffer.as_ref().ok_or(
                    "graph executor (vulkan): GBufferPrepass emitted but self.gbuffer is None",
                )?;
                let velocity_active = self.taa.is_some() || self.upscale.is_some();
                self.encode_gbuffer_prepass(
                    gb,
                    cmd,
                    params.frame_idx,
                    GbufferPrepassView {
                        jittered_vp: params.vp_mat,
                        cur_vp: params.cur_vp,
                        cam_pos: params.cam_pos,
                        frustum: params.frustum,
                    },
                    params.visible,
                    velocity_active,
                );
            }
            other => {
                return Err(format!(
                    "graph executor (vulkan): pass {} is not handled by this \
                     executor; it should not appear in the frame graph",
                    other.name()
                ));
            }
        }
        Ok(())
    }
}
