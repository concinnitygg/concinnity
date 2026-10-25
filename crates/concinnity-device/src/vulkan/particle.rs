//! GPU-compute particle system for the Vulkan backend. Each `ParticleEmitter`
//! declared in the world produces one persistent `ParticleEmitterGpuState`
//! carrying a device-local pool SSBO (read-write in the compute pass, read-only
//! in the vertex pass). Each frame the renderer:
//!
//!   1. Computes the per-emitter spawn run CPU-side (a fractional accumulator
//!      drives integer particle spawns per dispatch, into the pool slots a ring
//!      cursor names).
//!   2. Dispatches the `particle_simulate` compute kernel to age + integrate +
//!      respawn each pool.
//!   3. Rasterizes one alpha-blended billboard quad per live particle into
//!      `hdr_resolve_images[frame_idx]`, its vertex stage reading the pool the
//!      dispatch wrote. The compute -> vertex transition is the graph's: the two
//!      halves are the `ParticlesSim` and `ParticlesDraw` nodes, and the pool set
//!      is the `particle_pool` graph resource.
//!
//! Runs after the volumetric-fog pass and before SSR / TAA so particles
//! appear in screen-space reflections and are temporally stabilized by the
//! TAA history. The pass attaches no depth buffer; the fragment tests the main
//! depth itself, so opaque geometry hides a sprite behind it. Mirrors
//! src/directx/particle.rs and src/metal/particle.rs.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_core::math::saturating_u32;
use concinnity_engine::gfx::frustum::Frustum;
use concinnity_engine::gfx::render_types::ParticleParams;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::particles::burst::PARTICLE_FRAME_ROWS;
use concinnity_engine::render::particles::{
    ParticleBurst, ParticleEmitterRecord, ParticleFrame, ParticleSystem,
};
use concinnity_engine::render::temporal::reactive_mask::ReactiveWrite;
use concinnity_engine::render::uniforms::{GpuParticle, ParticleView};

use super::allocator::PooledBuffer;
use super::context::{HDR_FORMAT, VkContext};
use super::descriptor_layout::{Binding, PoolSizes};
use super::pipeline_desc::{GraphicsPipelineDesc, compute_pipeline};
use super::reactive_mask::{self, WriterRenderPasses};
use super::record::cmd_push_constants;
use super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::set_writes::SetWrites;
use super::texture::GpuUploadContext;
use crate::vulkan::builtin_shaders::CompileProgram;
use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedFramebuffer, OwnedPipeline, OwnedPipelineLayout, OwnedRenderPass,
    OwnedSampler, OwnedSetLayout, VkDevice,
};

// The per-emitter descriptor pool reserves a fixed block of `2 * MAX_EMITTERS`
// sets at init (one compute set + one render set per emitter).
pub(in crate::vulkan) const MAX_EMITTERS: usize =
    concinnity_engine::render::particles::MAX_EMITTERS;

// `GpuParticle` (one simulation-pool slot) and `ParticleView` (the render-pass
// view UBO) are GPU-free layout structs that live in `core::render`
// (imported above).

// SPIR-V for the particle shader stages: the simulation, and the billboard
// pair for unlit and for lit emitters.
struct ParticleShaderSpirv {
    simulate: Vec<u8>,
    vertex: Vec<u8>,
    fragment: Vec<u8>,
    lit_vertex: Vec<u8>,
    lit_fragment: Vec<u8>,
}

// Compile the particle shaders to SPIR-V. Used by [`ParticleResources::new`]
// at init and by shader hot-reload to rebuild the pipelines against the
// existing layouts.
fn compile_particle_shaders(hot_reload: bool, msaa: bool) -> RenderResult<ParticleShaderSpirv> {
    Ok(ParticleShaderSpirv {
        simulate: super::builtin_shaders::PARTICLE_SIMULATE.compile(hot_reload)?,
        vertex: super::builtin_shaders::PARTICLE_VERT.compile(hot_reload)?,
        fragment: super::builtin_shaders::PARTICLE_FRAG
            .at(msaa)
            .compile(hot_reload)?,
        lit_vertex: super::builtin_shaders::PARTICLE_LIT_VERT.compile(hot_reload)?,
        lit_fragment: super::builtin_shaders::PARTICLE_LIT_FRAG
            .at(msaa)
            .compile(hot_reload)?,
    })
}

// Which particle draw to encode: how it treats the reactive mask, and whether
// it is the one after the see-through surfaces, into the pre-TAA scene, rather
// than the one before them, into `hdr_resolve`.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct ParticleStage {
    pub mask: ReactiveWrite,
    pub late: bool,
}

// The particle pipelines a hot reload rebuilds.
pub(in crate::vulkan) struct ParticlePipelines {
    compute: OwnedPipeline,
    render: OwnedPipeline,
    render_lit: OwnedPipeline,
}

// Per-emitter persistent GPU state: the particle pool and the descriptor sets that bind the pool. The pool sits in
// DEVICE_LOCAL memory.
pub(in crate::vulkan) struct ParticleEmitterGpuState {
    // Particle pool: `record.max_particles` slots of `GpuParticle`. Used
    // as a storage buffer by both the compute pass and the vertex pass.
    // Held for the emitter's lifetime; the descriptor sets alias it, and the
    // graph executor's barrier registry resolves `particle_pool` to it.
    pub pool_buffer: PooledBuffer,
    // Compute descriptor set (set 0): binding 0 the pool SSBO. Taken from the
    // particle descriptor pool at emitter creation and recycled on removal.
    pub compute_set: vk::DescriptorSet,
    // Render emitter descriptor set (set 1): binding 0 the pool SSBO
    // (read-only here), binding 1 the emitter's albedo image and binding 2 its
    // sampler. The albedo image is rewritten by [`VkContext::add_emitter`]
    // from the live texture pool.
    pub render_set: vk::DescriptorSet,
    // Texture-pool slot last written into `render_set`'s albedo binding.
    // Read by `rewrite_particle_albedo_slot` so a streamed or hot-reloaded
    // albedo swap that recreates this slot's view re-points the binding.
    pub texture_slot: usize,
}

impl ParticleEmitterGpuState {
    fn sets(&self) -> EmitterSets {
        EmitterSets {
            compute: self.compute_set,
            render: self.render_set,
        }
    }
}

// The emitter table with each emitter's GPU state, and the shared resources
// (pipelines, per-frame view UBO, descriptor pool, framebuffers), built only
// when the world declared at least one `ParticleEmitter` or on the first
// runtime `add_particle_emitter`.
pub(in crate::vulkan) struct ParticleState {
    pub resources: Option<ParticleResources>,
    pub system: ParticleSystem<ParticleEmitterGpuState>,
}

impl Default for ParticleState {
    fn default() -> Self {
        Self {
            resources: None,
            system: ParticleSystem::new(),
        }
    }
}

// Pipelines + per-frame view uniform ring + per-emitter descriptor pool
// shared across every emitter. Owned by `VkContext` at most once; built
// either at init (when the world declares ≥1 emitter) or on the first
// runtime `add_emitter`.
pub(in crate::vulkan) struct ParticleResources {
    // Compute pass: particle_simulate.hlsl.
    pub(in crate::vulkan) compute_pipeline: OwnedPipeline,
    pub(in crate::vulkan) compute_pipeline_layout: OwnedPipelineLayout,
    // set 0: the pool SSBO per emitter.
    pub(in crate::vulkan) compute_set_layout: OwnedSetLayout,
    // set 1: the frame's rows, one buffer and set per frame slot.
    _rows_set_layout: OwnedSetLayout,
    rows: Vec<PooledBuffer>,
    rows_sets: Vec<vk::DescriptorSet>,

    // Render passes: the particle.hlsl billboard pair over the scene and the
    // reactive mask, one per way the frame treats the mask.
    pub(in crate::vulkan) render_passes: WriterRenderPasses,
    pub(in crate::vulkan) render_pipeline: OwnedPipeline,
    // The same pair for a lit emitter.
    pub(in crate::vulkan) render_pipeline_lit: OwnedPipeline,
    pub(in crate::vulkan) render_pipeline_layout: OwnedPipelineLayout,
    // set 0: per-frame (ParticleView UBO, main depth).
    pub(in crate::vulkan) _view_set_layout: OwnedSetLayout,
    // set 1: per-emitter (pool SSBO, albedo, the albedo's sampler). Allocated for each
    // `ParticleEmitterGpuState` from `descriptor_pool` and written by
    // `add_emitter`.
    pub(in crate::vulkan) emitter_set_layout: OwnedSetLayout,

    // Per-emitter descriptor pool. Holds `MAX_EMITTERS` compute sets +
    // `MAX_EMITTERS` render emitter sets + `frames` view sets. Sized at
    // init; runtime `add_emitter` past the cap returns an error.
    pub(in crate::vulkan) descriptor_pool: OwnedDescriptorPool,
    // Set pairs of removed emitters, reused before allocating from the pool.
    spare_sets: EmitterSetFreeList,

    // Per-frame view UBO (single 96-byte block), persistently mapped.
    pub(in crate::vulkan) view_ubos: Vec<PooledBuffer>,
    // Per-frame view set (binding 0 view UBO, 1 main depth). One per frame slot.
    pub(in crate::vulkan) view_sets: Vec<vk::DescriptorSet>,

    // One framebuffer per frame-in-flight slot, each binding its frame
    // slot's `hdr_resolve_images[i].view` and reactive mask as the color
    // attachments.
    pub(in crate::vulkan) framebuffers: Vec<OwnedFramebuffer>,
    // The same over each slot's pre-TAA scene, which the emitters that draw
    // after the see-through surfaces blend into.
    late_framebuffers: Vec<OwnedFramebuffer>,

    // Linear-clamp sampler shared by every emitter's albedo binding.
    pub(in crate::vulkan) sampler: OwnedSampler,

    // Whether the main depth is multisampled; picks the fragment variant.
    msaa: bool,
}

// Render-target inputs the particle pass writes into / samples from: the
// per-frame resolved HDR color views, the per-frame main depth views, and the
// framebuffer extent.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct ParticlePassTargets<'a> {
    pub(in crate::vulkan) hdr_resolve_views: &'a [vk::ImageView],
    // The pre-TAA scene per frame slot, which the transparent pass draws into.
    pub(in crate::vulkan) scene_views: &'a [vk::ImageView],
    pub(in crate::vulkan) reactive_mask_views: &'a [vk::ImageView],
    pub(in crate::vulkan) depth_views: &'a [vk::ImageView],
    pub(in crate::vulkan) extent: vk::Extent2D,
}

// One framebuffer per frame slot over that slot's `scenes` view and reactive
// mask.
fn create_framebuffers(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    targets: &ParticlePassTargets<'_>,
    scenes: &[vk::ImageView],
    frames: usize,
) -> RenderResult<Vec<OwnedFramebuffer>> {
    scenes
        .iter()
        .zip(targets.reactive_mask_views)
        .take(frames)
        .map(|(&scene, &mask)| {
            let attachments = device.writer_targets().views(scene, mask);
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(targets.extent.width.max(1))
                .height(targets.extent.height.max(1))
                .layers(1);
            device
                .create_framebuffer(&fb_info)
                .map_err(|e| super::error::map_vk_result(e, "particle framebuffer"))
        })
        .collect()
}

impl ParticleResources {
    // Build the particle compute + render pipelines, the per-frame view
    // UBO ring, the shared sampler, the descriptor pool, and the per-frame
    // framebuffers. Called from `VkContext::new` only when the world
    // declared at least one `ParticleEmitter`. The encoder is a no-op
    // when this is `None`.
    pub(in crate::vulkan) fn new(
        gpu: &GpuUploadContext<'_>,
        frames: usize,
        targets: ParticlePassTargets<'_>,
        msaa: bool,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let &GpuUploadContext { alloc, device, .. } = gpu;
        let depth_views = targets.depth_views;
        let render_passes = WriterRenderPasses::new(device.writer_targets(), |mask| {
            create_render_pass(device, HDR_FORMAT, mask)
        })?;
        let compute_set_layout = create_descriptor_set_layout(device, &compute_set_bindings())?;
        let rows_set_layout = create_descriptor_set_layout(device, &rows_set_bindings())?;
        let view_set_layout = create_descriptor_set_layout(device, &view_set_bindings())?;
        let emitter_set_layout = create_descriptor_set_layout(device, &emitter_set_bindings())?;
        let compute_pipeline_layout = create_compute_pipeline_layout(
            device,
            compute_set_layout.handle(),
            rows_set_layout.handle(),
        )?;
        let render_pipeline_layout = create_render_pipeline_layout(
            device,
            view_set_layout.handle(),
            emitter_set_layout.handle(),
        )?;

        let spirv = compile_particle_shaders(hot_reload, msaa)?;
        let compute_pipeline = compute_pipeline(
            device,
            compute_pipeline_layout.handle(),
            &spirv.simulate,
            "particle compute",
        )?;
        let render = |vertex: &[u8], fragment: &[u8]| {
            create_render_pipeline(
                device,
                render_passes.compatible(),
                render_pipeline_layout.handle(),
                vertex,
                fragment,
            )
        };
        let render_pipeline = render(&spirv.vertex, &spirv.fragment)?;
        let render_pipeline_lit = render(&spirv.lit_vertex, &spirv.lit_fragment)?;

        // Per-frame ParticleView UBOs (HOST_VISIBLE | HOST_COHERENT,
        // persistently mapped).
        let view_size = size_of::<ParticleView>() as u64;
        let mut view_ubos = Vec::with_capacity(frames);
        for _ in 0..frames {
            let buf = alloc.create_buffer(
                view_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            view_ubos.push(buf);
        }

        let sampler = create_sampler(device)?;
        let descriptor_pool = create_descriptor_pool(device, frames)?;

        // Per-frame view sets (one per frame slot).
        let view_layouts: Vec<_> = (0..frames).map(|_| view_set_layout.handle()).collect();
        let view_sets = alloc_descriptor_sets(device, descriptor_pool.handle(), &view_layouts)?;
        for (i, &set) in view_sets.iter().enumerate() {
            SetWrites::new(set)
                .uniform_buffer(0, view_ubos[i].buffer(), view_size)
                .sampled_image(1, frame_view(depth_views, i))
                .apply(device);
        }

        // Per-frame rows (HOST_VISIBLE | HOST_COHERENT, persistently mapped)
        // and the set each frame's simulation binds them through.
        let rows_bytes = (PARTICLE_FRAME_ROWS * size_of::<[f32; 4]>()) as u64;
        let rows = (0..frames)
            .map(|_| {
                alloc.create_buffer(
                    rows_bytes,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            })
            .collect::<RenderResult<Vec<_>>>()?;
        let rows_layouts: Vec<_> = (0..frames).map(|_| rows_set_layout.handle()).collect();
        let rows_sets = alloc_descriptor_sets(device, descriptor_pool.handle(), &rows_layouts)?;
        for (set, buffer) in rows_sets.iter().zip(&rows) {
            SetWrites::new(*set)
                .storage_buffer(0, buffer.buffer(), rows_bytes)
                .apply(device);
        }

        let framebuffers = create_framebuffers(
            device,
            render_passes.compatible(),
            &targets,
            targets.hdr_resolve_views,
            frames,
        )?;
        let late_framebuffers = create_framebuffers(
            device,
            render_passes.compatible(),
            &targets,
            targets.scene_views,
            frames,
        )?;

        Ok(Self {
            compute_pipeline,
            compute_pipeline_layout,
            compute_set_layout,
            _rows_set_layout: rows_set_layout,
            rows,
            rows_sets,
            render_passes,
            render_pipeline,
            render_pipeline_lit,
            render_pipeline_layout,
            _view_set_layout: view_set_layout,
            emitter_set_layout,
            descriptor_pool,
            spare_sets: EmitterSetFreeList::default(),
            view_ubos,
            view_sets,
            framebuffers,
            late_framebuffers,
            sampler,
            msaa,
        })
    }

    // Rebuild the framebuffers + re-point the per-frame depth binding after a
    // swapchain resize. Called from `VkContext::rebuild_swapchain`; same
    // pattern as `FogResources` / `DecalResources`. The pipelines, layouts,
    // buffers, sampler, and per-emitter descriptor sets all survive.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        device: &VkDevice,
        targets: ParticlePassTargets<'_>,
    ) -> RenderResult<()> {
        for (i, &set) in self.view_sets.iter().enumerate() {
            write_depth_binding(device, set, frame_view(targets.depth_views, i));
        }
        self.framebuffers.clear();
        self.late_framebuffers.clear();
        let frames = self.view_ubos.len();
        let pass = self.render_passes.compatible();
        self.framebuffers =
            create_framebuffers(device, pass, &targets, targets.hdr_resolve_views, frames)?;
        self.late_framebuffers =
            create_framebuffers(device, pass, &targets, targets.scene_views, frames)?;
        Ok(())
    }

    // Re-point frame slot `frame`'s late framebuffer at `scene`, the post
    // chain's scene image a settings change moved. The slot's fence has
    // passed; the replaced framebuffer waits out the owned-handle retire.
    pub(in crate::vulkan) fn rescene(
        &mut self,
        device: &VkDevice,
        frame: usize,
        (scene, mask): (vk::ImageView, vk::ImageView),
        extent: vk::Extent2D,
    ) -> RenderResult<()> {
        let attachments = device.writer_targets().views(scene, mask);
        let fb_info = vk::FramebufferCreateInfo::default()
            .render_pass(self.render_passes.compatible())
            .attachments(&attachments)
            .width(extent.width.max(1))
            .height(extent.height.max(1))
            .layers(1);
        let fb = device
            .create_framebuffer(&fb_info)
            .map_err(|e| super::error::map_vk_result(e, "particle framebuffer"))?;
        if let Some(slot) = self.late_framebuffers.get_mut(frame) {
            *slot = fb;
        }
        Ok(())
    }

    // Construct the compute + render pipelines against the existing
    // layouts. Used by the shader hot-reload pass.
    pub(in crate::vulkan) fn rebuild_pipelines(
        &self,
        device: &VkDevice,
        hot_reload: bool,
    ) -> RenderResult<ParticlePipelines> {
        let spirv = compile_particle_shaders(hot_reload, self.msaa)?;
        let compute = compute_pipeline(
            device,
            self.compute_pipeline_layout.handle(),
            &spirv.simulate,
            "particle compute",
        )?;
        let render = |vertex: &[u8], fragment: &[u8]| {
            create_render_pipeline(
                device,
                self.render_passes.compatible(),
                self.render_pipeline_layout.handle(),
                vertex,
                fragment,
            )
        };
        Ok(ParticlePipelines {
            compute,
            render: render(&spirv.vertex, &spirv.fragment)?,
            render_lit: render(&spirv.lit_vertex, &spirv.lit_fragment)?,
        })
    }

    // Swap the freshly-built pipelines in. The caller has already
    // `device_wait_idle`'d so the old pipelines are not in flight.
    pub(in crate::vulkan) fn swap_pipelines(&mut self, pipelines: ParticlePipelines) {
        self.compute_pipeline = pipelines.compute;
        self.render_pipeline = pipelines.render;
        self.render_pipeline_lit = pipelines.render_lit;
    }

    // Free every owned handle. Called from `Drop for VkContext` after
    // `device_wait_idle`. Per-emitter pools live in
    // `VkContext`'s `particle.system`; their destruction is the
    // caller's responsibility.
    pub(in crate::vulkan) fn destroy(&mut self, _device: &VkDevice) {
        self.framebuffers.clear();
        self.late_framebuffers.clear();
        self.view_ubos.clear();
    }
}

// Allocate the per-emitter GPU state: a zero-initialized DEVICE_LOCAL pool
// SSBO. Also takes the emitter's compute + render descriptor sets (a removed
// emitter's pair when one is spare) and writes the pool bindings. The albedo
// binding stays stale; `add_emitter` writes it from the live texture pool.
pub(in crate::vulkan) fn build_emitter_gpu_state(
    gpu: GpuUploadContext<'_>,
    resources: &mut ParticleResources,
    record: &ParticleEmitterRecord,
) -> RenderResult<ParticleEmitterGpuState> {
    // Destructure the handles the buffer allocations need directly; the
    // one-shot zero-fills below take the whole `gpu` context (it is Copy).
    let GpuUploadContext { alloc, device, .. } = gpu;
    let slots = u64::from(record.max_particles);
    let pool_bytes = slots * size_of::<GpuParticle>() as u64;

    // Pool buffer: DEVICE_LOCAL, used as STORAGE by both passes. The
    // compute kernel writes through it; the vertex stage reads it. A WAR
    // barrier in the encoder transitions accesses between dispatches.
    let pool_buffer = alloc.create_buffer(
        pool_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    zero_device_buffer(gpu, pool_buffer.buffer(), pool_bytes)?;

    let set_layouts = [
        resources.compute_set_layout.handle(),
        resources.emitter_set_layout.handle(),
    ];
    let pool = resources.descriptor_pool.handle();
    let EmitterSets {
        compute: compute_set,
        render: render_set,
    } = resources.spare_sets.take_or(|| {
        let sets = alloc_descriptor_sets(device, pool, &set_layouts)?;
        Ok(EmitterSets {
            compute: sets[0],
            render: sets[1],
        })
    })?;

    SetWrites::new(compute_set)
        .storage_buffer(0, pool_buffer.buffer(), pool_bytes)
        .apply(device);
    // The render set's pool binding and the albedo's sampler never change. The
    // albedo image (binding 1) is written by `add_emitter` from the live
    // texture pool.
    SetWrites::new(render_set)
        .storage_buffer(0, pool_buffer.buffer(), pool_bytes)
        .sampler(2, resources.sampler.handle())
        .apply(device);

    Ok(ParticleEmitterGpuState {
        pool_buffer,
        compute_set,
        render_set,
        texture_slot: usize::MAX,
    })
}

// An emitter's (compute, render) descriptor set pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EmitterSets {
    compute: vk::DescriptorSet,
    render: vk::DescriptorSet,
}

// Set pairs handed back by removed emitters. The pool is never reset, so
// reusing these keeps add/remove cycles from draining it.
#[derive(Default)]
struct EmitterSetFreeList(Vec<EmitterSets>);

impl EmitterSetFreeList {
    // Pop a spare pair, or allocate a fresh one with `alloc` when none is spare.
    fn take_or(
        &mut self,
        alloc: impl FnOnce() -> RenderResult<EmitterSets>,
    ) -> RenderResult<EmitterSets> {
        match self.0.pop() {
            Some(sets) => Ok(sets),
            None => alloc(),
        }
    }

    // Return a pair whose emitter is gone and no longer referenced by the GPU.
    fn recycle(&mut self, sets: EmitterSets) {
        self.0.push(sets);
    }
}

// Render pass / descriptor / pipeline construction

fn create_render_pass(
    device: &VkDevice,
    format: vk::Format,
    mask: Option<ReactiveWrite>,
) -> RenderResult<OwnedRenderPass> {
    // The resolved HDR scene, and the reactive mask when the writers carry it.
    // The fog pass left the scene in SHADER_READ_ONLY_OPTIMAL; we want it in
    // COLOR_ATTACHMENT during the subpass and SHADER_READ_ONLY_OPTIMAL again
    // on exit so SSR / TAA / bloom / composite can sample it. Mirrors the
    // decal / fog render passes.
    let attachment = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let attachments = device.writer_targets().attachments(attachment, mask);
    let color_refs: Vec<_> = (0..u32::saturating_from(attachments.len()))
        .map(|i| {
            vk::AttachmentReference::default()
                .attachment(i)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        })
        .collect();
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs);
    let dep_in = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::COLOR_ATTACHMENT_READ,
        );
    let dep_out = vk::SubpassDependency::default()
        .src_subpass(0)
        .dst_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
    let deps = [dep_in, dep_out];
    let info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(&deps);
    device
        .create_render_pass(&info)
        .map_err(|e| super::error::map_vk_result(e, "particle render pass"))
}

// Compute set 0: the pool SSBO.
// Compute set 1, per frame: the frame's rows.
fn rows_set_bindings() -> [Binding; 1] {
    [(
        0,
        vk::DescriptorType::STORAGE_BUFFER,
        vk::ShaderStageFlags::COMPUTE,
    )]
}

fn compute_set_bindings() -> [Binding; 1] {
    [(
        0,
        vk::DescriptorType::STORAGE_BUFFER,
        vk::ShaderStageFlags::COMPUTE,
    )]
}

// Render set 0, per frame: the ParticleView UBO (vertex) and the main depth
// (fragment).
fn view_set_bindings() -> [Binding; 2] {
    use vk::DescriptorType as T;
    [
        (0, T::UNIFORM_BUFFER, vk::ShaderStageFlags::VERTEX),
        (1, T::SAMPLED_IMAGE, vk::ShaderStageFlags::FRAGMENT),
    ]
}

// Render set 1, per emitter: the pool SSBO, the albedo and its sampler.
fn emitter_set_bindings() -> [Binding; 3] {
    use vk::DescriptorType as T;
    let frag = vk::ShaderStageFlags::FRAGMENT;
    [
        (0, T::STORAGE_BUFFER, vk::ShaderStageFlags::VERTEX),
        (1, T::SAMPLED_IMAGE, frag),
        (2, T::SAMPLER, frag),
    ]
}

// Push-constant range covering the full `ParticleParams` block.
// Visible to vertex (size_start/end, color_start/end) + fragment (none:
// vertex emits the color; fragment reads it via varyings) + compute
// (every field). The vertex stage actually only reads the gradient + size
// fields, but binding the full struct keeps the host upload single-shot.
const PARTICLE_PUSH_BYTES: u32 = saturating_u32(size_of::<ParticleParams>());
// 128 bytes is the `maxPushConstantsSize` every Vulkan device offers.
const _: () = assert!(PARTICLE_PUSH_BYTES <= 128);

fn create_compute_pipeline_layout(
    device: &VkDevice,
    compute_set_layout: vk::DescriptorSetLayout,
    rows_set_layout: vk::DescriptorSetLayout,
) -> RenderResult<OwnedPipelineLayout> {
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(PARTICLE_PUSH_BYTES);
    let set_layouts = [compute_set_layout, rows_set_layout];
    let info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(std::slice::from_ref(&push_range));
    device
        .create_pipeline_layout(&info)
        .map_err(|e| super::error::map_vk_result(e, "particle compute pipeline layout"))
}

fn create_render_pipeline_layout(
    device: &VkDevice,
    view_set_layout: vk::DescriptorSetLayout,
    emitter_set_layout: vk::DescriptorSetLayout,
) -> RenderResult<OwnedPipelineLayout> {
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(PARTICLE_PUSH_BYTES);
    let set_layouts = [view_set_layout, emitter_set_layout];
    let info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(std::slice::from_ref(&push_range));
    device
        .create_pipeline_layout(&info)
        .map_err(|e| super::error::map_vk_result(e, "particle render pipeline layout"))
}

fn create_descriptor_pool(device: &VkDevice, frames: usize) -> RenderResult<OwnedDescriptorPool> {
    let frames = u32::saturating_from(frames);
    let max_emitters = u32::saturating_from(MAX_EMITTERS);
    // One view and one rows set per frame slot, one compute + one render set
    // per emitter.
    let sizes = PoolSizes::default()
        .sets(&view_set_bindings(), frames)
        .sets(&rows_set_bindings(), frames)
        .sets(&compute_set_bindings(), max_emitters)
        .sets(&emitter_set_bindings(), max_emitters)
        .build();
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(2 * frames + 2 * max_emitters)
        .pool_sizes(&sizes);
    device
        .create_descriptor_pool(&info)
        .map_err(|e| super::error::map_vk_result(e, "particle descriptor pool"))
}

// The main depth view for frame slot `i`, clamped to the last one when there
// are fewer depth images than frame slots.
fn frame_view(views: &[vk::ImageView], i: usize) -> vk::ImageView {
    views[i.min(views.len().saturating_sub(1))]
}

// The main depth, which the fragment reads by texel.
fn write_depth_binding(device: &VkDevice, set: vk::DescriptorSet, depth_view: vk::ImageView) {
    SetWrites::new(set)
        .sampled_image(1, depth_view)
        .apply(device);
}

fn create_sampler(device: &VkDevice) -> RenderResult<OwnedSampler> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK)
        .max_lod(vk::LOD_CLAMP_NONE);
    device
        .create_sampler(&info)
        .map_err(|e| super::error::map_vk_result(e, "particle sampler"))
}

// One alpha-blended billboard quad per instance: the vertex shader emits the
// quad from gl_VertexIndex and reads the particle from the pool by
// gl_InstanceIndex, so there are no vertex buffers.
fn create_render_pipeline(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> RenderResult<OwnedPipeline> {
    GraphicsPipelineDesc {
        topology: vk::PrimitiveTopology::TRIANGLE_STRIP,
        ..GraphicsPipelineDesc::fullscreen(
            vert_spv,
            frag_spv,
            layout,
            render_pass,
            device.writer_targets().blends(),
        )
    }
    .build(device, "particle render")
}

// Zero-initialize a DEVICE_LOCAL buffer by recording a `vkCmdFillBuffer`
// inside a one-shot command buffer. Cheaper than the staging-buffer
// alternative and trivially correct since `vkCmdFillBuffer` writes a
// 32-bit pattern; `bytes` is a multiple of 4 since a pool slot is 32 bytes.
fn zero_device_buffer(
    gpu: GpuUploadContext<'_>,
    target: vk::Buffer,
    bytes: u64,
) -> RenderResult<()> {
    let GpuUploadContext {
        device,
        command_pool,
        queue,
        ..
    } = gpu;
    super::texture::one_shot_submit(device, command_pool, queue, |cmd| {
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe { device.cmd_fill_buffer(cmd, target, 0, bytes, 0) };
    })
}

// Encoder

impl VkContext {
    // Advance the particle clock and every emitter's spawn run on `&mut self`
    // before the render-graph fan-out, so the `&self` sim and draw halves can
    // record on parallel workers against one frame, and write the frame's view
    // block both draw passes read. `None` when the pass is inert.
    pub(in crate::vulkan) fn prepare_particle_pass(
        &mut self,
        elapsed: f32,
        bursts: &[ParticleBurst],
        frame_idx: usize,
        vp: [[f32; 4]; 4],
    ) -> Option<ParticleFrame> {
        let resources = self.particle.resources.as_ref()?;
        let frame = self.particle.system.prepare(elapsed, bursts)?;
        if let Some(rows) = resources.rows.get(frame_idx) {
            rows.write_slice(0, self.particle.system.rows());
        }
        if let Some(view) = resources.view_ubos.get(frame_idx) {
            let view_uni = ParticleView::new(
                vp,
                &self.state.view.matrix,
                self.pre_exposure.value(),
                self.uniforms.lights.uniforms(),
                &self.state.view.daylight,
            );
            view.write_val(0, &view_uni);
        }
        Some(frame)
    }

    // Encode the `ParticlesSim` node: dispatch the simulation kernel over each
    // live emitter's pool. A no-op when no pipeline has been built (no emitter
    // has ever existed in this session) or when every slot is tombstoned.
    // `frame` is the state `prepare_particle_pass` computed on `&mut self`; this
    // method takes `&self` (no spawn-state mutation) so it can run on a
    // parallel-recording worker.
    //
    // Both hazards on the pool are the graph's: the compute -> vertex one against
    // the draw, from the `particle_pool` read the draw declares in the VERTEX
    // stage, and the one against the previous frame's dispatch and draw, from the
    // pool's carried resting state.
    pub(in crate::vulkan) fn encode_particles_sim(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        frame: ParticleFrame,
    ) {
        let Some(resources) = self.particle.resources.as_ref() else {
            return;
        };
        if self.particle.system.is_empty() {
            return;
        }
        let device = &self.hw.device;

        // One dispatch per live emitter; resources are disjoint between
        // emitters so no inter-dispatch barrier is needed.
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                resources.compute_pipeline.handle(),
            );
        }
        let Some(&rows_set) = resources.rows_sets.get(frame_idx) else {
            return;
        };
        // SAFETY: `cmd` is recording, and the frame's rows set is owned by `resources` and
        // matches set 1 of the compute layout.
        unsafe {
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                resources.compute_pipeline_layout.handle(),
                1,
                std::slice::from_ref(&rows_set),
                &[],
            );
        }
        for emitter in self.particle.system.emitters(&frame) {
            let (rec, gpu, params) = (emitter.record, emitter.gpu, &emitter.params);
            // SAFETY: `cmd` is recording, and the emitter's compute set is owned by its GPU slot
            // and matches the compute layout.
            unsafe {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    resources.compute_pipeline_layout.handle(),
                    0,
                    std::slice::from_ref(&gpu.compute_set),
                    &[],
                );
            }
            // SAFETY: the compute layout is owned by `resources` and matches the bound pipeline;
            // `params` is borrowed for the call.
            unsafe {
                cmd_push_constants(
                    device,
                    cmd,
                    resources.compute_pipeline_layout.handle(),
                    vk::ShaderStageFlags::COMPUTE,
                    params,
                );
            }
            let groups = rec.max_particles.div_ceil(64);
            // SAFETY: the sim pipeline, this emitter's set and its push constants are bound.
            unsafe { device.cmd_dispatch(cmd, groups, 1, 1) };
        }
    }

    // Encode the `ParticlesDraw` node: one alpha-blended camera-facing quad per
    // live particle of every visible emitter, into this frame's `hdr_resolve`.
    // The vertex stage reads the pool `encode_particles_sim` wrote; the graph
    // derives that compute -> vertex transition from the declared read, so
    // nothing here transitions the pool.
    pub(in crate::vulkan) fn encode_particles_draw(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        frame: ParticleFrame,
        frustum: &Frustum,
        stage: ParticleStage,
    ) -> bool {
        let Some(resources) = self.particle.resources.as_ref() else {
            return false;
        };
        let ParticleStage { mask, late } = stage;
        let device = &self.hw.device;
        let extent = self.targets.render_extent;

        // Visibility-cull per emitter, for the draw alone: the simulation ticked
        // every live pool so off-screen emitters stay in a realistic mid-life
        // state when the camera turns back. Tombstoned (None) slots are always
        // invisible.
        let mut visible = self
            .particle
            .system
            .visible(&frame, frustum, late)
            .peekable();
        if visible.peek().is_none() {
            return false;
        }

        // Begin the render pass into this frame's framebuffer (which
        // binds the resolved HDR target as color attachment 0). The
        // render pass declares the round-trip
        // SHADER_READ_ONLY_OPTIMAL → COLOR_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
        // via its subpass dependencies, so no explicit image barrier is
        // needed here.
        let clears = [reactive_mask::REACTIVE_MASK_CLEAR; 2];
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(resources.render_passes.get(mask))
            .framebuffer(
                if late {
                    &resources.late_framebuffers
                } else {
                    &resources.framebuffers
                }[frame_idx]
                    .handle(),
            )
            .render_area(vk::Rect2D::default().extent(extent))
            .clear_values(&clears);
        // Negative-height viewport flips clip-space Y to match the main +
        // shadow + decal passes (the engine's `camera_projection()` produces +Y-up
        // clip coords, OpenGL-style; the Vulkan framebuffer has +Y down, so
        // the flip happens in the viewport). The fog pass dodges this with
        // a positive-height viewport because it emits NDC-space verts
        // directly; we MVP-transform world geometry, so we need the same
        // convention as the main pass.
        let viewport = vk::Viewport {
            x: 0.0,
            y: extent.height as f32,
            width: extent.width as f32,
            height: -(extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(extent);

        // SAFETY: `cmd` is recording outside a render pass, and the particle render pass and
        // framebuffer named by `rp_begin` are owned by `resources`.
        unsafe { device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE) };
        // SAFETY: the particle render pass is open on `cmd`; `viewport` is borrowed for the call.
        unsafe { device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&viewport)) };
        // SAFETY: same open render pass; `scissor` is borrowed for the call.
        unsafe { device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor)) };
        // SAFETY: the particle render pipeline is owned by `resources` and was built against
        // this render pass.
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                resources.render_pipeline.handle(),
            );
        }
        // SAFETY: the frame's view set is owned by `resources` and matches the render layout.
        unsafe {
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                resources.render_pipeline_layout.handle(),
                0,
                std::slice::from_ref(&resources.view_sets[frame_idx]),
                &[],
            );
        }

        let mut lit = false;
        for emitter in visible {
            let (rec, gpu, params) = (emitter.record, emitter.gpu, &emitter.params);
            if rec.lit != lit {
                lit = rec.lit;
                let pipeline = if lit {
                    &resources.render_pipeline_lit
                } else {
                    &resources.render_pipeline
                };
                // SAFETY: both render pipelines are owned by `resources`, built against this
                // render pass and the one render layout, so the bound sets stay valid.
                unsafe {
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.handle(),
                    );
                }
            }
            // SAFETY: the emitter's render set is owned by its GPU slot and reads the pool the
            // sim pass wrote; it matches set 1 of the render layout.
            unsafe {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    resources.render_pipeline_layout.handle(),
                    1,
                    std::slice::from_ref(&gpu.render_set),
                    &[],
                );
            }
            // SAFETY: the render layout is owned by `resources` and matches the bound pipeline;
            // `params` is borrowed for the call.
            unsafe {
                cmd_push_constants(
                    device,
                    cmd,
                    resources.render_pipeline_layout.handle(),
                    vk::ShaderStageFlags::VERTEX,
                    params,
                );
            }
            // SAFETY: pipeline, both sets and push constants are bound inside the open pass; one
            // instance per pool slot of `rec.max_particles`.
            unsafe { device.cmd_draw(cmd, 4, rec.max_particles, 0, 0) };
            self.inc_draw_calls(1);
        }
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_end_render_pass(cmd);
        }
        true
    }
}

// Runtime mutation (RenderBackend::add_emitter / remove_emitter)

impl VkContext {
    // Append a runtime emitter. Builds the particle pipelines + per-frame
    // uniform ring on first use (matching the init-time path) so a world
    // that never declared an emitter pays zero pipeline cost until the
    // first add. Reuses tombstoned slots from a prior `remove_emitter`
    // before growing the vec.
    pub(in crate::vulkan) fn add_particle_emitter(
        &mut self,
        record: ParticleEmitterRecord,
    ) -> RenderResult<usize> {
        if self.particle.resources.is_none() {
            let hdr_resolve_views: Vec<vk::ImageView> = self
                .targets
                .hdr_resolve_images
                .iter()
                .map(|img| img.view)
                .collect();
            let depth_views: Vec<vk::ImageView> = self
                .targets
                .depth_images
                .iter()
                .map(|img| img.view)
                .collect();
            let reactive_mask_views: Vec<vk::ImageView> = self
                .targets
                .reactive_mask_images
                .iter()
                .map(|img| img.view)
                .collect();
            let scene_views: Vec<vk::ImageView> = (0..self.frames_in_flight)
                .map(|i| self.post_scene_image(i).view)
                .collect();
            let resources = ParticleResources::new(
                &GpuUploadContext {
                    alloc: &self.hw.alloc,
                    device: &self.hw.device,
                    command_pool: self.commands.command_pool,
                    queue: self.hw.graphics_queue,
                },
                self.frames_in_flight,
                ParticlePassTargets {
                    hdr_resolve_views: &hdr_resolve_views,
                    scene_views: &scene_views,
                    reactive_mask_views: &reactive_mask_views,
                    depth_views: &depth_views,
                    extent: self.targets.render_extent,
                },
                self.targets.msaa_samples != vk::SampleCountFlags::TYPE_1,
                self.hot_reload.enabled,
            )?;
            self.particle.resources = Some(resources);
        }

        // Reuse a tombstoned slot if available; otherwise grow the vec.
        // The cap check is independent of slot availability.
        if !self.particle.system.has_room() {
            return Err(RenderError::Other(format!(
                "add_emitter: MAX_EMITTERS ({MAX_EMITTERS}) exceeded"
            )));
        }

        let gpu_state = build_emitter_gpu_state(
            GpuUploadContext {
                alloc: &self.hw.alloc,
                device: &self.hw.device,
                command_pool: self.commands.command_pool,
                queue: self.hw.graphics_queue,
            },
            self.particle
                .resources
                .as_mut()
                .expect("particle resources are live"),
            &record,
        )?;

        // Write the albedo binding from the live texture pool.
        let last_tex = self.scene.textures.len().saturating_sub(1);
        let tex_idx = record.texture_slot.min(last_tex);
        write_render_albedo_binding(
            &self.hw.device,
            gpu_state.render_set,
            self.scene.textures[tex_idx].view,
        );

        let state = ParticleEmitterGpuState {
            texture_slot: tex_idx,
            ..gpu_state
        };
        let sets = state.sets();
        self.particle.system.insert(record, state).map_err(|_| {
            // Nothing has recorded the rejected emitter's sets yet.
            self.recycle_emitter_sets(sets);
            RenderError::Other(format!(
                "add_emitter: MAX_EMITTERS ({MAX_EMITTERS}) exceeded"
            ))
        })
    }

    // Tombstone a runtime emitter slot. The id becomes invalid; the next
    // `add_emitter` may reuse it. The pool + counter buffers are dropped
    // after a `device_wait_idle`: Vulkan has no driver-side keep-alive
    // for in-flight buffer references, so we must drain the queue before
    // freeing the backing memory. Reached only through the bin's `cn debug`
    // runtime-mutation path (dead in the FFI lib, live in the bin).
    pub(in crate::vulkan) fn remove_particle_emitter(
        &mut self,
        emitter_id: usize,
    ) -> RenderResult<()> {
        let state = self
            .particle
            .system
            .remove(emitter_id)
            .map_err(|e| RenderError::Other(format!("remove_emitter: id {emitter_id} {e}")))?;
        // Drain the queue before freeing the pool so an in-flight command
        // buffer can't dereference the freed memory or the recycled sets.
        self.wait_idle();
        self.recycle_emitter_sets(state.sets());
        drop(state);
        Ok(())
    }

    fn recycle_emitter_sets(&mut self, sets: EmitterSets) {
        if let Some(resources) = self.particle.resources.as_mut() {
            resources.spare_sets.recycle(sets);
        }
    }

    // Wire every world-authored particle emitter through `add_particle_emitter`
    // so the same descriptor / SRV / GPU-state path serves both init and
    // runtime adds. Called from `VkContext::new` after the texture pool
    // is uploaded.
    pub(in crate::vulkan) fn upload_initial_particles(
        &mut self,
        records: Vec<ParticleEmitterRecord>,
    ) -> RenderResult<()> {
        if records.is_empty() {
            return Ok(());
        }
        if records.len() > MAX_EMITTERS {
            return Err(RenderError::Other(format!(
                "particles: {} authored emitters exceed MAX_EMITTERS ({})",
                records.len(),
                MAX_EMITTERS
            )));
        }
        for record in records {
            self.add_particle_emitter(record)?;
        }
        Ok(())
    }

    // Re-point every emitter's albedo binding (set 1, binding 1) that samples
    // texture-pool `slot` at the just-swapped `self.scene.textures[slot]` view. The
    // emitter albedo lives in the shared texture pool, so a streamed or
    // hot-reloaded albedo swap recreates the view and leaves a dangling
    // descriptor unless every emitter sampling that slot is re-pointed. Called
    // from `rewrite_texture_slot`, the sibling of the per-object / clone rewires.
    // Whether any live emitter's render set samples texture-pool `slot`. The
    // streaming fast path checks this: emitter sets are single-copy and bound
    // whenever the particle pass runs, so a swap of a slot they sample must
    // drain the device before rewriting.
    pub(in crate::vulkan) fn particle_samples_slot(&self, slot: usize) -> bool {
        let last = self.scene.textures.len().saturating_sub(1);
        self.particle
            .system
            .gpu_states()
            .any(|state| state.texture_slot.min(last) == slot)
    }

    pub(in crate::vulkan) fn rewrite_particle_albedo_slot(&self, slot: usize) {
        if self.particle.resources.is_none() {
            return;
        }
        let last = self.scene.textures.len().saturating_sub(1);
        let view = self.scene.textures[slot].view;
        for state in self.particle.system.gpu_states() {
            if state.texture_slot.min(last) == slot {
                write_render_albedo_binding(&self.hw.device, state.render_set, view);
            }
        }
    }

    // Free every per-emitter pool/counter buffer. Called from
    // `Drop for VkContext` after `device_wait_idle`. Sibling of
    // `ParticleResources::destroy`, which handles the shared pipelines.
    pub(in crate::vulkan) fn destroy_particle_emitter_states(&mut self, _device: &VkDevice) {
        // The pooled per-emitter buffers retire through the allocator as the
        // states drop.
        self.particle.system.clear();
    }
}

// Point an emitter's render set at albedo `view`. Its sampler was written with
// the set.
fn write_render_albedo_binding(device: &VkDevice, set: vk::DescriptorSet, view: vk::ImageView) {
    SetWrites::new(set).sampled_image(1, view).apply(device);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    fn sets(n: u64) -> EmitterSets {
        EmitterSets {
            compute: vk::DescriptorSet::from_raw(2 * n + 1),
            render: vk::DescriptorSet::from_raw(2 * n + 2),
        }
    }

    #[test]
    fn take_allocates_only_when_nothing_is_spare() {
        let mut list = EmitterSetFreeList::default();
        assert_eq!(list.take_or(|| Ok(sets(0))).unwrap(), sets(0));
        list.recycle(sets(7));
        let reused = list.take_or(|| panic!("a spare pair was available"));
        assert_eq!(reused.unwrap(), sets(7));
    }

    #[test]
    fn add_remove_cycles_past_the_cap_allocate_once_per_live_emitter() {
        let mut list = EmitterSetFreeList::default();
        let mut allocated = 0u64;
        for _ in 0..3 * MAX_EMITTERS {
            let pair = list
                .take_or(|| {
                    allocated += 1;
                    Ok(sets(allocated))
                })
                .unwrap();
            list.recycle(pair);
        }
        assert_eq!(allocated, 1);
    }

    #[test]
    fn a_failed_allocation_leaves_spares_untouched() {
        let mut list = EmitterSetFreeList::default();
        let err = list.take_or(|| Err(RenderError::Other("pool exhausted".into())));
        assert!(err.is_err());
        list.recycle(sets(1));
        assert_eq!(list.take_or(|| Ok(sets(2))).unwrap(), sets(1));
    }
}
