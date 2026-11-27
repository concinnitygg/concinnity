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
use concinnity_engine::protocol::ParticleBurst;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::backend::{EmitterPass, RenderBackend};
use concinnity_engine::render::particles::burst::PARTICLE_FRAME_ROWS;
use concinnity_engine::render::particles::{EmitterGpu, ParticleEmitterRecord, ParticleFrame};
use concinnity_engine::render::shader::programs::abi::particle::SIMULATE_GROUP;
use concinnity_engine::render::temporal::reactive_mask::ReactiveWrite;
use concinnity_engine::render::uniforms::{GpuParticle, ParticleView};

use super::allocator::PooledBuffer;
use super::context::{HDR_FORMAT, VkContext};
use super::descriptor_layout::{Binding, PoolSizes};
use super::pipeline_desc::{GraphicsPipelineDesc, compute_pipeline};
use super::reactive_mask;
use super::rendering::ImageUse;
use super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::set_writes::SetWrites;
use super::texture::GpuUploadContext;
use crate::vulkan::builtin_shaders::CompileProgram;
use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedPipeline, OwnedPipelineLayout, OwnedSampler, OwnedSetLayout, VkDevice,
};
use crate::vulkan::record::Recorder;

// The per-emitter descriptor pool reserves a fixed block of `2 * MAX_EMITTERS`
// sets at init (one compute set + one render set per emitter).
pub(in crate::vulkan) const MAX_EMITTERS: usize =
    concinnity_engine::render::particles::MAX_EMITTERS;

// `GpuParticle` (one simulation-pool slot) and `ParticleView` (the draw's
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

// Each emitter's GPU state, and the shared resources (pipelines, per-frame
// view UBO, descriptor pool), built on the first `add_emitter`,
// authored emitters included.
#[derive(Default)]
pub(in crate::vulkan) struct ParticleState {
    pub resources: Option<ParticleResources>,
    // Each emitter's pool and sets, by the id the scene's particle system
    // hands out.
    pub gpu: EmitterGpu<ParticleEmitterGpuState>,
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

    // The particle.hlsl billboard pair over the scene and the reactive mask.
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

    // Linear-clamp sampler shared by every emitter's albedo binding.
    pub(in crate::vulkan) sampler: OwnedSampler,

    // Whether the main depth is multisampled; picks the fragment variant.
    msaa: bool,
}

// Render-target inputs the particle pass samples from: the per-frame main
// depth views.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct ParticlePassTargets<'a> {
    pub(in crate::vulkan) depth_views: &'a [vk::ImageView],
}

impl ParticleResources {
    // Build the particle pipelines, view UBO ring, sampler and descriptor pool,
    // when the world declares an emitter.
    pub(in crate::vulkan) fn new(
        gpu: &GpuUploadContext<'_>,
        frames: usize,
        targets: ParticlePassTargets<'_>,
        msaa: bool,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let &GpuUploadContext { alloc, device, .. } = gpu;
        let depth_views = targets.depth_views;
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
            create_render_pipeline(device, render_pipeline_layout.handle(), vertex, fragment)
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

        Ok(Self {
            compute_pipeline,
            compute_pipeline_layout,
            compute_set_layout,
            _rows_set_layout: rows_set_layout,
            rows,
            rows_sets,
            render_pipeline,
            render_pipeline_lit,
            render_pipeline_layout,
            _view_set_layout: view_set_layout,
            emitter_set_layout,
            descriptor_pool,
            spare_sets: EmitterSetFreeList::default(),
            view_ubos,
            view_sets,
            sampler,
            msaa,
        })
    }

    // Re-point the per-frame depth binding after a swapchain resize. The
    // pipelines, layouts, buffers, sampler, and per-emitter descriptor sets all
    // survive.
    pub(in crate::vulkan) fn repoint_depth(
        &self,
        device: &VkDevice,
        targets: ParticlePassTargets<'_>,
    ) {
        for (i, &set) in self.view_sets.iter().enumerate() {
            SetWrites::new(set)
                .sampled_image(1, frame_view(targets.depth_views, i))
                .apply(device);
        }
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
    // `VkContext`'s `particle.gpu`; their destruction is the
    // caller's responsibility.
    pub(in crate::vulkan) fn destroy(&mut self, _device: &VkDevice) {
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

// Descriptor / pipeline construction

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

// Push-constant range covering the full `ParticleParams` block, so the upload
// is one push.
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
            device.writer_targets().formats(HDR_FORMAT),
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
    super::texture::one_shot_submit(device, command_pool, queue, |rec| {
        rec.fill_buffer(target, 0, bytes, 0);
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
        let frame = self.state.particles.prepare(elapsed, bursts)?;
        if let Some(rows) = resources.rows.get(frame_idx) {
            rows.write_slice(0, self.state.particles.rows());
        }
        if let Some(view) = resources.view_ubos.get(frame_idx) {
            let view_uni = ParticleView::new(
                vp,
                &self.state.view.matrix,
                self.targets.render_extent.height as f32,
                self.pre_exposure.value(),
                self.state.lights.uniforms(),
                &self.state.view.daylight,
            );
            view.write_val(0, &view_uni);
        }
        Some(frame)
    }

    // Encode `ParticlesSim` over each live emitter's pool, from the state
    // `prepare_particle_pass` computed. Takes `&self` so it can record on a worker;
    // the pool's hazards are the graph's.
    pub(in crate::vulkan) fn encode_particles_sim(
        &self,
        rec: &Recorder<'_>,
        frame_idx: usize,
        frame: ParticleFrame,
    ) {
        let Some(resources) = self.particle.resources.as_ref() else {
            return;
        };
        if self.state.particles.is_empty() {
            return;
        }

        // One dispatch per live emitter; resources are disjoint between
        // emitters so no inter-dispatch barrier is needed.
        rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &resources.compute_pipeline);
        let Some(&rows_set) = resources.rows_sets.get(frame_idx) else {
            return;
        };
        rec.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            &resources.compute_pipeline_layout,
            1,
            std::slice::from_ref(&rows_set),
            &[],
        );
        for emitter in self.state.particles.emitters(&frame) {
            let Some(gpu) = self.particle.gpu.get(emitter.id) else {
                continue;
            };
            let (record, params) = (emitter.record, &emitter.params);
            rec.bind_descriptor_sets(
                vk::PipelineBindPoint::COMPUTE,
                &resources.compute_pipeline_layout,
                0,
                std::slice::from_ref(&gpu.compute_set),
                &[],
            );
            rec.push_constants(
                &resources.compute_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                params,
            );
            rec.dispatch_threads([record.max_particles, 1, 1], SIMULATE_GROUP);
        }
    }

    // Encode `ParticlesDraw`: one alpha-blended quad per live particle into
    // `hdr_resolve`. The graph transitions the pool.
    pub(in crate::vulkan) fn encode_particles_draw(
        &self,
        rec: &Recorder<'_>,
        frame_idx: usize,
        frame: ParticleFrame,
        frustum: &Frustum,
        stage: ParticleStage,
    ) -> bool {
        let Some(resources) = self.particle.resources.as_ref() else {
            return false;
        };
        let ParticleStage { mask, late } = stage;
        let extent = self.targets.render_extent;

        // Visibility-cull per emitter, for the draw alone: the simulation ticked
        // every live pool so off-screen emitters stay in a realistic mid-life
        // state when the camera turns back. Tombstoned (None) slots are always
        // invisible.
        let mut visible = self
            .state
            .particles
            .visible(&frame, frustum, late)
            .peekable();
        if visible.peek().is_none() {
            return false;
        }

        // The early emitters blend into the resolved scene, the late ones into
        // the pre-TAA scene the see-through surfaces drew into.
        let scene = if late {
            self.post_scene_image(frame_idx)
        } else {
            &self.targets.hdr_resolve_images[frame_idx]
        };
        // Negative-height viewport, matching the main pass's +Y-up clip space.
        let viewport = vk::Viewport {
            x: 0.0,
            y: extent.height as f32,
            width: extent.width as f32,
            height: -(extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(extent);

        let pass = reactive_mask::WriterPass {
            scene,
            scene_from: ImageUse::Sampled,
            mask: self.targets.reactive_mask_images.get(frame_idx),
            write: mask,
            extent,
        };
        self.hw.device.writer_targets().render(rec, pass, |rec| {
            rec.set_viewport(&viewport);
            rec.set_scissor(&scissor);
            rec.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, &resources.render_pipeline);
            rec.bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                &resources.render_pipeline_layout,
                0,
                std::slice::from_ref(&resources.view_sets[frame_idx]),
                &[],
            );

            let mut lit = false;
            for emitter in visible {
                let Some(gpu) = self.particle.gpu.get(emitter.id) else {
                    continue;
                };
                let (record, params) = (emitter.record, &emitter.params);
                if record.lit != lit {
                    lit = record.lit;
                    let pipeline = if lit {
                        &resources.render_pipeline_lit
                    } else {
                        &resources.render_pipeline
                    };
                    rec.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline);
                }
                rec.bind_descriptor_sets(
                    vk::PipelineBindPoint::GRAPHICS,
                    &resources.render_pipeline_layout,
                    1,
                    std::slice::from_ref(&gpu.render_set),
                    &[],
                );
                rec.push_constants(
                    &resources.render_pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    params,
                );
                rec.draw(4, record.max_particles, 0, 0);
                self.inc_draw_calls(1);
            }
        });
        true
    }
}

// Runtime emitters (`RenderBackend::add_emitter` / `remove_emitter`)

impl VkContext {
    fn recycle_emitter_sets(&mut self, sets: EmitterSets) {
        if let Some(resources) = self.particle.resources.as_mut() {
            resources.spare_sets.recycle(sets);
        }
    }

    // Wire every world-authored particle emitter through `add_emitter`
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
            self.add_emitter(record)?;
        }
        Ok(())
    }

    // Whether any live emitter samples texture-pool `slot`. Emitter sets are
    // single-copy and always bound, so a swap of such a slot must drain the device
    // before rewriting.
    pub(in crate::vulkan) fn particle_samples_slot(&self, slot: usize) -> bool {
        let last = self.scene.textures.len().saturating_sub(1);
        self.particle
            .gpu
            .iter()
            .any(|state| state.texture_slot.min(last) == slot)
    }

    // Re-point every emitter's albedo binding that samples pool `slot` at the
    // swapped-in view, or it dangles.
    pub(in crate::vulkan) fn rewrite_particle_albedo_slot(&self, slot: usize) {
        if self.particle.resources.is_none() {
            return;
        }
        let last = self.scene.textures.len().saturating_sub(1);
        let view = self.scene.textures[slot].view;
        for state in self.particle.gpu.iter() {
            if state.texture_slot.min(last) == slot {
                write_render_albedo_binding(&self.hw.device, state.render_set, view);
            }
        }
    }
}

// Point an emitter's render set at albedo `view`. Its sampler was written with
// the set.
fn write_render_albedo_binding(device: &VkDevice, set: vk::DescriptorSet, view: vk::ImageView) {
    SetWrites::new(set).sampled_image(1, view).apply(device);
}

impl EmitterPass for VkContext {
    // Build the particle pipelines + per-frame uniform ring on first use, so a
    // world that never had an emitter pays no pipeline cost until the first
    // add.
    fn ensure_emitter_pass(&mut self) -> RenderResult<()> {
        if self.particle.resources.is_none() {
            let depth_views: Vec<vk::ImageView> = self
                .targets
                .depth_images
                .iter()
                .map(|img| img.view)
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
                    depth_views: &depth_views,
                },
                self.targets.msaa_samples != vk::SampleCountFlags::TYPE_1,
                self.hot_reload.enabled,
            )?;
            self.particle.resources = Some(resources);
        }
        Ok(())
    }

    // Allocate the emitter's pool, counter and sets, and point its albedo
    // binding at the live texture pool.
    fn create_emitter(&mut self, id: usize, record: &ParticleEmitterRecord) -> RenderResult<()> {
        let Some(resources) = self.particle.resources.as_mut() else {
            return Err(RenderError::Other(
                "add_emitter: particle pipeline unavailable".to_string(),
            ));
        };
        let gpu_state = build_emitter_gpu_state(
            GpuUploadContext {
                alloc: &self.hw.alloc,
                device: &self.hw.device,
                command_pool: self.commands.command_pool,
                queue: self.hw.graphics_queue,
            },
            resources,
            record,
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
        self.particle.gpu.set(id, state);
        Ok(())
    }

    // Drain the queue before freeing the pool: Vulkan has no driver-side
    // keep-alive for in-flight buffer references, so an in-flight command
    // buffer could otherwise dereference the freed memory or the recycled sets.
    fn destroy_emitter(&mut self, id: usize) {
        if let Some(state) = self.particle.gpu.take(id) {
            self.wait_idle();
            self.recycle_emitter_sets(state.sets());
            drop(state);
        }
    }
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
