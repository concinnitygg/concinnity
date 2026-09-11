// src/vulkan/post/gbuffer.rs
//
// Unified geometry G-buffer pre-pass for the Vulkan backend. One jittered
// traversal of the visible set (static + instanced + skinned) rasterizes into a
// single MRT:
//
//   target 0  RGBA16F  view-space normal (rgb) + positive linear view depth (a)
//   target 1  R8       perceptual roughness
//   target 2  RG16F    screen-space motion (prev_uv - cur_uv)
//
// plus a private single-sample depth buffer. Every screen-space consumer (SSR
// resolve, SSAO, SSGI, TAA, FSR) reads this one output instead of
// re-rasterizing, replacing the separate SSR pre-pass + SSAO pre-pass +
// velocity pre-pass. Rasterization uses the jittered VP (matching the main pass
// coverage); the motion vector derives from the un-jittered current / previous
// VPs in-shader so projection jitter never contaminates motion. Fuses the
// former SSR depth+normal pre-pass and TAA velocity pre-pass into one node;
// mirrors src/directx/post/gbuffer.rs.
//
// Unlike DirectX's single-resource G-buffer, the Vulkan unified buffer holds a
// per-frame `Vec<GpuImage>` for every MRT target (and per-frame framebuffers),
// because the temporal resolve reads `velocity_images[frame_idx]` and the engine
// pipelines frames-in-flight deep.

use ash::vk;
use concinnity_core::gfx::render_types::{GpuDrawArgs, GpuObjectData};
use concinnity_core::gfx::transform::IDENTITY;
use concinnity_core::render::uniforms::{GBufferView, ModelHistoryParams};

use super::super::allocator::{DeviceAllocator, PooledBuffer};
use super::super::context::VkContext;
use super::super::pipeline::*;
use super::super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::super::texture::*;
use crate::vulkan::owned::{
    OwnedFramebuffer, OwnedPipeline, OwnedPipelineLayout, OwnedRenderPass, OwnedSetLayout, VkDevice,
};
use crate::vulkan::slang_builtins::SlangCompile;

// Threads per group, matching `[numthreads(64, 1, 1)]` in model_history.slang.
const MODEL_HISTORY_THREADGROUP: usize = 64;

// Normal+depth target: rgb = unit view-space normal, a = positive linear view
// depth (-view_z). Alpha 0 (cleared background) marks "no geometry". Matches
// the SSR G-buffer so the resolve maths is byte-identical.
pub(in crate::vulkan) const GBUFFER_NORMAL_DEPTH_FORMAT: vk::Format =
    vk::Format::R16G16B16A16_SFLOAT;

// Single-channel perceptual roughness. 1.0 (cleared background) = no reflection;
// 0.0 = mirror.
pub(in crate::vulkan) const GBUFFER_ROUGHNESS_FORMAT: vk::Format = vk::Format::R8_UNORM;

// Screen-space motion (prev_uv - cur_uv). Cleared to 0 (no motion).
pub(in crate::vulkan) const GBUFFER_VELOCITY_FORMAT: vk::Format = vk::Format::R16G16_SFLOAT;

// Size of the per-frame view UBO: jittered_vp + cur_vp + prev_vp + view_mat
// (four std140 mat4 = 256 B). Matches the `GbView` UBO in every pre-pass VS.
pub(in crate::vulkan) const GBUFFER_VIEW_UBO_SIZE: vk::DeviceSize = 256;

// `GBufferView` (the std140 `GbView` UBO) is a GPU-free layout struct that
// lives in `core::render` (imported above).

// Pre-pass render pass: an RGBA16F normal+depth target, an R8 roughness target,
// and an RG16F velocity target, plus a private depth buffer. All color
// attachments clear and end shader-readable so the consumers can sample them
// without an extra barrier. The depth is STORE'd because the temporal upscaler
// (FSR) consumes this render-resolution single-sample depth alongside the
// motion vectors.
fn create_prepass_render_pass(device: &VkDevice) -> Result<OwnedRenderPass, String> {
    let attachments = [
        vk::AttachmentDescription::default()
            .format(GBUFFER_NORMAL_DEPTH_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        vk::AttachmentDescription::default()
            .format(GBUFFER_ROUGHNESS_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        vk::AttachmentDescription::default()
            .format(GBUFFER_VELOCITY_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];
    let color_refs = [
        vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        vk::AttachmentReference::default()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
    ];
    let depth_ref = vk::AttachmentReference::default()
        .attachment(3)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs)
        .depth_stencil_attachment(&depth_ref);
    let dep = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .src_access_mask(vk::AccessFlags::SHADER_READ)
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );
    let info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&dep));
    device
        .create_render_pass(&info)
        .map_err(|e| format!("gbuffer prepass render pass: {e}"))
}

// Render pass + pipeline layout a pre-pass pipeline binds against.
#[derive(Clone, Copy)]
struct PrepassPipelineTargets {
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
}

// The compiled SPIR-V + vertex input layout a pre-pass pipeline is built from.
struct PrepassPipelineShaders<'a> {
    vert_spv: &'a [u8],
    frag_spv: &'a [u8],
    bindings: &'a [vk::VertexInputBindingDescription],
    attrs: &'a [vk::VertexInputAttributeDescription],
}

// Build a pre-pass pipeline. Three MRT color targets (normal+depth, roughness,
// velocity) over a private depth buffer; same no-cull / LESS depth as the main
// pass.
fn create_prepass_pipeline(
    device: &VkDevice,
    targets: PrepassPipelineTargets,
    shaders: PrepassPipelineShaders,
) -> Result<OwnedPipeline, String> {
    let PrepassPipelineTargets {
        render_pass,
        layout,
    } = targets;
    let PrepassPipelineShaders {
        vert_spv,
        frag_spv,
        bindings,
        attrs,
    } = shaders;
    let modules = GraphicsStages::new(device, vert_spv, frag_spv)?;
    let stages = modules.infos();
    let vert_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(bindings)
        .vertex_attribute_descriptions(attrs);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS);
    // All three attachments must be byte-identical without `independentBlend`
    // enabled at device creation. The R8 roughness target stores only R, so a
    // uniform RGBA write-mask is the smallest-diff way to satisfy the spec.
    let blend_attaches = [
        vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false),
        vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false),
        vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false),
    ];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attaches);
    let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);

    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vert_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);
    let pipeline = crate::vulkan::pipeline_cache::create_graphics_pipeline(device, &info)
        .map_err(|e| format!("create gbuffer prepass pso: {e}"))?;
    Ok(pipeline)
}

// Vertex input for the GPU-driven (bindless) G-buffer pre-pass: the current
// attributes the VS reads (position 0, normal 1, skybox-sentinel color 3) on
// binding 0, plus the previous-frame position (location 5) on binding 1. Both
// bindings carry the 56-byte `Vertex`; the static prefix binds the static VB to
// both (prev_pos == cur_pos), the skinned tail binds the current deformed buffer
// to binding 0 and the previous-frame deformed buffer to binding 1. Tangent + UV
// are unused (the pre-pass samples no textures).
fn vertex_56_dual_input() -> (
    [vk::VertexInputBindingDescription; 2],
    [vk::VertexInputAttributeDescription; 4],
) {
    let bindings = [
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(56)
            .input_rate(vk::VertexInputRate::VERTEX),
        vk::VertexInputBindingDescription::default()
            .binding(1)
            .stride(56)
            .input_rate(vk::VertexInputRate::VERTEX),
    ];
    let attrs = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(12),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(3)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(36),
        vk::VertexInputAttributeDescription::default()
            .binding(1)
            .location(5)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
    ];
    (bindings, attrs)
}

// GPU-driven G-buffer pre-pass resources, built when the bindless cull path is
// active AND the G-buffer is enabled. Stored on `VkCull`. The pipeline reuses the
// G-buffer render pass; the per-frame model-history SSBOs supply the velocity
// history, filled on the GPU by the snapshot kernel below; the per-frame set 0
// binds the G-buffer view UBO, the PREVIOUS frame's history slot and this
// frame's draw args, and set 1 reuses the bindless GpuObjectData set.
pub(in crate::vulkan) struct GbufferBindless {
    pub(in crate::vulkan) pipeline: OwnedPipeline,
    pub(in crate::vulkan) pipeline_layout: OwnedPipelineLayout,
    pub(in crate::vulkan) set_layout: OwnedSetLayout,
    pub(in crate::vulkan) sets: Vec<vk::DescriptorSet>,
    pub(in crate::vulkan) prev_model_buffers: Vec<PooledBuffer>,
    pub(in crate::vulkan) history: ModelHistoryPipeline,
}

// The model-history snapshot kernel: one thread per cull record copying this
// frame's model matrix out of the object buffer into a history slot, which a
// later frame's pre-pass reprojects through. `sets` is a square table indexed
// `[frame * frames + slot]`, binding the record-count UBO, that frame's object
// buffer and that slot's history buffer: a normal frame takes the diagonal, and
// the frame that primes a rebuilt ring walks its row to fill every slot at once.
// Every buffer is stable for the world's lifetime, so the sets are written once
// at build.
pub(in crate::vulkan) struct ModelHistoryPipeline {
    pub(in crate::vulkan) pipeline: OwnedPipeline,
    pub(in crate::vulkan) pipeline_layout: OwnedPipelineLayout,
    // Owners only: the sets above are written once at build and the params are
    // a build-time constant, but both must outlive every frame that binds them.
    pub(in crate::vulkan) _set_layout: OwnedSetLayout,
    pub(in crate::vulkan) sets: Vec<vk::DescriptorSet>,
    // Set by the draw-args build when the ring was rebuilt, consumed by the
    // dispatch. An atomic rather than a borrow of the tracker: passes encode on
    // worker threads, and this is the one piece of that state the encode reads.
    pub(in crate::vulkan) prime: std::sync::atomic::AtomicBool,
    pub(in crate::vulkan) _params: PooledBuffer,
}

// Vulkan device handles every G-buffer builder threads through: the instance,
// logical device, and physical device used to allocate images / buffers.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GbufferDeviceCtx<'a> {
    pub alloc: &'a DeviceAllocator,
    pub device: &'a VkDevice,
}

// Descriptor wiring the GPU-driven pre-pass allocates against: the shared pool
// its per-frame set 0 comes from and the bindless GpuObjectData set layout it
// reuses as set 1.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GbufferBindlessDescriptors {
    pub descriptor_pool: vk::DescriptorPool,
    pub bindless_set_layout: vk::DescriptorSetLayout,
}

// The per-frame record buffers the pre-pass and its snapshot kernel read: the
// object buffer the snapshot copies models out of, and the draw args the
// pre-pass reads `NO_HISTORY` from.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GbufferBindlessRecords<'a> {
    pub object_buffers: &'a [PooledBuffer],
    pub draw_args_buffers: &'a [PooledBuffer],
}

// Scene sizing that dimensions the per-frame model-history SSBOs: `n_cull` is
// the cull-record count they hold one matrix each for, and `frames` the number
// of frames in flight the ring is deep.
pub(in crate::vulkan) struct GbufferBindlessScene {
    pub n_cull: usize,
    pub frames: usize,
}

// Build the GPU-driven G-buffer pre-pass pipeline, the model-history ring it
// reprojects through, the snapshot kernel that fills that ring, and the
// descriptor sets for both. Set 0 = G-buffer view UBO + the previous frame's
// history slot + this frame's draw args; set 1 = the shared bindless
// GpuObjectData set (object id via gl_InstanceIndex).
pub(in crate::vulkan) fn build_gbuffer_bindless(
    ctx: GbufferDeviceCtx,
    descriptors: GbufferBindlessDescriptors,
    records: GbufferBindlessRecords,
    gb: &GbufferResources,
    scene: GbufferBindlessScene,
    hot_reload: bool,
) -> Result<GbufferBindless, String> {
    use super::super::builtins;

    let GbufferDeviceCtx { alloc, device } = ctx;
    let GbufferBindlessDescriptors {
        descriptor_pool,
        bindless_set_layout,
    } = descriptors;
    let GbufferBindlessScene { n_cull, frames } = scene;
    let GbufferBindlessRecords {
        object_buffers,
        draw_args_buffers,
    } = records;

    let compile_ctx = builtins::Ctx::plain(hot_reload);
    let vs = super::super::slang_builtins::GBUFFER_BINDLESS_VERT.compile(&compile_ctx)?;
    let fs = super::super::slang_builtins::GBUFFER_BINDLESS_FRAG.compile(&compile_ctx)?;

    // Set 0: GbView UBO (binding 0), the previous frame's model-history slot
    // (binding 1) and this frame's draw args (binding 2), all VERTEX.
    let set_layout = create_descriptor_set_layout(
        device,
        &[
            (
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::ShaderStageFlags::VERTEX,
            ),
            (
                1,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::VERTEX,
            ),
            (
                2,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::VERTEX,
            ),
        ],
    )?;
    let layouts = [set_layout.handle(), bindless_set_layout];
    let pipeline_layout = device
        .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts))
        .map_err(|e| format!("gbuffer bindless pipeline layout: {e}"))?;

    let (bindings, attrs) = vertex_56_dual_input();
    let pipeline = create_prepass_pipeline(
        device,
        PrepassPipelineTargets {
            render_pass: gb.prepass_render_pass.handle(),
            layout: pipeline_layout.handle(),
        },
        PrepassPipelineShaders {
            vert_spv: &vs,
            frag_spv: &fs,
            bindings: &bindings,
            attrs: &attrs,
        },
    )?;

    // Per-frame model-history SSBOs, sized for `n_cull` column-major `float4x4`
    // records, parallel to the object buffer. Device-local: only the snapshot
    // kernel writes them and only the pre-pass reads them, so the host never
    // touches their bytes.
    let buf_size = (n_cull * std::mem::size_of::<[[f32; 4]; 4]>()) as u64;
    let mut prev_model_buffers = Vec::with_capacity(frames);
    for _ in 0..frames {
        prev_model_buffers.push(alloc.create_buffer(
            buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?);
    }

    // One set 0 per frame: binding 0 = that frame's GbView UBO, binding 1 = the
    // history slot the PREVIOUS frame filled, binding 2 = that frame's draw
    // args. The frame index cycles, so the previous slot is a fixed offset and
    // every set can be written once here.
    let draw_args_size = (n_cull * std::mem::size_of::<GpuDrawArgs>()) as u64;
    let set_layouts: Vec<_> = (0..frames).map(|_| set_layout.handle()).collect();
    let sets = alloc_descriptor_sets(device, descriptor_pool, &set_layouts)?;
    for (f, &set) in sets.iter().enumerate() {
        let view_info = vk::DescriptorBufferInfo::default()
            .buffer(gb.view_ubo_buffers[f].buffer())
            .offset(0)
            .range(GBUFFER_VIEW_UBO_SIZE);
        let pm_info = vk::DescriptorBufferInfo::default()
            .buffer(prev_model_buffers[(f + frames - 1) % frames].buffer())
            .offset(0)
            .range(buf_size);
        let da_info = vk::DescriptorBufferInfo::default()
            .buffer(draw_args_buffers[f].buffer())
            .offset(0)
            .range(draw_args_size);
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&view_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&pm_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&da_info)),
        ];
        // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every
        // set and resource it names belongs to this device.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    let history = build_model_history(
        ctx,
        descriptor_pool,
        &prev_model_buffers,
        object_buffers,
        ModelHistoryScene { n_cull, frames },
        hot_reload,
    )?;

    Ok(GbufferBindless {
        pipeline,
        pipeline_layout,
        set_layout,
        sets,
        prev_model_buffers,
        history,
    })
}

// Sizing for the snapshot kernel's per-frame sets.
#[derive(Clone, Copy)]
struct ModelHistoryScene {
    n_cull: usize,
    frames: usize,
}

// Build the model-history snapshot kernel: set 0 binds the record-count UBO at
// binding 0, the frame's object buffer at 1 and the frame's history slot at 2,
// which is the declaration order `model_history.slang` fixes.
fn build_model_history(
    ctx: GbufferDeviceCtx,
    descriptor_pool: vk::DescriptorPool,
    history_buffers: &[PooledBuffer],
    object_buffers: &[PooledBuffer],
    scene: ModelHistoryScene,
    hot_reload: bool,
) -> Result<ModelHistoryPipeline, String> {
    let GbufferDeviceCtx { alloc, device } = ctx;
    let ModelHistoryScene { n_cull, frames } = scene;
    let compile_ctx = super::super::builtins::Ctx::plain(hot_reload);
    let cs = super::super::slang_builtins::MODEL_HISTORY.compile(&compile_ctx)?;

    let set_layout = create_descriptor_set_layout(
        device,
        &[
            (
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
            ),
            (
                1,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
            ),
            (
                2,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
            ),
        ],
    )?;
    let layouts = [set_layout.handle()];
    let pipeline_layout = device
        .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts))
        .map_err(|e| format!("model history pipeline layout: {e}"))?;
    let pipeline = create_cull_pipeline(device, pipeline_layout.handle(), &cs)?;

    // The record count never moves for a built world, so one host-visible UBO
    // serves every frame's set.
    let params = ModelHistoryParams {
        record_count: n_cull as u32,
        _pad: [0; 3],
    };
    let params_size = std::mem::size_of::<ModelHistoryParams>() as u64;
    let params_buf = alloc.create_buffer(
        params_size,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    params_buf.write_val(0, &params);

    let object_size = (n_cull * std::mem::size_of::<GpuObjectData>()) as u64;
    let history_size = (n_cull * std::mem::size_of::<[[f32; 4]; 4]>()) as u64;
    let set_layouts: Vec<_> = (0..frames * frames).map(|_| set_layout.handle()).collect();
    let sets = alloc_descriptor_sets(device, descriptor_pool, &set_layouts)?;
    for (i, &set) in sets.iter().enumerate() {
        let (f, slot) = (i / frames, i % frames);
        let p_info = vk::DescriptorBufferInfo::default()
            .buffer(params_buf.buffer())
            .offset(0)
            .range(params_size);
        let o_info = vk::DescriptorBufferInfo::default()
            .buffer(object_buffers[f].buffer())
            .offset(0)
            .range(object_size);
        let h_info = vk::DescriptorBufferInfo::default()
            .buffer(history_buffers[slot].buffer())
            .offset(0)
            .range(history_size);
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&p_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&o_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&h_info)),
        ];
        // SAFETY: `writes` and the buffer infos it borrows are live for the call, and every set
        // and resource it names belongs to this device.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    Ok(ModelHistoryPipeline {
        pipeline,
        pipeline_layout,
        _set_layout: set_layout,
        sets,
        prime: std::sync::atomic::AtomicBool::new(false),
        _params: params_buf,
    })
}

// One pooled color channel for one frame in flight. The transient pool owns
// the image, its memory and its view; this is a borrowed record so the G-buffer
// can build framebuffers over them and hand views to readers. Field names match
// `GpuImage` so a consumer reading `.image` / `.view` does not care which it
// holds -- but it must NOT be destroyed here.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct PooledTarget {
    pub image: vk::Image,
    pub view: vk::ImageView,
}

// The pooled color channels for every frame in flight, as the transient pool
// hands them over. Each `Vec` has one entry per frame; the caller builds this
// from `pairs_for_frames` right after the pool is built or rebuilt, and passing
// a stale one is what a use-after-free would look like.
#[derive(Clone, Default)]
pub(in crate::vulkan) struct GbufferPooled {
    pub normal_depth: Vec<PooledTarget>,
    pub roughness: Vec<PooledTarget>,
    pub velocity: Vec<PooledTarget>,
}

// Unified G-buffer pre-pass resources held by `VkContext` when any screen-space
// consumer is enabled. Every `vk::*` handle here is owned by this struct and
// freed on `destroy`, EXCEPT the three pooled color channels (see
// `PooledTarget`). Holds per-frame MRT targets / framebuffers because the
// velocity target is read per frame in flight by the temporal resolve.
pub(in crate::vulkan) struct GbufferResources {
    // Render pass.
    pub(in crate::vulkan) prepass_render_pass: OwnedRenderPass,

    // Per-frame view UBO (jittered_vp + cur_vp + prev_vp + view_mat),
    // host-mapped. The pre-pass pipeline's own set 0 points at it.
    pub(in crate::vulkan) view_ubo_buffers: Vec<PooledBuffer>,

    // Per-frame MRT targets + private depth + framebuffers (rebuilt on resize).
    // One slot per frame in flight: TAA reads `velocity_images[frame_idx]`.
    //
    // The three color channels are `PooledTarget`: the transient pool owns
    // their images, memory and views, so this struct only records the handles it
    // needs to build framebuffers and hand views to readers. The private depth
    // stays feature-owned (`GpuImage`, retired through the allocator on drop).
    pub(in crate::vulkan) normal_depth_images: Vec<PooledTarget>,
    pub(in crate::vulkan) roughness_images: Vec<PooledTarget>,
    pub(in crate::vulkan) velocity_images: Vec<PooledTarget>,
    pub(in crate::vulkan) depth_images: Vec<GpuImage>,
    pub(in crate::vulkan) framebuffers: Vec<OwnedFramebuffer>,

    // Last frame's un-jittered VP, owned here so the velocity channel works for
    // any consumer (TAA or FSR) independent of whether engine-TAA is on. The
    // per-object half of the same history is the GPU-filled model-history ring.
    pub(in crate::vulkan) prev_view_proj: [[f32; 4]; 4],
}

// Command pool + queue the target builders use to lay out the private depth
// image (its layout transition is submitted on this queue).
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GbufferQueueCtx {
    pub command_pool: vk::CommandPool,
    pub queue: vk::Queue,
}

// Render-resolution extent + frame-in-flight count the per-frame MRT targets are
// sized and multiplied by.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GbufferExtent {
    pub width: u32,
    pub height: u32,
    pub frames: usize,
}

impl GbufferResources {
    // Build every G-buffer pre-pass resource. The pipeline that rasterizes into
    // them is built separately by [`build_gbuffer_bindless`].
    pub(in crate::vulkan) fn new(
        ctx: GbufferDeviceCtx,
        queue: GbufferQueueCtx,
        extent: GbufferExtent,
        pooled: &GbufferPooled,
    ) -> Result<Self, String> {
        let GbufferDeviceCtx { alloc, device } = ctx;
        // Only the frame count is needed here (for the view-UBO ring); the
        // sized targets are built by `build_targets`, which takes the full
        // `extent` below and reads width/height itself.
        let GbufferExtent { frames, .. } = extent;
        let prepass_render_pass = create_prepass_render_pass(device)?;

        // Per-frame view UBO (jittered_vp + cur_vp + prev_vp + view_mat).
        let mut view_ubo_buffers = Vec::with_capacity(frames);
        for _ in 0..frames {
            let buf = alloc.create_buffer(
                GBUFFER_VIEW_UBO_SIZE,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            view_ubo_buffers.push(buf);
        }

        let mut me = Self {
            prepass_render_pass,
            view_ubo_buffers,
            normal_depth_images: Vec::new(),
            roughness_images: Vec::new(),
            velocity_images: Vec::new(),
            depth_images: Vec::new(),
            framebuffers: Vec::new(),
            prev_view_proj: IDENTITY,
        };
        me.build_targets(ctx, queue, extent, pooled)?;
        Ok(me)
    }

    // Allocate the per-frame MRT targets + private depth + framebuffers at the
    // given extent. One slot per frame in flight.
    fn build_targets(
        &mut self,
        ctx: GbufferDeviceCtx,
        queue: GbufferQueueCtx,
        extent: GbufferExtent,
        pooled: &GbufferPooled,
    ) -> Result<(), String> {
        let GbufferDeviceCtx { alloc, device } = ctx;
        let GbufferQueueCtx {
            command_pool,
            queue,
        } = queue;
        let GbufferExtent {
            width,
            height,
            frames,
        } = extent;
        let w = width.max(1);
        let h = height.max(1);
        for f in 0..frames {
            // The three color channels come from the transient pool, which
            // holds one image per (label, frame) exactly as this loop expects.
            let normal_depth = *pooled
                .normal_depth
                .get(f)
                .ok_or("gbuffer: pooled normal_depth slot out of range")?;
            let roughness = *pooled
                .roughness
                .get(f)
                .ok_or("gbuffer: pooled roughness slot out of range")?;
            let velocity = *pooled
                .velocity
                .get(f)
                .ok_or("gbuffer: pooled velocity slot out of range")?;
            let depth = create_depth_image(
                &GpuUploadContext {
                    alloc,
                    device,
                    command_pool,
                    queue,
                },
                w,
                h,
                vk::SampleCountFlags::TYPE_1,
            )?;
            let attachments = [normal_depth.view, roughness.view, velocity.view, depth.view];
            let framebuffer = device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(self.prepass_render_pass.handle())
                        .attachments(&attachments)
                        .width(w)
                        .height(h)
                        .layers(1),
                )
                .map_err(|e| format!("gbuffer prepass framebuffer: {e}"))?;
            self.normal_depth_images.push(normal_depth);
            self.roughness_images.push(roughness);
            self.velocity_images.push(velocity);
            self.depth_images.push(depth);
            self.framebuffers.push(framebuffer);
        }
        // The sampled channels rest in `SHADER_READ_ONLY_OPTIMAL`, where the
        // pre-pass render pass leaves them, so a consumer that samples one before
        // the pre-pass has ever run binds a valid layout (the Composite samples
        // normal+depth and roughness unconditionally for the debug view modes, but
        // a world hidden behind an opaque menu masks the pre-pass off). The pool
        // already puts every image it allocates in that layout, so there is
        // nothing to do here now that these three are pooled.
        Ok(())
    }

    // The per-frame normal+depth view a reader (SSR resolve, SSAO, SSGI) binds.
    pub(in crate::vulkan) fn normal_depth_view(&self, frame: usize) -> vk::ImageView {
        self.normal_depth_images[frame].view
    }

    // The per-frame roughness view a reader binds.
    pub(in crate::vulkan) fn roughness_view(&self, frame: usize) -> vk::ImageView {
        self.roughness_images[frame].view
    }

    // The per-frame velocity view the TAA resolve / FSR binds.
    pub(in crate::vulkan) fn velocity_view(&self, frame: usize) -> vk::ImageView {
        self.velocity_images[frame].view
    }

    // Per-frame normal+depth views, one per frame in flight. The readers that
    // bind a per-frame descriptor set (SSR resolve, SSAO kernel/blur, SSGI, RT)
    // slice this so each set samples its own frame's unified G-buffer.
    pub(in crate::vulkan) fn normal_depth_views(&self) -> Vec<vk::ImageView> {
        (0..self.normal_depth_images.len())
            .map(|f| self.normal_depth_view(f))
            .collect()
    }

    // Per-frame roughness views, one per frame in flight.
    pub(in crate::vulkan) fn roughness_views(&self) -> Vec<vk::ImageView> {
        (0..self.roughness_images.len())
            .map(|f| self.roughness_view(f))
            .collect()
    }

    // Per-frame velocity views, one per frame in flight. The TAA resolve binds
    // its frame's slot; FSR reads `velocity_images[frame]` directly.
    pub(in crate::vulkan) fn velocity_views(&self) -> Vec<vk::ImageView> {
        (0..self.velocity_images.len())
            .map(|f| self.velocity_view(f))
            .collect()
    }

    fn destroy_targets(&mut self, _device: &VkDevice) {
        self.framebuffers.clear();
        // The three color channels are pool-owned: dropping these records frees
        // nothing, which is the point. The private depth images retire through
        // the allocator as they drop.
        self.normal_depth_images.clear();
        self.roughness_images.clear();
        self.velocity_images.clear();
        self.depth_images.clear();
    }

    // Rebuild the per-frame targets at a new swapchain extent. The caller has
    // already idled the device and rebuilt the transient pool, so `pooled` names
    // the new images; the framebuffers built here reference them, which is why
    // this must run after every pool rebuild and not only after a resize. The
    // descriptor sets and UBOs are resolution-independent and untouched.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        ctx: GbufferDeviceCtx,
        queue: GbufferQueueCtx,
        extent: GbufferExtent,
        pooled: &GbufferPooled,
    ) -> Result<(), String> {
        self.destroy_targets(ctx.device);
        self.build_targets(ctx, queue, extent, pooled)?;
        Ok(())
    }

    // Destroy every G-buffer pre-pass resource. The caller has already idled the
    // device.
    pub(in crate::vulkan) fn destroy(&mut self, device: &VkDevice) {
        self.destroy_targets(device);
    }
}

// Camera / view state the G-buffer pre-pass rasterizes with. `jittered_vp` is
// the jittered VP that rasterizes (matching the main pass); `cur_vp` is the
// un-jittered current VP the shader pairs with the previous VP for the motion
// vector.
pub(in crate::vulkan) struct GbufferPrepassView {
    pub jittered_vp: [[f32; 4]; 4],
    pub cur_vp: [[f32; 4]; 4],
}

impl VkContext {
    // Encode the unified G-buffer pre-pass: one jittered traversal of the cull
    // records into the per-frame normal+depth / roughness / velocity MRT plus a
    // private depth buffer. Runs
    // before the main pass. `velocity_active` is true when a consumer (TAA or
    // FSR) reads motion; when false, prev == cur so the motion channel is a
    // harmless zero. Fuses the former SSR depth+normal and TAA velocity
    // pre-passes.
    //
    // `gb` is borrowed from the owning `self.gbuffer` field by the caller,
    // matching how the SSR / TAA encoders take their resources.
    pub(in crate::vulkan) fn encode_gbuffer_prepass(
        &self,
        gb: &GbufferResources,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        view: GbufferPrepassView,
        velocity_active: bool,
    ) {
        let GbufferPrepassView {
            jittered_vp,
            cur_vp,
        } = view;
        let device = &self.device;
        let extent = self.render_extent;

        // Upload this frame's view UBO. When velocity is inactive the previous
        // VP equals the current one, so instanced + sky motion is zero.
        let prev_vp = if velocity_active {
            gb.prev_view_proj
        } else {
            cur_vp
        };
        let view_uni = GBufferView {
            jittered_vp,
            cur_vp,
            prev_vp,
            view: self.view.matrix,
        };
        gb.view_ubo_buffers[frame_idx].write_val(0, &view_uni);

        // Clears: alpha-0 normal+depth = "no geometry"; roughness 1.0 = no SSR;
        // velocity 0 = no motion.
        let clears = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [1.0, 0.0, 0.0, 0.0],
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0; 4] },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(gb.prepass_render_pass.handle())
            .framebuffer(gb.framebuffers[frame_idx].handle())
            .render_area(vk::Rect2D::default().extent(extent))
            .clear_values(&clears);
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe { device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE) };

        // Negative-height viewport: matches the main pass so the G-buffer lines
        // up with the main pass at pixel coordinates; the fragment shader's
        // upright-UV math expects this orientation.
        let vp = vk::Viewport {
            x: 0.0,
            y: extent.height as f32,
            width: extent.width as f32,
            height: -(extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(extent);
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
        }

        // The pre-pass is GPU-driven: it reuses the main pass's per-frame indirect
        // buffer (same camera frustum + active LOD) with two
        // `cmd_draw_indexed_indirect` draws (static + instance prefix, then the
        // skinned tail over the deformed VB); streamed chunks and runtime clones
        // ride the cull records' runtime reserve. With nothing to draw the pass
        // is the clears above, which is what "no geometry" means to every reader.
        self.encode_gbuffer_prepass_gpu_driven(cmd, frame_idx, velocity_active);

        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe { device.cmd_end_render_pass(cmd) };

        // Snapshot this frame's models into this frame's history slot, AFTER the
        // pass above read the previous one -- which is what keeps a single frame
        // in flight (one slot, read then rewritten) correct.
        self.encode_model_history(cmd, frame_idx);
    }

    // Dispatch the model-history snapshot: one thread per cull record copying
    // `objects[i].model` into this frame's history slot. The slot is the one a
    // pre-pass `frames_in_flight - 1` frames ago also read, and that frame may
    // still be in flight, so the write is fenced behind its vertex reads; the
    // matching release makes it visible to the next frame's pre-pass.
    fn encode_model_history(&self, cmd: vk::CommandBuffer, frame_idx: usize) {
        let Some(history) = self.cull.model_history.as_ref() else {
            return;
        };
        let records = self.cull_count();
        if records == 0 {
            return;
        }
        // A rebuilt ring holds nothing these records were written for, so the
        // priming frame fills every slot rather than only its own: the instance
        // region is the one the draw args cannot flag, being init-written.
        let frames = self.cull.prev_model_buffers.len();
        let slots = match history
            .prime
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            true => 0..frames,
            false => frame_idx..frame_idx + 1,
        };
        let device = &self.device;
        let groups = records.div_ceil(MODEL_HISTORY_THREADGROUP) as u32;
        for slot in slots {
            let Some(&set) = history.sets.get(frame_idx * frames + slot) else {
                continue;
            };
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                self.model_history_barrier(
                    cmd,
                    slot,
                    (
                        vk::AccessFlags::SHADER_READ,
                        vk::AccessFlags::SHADER_WRITE,
                        vk::PipelineStageFlags::VERTEX_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                    ),
                );
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    history.pipeline.handle(),
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    history.pipeline_layout.handle(),
                    0,
                    &[set],
                    &[],
                );
                device.cmd_dispatch(cmd, groups, 1, 1);
                self.model_history_barrier(
                    cmd,
                    slot,
                    (
                        vk::AccessFlags::SHADER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::VERTEX_SHADER,
                    ),
                );
            }
        }
    }

    // One access/stage dependency over a whole model-history slot.
    //
    // SAFETY: the caller must pass a command buffer in the recording state.
    unsafe fn model_history_barrier(
        &self,
        cmd: vk::CommandBuffer,
        slot: usize,
        deps: (
            vk::AccessFlags,
            vk::AccessFlags,
            vk::PipelineStageFlags,
            vk::PipelineStageFlags,
        ),
    ) {
        let Some(buf) = self.cull.prev_model_buffers.get(slot) else {
            return;
        };
        let (src_access, dst_access, src_stage, dst_stage) = deps;
        let barrier = vk::BufferMemoryBarrier::default()
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(buf.buffer())
            .offset(0)
            .size(vk::WHOLE_SIZE)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access);
        // SAFETY: the caller guarantees `cmd` is recording, and the barrier and the buffer it
        // names are live for the call and belong to this device.
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&barrier),
                &[],
            )
        };
    }

    // GPU-driven G-buffer pre-pass raster (inside the render pass the caller
    // began). Reuses the main pass's per-frame indirect buffer (the camera-frustum
    // cull already produced it, so no extra cull dispatch) with two indirect draws:
    // the static + instance prefix `[0, skinned_record_base())` over the static VB
    // (bound to BOTH vertex bindings, so prev_pos == cur_pos and the motion is the
    // per-object model delta plus camera), then the skinned tail over the current
    // deformed VB (binding 0) + the previous-frame deformed VB (binding 1) for
    // per-vertex deformation motion. model + roughness ride the per-frame
    // GpuObjectData SSBO (gl_InstanceIndex); the previous-frame model a parallel
    // SSBO. The CPU never walks the draw lists.
    fn encode_gbuffer_prepass_gpu_driven(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        velocity_active: bool,
    ) {
        let device = &self.device;
        let (Some(pipeline), Some(layout)) = (
            self.cull.gbuffer_bindless_pipeline.as_ref(),
            self.cull.gbuffer_bindless_pipeline_layout.as_ref(),
        ) else {
            return;
        };
        let Some(indirect) = self
            .cull
            .indirect_buffers
            .get(frame_idx)
            .map(|b| b.buffer())
        else {
            return;
        };
        let Some(&gset) = self.cull.gbuffer_sets.get(frame_idx) else {
            return;
        };
        let stride = std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32;
        let prefix = self.skinned_record_base() as u32;

        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.handle());
            // set 0 = GbView UBO + prev_model SSBO; set 1 = bindless GpuObjectData.
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout.handle(),
                0,
                &[gset, self.cull.bindless_sets[frame_idx]],
                &[],
            );

            // Static + instance prefix: the static VB bound to BOTH vertex bindings
            // (prev_pos == cur_pos) + the static u32 IB.
            device.cmd_bind_vertex_buffers(
                cmd,
                0,
                &[
                    self.geometry.vertex_buffer.buffer(),
                    self.geometry.vertex_buffer.buffer(),
                ],
                &[0, 0],
            );
            device.cmd_bind_index_buffer(
                cmd,
                self.geometry.index_buffer.buffer(),
                0,
                vk::IndexType::UINT32,
            );
            if prefix > 0 {
                device.cmd_draw_indexed_indirect(cmd, indirect, 0, prefix, stride);
                self.inc_draw_calls(1);
            }
        }
        // The material-referenced shader buckets write their own regions of the
        // command buffer. The pre-pass shades nothing, so every bucket runs under
        // this single pipeline; a bucket whose Shader is not resident is skipped,
        // matching what the color pass will draw.
        if prefix > 0 {
            self.inc_draw_calls(self.draw_bucket_regions_shared_pipeline(cmd, indirect, prefix));
        }

        // Skinned tail: the current deformed VB (binding 0) + the previous-frame
        // deformed VB (binding 1) + the skinned IB. Records carry base_vertex
        // = 0 (global skinned indexing). The previous deformed buffer is read only
        // once the ring is primed (a prior frame posed that slot); before then (or
        // when velocity is inactive) it is the current buffer, so prev_pos ==
        // cur_pos gives a harmless zero skinned motion vector.
        if self.draw.n_skinned > 0
            && let Some(cur) = self.skinned.deformed.get(frame_idx)
        {
            let frames = self.frames_in_flight.max(1);
            let use_prev = velocity_active
                && frames >= 2
                && self
                    .skinned
                    .deformed_primed
                    .load(std::sync::atomic::Ordering::Relaxed);
            let prev_idx = if use_prev {
                (frame_idx + frames - 1) % frames
            } else {
                frame_idx
            };
            let prev = self.skinned.deformed.get(prev_idx).unwrap_or(cur);
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                device.cmd_bind_vertex_buffers(cmd, 0, &[cur.buffer, prev.buffer], &[0, 0]);
                device.cmd_bind_index_buffer(
                    cmd,
                    self.skinned.index_buffer.buffer(),
                    0,
                    vk::IndexType::UINT32,
                );
                device.cmd_draw_indexed_indirect(
                    cmd,
                    indirect,
                    (self.skinned_record_base() * stride as usize) as u64,
                    self.draw.n_skinned as u32,
                    stride,
                );
            }
            self.inc_draw_calls(1);
            // The current deformed buffer is posed this frame, so next frame's
            // history slot (this slot) is valid -- prime the ring.
            self.skinned
                .deformed_primed
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The `GBufferView` layout test lives with the struct in
    // `concinnity_core::render::vulkan::uniforms`. `GBufferView` fitting the
    // `GBUFFER_VIEW_UBO_SIZE` allocation is checked here, where the size const
    // (typed `vk::DeviceSize`) lives.
    #[test]
    fn gb_view_uniforms_fits_ubo_allocation() {
        assert!(std::mem::size_of::<GBufferView>() as u64 <= GBUFFER_VIEW_UBO_SIZE);
    }

    // The GPU-driven pre-pass pair compiles to SPIR-V. Exercises the fused
    // ssr_prepass + velocity contract: the vertex shader emits cur_clip /
    // prev_clip the fragment consumes for the motion vector.
    #[test]
    fn gbuffer_shaders_compile() {
        if !concinnity_slang::shader_tests_enabled() {
            return;
        }
        let ctx = super::super::super::builtins::Ctx::plain(false);
        super::super::super::slang_builtins::GBUFFER_BINDLESS_VERT
            .compile(&ctx)
            .expect("gbuffer bindless vertex compiles");
        super::super::super::slang_builtins::GBUFFER_BINDLESS_FRAG
            .compile(&ctx)
            .expect("gbuffer bindless fragment compiles");
    }
}
