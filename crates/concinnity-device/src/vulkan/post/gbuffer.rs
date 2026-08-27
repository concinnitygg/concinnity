// src/vulkan/post/gbuffer.rs
//
// Unified geometry G-buffer pre-pass for the Vulkan backend. One jittered
// traversal of the visible set (static + instanced + skinned) rasterises into a
// single MRT:
//
//   target 0  RGBA16F  view-space normal (rgb) + positive linear view depth (a)
//   target 1  R8       perceptual roughness
//   target 2  RG16F    screen-space motion (prev_uv - cur_uv)
//
// plus a private single-sample depth buffer. Every screen-space consumer (SSR
// resolve, SSAO, SSGI, TAA, FSR) reads this one output instead of
// re-rasterising, replacing the separate SSR pre-pass + SSAO pre-pass +
// velocity pre-pass. Rasterisation uses the jittered VP (matching the main pass
// coverage); the motion vector derives from the un-jittered current / previous
// VPs in-shader so projection jitter never contaminates motion. Fuses the
// former SSR depth+normal pre-pass and TAA velocity pre-pass into one node;
// mirrors src/directx/post/gbuffer.rs.
//
// Unlike DirectX's single-resource G-buffer, the Vulkan unified buffer holds a
// per-frame `Vec<GpuImage>` for every MRT target (and per-frame framebuffers),
// because TAA reads `velocity_images[frame_idx]` and the engine pipelines
// frames-in-flight deep; this follows the per-frame `Vec` shape of taa.rs.

use ash::vk;
use concinnity_core::gfx::transform::IDENTITY;

use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedFramebuffer, OwnedPipeline, OwnedPipelineLayout, OwnedRenderPass,
    OwnedSetLayout, VkDevice,
};

use crate::vulkan::uniforms::GBUFFER_PREPASS_PUSH_BYTES;
use crate::vulkan::uniforms::GbModelPush;
use concinnity_render::uniforms::GBufferView;

use super::super::allocator::{DeviceAllocator, PooledBuffer};
use super::super::context::VkContext;
use super::super::pipeline::*;
use super::super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::super::texture::*;

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

// `GBufferView` (the std140 `GbView` UBO) and `GbModelPush` (the pre-pass
// push constant), plus its `GBUFFER_PREPASS_PUSH_BYTES` size, are GPU-free
// layout structs that live in concinnity-render (imported above).

// SPIR-V blobs for every G-buffer pre-pass pipeline. Produced by
// [`compile_gbuffer_shaders`]; consumed by `GbufferResources::new` at init and
// by the hot-reload pass. Mirrors the matching SSR struct.
pub(in crate::vulkan) struct GbufferShaders {
    pub prepass_vs: Vec<u8>,
    pub prepass_instanced_vs: Vec<u8>,
    pub prepass_skinned_vs: Vec<u8>,
    pub prepass_fs: Vec<u8>,
}

// Compile every G-buffer pre-pass GLSL source. `hot_reload` routes each source
// resolve through the builtins' disk-first path.
pub(in crate::vulkan) fn compile_gbuffer_shaders(
    hot_reload: bool,
) -> Result<GbufferShaders, String> {
    use super::super::{builtins, slang_builtins};
    let ctx = builtins::Ctx::plain(hot_reload);
    Ok(GbufferShaders {
        prepass_vs: slang_builtins::GBUFFER_PREPASS_VERT.compile(&ctx)?,
        prepass_instanced_vs: slang_builtins::GBUFFER_PREPASS_VERT_INSTANCED.compile(&ctx)?,
        prepass_skinned_vs: slang_builtins::GBUFFER_PREPASS_VERT_SKINNED.compile(&ctx)?,
        prepass_fs: slang_builtins::GBUFFER_PREPASS_FRAG.compile(&ctx)?,
    })
}

// Pre-pass render pass: an RGBA16F normal+depth target, an R8 roughness target,
// and an RG16F velocity target, plus a private depth buffer. All colour
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

// Build a pre-pass pipeline. Three MRT colour targets (normal+depth, roughness,
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
    let vert_mod = spv_module(device, vert_spv)?;
    let frag_mod = spv_module(device, frag_spv)?;
    let entry = std::ffi::CString::new("main").unwrap();

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_mod.handle())
            .name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_mod.handle())
            .name(&entry),
    ];
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

// Vertex input for the static / instanced G-buffer pre-pass over the 56-byte
// `Vertex`. Declares only the attributes the pre-pass vertex shaders consume:
// position (0), normal (1), and the skybox-sentinel colour (3); the instanced
// variant slices `[..2]` (position + normal, model comes from the instance
// SSBO). Stride stays the full 56 bytes; tangent (2) + uv (4) are not fetched.
fn vertex_56_input() -> (
    [vk::VertexInputBindingDescription; 1],
    [vk::VertexInputAttributeDescription; 3],
) {
    let binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(56)
        .input_rate(vk::VertexInputRate::VERTEX);
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
    ];
    ([binding], attrs)
}

// Vertex input for the skinned G-buffer pre-pass over the 80-byte
// `SkinnedVertex`. `gbuffer_prepass_skinned.vert` consumes position (0), normal
// (1), and the skinning joints (5) + weights (6); tangent (2), colour (3), and
// uv (4) are omitted so the pipeline matches the shader interface. Stride stays
// 80.
fn skinned_vertex_input() -> (
    [vk::VertexInputBindingDescription; 1],
    [vk::VertexInputAttributeDescription; 4],
) {
    let binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(80)
        .input_rate(vk::VertexInputRate::VERTEX);
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
            .location(5)
            .format(vk::Format::R16G16B16A16_UINT)
            .offset(56),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(6)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(64),
    ];
    ([binding], attrs)
}

// Vertex input for the GPU-driven (bindless) G-buffer pre-pass: the current
// attributes the VS reads (position 0, normal 1, skybox-sentinel colour 3) on
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
// G-buffer render pass; the per-frame `prev_model` SSBOs supply the velocity
// history (instance region init-written, static + skinned rewritten each frame);
// the per-frame set 0 binds the G-buffer view UBO + that frame's prev_model SSBO,
// and set 1 reuses the bindless GpuObjectData set.
pub(in crate::vulkan) struct GbufferBindless {
    pub(in crate::vulkan) pipeline: OwnedPipeline,
    pub(in crate::vulkan) pipeline_layout: OwnedPipelineLayout,
    pub(in crate::vulkan) set_layout: OwnedSetLayout,
    pub(in crate::vulkan) sets: Vec<vk::DescriptorSet>,
    pub(in crate::vulkan) prev_model_buffers: Vec<PooledBuffer>,
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

// Scene sizing that dimensions the per-frame prev_model SSBOs. `instance_models`
// are the per-instance current transforms written once into the immutable
// instance region; `draw.n_objects` is the static prefix length that region starts
// after; `n_cull` is the total cull-record count (the SSBO stride); `frames` is
// the number of frames in flight.
pub(in crate::vulkan) struct GbufferBindlessScene<'a> {
    pub instance_models: &'a [[[f32; 4]; 4]],
    pub n_objects: usize,
    pub n_cull: usize,
    pub frames: usize,
}

// Build the GPU-driven G-buffer pre-pass pipeline + its per-frame previous-frame
// model SSBOs + descriptor sets. The previous-frame model buffers' instance
// region `[draw.n_objects, draw.n_objects + n_instances)` is written once here (immutable,
// camera-only motion); the static + skinned regions are rewritten each frame by
// `build_gbuffer_prev_models`. Set 0 = G-buffer view UBO + prev_model SSBO; set 1
// = the shared bindless GpuObjectData set (object id via gl_InstanceIndex).
pub(in crate::vulkan) fn build_gbuffer_bindless(
    ctx: GbufferDeviceCtx,
    descriptors: GbufferBindlessDescriptors,
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
    let GbufferBindlessScene {
        instance_models,
        n_objects,
        n_cull,
        frames,
    } = scene;

    let compile_ctx = builtins::Ctx::plain(hot_reload);
    let vs = super::super::slang_builtins::GBUFFER_BINDLESS_VERT.compile(&compile_ctx)?;
    let fs = super::super::slang_builtins::GBUFFER_BINDLESS_FRAG.compile(&compile_ctx)?;

    // Set 0: GbView UBO (binding 0) + prev_model SSBO (binding 1), both VERTEX.
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

    // Per-frame prev_model SSBOs (host-visible, persistently mapped), sized for
    // `n_cull` column-major `float4x4` records, parallel to the object buffer.
    let buf_size = (n_cull * std::mem::size_of::<[[f32; 4]; 4]>()) as u64;
    let mut prev_model_buffers = Vec::with_capacity(frames);
    for _ in 0..frames {
        let buf = alloc.create_buffer(
            buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        // Instance region: the instances' current models (immutable, camera-only
        // motion). Written once into every frame buffer after the static prefix;
        // the per-frame fill rewrites only the static + skinned regions.
        if !instance_models.is_empty() {
            let stride = std::mem::size_of::<[[f32; 4]; 4]>();
            buf.write_slice(n_objects * stride, instance_models);
        }
        prev_model_buffers.push(buf);
    }

    // One set 0 per frame: binding 0 = that frame's GbView UBO, binding 1 = that
    // frame's prev_model SSBO. Both buffers are stable for the world's lifetime,
    // so the sets are written once here.
    let set_layouts: Vec<_> = (0..frames).map(|_| set_layout.handle()).collect();
    let sets = alloc_descriptor_sets(device, descriptor_pool, &set_layouts)?;
    for (f, &set) in sets.iter().enumerate() {
        let view_info = vk::DescriptorBufferInfo::default()
            .buffer(gb.view_ubo_buffers[f].buffer())
            .offset(0)
            .range(GBUFFER_VIEW_UBO_SIZE);
        let pm_info = vk::DescriptorBufferInfo::default()
            .buffer(prev_model_buffers[f].buffer())
            .offset(0)
            .range(buf_size);
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
        ];
        // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every
        // set and resource it names belongs to this device.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    Ok(GbufferBindless {
        pipeline,
        pipeline_layout,
        set_layout,
        sets,
        prev_model_buffers,
    })
}

// One pooled colour channel for one frame in flight. The transient pool owns
// the image, its memory and its view; this is a borrowed record so the G-buffer
// can build framebuffers over them and hand views to readers. Field names match
// `GpuImage` so a consumer reading `.image` / `.view` does not care which it
// holds -- but it must NOT be destroyed here.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct PooledTarget {
    pub image: vk::Image,
    pub view: vk::ImageView,
}

// The pooled colour channels for every frame in flight, as the transient pool
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
// freed on `destroy`, EXCEPT the three pooled colour channels (see
// `PooledTarget`). Holds per-frame MRT targets / framebuffers because the
// velocity target is read per-frame-in-flight by TAA, mirroring taa.rs's `Vec`
// shape.
pub(in crate::vulkan) struct GbufferResources {
    // Render pass.
    pub(in crate::vulkan) prepass_render_pass: OwnedRenderPass,

    // Pre-pass pipelines (static always, instanced / skinned conditional).
    pub(in crate::vulkan) prepass_set_layout: OwnedSetLayout,
    pub(in crate::vulkan) prepass_layout_static: OwnedPipelineLayout,
    pub(in crate::vulkan) prepass_layout_instanced: Option<OwnedPipelineLayout>,
    pub(in crate::vulkan) prepass_layout_skinned: Option<OwnedPipelineLayout>,
    pub(in crate::vulkan) prepass_pso_static: OwnedPipeline,
    pub(in crate::vulkan) prepass_pso_instanced: Option<OwnedPipeline>,
    pub(in crate::vulkan) prepass_pso_skinned: Option<OwnedPipeline>,

    // Per-frame view UBO (jittered_vp + cur_vp + prev_vp + view_mat),
    // host-mapped + descriptor set.
    pub(in crate::vulkan) view_ubo_buffers: Vec<PooledBuffer>,
    pub(in crate::vulkan) prepass_sets: Vec<vk::DescriptorSet>,
    pub(in crate::vulkan) _descriptor_pool: OwnedDescriptorPool,

    // Per-frame MRT targets + private depth + framebuffers (rebuilt on resize).
    // One slot per frame in flight: TAA reads `velocity_images[frame_idx]`.
    //
    // The three colour channels are `PooledTarget`: the transient pool owns
    // their images, memory and views, so this struct only records the handles it
    // needs to build framebuffers and hand views to readers. The private depth
    // stays feature-owned (`GpuImage`, retired through the allocator on drop).
    pub(in crate::vulkan) normal_depth_images: Vec<PooledTarget>,
    pub(in crate::vulkan) roughness_images: Vec<PooledTarget>,
    pub(in crate::vulkan) velocity_images: Vec<PooledTarget>,
    pub(in crate::vulkan) depth_images: Vec<GpuImage>,
    pub(in crate::vulkan) framebuffers: Vec<OwnedFramebuffer>,

    // Previous-frame motion state, owned here so the velocity channel works for
    // any consumer (TAA or FSR) independent of whether engine-TAA is on.
    // `prev_view_proj` is last frame's un-jittered VP; `prev_models` is each
    // draw's previous transform. Both advance once per frame.
    pub(in crate::vulkan) prev_view_proj: [[f32; 4]; 4],
    pub(in crate::vulkan) prev_models: Vec<[[f32; 4]; 4]>,

    // True only under `cn debug`. Stored so the lazy
    // `ensure_skinned_gbuffer_pso` path and the shader hot-reload pass read
    // every GLSL source through the disk-first helper. Mirrors
    // `SsrResources::hot_reload`.
    pub(in crate::vulkan) hot_reload: bool,
}

// Replacement G-buffer pre-pass pipelines built by the hot-reload pass.
// Conditional variants are `Some` exactly when the corresponding
// `prepass_pso_*` is `Some` on the live resource.
pub(in crate::vulkan) struct RebuiltGbufferPipelines {
    pub prepass_static: OwnedPipeline,
    pub prepass_instanced: Option<OwnedPipeline>,
    pub prepass_skinned: Option<OwnedPipeline>,
}

// Rebuild every live G-buffer pre-pass pipeline from disk-resident GLSL source
// against the existing layouts + render pass. Same shape as
// [`rebuild_ssr_pipelines`].
pub(in crate::vulkan) fn rebuild_gbuffer_pipelines(
    device: &VkDevice,
    gbuffer: &GbufferResources,
    hot_reload: bool,
) -> Result<RebuiltGbufferPipelines, String> {
    let shaders = compile_gbuffer_shaders(hot_reload)?;
    let (vbindings, vattrs) = vertex_56_input();
    let prepass_static = create_prepass_pipeline(
        device,
        PrepassPipelineTargets {
            render_pass: gbuffer.prepass_render_pass.handle(),
            layout: gbuffer.prepass_layout_static.handle(),
        },
        PrepassPipelineShaders {
            vert_spv: &shaders.prepass_vs,
            frag_spv: &shaders.prepass_fs,
            bindings: &vbindings,
            attrs: &vattrs,
        },
    )?;
    let prepass_instanced = if let (Some(layout), Some(_)) = (
        gbuffer.prepass_layout_instanced.as_ref(),
        gbuffer.prepass_pso_instanced.as_ref(),
    ) {
        // Instanced pre-pass reads only position + normal (model comes from the
        // instance SSBO), so bind just those two attributes.
        Some(create_prepass_pipeline(
            device,
            PrepassPipelineTargets {
                render_pass: gbuffer.prepass_render_pass.handle(),
                layout: layout.handle(),
            },
            PrepassPipelineShaders {
                vert_spv: &shaders.prepass_instanced_vs,
                frag_spv: &shaders.prepass_fs,
                bindings: &vbindings,
                attrs: &vattrs[..2],
            },
        )?)
    } else {
        None
    };
    let prepass_skinned = if let (Some(layout), Some(_)) = (
        gbuffer.prepass_layout_skinned.as_ref(),
        gbuffer.prepass_pso_skinned.as_ref(),
    ) {
        let (sbindings, sattrs) = skinned_vertex_input();
        Some(create_prepass_pipeline(
            device,
            PrepassPipelineTargets {
                render_pass: gbuffer.prepass_render_pass.handle(),
                layout: layout.handle(),
            },
            PrepassPipelineShaders {
                vert_spv: &shaders.prepass_skinned_vs,
                frag_spv: &shaders.prepass_fs,
                bindings: &sbindings,
                attrs: &sattrs,
            },
        )?)
    } else {
        None
    };
    Ok(RebuiltGbufferPipelines {
        prepass_static,
        prepass_instanced,
        prepass_skinned,
    })
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

// The existing main-pass storage-buffer set layouts the pre-pass reuses: the
// per-instance model SSBO layout and the per-object joint-palette SSBO layout.
// Either is `None` when the world has no instanced / skinned geometry.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GbufferSsboLayouts {
    pub instance: Option<vk::DescriptorSetLayout>,
    pub skinned: Option<vk::DescriptorSetLayout>,
}

impl GbufferResources {
    // Build every G-buffer pre-pass resource. `ssbo_layouts` are the existing
    // per-instance / per-object joint storage-buffer layouts the main pass uses;
    // the pre-pass reuses those buffers directly.
    pub(in crate::vulkan) fn new(
        ctx: GbufferDeviceCtx,
        queue: GbufferQueueCtx,
        extent: GbufferExtent,
        ssbo_layouts: GbufferSsboLayouts,
        object_count: usize,
        hot_reload: bool,
        pooled: &GbufferPooled,
    ) -> Result<Self, String> {
        let GbufferDeviceCtx { alloc, device } = ctx;
        // Only the frame count is needed here (for the view-UBO ring); the
        // sized targets are built by `build_targets`, which takes the full
        // `extent` below and reads width/height itself.
        let GbufferExtent { frames, .. } = extent;
        let GbufferSsboLayouts {
            instance: instance_ssbo_set_layout,
            skinned: skinned_ssbo_set_layout,
        } = ssbo_layouts;
        let prepass_render_pass = create_prepass_render_pass(device)?;

        // Pre-pass set 0: GbView UBO. Set 1 (instance/joint SSBO) is supplied by
        // the caller from the existing main-pass / skinned pipeline so the
        // pre-pass reuses those buffers directly.
        let prepass_set_layout = create_descriptor_set_layout(
            device,
            &[(
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::ShaderStageFlags::VERTEX,
            )],
        )?;

        // Pipeline layouts. Both stages see the full prepass push block; the VS
        // reads cur/prev model, the FS only reads roughness.
        let prepass_push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(GBUFFER_PREPASS_PUSH_BYTES);
        let static_layouts = [prepass_set_layout.handle()];
        let prepass_layout_static = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&static_layouts)
                    .push_constant_ranges(std::slice::from_ref(&prepass_push)),
            )
            .map_err(|e| format!("gbuffer prepass static layout: {e}"))?;

        let prepass_layout_instanced = if let Some(isl) = instance_ssbo_set_layout {
            let layouts = [prepass_set_layout.handle(), isl];
            Some(
                device
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default()
                            .set_layouts(&layouts)
                            .push_constant_ranges(std::slice::from_ref(&prepass_push)),
                    )
                    .map_err(|e| format!("gbuffer prepass instanced layout: {e}"))?,
            )
        } else {
            None
        };

        let prepass_layout_skinned = if let Some(jsl) = skinned_ssbo_set_layout {
            // Set 1 = current joint palette, set 2 = previous joint palette.
            // Both reuse the single main-pass joint set layout.
            let layouts = [prepass_set_layout.handle(), jsl, jsl];
            Some(
                device
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default()
                            .set_layouts(&layouts)
                            .push_constant_ranges(std::slice::from_ref(&prepass_push)),
                    )
                    .map_err(|e| format!("gbuffer prepass skinned layout: {e}"))?,
            )
        } else {
            None
        };

        // Pipelines.
        let shaders = compile_gbuffer_shaders(hot_reload)?;
        let (vbindings, vattrs) = vertex_56_input();
        let prepass_pso_static = create_prepass_pipeline(
            device,
            PrepassPipelineTargets {
                render_pass: prepass_render_pass.handle(),
                layout: prepass_layout_static.handle(),
            },
            PrepassPipelineShaders {
                vert_spv: &shaders.prepass_vs,
                frag_spv: &shaders.prepass_fs,
                bindings: &vbindings,
                attrs: &vattrs,
            },
        )?;
        let prepass_pso_instanced = if let Some(layout) = prepass_layout_instanced.as_ref() {
            // Instanced pre-pass reads only position + normal (model comes from
            // the instance SSBO), so bind just those two attributes.
            Some(create_prepass_pipeline(
                device,
                PrepassPipelineTargets {
                    render_pass: prepass_render_pass.handle(),
                    layout: layout.handle(),
                },
                PrepassPipelineShaders {
                    vert_spv: &shaders.prepass_instanced_vs,
                    frag_spv: &shaders.prepass_fs,
                    bindings: &vbindings,
                    attrs: &vattrs[..2],
                },
            )?)
        } else {
            None
        };
        let prepass_pso_skinned = if let Some(layout) = prepass_layout_skinned.as_ref() {
            let (sbindings, sattrs) = skinned_vertex_input();
            Some(create_prepass_pipeline(
                device,
                PrepassPipelineTargets {
                    render_pass: prepass_render_pass.handle(),
                    layout: layout.handle(),
                },
                PrepassPipelineShaders {
                    vert_spv: &shaders.prepass_skinned_vs,
                    frag_spv: &shaders.prepass_fs,
                    bindings: &sbindings,
                    attrs: &sattrs,
                },
            )?)
        } else {
            None
        };

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

        // Descriptor pool: `frames` prepass sets (1 UBO each).
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(frames as u32)];
        let descriptor_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(frames as u32),
            )
            .map_err(|e| format!("gbuffer descriptor pool: {e}"))?;

        let prepass_layouts: Vec<_> = (0..frames).map(|_| prepass_set_layout.handle()).collect();
        let prepass_sets =
            alloc_descriptor_sets(device, descriptor_pool.handle(), &prepass_layouts)?;
        for (i, &set) in prepass_sets.iter().enumerate() {
            let buf_info = vk::DescriptorBufferInfo::default()
                .buffer(view_ubo_buffers[i].buffer())
                .offset(0)
                .range(GBUFFER_VIEW_UBO_SIZE);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_info));
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        }

        let mut me = Self {
            prepass_render_pass,
            prepass_set_layout,
            prepass_layout_static,
            prepass_layout_instanced,
            prepass_layout_skinned,
            prepass_pso_static,
            prepass_pso_instanced,
            prepass_pso_skinned,
            view_ubo_buffers,
            prepass_sets,
            _descriptor_pool: descriptor_pool,
            normal_depth_images: Vec::new(),
            roughness_images: Vec::new(),
            velocity_images: Vec::new(),
            depth_images: Vec::new(),
            framebuffers: Vec::new(),
            prev_view_proj: IDENTITY,
            prev_models: vec![IDENTITY; object_count],
            hot_reload,
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
            // The three colour channels come from the transient pool, which
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
        // The three colour channels are pool-owned: dropping these records frees
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

    // Build (or rebuild) the skinned G-buffer pre-pass pipeline lazily, once a
    // `SkinnedMesh` has been uploaded and the joint descriptor set layout
    // exists. Idempotent: re-calling replaces the existing pipeline.
    pub(in crate::vulkan) fn ensure_skinned_gbuffer_pso(
        &mut self,
        device: &VkDevice,
        joint_set_layout: vk::DescriptorSetLayout,
    ) -> Result<(), String> {
        if let Some(_p) = self.prepass_pso_skinned.take() {
            // SAFETY: the handle was created from this device and is destroyed exactly once; the
            // caller has already waited for the device to go idle, so no submission still
            // references it.
        }
        if let Some(_l) = self.prepass_layout_skinned.take() {
            // SAFETY: the handle was created from this device and is destroyed exactly once; the
            // caller has already waited for the device to go idle, so no submission still
            // references it.
        }
        let prepass_push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(GBUFFER_PREPASS_PUSH_BYTES);
        // Set 1 = current joint palette, set 2 = previous joint palette; the
        // skinned VS deforms both poses to emit a real deformation motion
        // vector. Both reuse the single main-pass joint set layout.
        let layouts = [
            self.prepass_set_layout.handle(),
            joint_set_layout,
            joint_set_layout,
        ];
        let layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(std::slice::from_ref(&prepass_push)),
            )
            .map_err(|e| format!("gbuffer prepass skinned layout: {e}"))?;
        use super::super::builtins;
        let compile_ctx = builtins::Ctx::plain(self.hot_reload);
        let sk_vs =
            super::super::slang_builtins::GBUFFER_PREPASS_VERT_SKINNED.compile(&compile_ctx)?;
        let prepass_fs =
            super::super::slang_builtins::GBUFFER_PREPASS_FRAG.compile(&compile_ctx)?;
        let (sbindings, sattrs) = skinned_vertex_input();
        let pso = create_prepass_pipeline(
            device,
            PrepassPipelineTargets {
                render_pass: self.prepass_render_pass.handle(),
                layout: layout.handle(),
            },
            PrepassPipelineShaders {
                vert_spv: &sk_vs,
                frag_spv: &prepass_fs,
                bindings: &sbindings,
                attrs: &sattrs,
            },
        )?;
        self.prepass_layout_skinned = Some(layout);
        self.prepass_pso_skinned = Some(pso);
        Ok(())
    }

    // Swap the freshly-built pipelines into the live resources. The caller has
    // already `device_wait_idle`'d so the old pipelines are not in flight.
    pub(in crate::vulkan) fn swap_pipelines(&mut self, rebuilt: RebuiltGbufferPipelines) {
        self.prepass_pso_static = rebuilt.prepass_static;
        self.prepass_pso_instanced = rebuilt.prepass_instanced;
        self.prepass_pso_skinned = rebuilt.prepass_skinned;
    }

    // Destroy every G-buffer pre-pass resource. The caller has already idled the
    // device.
    pub(in crate::vulkan) fn destroy(&mut self, device: &VkDevice) {
        self.destroy_targets(device);
    }
}

// Camera / view state the G-buffer pre-pass rasterises with. `jittered_vp` is
// the jittered VP that rasterises (matching the main pass); `cur_vp` is the
// un-jittered current VP the shader pairs with the previous VP for the motion
// vector; `cam_pos` + `frustum` drive LOD selection and instanced-cluster
// culling.
pub(in crate::vulkan) struct GbufferPrepassView<'a> {
    pub jittered_vp: [[f32; 4]; 4],
    pub cur_vp: [[f32; 4]; 4],
    pub cam_pos: [f32; 3],
    pub frustum: &'a crate::gfx::frustum::Frustum,
}

impl VkContext {
    // Encode the unified G-buffer pre-pass: one jittered traversal of the
    // visible set (static + GPU-instanced + skinned) into the per-frame
    // normal+depth / roughness / velocity MRT plus a private depth buffer. Runs
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
        visible: &[u32],
        velocity_active: bool,
    ) {
        let GbufferPrepassView {
            jittered_vp,
            cur_vp,
            cam_pos,
            frustum,
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

        // When the bindless GPU-cull path is active, the pre-pass is GPU-driven:
        // it reuses the main pass's per-frame indirect buffer (same camera frustum
        // + active LOD) with two `cmd_draw_indexed_indirect` draws (static +
        // instance prefix, then the skinned tail over the deformed VB) instead of
        // the CPU per-object loops, plus a legacy extra loop for streamed chunks /
        // runtime clones not in the cull records. A non-bindless world (custom
        // shader) keeps the legacy path below. Both write the same MRT.
        if self.cull.gbuffer_bindless_pipeline.is_some() && self.cull_count() > 0 {
            self.encode_gbuffer_prepass_gpu_driven(
                gb,
                cmd,
                frame_idx,
                visible,
                cam_pos,
                velocity_active,
            );
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe { device.cmd_end_render_pass(cmd) };
            return;
        }

        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.geometry.vertex_buffer.buffer()], &[0]);
            device.cmd_bind_index_buffer(
                cmd,
                self.geometry.index_buffer.buffer(),
                0,
                vk::IndexType::UINT32,
            );
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                gb.prepass_pso_static.handle(),
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                gb.prepass_layout_static.handle(),
                0,
                std::slice::from_ref(&gb.prepass_sets[frame_idx]),
                &[],
            );
        }

        // Static geometry: same visible set + LOD pick as the main pass so the
        // G-buffer covers exactly what main rasterised.
        let last_obj = self.draw.objects.len().saturating_sub(1);
        let skip_seethrough = self.mesh_glass_active();
        for &draw_idx in visible {
            let i = (draw_idx as usize).min(last_obj);
            let obj = match self.draw.objects.get(i) {
                Some(o) => o,
                None => continue,
            };
            if !obj.visible || !obj.resident {
                continue;
            }
            // See-through glass meshes (Layer 2) are rerouted to the transparent
            // pass while RT is live, so they must not stamp depth here either --
            // the refraction tap reads the scene they would otherwise occlude.
            if skip_seethrough && obj.material.see_through != 0 {
                continue;
            }
            let d = crate::gfx::lod::camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            let prev_model = if velocity_active {
                gb.prev_models.get(i).copied().unwrap_or(obj.model)
            } else {
                obj.model
            };
            let push = GbModelPush {
                cur_model: obj.model,
                prev_model,
                roughness: obj.material.roughness,
                _pad: [0.0; 3],
            };
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                device.cmd_push_constants(
                    cmd,
                    gb.prepass_layout_static.handle(),
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    std::slice::from_raw_parts(
                        &push as *const GbModelPush as *const u8,
                        std::mem::size_of::<GbModelPush>(),
                    ),
                );
                device.cmd_draw_indexed(
                    cmd,
                    index_count as u32,
                    1,
                    index_offset as u32,
                    obj.base_vertex,
                    0,
                );
            }
        }

        // GPU-instanced clusters: instance transforms never change, so the
        // motion is camera-only (the instanced VS feeds the same matrix to cur
        // and prev clip). Reuses the per-cluster instance SSBO the main
        // instanced pass already filled this frame.
        if let (Some(inst_pso), Some(inst_layout)) = (
            gb.prepass_pso_instanced.as_ref(),
            gb.prepass_layout_instanced.as_ref(),
        ) && !self.instanced.clusters.is_empty()
            && !self.instanced.sets.is_empty()
        {
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, inst_pso.handle());
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    inst_layout.handle(),
                    0,
                    std::slice::from_ref(&gb.prepass_sets[frame_idx]),
                    &[],
                );
            }
            for (cluster_idx, cluster) in self.instanced.clusters.iter().enumerate() {
                if cluster.instances.is_empty() {
                    continue;
                }
                if cluster.cullable() {
                    if !frustum.intersects_aabb(cluster.cluster_bb_min, cluster.cluster_bb_max) {
                        continue;
                    }
                    if cluster.cull_distance > 0.0 {
                        let d2 = crate::gfx::frustum::aabb_distance_sq(
                            cam_pos,
                            cluster.cluster_bb_min,
                            cluster.cluster_bb_max,
                        );
                        if d2 > cluster.cull_distance * cluster.cull_distance {
                            continue;
                        }
                    }
                }
                let Some(buckets) = self.instanced.lod_buckets.get(cluster_idx) else {
                    continue;
                };
                let inst_set = self.instanced.sets[frame_idx][cluster_idx];
                let push = GbModelPush {
                    cur_model: [[0.0; 4]; 4],  // ignored by instanced VS
                    prev_model: [[0.0; 4]; 4], // ignored by instanced VS
                    roughness: cluster.material.roughness,
                    _pad: [0.0; 3],
                };
                // SAFETY: `cmd` is a command buffer in the recording state, and every handle and
                // slice these commands name is live for the call.
                unsafe {
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        inst_layout.handle(),
                        1,
                        std::slice::from_ref(&inst_set),
                        &[],
                    );
                    device.cmd_push_constants(
                        cmd,
                        inst_layout.handle(),
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        std::slice::from_raw_parts(
                            &push as *const GbModelPush as *const u8,
                            std::mem::size_of::<GbModelPush>(),
                        ),
                    );
                    // One draw per LOD bucket, matching the Main pass partition
                    // so the G-buffer stays pixel-aligned with the scene.
                    let mut first_instance: u32 = 0;
                    for bucket in buckets {
                        let count = bucket.instances.len() as u32;
                        device.cmd_draw_indexed(
                            cmd,
                            bucket.index_count as u32,
                            count,
                            bucket.index_offset as u32,
                            0,
                            first_instance,
                        );
                        first_instance += count;
                    }
                }
            }
        }

        // Skinned meshes: drawn last so the G-buffer reflects animated
        // characters too. The current palette (set 1) and the previous-frame
        // palette (set 2) deform the two poses so per-vertex skinned motion
        // produces a correct motion vector. The model matrix is static (skinned
        // meshes are self-placing), so cur and prev model are usually identical;
        // it is threaded through identically to the static path. The previous
        // palette lives at the prior slot of the joint-set ring; with fewer than
        // two frames in flight there is no distinct prior slot, so prev = cur
        // (it cannot ghost without a second in-flight frame anyway).
        if let (Some(sk_pso), Some(sk_layout)) = (
            gb.prepass_pso_skinned.as_ref(),
            gb.prepass_layout_skinned.as_ref(),
        ) && !self.skinned.draw_objects.is_empty()
        {
            let frames = self.frames_in_flight.max(1);
            let prev_frame_idx = if velocity_active && frames >= 2 {
                (frame_idx + frames - 1) % frames
            } else {
                frame_idx
            };
            let (sk_vbuf, sk_ibuf) = self.skinned_geometry();
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, sk_pso.handle());
                device.cmd_bind_vertex_buffers(cmd, 0, std::slice::from_ref(&sk_vbuf), &[0]);
                device.cmd_bind_index_buffer(cmd, sk_ibuf, 0, vk::IndexType::UINT32);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    sk_layout.handle(),
                    0,
                    std::slice::from_ref(&gb.prepass_sets[frame_idx]),
                    &[],
                );
            }
            for (i, obj) in self.skinned.draw_objects.iter().enumerate() {
                if !obj.visible {
                    continue;
                }
                let d = crate::gfx::lod::skinned_camera_distance(obj, cam_pos);
                let (index_offset, index_count) = obj.active_lod(d);
                // Skinned meshes are self-placing, so cur and prev model are
                // identical; the deformation motion comes from the current vs
                // previous joint palettes bound at sets 1 / 2. Threaded through
                // identically to the static path. `prev_models` is keyed by
                // static draw-object index, so it is not consulted here.
                let push = GbModelPush {
                    cur_model: obj.model,
                    prev_model: obj.model,
                    roughness: obj.material.roughness,
                    _pad: [0.0; 3],
                };
                // SAFETY: `cmd` is a command buffer in the recording state, and every handle and
                // slice these commands name is live for the call.
                unsafe {
                    // Set 1 = current palette, set 2 = previous-frame palette.
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        sk_layout.handle(),
                        1,
                        std::slice::from_ref(&self.skinned.joint_sets[frame_idx][i]),
                        &[],
                    );
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        sk_layout.handle(),
                        2,
                        std::slice::from_ref(&self.skinned.joint_sets[prev_frame_idx][i]),
                        &[],
                    );
                    device.cmd_push_constants(
                        cmd,
                        sk_layout.handle(),
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        std::slice::from_raw_parts(
                            &push as *const GbModelPush as *const u8,
                            std::mem::size_of::<GbModelPush>(),
                        ),
                    );
                    device.cmd_draw_indexed(cmd, index_count as u32, 1, index_offset as u32, 0, 0);
                }
            }
            // Restore the static vertex/index buffers for any later pass that
            // does not rebind them itself.
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                device.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    &[self.geometry.vertex_buffer.buffer()],
                    &[0],
                );
                device.cmd_bind_index_buffer(
                    cmd,
                    self.geometry.index_buffer.buffer(),
                    0,
                    vk::IndexType::UINT32,
                );
            }
        }

        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe { device.cmd_end_render_pass(cmd) };
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
    // SSBO. Streamed chunks / runtime clones keep a legacy per-object loop.
    fn encode_gbuffer_prepass_gpu_driven(
        &self,
        gb: &GbufferResources,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        visible: &[u32],
        cam_pos: [f32; 3],
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

        // Build this frame's previous-frame model buffer (static + skinned regions;
        // the instance region is init-written + immutable). Honours velocity_active.
        self.build_gbuffer_prev_models(gb, frame_idx, velocity_active);

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
        // matching what the colour pass will draw.
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

        // Legacy extra: streamed chunks + runtime clones (records past `draw.n_objects`)
        // are not in the GpuObjectData buffer, so draw them with the legacy
        // per-object pipeline into the same MRT.
        self.encode_gbuffer_legacy_extra(gb, cmd, frame_idx, visible, cam_pos, velocity_active);
    }

    // Legacy per-object G-buffer draws for runtime clones past the bindless range
    // (`i >= draw.n_objects` AND in `clone.slot_by_draw_idx`). Streamed VoxelWorld chunks
    // now fold into the GPU-driven cull records (drawn by the prefix indirect draw),
    // so they are skipped here. Mirrors the legacy static loop, appending into the
    // same MRT after the indirect draws. A no-op for worlds with no clones.
    fn encode_gbuffer_legacy_extra(
        &self,
        gb: &GbufferResources,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        visible: &[u32],
        cam_pos: [f32; 3],
        velocity_active: bool,
    ) {
        if self.clone.slot_by_draw_idx.is_empty() {
            return;
        }
        let device = &self.device;
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                gb.prepass_pso_static.handle(),
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                gb.prepass_layout_static.handle(),
                0,
                std::slice::from_ref(&gb.prepass_sets[frame_idx]),
                &[],
            );
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.geometry.vertex_buffer.buffer()], &[0]);
            device.cmd_bind_index_buffer(
                cmd,
                self.geometry.index_buffer.buffer(),
                0,
                vk::IndexType::UINT32,
            );
        }
        let skip_seethrough = self.mesh_glass_active();
        for &draw_idx in visible {
            let i = draw_idx as usize;
            if i < self.draw.n_objects {
                continue; // build-time object, already drawn via indirect
            }
            if !self.clone.slot_by_draw_idx.contains_key(&i) {
                continue; // streamed chunk -> folded into the cull records
            }
            let Some(obj) = self.draw.objects.get(i) else {
                continue;
            };
            if !obj.visible || !obj.resident {
                continue;
            }
            // See-through glass meshes (Layer 2) are rerouted to the transparent
            // pass while RT is live, so they must not stamp depth here either --
            // the refraction tap reads the scene they would otherwise occlude.
            if skip_seethrough && obj.material.see_through != 0 {
                continue;
            }
            let d = crate::gfx::lod::camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            let prev_model = if velocity_active {
                gb.prev_models.get(i).copied().unwrap_or(obj.model)
            } else {
                obj.model
            };
            let push = GbModelPush {
                cur_model: obj.model,
                prev_model,
                roughness: obj.material.roughness,
                _pad: [0.0; 3],
            };
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                device.cmd_push_constants(
                    cmd,
                    gb.prepass_layout_static.handle(),
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    std::slice::from_raw_parts(
                        &push as *const GbModelPush as *const u8,
                        std::mem::size_of::<GbModelPush>(),
                    ),
                );
                device.cmd_draw_indexed(
                    cmd,
                    index_count as u32,
                    1,
                    index_offset as u32,
                    obj.base_vertex,
                    0,
                );
            }
        }
    }

    // Fill this frame's previous-frame model SSBO for the GPU-driven G-buffer
    // velocity. Indexed by cull record id, parallel to the GpuObjectData buffer:
    // the static prefix `[0, draw.n_objects)` gets last frame's model (so a moving
    // static object reprojects correctly), the skinned tail gets the current model
    // (skinned deformation motion comes from the previous-frame deformed buffer,
    // not the model matrix). The instance region is init-written + immutable
    // (camera-only motion), so it is left untouched. When velocity is inactive
    // every written record gets its current model, so the motion stays zero (GbView
    // prev_vp also equals cur_vp). Mirrors build_object_buffer's record indexing.
    fn build_gbuffer_prev_models(
        &self,
        gb: &GbufferResources,
        frame_idx: usize,
        velocity_active: bool,
    ) {
        let Some(buf) = self.cull.prev_model_buffers.get(frame_idx) else {
            return;
        };
        let stride = std::mem::size_of::<[[f32; 4]; 4]>();
        for (i, obj) in self
            .draw
            .objects
            .iter()
            .take(self.draw.n_objects)
            .enumerate()
        {
            let prev = if velocity_active {
                gb.prev_models.get(i).copied().unwrap_or(obj.model)
            } else {
                obj.model
            };
            buf.write_val(i * stride, &prev);
        }
        // Streamed chunks: current model -> camera-only velocity (chunk terrain is
        // static-in-world; the unused reserve slots keep stale prev_models but their
        // draw-args are disabled, so the gbuffer never rasterises them).
        let chunk_base = self.chunk_record_base();
        self.for_each_chunk_record(|k, obj| {
            buf.write_val((chunk_base + k) * stride, &obj.model);
        });
        let base = self.skinned_record_base();
        for (k, obj) in self
            .skinned
            .draw_objects
            .iter()
            .take(self.draw.n_skinned)
            .enumerate()
        {
            // Skinned motion is per-vertex (previous deformed buffer), so the model
            // matrix is the current one (cur == prev model, like the legacy path).
            buf.write_val((base + k) * stride, &obj.model);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The `GBufferView` / `GbModelPush` layout tests live with the structs in
    // `concinnity_render::vulkan::uniforms`. `GBufferView` fitting the
    // `GBUFFER_VIEW_UBO_SIZE` allocation is checked here, where the size const
    // (typed `vk::DeviceSize`) lives.
    #[test]
    fn gb_view_uniforms_fits_ubo_allocation() {
        assert!(std::mem::size_of::<GBufferView>() as u64 <= GBUFFER_VIEW_UBO_SIZE);
    }

    // Every G-buffer pre-pass GLSL (static + instanced + skinned vertex shaders
    // and the shared fragment) compiles to SPIR-V. Exercises the fused
    // ssr_prepass + velocity contract: the vertex shaders emit cur_clip /
    // prev_clip the fragment consumes for the motion vector.
    #[test]
    fn gbuffer_shaders_compile() {
        compile_gbuffer_shaders(false).expect("gbuffer shaders compile");
    }
}
