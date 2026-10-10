//! Unified geometry G-buffer pre-pass for the Vulkan backend. One jittered
//! traversal of the visible set (static + instanced + skinned) rasterizes into a
//! single MRT:
//!
//!   target 0  RGBA16F  view-space normal (rgb) + positive linear view depth (a)
//!   target 1  R8       perceptual roughness
//!   target 2  RG16F    screen-space motion (prev_uv - cur_uv)
//!
//! plus a private single-sample depth buffer. Every screen-space consumer (SSR
//! resolve, SSAO, SSGI, TAA, FSR) reads this one output instead of
//! re-rasterizing, replacing the separate SSR pre-pass + SSAO pre-pass +
//! velocity pre-pass. Rasterization uses the jittered VP (matching the main pass
//! coverage); the motion vector derives from the un-jittered current / previous
//! VPs in-shader so projection jitter never contaminates motion. Fuses the
//! former SSR depth+normal pre-pass and TAA velocity pre-pass into one node;
//! mirrors src/directx/post/gbuffer.rs.
//!
//! Unlike DirectX's single-resource G-buffer, the Vulkan unified buffer holds a
//! per-frame `Vec<GpuImage>` for every MRT target (and per-frame framebuffers),
//! because the temporal resolve reads `velocity_images[frame_idx]` and the engine
//! pipelines frames-in-flight deep.

use ash::vk;
use concinnity_core::gfx::render_types::{GpuDrawArgs, GpuObjectData};
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::uniforms::{GBufferView, ModelHistoryParams};
use concinnity_core::render::view_history::{ViewFrame, ViewHistory};

use super::super::allocator::{DeviceAllocator, PooledBuffer};
use super::super::context::VkContext;
use super::super::descriptor_layout::Binding;
use super::super::pipeline::{MAIN_VERTEX_ATTRS, MeshPipelineTargets, spirv_words};
use super::super::pipeline_desc::{Blend, Depth, GraphicsPipelineDesc, compute_pipeline};
use super::super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::super::set_writes::SetWrites;
use super::super::spirv_inputs::input_locations;
use super::super::texture::*;
use super::gbuffer_sky::GbufferSky;
use crate::vulkan::builtin_shaders::CompileProgram;
use crate::vulkan::depth;
use crate::vulkan::owned::{
    OwnedFramebuffer, OwnedPipeline, OwnedPipelineLayout, OwnedRenderPass, OwnedSetLayout, VkDevice,
};

// Threads per group, matching `[numthreads(64, 1, 1)]` in model_history.hlsl.
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

// Size of the per-frame view UBO: jittered_vp + cur_vp + prev_vp + view_mat and
// the previous clock. Matches the `GbView` UBO in `gbuffer_common.hlsl`.
pub(in crate::vulkan) const GBUFFER_VIEW_UBO_SIZE: vk::DeviceSize =
    std::mem::size_of::<GBufferView>() as vk::DeviceSize;

// `GBufferView` (the std140 `GbView` UBO) is a GPU-free layout struct that
// lives in `core::render` (imported above).

// Pre-pass render pass: an RGBA16F normal+depth target, an R8 roughness target,
// and an RG16F velocity target, plus a private depth buffer. All color
// attachments clear and end shader-readable so the consumers can sample them
// without an extra barrier. The depth is STORE'd because the temporal upscaler
// (FSR) consumes this render-resolution single-sample depth alongside the
// motion vectors.
pub(in crate::vulkan) fn create_prepass_render_pass(
    device: &VkDevice,
) -> RenderResult<OwnedRenderPass> {
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
        .map_err(|e| crate::vulkan::error::map_vk_result(e, "gbuffer prepass render pass"))
}

// The pre-pass's three MRT targets: normal+depth, roughness, velocity. All three
// must be byte-identical without `independentBlend` enabled at device creation;
// the R8 roughness target stores only R under the uniform RGBA write mask.
pub(in crate::vulkan) const PREPASS_TARGETS: [Blend; 3] = [Blend::Opaque; 3];

// Vertex input for the G-buffer pre-pass: the main pass's attributes on binding
// 0, which the vertex hook reads in full, plus the previous-frame position
// (location 5) on binding 1. Both bindings carry the 56-byte `Vertex`; the
// static prefix binds the static VB to both (prev_pos == cur_pos), the skinned
// tail binds the current deformed buffer to binding 0 and the previous-frame
// deformed buffer to binding 1.
const VERTEX_56_DUAL_BINDINGS: [vk::VertexInputBindingDescription; 2] = [
    vk::VertexInputBindingDescription {
        binding: 0,
        stride: 56,
        input_rate: vk::VertexInputRate::VERTEX,
    },
    vk::VertexInputBindingDescription {
        binding: 1,
        stride: 56,
        input_rate: vk::VertexInputRate::VERTEX,
    },
];
const PREPASS_VERTEX_ATTRIBUTES: [vk::VertexInputAttributeDescription; 6] = [
    MAIN_VERTEX_ATTRS[0],
    MAIN_VERTEX_ATTRS[1],
    MAIN_VERTEX_ATTRS[2],
    MAIN_VERTEX_ATTRS[3],
    MAIN_VERTEX_ATTRS[4],
    vk::VertexInputAttributeDescription {
        location: 5,
        binding: 1,
        format: vk::Format::R32G32B32_SFLOAT,
        offset: 0,
    },
];

// The layout every shader bucket's G-buffer pre-pass pipeline binds: the main
// pass's global and bindless sets at 0 and 1 and the pre-pass's own view
// block, model history and draw args at set 2. Built with the bindless main
// pass, so a quality change that adds a G-buffer consumer can build the
// pre-pass sets and pipelines against it; the pipelines themselves take the
// G-buffer's render pass.
pub(in crate::vulkan) struct PrepassLayout {
    pub(in crate::vulkan) set_layout: OwnedSetLayout,
    pub(in crate::vulkan) pipeline_layout: OwnedPipelineLayout,
}

pub(in crate::vulkan) fn build_prepass_layout(
    device: &VkDevice,
    global_set_layout: vk::DescriptorSetLayout,
    bindless_set_layout: vk::DescriptorSetLayout,
) -> RenderResult<PrepassLayout> {
    let set_layout = create_descriptor_set_layout(
        device,
        &history_set_bindings(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
    )?;
    let layouts = [global_set_layout, bindless_set_layout, set_layout.handle()];
    let pipeline_layout = device
        .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts))
        .map_err(|e| crate::vulkan::error::map_vk_result(e, "gbuffer prepass pipeline layout"))?;
    Ok(PrepassLayout {
        set_layout,
        pipeline_layout,
    })
}

// One shader bucket's pre-pass pipeline over its `vertex_prepass_bindless` /
// `fragment_prepass_bindless` pair: the same no-cull / camera depth test as the
// main pass, over a private depth buffer. It binds only the attributes the
// vertex module reads, which depend on the bucket's vertex hook.
pub(in crate::vulkan) fn create_prepass_pipeline(
    device: &VkDevice,
    targets: MeshPipelineTargets<'_>,
) -> RenderResult<OwnedPipeline> {
    let read = input_locations(&spirv_words(targets.vert_spv)?);
    let attributes = prepass_attributes(&read)?;
    GraphicsPipelineDesc {
        depth: Depth::write(),
        vertex_bindings: &VERTEX_56_DUAL_BINDINGS,
        vertex_attributes: &attributes,
        ..GraphicsPipelineDesc::fullscreen(
            targets.vert_spv,
            targets.frag_spv,
            targets.layout,
            targets.render_pass,
            &PREPASS_TARGETS,
        )
    }
    .build(device, "gbuffer prepass")
}

// The attributes a pre-pass vertex module reading the input locations `read`
// binds: each one it reads, and only those. A location the layout does not
// offer is an error, since the module would read an unbound input.
fn prepass_attributes(read: &[u32]) -> RenderResult<Vec<vk::VertexInputAttributeDescription>> {
    if let Some(missing) = read
        .iter()
        .find(|l| !PREPASS_VERTEX_ATTRIBUTES.iter().any(|a| a.location == **l))
    {
        return Err(RenderError::ShaderCompile(format!(
            "G-buffer pre-pass vertex stage reads input location {missing}, which the pre-pass \
             vertex layout does not offer"
        )));
    }
    Ok(PREPASS_VERTEX_ATTRIBUTES
        .iter()
        .filter(|a| read.contains(&a.location))
        .copied()
        .collect())
}

// The G-buffer pre-pass's model history and its per-frame sets, built when the
// bindless cull path is active AND the G-buffer is enabled. Stored on `VkCull`.
// The per-frame set (`PrepassLayout`'s set 2) binds the G-buffer view UBO, the
// PREVIOUS frame's history slot and this frame's draw args; the per-frame
// model-history SSBOs are filled on the GPU by the snapshot kernel below.
pub(in crate::vulkan) struct GbufferBindless {
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

// Descriptor wiring the pre-pass's per-frame sets allocate against: the shared
// pool, and `PrepassLayout`'s set 2 layout.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GbufferBindlessDescriptors {
    pub descriptor_pool: vk::DescriptorPool,
    pub prepass_set_layout: vk::DescriptorSetLayout,
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

// Build the G-buffer pre-pass's model-history ring, the snapshot kernel that
// fills it, and the descriptor sets for both. The pre-pass's per-frame set binds
// the G-buffer view UBO, the previous frame's history slot and this frame's
// draw args.
pub(in crate::vulkan) fn build_gbuffer_bindless(
    ctx: GbufferDeviceCtx,
    descriptors: GbufferBindlessDescriptors,
    records: GbufferBindlessRecords,
    gb: &GbufferResources,
    scene: GbufferBindlessScene,
    hot_reload: bool,
) -> RenderResult<GbufferBindless> {
    let GbufferDeviceCtx { alloc, device } = ctx;
    let GbufferBindlessDescriptors {
        descriptor_pool,
        prepass_set_layout,
    } = descriptors;
    let GbufferBindlessScene { n_cull, frames } = scene;
    let GbufferBindlessRecords {
        object_buffers,
        draw_args_buffers,
    } = records;

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

    // One set per frame: binding 0 = that frame's GbView UBO, binding 1 = the
    // history slot the PREVIOUS frame filled, binding 2 = that frame's draw
    // args. The frame index cycles, so the previous slot is a fixed offset and
    // every set can be written once here.
    let draw_args_size = (n_cull * std::mem::size_of::<GpuDrawArgs>()) as u64;
    let set_layouts: Vec<_> = (0..frames).map(|_| prepass_set_layout).collect();
    let sets = alloc_descriptor_sets(device, descriptor_pool, &set_layouts)?;
    for (f, &set) in sets.iter().enumerate() {
        SetWrites::new(set)
            .uniform_buffer(0, gb.view_ubo_buffers[f].buffer(), GBUFFER_VIEW_UBO_SIZE)
            .storage_buffer(
                1,
                prev_model_buffers[(f + frames - 1) % frames].buffer(),
                buf_size,
            )
            .storage_buffer(2, draw_args_buffers[f].buffer(), draw_args_size)
            .apply(device);
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
        sets,
        prev_model_buffers,
        history,
    })
}

// A UBO at binding 0 and two storage buffers at 1 and 2, read by `stages`: the
// shape of both the pre-pass's set 0 and the snapshot kernel's.
fn history_set_bindings(stages: vk::ShaderStageFlags) -> [Binding; 3] {
    use vk::DescriptorType as T;
    [
        (0, T::UNIFORM_BUFFER, stages),
        (1, T::STORAGE_BUFFER, stages),
        (2, T::STORAGE_BUFFER, stages),
    ]
}

// Sizing for the snapshot kernel's per-frame sets.
#[derive(Clone, Copy)]
struct ModelHistoryScene {
    n_cull: usize,
    frames: usize,
}

// Build the model-history snapshot kernel: set 0 binds the record-count UBO at
// binding 0, the frame's object buffer at 1 and the frame's history slot at 2,
// which is the declaration order `model_history.hlsl` fixes.
fn build_model_history(
    ctx: GbufferDeviceCtx,
    descriptor_pool: vk::DescriptorPool,
    history_buffers: &[PooledBuffer],
    object_buffers: &[PooledBuffer],
    scene: ModelHistoryScene,
    hot_reload: bool,
) -> RenderResult<ModelHistoryPipeline> {
    let GbufferDeviceCtx { alloc, device } = ctx;
    let ModelHistoryScene { n_cull, frames } = scene;
    let cs = super::super::builtin_shaders::MODEL_HISTORY.compile(hot_reload)?;

    let set_layout =
        create_descriptor_set_layout(device, &history_set_bindings(vk::ShaderStageFlags::COMPUTE))?;
    let layouts = [set_layout.handle()];
    let pipeline_layout = device
        .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts))
        .map_err(|e| crate::vulkan::error::map_vk_result(e, "model history pipeline layout"))?;
    let pipeline = compute_pipeline(device, pipeline_layout.handle(), &cs, "model history")?;

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
        SetWrites::new(set)
            .uniform_buffer(0, params_buf.buffer(), params_size)
            .storage_buffer(1, object_buffers[f].buffer(), object_size)
            .storage_buffer(2, history_buffers[slot].buffer(), history_size)
            .apply(device);
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

// One frame slot's sampled G-buffer channels.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GbufferFrame {
    pub normal_depth: vk::ImageView,
    pub roughness: vk::ImageView,
    pub velocity: vk::ImageView,
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
    // host-mapped. The pre-pass pipeline's set 0 and the sky's per-frame sets
    // point at it.
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
    pub(in crate::vulkan) view_history: ViewHistory,

    // The sky's motion behind the geometry.
    pub(in crate::vulkan) sky: GbufferSky,
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
        hot_reload: bool,
    ) -> RenderResult<Self> {
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
        let sky = GbufferSky::build(
            device,
            prepass_render_pass.handle(),
            &view_ubo_buffers,
            hot_reload,
        )?;

        let mut me = Self {
            prepass_render_pass,
            view_ubo_buffers,
            normal_depth_images: Vec::new(),
            roughness_images: Vec::new(),
            velocity_images: Vec::new(),
            depth_images: Vec::new(),
            framebuffers: Vec::new(),
            view_history: ViewHistory::default(),
            sky,
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
    ) -> RenderResult<()> {
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
            let normal_depth = *pooled.normal_depth.get(f).ok_or_else(|| {
                RenderError::Other("gbuffer: pooled normal_depth slot out of range".to_string())
            })?;
            let roughness = *pooled.roughness.get(f).ok_or_else(|| {
                RenderError::Other("gbuffer: pooled roughness slot out of range".to_string())
            })?;
            let velocity = *pooled.velocity.get(f).ok_or_else(|| {
                RenderError::Other("gbuffer: pooled velocity slot out of range".to_string())
            })?;
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
                .map_err(|e| {
                    crate::vulkan::error::map_vk_result(e, "gbuffer prepass framebuffer")
                })?;
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

    // This frame slot's sampled channels, or `None` while the targets are
    // missing after a failed rebuild.
    pub(in crate::vulkan) fn frame(&self, frame: usize) -> Option<GbufferFrame> {
        Some(GbufferFrame {
            normal_depth: self.normal_depth_images.get(frame)?.view,
            roughness: self.roughness_images.get(frame)?.view,
            velocity: self.velocity_images.get(frame)?.view,
        })
    }

    // Whether the per-frame targets exist. A rebuild that fails leaves none,
    // never a partial set.
    pub(in crate::vulkan) fn has_targets(&self) -> bool {
        !self.framebuffers.is_empty()
    }

    // Per-frame normal+depth views, one per frame in flight. The readers that
    // bind a per-frame descriptor set (SSR resolve, SSAO kernel/blur, SSGI, RT)
    // slice this so each set samples its own frame's unified G-buffer.
    pub(in crate::vulkan) fn normal_depth_views(&self) -> Vec<vk::ImageView> {
        self.normal_depth_images
            .iter()
            .map(|img| img.view)
            .collect()
    }

    // Per-frame roughness views, one per frame in flight.
    pub(in crate::vulkan) fn roughness_views(&self) -> Vec<vk::ImageView> {
        self.roughness_images.iter().map(|img| img.view).collect()
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
    // descriptor sets and UBOs are resolution-independent and untouched. On
    // failure the targets are left empty, so readers skip until a later
    // rebuild succeeds.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        ctx: GbufferDeviceCtx,
        queue: GbufferQueueCtx,
        extent: GbufferExtent,
        pooled: &GbufferPooled,
    ) -> RenderResult<()> {
        self.destroy_targets(ctx.device);
        let built = self.build_targets(ctx, queue, extent, pooled);
        if built.is_err() {
            self.destroy_targets(ctx.device);
        }
        built
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
    // The main pass's clock and camera position, which a vertex hook reads.
    pub elapsed: f32,
    pub cam_pos: [f32; 3],
}

impl VkContext {
    // The G-buffer while it has targets to draw into and sample.
    pub(in crate::vulkan) fn gbuffer_targets(&self) -> Option<&GbufferResources> {
        self.gbuffer.as_ref().filter(|gb| gb.has_targets())
    }

    // Whether anything reprojects through the pre-pass's motion channel this
    // frame: TAA, the FSR upscaler, or the SSGI accumulation.
    pub(in crate::vulkan) fn reads_motion(&self) -> bool {
        self.taa.is_some()
            || self.upscale.is_some()
            || self.ssgi.as_ref().is_some_and(|s| s.settings.contributes())
    }

    // Give the G-buffer whatever its pre-pass still lacks to draw with: the
    // per-frame sets, the model-history ring and its snapshot kernel, which
    // init builds only for a world that starts with a G-buffer consumer, and
    // each shader bucket's pre-pass pipeline. Builds only what is missing, so
    // it is safe to repeat. A world with no GPU-driven pass has nothing to
    // draw here.
    pub(in crate::vulkan) fn enable_gbuffer_prepass(&mut self) -> RenderResult<()> {
        if let (Some(gb), Some(prepass), None) = (
            self.gbuffer.as_ref(),
            self.cull.prepass_layout.as_ref(),
            self.cull.model_history.as_ref(),
        ) {
            let built = build_gbuffer_bindless(
                GbufferDeviceCtx {
                    alloc: &self.hw.alloc,
                    device: &self.hw.device,
                },
                GbufferBindlessDescriptors {
                    descriptor_pool: self.descriptors.descriptor_pool.handle(),
                    prepass_set_layout: prepass.set_layout.handle(),
                },
                GbufferBindlessRecords {
                    object_buffers: &self.cull.object_buffers,
                    draw_args_buffers: &self.cull.draw_args_buffers,
                },
                gb,
                GbufferBindlessScene {
                    n_cull: self.cull.bucket_stride,
                    frames: self.frames_in_flight,
                },
                self.hot_reload.enabled,
            )?;
            self.cull.gbuffer_sets = built.sets;
            self.cull.prev_model_buffers = built.prev_model_buffers;
            self.cull.model_history = Some(built.history);
            // Nothing has written the fresh ring, so the first pre-pass primes it.
            self.state.model_history.borrow_mut().request_prime();
        }
        self.sync_prepass_pipelines();
        Ok(())
    }

    // Encode the unified G-buffer pre-pass: one jittered traversal of the cull
    // records into the per-frame normal+depth / roughness / velocity MRT plus a
    // private depth buffer. Runs
    // before the main pass. `velocity_active` is true when a consumer (TAA, FSR
    // or SSGI) reads motion; when false, prev == cur so the motion channel is a
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
        grass: Option<&crate::vulkan::grass::GrassFrame>,
    ) {
        let GbufferPrepassView {
            jittered_vp,
            cur_vp,
            elapsed,
            cam_pos,
        } = view;
        let Some(framebuffer) = gb.framebuffers.get(frame_idx) else {
            return;
        };
        let device = &self.hw.device;
        let extent = self.targets.render_extent;

        // Upload this frame's view UBO. When velocity is inactive the previous
        // camera and clock equal the current ones, so instanced + sky motion
        // is zero and the surfaces skip reprojecting.
        let cur = ViewFrame {
            vp: cur_vp,
            elapsed,
            cam_pos,
        };
        let view_uni = GBufferView::new(
            jittered_vp,
            self.state.view.matrix,
            cur,
            gb.view_history.prev_or(cur),
            velocity_active,
        );
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
            depth::CLEAR_VALUE,
        ];
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(gb.prepass_render_pass.handle())
            .framebuffer(framebuffer.handle())
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
        if let Some(grass) = grass {
            self.encode_grass_prepass(cmd, frame_idx, grass);
        }
        self.encode_raymarch_prepass(cmd, frame_idx, &view, &view_uni);
        // The sky keeps the "no geometry" depth and roughness and adds the
        // camera's motion where nothing was drawn.
        if self.draws_sky(self.state.view.mode) {
            gb.sky.encode(device, cmd, frame_idx);
            self.inc_draw_calls(1);
        }

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
        let device = &self.hw.device;
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
            self.hw.device.cmd_pipeline_barrier(
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
    // cull already produced it, so no extra cull dispatch): the static + instance
    // prefix `[0, skinned_record_base())` once per shader bucket, each under that
    // bucket's pre-pass pipeline, over the static VB (bound to BOTH vertex
    // bindings, so prev_pos == cur_pos and the motion is the per-object model
    // delta plus camera), then the skinned tail under bucket 0's over the current
    // deformed VB (binding 0) + the previous-frame deformed VB (binding 1) for
    // per-vertex deformation motion. The main pass's global and bindless sets
    // carry the view block, records, parameter table and texture pool the
    // vertex hook and the surface read; set 2 the pre-pass's own. The CPU never
    // walks the draw lists.
    fn encode_gbuffer_prepass_gpu_driven(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        velocity_active: bool,
    ) {
        let device = &self.hw.device;
        // Bucket 0's pipeline is `None` when its pre-pass failed to build; the
        // other buckets still draw.
        let pipeline = self.cull.prepass_pipeline.as_ref();
        let Some(prepass) = self.cull.prepass_layout.as_ref() else {
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
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                prepass.pipeline_layout.handle(),
                0,
                &[
                    self.descriptors.global_sets[frame_idx],
                    self.cull.bindless_sets[frame_idx],
                    gset,
                ],
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
            if let (true, Some(pipeline)) = (prefix > 0, pipeline) {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.handle());
                device.cmd_draw_indexed_indirect(cmd, indirect, 0, prefix, stride);
                self.inc_draw_calls(1);
            }
        }
        // The material-referenced shader buckets write their own regions of the
        // command buffer, each drawn under its own pre-pass pipeline; a bucket
        // whose Shader is not resident is skipped, matching what the color pass
        // will draw.
        if prefix > 0 {
            self.inc_draw_calls(self.draw_prepass_bucket_regions(cmd, indirect, prefix));
        }

        // Skinned tail: the current deformed VB (binding 0) + the previous-frame
        // deformed VB (binding 1) + the skinned IB. Records carry base_vertex
        // = 0 (global skinned indexing). The previous deformed buffer is read only
        // once the ring is primed (a prior frame posed that slot); before then (or
        // when velocity is inactive) it is the current buffer, so prev_pos ==
        // cur_pos gives a harmless zero skinned motion vector.
        if self.state.draw.n_skinned > 0
            && let (Some(pipeline), Some(cur)) = (pipeline, self.skinned.deformed.get(frame_idx))
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
                // Skinned records are always bucket 0.
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.handle());
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
                    self.state.draw.n_skinned as u32,
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
    // The sky's share of the pre-pass compiles to SPIR-V; the surface entries
    // compile with the main pass's (`pipeline::tests`).
    #[test]
    fn gbuffer_sky_shaders_compile() {
        use crate::vulkan::builtin_shaders::{CompileProgram, GBUFFER_SKY_FRAG, GBUFFER_SKY_VERT};
        concinnity_shader::require_dxc!();
        GBUFFER_SKY_VERT
            .compile(false)
            .expect("gbuffer sky vertex compiles");
        GBUFFER_SKY_FRAG
            .compile(false)
            .expect("gbuffer sky fragment compiles");
    }

    // Only what the module reads is bound, in the layout's own order.
    #[test]
    fn the_prepass_binds_exactly_what_its_module_reads() {
        let bound = |read: &[u32]| -> Vec<u32> {
            super::prepass_attributes(read)
                .expect("offered")
                .iter()
                .map(|a| a.location)
                .collect()
        };
        assert_eq!(bound(&[0, 1, 2, 4, 5]), [0, 1, 2, 4, 5]);
        assert_eq!(bound(&[5, 0]), [0, 5]);
        assert!(bound(&[]).is_empty());
    }

    // A location the layout does not offer is refused rather than left
    // unbound.
    #[test]
    fn an_input_the_layout_does_not_offer_is_an_error() {
        assert!(super::prepass_attributes(&[0, 6]).is_err());
        assert!(super::prepass_attributes(&[9]).is_err());
    }

    // The pre-pass binds the attributes its vertex module reads out of the ones
    // it offers, so every one the engine's module reads must be offered, the
    // current and previous positions among them.
    #[test]
    fn the_prepass_offers_every_attribute_its_vertex_shader_reads() {
        use super::{PREPASS_VERTEX_ATTRIBUTES, input_locations, spirv_words};
        use crate::vulkan::builtin_shaders::{CompileProgram, MAIN_PREPASS_VERT};
        concinnity_shader::require_dxc!();
        let vs = MAIN_PREPASS_VERT
            .compile(false)
            .expect("pre-pass vertex compiles");
        let read = input_locations(&spirv_words(&vs).expect("whole words"));
        let offered: Vec<u32> = PREPASS_VERTEX_ATTRIBUTES
            .iter()
            .map(|a| a.location)
            .collect();
        assert!(
            read.iter().all(|l| offered.contains(l)),
            "{read:?} vs {offered:?}"
        );
        assert!(read.contains(&0) && read.contains(&5), "{read:?}");
    }
}
