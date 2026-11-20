//! Unified geometry G-buffer pre-pass for the Vulkan backend. One jittered
//! traversal of the visible set (static + instanced + skinned) rasterizes into a
//! single MRT:
//!
//!   target 0  RGBA16F  view-space normal (rgb) + positive linear view depth (a)
//!   target 1  R8       perceptual roughness
//!   target 2  RG16F    screen-space motion (`prev_uv` - `cur_uv`)
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
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::frustum::Frustum;
use concinnity_engine::gfx::render_types::{GpuDrawArgs, GpuObjectData};
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::shader::programs::abi::history::SNAPSHOT_GROUP;
use concinnity_engine::render::uniforms::{GBufferView, ModelHistoryParams};

use super::super::allocator::{DeviceAllocator, PooledBuffer};
use super::super::context::VkContext;
use super::super::descriptor_layout::Binding;
use super::super::pipeline::{MAIN_VERTEX_ATTRS, MeshPipelineTargets, spirv_words};
use super::super::pipeline_desc::{Blend, Depth, GraphicsPipelineDesc, compute_pipeline};
use super::super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::super::set_writes::SetWrites;
use super::super::spirv_inputs::input_locations;
use super::super::texture::{GpuImage, GpuUploadContext, create_depth_image};
use super::gbuffer_sky::GbufferSky;
use crate::vulkan::builtin_shaders::CompileProgram;
use crate::vulkan::depth;
use crate::vulkan::owned::{
    OwnedFramebuffer, OwnedPipeline, OwnedPipelineLayout, OwnedRenderPass, OwnedSetLayout, VkDevice,
};
use crate::vulkan::record::Recorder;

// Normal+depth target: rgb = unit view-space normal, a = positive linear view
// depth (-view_z). Alpha 0 (cleared background) marks "no geometry". Matches
// the SSR G-buffer so the resolve maths is byte-identical.
pub(in crate::vulkan) const GBUFFER_NORMAL_DEPTH_FORMAT: vk::Format =
    vk::Format::R16G16B16A16_SFLOAT;

// Perceptual roughness (r) + specular F0 (gba). Roughness 1.0 (cleared
// background) = no reflection; 0.0 = mirror.
pub(in crate::vulkan) const GBUFFER_ROUGHNESS_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

// Screen-space motion (prev_uv - cur_uv). Cleared to 0 (no motion).
pub(in crate::vulkan) const GBUFFER_VELOCITY_FORMAT: vk::Format = vk::Format::R16G16_SFLOAT;

// Size of the per-frame view UBO: jittered_vp + cur_vp + prev_vp + view_mat +
// prev_model_carry and the previous clock. Matches the `GbView` UBO in
// `gbuffer_common.hlsl`.
pub(in crate::vulkan) const GBUFFER_VIEW_UBO_SIZE: vk::DeviceSize =
    size_of::<GBufferView>() as vk::DeviceSize;

// `GBufferView` (the std140 `GbView` UBO) is a GPU-free layout struct that
// lives in `core::render` (imported above).

// Pre-pass render pass: RGBA16F normal+depth, R8 roughness and RG16F velocity,
// plus a private depth. The colors clear and end shader-readable; the depth is
// stored because the temporal upscaler reads it with the motion vectors.
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

// The main pass's attributes on binding 0 plus the previous-frame position
// (location 5) on binding 1. Static geometry binds the same buffer to both;
// the skinned tail binds the current and previous deformed buffers.
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

// The layout every bucket's pre-pass pipeline binds: the main pass's global and
// bindless sets at 0 and 1, and the pre-pass's view block, model history and
// draw args at set 2. Built with the bindless main pass so a later G-buffer
// consumer can build against it.
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

// The pre-pass's model history and per-frame sets, built with the bindless cull
// path and the G-buffer on. Set 2 binds the view UBO, the previous frame's
// history slot and this frame's draw args.
pub(in crate::vulkan) struct GbufferBindless {
    pub(in crate::vulkan) sets: Vec<vk::DescriptorSet>,
    pub(in crate::vulkan) prev_model_buffers: Vec<PooledBuffer>,
    pub(in crate::vulkan) history: ModelHistoryPipeline,
}

// The model-history snapshot kernel: one thread per cull record copies this
// frame's model matrix into a history slot. `sets` is indexed
// `[frame * frames + slot]`: a normal frame takes the diagonal, and priming a
// rebuilt ring walks a row. Written once at build.
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
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GbufferBindlessScene {
    pub n_cull: usize,
    pub frames: usize,
}

// Build the G-buffer pre-pass's model-history ring, the snapshot kernel that
// fills it, and the descriptor sets for both. The pre-pass's per-frame set binds
// the G-buffer view UBO, the previous frame's history slot and this frame's
// draw args.
pub(in crate::vulkan) fn build_gbuffer_bindless(
    ctx: GbufferDeviceCtx<'_>,
    descriptors: GbufferBindlessDescriptors,
    records: GbufferBindlessRecords<'_>,
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
    let buf_size = (n_cull * size_of::<[[f32; 4]; 4]>()) as u64;
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
    let draw_args_size = (n_cull * size_of::<GpuDrawArgs>()) as u64;
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
    ctx: GbufferDeviceCtx<'_>,
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
        record_count: u32::saturating_from(n_cull),
        _pad: [0; 3],
    };
    let params_size = size_of::<ModelHistoryParams>() as u64;
    let params_buf = alloc.create_buffer(
        params_size,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    params_buf.write_val(0, &params);

    let object_size = (n_cull * size_of::<GpuObjectData>()) as u64;
    let history_size = (n_cull * size_of::<[[f32; 4]; 4]>()) as u64;
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

// One pooled color channel for one frame in flight, borrowed from the transient
// pool, which owns it. Never destroyed here.
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

// Unified G-buffer pre-pass resources. Owns every handle and frees it on
// `destroy`, except the pooled color channels. Per frame in flight, since the
// temporal resolve reads velocity per frame.
pub(in crate::vulkan) struct GbufferResources {
    // Render pass.
    pub(in crate::vulkan) prepass_render_pass: OwnedRenderPass,

    // Per-frame view UBO (jittered_vp + cur_vp + prev_vp + view_mat),
    // host-mapped. The pre-pass pipeline's set 0 and the sky's per-frame sets
    // point at it.
    pub(in crate::vulkan) view_ubo_buffers: Vec<PooledBuffer>,

    // Per-frame MRT targets, private depth and framebuffers, rebuilt on resize. The
    // color channels are pooled; the depth is feature-owned.
    pub(in crate::vulkan) normal_depth_images: Vec<PooledTarget>,
    pub(in crate::vulkan) roughness_images: Vec<PooledTarget>,
    pub(in crate::vulkan) velocity_images: Vec<PooledTarget>,
    pub(in crate::vulkan) depth_images: Vec<GpuImage>,
    pub(in crate::vulkan) framebuffers: Vec<OwnedFramebuffer>,

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
        ctx: GbufferDeviceCtx<'_>,
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
            sky,
        };
        me.build_targets(ctx, queue, extent, pooled)?;
        Ok(me)
    }

    // Allocate the per-frame MRT targets + private depth + framebuffers at the
    // given extent. One slot per frame in flight.
    fn build_targets(
        &mut self,
        ctx: GbufferDeviceCtx<'_>,
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
        // The sampled channels already rest in `SHADER_READ_ONLY_OPTIMAL`, as the pool
        // allocates them, so a consumer sampling them before the pre-pass ever ran
        // binds a valid layout.
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

    // Rebuild the per-frame targets after the transient pool was rebuilt, since
    // the framebuffers reference the pooled images. On failure the targets stay
    // empty and readers skip.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        ctx: GbufferDeviceCtx<'_>,
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

// Camera / view state the G-buffer pre-pass's raymarched surfaces rasterize
// with. `jittered_vp` is the jittered VP that rasterizes (matching the main
// pass).
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GbufferPrepassView<'a> {
    pub jittered_vp: [[f32; 4]; 4],
    // The main pass's clock and camera position, which a vertex hook reads.
    pub elapsed: f32,
    pub cam_pos: [f32; 3],
    // The camera frustum the raymarched surfaces are culled against.
    pub frustum: &'a Frustum,
}

impl VkContext {
    // The G-buffer while it has targets to draw into and sample.
    pub(in crate::vulkan) fn gbuffer_targets(&self) -> Option<&GbufferResources> {
        self.gbuffer.as_ref().filter(|gb| gb.has_targets())
    }

    // Build whatever the pre-pass still lacks to draw with (per-frame sets, model
    // history and snapshot kernel, each bucket's pipeline), since init builds them
    // only when the world starts with a consumer. Safe to repeat.
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

    // Encode the unified G-buffer pre-pass before the main pass: one jittered
    // traversal into the normal+depth / roughness / velocity MRT. Without a motion
    // consumer the previous frame is this one, so motion is zero.
    pub(in crate::vulkan) fn encode_gbuffer_prepass(
        &self,
        gb: &GbufferResources,
        rec: &Recorder<'_>,
        frame_idx: usize,
        view: GbufferPrepassView<'_>,
        // The frame's planned view block, collapsed onto this frame's own camera
        // when no pass reads motion.
        block: &GBufferView,
        grass: Option<&crate::vulkan::grass::GrassFrame>,
    ) {
        let Some(framebuffer) = gb.framebuffers.get(frame_idx) else {
            return;
        };
        let extent = self.targets.render_extent;
        let velocity_active = block.motion != 0;
        let view_uni = *block;
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
        rec.begin_render_pass(
            &gb.prepass_render_pass,
            framebuffer,
            vk::Rect2D::default().extent(extent),
            &clears,
        );

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
        rec.set_viewport(&vp);
        rec.set_scissor(&scissor);

        // Reuses the main pass's indirect buffer with two draws (static + instance,
        // then the skinned tail). With nothing to draw the pass is just the clears.
        self.encode_gbuffer_prepass_gpu_driven(rec, frame_idx, velocity_active);
        if let Some(grass) = grass {
            self.encode_grass_prepass(rec, frame_idx, grass);
        }
        self.encode_raymarch_prepass(rec, frame_idx, &view, &view_uni);
        // The sky keeps the "no geometry" depth and roughness and adds the
        // camera's motion where nothing was drawn.
        if self.draws_sky(self.state.view.mode) {
            gb.sky.encode(rec, frame_idx);
            self.inc_draw_calls(1);
        }

        rec.end_render_pass();

        // Snapshot this frame's models into this frame's history slot, AFTER the
        // pass above read the previous one -- which is what keeps a single frame
        // in flight (one slot, read then rewritten) correct.
        self.encode_model_history(rec, frame_idx);
    }

    // Copy each record's model into this frame's history slot. That slot was read
    // by a pre-pass `frames_in_flight - 1` frames ago, so the write waits on those
    // vertex reads and is released to the next frame's pre-pass.
    fn encode_model_history(&self, rec: &Recorder<'_>, frame_idx: usize) {
        let Some(history) = self.cull.model_history.as_ref() else {
            return;
        };
        let records = self.state.draw.record_count();
        if records == 0 {
            return;
        }
        // A rebuilt ring holds nothing these records were written for, so the
        // priming frame fills every slot rather than only its own: the instance
        // region is the one the draw args cannot flag, being init-written.
        let frames = self.cull.prev_model_buffers.len();
        let slots = if history
            .prime
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            0..frames
        } else {
            frame_idx..frame_idx + 1
        };
        let records = u32::saturating_from(records);
        for slot in slots {
            let Some(&set) = history.sets.get(frame_idx * frames + slot) else {
                continue;
            };
            self.model_history_barrier(
                rec,
                slot,
                (
                    vk::AccessFlags::SHADER_READ,
                    vk::AccessFlags::SHADER_WRITE,
                    vk::PipelineStageFlags::VERTEX_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                ),
            );
            rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &history.pipeline);
            rec.bind_descriptor_sets(
                vk::PipelineBindPoint::COMPUTE,
                &history.pipeline_layout,
                0,
                &[set],
                &[],
            );
            rec.dispatch_threads([records, 1, 1], SNAPSHOT_GROUP);
            self.model_history_barrier(
                rec,
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

    // One access/stage dependency over a whole model-history slot.
    fn model_history_barrier(
        &self,
        rec: &Recorder<'_>,
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
        rec.pipeline_barrier(
            src_stage,
            dst_stage,
            &[],
            std::slice::from_ref(&barrier),
            &[],
        );
    }

    // The GPU-driven pre-pass raster inside the caller's render pass: the static +
    // instance prefix once per shader bucket over the static VB bound to both
    // bindings, then the skinned tail over the current and previous deformed VBs.
    fn encode_gbuffer_prepass_gpu_driven(
        &self,
        rec: &Recorder<'_>,
        frame_idx: usize,
        velocity_active: bool,
    ) {
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
            .map(PooledBuffer::buffer)
        else {
            return;
        };
        let Some(&gset) = self.cull.gbuffer_sets.get(frame_idx) else {
            return;
        };
        let prefix = u32::saturating_from(self.state.draw.skinned_record_base());

        let sets = [
            self.descriptors.global_sets[frame_idx],
            self.cull.bindless_sets[frame_idx],
            gset,
        ];
        rec.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            &prepass.pipeline_layout,
            0,
            &sets,
            &[],
        );

        // Static + instance prefix: the static VB bound to BOTH vertex bindings
        // (prev_pos == cur_pos) + the static u32 IB.
        let static_vb = self.geometry.vertex_buffer.buffer();
        rec.bind_vertex_buffers(0, &[static_vb, static_vb], &[0, 0]);
        rec.bind_index_buffer(
            self.geometry.index_buffer.buffer(),
            0,
            vk::IndexType::UINT32,
        );
        if let (true, Some(pipeline)) = (prefix > 0, pipeline) {
            rec.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline);
            self.draw_main_stream(rec, indirect, 0);
            self.inc_draw_calls(1);
        }
        // The material-referenced shader buckets write their own regions of the
        // command buffer, each drawn under its own pre-pass pipeline; a bucket
        // whose Shader is not resident is skipped, matching what the color pass
        // will draw.
        if prefix > 0 {
            self.inc_draw_calls(self.draw_prepass_bucket_regions(rec, indirect, frame_idx));
        }

        // Skinned tail: current deformed VB (0), previous deformed VB (1), skinned IB.
        // Until the ring is primed, or without velocity, the previous buffer is the
        // current one, so skinned motion is zero.
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
            // Skinned records are always bucket 0.
            rec.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline);
            rec.bind_vertex_buffers(0, &[cur.buffer, prev.buffer], &[0, 0]);
            rec.bind_index_buffer(self.skinned.index_buffer.buffer(), 0, vk::IndexType::UINT32);
            self.draw_main_stream(
                rec,
                indirect,
                concinnity_engine::gfx::render_types::SKINNED_STREAM,
            );
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
