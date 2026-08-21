// src/vulkan/line.rs
//
// World-space line pass for the Vulkan backend. Runs at the tail of the
// hdr_resolve decoration chain, after the main pass resolved colour into the
// HDR scene target and depth into the main depth image, so the lines layer over
// the lit scene and SSR / TAA treat them like any other scene content.
//
// The ribbons arrive already expanded (`gfx::lines::build_vertices`):
// world-space quads whose width was sized off each corner's depth, so a line
// holds its pixel thickness at any distance. Like the decal pass this one
// attaches no depth buffer and instead samples the scene depth, so an occluded
// line fades to `OCCLUDED_ALPHA` rather than being clipped by hardware.
//
// Mirrors src/directx/line.rs and src/metal/line.rs.

use std::ffi::CString;

use ash::vk;

use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedFramebuffer, OwnedPipeline, OwnedPipelineLayout, OwnedRenderPass,
    OwnedSetLayout, VkDevice,
};

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::context::VkContext;
use super::pipeline::spv_module;
use crate::gfx::render_types::LineVertex;

// How much of a line still shows where scene geometry is in front of it. A
// faint trace keeps the lines readable inside a dense scene without letting
// them pretend to be unoccluded.
const OCCLUDED_ALPHA: f32 = 0.12;

// First allocation for a frame slot's ribbon-vertex buffer. The editor axes are
// a handful of segments; this covers a few thousand before the first growth.
const MIN_VERTEX_CAPACITY: u64 = 64 * 1024;

// `LineView` is a GPU-free layout struct that lives in concinnity-render;
// re-export it so `crate::vulkan::line::LineView` is the local path.
pub(in crate::vulkan) use concinnity_render::uniforms::LineView;

// Line-pass state on the context: the resources, built on the first frame that
// submits lines so a world that never draws any pays nothing, plus the
// build-failure latch that keeps a broken build from re-reporting every frame.
pub(in crate::vulkan) struct LineState {
    pub resources: Option<LineResources>,
    pub build_failed: bool,
}

impl LineState {
    pub(in crate::vulkan) fn empty() -> Self {
        Self {
            resources: None,
            build_failed: false,
        }
    }
}

// One frame slot's ribbon-vertex buffer: HOST_VISIBLE | HOST_COHERENT and
// persistently mapped, reallocated larger when a frame's expansion outgrows it.
// The frame fence (waited before a slot is reused) proves the GPU has finished
// reading a slot's buffer before it is overwritten or replaced.
struct VertexSlot {
    buffer: PooledBuffer,
    capacity: u64,
}

// Owned by `VkContext` at most once (built lazily): the line pipeline, its
// render pass + per-frame framebuffers over `hdr_resolve`, the per-frame view
// UBO ring, and the per-frame ribbon-vertex buffers.
//
// Line-pass descriptor sets are a single per-frame set:
//   * **set 0** (per-frame, FRAMES sets):
//       - binding 0: UNIFORM_BUFFER, `LineView`
//       - binding 1: COMBINED_IMAGE_SAMPLER, main depth view
pub(in crate::vulkan) struct LineResources {
    render_pass: OwnedRenderPass,
    pub(in crate::vulkan) pipeline: OwnedPipeline,
    pipeline_layout: OwnedPipelineLayout,
    _view_set_layout: OwnedSetLayout,
    _descriptor_pool: OwnedDescriptorPool,

    // Per-frame view UBO (LineView, 80 bytes). Persistently mapped.
    view_ubos: Vec<PooledBuffer>,

    // Per-frame ribbon vertices.
    vertex_slots: Vec<VertexSlot>,

    // Per-frame view sets (binding 0 view UBO, 1 depth).
    view_sets: Vec<vk::DescriptorSet>,

    // One framebuffer per frame-in-flight slot, each binding its frame slot's
    // `hdr_resolve_images[i].view` as the sole colour attachment.
    framebuffers: Vec<OwnedFramebuffer>,

    sampler: vk::Sampler,
}

// Vulkan handles needed to create the line pass's GPU resources. Borrowed for
// the duration of `LineResources::new`.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct LineDeviceContext<'a> {
    pub(in crate::vulkan) alloc: &'a DeviceAllocator,
    pub(in crate::vulkan) device: &'a VkDevice,
}

// Render-target inputs the line pass writes into / samples from: the resolved
// HDR colour attachment (format + per-frame views), the main depth views, the
// shared sampler, and the framebuffer extent.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct LinePassTargets<'a> {
    pub(in crate::vulkan) hdr_format: vk::Format,
    pub(in crate::vulkan) hdr_resolve_views: &'a [vk::ImageView],
    pub(in crate::vulkan) depth_views: &'a [vk::ImageView],
    pub(in crate::vulkan) sampler: vk::Sampler,
    pub(in crate::vulkan) extent: vk::Extent2D,
}

impl LineResources {
    fn new(
        ctx: LineDeviceContext,
        targets: LinePassTargets,
        frames: usize,
        msaa: bool,
        hot_reload: bool,
    ) -> Result<Self, String> {
        let LineDeviceContext { alloc, device } = ctx;
        let LinePassTargets {
            hdr_format,
            hdr_resolve_views,
            depth_views,
            sampler,
            extent,
        } = targets;
        let render_pass = create_line_render_pass(device, hdr_format)?;
        let view_set_layout = create_line_set_layout(device)?;
        let pipeline_layout = create_line_pipeline_layout(device, view_set_layout.handle())?;

        let (vert_spv, frag_spv) = compile_line_shaders(hot_reload, msaa)?;
        let pipeline = create_line_pipeline(
            device,
            render_pass.handle(),
            pipeline_layout.handle(),
            &vert_spv,
            &frag_spv,
        )?;

        let view_size = std::mem::size_of::<LineView>() as u64;
        let mut view_ubos = Vec::with_capacity(frames);
        let mut vertex_slots = Vec::with_capacity(frames);
        for _ in 0..frames {
            view_ubos.push(alloc.create_buffer(
                view_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?);
            vertex_slots.push(new_vertex_slot(alloc, MIN_VERTEX_CAPACITY)?);
        }

        let descriptor_pool = create_line_descriptor_pool(device, frames)?;
        let view_layouts: Vec<_> = (0..frames).map(|_| view_set_layout.handle()).collect();
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool.handle())
            .set_layouts(&view_layouts);
        // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
        // it names belongs to this device.
        let view_sets = unsafe { device.allocate_descriptor_sets(&info) }
            .map_err(|e| format!("line descriptor sets: {e}"))?;
        for (i, &set) in view_sets.iter().enumerate() {
            write_view_set(
                device,
                set,
                view_ubos[i].buffer(),
                depth_views[i.min(depth_views.len().saturating_sub(1))],
                sampler,
            );
        }

        let mut framebuffers = Vec::with_capacity(frames);
        for &view in hdr_resolve_views.iter().take(frames) {
            framebuffers.push(create_line_framebuffer(
                device,
                render_pass.handle(),
                view,
                extent,
            )?);
        }

        Ok(Self {
            render_pass,
            pipeline,
            pipeline_layout,
            _view_set_layout: view_set_layout,
            _descriptor_pool: descriptor_pool,
            view_ubos,
            vertex_slots,
            view_sets,
            framebuffers,
            sampler,
        })
    }

    // Rebuild the framebuffers + re-point the per-frame view set's depth
    // binding after a swapchain resize. Same pattern as `DecalResources`; the
    // pipeline, layouts, buffers, and sampler all survive.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        device: &VkDevice,
        hdr_resolve_views: &[vk::ImageView],
        depth_views: &[vk::ImageView],
        extent: vk::Extent2D,
    ) -> Result<(), String> {
        self.framebuffers.clear();
        for &view in hdr_resolve_views.iter().take(self.view_ubos.len()) {
            self.framebuffers.push(create_line_framebuffer(
                device,
                self.render_pass.handle(),
                view,
                extent,
            )?);
        }
        for (i, &set) in self.view_sets.iter().enumerate() {
            let depth_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(depth_views[i.min(depth_views.len().saturating_sub(1))])
                .sampler(self.sampler);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&depth_info));
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        }
        Ok(())
    }

    // Destroy every GPU resource. Called from `VkContext::destroy` after
    // `wait_idle`; the pooled buffers retire through the allocator as their
    // fields clear.
    pub(in crate::vulkan) fn destroy(&mut self, _device: &VkDevice) {
        self.framebuffers.clear();
        self.view_ubos.clear();
        self.vertex_slots.clear();
    }
}

// Allocate one persistently-mapped host-visible vertex slot of `capacity` bytes.
fn new_vertex_slot(alloc: &DeviceAllocator, capacity: u64) -> Result<VertexSlot, String> {
    let buffer = alloc.create_buffer(
        capacity,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    Ok(VertexSlot { buffer, capacity })
}

// New capacity for a slot that must hold at least `needed` bytes, given its
// current `capacity`. Grows geometrically so a burst of small growths
// amortizes, but never returns less than `needed`.
fn grow_capacity(capacity: u64, needed: u64) -> u64 {
    let mut cap = capacity.max(MIN_VERTEX_CAPACITY);
    while cap < needed {
        cap *= 2;
    }
    cap
}

fn create_line_framebuffer(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    view: vk::ImageView,
    extent: vk::Extent2D,
) -> Result<OwnedFramebuffer, String> {
    let attachments = [view];
    let info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(&attachments)
        .width(extent.width.max(1))
        .height(extent.height.max(1))
        .layers(1);
    device
        .create_framebuffer(&info)
        .map_err(|e| format!("line framebuffer: {e}"))
}

// Render pass / pipeline construction

fn create_line_render_pass(
    device: &VkDevice,
    format: vk::Format,
) -> Result<OwnedRenderPass, String> {
    // One colour attachment: the resolved HDR scene. The preceding pass left it
    // in SHADER_READ_ONLY_OPTIMAL; we want it in COLOR_ATTACHMENT during the
    // subpass, then SHADER_READ_ONLY_OPTIMAL again on exit so SSR / TAA / bloom
    // / composite can sample it. Mirrors the decal render pass.
    let attachment = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref));
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
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(&deps);
    device
        .create_render_pass(&info)
        .map_err(|e| format!("line render pass: {e}"))
}

fn create_line_set_layout(device: &VkDevice) -> Result<OwnedSetLayout, String> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    device
        .create_descriptor_set_layout(&info)
        .map_err(|e| format!("line view set layout: {e}"))
}

fn create_line_pipeline_layout(
    device: &VkDevice,
    view_set_layout: vk::DescriptorSetLayout,
) -> Result<OwnedPipelineLayout, String> {
    let set_layouts = [view_set_layout];
    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    device
        .create_pipeline_layout(&info)
        .map_err(|e| format!("line pipeline layout: {e}"))
}

fn create_line_descriptor_pool(
    device: &VkDevice,
    frames: usize,
) -> Result<OwnedDescriptorPool, String> {
    let frames = frames as u32;
    let sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: frames,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: frames,
        },
    ];
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(frames)
        .pool_sizes(&sizes);
    device
        .create_descriptor_pool(&info)
        .map_err(|e| format!("line descriptor pool: {e}"))
}

fn write_view_set(
    device: &VkDevice,
    set: vk::DescriptorSet,
    view_ubo: vk::Buffer,
    depth_view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let view_info = vk::DescriptorBufferInfo::default()
        .buffer(view_ubo)
        .offset(0)
        .range(std::mem::size_of::<LineView>() as u64);
    let depth_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(depth_view)
        .sampler(sampler);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&view_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&depth_info)),
    ];
    // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every set
    // and resource it names belongs to this device.
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

fn compile_line_shaders(hot_reload: bool, msaa: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let ctx = super::builtins::Ctx {
        msaa,
        ..super::builtins::Ctx::plain(hot_reload)
    };
    let vert = super::slang_builtins::LINE_VERT.compile(&ctx)?;
    let frag = super::slang_builtins::LINE_FRAG.compile(&ctx)?;
    Ok((vert, frag))
}

// Rebuild the line graphics pipeline against the existing render pass + layout.
// Used by the Vulkan shader hot-reload path. The caller destroys the previous
// pipeline only after this call succeeds.
pub(in crate::vulkan) fn rebuild_line_pipeline(
    device: &VkDevice,
    lines: &LineResources,
    msaa: bool,
    hot_reload: bool,
) -> Result<OwnedPipeline, String> {
    let (vert_spv, frag_spv) = compile_line_shaders(hot_reload, msaa)?;
    create_line_pipeline(
        device,
        lines.render_pass.handle(),
        lines.pipeline_layout.handle(),
        &vert_spv,
        &frag_spv,
    )
}

fn create_line_pipeline(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> Result<OwnedPipeline, String> {
    let vert = spv_module(device, vert_spv)?;
    let frag = spv_module(device, frag_spv)?;
    let entry = CString::new("main").unwrap();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert.handle())
            .name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag.handle())
            .name(&entry),
    ];
    // `LineVertex` (position, edge, colour) at 32 bytes, asserted by
    // `line_vertex_layout_matches_shaders`.
    let bindings = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(std::mem::size_of::<LineVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)];
    let attrs = [
        vk::VertexInputAttributeDescription::default()
            .location(0)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .location(1)
            .binding(0)
            .format(vk::Format::R32_SFLOAT)
            .offset(12),
        vk::VertexInputAttributeDescription::default()
            .location(2)
            .binding(0)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(16),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attrs);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        // No culling: a ribbon faces the camera, but its winding depends on
        // which way the line runs.
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        // The pass writes the SINGLE-SAMPLE resolved HDR, not the MSAA colour.
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false);
    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let blend_attachments = [blend_attachment];
    let blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&blend_state)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass);
    let pipeline = crate::vulkan::pipeline_cache::create_graphics_pipeline(device, &info)
        .map_err(|e| format!("create line pipeline: {e}"))?;
    Ok(pipeline)
}

// Encoder

impl VkContext {
    // Build the line resources if this frame has lines to draw and they are not
    // built yet, then make sure this frame slot's vertex buffer holds them. A
    // failed build latches, so the error is reported once and the pass stays
    // skipped for the rest of the run.
    //
    // Called from `record_frame` (after the frame fence proved this slot's
    // previous work retired), so growing the slot's buffer is safe here.
    pub(in crate::vulkan) fn ensure_line_pipeline(
        &mut self,
        frame_idx: usize,
        vertices: &[LineVertex],
    ) {
        if vertices.is_empty() || self.lines.build_failed {
            return;
        }
        if self.lines.resources.is_none() {
            let depth_views: Vec<vk::ImageView> =
                self.depth_images.iter().map(|img| img.view).collect();
            let hdr_resolve_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|img| img.view).collect();
            let built = LineResources::new(
                LineDeviceContext {
                    alloc: &self.alloc,
                    device: &self.device,
                },
                LinePassTargets {
                    hdr_format: super::context::HDR_FORMAT,
                    hdr_resolve_views: &hdr_resolve_views,
                    depth_views: &depth_views,
                    sampler: self.linear_sampler.handle(),
                    extent: self.render_extent,
                },
                self.frames_in_flight,
                self.msaa_samples != vk::SampleCountFlags::TYPE_1,
                self.hot_reload.enabled,
            );
            match built {
                Ok(r) => self.lines.resources = Some(r),
                Err(e) => {
                    self.lines.build_failed = true;
                    tracing::error!("line pipeline: {}", e);
                    return;
                }
            }
        }
        if let Err(e) =
            self.grow_line_vertex_slot(frame_idx, std::mem::size_of_val(vertices) as u64)
        {
            self.lines.build_failed = true;
            tracing::error!("line vertex buffer: {}", e);
        }
    }

    // Reallocate this frame slot's ribbon-vertex buffer when the frame's
    // expansion outgrows it. The replaced buffer retires through the
    // allocator.
    fn grow_line_vertex_slot(&mut self, frame_idx: usize, needed: u64) -> Result<(), String> {
        let Some(lines) = self.lines.resources.as_mut() else {
            return Ok(());
        };
        let Some(slot) = lines.vertex_slots.get_mut(frame_idx) else {
            return Ok(());
        };
        if needed <= slot.capacity {
            return Ok(());
        }
        let capacity = grow_capacity(slot.capacity, needed);
        *slot = new_vertex_slot(&self.alloc, capacity)?;
        Ok(())
    }

    // Encode the line pass: one unindexed triangle list covering every expanded
    // ribbon, alpha-blended into the resolved HDR target. `vp` is the same
    // view-projection the main pass rasterised with (jittered under TAA), so a
    // line sits on the pixel its geometry did.
    pub(in crate::vulkan) fn encode_lines(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        vp: [[f32; 4]; 4],
        vertices: &[LineVertex],
    ) {
        let Some(lines) = self.lines.resources.as_ref() else {
            return;
        };
        if vertices.is_empty() {
            return;
        }
        let Some(slot) = lines.vertex_slots.get(frame_idx) else {
            return;
        };
        let bytes = std::mem::size_of_val(vertices) as u64;
        if bytes > slot.capacity {
            return;
        }

        let device = &self.device;
        let extent = self.render_extent;

        let view_uni = LineView {
            vp,
            occluded_alpha: OCCLUDED_ALPHA,
            _pad: [0.0; 3],
        };
        lines.view_ubos[frame_idx].write_val(0, &view_uni);
        slot.buffer.write_slice(0, vertices);

        // Main depth is already in SHADER_READ_ONLY for the fragment's occlusion
        // sample: the graph declares this pass's depth read and the executor emits
        // the transition ahead of this command buffer.
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(lines.render_pass.handle())
            .framebuffer(lines.framebuffers[frame_idx].handle())
            .render_area(vk::Rect2D::default().extent(extent));

        // Negative-height viewport matches the main pass so the rasterised
        // pixel grid lines up with the depth attachment being sampled.
        let vp_state = vk::Viewport {
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
            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp_state));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                lines.pipeline.handle(),
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                lines.pipeline_layout.handle(),
                0,
                std::slice::from_ref(&lines.view_sets[frame_idx]),
                &[],
            );
            device.cmd_bind_vertex_buffers(cmd, 0, &[slot.buffer.buffer()], &[0]);
            device.cmd_draw(cmd, vertices.len() as u32, 1, 0, 0);
            device.cmd_end_render_pass(cmd);
        }
        self.inc_draw_calls(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grow_capacity_starts_at_minimum() {
        assert_eq!(grow_capacity(0, 1), MIN_VERTEX_CAPACITY);
    }

    #[test]
    fn grow_capacity_doubles_until_it_fits() {
        let need = MIN_VERTEX_CAPACITY * 3 + 1;
        let cap = grow_capacity(0, need);
        assert!(cap >= need);
        assert_eq!(cap, MIN_VERTEX_CAPACITY * 4);
    }

    #[test]
    fn grow_capacity_never_shrinks_below_existing() {
        assert_eq!(
            grow_capacity(MIN_VERTEX_CAPACITY * 8, 10),
            MIN_VERTEX_CAPACITY * 8
        );
    }
}
