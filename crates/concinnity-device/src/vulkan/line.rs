//! World-space line pass for the Vulkan backend. Runs at the tail of the
//! `hdr_resolve` decoration chain, after the main pass resolved color into the
//! HDR scene target and depth into the main depth image, so the lines layer over
//! the lit scene and SSR / TAA treat them like any other scene content.
//!
//! The ribbons arrive already expanded (`gfx::lines::build_vertices`):
//! world-space quads whose width was sized off each corner's depth, so a line
//! holds its pixel thickness at any distance. Like the decal pass this one
//! attaches no depth buffer and instead samples the scene depth, so an occluded
//! line fades to `OCCLUDED_ALPHA` rather than being clipped by hardware.
//!
//! Mirrors src/directx/line.rs and src/metal/line.rs.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types::LineVertex;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::buffers::growth::grow_capacity;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::context::VkContext;
use super::descriptor_layout::{Binding, PoolSizes};
use super::pipeline_desc::{Blend, GraphicsPipelineDesc};
use super::rendering::PassFormats;
use super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::set_writes::SetWrites;
use crate::vulkan::builtin_shaders::CompileProgram;
use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedPipeline, OwnedPipelineLayout, OwnedSetLayout, VkDevice,
};
use crate::vulkan::record::Recorder;

// How much of a line still shows where scene geometry is in front of it. A
// faint trace keeps the lines readable inside a dense scene without letting
// them pretend to be unoccluded.
const OCCLUDED_ALPHA: f32 = 0.12;

// First allocation for a frame slot's ribbon-vertex buffer. The editor axes are
// a handful of segments; this covers a few thousand before the first growth.
const MIN_VERTEX_CAPACITY: u64 = 64 * 1024;

// `LineView` is a GPU-free layout struct that lives in `core::render`;
// re-export it so `crate::vulkan::line::LineView` is the local path.
pub(in crate::vulkan) use concinnity_engine::render::uniforms::LineView;

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

// The line pass, built lazily: the pipeline drawing over `hdr_resolve`, the
// view UBO ring, and per-frame vertex buffers. Set 0 is per frame: `LineView`
// (0) and the main depth (1).
pub(in crate::vulkan) struct LineResources {
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
}

// Vulkan handles needed to create the line pass's GPU resources. Borrowed for
// the duration of `LineResources::new`.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct LineDeviceContext<'a> {
    pub(in crate::vulkan) alloc: &'a DeviceAllocator,
    pub(in crate::vulkan) device: &'a VkDevice,
}

// Render-target inputs the line pass writes into / samples from: the resolved
// HDR color format and the main depth views.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct LinePassTargets<'a> {
    pub(in crate::vulkan) hdr_format: vk::Format,
    pub(in crate::vulkan) depth_views: &'a [vk::ImageView],
}

impl LineResources {
    fn new(
        ctx: LineDeviceContext<'_>,
        targets: LinePassTargets<'_>,
        frames: usize,
        msaa: bool,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let LineDeviceContext { alloc, device } = ctx;
        let LinePassTargets {
            hdr_format,
            depth_views,
        } = targets;
        let view_set_layout = create_descriptor_set_layout(device, &view_set_bindings())?;
        let pipeline_layout = create_line_pipeline_layout(device, view_set_layout.handle())?;

        let (vert_spv, frag_spv) = compile_line_shaders(hot_reload, msaa)?;
        let pipeline = create_line_pipeline(
            device,
            hdr_format,
            pipeline_layout.handle(),
            &vert_spv,
            &frag_spv,
        )?;

        let view_size = size_of::<LineView>() as u64;
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
        let view_sets = alloc_descriptor_sets(device, descriptor_pool.handle(), &view_layouts)?;
        for (i, &set) in view_sets.iter().enumerate() {
            SetWrites::new(set)
                .uniform_buffer(0, view_ubos[i].buffer(), view_size)
                .sampled_image(1, depth_views[i.min(depth_views.len().saturating_sub(1))])
                .apply(device);
        }

        Ok(Self {
            pipeline,
            pipeline_layout,
            _view_set_layout: view_set_layout,
            _descriptor_pool: descriptor_pool,
            view_ubos,
            vertex_slots,
            view_sets,
        })
    }

    // Re-point the per-frame view set's depth binding after a swapchain
    // resize; the pipeline, layouts and buffers all survive.
    pub(in crate::vulkan) fn repoint_depth(
        &self,
        device: &VkDevice,
        depth_views: &[vk::ImageView],
    ) {
        for (i, &set) in self.view_sets.iter().enumerate() {
            SetWrites::new(set)
                .sampled_image(1, depth_views[i.min(depth_views.len().saturating_sub(1))])
                .apply(device);
        }
    }

    // Destroy every GPU resource. Called from `VkContext::destroy` after
    // `wait_idle`; the pooled buffers retire through the allocator as their
    // fields clear.
    pub(in crate::vulkan) fn destroy(&mut self, _device: &VkDevice) {
        self.view_ubos.clear();
        self.vertex_slots.clear();
    }
}

// Allocate one persistently-mapped host-visible vertex slot of `capacity` bytes.
fn new_vertex_slot(alloc: &DeviceAllocator, capacity: u64) -> RenderResult<VertexSlot> {
    let buffer = alloc.create_buffer(
        capacity,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    Ok(VertexSlot { buffer, capacity })
}

// Pipeline construction

// The per-frame view set: the LineView UBO and the main depth.
fn view_set_bindings() -> [Binding; 2] {
    use vk::DescriptorType as T;
    let frag = vk::ShaderStageFlags::FRAGMENT;
    [
        (0, T::UNIFORM_BUFFER, vk::ShaderStageFlags::VERTEX | frag),
        (1, T::SAMPLED_IMAGE, frag),
    ]
}

fn create_line_pipeline_layout(
    device: &VkDevice,
    view_set_layout: vk::DescriptorSetLayout,
) -> RenderResult<OwnedPipelineLayout> {
    let set_layouts = [view_set_layout];
    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    device
        .create_pipeline_layout(&info)
        .map_err(|e| super::error::map_vk_result(e, "line pipeline layout"))
}

fn create_line_descriptor_pool(
    device: &VkDevice,
    frames: usize,
) -> RenderResult<OwnedDescriptorPool> {
    let frames = u32::saturating_from(frames);
    let sizes = PoolSizes::default()
        .sets(&view_set_bindings(), frames)
        .build();
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(frames)
        .pool_sizes(&sizes);
    device
        .create_descriptor_pool(&info)
        .map_err(|e| super::error::map_vk_result(e, "line descriptor pool"))
}

fn compile_line_shaders(hot_reload: bool, msaa: bool) -> RenderResult<(Vec<u8>, Vec<u8>)> {
    let vert = super::builtin_shaders::LINE_VERT.compile(hot_reload)?;
    let frag = super::builtin_shaders::LINE_FRAG
        .at(msaa)
        .compile(hot_reload)?;
    Ok((vert, frag))
}

// Rebuild the line graphics pipeline against the existing layout.
// Used by the Vulkan shader hot-reload path. The caller destroys the previous
// pipeline only after this call succeeds.
pub(in crate::vulkan) fn rebuild_line_pipeline(
    device: &VkDevice,
    lines: &LineResources,
    msaa: bool,
    hot_reload: bool,
) -> RenderResult<OwnedPipeline> {
    let (vert_spv, frag_spv) = compile_line_shaders(hot_reload, msaa)?;
    create_line_pipeline(
        device,
        super::context::HDR_FORMAT,
        lines.pipeline_layout.handle(),
        &vert_spv,
        &frag_spv,
    )
}

// Alpha-blended ribbons into the SINGLE-SAMPLE resolved HDR. No culling: a
// ribbon faces the camera, but its winding depends on which way the line runs.
fn create_line_pipeline(
    device: &VkDevice,
    hdr_format: vk::Format,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> RenderResult<OwnedPipeline> {
    // `LineVertex` (position, edge, color) at 32 bytes, asserted by
    // `line_vertex_layout_matches_shaders`.
    let bindings = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(u32::saturating_from(size_of::<LineVertex>()))
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
    GraphicsPipelineDesc {
        vertex_bindings: &bindings,
        vertex_attributes: &attrs,
        ..GraphicsPipelineDesc::fullscreen(
            vert_spv,
            frag_spv,
            layout,
            PassFormats::color(hdr_format),
            &[Blend::AlphaOver],
        )
    }
    .build(device, "line")
}

// Encoder

impl VkContext {
    // Build the line resources the first frame with lines and size this slot's
    // vertex buffer; the frame fence already freed it. A failed build latches and
    // the pass stays skipped.
    pub(in crate::vulkan) fn ensure_line_pipeline(
        &mut self,
        frame_idx: usize,
        vertices: &[LineVertex],
    ) {
        if vertices.is_empty() || self.lines.build_failed {
            return;
        }
        if self.lines.resources.is_none() {
            let depth_views: Vec<vk::ImageView> = self
                .targets
                .depth_images
                .iter()
                .map(|img| img.view)
                .collect();
            let built = LineResources::new(
                LineDeviceContext {
                    alloc: &self.hw.alloc,
                    device: &self.hw.device,
                },
                LinePassTargets {
                    hdr_format: super::context::HDR_FORMAT,
                    depth_views: &depth_views,
                },
                self.frames_in_flight,
                self.targets.msaa_samples != vk::SampleCountFlags::TYPE_1,
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
        if let Err(e) = self.grow_line_vertex_slot(frame_idx, size_of_val(vertices) as u64) {
            self.lines.build_failed = true;
            tracing::error!("line vertex buffer: {}", e);
        }
    }

    // Reallocate this frame slot's ribbon-vertex buffer when the frame's
    // expansion outgrows it. The replaced buffer retires through the
    // allocator.
    fn grow_line_vertex_slot(&mut self, frame_idx: usize, needed: u64) -> RenderResult<()> {
        let Some(lines) = self.lines.resources.as_mut() else {
            return Ok(());
        };
        let Some(slot) = lines.vertex_slots.get_mut(frame_idx) else {
            return Ok(());
        };
        let Some(capacity) = grow_capacity(slot.capacity, needed, MIN_VERTEX_CAPACITY) else {
            return Ok(());
        };
        *slot = new_vertex_slot(&self.hw.alloc, capacity)?;
        Ok(())
    }

    // Encode the line pass: one unindexed triangle list covering every expanded
    // ribbon, alpha-blended into the resolved HDR target. `vp` is the same
    // view-projection the main pass rasterized with (jittered under TAA), so a
    // line sits on the pixel its geometry did.
    pub(in crate::vulkan) fn encode_lines(
        &self,
        rec: &Recorder<'_>,
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
        let bytes = size_of_val(vertices) as u64;
        if bytes > slot.capacity {
            return;
        }

        let extent = self.targets.render_extent;

        let view_uni = LineView {
            vp,
            occluded_alpha: OCCLUDED_ALPHA,
            pre_exposure: self.pre_exposure.value(),
            _pad: [0.0; 2],
        };
        lines.view_ubos[frame_idx].write_val(0, &view_uni);
        slot.buffer.write_slice(0, vertices);

        // Main depth is already in SHADER_READ_ONLY for the fragment's occlusion
        // sample: the graph declares this pass's depth read and the executor emits
        // the transition ahead of this command buffer.

        // Negative-height viewport matches the main pass so the rasterized
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

        rec.render_over_sampled(&self.targets.hdr_resolve_images[frame_idx], extent, |rec| {
            rec.set_viewport(&vp_state);
            rec.set_scissor(&scissor);
            rec.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, &lines.pipeline);
            rec.bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                &lines.pipeline_layout,
                0,
                std::slice::from_ref(&lines.view_sets[frame_idx]),
                &[],
            );
            rec.bind_vertex_buffers(0, &[slot.buffer.buffer()], &[0]);
            rec.draw(u32::saturating_from(vertices.len()), 1, 0, 0);
        });

        self.inc_draw_calls(1);
    }
}
