//! The engine's `PassId::Transparent` slot on the Vulkan backend: one render
//! pass, drawn after the SSR resolve and before TAA, with two producers -- glass
//! panes (`glass.rs`) and water surfaces (`water.rs`). Each contributes records
//! built once at init; the pass snapshots the pre-transparent scene, orders every
//! record of both producers back-to-front by camera distance, and draws them into
//! the post-SSR scene image (`SsrResources::output` when SSR is on, else
//! `hdr_resolve_images[frame]`), alpha-blending over it. Downstream TAA / bloom /
//! composite pick the translucent geometry up unchanged.
//!
//! One pass rather than one per producer, mirroring the Metal and DirectX
//! backends: the scene snapshot the refraction taps is a full render-resolution
//! HDR image and a copy of it every frame, so a second one would be pure waste;
//! and a single ordering over both producers is what puts a pane standing in a
//! pool on the correct side of the water.
//!
//! The producers also share every descriptor set layout and pipeline layout,
//! because `glass.hlsl` and `water.hlsl` declare the same bindings on purpose:
//! the view set (0) carries the per-frame view UBO plus the snapshot and main
//! depth, the params set (1) one record's uniforms plus its planar reflection
//! target, the global set (2) is the forward one the probe / sky taps read, and
//! the RT variants add the trace's geometry (3) and the bindless pool (4).
//!
//! Same uniform layouts, back-to-front ordering and manual depth-occlusion test
//! as the DirectX and Metal hosts.

use ash::vk;
use concinnity_core::components::{GlassPanel, WaterSurface};
use concinnity_core::gfx::lod;
use concinnity_core::gfx::mesh_payload::Vertex;
use concinnity_core::gfx::render_types::RtParams;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::fullscreen::align_up;
use concinnity_core::render::lights;
use concinnity_core::render::post::rt_reflections::RtParamsInputs;
pub(in crate::vulkan) use concinnity_core::render::uniforms::TransparentView;
use concinnity_core::transform::mat4_inverse;
// `TransparentView` (the per-frame view UBO) is a GPU-free layout struct that
// lives in `core::render`; re-export it so the encode path and the graph's
// view builder can keep naming it through this module.
use concinnity_core::render::uniforms::GlassMeshParams;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::context::{HDR_FORMAT, VkContext};
use super::descriptor_layout::{Binding, PoolSizes};
use super::pipeline::GraphicsStages;
use super::resources::{alloc_descriptor_sets, create_descriptor_set_layout, write_samplers};
use super::texture::{
    GpuImage, GpuUploadContext, ImageSpec, LayoutTransition, SubresourceRange, create_image,
    create_image_view, one_shot_submit, transition_image_layout_range, upload_texture,
};
use super::wire_cache::WireCache;
use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedFramebuffer, OwnedPipeline, OwnedPipelineLayout, OwnedRenderPass,
    OwnedSetLayout, VkDevice,
};

// The live acceleration-structure handles wired into the transparent RT
// descriptor ring. Passed once at init (`None` when RT is not live at launch)
// and re-pointed every frame thereafter by `VkContext::rt_dynamic_update`, so
// the ring tracks dynamic TLAS / geometry-table / deformed-buffer rebuilds.
// Mirrors the per-frame inputs `post::rt_reflections::wire_dynamic` takes.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct TransparentRtInputs {
    pub tlas: vk::AccelerationStructureKHR,
    pub geom_buffer: vk::Buffer,
    pub geom_size: vk::DeviceSize,
    pub deformed_verts: vk::Buffer,
    pub skinned_indices: vk::Buffer,
}

// The live acceleration-structure handles re-pointed into one frame's
// transparent RT descriptor set every frame by `wire_dynamic` /
// `wire_rt_dynamic`. Same handles the RT-reflection pass rewires; the deformed
// buffer is always valid while `skinned_indices` is null until the first skinned
// rebuild. Compared frame to frame so a frame that rebuilt nothing rewrites
// nothing.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::vulkan) struct TransparentRtDynamic {
    pub tlas: vk::AccelerationStructureKHR,
    pub geom_buffer: vk::Buffer,
    pub geom_size: vk::DeviceSize,
    pub deformed: vk::Buffer,
    pub skinned_indices: vk::Buffer,
}

pub(in crate::vulkan) use concinnity_core::render::transparent::Producer;

// Per-record GPU state: the static world-space quad (glass) or origin-centered
// grid (water) VB + IB, the per-record params UBO + its descriptor set, and the
// visibility flag.
pub(in crate::vulkan) struct TransparentRecord {
    vertex_buffer: PooledBuffer,
    index_buffer: PooledBuffer,
    index_count: u32,
    _params_ubo: PooledBuffer,
    params_set: vk::DescriptorSet,
    visible: bool,
    // World-space center, used for the back-to-front camera-distance sort.
    center: [f32; 3],
    // The record's planar reflection slot (its mirror render's target), or `None`
    // when it falls back to the probe cube. Drives the resize re-point of the
    // planar binding (binding 1 of `params_set`).
    planar_slot: Option<usize>,
}

// The geometry, uniform payload and per-record state one producer hands over for
// a `TransparentRecord`. Keeps the buffer uploads and the descriptor write in
// one place instead of once per producer.
pub(in crate::vulkan) struct RecordUpload<'a> {
    pub vertices: &'a [Vertex],
    pub indices: &'a [u16],
    pub params: &'a [u8],
    pub visible: bool,
    pub center: [f32; 3],
    pub planar_slot: Option<usize>,
}

// The descriptor plumbing a record needs: the pool + layout its params set comes
// from, the planar target it samples (or the 1x1 stand-in), and the linear
// sampler bound alongside.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct RecordDescriptors<'a> {
    pub device: &'a VkDevice,
    pub pool: vk::DescriptorPool,
    pub params_set_layout: vk::DescriptorSetLayout,
    pub planar_view: vk::ImageView,
    pub sampler: vk::Sampler,
}

impl TransparentRecord {
    // Upload one record's static geometry (host-visible, written once) and its
    // per-record params UBO, then allocate + write the record's descriptor set.
    pub(in crate::vulkan) fn upload(
        alloc: &DeviceAllocator,
        descriptors: RecordDescriptors,
        upload: RecordUpload<'_>,
    ) -> RenderResult<Self> {
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let vb_bytes = std::mem::size_of_val(upload.vertices) as u64;
        let ib_bytes = std::mem::size_of_val(upload.indices) as u64;
        let vertex_buffer =
            alloc.create_buffer(vb_bytes, vk::BufferUsageFlags::VERTEX_BUFFER, host)?;
        let index_buffer =
            alloc.create_buffer(ib_bytes, vk::BufferUsageFlags::INDEX_BUFFER, host)?;
        vertex_buffer.write_slice(0, upload.vertices);
        index_buffer.write_slice(0, upload.indices);

        let params_ubo = alloc.create_buffer(
            upload.params.len() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            host,
        )?;
        params_ubo.write_slice(0, upload.params);

        let params_set = alloc_descriptor_sets(
            descriptors.device,
            descriptors.pool,
            &[descriptors.params_set_layout],
        )?[0];
        write_params_set(
            descriptors.device,
            params_set,
            params_ubo.buffer(),
            upload.params.len() as u64,
            descriptors.planar_view,
            descriptors.sampler,
        );

        Ok(Self {
            vertex_buffer,
            index_buffer,
            index_count: upload.indices.len() as u32,
            _params_ubo: params_ubo,
            params_set,
            visible: upload.visible,
            center: upload.center,
            planar_slot: upload.planar_slot,
        })
    }
}

// One producer's pipelines plus its records. The RT pair is `Some` only when the
// device is RT-capable and the compile succeeded; a live RT toggle then selects
// it with no rebuild. The textured variant additionally needs the bindless pool.
pub(in crate::vulkan) struct TransparentProducer {
    pub pipeline: OwnedPipeline,
    pub flat_rt_pso: Option<OwnedPipeline>,
    pub textured_rt_pso: Option<OwnedPipeline>,
    // The glass reflection pre-pass pair, built beside the RT pair for glass
    // panes; `None` for water, which traces in place.
    pub reflection_flat_pso: Option<OwnedPipeline>,
    pub reflection_textured_pso: Option<OwnedPipeline>,
    pub records: Vec<TransparentRecord>,
}

impl TransparentProducer {
    // Pick this producer's pipeline for the frame: the sharp per-pixel trace when
    // RT is live, the textured variant when the bindless pool exists as well, and
    // the probe / planar pipeline otherwise.
    //
    // The `expect`s are the point rather than an inconvenience: each variant is
    // built against a different pipeline layout, and the encoder binds the
    // descriptor sets under the layout it picked for the whole pass, so falling
    // back across the three would draw with sets bound under an incompatible
    // layout. Both choices are whole-pass decisions the encoder makes from
    // `rt_pipelines_ready` / `rt_textured_ready`, which require the pipeline of
    // every live producer -- so a producer is never asked for one it lacks.
    fn pipeline(&self, rt_live: bool, textured: bool) -> &OwnedPipeline {
        match (rt_live, textured) {
            (true, true) => self
                .textured_rt_pso
                .as_ref()
                .expect("rt_textured_ready gated the frame on every producer's textured pipeline"),
            (true, false) => self
                .flat_rt_pso
                .as_ref()
                .expect("rt_pipelines_ready gated the frame on every producer's flat RT pipeline"),
            _ => &self.pipeline,
        }
    }

    // This producer's reflection pre-pass pipeline, or `None` when it traces in
    // place.
    fn reflection_pipeline(&self, textured: bool) -> Option<&OwnedPipeline> {
        match textured {
            true => self.reflection_textured_pso.as_ref(),
            false => self.reflection_flat_pso.as_ref(),
        }
    }
}

// The see-through glass MESH producer. Ray-traced only: what makes the mesh
// see-through rather than the opaque reflective glass of the main pass is a real
// per-pixel reflection ray, so there is no probe-path pipeline and the whole
// producer is inert while RT is off (those meshes then render opaque).
//
// It holds no `TransparentRecord`s. A mesh draws from the shared scene vertex /
// index buffers at its `DrawObject`'s offsets, and both those offsets (LOD picks
// per frame) and its params change at runtime, so the encoder rebuilds the list
// every frame and writes each mesh's params into this frame's slice of the ring.
pub(in crate::vulkan) struct GlassMeshProducer {
    pipeline_flat: OwnedPipeline,
    // `Some` only when the bindless pool is live, matching the other producers.
    pipeline_textured: Option<OwnedPipeline>,
    // The glass reflection pre-pass pair, gated the same way.
    reflection_flat: OwnedPipeline,
    reflection_textured: Option<OwnedPipeline>,
    // Indices into `VkContext::draw.objects` of every see-through mesh,
    // precomputed at init. The objects stay IN `draw.objects` -- a slot is a key
    // into the cull / prev-model / RT parallel arrays -- this only marks which to
    // reroute.
    object_indices: Vec<usize>,
    // Per-frame params ring: one host-mapped buffer per frame slot holding one
    // `params_stride`-aligned `GlassMeshParams` block per mesh.
    params_buffers: Vec<PooledBuffer>,
    params_stride: u64,
    // One params descriptor set per (frame, mesh), written once at init to point
    // at that block. Indexed `frame * object_indices.len() + slot`.
    params_sets: Vec<vk::DescriptorSet>,
    _pool: OwnedDescriptorPool,
}

// One see-through mesh's draw for this frame: the shared-buffer slice its
// `DrawObject` resolved to, the params set covering its block of this frame's
// ring, and its world-space center for the back-to-front sort.
struct GlassMeshDraw {
    index_offset: u32,
    index_count: u32,
    base_vertex: i32,
    params_set: vk::DescriptorSet,
    center: [f32; 3],
}

impl GlassMeshProducer {
    // Allocate the per-frame params ring and one descriptor set per (frame,
    // mesh) pointing at that mesh's block of it, then take ownership of the
    // pipelines. The sets come from a pool of this producer's own rather than the
    // pass's, whose size is fixed to the pane + water record count.
    //
    // Each set reuses the shared params layout, so its planar binding (1) is
    // written with the 1x1 stand-in: a mesh never samples a planar reflection,
    // but the layout the pipeline was built against still declares the
    // binding.
    pub(in crate::vulkan) fn new(
        ctx: &ProducerCtx,
        pipelines: TracedGlassPipelines,
        object_indices: Vec<usize>,
    ) -> RenderResult<Self> {
        let device = ctx.device;
        let count = object_indices.len();
        let frames = ctx.frames;
        let params_stride = align_up(
            std::mem::size_of::<GlassMeshParams>() as u64,
            ctx.ubo_offset_alignment,
        );

        let mut params_buffers = Vec::with_capacity(frames);
        for _ in 0..frames {
            params_buffers.push(ctx.alloc.create_buffer(
                params_stride * count.max(1) as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?);
        }

        let sets_needed = (frames * count) as u32;
        let sizes = PoolSizes::default()
            .sets(&params_set_bindings(), sets_needed)
            .build();
        let pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(sets_needed.max(1))
                    .pool_sizes(&sizes),
            )
            .map_err(|e| super::error::map_vk_result(e, "glass mesh descriptor pool"))?;
        let layouts: Vec<_> = (0..frames * count).map(|_| ctx.params_set_layout).collect();
        let params_sets = alloc_descriptor_sets(device, pool.handle(), &layouts)?;
        for frame in 0..frames {
            for slot in 0..count {
                write_params_set_at(
                    device,
                    params_sets[frame * count + slot],
                    params_buffers[frame].buffer(),
                    slot as u64 * params_stride,
                    std::mem::size_of::<GlassMeshParams>() as u64,
                    ctx.stand_in_view,
                    ctx.sampler,
                );
            }
        }

        let TracedGlassPipelines {
            shade_flat: pipeline_flat,
            shade_textured: pipeline_textured,
            reflection_flat,
            reflection_textured,
        } = pipelines;
        Ok(Self {
            pipeline_flat,
            pipeline_textured,
            reflection_flat,
            reflection_textured,
            object_indices,
            params_buffers,
            params_stride,
            params_sets,
            _pool: pool,
        })
    }

    // Pick the frame's pipeline. Same all-or-nothing gate as the other producers:
    // `rt_textured_ready` requires every live producer's textured pipeline, so
    // this is never asked for one it lacks.
    fn pipeline(&self, textured: bool) -> &OwnedPipeline {
        match textured {
            true => self
                .pipeline_textured
                .as_ref()
                .expect("rt_textured_ready gated the frame on every producer's textured pipeline"),
            false => &self.pipeline_flat,
        }
    }

    // The reflection pre-pass pipeline, under the same gate as `pipeline`.
    fn reflection_pipeline(&self, textured: bool) -> &OwnedPipeline {
        match textured {
            true => self
                .reflection_textured
                .as_ref()
                .expect("rt_textured_ready gated the frame on every producer's textured pipeline"),
            false => &self.reflection_flat,
        }
    }
}

// A traced glass producer's pipelines: the shading pair and the reflection
// pre-pass pair, each textured variant `Some` only with the bindless pool.
pub(in crate::vulkan) struct TracedGlassPipelines {
    pub shade_flat: OwnedPipeline,
    pub shade_textured: Option<OwnedPipeline>,
    pub reflection_flat: OwnedPipeline,
    pub reflection_textured: Option<OwnedPipeline>,
}

// Per-pixel ray-traced reflection state shared by both producers: the two
// pipeline layouts (flat material-tint + textured bindless), the per-frame
// RtParams UBO ring, and the per-frame RT descriptor ring (set 3: TLAS +
// geometry table + the static + skinned vertex/index buffers). Mirrors the RT
// half of `directx::transparent::TransparentResources`.
struct TransparentRt {
    _set_layout: OwnedSetLayout,
    layout_flat: OwnedPipelineLayout,
    // The textured layout is `Some` only when the bindless texture pool is live
    // (the same gate the bindless static + RT-reflection passes use).
    layout_textured: Option<OwnedPipelineLayout>,

    // Per-frame RtParams UBO ring (144 B, host-mapped). The encoder fills this
    // frame's slot (sun + ray tunables) before binding, mirroring
    // `encode_rt_reflections`.
    params_buffers: Vec<PooledBuffer>,

    // Per-frame RT descriptor ring (set 3). Static bindings (RtParams UBO, the
    // shared static verts / indices) are written once; the TLAS / geom table /
    // deformed verts / skinned indices (bindings 1/2/5/6) are re-pointed every
    // frame by `wire_dynamic` because a dynamic rebuild fresh-allocates them.
    sets: Vec<vk::DescriptorSet>,
    _pool: OwnedDescriptorPool,

    // 1-element dummy SSBO bound to the skinned vertex/index bindings (5/6) when
    // the scene carries no skinned geometry (the accel data's skinned-index handle
    // is then `vk::Buffer::null()`), so the descriptor stays valid. Mirrors the
    // RT-reflection pass's dummy.
    dummy_ssbo: PooledBuffer,

    // What each frame's dynamic bindings (1/2/5/6) already point at, so a frame
    // whose acceleration structures did not move skips four descriptor writes,
    // one of them an acceleration-structure write.
    wired_accel: WireCache<TransparentRtDynamic>,
}

// Engine-side transparent-pass resources. Built only when the world declared at
// least one `GlassPanel` or `WaterSurface`; `VkContext::transparent` stays
// `None` otherwise and the Transparent pass is omitted from the frame graph.
pub(in crate::vulkan) struct TransparentResources {
    render_pass: OwnedRenderPass,
    pipeline_layout: OwnedPipelineLayout,
    _view_set_layout: OwnedSetLayout,
    _params_set_layout: OwnedSetLayout,
    _descriptor_pool: OwnedDescriptorPool,

    // Per-frame `TransparentView` UBO ring. Host-mapped; the encoder writes this
    // frame's view before binding.
    view_ubos: Vec<PooledBuffer>,
    view_sets: Vec<FrameViewSets>,

    // Per-frame scene target the pass blends into: `SsrResources::output`
    // (repeated for every frame slot) when SSR is on, else this slot's
    // `hdr_resolve_images[i]`. The framebuffer targets the view; the snapshot
    // copy reads the image.
    scene_images: Vec<vk::Image>,
    framebuffers: Vec<OwnedFramebuffer>,

    // Pre-transparent HDR scene snapshot for the refraction tap. The encoder
    // copies the scene image into this at the head of the pass; sized to render
    // dims, recreated by `rebuild` on resize. Single image shared across frames
    // (the same single-shared-snapshot pattern as the raymarch pass).
    snapshot: GpuImage,

    glass: Option<TransparentProducer>,
    water: Option<TransparentProducer>,
    glass_mesh: Option<GlassMeshProducer>,

    rt: Option<TransparentRt>,

    // The glass reflection pre-pass: its render pass (the reflection pipelines
    // are built against it whenever glass can trace), the reduced layers while the
    // trace divisor is above 1, and the 1x1 empty layer the first one peels behind.
    reflection_render_pass: OwnedRenderPass,
    reflection: Option<GlassReflectionLayers>,
    empty_layer: GpuImage,
}

use concinnity_core::render::transparent::ordered_visible;

fn rt_set_bindings() -> [Binding; 7] {
    use vk::DescriptorType as T;
    let frag = vk::ShaderStageFlags::FRAGMENT;
    [
        (0, T::UNIFORM_BUFFER, frag),
        (1, T::ACCELERATION_STRUCTURE_KHR, frag),
        (2, T::STORAGE_BUFFER, frag),
        (3, T::STORAGE_BUFFER, frag),
        (4, T::STORAGE_BUFFER, frag),
        (5, T::STORAGE_BUFFER, frag),
        (6, T::STORAGE_BUFFER, frag),
    ]
}

impl TransparentRt {
    // Write the per-frame static RT bindings: the RtParams UBO (0) + the shared
    // static verts (3) + u32 indices (4). The TLAS / geom table / skinned buffers
    // (1/2/5/6) are filled by `wire_dynamic`. Called once at init.
    fn wire_static(&self, device: &VkDevice, vertex_buffer: vk::Buffer, index_buffer: vk::Buffer) {
        for (i, &set) in self.sets.iter().enumerate() {
            let ubo_info = vk::DescriptorBufferInfo::default()
                .buffer(self.params_buffers[i].buffer())
                .offset(0)
                .range(std::mem::size_of::<RtParams>() as vk::DeviceSize);
            let writes = [vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&ubo_info))];
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        self.rewire_geometry(device, vertex_buffer, index_buffer);
    }

    // Re-point every frame's shared static verts (3) + u32 indices (4) at the
    // given buffers. Called by `wire_static`, and again on its own when an asset
    // hot-reload replaces the shared geometry buffers under the pass.
    fn rewire_geometry(
        &self,
        device: &VkDevice,
        vertex_buffer: vk::Buffer,
        index_buffer: vk::Buffer,
    ) {
        let verts_info = vk::DescriptorBufferInfo::default()
            .buffer(vertex_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let indices_info = vk::DescriptorBufferInfo::default()
            .buffer(index_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        for &set in &self.sets {
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&verts_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&indices_info)),
            ];
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }

    // Re-point one frame's TLAS (1), geometry table (2), deformed skinned verts
    // (5), and skinned indices (6) at the live handles. Called every frame from
    // `VkContext::rt_dynamic_update` (the current frame's set is fence-gated). A
    // frame that rebuilt nothing hands over the same five handles this slot
    // already holds and writes nothing. The deformed buffer is always a valid
    // handle (the accel data holds a 1-element
    // dummy when there is no skinned geometry); `skinned_indices` is null until the
    // first skinned rebuild, in which case the 1-element dummy SSBO binds so the
    // descriptor stays valid. Mirrors `post::rt_reflections::wire_dynamic`.
    fn wire_dynamic(&mut self, device: &VkDevice, frame_idx: usize, dynamic: TransparentRtDynamic) {
        if !self.wired_accel.changed(frame_idx, dynamic) {
            return;
        }
        let TransparentRtDynamic {
            tlas,
            geom_buffer,
            geom_size,
            deformed,
            skinned_indices,
        } = dynamic;
        let set = self.sets[frame_idx];
        let accels = [tlas];
        let mut accel_write = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&accels);
        let mut tlas_write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .push_next(&mut accel_write);
        // `push_next` does not set the count for an acceleration-structure write.
        tlas_write.descriptor_count = 1;
        let geom_info = vk::DescriptorBufferInfo::default()
            .buffer(geom_buffer)
            .offset(0)
            .range(geom_size);
        let geom_write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&geom_info));
        let deformed_info = vk::DescriptorBufferInfo::default()
            .buffer(deformed)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let deformed_write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(5)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&deformed_info));
        let sidx_buffer = if skinned_indices != vk::Buffer::null() {
            skinned_indices
        } else {
            self.dummy_ssbo.buffer()
        };
        let sidx_info = vk::DescriptorBufferInfo::default()
            .buffer(sidx_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let sidx_write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(6)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&sidx_info));
        // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every
        // set and resource it names belongs to this device.
        unsafe {
            device
                .update_descriptor_sets(&[tlas_write, geom_write, deformed_write, sidx_write], &[])
        };
    }

    fn destroy(&mut self, _device: &VkDevice) {
        self.params_buffers.clear();
        self.dummy_ssbo = PooledBuffer::null();
    }
}

// The descriptor set layouts the RT pipeline layouts reference: the shared view /
// params / global sets (0/1/2) plus the bindless texture pool set that gates the
// textured hit-shading variant.
#[derive(Clone, Copy)]
struct RtSetLayouts {
    view: vk::DescriptorSetLayout,
    params: vk::DescriptorSetLayout,
    global: vk::DescriptorSetLayout,
    bindless: Option<vk::DescriptorSetLayout>,
}

// Build the shared RT pipeline layouts + descriptor ring. Called from
// `TransparentResources::new` when the device is RT-capable. The two layouts
// share the view / params / global set layouts (sets 0/1/2) so the same
// descriptor sets the base path binds carry over; the RT geometry rides a
// dedicated set 3 (bindless pool on set 4 for the textured variant).
fn build_transparent_rt(
    alloc: &DeviceAllocator,
    instance: &ash::Instance,
    device: &VkDevice,
    physical_device: vk::PhysicalDevice,
    frames: usize,
    layouts: RtSetLayouts,
    geometry: TransparentRtGeometry,
) -> RenderResult<TransparentRt> {
    let set_layout = create_descriptor_set_layout(device, &rt_set_bindings())?;

    let flat_layouts = [
        layouts.view,
        layouts.params,
        layouts.global,
        set_layout.handle(),
    ];
    let layout_flat = device
        .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&flat_layouts))
        .map_err(|e| super::error::map_vk_result(e, "transparent rt flat pipeline layout"))?;
    // The textured variant binds 5 sets (view / params / global / rt-geom / bindless
    // pool); the flat variant binds 4. The Vulkan spec only guarantees
    // `maxBoundDescriptorSets >= 4`, so on a device that reports exactly 4 fall back
    // to the flat trace (the bindless pool is unreachable there). Every RT-capable
    // desktop GPU reports >= 8; this mirrors the `rt_capable -> flat -> base`
    // degradation ladder.
    // SAFETY: a property query on a live handle; it only reads.
    let max_bound_sets = unsafe { instance.get_physical_device_properties(physical_device) }
        .limits
        .max_bound_descriptor_sets;
    let layout_textured = match layouts.bindless {
        Some(bsl) if max_bound_sets >= 5 => {
            let set_layouts = [
                layouts.view,
                layouts.params,
                layouts.global,
                set_layout.handle(),
                bsl,
            ];
            Some(
                device
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                    )
                    .map_err(|e| {
                        super::error::map_vk_result(e, "transparent rt textured pipeline layout")
                    })?,
            )
        }
        _ => None,
    };

    // Per-frame RtParams UBO ring (host-mapped).
    let params_size = std::mem::size_of::<RtParams>() as vk::DeviceSize;
    let mut params_buffers = Vec::with_capacity(frames);
    for _ in 0..frames {
        params_buffers.push(alloc.create_buffer(
            params_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?);
    }

    // Pool: one set per frame.
    let f = frames as u32;
    let pool_sizes = PoolSizes::default().sets(&rt_set_bindings(), f).build();
    let pool = device
        .create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(f),
        )
        .map_err(|e| super::error::map_vk_result(e, "transparent rt descriptor pool"))?;
    let set_handles: Vec<_> = (0..frames).map(|_| set_layout.handle()).collect();
    let sets = alloc_descriptor_sets(device, pool.handle(), &set_handles)?;

    // 1-element dummy SSBO for the skinned-index binding when there is no skinned
    // geometry.
    let dummy_ssbo = alloc.create_buffer(
        16,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    let mut rt = TransparentRt {
        _set_layout: set_layout,
        layout_flat,
        layout_textured,
        params_buffers,
        sets,
        _pool: pool,
        dummy_ssbo,
        wired_accel: WireCache::new(frames),
    };
    rt.wire_static(device, geometry.vertex_buffer, geometry.index_buffer);
    if let Some(inputs) = geometry.rt_inputs {
        for i in 0..frames {
            rt.wire_dynamic(
                device,
                i,
                TransparentRtDynamic {
                    tlas: inputs.tlas,
                    geom_buffer: inputs.geom_buffer,
                    geom_size: inputs.geom_size,
                    deformed: inputs.deformed_verts,
                    skinned_indices: inputs.skinned_indices,
                },
            );
        }
    }
    Ok(rt)
}

// The transparent render pass: load + store the single-sample scene image (the
// post-SSR scene rests in SHADER_READ_ONLY) with no depth attachment (the
// fragments do the manual occlusion test). Mirrors the decal render pass shape.
fn create_transparent_render_pass(
    device: &VkDevice,
    format: vk::Format,
) -> RenderResult<OwnedRenderPass> {
    let color = vk::AttachmentDescription::default()
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
    // The encoder's explicit barrier (scene back to SHADER_READ_ONLY after the
    // snapshot copy) makes the load available; this dependency orders the load
    // after it.
    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::TRANSFER,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        );
    let info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&color))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&dependency));
    device
        .create_render_pass(&info)
        .map_err(|e| super::error::map_vk_result(e, "transparent render pass"))
}

// The glass reflection pre-pass: one reduced layer (CLEAR, left
// SHADER_READ_ONLY for the next layer and the scene pass to read) over a depth
// attachment the pass clears and tests against. The incoming dependency orders
// the clears after the previous layer's depth writes and after the prior frame's
// reads of the layer; the outgoing one publishes the layer to fragment reads.
fn create_reflection_render_pass(device: &VkDevice) -> RenderResult<OwnedRenderPass> {
    let color = vk::AttachmentDescription::default()
        .format(HDR_FORMAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let depth = vk::AttachmentDescription::default()
        .format(REFLECTION_DEPTH_FORMAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let depth_ref = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref))
        .depth_stencil_attachment(&depth_ref);
    let fragment_tests =
        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
    let dependencies = [
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                fragment_tests
                    | vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            )
            .src_access_mask(
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .dst_stage_mask(
                fragment_tests
                    | vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            )
            .dst_access_mask(
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::SHADER_READ,
            ),
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags::SHADER_READ),
    ];
    let attachments = [color, depth];
    let info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(&dependencies);
    device
        .create_render_pass(&info)
        .map_err(|e| super::error::map_vk_result(e, "glass reflection render pass"))
}

const REFLECTION_DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

// The reduced glass reflection pre-pass targets: two layers (rgb the traced
// reflection, a the surface's distance from the camera) over one depth
// attachment, with a framebuffer per layer.
struct GlassReflectionLayers {
    layers: [GpuImage; 2],
    _depth: GpuImage,
    framebuffers: [OwnedFramebuffer; 2],
    extent: vk::Extent2D,
}

impl GlassReflectionLayers {
    fn new(
        alloc: &DeviceAllocator,
        device: &VkDevice,
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
    ) -> RenderResult<Self> {
        let image = |format: vk::Format, usage: vk::ImageUsageFlags, aspect| -> RenderResult<_> {
            let pooled = create_image(
                alloc,
                &ImageSpec {
                    width: extent.width,
                    height: extent.height,
                    format,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage,
                    mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    samples: vk::SampleCountFlags::TYPE_1,
                },
            )?;
            let view = create_image_view(device, pooled.image(), format, aspect)?;
            Ok(GpuImage::from_pooled(pooled, view))
        };
        let layer = || {
            image(
                HDR_FORMAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::COLOR,
            )
        };
        let layers = [layer()?, layer()?];
        let depth = image(
            REFLECTION_DEPTH_FORMAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
        )?;
        let framebuffer = |color: &GpuImage| {
            let attachments = [color.view, depth.view];
            device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(render_pass)
                        .attachments(&attachments)
                        .width(extent.width)
                        .height(extent.height)
                        .layers(1),
                )
                .map_err(|e| super::error::map_vk_result(e, "glass reflection framebuffer"))
        };
        let framebuffers = [framebuffer(&layers[0])?, framebuffer(&layers[1])?];
        Ok(Self {
            layers,
            _depth: depth,
            framebuffers,
            extent,
        })
    }
}

// One frame's set-0 variants, differing only in the reflection layers bound at
// 3 / 4: the scene pass reads both layers, and each pre-pass layer reads the
// layer it peels behind (the empty layer for the first).
#[derive(Clone, Copy)]
struct FrameViewSets {
    scene: vk::DescriptorSet,
    layers: [vk::DescriptorSet; 2],
}

impl FrameViewSets {
    const COUNT: usize = 3;
}

// Set 0: the per-frame view UBO (0), the scene snapshot (1), this frame's main
// depth (2), the two glass reflection layers (3, 4), and the sampler the
// snapshot is read through (5). The sky prefilter cube and its sampler are the
// global set's (set 2). The view UBO is
// visible to the vertex stage as well: both producers project through `vp`, and
// water reads `time` there for its wave phase.
fn view_set_bindings() -> [Binding; 6] {
    use vk::DescriptorType as T;
    let frag = vk::ShaderStageFlags::FRAGMENT;
    [
        (0, T::UNIFORM_BUFFER, vk::ShaderStageFlags::VERTEX | frag),
        (1, T::SAMPLED_IMAGE, frag),
        (2, T::SAMPLED_IMAGE, frag),
        (3, T::SAMPLED_IMAGE, frag),
        (4, T::SAMPLED_IMAGE, frag),
        (5, T::SAMPLER, frag),
    ]
}

// Set 1: one record's params UBO (0), the planar reflection target it samples
// (1) and that target's sampler (2). The UBO is visible to the vertex stage
// because the water vertex stage reads its wave table out of it.
fn params_set_bindings() -> [Binding; 3] {
    use vk::DescriptorType as T;
    let frag = vk::ShaderStageFlags::FRAGMENT;
    [
        (0, T::UNIFORM_BUFFER, vk::ShaderStageFlags::VERTEX | frag),
        (1, T::SAMPLED_IMAGE, frag),
        (2, T::SAMPLER, frag),
    ]
}

fn create_descriptor_pool(
    device: &VkDevice,
    frames: usize,
    records: usize,
) -> RenderResult<OwnedDescriptorPool> {
    // One view set per frame per `FrameViewSets` slot.
    let v = (frames * FrameViewSets::COUNT) as u32;
    let r = records as u32;
    let sizes = PoolSizes::default()
        .sets(&view_set_bindings(), v)
        .sets(&params_set_bindings(), r)
        .build();
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(v + r)
        .pool_sizes(&sizes);
    device
        .create_descriptor_pool(&info)
        .map_err(|e| super::error::map_vk_result(e, "transparent descriptor pool"))
}

// The images one view set points at: the shared scene snapshot (binding 1),
// this frame's main depth (2) and the two reflection layers (3, 4).
#[derive(Clone, Copy)]
struct ViewSetImages {
    snapshot_view: vk::ImageView,
    depth_view: vk::ImageView,
    reflection: [vk::ImageView; 2],
}

// Write the view set's resolution-independent bindings once: the view UBO (0)
// and the snapshot's sampler (5).
fn write_view_set_statics(
    device: &VkDevice,
    set: vk::DescriptorSet,
    view_ubo: vk::Buffer,
    sampler: vk::Sampler,
) {
    let view_info = vk::DescriptorBufferInfo::default()
        .buffer(view_ubo)
        .offset(0)
        .range(std::mem::size_of::<TransparentView>() as u64);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(std::slice::from_ref(&view_info));
    // SAFETY: the write and the buffer info it borrows are live for the call, and the set and
    // buffer belong to this device.
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
    write_samplers(device, set, 5, &[sampler]);
}

// Write the view set's images, which a resize replaces.
fn write_view_set_images(device: &VkDevice, set: vk::DescriptorSet, inputs: ViewSetImages) {
    let images = [
        inputs.snapshot_view,
        inputs.depth_view,
        inputs.reflection[0],
        inputs.reflection[1],
    ]
    .map(|view| {
        vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)
    });
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(1)
        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
        .image_info(&images);
    // SAFETY: the write and the image infos it borrows are live for the call, and the set and
    // every view belong to this device.
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

// Write a record's params set: its uniform block (binding 0), the planar
// reflection target it samples (binding 1) -- its slot's mirror render, or the
// 1x1 stand-in for a slotless record (the shaders gate on the `planar` flag) --
// and the sampler that target is read through (binding 2).
fn write_params_set(
    device: &VkDevice,
    set: vk::DescriptorSet,
    params_ubo: vk::Buffer,
    params_size: u64,
    planar_view: vk::ImageView,
    sampler: vk::Sampler,
) {
    write_params_set_at(
        device,
        set,
        params_ubo,
        0,
        params_size,
        planar_view,
        sampler,
    );
}

// The same write at an explicit offset into the buffer, so the mesh producer can
// point one set per mesh at its own block of a shared per-frame ring.
fn write_params_set_at(
    device: &VkDevice,
    set: vk::DescriptorSet,
    params_ubo: vk::Buffer,
    params_offset: u64,
    params_size: u64,
    planar_view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let info = vk::DescriptorBufferInfo::default()
        .buffer(params_ubo)
        .offset(params_offset)
        .range(params_size);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(std::slice::from_ref(&info));
    // SAFETY: the write and the buffer info it borrows are live for the call, and the set and
    // buffer belong to this device.
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
    write_planar_view(device, set, planar_view);
    write_samplers(device, set, 2, &[sampler]);
}

// Point a params set's planar binding (1) at `view`.
fn write_planar_view(device: &VkDevice, set: vk::DescriptorSet, view: vk::ImageView) {
    let info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(view);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(1)
        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
        .image_info(std::slice::from_ref(&info));
    // SAFETY: the write and the image info it borrows are live for the call, and the set and
    // view belong to this device.
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

// Which attributes of the standard engine `Vertex` a transparent vertex stage
// fetches. Panes are pre-transformed into world space and water grids carry
// their frame in the wave sum, so neither reads anything but the position; a
// see-through mesh is local-space and shades off the stored normal.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::vulkan) enum TransparentVertexInput {
    Position,
    PositionAndNormal,
}

// Build one transparent graphics pipeline. No face culling (the shaders are
// two-sided), no depth attachment / test (the fragments do the manual occlusion
// test), and SRC_ALPHA / ONE_MINUS_SRC_ALPHA blending into the single-sample
// scene target. The standard engine `Vertex` stride is bound with the attributes
// `vertex_input` names. Negative-height viewport applied dynamically at encode.
pub(in crate::vulkan) fn create_transparent_pipeline(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
    vertex_input: TransparentVertexInput,
) -> RenderResult<OwnedPipeline> {
    let shaders = TransparentShaders {
        vert_spv,
        frag_spv,
        vertex_input,
    };
    transparent_pipeline(
        device,
        render_pass,
        layout,
        shaders,
        TransparentOutput::Scene,
    )
}

// Build one glass reflection pre-pass pipeline: the transparent pipeline's
// stages, overwriting a reflection layer and depth-tested (LESS, writing) against
// the layer's depth attachment, so each layer keeps the nearest surface it
// accepts. `render_pass` is the pre-pass's own.
pub(in crate::vulkan) fn create_glass_reflection_pipeline(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
    vertex_input: TransparentVertexInput,
) -> RenderResult<OwnedPipeline> {
    let shaders = TransparentShaders {
        vert_spv,
        frag_spv,
        vertex_input,
    };
    transparent_pipeline(
        device,
        render_pass,
        layout,
        shaders,
        TransparentOutput::ReflectionLayer,
    )
}

// A transparent pipeline's two stages and the vertex attributes its vertex
// stage fetches.
struct TransparentShaders<'a> {
    vert_spv: &'a [u8],
    frag_spv: &'a [u8],
    vertex_input: TransparentVertexInput,
}

// What a transparent pipeline draws into.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TransparentOutput {
    // Straight-alpha blended over the scene, with no depth attachment.
    Scene,
    // Overwriting a glass reflection layer, depth-tested against its attachment.
    ReflectionLayer,
}

fn transparent_pipeline(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    shaders: TransparentShaders,
    output: TransparentOutput,
) -> RenderResult<OwnedPipeline> {
    let TransparentShaders {
        vert_spv,
        frag_spv,
        vertex_input,
    } = shaders;
    let blend = output == TransparentOutput::Scene;
    let modules = GraphicsStages::new(device, vert_spv, frag_spv)?;
    let stages = modules.infos();

    let binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(std::mem::size_of::<Vertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let attr = |location: u32, offset: u32| {
        vk::VertexInputAttributeDescription::default()
            .location(location)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset)
    };
    // Normal sits at byte 12 of `Vertex`, after the position.
    let attributes: &[vk::VertexInputAttributeDescription] = match vertex_input {
        TransparentVertexInput::Position => &[attr(0, 0)],
        TransparentVertexInput::PositionAndNormal => &[attr(0, 0), attr(1, 12)],
    };
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(std::slice::from_ref(&binding))
        .vertex_attribute_descriptions(attributes);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    // The scene target is single-sample regardless of the main pass's MSAA.
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    // The scene pass has no depth attachment (the fragment shader does the manual
    // occlusion test); a reflection layer keeps its nearest surface.
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(!blend)
        .depth_write_enable(!blend)
        .depth_compare_op(vk::CompareOp::LESS);
    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(blend)
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
        .vertex_input_state(&vertex_input_state)
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
        .map_err(|e| super::error::map_vk_result(e, "create transparent pipeline"))?;
    Ok(pipeline)
}

// Create the pre-transparent HDR scene snapshot (SAMPLED | TRANSFER_DST,
// GPU-local) and rest it in SHADER_READ_ONLY so the first frame's snapshot
// barrier (SHADER_READ_ONLY -> TRANSFER_DST) matches. Mirrors the raymarch
// scene snapshot.
fn create_snapshot(
    alloc: &DeviceAllocator,
    device: &VkDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    width: u32,
    height: u32,
) -> RenderResult<GpuImage> {
    let pooled = create_image(
        alloc,
        &ImageSpec {
            width: width.max(1),
            height: height.max(1),
            format: HDR_FORMAT,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            samples: vk::SampleCountFlags::TYPE_1,
        },
    )?;
    let image = pooled.image();
    one_shot_submit(device, command_pool, queue, |cmd| {
        transition_image_layout_range(
            device,
            cmd,
            image,
            LayoutTransition {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                aspect: vk::ImageAspectFlags::COLOR,
            },
            SubresourceRange {
                base_layer: 0,
                layer_count: 1,
                base_mip: 0,
                mip_count: 1,
            },
        );
    })?;
    let view = create_image_view(device, image, HDR_FORMAT, vk::ImageAspectFlags::COLOR)?;
    Ok(GpuImage::from_pooled(pooled, view))
}

// The Vulkan device handles the transparent build + rebuild need: the instance,
// logical + physical device, and the transient command pool + queue used for the
// one-shot snapshot layout transition.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct TransparentDeviceCtx<'a> {
    pub alloc: &'a DeviceAllocator,
    pub instance: &'a ash::Instance,
    pub device: &'a VkDevice,
    pub physical_device: vk::PhysicalDevice,
    pub command_pool: vk::CommandPool,
    pub queue: vk::Queue,
}

// The non-resource build config: the render dims + ring depth + MSAA sample
// count, the per-frame global descriptor set layout bound as set 2, and the
// hot-reload shader source toggle.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct TransparentBuildConfig {
    pub frames: usize,
    pub msaa_samples: vk::SampleCountFlags,
    pub width: u32,
    pub height: u32,
    // The per-frame global descriptor set layout (ViewUniforms, IBL cubes, probe
    // set). Bound as set 2 so the fragment shaders reflect the probe set / sky
    // prefilter cube; the pipeline layout must reference it even though the pass
    // reads only bindings 5, 7, 8, 10, 11, 17 and 19.
    pub global_set_layout: vk::DescriptorSetLayout,
    pub hot_reload: bool,
    // Per-axis divisor of the glass reflection pre-pass; 1 traces in place.
    pub reflection_divisor: u32,
}

// The post-SSR scene target per frame slot plus the per-frame main-depth views.
// `scene_views` / `scene_images` are the post-SSR scene target per frame slot
// (SSR output repeated, or `hdr_resolve_images[i]`); `depth_views` are the main-
// depth views the manual occlusion test samples. `sampler` is the linear sampler
// the snapshot and the planar targets are read through.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct TransparentSceneTargets<'a> {
    pub scene_views: &'a [vk::ImageView],
    pub scene_images: &'a [vk::Image],
    pub depth_views: &'a [vk::ImageView],
    pub sampler: vk::Sampler,
}

// The world's transparent content, each producer's per-record planar slot
// assignment (aligned with its slice; `None` records keep the probe cube), and
// the per-distinct-plane mirror target views the assigned records sample. A
// slotless record (or an empty `planar_target_views`) binds the snapshot as a
// valid stand-in and never samples it (the shaders gate on the flag). From
// `assign_planar_slots`, which numbers water first.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct TransparentContent<'a> {
    pub glass_panels: &'a [GlassPanel],
    pub glass_planar_slots: &'a [Option<usize>],
    pub water_surfaces: &'a [WaterSurface],
    pub water_planar_slots: &'a [Option<usize>],
    pub planar_target_views: &'a [vk::ImageView],
    // Indices into the context's draw objects of every see-through material.
    // Empty when no material opted in; those meshes then render opaque.
    pub seethrough_mesh_indices: &'a [usize],
}

// Per-pixel RT reflection inputs, built whenever the device is RT-capable (so a
// live quality toggle can bring RT up), independent of whether RT is on at launch.
// `vertex_buffer` / `index_buffer` are the shared static geometry the trace reads;
// `rt_inputs` is the initial acceleration-structure handles (`None` when RT is off
// at launch, then filled per frame by `rt_dynamic_update`); `bindless_set_layout` +
// pool size enable the textured hit-shading variant.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct TransparentRtSetup {
    pub rt_capable: bool,
    pub vertex_buffer: vk::Buffer,
    pub index_buffer: vk::Buffer,
    pub rt_inputs: Option<TransparentRtInputs>,
    pub bindless_set_layout: Option<vk::DescriptorSetLayout>,
    pub bindless_pool_size: usize,
}

// The shared static geometry the trace reads plus the initial acceleration-
// structure handles.
#[derive(Clone, Copy)]
struct TransparentRtGeometry {
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    rt_inputs: Option<TransparentRtInputs>,
}

// What a producer module needs to build its pipelines and records: the shared
// render pass and pipeline layouts, the descriptor pool + params layout its
// records allocate from, the planar stand-in, the sampler, and the shader
// assembly inputs.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct ProducerCtx<'a> {
    pub alloc: &'a DeviceAllocator,
    pub device: &'a VkDevice,
    pub render_pass: vk::RenderPass,
    // The glass reflection pre-pass render pass the reflection pipelines target.
    pub reflection_render_pass: vk::RenderPass,
    pub layout: vk::PipelineLayout,
    pub rt_layout_flat: Option<vk::PipelineLayout>,
    pub rt_layout_textured: Option<vk::PipelineLayout>,
    pub pool: vk::DescriptorPool,
    pub params_set_layout: vk::DescriptorSetLayout,
    // The resolution-independent 1x1 image a params set's planar binding holds
    // when its record samples no mirror.
    pub stand_in_view: vk::ImageView,
    pub planar_target_views: &'a [vk::ImageView],
    pub sampler: vk::Sampler,
    pub msaa: bool,
    pub hot_reload: bool,
    pub bindless_pool_size: usize,
    // Ring depth, for the mesh producer's per-frame params buffers.
    pub frames: usize,
    // The device's `minUniformBufferOffsetAlignment`, which the mesh producer's
    // per-mesh params blocks must be spaced by.
    pub ubo_offset_alignment: u64,
}

impl<'a> ProducerCtx<'a> {
    // The descriptor plumbing for one record, resolving its planar slot to the
    // mirror target it samples (or the stand-in).
    pub(in crate::vulkan) fn record_descriptors(
        &self,
        planar_slot: Option<usize>,
    ) -> RecordDescriptors<'a> {
        RecordDescriptors {
            device: self.device,
            pool: self.pool,
            params_set_layout: self.params_set_layout,
            planar_view: planar_slot
                .and_then(|s| self.planar_target_views.get(s).copied())
                .unwrap_or(self.stand_in_view),
            sampler: self.sampler,
        }
    }
}

// The resized post-SSR scene target + per-frame depth views a `rebuild` re-points
// into. `planar_target_views` are the resized per-distinct-plane mirror target
// views (the planar set is rebuilt just before this), re-pointed into each
// slotted record's binding 1.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct TransparentRebuildTargets<'a> {
    pub scene_views: &'a [vk::ImageView],
    pub scene_images: &'a [vk::Image],
    pub depth_views: &'a [vk::ImageView],
    pub planar_target_views: &'a [vk::ImageView],
    // Per-axis divisor of the glass reflection pre-pass; 1 traces in place.
    pub reflection_divisor: u32,
}

impl TransparentResources {
    // Build the render pass, the shared layouts, each live producer's pipelines +
    // records, the per-frame view ring, the scene snapshot and the per-frame
    // framebuffers. Called from `VkContext::new` when the world declares any
    // `GlassPanel` or `WaterSurface`.
    pub(in crate::vulkan) fn new(
        ctx: TransparentDeviceCtx,
        config: TransparentBuildConfig,
        scene: TransparentSceneTargets,
        content: TransparentContent,
        rt_setup: TransparentRtSetup,
    ) -> RenderResult<Self> {
        let TransparentDeviceCtx {
            alloc,
            instance,
            device,
            physical_device,
            command_pool,
            queue,
        } = ctx;
        let TransparentBuildConfig {
            frames,
            msaa_samples,
            width,
            height,
            global_set_layout,
            hot_reload,
            reflection_divisor,
        } = config;
        let TransparentSceneTargets {
            scene_views,
            scene_images,
            depth_views,
            sampler,
        } = scene;
        let TransparentRtSetup {
            rt_capable,
            vertex_buffer,
            index_buffer,
            rt_inputs,
            bindless_set_layout,
            bindless_pool_size,
        } = rt_setup;
        let msaa = msaa_samples != vk::SampleCountFlags::TYPE_1;
        let render_pass = create_transparent_render_pass(device, HDR_FORMAT)?;
        let reflection_render_pass = create_reflection_render_pass(device)?;
        let view_set_layout = create_descriptor_set_layout(device, &view_set_bindings())?;
        let params_set_layout = create_descriptor_set_layout(device, &params_set_bindings())?;
        let set_layouts = [
            view_set_layout.handle(),
            params_set_layout.handle(),
            global_set_layout,
        ];
        let pipeline_layout = {
            let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            device
                .create_pipeline_layout(&info)
                .map_err(|e| super::error::map_vk_result(e, "transparent pipeline layout"))?
        };

        // The shared RT layouts + descriptor ring, when the device is RT-capable. A
        // failure here leaves `rt` `None` and every producer keeps the probe /
        // planar path (mirrors DirectX's graceful fallback).
        let rt = if rt_capable {
            match build_transparent_rt(
                alloc,
                instance,
                device,
                physical_device,
                frames,
                RtSetLayouts {
                    view: view_set_layout.handle(),
                    params: params_set_layout.handle(),
                    global: global_set_layout,
                    bindless: bindless_set_layout,
                },
                TransparentRtGeometry {
                    vertex_buffer,
                    index_buffer,
                    rt_inputs,
                },
            ) {
                Ok(rt) => Some(rt),
                Err(e) => {
                    tracing::warn!(
                        "transparent RT setup failed ({e}); using the probe / planar path"
                    );
                    None
                }
            }
        } else {
            None
        };

        let snapshot = create_snapshot(alloc, device, command_pool, queue, width, height)?;

        // Per-frame view UBO ring (HOST_VISIBLE | HOST_COHERENT, mapped).
        let view_size = std::mem::size_of::<TransparentView>() as u64;
        let mut view_ubos = Vec::with_capacity(frames);
        for _ in 0..frames {
            view_ubos.push(alloc.create_buffer(
                view_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?);
        }

        let records = content.glass_panels.len() + content.water_surfaces.len();
        let descriptor_pool = create_descriptor_pool(device, frames, records)?;
        let view_layouts: Vec<_> = (0..frames * FrameViewSets::COUNT)
            .map(|_| view_set_layout.handle())
            .collect();
        let view_sets: Vec<FrameViewSets> =
            alloc_descriptor_sets(device, descriptor_pool.handle(), &view_layouts)?
                .chunks_exact(FrameViewSets::COUNT)
                .map(|sets| FrameViewSets {
                    scene: sets[0],
                    layers: [sets[1], sets[2]],
                })
                .collect();
        let empty_layer = upload_texture(
            &GpuUploadContext {
                alloc,
                device,
                command_pool,
                queue,
            },
            1,
            1,
            &[0u8; 4],
        )?;

        // Per-frame framebuffers targeting the scene image for that slot.
        let framebuffers =
            create_framebuffers(device, render_pass.handle(), scene_views, width, height)?;

        let producer_ctx = ProducerCtx {
            alloc,
            device,
            render_pass: render_pass.handle(),
            reflection_render_pass: reflection_render_pass.handle(),
            layout: pipeline_layout.handle(),
            rt_layout_flat: rt.as_ref().map(|r| r.layout_flat.handle()),
            rt_layout_textured: rt
                .as_ref()
                .and_then(|r| r.layout_textured.as_ref())
                .map(|l| l.handle()),
            pool: descriptor_pool.handle(),
            params_set_layout: params_set_layout.handle(),
            stand_in_view: empty_layer.view,
            planar_target_views: content.planar_target_views,
            sampler,
            msaa,
            hot_reload,
            bindless_pool_size,
            frames,
            // SAFETY: a property query on a live handle; it only reads.
            ubo_offset_alignment: unsafe {
                instance.get_physical_device_properties(physical_device)
            }
            .limits
            .min_uniform_buffer_offset_alignment,
        };
        let glass = if content.glass_panels.is_empty() {
            None
        } else {
            Some(super::glass::build_glass_producer(
                producer_ctx,
                content.glass_panels,
                content.glass_planar_slots,
            )?)
        };
        let water = if content.water_surfaces.is_empty() {
            None
        } else {
            Some(super::water::build_water_producer(
                producer_ctx,
                content.water_surfaces,
                content.water_planar_slots,
            )?)
        };

        // The see-through mesh producer, built only when a material opted in AND
        // the pass has RT pipeline layouts: the trace is the whole feature, so
        // without them there is nothing to build and those meshes stay opaque. A
        // shader-compile failure is non-fatal for the same reason -- it is logged
        // and the meshes keep the Layer 1 opaque-reflective look.
        let glass_mesh = match (
            content.seethrough_mesh_indices.is_empty(),
            producer_ctx.rt_layout_flat,
        ) {
            (false, Some(flat_layout)) => match super::glass::build_glass_mesh_producer(
                producer_ctx,
                flat_layout,
                content.seethrough_mesh_indices,
            ) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(
                        "see-through glass mesh pipeline build failed ({e});                          those meshes render opaque"
                    );
                    None
                }
            },
            _ => None,
        };

        let mut me = Self {
            render_pass,
            pipeline_layout,
            _view_set_layout: view_set_layout,
            _params_set_layout: params_set_layout,
            _descriptor_pool: descriptor_pool,
            view_ubos,
            view_sets,
            scene_images: scene_images.to_vec(),
            framebuffers,
            snapshot,
            glass,
            water,
            glass_mesh,
            rt,
            reflection_render_pass,
            reflection: None,
            empty_layer,
        };
        me.reflection =
            me.build_reflection_layers(alloc, device, width, height, reflection_divisor)?;
        for (sets, ubo) in me.view_sets.iter().zip(&me.view_ubos) {
            for set in [sets.scene, sets.layers[0], sets.layers[1]] {
                write_view_set_statics(device, set, ubo.buffer(), sampler);
            }
        }
        me.write_view_sets(device, depth_views);
        Ok(me)
    }

    // The reduced reflection layers for a `width` x `height` render at
    // `divisor`, or `None` when glass traces in place: a divisor of 1, or no
    // glass producer that can trace.
    fn build_reflection_layers(
        &self,
        alloc: &DeviceAllocator,
        device: &VkDevice,
        width: u32,
        height: u32,
        divisor: u32,
    ) -> RenderResult<Option<GlassReflectionLayers>> {
        let traced = self
            .glass
            .as_ref()
            .is_some_and(|p| p.reflection_flat_pso.is_some())
            || self.glass_mesh.is_some();
        if divisor <= 1 || !traced {
            return Ok(None);
        }
        let extent = vk::Extent2D {
            width: (width / divisor).max(1),
            height: (height / divisor).max(1),
        };
        GlassReflectionLayers::new(alloc, device, self.reflection_render_pass.handle(), extent)
            .map(Some)
    }

    // Write the images of every frame's set-0 variants. The scene pass reads both
    // reflection layers (the snapshot stands in while there are none, which the
    // shaders never read then); the first pre-pass layer peels behind the empty
    // layer and the second behind the first.
    fn write_view_sets(&self, device: &VkDevice, depth_views: &[vk::ImageView]) {
        let empty = self.empty_layer.view;
        let (scene, first) = match &self.reflection {
            Some(r) => ([r.layers[0].view, r.layers[1].view], r.layers[0].view),
            None => ([self.snapshot.view; 2], empty),
        };
        for (i, sets) in self.view_sets.iter().enumerate() {
            let inputs = |reflection| ViewSetImages {
                snapshot_view: self.snapshot.view,
                depth_view: depth_views[i.min(depth_views.len().saturating_sub(1))],
                reflection,
            };
            write_view_set_images(device, sets.scene, inputs(scene));
            write_view_set_images(device, sets.layers[0], inputs([empty; 2]));
            write_view_set_images(device, sets.layers[1], inputs([first, empty]));
        }
    }

    // True when the per-pixel RT pipelines are built (RT-capable device + the
    // shader compile + descriptor setup succeeded) for every live producer.
    // Single-sources the "the transparent pass can trace" half of
    // `VkContext::rt_transparent_active`: gating on the whole set is what keeps
    // the RT choice a per-frame one rather than a per-producer one, so the planar
    // mirror render the graph skips is never one a producer still needs. Mirrors
    // DirectX's `rt_pipelines_ready`.
    pub(in crate::vulkan) fn rt_pipelines_ready(&self) -> bool {
        self.rt.is_some()
            && self.glass.as_ref().is_none_or(|p| p.flat_rt_pso.is_some())
            && self.water.as_ref().is_none_or(|p| p.flat_rt_pso.is_some())
    }

    // The see-through meshes this pass was built over, or an empty slice when
    // the world declared none.
    pub(in crate::vulkan) fn seethrough_mesh_indices(&self) -> &[usize] {
        self.glass_mesh
            .as_ref()
            .map(|p| p.object_indices.as_slice())
            .unwrap_or_default()
    }

    // True when the see-through mesh pipelines are built, so the Layer 2 reroute
    // can engage as soon as RT is live. Independent of `rt.accel`, because the
    // init-time BLAS build has to exclude the meshes it will reroute before the
    // acceleration structure it gates on exists.
    pub(in crate::vulkan) fn mesh_pipelines_ready(&self) -> bool {
        self.glass_mesh.is_some()
    }

    // True when the textured RT layout exists AND every live producer built its
    // textured pipeline, so the whole pass can take the bindless hit-shading
    // variant. All-or-nothing across producers: the encoder binds one pipeline
    // layout for the pass, so a per-producer split would leave one drawing with
    // sets bound under an incompatible layout.
    fn rt_textured_ready(&self) -> bool {
        self.rt
            .as_ref()
            .is_some_and(|r| r.layout_textured.is_some())
            && self
                .glass
                .as_ref()
                .is_none_or(|p| p.textured_rt_pso.is_some())
            && self
                .water
                .as_ref()
                .is_none_or(|p| p.textured_rt_pso.is_some())
            && self
                .glass_mesh
                .as_ref()
                .is_none_or(|p| p.pipeline_textured.is_some())
    }

    // Re-point this frame's transparent RT descriptor set at the live TLAS +
    // geometry handles. A no-op when the RT pipelines are absent. Called from
    // `VkContext::rt_dynamic_update` alongside the RT-reflection pass's re-point,
    // so the transparent traces sample the same per-frame acceleration structure.
    pub(in crate::vulkan) fn wire_rt_dynamic(
        &mut self,
        device: &VkDevice,
        frame_idx: usize,
        dynamic: TransparentRtDynamic,
    ) {
        if let Some(rt) = self.rt.as_mut() {
            rt.wire_dynamic(device, frame_idx, dynamic);
        }
    }

    // Re-point the transparent RT set's shared static verts + indices at new
    // buffers, after an asset hot-reload replaced the shared geometry buffers. A
    // no-op when the RT pipelines are absent.
    pub(in crate::vulkan) fn wire_rt_geometry(
        &self,
        device: &VkDevice,
        vertex_buffer: vk::Buffer,
        index_buffer: vk::Buffer,
    ) {
        if let Some(rt) = self.rt.as_ref() {
            rt.rewire_geometry(device, vertex_buffer, index_buffer);
        }
    }

    // True when a visible water surface holds a planar slot, so the mirror
    // re-render has a consumer this frame even while the trace is live. Water
    // takes the mirror over its own trace (see `water.hlsl`), so this is what
    // `planar_pass_needed` reads; glass is deliberately not counted.
    pub(in crate::vulkan) fn water_planar_slot_live(&self) -> bool {
        self.water.as_ref().is_some_and(|p| {
            p.records
                .iter()
                .any(|r| r.visible && r.planar_slot.is_some())
        })
    }

    // True when any record of the pane or water producer is currently visible.
    // The mesh producer is not covered here: its visibility is per-frame state
    // that lives in `draw.objects`, so `VkContext::transparent_enabled` asks it
    // separately. Together they drive `FrameGraphInputs::transparent_enabled` and
    // the encoder early-out.
    pub(in crate::vulkan) fn any_visible(&self) -> bool {
        let live = |p: &Option<TransparentProducer>| {
            p.as_ref()
                .is_some_and(|p| p.records.iter().any(|r| r.visible))
        };
        live(&self.glass) || live(&self.water)
    }

    // Every visible record of the static producers plus this frame's mesh draws,
    // farthest first.
    fn draw_order(&self, meshes: &[[f32; 3]], cam: [f32; 3]) -> Vec<(Producer, usize)> {
        let centers = |p: &Option<TransparentProducer>| -> Vec<([f32; 3], bool)> {
            p.as_ref()
                .map(|p| p.records.iter().map(|r| (r.center, r.visible)).collect())
                .unwrap_or_default()
        };
        ordered_visible(&centers(&self.glass), &centers(&self.water), meshes, cam)
    }

    // The static producer a draw-order entry names. Mesh entries never reach
    // here: they resolve to a per-frame draw, not a record.
    fn producer(&self, kind: Producer) -> &TransparentProducer {
        match kind {
            Producer::Glass => self.glass.as_ref(),
            Producer::Water => self.water.as_ref(),
            Producer::GlassMesh => {
                unreachable!("mesh draws are per-frame and never resolve to a static record")
            }
        }
        .expect("the draw order only names live producers")
    }

    // Recreate the scene snapshot + per-frame framebuffers at new render dims +
    // re-point the snapshot (binding 1) and per-frame depth (binding 2) of every
    // view set. The pipelines, layouts, UBOs, record buffers, and render pass all
    // survive. Called from the swapchain-resize handler after the SSR / HDR
    // resolve targets have been rebuilt (so `scene_views` / `scene_images` carry
    // the new handles).
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        ctx: TransparentDeviceCtx,
        width: u32,
        height: u32,
        targets: TransparentRebuildTargets,
    ) -> RenderResult<()> {
        let TransparentDeviceCtx {
            alloc,
            device,
            command_pool,
            queue,
            ..
        } = ctx;
        let TransparentRebuildTargets {
            scene_views,
            scene_images,
            depth_views,
            planar_target_views,
            reflection_divisor,
        } = targets;
        let old = std::mem::replace(
            &mut self.snapshot,
            create_snapshot(alloc, device, command_pool, queue, width, height)?,
        );
        drop(old);

        self.framebuffers = create_framebuffers(
            device,
            self.render_pass.handle(),
            scene_views,
            width,
            height,
        )?;
        self.scene_images = scene_images.to_vec();

        self.reflection = None;
        self.reflection =
            self.build_reflection_layers(alloc, device, width, height, reflection_divisor)?;
        self.write_view_sets(device, depth_views);

        // Re-point each slotted record's planar binding (binding 1) at its slot's
        // resized target. Slotless records and the mesh sets hold the
        // resolution-independent stand-in, which a resize leaves in place.
        for producer in [self.glass.as_ref(), self.water.as_ref()]
            .into_iter()
            .flatten()
        {
            for r in &producer.records {
                if let Some(&view) = r.planar_slot.and_then(|s| planar_target_views.get(s)) {
                    write_planar_view(device, r.params_set, view);
                }
            }
        }
        Ok(())
    }

    // Destroy every owned GPU resource.
    pub(in crate::vulkan) fn destroy(&mut self, device: &VkDevice) {
        if let Some(mut rt) = self.rt.take() {
            rt.destroy(device);
        }
        self.glass = None;
        self.water = None;
        self.glass_mesh = None;
        self.reflection = None;
        self.empty_layer = GpuImage::null();
        self.view_ubos.clear();
        self.snapshot = GpuImage::null();
        self.framebuffers.clear();
        self.scene_images.clear();
    }
}

// One framebuffer per frame slot, each binding that slot's scene image view as
// the sole color attachment.
fn create_framebuffers(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    scene_views: &[vk::ImageView],
    width: u32,
    height: u32,
) -> RenderResult<Vec<OwnedFramebuffer>> {
    let mut out = Vec::with_capacity(scene_views.len());
    for &view in scene_views {
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(std::slice::from_ref(&view))
            .width(width.max(1))
            .height(height.max(1))
            .layers(1);
        let fb = device
            .create_framebuffer(&info)
            .map_err(|e| super::error::map_vk_result(e, "transparent framebuffer"))?;
        out.push(fb);
    }
    Ok(out)
}

// Refraction offset + Fresnel falloff for a see-through glass MESH. A `Material`
// carries no glass-specific tunables (unlike a `GlassPanel`), so these match the
// `GlassPanel` defaults: a gentle screen-space refraction and a fresnel power of
// 1 (subtle reflection head-on, full mirror at grazing). The same constants the
// other backends use, so a mesh reads the same everywhere.
const GLASS_MESH_REFRACTION: f32 = 0.02;
const GLASS_MESH_FRESNEL_POWER: f32 = 1.0;

impl VkContext {
    // Whether a material opted into Layer 2 see-through glass AND the device can
    // drive it (the mesh pipelines built). Independent of `rt.accel`, so it
    // answers "would the see-through path run if RT is on" -- used at the RT-BLAS
    // build, which must exclude the meshes it will reroute before the
    // acceleration structure it gates on exists. Data-driven: see-through is
    // opt-in per `Material::see_through`, so a scene with no see-through material
    // never engages Layer 2 and its transparent glass stays Layer 1 (opaque, low
    // roughness, reflective).
    pub(in crate::vulkan) fn seethrough_meshes_enabled(&self) -> bool {
        self.transparent
            .as_ref()
            .is_some_and(|t| t.mesh_pipelines_ready())
    }

    // Whether the see-through mesh (Layer 2) path is live this frame: the
    // pipelines built AND the pass can trace (`rt_transparent_active`, which needs
    // the TLAS). When false, those meshes render opaque + reflective in the main
    // pass (Layer 1) and the producer / opaque-skip / BLAS-exclude all stay inert.
    // Mirrors `DxContext::mesh_glass_active`.
    pub(in crate::vulkan) fn mesh_glass_active(&self) -> bool {
        self.seethrough_meshes_enabled() && self.rt_transparent_active()
    }

    // Whether any see-through mesh would actually draw this frame. Only then does
    // the mesh producer contribute, so the graph's Transparent node is not
    // scheduled for a world whose glass is all hidden or evicted.
    pub(in crate::vulkan) fn mesh_glass_visible(&self) -> bool {
        self.mesh_glass_active()
            && self.transparent.as_ref().is_some_and(|t| {
                t.seethrough_mesh_indices().iter().any(|&i| {
                    self.draw
                        .objects
                        .get(i)
                        .is_some_and(|o| o.visible && o.resident)
                })
            })
    }

    // Build this frame's see-through mesh draw list and write each mesh's params
    // into its block of the producer's ring. Only called while RT is live.
    //
    // Each mesh resolves its own LOD slice by camera distance exactly as the
    // opaque passes do, so a mesh rerouted here rasterizes the same triangles it
    // would have rasterized opaque.
    fn collect_mesh_draws(
        &self,
        transparent: &TransparentResources,
        frame_idx: usize,
        cam: [f32; 3],
    ) -> Vec<GlassMeshDraw> {
        let Some(producer) = transparent.glass_mesh.as_ref() else {
            return Vec::new();
        };
        let count = producer.object_indices.len();
        let Some(ring) = producer.params_buffers.get(frame_idx) else {
            return Vec::new();
        };
        let prefilter_mip_count = self.scene.prefilter_mip_count as f32;

        let mut draws = Vec::with_capacity(count);
        for (slot, &idx) in producer.object_indices.iter().enumerate() {
            let Some(obj) = self.draw.objects.get(idx) else {
                continue;
            };
            // The flag is re-read rather than trusted from the init list, so this
            // producer and the opaque-pass skip decide from the same live
            // predicate and cannot disagree about which meshes are rerouted.
            if !obj.visible || !obj.resident || obj.material.see_through == 0 {
                continue;
            }
            let center = [
                0.5 * (obj.bb_min[0] + obj.bb_max[0]),
                0.5 * (obj.bb_min[1] + obj.bb_max[1]),
                0.5 * (obj.bb_min[2] + obj.bb_max[2]),
            ];
            let d = lod::camera_distance(obj, cam);
            let (index_offset, index_count) = obj.active_lod(d);
            let t = obj.material.tint;
            let params = GlassMeshParams {
                model: obj.model,
                tint: [t[0], t[1], t[2], 0.0],
                opacity: obj.material.opacity,
                refraction_strength: GLASS_MESH_REFRACTION,
                fresnel_power: GLASS_MESH_FRESNEL_POWER,
                prefilter_mip_count,
            };
            ring.write_val((slot as u64 * producer.params_stride) as usize, &params);
            draws.push(GlassMeshDraw {
                index_offset: index_offset as u32,
                index_count: index_count as u32,
                base_vertex: obj.base_vertex,
                params_set: producer.params_sets[frame_idx * count + slot],
                center,
            });
        }
        draws
    }

    // Assemble the per-frame transparent view from the frame's jittered VP (the
    // matrix the main pass rasterized the depth buffer with, so a transparent
    // record's clip-space depth matches the stored main-depth) + camera position.
    // Mirrors `directx::graph_exec::build_transparent_view`.
    pub(in crate::vulkan) fn build_transparent_view(
        &self,
        vp: [[f32; 4]; 4],
        cam_pos: [f32; 3],
        time: f32,
    ) -> TransparentView {
        let (sun_dir, sun_color) = lights::glint_sun(&self.uniforms.light_uniforms);
        TransparentView {
            vp,
            inv_vp: mat4_inverse(vp),
            camera_pos: [cam_pos[0], cam_pos[1], cam_pos[2], 0.0],
            viewport: [
                self.targets.render_extent.width as f32,
                self.targets.render_extent.height as f32,
            ],
            time,
            prefilter_mip_count: self.scene.prefilter_mip_count as f32,
            sky_rot: self.view.sky_rot,
            sun_dir,
            sun_color,
        }
    }

    // Encode the transparent pass. Runs after `SsrResolve` and before
    // `TaaResolve` / `Upscale`. Snapshots the post-SSR scene into `snapshot` for
    // refractive taps, then draws every visible glass pane and water surface
    // back-to-front into the scene image with SRC_ALPHA blending; the manual
    // occlusion test samples the main depth. No-op when the world has no
    // transparent content or nothing is visible. Leaves the scene image
    // SHADER_READ_ONLY and the main depth DEPTH_STENCIL_ATTACHMENT_OPTIMAL for the
    // downstream stack.
    pub(in crate::vulkan) fn encode_transparent(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        view: &TransparentView,
        // Projection inputs for the per-pixel RT reflection trace's RtParams (the
        // same values the RT-reflection resolve uses); only consumed on the RT path.
        fov_y_radians: f32,
        aspect: f32,
    ) -> RenderResult<()> {
        let Some(transparent) = self.transparent.as_ref() else {
            return Ok(());
        };
        let cam = [view.camera_pos[0], view.camera_pos[1], view.camera_pos[2]];

        // Per-pixel RT reflection is selected over the probe / planar path when RT
        // is live (the scene TLAS is built) AND every live producer's RT pipelines
        // compiled -- single-sourced via `rt_transparent_active`, the same predicate
        // `graph_exec` uses to skip the planar mirror re-render, so the two always
        // agree. The textured variant additionally needs the bindless albedo/normal
        // pool the GPU-cull path populates; without it the flat-tint trace runs.
        // Mirrors DirectX's selection.
        let rt_live = self.rt_transparent_active();
        let textured =
            rt_live && self.cull.bindless_pipeline.is_some() && transparent.rt_textured_ready();

        // This frame's see-through mesh draws. Empty unless RT is live: the
        // per-pixel trace is the feature, and with RT off those meshes rasterize
        // opaque in the main pass instead.
        let mesh_draws = if rt_live {
            self.collect_mesh_draws(transparent, frame_idx, cam)
        } else {
            Vec::new()
        };
        let mesh_centers: Vec<[f32; 3]> = mesh_draws.iter().map(|d| d.center).collect();
        let order = transparent.draw_order(&mesh_centers, cam);
        if order.is_empty() {
            return Ok(());
        }

        // The glass reflection pre-pass runs when RT is live, the reduced layers
        // exist, and a traced glass surface draws this frame.
        let reflection_layers = transparent.reflection.as_ref().filter(|_| {
            rt_live
                && order
                    .iter()
                    .any(|&(kind, _)| matches!(kind, Producer::Glass | Producer::GlassMesh))
        });

        let device = &self.hw.device;
        let extent = self.targets.render_extent;
        let scene_image = *transparent
            .scene_images
            .get(frame_idx)
            .ok_or_else(|| RenderError::Other("transparent: scene image index OOB".to_string()))?;
        let snapshot = transparent.snapshot.image;

        // Upload this frame's view UBO.
        transparent
            .view_ubos
            .get(frame_idx)
            .ok_or_else(|| RenderError::Other("transparent: view_ubos index OOB".to_string()))?
            .write_val(0, view);

        // On the RT path, upload this frame's RtParams (sun + ray tunables) into the
        // shared RtParams ring, mirroring `encode_rt_reflections`'s build. The
        // settings come from the RT-reflection pass (always present when `rt_live`).
        if rt_live {
            let rtres = self.rt_reflections.as_ref().ok_or_else(|| {
                RenderError::Other("transparent rt_live but rt_reflections missing".to_string())
            })?;
            let rt = transparent.rt.as_ref().ok_or_else(|| {
                RenderError::Other("transparent rt_live but rt pipelines missing".to_string())
            })?;
            let v = self.view.matrix;
            let inv_view_rot = [
                [v[0][0], v[1][0], v[2][0], 0.0],
                [v[0][1], v[1][1], v[2][1], 0.0],
                [v[0][2], v[1][2], v[2][2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let params = rtres.settings.params(RtParamsInputs {
                fov_y_radians,
                aspect,
                inv_view_rot,
                cam_pos: cam,
                sun_dir: self.fog.sun_dir,
                sun_color: self.fog.sun_color,
                prefilter_mip_count: self.scene.prefilter_mip_count as f32,
                sky_rot: self.view.sky_rot,
            });
            // Traced glass reads the reduced layers back only when the pre-pass
            // below fills them this frame; otherwise it traces in place.
            let params = RtParams {
                trace_divisor: if reflection_layers.is_some() {
                    params.trace_divisor
                } else {
                    1.0
                },
                ..params
            };
            rt.params_buffers[frame_idx].write_val(0, &params);
        }

        let color_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let color_barrier = |image: vk::Image,
                             old: vk::ImageLayout,
                             new: vk::ImageLayout,
                             src: vk::AccessFlags,
                             dst: vk::AccessFlags| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(src)
                .dst_access_mask(dst)
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_range)
        };

        // 1) Open the scene image + snapshot for the refraction snapshot copy.
        // The src scopes order the scene's last writer (SSR resolve / particles
        // color write) and the prior frame's snapshot read ahead of the
        // transfer.
        let scene_to_src = color_barrier(
            scene_image,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::TRANSFER_READ,
        );
        let snapshot_to_dst = color_barrier(
            snapshot,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::TRANSFER_WRITE,
        );
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[scene_to_src, snapshot_to_dst],
            );
            let region = vk::ImageCopy::default()
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                });
            device.cmd_copy_image(
                cmd,
                scene_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                snapshot,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
            );
        }

        // 2) Close the snapshot for the fragment read and restore the scene
        // image to SHADER_READ_ONLY, so the render pass's color LOAD matches
        // its declared initial layout. Main depth is already sampled here: the
        // graph transitions it once for the whole decoration run.
        let snapshot_to_read = color_barrier(
            snapshot,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        let scene_to_read = color_barrier(
            scene_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::TRANSFER_READ,
            vk::AccessFlags::COLOR_ATTACHMENT_READ,
        );
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[snapshot_to_read, scene_to_read],
            );
        }

        // Every producer shares every set layout, so one pipeline layout binds the
        // view / global / RT sets for the whole pass and only the pipeline changes
        // across the draw loop.
        let layout = match (rt_live, transparent.rt.as_ref()) {
            (true, Some(r)) if textured => r
                .layout_textured
                .as_ref()
                .expect("textured implies a textured layout")
                .handle(),
            (true, Some(r)) => r.layout_flat.handle(),
            _ => transparent.pipeline_layout.handle(),
        };
        let frame = TransparentFrame {
            cmd,
            frame_idx,
            layout,
            rt_live,
            textured,
            order: &order,
            mesh_draws: &mesh_draws,
        };

        if let Some(layers) = reflection_layers {
            self.encode_glass_reflection_layers(transparent, layers, &frame);
        }

        // 3) The render pass: LOAD the scene color, draw each visible record
        // back-to-front, STORE. The negative-height viewport matches the main
        // pass so the manual depth test + refraction taps line up at pixel
        // coordinates.
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(transparent.render_pass.handle())
            .framebuffer(transparent.framebuffers[frame_idx].handle())
            .render_area(vk::Rect2D::default().extent(extent));
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            set_flipped_viewport(device, cmd, extent);
        }
        self.bind_transparent_sets(transparent, &frame, transparent.view_sets[frame_idx].scene);
        let mut bound: Option<Producer> = None;
        for &(kind, i) in &order {
            if bound != Some(kind) {
                let pipeline = match kind {
                    Producer::GlassMesh => transparent
                        .glass_mesh
                        .as_ref()
                        .expect("the draw order only names live producers")
                        .pipeline(textured),
                    _ => transparent.producer(kind).pipeline(rt_live, textured),
                };
                // SAFETY: `cmd` is a command buffer in the recording state, and the pipeline is
                // live for the call.
                unsafe {
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.handle(),
                    );
                }
                bound = Some(kind);
            }
            self.draw_transparent_entry(transparent, &frame, kind, i);
        }
        // SAFETY: `cmd` is a command buffer in the recording state inside the render pass begun
        // above.
        unsafe { device.cmd_end_render_pass(cmd) };

        Ok(())
    }

    // Draw the traced glass into the two reduced reflection layers: each layer
    // clears its color and the shared depth, then draws every glass surface
    // behind the layer bound at set-0 binding 3, keeping the nearest. Water
    // traces in place and is skipped.
    fn encode_glass_reflection_layers(
        &self,
        transparent: &TransparentResources,
        layers: &GlassReflectionLayers,
        frame: &TransparentFrame,
    ) {
        let device = &self.hw.device;
        let clears = [
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
        for layer in 0..2 {
            let rp_begin = vk::RenderPassBeginInfo::default()
                .render_pass(transparent.reflection_render_pass.handle())
                .framebuffer(layers.framebuffers[layer].handle())
                .render_area(vk::Rect2D::default().extent(layers.extent))
                .clear_values(&clears);
            // SAFETY: `frame.cmd` is a command buffer in the recording state, and every handle
            // and slice these commands name is live for the call.
            unsafe {
                device.cmd_begin_render_pass(frame.cmd, &rp_begin, vk::SubpassContents::INLINE);
                set_flipped_viewport(device, frame.cmd, layers.extent);
            }
            self.bind_transparent_sets(
                transparent,
                frame,
                transparent.view_sets[frame.frame_idx].layers[layer],
            );
            for &(kind, i) in frame.order {
                let pipeline = match kind {
                    Producer::GlassMesh => transparent
                        .glass_mesh
                        .as_ref()
                        .map(|m| m.reflection_pipeline(frame.textured)),
                    _ => transparent
                        .producer(kind)
                        .reflection_pipeline(frame.textured),
                };
                let Some(pipeline) = pipeline else {
                    continue;
                };
                // SAFETY: `frame.cmd` is a command buffer in the recording state, and the
                // pipeline is live for the call.
                unsafe {
                    device.cmd_bind_pipeline(
                        frame.cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.handle(),
                    );
                }
                self.draw_transparent_entry(transparent, frame, kind, i);
            }
            // SAFETY: `frame.cmd` is a command buffer in the recording state inside the render
            // pass begun above.
            unsafe { device.cmd_end_render_pass(frame.cmd) };
        }
    }

    // Bind the sets every transparent draw shares: `view_set` at 0, the frame's
    // global set at 2, and while RT is live the RT geometry at 3 and (textured)
    // the bindless pool at 4.
    fn bind_transparent_sets(
        &self,
        transparent: &TransparentResources,
        frame: &TransparentFrame,
        view_set: vk::DescriptorSet,
    ) {
        let device = &self.hw.device;
        let bind = |index: u32, set: vk::DescriptorSet| {
            // SAFETY: `frame.cmd` is a command buffer in the recording state, and the set and
            // layout are live for the call.
            unsafe {
                device.cmd_bind_descriptor_sets(
                    frame.cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    frame.layout,
                    index,
                    std::slice::from_ref(&set),
                    &[],
                );
            }
        };
        bind(0, view_set);
        // The per-frame global set (set 2): the fragment shaders reflect its probe
        // set / cube array (bindings 7 / 8) + sky prefilter cube (binding 5).
        bind(2, self.descriptors.global_sets[frame.frame_idx]);
        if frame.rt_live {
            let r = transparent
                .rt
                .as_ref()
                .expect("rt_live implies the RT pipelines");
            // set 3: this frame's RT geometry (TLAS + geom table + the static +
            // skinned vertex/index buffers).
            bind(3, r.sets[frame.frame_idx]);
            if frame.textured {
                // set 4: the bindless albedo/normal pool for textured hit shading
                // (the same set the main bindless pass binds).
                bind(4, self.cull.bindless_sets[frame.frame_idx]);
            }
        }
    }

    // Bind one draw-order entry's params set and geometry and draw it under the
    // bound pipeline. A mesh draws its DrawObject slice of the shared scene
    // buffers; a pane or water surface draws its record's own pair.
    fn draw_transparent_entry(
        &self,
        transparent: &TransparentResources,
        frame: &TransparentFrame,
        kind: Producer,
        i: usize,
    ) {
        let device = &self.hw.device;
        let cmd = frame.cmd;
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            if kind == Producer::GlassMesh {
                let d = &frame.mesh_draws[i];
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    frame.layout,
                    1,
                    std::slice::from_ref(&d.params_set),
                    &[],
                );
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
                device.cmd_draw_indexed(cmd, d.index_count, 1, d.index_offset, d.base_vertex, 0);
            } else {
                let r = &transparent.producer(kind).records[i];
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    frame.layout,
                    1,
                    std::slice::from_ref(&r.params_set),
                    &[],
                );
                device.cmd_bind_vertex_buffers(cmd, 0, &[r.vertex_buffer.buffer()], &[0]);
                device.cmd_bind_index_buffer(
                    cmd,
                    r.index_buffer.buffer(),
                    0,
                    vk::IndexType::UINT16,
                );
                device.cmd_draw_indexed(cmd, r.index_count, 1, 0, 0, 0);
            }
        }
        self.inc_draw_calls(1);
    }
}

// One frame's transparent recording state, shared by the reflection pre-pass
// and the scene pass.
struct TransparentFrame<'a> {
    cmd: vk::CommandBuffer,
    frame_idx: usize,
    layout: vk::PipelineLayout,
    rt_live: bool,
    textured: bool,
    order: &'a [(Producer, usize)],
    mesh_draws: &'a [GlassMeshDraw],
}

// Set the negative-height viewport the main pass rasterizes with over `extent`,
// so the fragment positions and the manual depth test line up at pixel
// coordinates, plus a matching scissor.
fn set_flipped_viewport(device: &VkDevice, cmd: vk::CommandBuffer, extent: vk::Extent2D) {
    let vp = vk::Viewport {
        x: 0.0,
        y: extent.height as f32,
        width: extent.width as f32,
        height: -(extent.height as f32),
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let scissor = vk::Rect2D::default().extent(extent);
    // SAFETY: `cmd` is a command buffer in the recording state, and the viewport and scissor
    // slices are live for the call.
    unsafe {
        device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
        device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
    }
}
