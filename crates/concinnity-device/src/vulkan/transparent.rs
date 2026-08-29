// src/vulkan/transparent.rs
//
// The engine's `PassId::Transparent` slot on the Vulkan backend: one render
// pass, drawn after the SSR resolve and before TAA, with two producers -- glass
// panes (`glass.rs`) and water surfaces (`water.rs`). Each contributes records
// built once at init; the pass snapshots the pre-transparent scene, orders every
// record of both producers back-to-front by camera distance, and draws them into
// the post-SSR scene image (`SsrResources::output` when SSR is on, else
// `hdr_resolve_images[frame]`), alpha-blending over it. Downstream TAA / bloom /
// composite pick the translucent geometry up unchanged.
//
// One pass rather than one per producer, mirroring the Metal and DirectX
// backends: the scene snapshot the refraction taps is a full render-resolution
// HDR image and a copy of it every frame, so a second one would be pure waste;
// and a single ordering over both producers is what puts a pane standing in a
// pool on the correct side of the water.
//
// The producers also share every descriptor set layout and pipeline layout,
// because `glass.slang` and `water.slang` declare the same bindings on purpose:
// the view set (0) carries the per-frame view UBO plus the snapshot and main
// depth, the params set (1) one record's uniforms plus its planar reflection
// target, the global set (2) is the forward one the probe / sky taps read, and
// the RT variants add the trace's geometry (3) and the bindless pool (4).
//
// Same uniform layouts, back-to-front ordering and manual depth-occlusion test
// as the DirectX and Metal hosts.

use ash::vk;
use concinnity_core::gfx::transform::mat4_inverse;

use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedFramebuffer, OwnedPipeline, OwnedPipelineLayout, OwnedRenderPass,
    OwnedSetLayout, VkDevice,
};

use super::allocator::{DeviceAllocator, PooledBuffer};
use crate::components::{GlassPanel, WaterSurface};
use crate::gfx::mesh_payload::Vertex;

use crate::gfx::render_types::RtParams;
use crate::gfx::rt_reflections::RtParamsInputs;

use super::context::{HDR_FORMAT, VkContext};
use super::pipeline::spv_module;
use super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::texture::{
    GpuImage, ImageSpec, LayoutTransition, SubresourceRange, create_image, create_image_view,
    one_shot_submit, transition_image_layout_range,
};

// `TransparentView` (the per-frame view UBO) is a GPU-free layout struct that
// lives in `core::render`; re-export it so the encode path and the graph's
// view builder can keep naming it through this module.
use concinnity_core::render::uniforms::GlassMeshParams;
pub(in crate::vulkan) use concinnity_core::render::uniforms::TransparentView;

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
// rebuild.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct TransparentRtDynamic {
    pub tlas: vk::AccelerationStructureKHR,
    pub geom_buffer: vk::Buffer,
    pub geom_size: vk::DeviceSize,
    pub deformed: vk::Buffer,
    pub skinned_indices: vk::Buffer,
}

// Which producer a record belongs to, and so which pipeline draws it. The
// records themselves are identical in shape, so this is the only thing the
// combined draw loop needs to tell them apart.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::vulkan) enum Producer {
    Glass,
    Water,
    GlassMesh,
}

// Per-record GPU state: the static world-space quad (glass) or origin-centred
// grid (water) VB + IB, the per-record params UBO + its descriptor set, and the
// visibility flag.
pub(in crate::vulkan) struct TransparentRecord {
    vertex_buffer: PooledBuffer,
    index_buffer: PooledBuffer,
    index_count: u32,
    params_ubo: PooledBuffer,
    // Byte size of the record's uniform block, so the resize re-point can
    // rewrite binding 0 with the same range the initial write used.
    params_size: u64,
    params_set: vk::DescriptorSet,
    visible: bool,
    // World-space centre, used for the back-to-front camera-distance sort.
    centre: [f32; 3],
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
    pub centre: [f32; 3],
    pub planar_slot: Option<usize>,
}

// The descriptor plumbing a record needs: the pool + layout its params set comes
// from, the planar target it samples (or the snapshot stand-in), and the linear
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
    ) -> Result<Self, String> {
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
            params_ubo,
            params_size: upload.params.len() as u64,
            params_set,
            visible: upload.visible,
            centre: upload.centre,
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
// ring, and its world-space centre for the back-to-front sort.
struct GlassMeshDraw {
    index_offset: u32,
    index_count: u32,
    base_vertex: i32,
    params_set: vk::DescriptorSet,
    centre: [f32; 3],
}

impl GlassMeshProducer {
    // Allocate the per-frame params ring and one descriptor set per (frame,
    // mesh) pointing at that mesh's block of it, then take ownership of the
    // pipelines. The sets come from a pool of this producer's own rather than the
    // pass's, whose size is fixed to the pane + water record count.
    //
    // Each set reuses the shared params layout, so its planar binding (1) is
    // written with the snapshot stand-in: a mesh never samples a planar
    // reflection, but the layout the pipeline was built against still declares
    // the binding.
    pub(in crate::vulkan) fn new(
        ctx: &ProducerCtx,
        pipeline_flat: OwnedPipeline,
        pipeline_textured: Option<OwnedPipeline>,
        object_indices: Vec<usize>,
    ) -> Result<Self, String> {
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
        let sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: sets_needed,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: sets_needed,
            },
        ];
        let pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(sets_needed.max(1))
                    .pool_sizes(&sizes),
            )
            .map_err(|e| format!("glass mesh descriptor pool: {e}"))?;
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
                    ctx.snapshot_view,
                    ctx.sampler,
                );
            }
        }

        Ok(Self {
            pipeline_flat,
            pipeline_textured,
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
    view_sets: Vec<vk::DescriptorSet>,

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
    // Linear sampler bound alongside the snapshot (binding 1) and the main
    // depth (binding 2). Borrowed from `VkContext`; not owned, never destroyed
    // here.
    sampler: vk::Sampler,

    glass: Option<TransparentProducer>,
    water: Option<TransparentProducer>,
    glass_mesh: Option<GlassMeshProducer>,

    rt: Option<TransparentRt>,
}

// Round `size` up to a multiple of `align` (a power of two, from the device's
// `minUniformBufferOffsetAlignment`). Pure; unit tested.
fn align_up(size: u64, align: u64) -> u64 {
    let align = align.max(1);
    size.div_ceil(align) * align
}

// World-space distance from the camera to a record centre. Larger = farther =
// drawn first. Pure; unit tested.
fn sort_distance(centre: [f32; 3], cam: [f32; 3]) -> f32 {
    let dx = centre[0] - cam[0];
    let dy = centre[1] - cam[1];
    let dz = centre[2] - cam[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// Every visible record of every producer, ordered farthest-camera-distance
// first. Pure; unit tested. Invisible records are excluded, and the producers
// interleave so a pane standing in a pool composites in the right order; the
// visible set is sorted via the shared `gfx::transparent::back_to_front_order`.
//
// The mesh slice is already filtered to this frame's visible meshes (the encoder
// builds it), so every entry it carries is live.
fn ordered_visible(
    glass: &[([f32; 3], bool)],
    water: &[([f32; 3], bool)],
    meshes: &[[f32; 3]],
    cam: [f32; 3],
) -> Vec<(Producer, usize)> {
    let live_of = |records: &[([f32; 3], bool)], kind: Producer| -> Vec<(Producer, usize)> {
        records
            .iter()
            .enumerate()
            .filter(|(_, (_, vis))| *vis)
            .map(|(i, _)| (kind, i))
            .collect()
    };
    let live: Vec<(Producer, usize)> = live_of(glass, Producer::Glass)
        .into_iter()
        .chain(live_of(water, Producer::Water))
        .chain((0..meshes.len()).map(|i| (Producer::GlassMesh, i)))
        .collect();
    let dists: Vec<f32> = live
        .iter()
        .map(|&(kind, i)| {
            let centre = match kind {
                Producer::Glass => glass[i].0,
                Producer::Water => water[i].0,
                Producer::GlassMesh => meshes[i],
            };
            sort_distance(centre, cam)
        })
        .collect();
    crate::gfx::transparent::back_to_front_order(&dists)
        .into_iter()
        .map(|oi| live[oi])
        .collect()
}

fn create_rt_set_layout(device: &VkDevice) -> Result<OwnedSetLayout, String> {
    let frag = vk::ShaderStageFlags::FRAGMENT;
    create_descriptor_set_layout(
        device,
        &[
            (0, vk::DescriptorType::UNIFORM_BUFFER, frag),
            (1, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, frag),
            (2, vk::DescriptorType::STORAGE_BUFFER, frag),
            (3, vk::DescriptorType::STORAGE_BUFFER, frag),
            (4, vk::DescriptorType::STORAGE_BUFFER, frag),
            (5, vk::DescriptorType::STORAGE_BUFFER, frag),
            (6, vk::DescriptorType::STORAGE_BUFFER, frag),
        ],
    )
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
    // `VkContext::rt_dynamic_update` (the current frame's set is fence-gated). The
    // deformed buffer is always a valid handle (the accel data holds a 1-element
    // dummy when there is no skinned geometry); `skinned_indices` is null until the
    // first skinned rebuild, in which case the 1-element dummy SSBO binds so the
    // descriptor stays valid. Mirrors `post::rt_reflections::wire_dynamic`.
    fn wire_dynamic(&self, device: &VkDevice, frame_idx: usize, dynamic: TransparentRtDynamic) {
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
) -> Result<TransparentRt, String> {
    let set_layout = create_rt_set_layout(device)?;

    let flat_layouts = [
        layouts.view,
        layouts.params,
        layouts.global,
        set_layout.handle(),
    ];
    let layout_flat = device
        .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&flat_layouts))
        .map_err(|e| format!("transparent rt flat pipeline layout: {e}"))?;
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
                    .map_err(|e| format!("transparent rt textured pipeline layout: {e}"))?,
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

    // Pool: per-frame sets, each 1 UBO + 1 TLAS + 5 SSBO (geom, verts, indices,
    // deformed verts, skinned indices).
    let f = frames as u32;
    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(f),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(f),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(f * 5),
    ];
    let pool = device
        .create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(f),
        )
        .map_err(|e| format!("transparent rt descriptor pool: {e}"))?;
    let set_handles: Vec<_> = (0..frames).map(|_| set_layout.handle()).collect();
    let sets = alloc_descriptor_sets(device, pool.handle(), &set_handles)?;

    // 1-element dummy SSBO for the skinned-index binding when there is no skinned
    // geometry.
    let dummy_ssbo = alloc.create_buffer(
        16,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    let rt = TransparentRt {
        _set_layout: set_layout,
        layout_flat,
        layout_textured,
        params_buffers,
        sets,
        _pool: pool,
        dummy_ssbo,
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
) -> Result<OwnedRenderPass, String> {
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
        .map_err(|e| format!("transparent render pass: {e}"))
}

// Set 0: the per-frame view UBO (0), the scene snapshot (1) and this frame's
// main depth (2). The view UBO is visible to the vertex stage as well: both
// producers project through `vp`, and water reads `time` there for its wave
// phase.
fn create_view_set_layout(device: &VkDevice) -> Result<OwnedSetLayout, String> {
    let frag = vk::ShaderStageFlags::FRAGMENT;
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
            .stage_flags(frag),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(frag),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    device
        .create_descriptor_set_layout(&info)
        .map_err(|e| format!("transparent view set layout: {e}"))
}

// Set 1: one record's params UBO (0) and the planar reflection target it samples
// (1). The UBO is visible to the vertex stage because the water vertex stage
// reads its wave table out of it.
fn create_params_set_layout(device: &VkDevice) -> Result<OwnedSetLayout, String> {
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
        .map_err(|e| format!("transparent params set layout: {e}"))
}

fn create_descriptor_pool(
    device: &VkDevice,
    frames: usize,
    records: usize,
) -> Result<OwnedDescriptorPool, String> {
    let f = frames as u32;
    let r = records as u32;
    let sizes = [
        // view UBO per frame + params UBO per record.
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: f + r,
        },
        // snapshot + depth per per-frame view set, plus one planar target per record.
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 2 * f + r,
        },
    ];
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(f + r)
        .pool_sizes(&sizes);
    device
        .create_descriptor_pool(&info)
        .map_err(|e| format!("transparent descriptor pool: {e}"))
}

// Write one per-frame view set: the view UBO (binding 0), the shared scene
// snapshot (binding 1), and this frame's main-depth view (binding 2).
fn write_view_set(
    device: &VkDevice,
    set: vk::DescriptorSet,
    view_ubo: vk::Buffer,
    snapshot_view: vk::ImageView,
    depth_view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let view_info = vk::DescriptorBufferInfo::default()
        .buffer(view_ubo)
        .offset(0)
        .range(std::mem::size_of::<TransparentView>() as u64);
    let img = |view: vk::ImageView| {
        vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)
            .sampler(sampler)
    };
    let snapshot_info = img(snapshot_view);
    let depth_info = img(depth_view);
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
            .image_info(std::slice::from_ref(&snapshot_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&depth_info)),
    ];
    // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every set
    // and resource it names belongs to this device.
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

// Write a record's params set: its uniform block (binding 0) and the planar
// reflection target it samples (binding 1) -- its slot's mirror render, or the
// snapshot stand-in for a slotless record (the shaders gate on the `planar` flag).
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
    let planar_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(planar_view)
        .sampler(sampler);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&planar_info)),
    ];
    // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every set
    // and resource it names belongs to this device.
    unsafe { device.update_descriptor_sets(&writes, &[]) };
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
) -> Result<OwnedPipeline, String> {
    let vert = spv_module(device, vert_spv)?;
    let frag = spv_module(device, frag_spv)?;
    let entry = std::ffi::CString::new("main").unwrap();
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
    // No depth attachment: the fragment shader does the manual occlusion test.
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
        .map_err(|e| format!("create transparent pipeline: {e}"))?;
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
) -> Result<GpuImage, String> {
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
    // set + cube array). Bound as set 2 so the fragment shaders reflect the probe
    // set / sky prefilter cube; the pipeline layout must reference it even though
    // the pass only samples bindings 5 / 7 / 8. `probe_cube_count` is that
    // layout's binding-8 descriptor count, sizing the fragments' cube array.
    pub global_set_layout: vk::DescriptorSetLayout,
    pub probe_cube_count: u32,
    pub hot_reload: bool,
}

// The post-SSR scene target per frame slot plus the per-frame main-depth views.
// `scene_views` / `scene_images` are the post-SSR scene target per frame slot
// (SSR output repeated, or `hdr_resolve_images[i]`); `depth_views` are the main-
// depth views the manual occlusion test samples. `sampler` is the linear sampler
// bound alongside the snapshot + depth.
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
// records allocate from, the snapshot stand-in, the sampler, and the shader
// assembly inputs.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct ProducerCtx<'a> {
    pub alloc: &'a DeviceAllocator,
    pub device: &'a VkDevice,
    pub render_pass: vk::RenderPass,
    pub layout: vk::PipelineLayout,
    pub rt_layout_flat: Option<vk::PipelineLayout>,
    pub rt_layout_textured: Option<vk::PipelineLayout>,
    pub pool: vk::DescriptorPool,
    pub params_set_layout: vk::DescriptorSetLayout,
    pub snapshot_view: vk::ImageView,
    pub planar_target_views: &'a [vk::ImageView],
    pub sampler: vk::Sampler,
    pub msaa: bool,
    pub hot_reload: bool,
    pub probe_cube_count: u32,
    pub bindless_pool_size: usize,
    // Ring depth, for the mesh producer's per-frame params buffers.
    pub frames: usize,
    // The device's `minUniformBufferOffsetAlignment`, which the mesh producer's
    // per-mesh params blocks must be spaced by.
    pub ubo_offset_alignment: u64,
}

impl<'a> ProducerCtx<'a> {
    // The descriptor plumbing for one record, resolving its planar slot to the
    // mirror target it samples (or the snapshot stand-in).
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
                .unwrap_or(self.snapshot_view),
            sampler: self.sampler,
        }
    }
}

// The resized post-SSR scene target + per-frame depth views a `rebuild` re-points
// into. `planar_target_views` are the resized per-distinct-plane mirror target
// views (the planar set is rebuilt just before this), re-pointed into each
// record's binding 1. The sampler is borrowed from `VkContext` and survives on
// the resource.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct TransparentRebuildTargets<'a> {
    pub scene_views: &'a [vk::ImageView],
    pub scene_images: &'a [vk::Image],
    pub depth_views: &'a [vk::ImageView],
    pub planar_target_views: &'a [vk::ImageView],
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
    ) -> Result<Self, String> {
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
            probe_cube_count,
            hot_reload,
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
        let view_set_layout = create_view_set_layout(device)?;
        let params_set_layout = create_params_set_layout(device)?;
        let set_layouts = [
            view_set_layout.handle(),
            params_set_layout.handle(),
            global_set_layout,
        ];
        let pipeline_layout = {
            let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            device
                .create_pipeline_layout(&info)
                .map_err(|e| format!("transparent pipeline layout: {e}"))?
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
        let view_layouts: Vec<_> = (0..frames).map(|_| view_set_layout.handle()).collect();
        let view_sets = alloc_descriptor_sets(device, descriptor_pool.handle(), &view_layouts)?;
        for (i, &set) in view_sets.iter().enumerate() {
            write_view_set(
                device,
                set,
                view_ubos[i].buffer(),
                snapshot.view,
                depth_views[i.min(depth_views.len().saturating_sub(1))],
                sampler,
            );
        }

        // Per-frame framebuffers targeting the scene image for that slot.
        let framebuffers =
            create_framebuffers(device, render_pass.handle(), scene_views, width, height)?;

        let producer_ctx = ProducerCtx {
            alloc,
            device,
            render_pass: render_pass.handle(),
            layout: pipeline_layout.handle(),
            rt_layout_flat: rt.as_ref().map(|r| r.layout_flat.handle()),
            rt_layout_textured: rt
                .as_ref()
                .and_then(|r| r.layout_textured.as_ref())
                .map(|l| l.handle()),
            pool: descriptor_pool.handle(),
            params_set_layout: params_set_layout.handle(),
            snapshot_view: snapshot.view,
            planar_target_views: content.planar_target_views,
            sampler,
            msaa,
            hot_reload,
            probe_cube_count,
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

        Ok(Self {
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
            sampler,
            glass,
            water,
            glass_mesh,
            rt,
        })
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
    // can engage as soon as RT is live. Independent of `rt_accel`, because the
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
        &self,
        device: &VkDevice,
        frame_idx: usize,
        dynamic: TransparentRtDynamic,
    ) {
        if let Some(rt) = self.rt.as_ref() {
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
    // takes the mirror over its own trace (see `water.slang`), so this is what
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
        let centres = |p: &Option<TransparentProducer>| -> Vec<([f32; 3], bool)> {
            p.as_ref()
                .map(|p| p.records.iter().map(|r| (r.centre, r.visible)).collect())
                .unwrap_or_default()
        };
        ordered_visible(&centres(&self.glass), &centres(&self.water), meshes, cam)
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
    ) -> Result<(), String> {
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

        for (i, &set) in self.view_sets.iter().enumerate() {
            write_view_set(
                device,
                set,
                self.view_ubos[i].buffer(),
                self.snapshot.view,
                depth_views[i.min(depth_views.len().saturating_sub(1))],
                self.sampler,
            );
        }

        // Re-point each record's planar binding (binding 1) at its slot's resized
        // target, or the new snapshot for a slotless record (the moved snapshot
        // view must be refreshed there too, even though the shader never samples
        // it).
        for producer in [self.glass.as_ref(), self.water.as_ref()]
            .into_iter()
            .flatten()
        {
            for r in &producer.records {
                let planar_view = r
                    .planar_slot
                    .and_then(|s| planar_target_views.get(s).copied())
                    .unwrap_or(self.snapshot.view);
                write_params_set(
                    device,
                    r.params_set,
                    r.params_ubo.buffer(),
                    r.params_size,
                    planar_view,
                    self.sampler,
                );
            }
        }
        Ok(())
    }

    // Destroy every owned GPU resource. The `sampler` is borrowed from
    // `VkContext` and is not destroyed here.
    pub(in crate::vulkan) fn destroy(&mut self, device: &VkDevice) {
        if let Some(mut rt) = self.rt.take() {
            rt.destroy(device);
        }
        self.glass = None;
        self.water = None;
        self.glass_mesh = None;
        self.view_ubos.clear();
        self.snapshot = GpuImage::null();
        self.framebuffers.clear();
        self.scene_images.clear();
    }
}

// One framebuffer per frame slot, each binding that slot's scene image view as
// the sole colour attachment.
fn create_framebuffers(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    scene_views: &[vk::ImageView],
    width: u32,
    height: u32,
) -> Result<Vec<OwnedFramebuffer>, String> {
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
            .map_err(|e| format!("transparent framebuffer: {e}"))?;
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
    // drive it (the mesh pipelines built). Independent of `rt_accel`, so it
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
    // opaque passes do, so a mesh rerouted here rasterises the same triangles it
    // would have rasterised opaque.
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
        let prefilter_mip_count = self.prefilter_mip_count as f32;

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
            let centre = [
                0.5 * (obj.bb_min[0] + obj.bb_max[0]),
                0.5 * (obj.bb_min[1] + obj.bb_max[1]),
                0.5 * (obj.bb_min[2] + obj.bb_max[2]),
            ];
            let d = crate::gfx::lod::camera_distance(obj, cam);
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
                centre,
            });
        }
        draws
    }

    // Assemble the per-frame transparent view from the frame's jittered VP (the
    // matrix the main pass rasterised the depth buffer with, so a transparent
    // record's clip-space depth matches the stored main-depth) + camera position.
    // Mirrors `directx::graph_exec::build_transparent_view`.
    pub(in crate::vulkan) fn build_transparent_view(
        &self,
        vp: [[f32; 4]; 4],
        cam_pos: [f32; 3],
        time: f32,
    ) -> TransparentView {
        TransparentView {
            vp,
            inv_vp: mat4_inverse(vp),
            camera_pos: [cam_pos[0], cam_pos[1], cam_pos[2], 0.0],
            viewport: [
                self.render_extent.width as f32,
                self.render_extent.height as f32,
            ],
            time,
            prefilter_mip_count: self.prefilter_mip_count as f32,
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
    ) -> Result<(), String> {
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
        // per-pixel trace is the feature, and with RT off those meshes rasterise
        // opaque in the main pass instead.
        let mesh_draws = if rt_live {
            self.collect_mesh_draws(transparent, frame_idx, cam)
        } else {
            Vec::new()
        };
        let mesh_centres: Vec<[f32; 3]> = mesh_draws.iter().map(|d| d.centre).collect();
        let order = transparent.draw_order(&mesh_centres, cam);
        if order.is_empty() {
            return Ok(());
        }

        let device = &self.device;
        let extent = self.render_extent;
        let scene_image = *transparent
            .scene_images
            .get(frame_idx)
            .ok_or("transparent: scene image index OOB")?;
        let snapshot = transparent.snapshot.image;

        // Upload this frame's view UBO.
        transparent
            .view_ubos
            .get(frame_idx)
            .ok_or("transparent: view_ubos index OOB")?
            .write_val(0, view);

        // On the RT path, upload this frame's RtParams (sun + ray tunables) into the
        // shared RtParams ring, mirroring `encode_rt_reflections`'s build. The
        // settings come from the RT-reflection pass (always present when `rt_live`).
        if rt_live {
            let rtres = self
                .rt_reflections
                .as_ref()
                .ok_or("transparent rt_live but rt_reflections missing")?;
            let rt = transparent
                .rt
                .as_ref()
                .ok_or("transparent rt_live but rt pipelines missing")?;
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
                prefilter_mip_count: self.prefilter_mip_count as f32,
            });
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
        // colour write) and the prior frame's snapshot read ahead of the
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
        // image to SHADER_READ_ONLY, so the render pass's colour LOAD matches
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

        // 3) The render pass: LOAD the scene colour, draw each visible record
        // back-to-front, STORE. The negative-height viewport matches the main
        // pass so the manual depth test + refraction taps line up at pixel
        // coordinates.
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(transparent.render_pass.handle())
            .framebuffer(transparent.framebuffers[frame_idx].handle())
            .render_area(vk::Rect2D::default().extent(extent));
        let vp = vk::Viewport {
            x: 0.0,
            y: extent.height as f32,
            width: extent.width as f32,
            height: -(extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(extent);

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
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                std::slice::from_ref(&transparent.view_sets[frame_idx]),
                &[],
            );
            // The per-frame global set (set 2): the fragment shaders reflect its
            // probe set / cube array (bindings 7 / 8) + sky prefilter cube
            // (binding 5). Bound once per frame; stable across the draw loop.
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                2,
                std::slice::from_ref(&self.descriptors.global_sets[frame_idx]),
                &[],
            );
            if rt_live {
                let r = transparent
                    .rt
                    .as_ref()
                    .expect("rt_live implies the RT pipelines");
                // set 3: this frame's RT geometry (TLAS + geom table + the static +
                // skinned vertex/index buffers). Bound once; stable across the loop.
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
                    3,
                    std::slice::from_ref(&r.sets[frame_idx]),
                    &[],
                );
                if textured {
                    // set 4: the bindless albedo/normal pool for textured hit shading
                    // (the same set the main bindless pass binds).
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        layout,
                        4,
                        std::slice::from_ref(&self.cull.bindless_sets[frame_idx]),
                        &[],
                    );
                }
            }
            let mut bound: Option<Producer> = None;
            for &(kind, i) in &order {
                if kind == Producer::GlassMesh {
                    let mesh = transparent
                        .glass_mesh
                        .as_ref()
                        .expect("the draw order only names live producers");
                    if bound != Some(kind) {
                        device.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            mesh.pipeline(textured).handle(),
                        );
                        bound = Some(kind);
                    }
                    // A mesh draws its DrawObject slice of the shared scene
                    // buffers, so the bound buffers are the scene ones rather than
                    // a record own pair and the slice rides the draw arguments.
                    let d = &mesh_draws[i];
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        layout,
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
                    device.cmd_draw_indexed(
                        cmd,
                        d.index_count,
                        1,
                        d.index_offset,
                        d.base_vertex,
                        0,
                    );
                    self.inc_draw_calls(1);
                    continue;
                }
                let producer = transparent.producer(kind);
                if bound != Some(kind) {
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        producer.pipeline(rt_live, textured).handle(),
                    );
                    bound = Some(kind);
                }
                let r = &producer.records[i];
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
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
                self.inc_draw_calls(1);
            }
            device.cmd_end_render_pass(cmd);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_distance_is_euclidean_and_monotone() {
        let cam = [0.0, 0.0, 0.0];
        let near = sort_distance([0.0, 0.0, 1.0], cam);
        let far = sort_distance([0.0, 0.0, 5.0], cam);
        assert!((near - 1.0).abs() < 1e-5);
        assert!((far - 5.0).abs() < 1e-5);
        assert!(far > near);
    }

    #[test]
    fn ordered_visible_excludes_hidden_and_sorts_back_to_front() {
        // Pane 1 is hidden; 0 (dist 5) and 2 (dist 3) are visible. Farthest
        // first => [0, 2]; the hidden pane never appears.
        let glass = [
            ([0.0, 0.0, 5.0], true),
            ([0.0, 0.0, 9.0], false),
            ([0.0, 0.0, 3.0], true),
        ];
        let order = ordered_visible(&glass, &[], &[], [0.0, 0.0, 0.0]);
        assert_eq!(order, vec![(Producer::Glass, 0), (Producer::Glass, 2)]);
    }

    #[test]
    fn ordered_visible_interleaves_the_two_producers() {
        // A pane standing in a pool has to composite in distance order, not in
        // producer order: the far pane draws first, then the water, then the near
        // pane.
        let glass = [([0.0, 0.0, 9.0], true), ([0.0, 0.0, 1.0], true)];
        let water = [([0.0, 0.0, 5.0], true), ([0.0, 0.0, 7.0], false)];
        let order = ordered_visible(&glass, &water, &[], [0.0, 0.0, 0.0]);
        assert_eq!(
            order,
            vec![
                (Producer::Glass, 0),
                (Producer::Water, 0),
                (Producer::Glass, 1),
            ]
        );
    }

    #[test]
    fn align_up_rounds_to_the_next_multiple() {
        // The mesh params ring spaces one block per mesh by the device's
        // `minUniformBufferOffsetAlignment`, so a 96-byte block has to round up to
        // whatever the device asks for. An already-aligned size must not grow.
        assert_eq!(align_up(96, 256), 256);
        assert_eq!(align_up(96, 64), 128);
        assert_eq!(align_up(128, 64), 128);
        assert_eq!(align_up(0, 256), 0);
        // A device reporting no alignment requirement leaves the size alone
        // rather than dividing by zero.
        assert_eq!(align_up(96, 0), 96);
        assert_eq!(align_up(96, 1), 96);
    }

    #[test]
    fn ordered_visible_interleaves_mesh_draws_with_the_static_producers() {
        // A see-through mesh sorts against panes and water by the same camera
        // distance, so it is not simply appended after them. Every mesh entry the
        // encoder passes is already visible, which is why the slice carries
        // centres alone.
        let glass = [([0.0, 0.0, 9.0], true)];
        let water = [([0.0, 0.0, 3.0], true)];
        let meshes = [[0.0, 0.0, 6.0], [0.0, 0.0, 1.0]];
        let order = ordered_visible(&glass, &water, &meshes, [0.0, 0.0, 0.0]);
        assert_eq!(
            order,
            vec![
                (Producer::Glass, 0),
                (Producer::GlassMesh, 0),
                (Producer::Water, 0),
                (Producer::GlassMesh, 1),
            ]
        );
    }

    #[test]
    fn ordered_visible_orders_meshes_alone_back_to_front() {
        // A world whose only transparent content is see-through meshes: the pass
        // still runs, and they still sort farthest first.
        let meshes = [[0.0, 0.0, 2.0], [0.0, 0.0, 8.0]];
        let order = ordered_visible(&[], &[], &meshes, [0.0, 0.0, 0.0]);
        assert_eq!(
            order,
            vec![(Producer::GlassMesh, 1), (Producer::GlassMesh, 0)]
        );
    }

    #[test]
    fn ordered_visible_is_empty_with_no_visible_records() {
        let glass = [([0.0, 0.0, 5.0], false)];
        let water = [([0.0, 0.0, 3.0], false)];
        assert!(ordered_visible(&glass, &water, &[], [0.0, 0.0, 0.0]).is_empty());
    }
}
