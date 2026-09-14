//! Scene data: the area-light table and LTC lookups, the geometry buffers, and
//! the per-frame uniform rings and clustered-light binning.

use ash::vk;
use concinnity_core::gfx::render_types::{self, AreaLightData, GpuLight, LightUniforms};
use concinnity_core::render::backend_init::SceneData;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::frame_dirty::FrameDirty;
use concinnity_core::render::ltc;

use super::InitGpu;
use crate::suballoc::range_alloc::RangeAllocator;
use crate::vulkan::context::{VkAreaLight, VkGeometry, VkUniforms};
use crate::vulkan::draw::{upload_light_uniforms, upload_static_records};
use crate::vulkan::light_cull::VkLightCull;
use crate::vulkan::resources::upload_geometry_buffer;
use crate::vulkan::texture::*;

// Rectangular area lights: the edge vectors that do not fit in
// `GpuLight`, indexed by its `data_index`, plus the two LTC tables the
// shading path samples. A world with no area light still gets a
// one-element buffer, since the shader never reads it (`data_index`
// stays -1) but the descriptor must be valid. The tables are
// scene-independent, so they are uploaded either way.
pub(super) fn build_area_lights(
    gpu: &InitGpu<'_>,
    area_lights: &[AreaLightData],
) -> RenderResult<VkAreaLight> {
    let hw = gpu.hw;
    let area_light_data = if area_lights.is_empty() {
        vec![render_types::AreaLightData::ZERO]
    } else {
        area_lights.to_vec()
    };
    let area_light_size = std::mem::size_of_val(area_light_data.as_slice()) as u64;
    let buffer = hw.alloc.create_buffer(
        area_light_size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    upload_static_records(&buffer, &area_light_data);
    let ltc_size = ltc::LTC_LUT_SIZE as u32;
    let ltc_upload = gpu.upload();
    let ltc_matrix = upload_float_lut(&ltc_upload, ltc_size, 4, ltc::matrix_texels())?;
    let ltc_magnitude = upload_float_lut(&ltc_upload, ltc_size, 2, ltc::magnitude_texels())?;
    // Linear clamp-to-edge: the LUT is indexed by roughness / view angle, so
    // an edge sample must not wrap.
    let sampler = create_sampler_cube_linear(&hw.device)?;
    Ok(VkAreaLight {
        buffer,
        ltc_matrix,
        ltc_magnitude,
        sampler,
    })
}

pub(super) struct SceneInputs<'a> {
    pub(super) world: &'a SceneData<'a>,
    pub(super) local_lights: &'a [GpuLight],
    pub(super) light_uniforms: LightUniforms,
}

// Upload the geometry, the per-frame uniform rings and the local lights, and
// build the clustered-light binning.
pub(super) fn build_scene_resources(
    gpu: &InitGpu<'_>,
    inputs: SceneInputs<'_>,
) -> RenderResult<(VkGeometry, VkUniforms, VkLightCull)> {
    let InitGpu {
        hw,
        command_pool,
        frames,
        hot_reload,
    } = *gpu;
    let SceneInputs {
        world,
        local_lights,
        light_uniforms,
    } = inputs;
    let (device, alloc, graphics_queue) = (&hw.device, &hw.alloc, hw.graphics_queue);
    let (vertices, indices) = (world.vertices, world.indices);
    // Geometry buffers. See `shared_geometry_usage` for why an RT-capable
    // device carries the acceleration-structure / storage usage here even
    // when RT is off at launch. Inert when RT is never built.
    let rt_geo_usage = crate::vulkan::resources::shared_geometry_usage(hw.rt_capable);
    let vertex_buffer = upload_geometry_buffer(
        alloc,
        device,
        command_pool,
        graphics_queue,
        vertices,
        vk::BufferUsageFlags::VERTEX_BUFFER | rt_geo_usage,
    )?;
    let index_buffer = upload_geometry_buffer(
        alloc,
        device,
        command_pool,
        graphics_queue,
        indices,
        vk::BufferUsageFlags::INDEX_BUFFER | rt_geo_usage,
    )?;
    // Empty geometry still allocates a 4-byte buffer (see
    // `upload_geometry_buffer_raw`); track the real allocation size so
    // `setup_chunk_streaming` copies the right prefix when it grows them.
    let vertex_buffer_bytes = (std::mem::size_of_val(vertices) as u64).max(4);
    let index_buffer_bytes = (std::mem::size_of_val(indices) as u64).max(4);

    let view_ubo_size = std::mem::size_of::<crate::vulkan::draw::ViewUniforms>() as u64;
    let light_ubo_size = std::mem::size_of::<LightUniforms>() as u64;
    // Per-scene local-light SSBO (global set 0 binding 9): created once from
    // `local_lights` and never updated per-frame. A zero-length buffer is
    // invalid, so an empty scene gets a 1-element placeholder;
    // `num_local_lights == 0` keeps the shader from reading it. Mirrors the
    // Metal `local_light_buffer`.
    let local_light_size = (local_lights.len().max(1) * std::mem::size_of::<GpuLight>()) as u64;

    let mut view_ubo_buffers = Vec::with_capacity(frames);
    for _ in 0..frames {
        view_ubo_buffers.push(alloc.create_buffer(
            view_ubo_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?);
    }

    // Per-frame `ProbeSet` UBO ring (global set 0 binding 7): the
    // reflection-probe count + per-probe parallax boxes. Persistently mapped;
    // `record_frame` writes `self.probe.set` here each frame.
    let probe_set_ubo_size =
        std::mem::size_of::<concinnity_core::render::uniforms::ProbeSet>() as u64;
    let mut probe_set_ubo_buffers = Vec::with_capacity(frames);
    for _ in 0..frames {
        probe_set_ubo_buffers.push(alloc.create_buffer(
            probe_set_ubo_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?);
    }

    // Per-frame-in-flight `LightUniforms` UBO ring, persistently mapped. One
    // slot per frame so a live directional-light or ambient change can be
    // written on the CPU without draining the queue: a turning sky rewrites
    // the set every frame, and a single shared buffer would stall on every
    // one of them.
    let mut light_ubo_buffers = Vec::with_capacity(frames);
    for _ in 0..frames {
        light_ubo_buffers.push(alloc.create_buffer(
            light_ubo_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?);
    }
    // Static per-scene local-light SSBO; uploaded once below, never per-frame.
    let local_light_buffer = alloc.create_buffer(
        local_light_size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    for ubo in &light_ubo_buffers {
        upload_light_uniforms(ubo, &light_uniforms);
    }
    // Empty scene keeps the 1-element placeholder (nothing copied in).
    upload_static_records(&local_light_buffer, local_lights);

    // Clustered light binning. The per-cluster list + `ClusterParams` buffers
    // are always allocated (the forward shaders reference bindings 10 + 11
    // unconditionally, guarded by `use_clusters`); the compute pipeline is
    // built only when the world has local lights to bin, which is also what
    // gates the `LightCull` graph node.
    let light_cull = crate::vulkan::light_cull::build_light_cull(
        alloc,
        device,
        frames,
        local_light_buffer.buffer(),
        local_light_size,
        !local_lights.is_empty(),
        hot_reload,
    )?;
    Ok((
        VkGeometry {
            vertex_buffer,
            index_buffer,
            mesh_vtx_alloc: RangeAllocator::new(),
            mesh_idx_alloc: RangeAllocator::new(),
            vertex_buffer_bytes,
            index_buffer_bytes,
        },
        VkUniforms {
            view_ubo_buffers,
            probe_set_ubo_buffers,
            light_ubo_buffers,
            light_dirty: FrameDirty::new(frames),
            local_light_buffer,
            local_light_size,
            light_uniforms,
        },
        light_cull,
    ))
}
