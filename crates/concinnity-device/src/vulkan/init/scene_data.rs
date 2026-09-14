//! Scene data: the area-light table and LTC lookups, the world's textures and
//! the engine samplers, the geometry buffers, the per-frame uniform rings and
//! clustered-light binning, the IBL cubes, and the color-grading LUT.

use ash::vk;
use concinnity_core::bake;
use concinnity_core::gfx::render_types::{
    self, AreaLightData, GpuLight, LightUniforms, ShadowUniforms,
};
use concinnity_core::render::backend_init::{MediaPayloads, SceneData};
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::frame_dirty::FrameDirty;
use concinnity_core::render::lights;
use concinnity_core::render::ltc;

use super::{InitGpu, PassStates};
use crate::suballoc::range_alloc::RangeAllocator;
use crate::vulkan::context::{VkAreaLight, VkGeometry, VkSceneAssets, VkShadow, VkUniforms};
use crate::vulkan::draw::{upload_light_uniforms, upload_shadow_uniforms, upload_static_records};
use crate::vulkan::light_cull::VkLightCull;
use crate::vulkan::owned::OwnedSampler;
use crate::vulkan::resources::upload_geometry_buffer;
use crate::vulkan::texture::{self, *};

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

// Upload the world's textures and the reserved fallbacks, the text atlases, and
// the engine samplers. The scene assets start here; their IBL cubes, LUT and
// SSAO fallback are filled in by later stages.
pub(super) fn build_textures_and_samplers(
    gpu: &InitGpu<'_>,
    media: &MediaPayloads<'_>,
    anisotropy: u32,
    passes: &mut PassStates,
) -> RenderResult<VkSceneAssets> {
    let hw = gpu.hw;
    let device = &hw.device;
    let textures: Vec<GpuImage> = if media.textures.is_empty() {
        vec![texture::create_fallback_white(&gpu.upload())?]
    } else {
        media
            .textures
            .iter()
            .enumerate()
            .map(|(i, image)| {
                texture::upload_texture_image(&gpu.upload(), image)
                    .map_err(|e| e.context(format_args!("texture[{i}]")))
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    // Reserved fallbacks, in the order `FALLBACK_TEXTURE_COUNT` documents:
    // the flat-normal image a draw with no normal map samples, then the
    // white image a draw with no albedo samples. Real normal maps and
    // albedos are textures in `textures` (the shared pool) at their own
    // handle; only these two live in `fallback_textures`, past the last
    // real texture.
    let upload_ctx = gpu.upload();
    let fallback_textures = vec![
        texture::create_fallback_flat_normal(&upload_ctx)?,
        texture::create_fallback_white(&upload_ctx)?,
    ];

    passes.text.atlas_textures = media
        .text_atlases
        .iter()
        .enumerate()
        .map(|(i, (w, h, px))| {
            upload_texture(&gpu.upload(), *w, *h, px)
                .map_err(|e| e.context(format_args!("text_atlas[{i}]")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Anisotropic degree for the scene sampler: enabled only when the device
    // supports `samplerAnisotropy` (the matching feature is turned on in
    // `device.rs`). Clamp the requested degree (GraphicsConfig.anisotropy) to
    // the GPU's 1..16 range and then to the device limit.
    let scene_aniso = {
        // SAFETY: a property query on a live handle; it only reads.
        let feats = unsafe { hw.instance.get_physical_device_features(hw.physical_device) };
        if feats.sampler_anisotropy != 0 {
            // SAFETY: a property query on a live handle; it only reads.
            let limit = unsafe {
                hw.instance
                    .get_physical_device_properties(hw.physical_device)
            }
            .limits
            .max_sampler_anisotropy;
            (anisotropy.clamp(1, 16) as f32).min(limit)
        } else {
            1.0
        }
    };
    let linear_sampler = create_sampler_linear_repeat(device, scene_aniso)?;
    passes.shadow.sampler = create_sampler_shadow(device)?;
    passes.text.sampler = create_sampler_linear_clamp(device)?;
    // Linear-clamp sampler the composite pass reads the HDR resolve with;
    // clamp keeps the FXAA neighbor taps from wrapping at screen edges.
    passes.composite.sampler = create_sampler_linear_clamp(device)?;
    Ok(VkSceneAssets {
        textures,
        fallback_textures,
        linear_sampler,
        cube_sampler: OwnedSampler::null(),
        env_map: EnvironmentMapTextures {
            irradiance: GpuImage::null(),
            prefilter: GpuImage::null(),
            prefilter_mip_count: 0,
        },
        prefilter_mip_count: 0,
        color_lut: GpuImage::null(),
        ssao_white: GpuImage::null(),
    })
}

pub(super) struct SceneInputs<'a> {
    pub(super) world: &'a SceneData<'a>,
    pub(super) media: &'a MediaPayloads<'a>,
    pub(super) local_lights: &'a [GpuLight],
    pub(super) light_uniforms: LightUniforms,
}

// Upload the geometry, the per-frame uniform rings and the local lights, build
// the clustered-light binning, and fill in the shadow uniform ring and the
// scene's IBL cubes and color-grading LUT.
pub(super) fn build_scene_resources(
    gpu: &InitGpu<'_>,
    inputs: SceneInputs<'_>,
    scene: &mut VkSceneAssets,
    shadow: &mut VkShadow,
) -> RenderResult<(VkGeometry, VkUniforms, VkLightCull)> {
    let InitGpu {
        hw,
        command_pool,
        frames,
        hot_reload,
    } = *gpu;
    let SceneInputs {
        world,
        media,
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
    let shadow_ubo_size = std::mem::size_of::<ShadowUniforms>() as u64;
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
    // Per-frame-in-flight `ShadowUniforms` UBO ring, persistently mapped.
    // One slot per frame so writing this frame's cascade VPs cannot land in
    // memory an in-flight frame is still sampling: under `Hybrid` a far
    // cascade's VP is frozen for several frames and then jumps a whole
    // texel-snap quantum, so an aliased read samples that cascade with the
    // jumped VP against depth rasterized with the old one.
    shadow.ubos.reserve(frames);
    for _ in 0..frames {
        shadow.ubos.push(alloc.create_buffer(
            shadow_ubo_size,
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

    // Per-frame CSM updates use the first directional light's direction;
    // we cache it here at init so subsequent frames don't have to look it
    // up. Matches the Metal/DirectX pattern.
    shadow.light_dir = lights::sun_direction(&light_uniforms);
    // Per-frame cascade computation lives in `gfx::csm::compute_shadow_uniforms`
    // and runs from `draw.rs` each frame; the shadow state starts from
    // `empty_shadow_uniforms()` so the descriptor write at startup has a valid
    // (fully-lit) buffer.
    for ubo in &shadow.ubos {
        upload_shadow_uniforms(ubo, &shadow.uniforms);
    }
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

    // IBL resources, always created so descriptor bindings 4/5 are valid.
    scene.cube_sampler = create_sampler_cube_linear(device)?;
    scene.env_map = if let Some(bytes) = media.env_map_bytes {
        let view = bake::environment_map::deserialize(bytes)
            .map_err(|e| format!("EnvironmentMap payload malformed: {}", e))?;
        upload_environment_map(
            &gpu.upload(),
            view.irradiance_face,
            view.irradiance_bytes,
            view.prefilter_face,
            &view.prefilter_mip_bytes,
        )?
    } else {
        EnvironmentMapTextures {
            irradiance: texture::create_fallback_cubemap(&gpu.upload(), [0.05, 0.05, 0.05, 1.0])?,
            prefilter: texture::create_fallback_cubemap(&gpu.upload(), [0.05, 0.05, 0.05, 1.0])?,
            prefilter_mip_count: 0,
        }
    };
    scene.prefilter_mip_count = scene.env_map.prefilter_mip_count;

    // Color-grading LUT: upload the declared `ColorLut` payload, or build a
    // 2x2x2 identity LUT so the composite pass always binds a valid 3D
    // texture. With the identity LUT the grade is a no-op at any strength.
    scene.color_lut = if let Some(bytes) = media.color_lut_bytes {
        let (size, data) = bake::color_lut::deserialize(bytes)
            .map_err(|e| format!("ColorLut payload malformed: {e}"))?;
        upload_color_lut(&gpu.upload(), size, data)?
    } else {
        create_fallback_color_lut(&gpu.upload())?
    };
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
