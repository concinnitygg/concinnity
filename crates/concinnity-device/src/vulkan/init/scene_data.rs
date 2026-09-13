//! Scene data: the area-light table and LTC lookups, the world's textures and
//! the engine samplers, the geometry buffers, the per-frame uniform rings and
//! clustered-light binning, the IBL cubes, and the color-grading LUT.

use ash::vk;
use concinnity_core::bake;
use concinnity_core::bake::texture::TextureImage;
use concinnity_core::gfx::mesh_payload::Vertex;
use concinnity_core::gfx::render_types::{
    self, AreaLightData, GpuLight, LightUniforms, ShadowUniforms,
};
use concinnity_core::render::csm;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::lights;
use concinnity_core::render::ltc;

use super::InitGpu;
use crate::vulkan::allocator::PooledBuffer;
use crate::vulkan::draw::{upload_light_uniforms, upload_shadow_uniforms, upload_static_records};
use crate::vulkan::light_cull::VkLightCull;
use crate::vulkan::owned::OwnedSampler;
use crate::vulkan::resources::upload_geometry_buffer;
use crate::vulkan::texture::{self, *};

pub(super) struct AreaLights {
    pub(super) area_light_buffer: PooledBuffer,
    pub(super) ltc_matrix_image: GpuImage,
    pub(super) ltc_magnitude_image: GpuImage,
    pub(super) ltc_sampler: OwnedSampler,
}

// Rectangular area lights: the edge vectors that do not fit in
// `GpuLight`, indexed by its `data_index`, plus the two LTC tables the
// shading path samples. A world with no area light still gets a
// one-element buffer, since the shader never reads it (`data_index`
// stays -1) but the descriptor must be valid. The tables are
// scene-independent, so they are uploaded either way.
pub(super) fn build_area_lights(
    gpu: &InitGpu<'_>,
    area_lights: &[AreaLightData],
) -> RenderResult<AreaLights> {
    let InitGpu {
        device,
        alloc,
        command_pool,
        queue: graphics_queue,
        ..
    } = *gpu;
    let area_light_data = if area_lights.is_empty() {
        vec![render_types::AreaLightData::ZERO]
    } else {
        area_lights.to_vec()
    };
    let area_light_size = std::mem::size_of_val(area_light_data.as_slice()) as u64;
    let area_light_buffer = alloc.create_buffer(
        area_light_size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    upload_static_records(&area_light_buffer, &area_light_data);
    let ltc_size = ltc::LTC_LUT_SIZE as u32;
    let ltc_upload = GpuUploadContext {
        alloc,
        device,
        command_pool,
        queue: graphics_queue,
    };
    let ltc_matrix_image = upload_float_lut(&ltc_upload, ltc_size, 4, ltc::matrix_texels())?;
    let ltc_magnitude_image = upload_float_lut(&ltc_upload, ltc_size, 2, ltc::magnitude_texels())?;
    // Linear clamp-to-edge: the LUT is indexed by roughness / view angle, so
    // an edge sample must not wrap.
    let ltc_sampler = create_sampler_cube_linear(device)?;
    Ok(AreaLights {
        area_light_buffer,
        ltc_matrix_image,
        ltc_magnitude_image,
        ltc_sampler,
    })
}

pub(super) struct SceneTextures {
    pub(super) gpu_textures: Vec<GpuImage>,
    pub(super) gpu_fallbacks: Vec<GpuImage>,
    pub(super) gpu_text_atlases: Vec<GpuImage>,
    pub(super) linear_sampler: OwnedSampler,
    pub(super) shadow_sampler: OwnedSampler,
    pub(super) text_sampler: OwnedSampler,
    pub(super) composite_sampler: OwnedSampler,
}

pub(super) fn build_textures_and_samplers(
    gpu: &InitGpu<'_>,
    textures: &[TextureImage],
    text_atlases: &[(u32, u32, Vec<u8>)],
    anisotropy: u32,
) -> RenderResult<SceneTextures> {
    let InitGpu {
        instance,
        device,
        physical_device,
        alloc,
        command_pool,
        queue: graphics_queue,
        ..
    } = *gpu;
    let gpu_textures: Vec<GpuImage> = if textures.is_empty() {
        vec![texture::create_fallback_white(&GpuUploadContext {
            alloc,
            device,
            command_pool,
            queue: graphics_queue,
        })?]
    } else {
        textures
            .iter()
            .enumerate()
            .map(|(i, image)| {
                texture::upload_texture_image(
                    &GpuUploadContext {
                        alloc,
                        device,
                        command_pool,
                        queue: graphics_queue,
                    },
                    image,
                )
                .map_err(|e| e.context(format_args!("texture[{i}]")))
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    // Reserved fallbacks, in the order `FALLBACK_TEXTURE_COUNT` documents:
    // the flat-normal image a draw with no normal map samples, then the
    // white image a draw with no albedo samples. Real normal maps and
    // albedos are textures in `gpu_textures` (the shared pool) at their own
    // handle; only these two live in `fallback_textures`, past the last
    // real texture.
    let upload_ctx = GpuUploadContext {
        alloc,
        device,
        command_pool,
        queue: graphics_queue,
    };
    let gpu_fallbacks = vec![
        texture::create_fallback_flat_normal(&upload_ctx)?,
        texture::create_fallback_white(&upload_ctx)?,
    ];

    let gpu_text_atlases: Vec<GpuImage> = text_atlases
        .iter()
        .enumerate()
        .map(|(i, (w, h, px))| {
            upload_texture(
                &GpuUploadContext {
                    alloc,
                    device,
                    command_pool,
                    queue: graphics_queue,
                },
                *w,
                *h,
                px,
            )
            .map_err(|e| e.context(format_args!("text_atlas[{i}]")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Anisotropic degree for the scene sampler: enabled only when the device
    // supports `samplerAnisotropy` (the matching feature is turned on in
    // `device.rs`). Clamp the requested degree (GraphicsConfig.anisotropy) to
    // the GPU's 1..16 range and then to the device limit.
    let scene_aniso = {
        // SAFETY: a property query on a live handle; it only reads.
        let feats = unsafe { instance.get_physical_device_features(physical_device) };
        if feats.sampler_anisotropy != 0 {
            // SAFETY: a property query on a live handle; it only reads.
            let limit = unsafe { instance.get_physical_device_properties(physical_device) }
                .limits
                .max_sampler_anisotropy;
            (anisotropy.clamp(1, 16) as f32).min(limit)
        } else {
            1.0
        }
    };
    let linear_sampler = create_sampler_linear_repeat(device, scene_aniso)?;
    let shadow_sampler = create_sampler_shadow(device)?;
    let text_sampler = create_sampler_linear_clamp(device)?;
    // Linear-clamp sampler the composite pass reads the HDR resolve with;
    // clamp keeps the FXAA neighbor taps from wrapping at screen edges.
    let composite_sampler = create_sampler_linear_clamp(device)?;
    Ok(SceneTextures {
        gpu_textures,
        gpu_fallbacks,
        gpu_text_atlases,
        linear_sampler,
        shadow_sampler,
        text_sampler,
        composite_sampler,
    })
}

pub(super) struct SceneInputs<'a> {
    pub(super) vertices: &'a [Vertex],
    pub(super) indices: &'a [u32],
    pub(super) rt_capable: bool,
    pub(super) local_lights: &'a [GpuLight],
    pub(super) light_uniforms: &'a LightUniforms,
    pub(super) env_map_bytes: Option<&'a [u8]>,
    pub(super) color_lut_bytes: Option<&'a [u8]>,
}

pub(super) struct SceneResources {
    pub(super) vertex_buffer: PooledBuffer,
    pub(super) index_buffer: PooledBuffer,
    pub(super) vertex_buffer_bytes: u64,
    pub(super) index_buffer_bytes: u64,
    pub(super) view_ubo_size: u64,
    pub(super) light_ubo_size: u64,
    pub(super) shadow_ubo_size: u64,
    pub(super) local_light_buffer_size: u64,
    pub(super) view_ubo_buffers: Vec<PooledBuffer>,
    pub(super) probe_set_ubo_size: u64,
    pub(super) probe_set_ubo_buffers: Vec<PooledBuffer>,
    pub(super) light_ubo_buffers: Vec<PooledBuffer>,
    pub(super) shadow_ubos: Vec<PooledBuffer>,
    pub(super) local_light_buffer: PooledBuffer,
    pub(super) shadow_light_dir: [f32; 3],
    pub(super) fog_sun_dir: [f32; 3],
    pub(super) fog_sun_color: [f32; 3],
    pub(super) shadow_uniforms: ShadowUniforms,
    pub(super) light_cull: VkLightCull,
    pub(super) cube_sampler: OwnedSampler,
    pub(super) env_map: EnvironmentMapTextures,
    pub(super) color_lut: GpuImage,
}

pub(super) fn build_scene_resources(
    gpu: &InitGpu<'_>,
    inputs: SceneInputs<'_>,
) -> RenderResult<SceneResources> {
    let InitGpu {
        device,
        alloc,
        command_pool,
        queue: graphics_queue,
        frames,
        hot_reload,
        ..
    } = *gpu;
    let SceneInputs {
        vertices,
        indices,
        rt_capable,
        local_lights,
        light_uniforms,
        env_map_bytes,
        color_lut_bytes,
    } = inputs;
    // Geometry buffers. See `shared_geometry_usage` for why an RT-capable
    // device carries the acceleration-structure / storage usage here even
    // when RT is off at launch. Inert when RT is never built.
    let rt_geo_usage = crate::vulkan::resources::shared_geometry_usage(rt_capable);
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
    let local_light_buffer_size =
        (local_lights.len().max(1) * std::mem::size_of::<GpuLight>()) as u64;

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
    let mut shadow_ubos = Vec::with_capacity(frames);
    for _ in 0..frames {
        shadow_ubos.push(alloc.create_buffer(
            shadow_ubo_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?);
    }
    // Static per-scene local-light SSBO; uploaded once below, never per-frame.
    let local_light_buffer = alloc.create_buffer(
        local_light_buffer_size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // Per-frame CSM updates use the first directional light's direction;
    // we cache it here at init so subsequent frames don't have to look it
    // up. Matches the Metal/DirectX pattern.
    let shadow_light_dir = lights::sun_direction(light_uniforms);
    // Sun direction + intensity-weighted color for the volumetric-fog
    // encoder, cached because the light UBO is uploaded rather than pushed
    // each frame. `update_directional_lights` re-derives both.
    let fog_sun_dir = shadow_light_dir;
    let fog_sun_color = lights::sun_color(light_uniforms);
    // Per-frame cascade computation lives in `gfx::csm::compute_shadow_uniforms`
    // and runs from `draw.rs` each frame; init stores `empty_shadow_uniforms()`
    // so the descriptor write at startup has a valid (fully-lit) buffer.
    let shadow_uniforms = csm::empty_shadow_uniforms();
    for ubo in &shadow_ubos {
        upload_shadow_uniforms(ubo, &shadow_uniforms);
    }
    for ubo in &light_ubo_buffers {
        upload_light_uniforms(ubo, light_uniforms);
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
        local_light_buffer_size,
        !local_lights.is_empty(),
        hot_reload,
    )?;

    // IBL resources, always created so descriptor bindings 4/5 are valid.
    let cube_sampler = create_sampler_cube_linear(device)?;
    let env_map = if let Some(bytes) = env_map_bytes {
        let view = bake::environment_map::deserialize(bytes)
            .map_err(|e| format!("EnvironmentMap payload malformed: {}", e))?;
        upload_environment_map(
            &GpuUploadContext {
                alloc,
                device,
                command_pool,
                queue: graphics_queue,
            },
            view.irradiance_face,
            view.irradiance_bytes,
            view.prefilter_face,
            &view.prefilter_mip_bytes,
        )?
    } else {
        EnvironmentMapTextures {
            irradiance: texture::create_fallback_cubemap(
                &GpuUploadContext {
                    alloc,
                    device,
                    command_pool,
                    queue: graphics_queue,
                },
                [0.05, 0.05, 0.05, 1.0],
            )?,
            prefilter: texture::create_fallback_cubemap(
                &GpuUploadContext {
                    alloc,
                    device,
                    command_pool,
                    queue: graphics_queue,
                },
                [0.05, 0.05, 0.05, 1.0],
            )?,
            prefilter_mip_count: 0,
        }
    };

    // Color-grading LUT: upload the declared `ColorLut` payload, or build a
    // 2x2x2 identity LUT so the composite pass always binds a valid 3D
    // texture. With the identity LUT the grade is a no-op at any strength.
    let color_lut = if let Some(bytes) = color_lut_bytes {
        let (size, data) = bake::color_lut::deserialize(bytes)
            .map_err(|e| format!("ColorLut payload malformed: {e}"))?;
        upload_color_lut(
            &GpuUploadContext {
                alloc,
                device,
                command_pool,
                queue: graphics_queue,
            },
            size,
            data,
        )?
    } else {
        create_fallback_color_lut(&GpuUploadContext {
            alloc,
            device,
            command_pool,
            queue: graphics_queue,
        })?
    };
    Ok(SceneResources {
        vertex_buffer,
        index_buffer,
        vertex_buffer_bytes,
        index_buffer_bytes,
        view_ubo_size,
        light_ubo_size,
        shadow_ubo_size,
        local_light_buffer_size,
        view_ubo_buffers,
        probe_set_ubo_size,
        probe_set_ubo_buffers,
        light_ubo_buffers,
        shadow_ubos,
        local_light_buffer,
        shadow_light_dir,
        fog_sun_dir,
        fog_sun_color,
        shadow_uniforms,
        light_cull,
        cube_sampler,
        env_map,
        color_lut,
    })
}
