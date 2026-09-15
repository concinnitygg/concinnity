//! The world's scene assets: the shared geometry buffers, the local and area
//! light tables, the LTC lookup tables, the texture pool with its fallbacks, the
//! samplers the pool and the IBL cubes are read through, the IBL cubes, and the
//! color-grading LUT.

use concinnity_core::bake;
use concinnity_core::gfx::mesh_payload::Vertex;
use concinnity_core::gfx::render_types::{AreaLightData, GpuLight};
use concinnity_core::render::backend_init::{MediaPayloads, SceneData};
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::ltc;
use objc2_metal::{
    MTLDevice as _, MTLResourceOptions, MTLSamplerAddressMode, MTLSamplerDescriptor,
    MTLSamplerMinMagFilter,
};

use super::InitGpu;
use crate::metal::context::{BINDLESS_TEXTURE_COUNT, MtlSceneAssets, bytes_of_slice};
use crate::metal::texture::{
    EnvironmentMapTextures, create_fallback_color_lut, create_fallback_cubemap,
    create_fallback_texture, create_lut_texture, upload_color_lut, upload_environment_map,
    upload_texture, upload_texture_image,
};

pub(super) struct SceneInputs<'a> {
    pub(super) world: &'a SceneData<'a>,
    pub(super) media: &'a MediaPayloads<'a>,
    pub(super) local_lights: &'a [GpuLight],
    pub(super) area_lights: &'a [AreaLightData],
    // Degree from GraphicsConfig.anisotropy (default 8).
    pub(super) anisotropy: u32,
    // Whether the bindless static pass binds the texture pool, whose capacity
    // is capped.
    pub(super) bindless: bool,
}

pub(super) fn build_scene_assets(
    gpu: &InitGpu<'_>,
    inputs: SceneInputs<'_>,
) -> RenderResult<MtlSceneAssets> {
    let device = &*gpu.hw.device;
    let allocator = &gpu.hw.allocator;
    let SceneInputs {
        world,
        media,
        local_lights,
        area_lights,
        anisotropy,
        bindless,
    } = inputs;
    let (vertices, indices) = (world.vertices, world.indices);

    // upload vertex and index data into GPU-accessible buffers. A
    // geometry-less world (text-only) has empty slices; Metal rejects a
    // zero-length buffer, so a minimal placeholder is allocated instead --
    // the draw list is empty so the placeholder is never read.
    let vertex_buffer = if vertices.is_empty() {
        allocator.alloc_buffer(
            std::mem::size_of::<Vertex>(),
            MTLResourceOptions::StorageModeShared,
        )
    } else {
        allocator.alloc_buffer_with_bytes(
            bytes_of_slice(vertices),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .map_err(|e| e.context("vertex buffer"))?;

    let index_buffer = if indices.is_empty() {
        allocator.alloc_buffer(
            std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
    } else {
        allocator.alloc_buffer_with_bytes(
            bytes_of_slice(indices),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .map_err(|e| e.context("index buffer"))?;

    // Per-scene local-light storage buffer bound to the forward pass at
    // fragment buffer(8). Metal rejects a zero-length buffer, so a scene with
    // no local lights gets a one-element placeholder; num_local_lights == 0
    // keeps the shader from reading it.
    let local_light_buffer = if local_lights.is_empty() {
        allocator.alloc_buffer(
            std::mem::size_of::<GpuLight>(),
            MTLResourceOptions::StorageModeShared,
        )
    } else {
        allocator.alloc_buffer_with_bytes(
            bytes_of_slice(local_lights),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .map_err(|e| e.context("local-light buffer"))?;

    // Per-scene rect area-light table, indexed by `GpuLight.data_index`.
    // Static like the lights themselves, so it uploads once. Metal rejects a
    // zero-length buffer, so a world with no area light gets a one-element
    // placeholder the shader never reads (every data_index stays -1).
    let area_light_buffer = if area_lights.is_empty() {
        allocator.alloc_buffer(
            std::mem::size_of::<AreaLightData>(),
            MTLResourceOptions::StorageModeShared,
        )
    } else {
        allocator.alloc_buffer_with_bytes(
            bytes_of_slice(area_lights),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .map_err(|e| e.context("area-light buffer"))?;

    // Area-light LTC tables. Scene-independent (they depend only on the
    // build-time fit), so they are created unconditionally and the shader
    // simply never samples them when no area light is declared.
    let ltc_matrix_texture =
        create_lut_texture(allocator, ltc::matrix_texels(), ltc::LTC_LUT_SIZE as u32, 4)?;
    let ltc_magnitude_texture = create_lut_texture(
        allocator,
        ltc::magnitude_texels(),
        ltc::LTC_LUT_SIZE as u32,
        2,
    )?;

    // upload textures; fall back to a 1x1 opaque white texture when none provided
    let textures = if media.textures.is_empty() {
        vec![create_fallback_texture(allocator)?]
    } else {
        media
            .textures
            .iter()
            .enumerate()
            .map(|(i, image)| {
                upload_texture_image(allocator, image)
                    .map_err(|e| e.context(format_args!("texture[{i}]")))
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    // Reserved fallbacks, in the order `FALLBACK_TEXTURE_COUNT` documents:
    // the 1x1 tangent-space (0,0,1) texture a draw with no normal map
    // samples, then the 1x1 white texture a draw with no albedo samples.
    // Real normal maps and albedos are textures in `textures` (the shared
    // pool) at their own handle; only these two live in `fallback_textures`,
    // past the last real texture.
    let flat_normal = upload_texture(allocator, 1, 1, &[128u8, 128, 255, 255])
        .map_err(|e| e.context("flat normal fallback"))?;
    let white = upload_texture(allocator, 1, 1, &[255u8, 255, 255, 255])
        .map_err(|e| e.context("white fallback"))?;
    let fallback_textures = vec![flat_normal, white];

    // The bindless static pass binds every texture plus the flat-normal
    // fallback into one capped pool. A world that exceeds the cap still
    // renders, but objects whose pool index would overflow get clamped to
    // the last slot.
    if bindless && textures.len() + fallback_textures.len() > BINDLESS_TEXTURE_COUNT {
        tracing::warn!(
            "Metal: texture pool ({} textures + 2 fallbacks) exceeds bindless \
             capacity {}; some objects will sample a clamped texture",
            textures.len(),
            BINDLESS_TEXTURE_COUNT,
        );
    }

    // linear filter, repeat wrap -- matches the room shader expectations.
    // Mipmap linear + anisotropy let minified scene textures trilinear-select
    // down the mip chain now that uploads carry one, instead of aliasing from
    // mip 0. The degree is clamped to Metal's guaranteed 1..16 range.
    let sampler = {
        let desc = MTLSamplerDescriptor::new();
        desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
        desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
        desc.setMipFilter(objc2_metal::MTLSamplerMipFilter::Linear);
        desc.setSAddressMode(MTLSamplerAddressMode::Repeat);
        desc.setTAddressMode(MTLSamplerAddressMode::Repeat);
        desc.setMaxAnisotropy(anisotropy.clamp(1, 16) as usize);
        // Written into the engine sampler block (an argument buffer) for
        // the single-source main program, which requires this flag.
        desc.setSupportArgumentBuffers(true);
        device
            .newSamplerStateWithDescriptor(&desc)
            .ok_or("failed to create sampler state")?
    };

    // Cube sampler: linear filter + clamp-to-edge + mipmap linear for prefilter
    // roughness lookups. Bound at sampler(2) and shared by both IBL cubes.
    let cube_sampler = {
        let desc = MTLSamplerDescriptor::new();
        desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
        desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
        desc.setMipFilter(objc2_metal::MTLSamplerMipFilter::Linear);
        desc.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
        desc.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
        desc.setRAddressMode(MTLSamplerAddressMode::ClampToEdge);
        // Rides the engine sampler block alongside the pool sampler.
        desc.setSupportArgumentBuffers(true);
        device
            .newSamplerStateWithDescriptor(&desc)
            .ok_or("failed to create cube sampler state")?
    };

    // IBL: either upload the supplied EnvironmentMap payload or build a
    // 1x1 gray fallback cube pair so texture(3) / texture(4) are always
    // bound. The fragment shader uses `prefilter_mip_count == 0` to
    // detect the fallback and skip IBL math.
    let env_map = if let Some(bytes) = media.env_map_bytes {
        let view = bake::environment_map::deserialize(bytes)
            .map_err(|e| RenderError::Other(format!("EnvironmentMap payload malformed: {e}")))?;
        upload_environment_map(
            allocator,
            view.irradiance_face,
            view.irradiance_bytes,
            view.prefilter_face,
            &view.prefilter_mip_bytes,
        )?
    } else {
        EnvironmentMapTextures {
            irradiance: create_fallback_cubemap(allocator, [0.05, 0.05, 0.05, 1.0])?,
            prefilter: create_fallback_cubemap(allocator, [0.05, 0.05, 0.05, 1.0])?,
            prefilter_mip_count: 0,
        }
    };

    // Color-grading LUT: upload the declared ColorLut payload, or build a
    // 2x2x2 identity LUT so the composite pass always binds a valid 3D
    // texture. With the identity LUT the grade is a no-op at any strength.
    let color_lut = if let Some(bytes) = media.color_lut_bytes {
        let (size, data) = bake::color_lut::deserialize(bytes)
            .map_err(|e| RenderError::Other(format!("ColorLut payload malformed: {e}")))?;
        upload_color_lut(allocator, size, data)?
    } else {
        create_fallback_color_lut(allocator)?
    };

    Ok(MtlSceneAssets {
        vertex_buffer,
        index_buffer,
        textures,
        fallback_textures,
        local_light_buffer,
        area_light_buffer,
        ltc_matrix_texture,
        ltc_magnitude_texture,
        env_map,
        color_lut,
        sampler,
        cube_sampler,
    })
}
