//! Scene assets: the world's textures and their reserved fallbacks, the scene
//! and cube samplers, the IBL cubes, the color-grading LUT, and the SSAO
//! fallback.

use concinnity_core::bake;
use concinnity_core::render::backend_init::MediaPayloads;
use concinnity_core::render::error::RenderResult;

use super::InitGpu;
use crate::vulkan::context::VkSceneAssets;
use crate::vulkan::texture::{self, *};

pub(super) fn build_scene_assets(
    gpu: &InitGpu<'_>,
    media: &MediaPayloads<'_>,
    anisotropy: u32,
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

    // IBL resources, always created so descriptor bindings 4/5 are valid.
    let cube_sampler = create_sampler_cube_linear(device)?;
    let env_map = if let Some(bytes) = media.env_map_bytes {
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

    // Color-grading LUT: upload the declared `ColorLut` payload, or build a
    // 2x2x2 identity LUT so the composite pass always binds a valid 3D
    // texture. With the identity LUT the grade is a no-op at any strength.
    let color_lut = if let Some(bytes) = media.color_lut_bytes {
        let (size, data) = bake::color_lut::deserialize(bytes)
            .map_err(|e| format!("ColorLut payload malformed: {e}"))?;
        upload_color_lut(&gpu.upload(), size, data)?
    } else {
        create_fallback_color_lut(&gpu.upload())?
    };

    // The 1x1 white image bound at set 0 binding 6 when SSAO is off, so the main
    // pass's `ambient *= ao` multiplier collapses to a pass-through.
    let ssao_white = texture::create_fallback_white(&gpu.upload())?;
    Ok(VkSceneAssets {
        textures,
        fallback_textures,
        linear_sampler,
        cube_sampler,
        prefilter_mip_count: env_map.prefilter_mip_count,
        env_map,
        color_lut,
        ssao_white,
    })
}
