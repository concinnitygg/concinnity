//! Scene assets: the area-light tables, the IBL cubes and the probe cube array
//! they seed, the world's textures with their reserved fallbacks and the flat
//! bindless pool views, the color-grading LUT, and the static geometry.

use concinnity_core::bake;
use concinnity_core::gfx::mesh_payload::Vertex;
use concinnity_core::gfx::render_types::{AreaLightData, FALLBACK_TEXTURE_COUNT};
use concinnity_core::render::backend_init::{MediaPayloads, SceneData};
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::ltc;
use windows::Win32::Graphics::Direct3D12::*;

use super::InitGpu;
use crate::directx::allocator::PooledTexture;
use crate::directx::com;
use crate::directx::context::{
    AreaLightState, DxDescriptors, DxGeometry, DxSceneAssets, FRAMES, align256,
};
use crate::directx::draw::upload_static_records;
use crate::directx::texture::*;

pub(super) fn build_scene_assets(
    gpu: &InitGpu<'_>,
    descriptors: &DxDescriptors,
    media: &MediaPayloads<'_>,
    area_lights: &[AreaLightData],
    world: &SceneData<'_>,
) -> RenderResult<DxSceneAssets> {
    let hw = gpu.hw;
    let layout = &descriptors.layout;
    let area_light = build_area_light(gpu, descriptors, area_lights)?;

    // IBL cubemaps (irradiance + prefilter)
    // When env_map_bytes is Some, deserialize the EnvironmentMap payload and
    // upload both cubes. Otherwise bind a 1x1 gray fallback for each; the
    // shader keys off prefilter_mip_count == 0 to skip IBL math.
    let env_map = if let Some(bytes) = media.env_map_bytes {
        let view = bake::environment_map::deserialize(bytes)
            .map_err(|e| format!("EnvironmentMap payload malformed: {e}"))?;
        upload_environment_map(
            &hw.alloc,
            crate::directx::texture::EnvironmentMapPayload {
                irradiance_face: view.irradiance_face,
                irradiance_bytes: view.irradiance_bytes,
                prefilter_face: view.prefilter_face,
                mip_bytes: &view.prefilter_mip_bytes,
            },
            crate::directx::texture::EnvironmentMapDescriptors {
                irr_srv_cpu: descriptors.slot_cpu(1),
                irr_srv_gpu: descriptors.slot_gpu(1),
                pre_srv_cpu: descriptors.slot_cpu(2),
                pre_srv_gpu: descriptors.slot_gpu(2),
            },
        )?
    } else {
        let irradiance = create_fallback_cubemap(
            &hw.alloc,
            [0.05, 0.05, 0.05, 1.0],
            descriptors.slot_cpu(1),
            descriptors.slot_gpu(1),
        )?;
        let prefilter = create_fallback_cubemap(
            &hw.alloc,
            [0.05, 0.05, 0.05, 1.0],
            descriptors.slot_cpu(2),
            descriptors.slot_gpu(2),
        )?;
        EnvironmentMapTextures {
            irradiance,
            prefilter,
            prefilter_mip_count: 0,
        }
    };

    // Reflection-probe cube array: point every slot at the sky prefilter cube so
    // the bindless main shader's `probe_cubes` table is valid before any probe
    // bakes (unbaked slots stay the sky; a baked probe overwrites its slot in
    // `probe_install`). The forward shader only samples a slot when
    // `ProbeSet.count` covers it, but the descriptor table must still be valid.
    let probe_sky_mips = env_map.prefilter_mip_count.max(1);
    for k in 0..concinnity_core::render::uniforms::MAX_PROBES {
        crate::directx::texture::write_cube_srv_mips(
            &hw.device,
            &env_map.prefilter.resource,
            probe_sky_mips,
            descriptors.slot_cpu(layout.probe_cube_base_slot + k),
        );
    }

    // Albedo texture pool
    // One ID3D12Resource per input texture; SRVs are written below at
    // per-object pair slots so a single texture can be referenced by many
    // objects. When no textures were declared, a single 1x1 white fallback
    // stands in so every object's albedo slot resolves to opaque white.
    let gpu_textures: Vec<PooledTexture> = if media.textures.is_empty() {
        vec![create_fallback_white_resource(&hw.alloc)?]
    } else {
        media
            .textures
            .iter()
            .enumerate()
            .map(|(i, image)| {
                upload_texture_image(&hw.alloc, image)
                    .map_err(|e| e.context(format!("texture[{i}]")))
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    // Reserved fallbacks, in the order `FALLBACK_TEXTURE_COUNT` documents:
    // the flat-normal resource a draw with no normal map samples, then the
    // white resource a draw with no albedo samples. Real normal maps and
    // albedos are textures in `gpu_textures` (the shared pool), addressed by
    // their own handle; only these two live in `fallback_textures`.
    let gpu_fallbacks: Vec<PooledTexture> = vec![
        create_fallback_flat_normal_resource(&hw.alloc)?,
        create_fallback_white_resource(&hw.alloc)?,
    ];

    // Flat deduplicated bindless pool: one SRV per distinct texture, then the
    // fallback pair at `flat_albedo_count`. The bindless main pass and
    // the RT hit shader bind this region's base and index it by a flat slot
    // (`albedo = texture_slot` or the white slot when the draw has none,
    // `normal = normal's own handle` or the flat-normal slot), mirroring
    // Vulkan/Metal. A shared texture resolves to ONE descriptor here.
    let flat_albedo_count = gpu_textures.len();
    debug_assert_eq!(gpu_fallbacks.len(), FALLBACK_TEXTURE_COUNT);
    debug_assert_eq!(
        flat_albedo_count + gpu_fallbacks.len(),
        descriptors.flat_pool_len
    );
    for f in 0..FRAMES {
        let copy_base = layout.flat_pool_base_slot + f * descriptors.flat_pool_len;
        for (k, tex) in gpu_textures.iter().enumerate() {
            write_texture_srv(&hw.device, tex, descriptors.slot_cpu(copy_base + k));
        }
        for (k, tex) in gpu_fallbacks.iter().enumerate() {
            write_texture_srv(
                &hw.device,
                tex,
                descriptors.slot_cpu(copy_base + flat_albedo_count + k),
            );
        }
    }

    // Color-grading LUT
    // Upload the declared `ColorLut` payload, or build a 2x2x2 identity LUT
    // so the composite pass always binds a valid Texture3D. With the
    // identity LUT the grade is a no-op at any `lut_strength`.
    let color_lut = if let Some(bytes) = media.color_lut_bytes {
        let (size, data) = bake::color_lut::deserialize(bytes)
            .map_err(|e| format!("ColorLut payload malformed: {e}"))?;
        upload_color_lut(
            &hw.alloc,
            size,
            data,
            descriptors.slot_cpu(layout.lut_srv_slot),
            descriptors.slot_gpu(layout.lut_srv_slot),
        )?
    } else {
        create_fallback_color_lut(
            &hw.alloc,
            descriptors.slot_cpu(layout.lut_srv_slot),
            descriptors.slot_gpu(layout.lut_srv_slot),
        )?
    };

    let geometry = build_geometry(gpu, world)?;
    Ok(DxSceneAssets {
        env_map,
        color_lut,
        area_light,
        geometry,
        textures: gpu_textures,
        _fallback_textures: gpu_fallbacks,
    })
}

fn build_area_light(
    gpu: &InitGpu<'_>,
    descriptors: &DxDescriptors,
    area_lights: &[AreaLightData],
) -> RenderResult<AreaLightState> {
    let hw = gpu.hw;
    let ltc_srv_base_slot = descriptors.layout.ltc_srv_base_slot;
    // Per-scene rectangular area lights: the edge vectors that do not fit in
    // `GpuLight`, indexed by its `data_index`. A world with no area light
    // still gets a one-element buffer, since the shader never reads it
    // (`data_index` stays -1) but the root SRV must be valid.
    let area_light_data = if area_lights.is_empty() {
        vec![AreaLightData::ZERO]
    } else {
        area_lights.to_vec()
    };
    let area_light_buffer = {
        let size = align256((area_light_data.len() * size_of::<AreaLightData>()) as u64);
        let buf = create_buffer(
            &hw.alloc,
            size,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        upload_static_records(&buf, &area_light_data, "area-light")?;
        buf
    };

    // Area-light LTC tables. Scene-independent (they depend only on the
    // build-time fit), so they are created unconditionally and the shader
    // simply never samples them when no area light is declared.
    let ltc_size = ltc::LTC_LUT_SIZE as u32;
    let ltc_matrix_texture = upload_float_lut(
        &hw.alloc,
        ltc_size,
        4,
        ltc::matrix_texels(),
        descriptors.slot_cpu(ltc_srv_base_slot),
        descriptors.slot_gpu(ltc_srv_base_slot),
    )?;
    let ltc_magnitude_texture = upload_float_lut(
        &hw.alloc,
        ltc_size,
        2,
        ltc::magnitude_texels(),
        descriptors.slot_cpu(ltc_srv_base_slot + 1),
        descriptors.slot_gpu(ltc_srv_base_slot + 1),
    )?;
    Ok(AreaLightState {
        buffer: area_light_buffer,
        ltc_matrix: ltc_matrix_texture,
        ltc_magnitude: ltc_magnitude_texture,
        ltc_table_gpu: descriptors.slot_gpu(ltc_srv_base_slot),
    })
}

fn build_geometry(gpu: &InitGpu<'_>, world: &SceneData<'_>) -> RenderResult<DxGeometry> {
    let hw = gpu.hw;
    let vert_bytes_raw = bytemuck::cast_slice(world.vertices);
    let idx_bytes_raw = bytemuck::cast_slice(world.indices);
    let vertex_buffer = upload_buffer(
        &hw.alloc,
        vert_bytes_raw,
        D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
    )?;
    let index_buffer = upload_buffer(&hw.alloc, idx_bytes_raw, D3D12_RESOURCE_STATE_INDEX_BUFFER)?;

    let vertex_buffer_view = D3D12_VERTEX_BUFFER_VIEW {
        BufferLocation: com::gpu_va(&vertex_buffer),
        SizeInBytes: vert_bytes_raw.len().max(4) as u32,
        StrideInBytes: std::mem::size_of::<Vertex>() as u32,
    };
    let index_buffer_view = D3D12_INDEX_BUFFER_VIEW {
        BufferLocation: com::gpu_va(&index_buffer),
        SizeInBytes: idx_bytes_raw.len().max(4) as u32,
        // Static IB is u32: the `indices: &[u32]` signature is honored
        // end-to-end. A previous half-completed migration left this as
        // R16_UINT while the byte count was already widened: the GPU then
        // read each u32 index as a pair of u16s, indexing into garbage
        // vertices and shearing every static prop's geometry.
        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT,
    };
    Ok(DxGeometry {
        vertex_buffer,
        index_buffer,
        vertex_buffer_view,
        index_buffer_view,
    })
}
