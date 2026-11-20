//! World-content effects: projected decals, volumetric fog, particles, water
//! surfaces, glass, the planar reflection set they share, and raymarched SDF
//! volumes. Each is built only when the world declares it, and a quality toggle
//! never rebuilds them.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_engine::components::GlassPanel;
use concinnity_engine::environment::water::WaterBody;
use concinnity_engine::environment::water::ocean::OceanPlan;
use concinnity_engine::geometry::proxy_cube::{CUBE_INDICES, cube_corners};
use concinnity_engine::gfx::render_types::DrawObject;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::backend::init::SdfVolumeSource;
use concinnity_engine::render::decal::{DecalRecord, DecalSet};
use concinnity_engine::render::environment::volumetric_fog::FogSettings;
use concinnity_engine::render::planar_reflection::PlanarReflectors;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLDevice, MTLRenderPipelineState, MTLResourceOptions, MTLSamplerAddressMode,
    MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerState,
};

use crate::metal::context::{GlassState, RaymarchState, WaterState};
use crate::metal::decal::{DecalState, build_decal_pipeline};
use crate::metal::error::allocation_failed;
use crate::metal::fog::{
    FogState, build_fog_froxel_pipeline, build_fog_froxel_volume, build_fog_pipeline,
};
use crate::metal::init::{Features, InitGpu};
use crate::metal::planar::PlanarReflectionSet;
use crate::metal::{glass, raymarch, raytrace, water};

// The projected-decal pass, built only with authored decals; a runtime
// `add_decal` builds it on demand. Also returns the slot table, seeded by the
// authored decals and uncapped on Metal.
pub(in crate::metal::init) fn build_decals(
    gpu: &InitGpu<'_>,
    decals: Vec<DecalRecord>,
) -> RenderResult<(DecalState, DecalSet)> {
    let (pipeline, cube_vertex_buffer, cube_index_buffer, sampler) = if decals.is_empty() {
        (None, None, None, None)
    } else {
        let (ps, vbuf, ibuf, samp) =
            build_decal_resources_for_runtime(&gpu.hw.device, gpu.hot_reload)?;
        (Some(ps), Some(vbuf), Some(ibuf), Some(samp))
    };
    let mut set = DecalSet::new(usize::MAX, gpu.frames_in_flight);
    for record in decals {
        set.insert(record)
            .map_err(|_| RenderError::Other("decals: decal slot table is full".to_string()))?;
    }
    let state = DecalState {
        pipeline,
        cube_vertex_buffer,
        cube_index_buffer,
        sampler,
    };
    Ok((state, set))
}

type DecalResources = (
    Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    Retained<ProtocolObject<dyn MTLBuffer>>,
    Retained<ProtocolObject<dyn MTLBuffer>>,
    Retained<ProtocolObject<dyn MTLSamplerState>>,
);

// The decal pipeline, unit cube (`[-0.5, 0.5]^3`, CCW outward) and sampler, at
// init or on the first runtime `add_decal`.
pub(in crate::metal) fn build_decal_resources_for_runtime(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> RenderResult<DecalResources> {
    let ps = build_decal_pipeline(device, hot_reload)?;
    const CUBE_VERTS: [[f32; 3]; 8] = cube_corners(0.5);
    // SAFETY: the pointer and length describe the live `CUBE_VERTS` allocation, and Metal copies
    // those bytes into the new buffer before the call returns.
    let vbuf = unsafe {
        let ptr = std::ptr::NonNull::new(CUBE_VERTS.as_ptr() as *mut _)
            .ok_or_else(|| RenderError::Other("decal cube vertex slice is null".to_string()))?;
        device
            .newBufferWithBytes_length_options(
                ptr,
                size_of_val(&CUBE_VERTS),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| allocation_failed("decal cube vertex buffer"))?
    };
    // SAFETY: the pointer and length describe the live `CUBE_INDICES` allocation, and Metal copies
    // those bytes into the new buffer before the call returns.
    let ibuf = unsafe {
        let ptr = std::ptr::NonNull::new(CUBE_INDICES.as_ptr() as *mut _)
            .ok_or_else(|| RenderError::Other("decal cube index slice is null".to_string()))?;
        device
            .newBufferWithBytes_length_options(
                ptr,
                size_of_val(&CUBE_INDICES),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| allocation_failed("decal cube index buffer"))?
    };
    let samp = {
        let desc = MTLSamplerDescriptor::new();
        desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
        desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
        desc.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
        desc.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
        device
            .newSamplerStateWithDescriptor(&desc)
            .ok_or_else(|| RenderError::Other("failed to create decal sampler state".to_string()))?
    };
    Ok((ps, vbuf, ibuf, samp))
}

// Volumetric-fog pipelines and froxel volume. Built only when the world
// declares a `VolumetricFog`; with none, the pipeline stays `None` and the fog
// pass is skipped by `draw_frame`.
pub(in crate::metal::init) fn build_fog(
    gpu: &InitGpu<'_>,
    settings: Option<FogSettings>,
) -> RenderResult<FogState> {
    let device = &*gpu.hw.device;
    let (pipeline, froxel_pipeline, froxel_volume) = if settings.is_some() {
        let render_ps = build_fog_pipeline(device, gpu.hot_reload)?;
        let compute_ps = build_fog_froxel_pipeline(device, gpu.hot_reload)?;
        let volume = build_fog_froxel_volume(device)?;
        (Some(render_ps), Some(compute_ps), Some(volume))
    } else {
        (None, None, None)
    };
    Ok(FogState {
        pipeline,
        froxel_pipeline,
        froxel_volume,
    })
}

// Water surfaces, built only with a `WaterSurface`: one tessellated grid per
// surface with its planar slot, beside the simulated oceans.
pub(in crate::metal::init) fn build_water(
    gpu: &InitGpu<'_>,
    water_surfaces: &[WaterBody],
    planar_slots: &[Option<usize>],
    ocean: &OceanPlan,
) -> RenderResult<WaterState> {
    let device = &*gpu.hw.device;
    let hot_reload = gpu.hot_reload;
    // The ray-traced reflections read the waves' slopes too, so a world without
    // water on a device that can trace still builds the stand-in arrays.
    let ocean = if !water_surfaces.is_empty() {
        Some(crate::metal::water_sim::OceanSim::build(
            device, ocean, hot_reload,
        )?)
    } else if raytrace::raytracing_supported(device) {
        Some(crate::metal::water_sim::OceanSim::stand_in(device)?)
    } else {
        None
    };
    let (pipeline, pipeline_rt, pipeline_rt_textured, surfaces) = if water_surfaces.is_empty() {
        (None, None, None, Vec::new())
    } else {
        let ps = water::build_water_pipeline(device, hot_reload)?;
        // RT variants build on any RT-capable device so a live toggle needs no
        // rebuild; the shader cannot compile without ray tracing.
        let (ps_rt, ps_rt_tex) = if raytrace::raytracing_supported(device) {
            (
                Some(water::build_water_pipeline_rt(device, hot_reload)?),
                Some(water::build_water_pipeline_rt_textured(device, hot_reload)?),
            )
        } else {
            (None, None)
        };
        let mut records = Vec::with_capacity(water_surfaces.len());
        for (s, slot) in water_surfaces.iter().zip(planar_slots) {
            let mut record = water::build_water_surface_record(device, s)?;
            record.planar_slot = *slot;
            records.push(record);
        }
        (Some(ps), ps_rt, ps_rt_tex, records)
    };
    let sea = water_surfaces
        .iter()
        .any(|b| b.sea.is_some())
        .then(|| {
            water::SeaPipelines::build(device, hot_reload, raytrace::raytracing_supported(device))
        })
        .transpose()?;
    Ok(WaterState {
        pipeline,
        pipeline_rt,
        pipeline_rt_textured,
        sea,
        surfaces,
        ocean,
        wake: concinnity_engine::render::environment::water::wake::WakeHistory::default(),
    })
}

// Translucent glass: the `GlassPanel` producer and the see-through mesh path.
pub(in crate::metal::init) fn build_glass(
    gpu: &InitGpu<'_>,
    glass_panels: &[GlassPanel],
    planar_slots: &[Option<usize>],
    draw_objects: &[DrawObject],
) -> RenderResult<GlassState> {
    let device = &*gpu.hw.device;
    let hot_reload = gpu.hot_reload;

    // Glass panels. Built only when the world declared ≥1 `GlassPanel`; rides
    // the same transparent pass as water. Per-panel world-space quads upload
    // once at init, each recording the planar slot `PlanarReflectors::plan` gave it.
    let (pipeline, pipeline_rt, pipeline_rt_textured, panels) = if glass_panels.is_empty() {
        (None, None, None, Vec::new())
    } else {
        let ps = glass::build_glass_pipeline(device, hot_reload)?;
        // RT variants build on any RT-capable device so a live toggle needs no
        // rebuild; the shader cannot compile without ray tracing.
        let (ps_rt, ps_rt_tex) = if raytrace::raytracing_supported(device) {
            (
                Some(glass::build_glass_pipeline_rt(device, hot_reload)?),
                Some(glass::build_glass_pipeline_rt_textured(device, hot_reload)?),
            )
        } else {
            (None, None)
        };
        let mut records = Vec::with_capacity(glass_panels.len());
        for (g, slot) in glass_panels.iter().zip(planar_slots) {
            let mut record = glass::build_glass_panel_record(device, g)?;
            record.planar_slot = *slot;
            records.push(record);
        }
        (Some(ps), ps_rt, ps_rt_tex, records)
    };

    // See-through glass mesh pipelines, built on any RT-capable device whether
    // or not the world has panels.
    let (mesh_pipeline_rt, mesh_pipeline_rt_textured) = if raytrace::raytracing_supported(device) {
        (
            Some(glass::build_glass_mesh_pipeline_rt(device, hot_reload)?),
            Some(glass::build_glass_mesh_pipeline_rt_textured(
                device, hot_reload,
            )?),
        )
    } else {
        (None, None)
    };
    // See-through glass is opt-in per `Material::see_through`; this list drives
    // the transparent producer, the opaque-pass skip and the BLAS exclusion.
    let seethrough_mesh_indices: Vec<usize> = draw_objects
        .iter()
        .enumerate()
        .filter(|(_, o)| o.material.transparent != 0 && o.material.see_through != 0)
        .map(|(i, _)| i)
        .collect();

    Ok(GlassState {
        pipeline,
        pipeline_rt,
        pipeline_rt_textured,
        mesh_pipeline_rt,
        mesh_pipeline_rt_textured,
        seethrough_mesh_indices,
        panels,
        reflection_targets: None,
    })
}

// The planar mirror targets, one set per assigned plane at the mirror
// resolution. Built only when the world has >=1 reflector holding a slot; the
// per-frame pass is additionally gated on what the transparent pass samples.
pub(in crate::metal::init) fn build_planar_reflection(
    gpu: &InitGpu<'_>,
    planar: PlanarReflectors,
    features: &Features,
) -> RenderResult<Option<PlanarReflectionSet>> {
    if planar.planes().is_empty() {
        return Ok(None);
    }
    Ok(Some(PlanarReflectionSet::new(
        &gpu.hw.device,
        planar,
        features.render,
        features.hdr_samples,
    )?))
}

// Raymarched SDF volumes. Each volume builds its own pipelines from the
// field the build compiled into its payload; the proxy-cube buffers are
// allocated once and shared across all volumes. Empty input list means
// both stay None / empty and the raymarch executor short-circuits.
pub(in crate::metal::init) fn build_raymarch(
    gpu: &InitGpu<'_>,
    sdf_volumes: &[SdfVolumeSource],
) -> RenderResult<RaymarchState> {
    let device = &*gpu.hw.device;
    let (volumes, cube_vertex_buffer, cube_index_buffer) = if sdf_volumes.is_empty() {
        (Vec::new(), None, None)
    } else {
        let mut records = Vec::with_capacity(sdf_volumes.len());
        for SdfVolumeSource {
            volume,
            fragment_source: payload,
            label,
        } in sdf_volumes
        {
            records.push(raymarch::build_raymarch_volume_record(
                device,
                volume,
                payload,
                gpu.hot_reload,
                label,
            )?);
        }
        let (vb, ib) = raymarch::build_raymarch_cube_buffers(device)?;
        (records, Some(vb), Some(ib))
    };
    Ok(RaymarchState {
        volumes,
        cube_vertex_buffer,
        cube_index_buffer,
    })
}
