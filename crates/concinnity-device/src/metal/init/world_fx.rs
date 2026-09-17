//! World-content effects: projected decals, volumetric fog, particles, water
//! surfaces, glass, the planar reflection set they share, and raymarched SDF
//! volumes. Each is built only when the world declares it, and a quality toggle
//! never rebuilds them.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::components::{GlassPanel, WaterSurface};
use concinnity_core::gfx::render_types::DrawObject;
use concinnity_core::render::backend_init::SdfVolumeSource;
use concinnity_core::render::decal::{DecalRecord, DecalSet};
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::particles::ParticleEmitterRecord;
use concinnity_core::render::planar_reflection::{self, PlanarAssignment};
use concinnity_core::render::volumetric_fog::FogSettings;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLDevice, MTLRenderPipelineState, MTLResourceOptions, MTLSamplerAddressMode,
    MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerState,
};

use super::{Features, InitGpu};
use crate::metal::context::{GlassState, RaymarchState, WaterState};
use crate::metal::decal::{DecalState, build_decal_pipeline};
use crate::metal::error::allocation_failed;
use crate::metal::fog::{
    FogState, build_fog_froxel_pipeline, build_fog_froxel_volume, build_fog_pipeline,
};
use crate::metal::particle::{ParticleState, build_emitter_gpu_state, build_particle_pipelines};
use crate::metal::planar::{MAX_PLANAR_PLANES, PlanarReflectionSet, create_planar_set};
use crate::metal::{glass, raymarch, raytrace, water};

// Projected-decal pass. The pipeline and unit cube are built only when the
// world declares at least one decal; with none, all four resources stay `None`
// and the pass is skipped by `draw_frame`. The first runtime
// [`MtlContext::add_decal`] for a world that started with no decals will
// rebuild the same four resources on demand. The slot table the decal pass
// draws from: authored decals seed it in order, and a runtime add reuses
// whatever `remove_decal` freed. Metal reserves no per-decal descriptors, so the
// table is uncapped.
pub(super) fn build_decals(
    gpu: &InitGpu<'_>,
    decals: Vec<DecalRecord>,
) -> RenderResult<DecalState> {
    let (pipeline, cube_vertex_buffer, cube_index_buffer, sampler) = if !decals.is_empty() {
        let (ps, vbuf, ibuf, samp) =
            build_decal_resources_for_runtime(&gpu.hw.device, gpu.hot_reload)?;
        (Some(ps), Some(vbuf), Some(ibuf), Some(samp))
    } else {
        (None, None, None, None)
    };
    let mut set = DecalSet::new(usize::MAX, gpu.frames_in_flight);
    for record in decals {
        set.insert(record)
            .map_err(|_| RenderError::Other("decals: decal slot table is full".to_string()))?;
    }
    Ok(DecalState {
        set,
        pipeline,
        cube_vertex_buffer,
        cube_index_buffer,
        sampler,
    })
}

type DecalResources = (
    Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    Retained<ProtocolObject<dyn MTLBuffer>>,
    Retained<ProtocolObject<dyn MTLBuffer>>,
    Retained<ProtocolObject<dyn MTLSamplerState>>,
);

// Build the projected-decal pipeline + the unit-cube vertex / index buffers
// + the shared sampler. Called either at init when the world declared ≥1
// decal, or lazily by [`crate::metal::MtlContext::add_decal`] on the first
// runtime add for a world that started with none. The unit cube spans
// `[-0.5, 0.5]^3` -- the same local space the decal `inv_model` maps a
// reconstructed world point into. 36 indices form 12 triangles wound CCW
// outward.
pub(in crate::metal) fn build_decal_resources_for_runtime(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> RenderResult<DecalResources> {
    let ps = build_decal_pipeline(device, hot_reload)?;
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct CubeVtx {
        p: [f32; 3],
    }
    const CUBE_VERTS: [CubeVtx; 8] = [
        CubeVtx {
            p: [-0.5, -0.5, -0.5],
        },
        CubeVtx {
            p: [0.5, -0.5, -0.5],
        },
        CubeVtx {
            p: [0.5, 0.5, -0.5],
        },
        CubeVtx {
            p: [-0.5, 0.5, -0.5],
        },
        CubeVtx {
            p: [-0.5, -0.5, 0.5],
        },
        CubeVtx {
            p: [0.5, -0.5, 0.5],
        },
        CubeVtx { p: [0.5, 0.5, 0.5] },
        CubeVtx {
            p: [-0.5, 0.5, 0.5],
        },
    ];
    const CUBE_INDICES: [u16; 36] = [
        // -Z face                    +Z face
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, // -Y                         +Y
        0, 1, 5, 0, 5, 4, 3, 6, 2, 3, 7, 6, // -X                         +X
        0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
    ];
    // SAFETY: the pointer and length describe the live `CUBE_VERTS` allocation, and Metal copies
    // those bytes into the new buffer before the call returns.
    let vbuf = unsafe {
        let ptr = std::ptr::NonNull::new(CUBE_VERTS.as_ptr() as *mut _)
            .ok_or_else(|| RenderError::Other("decal cube vertex slice is null".to_string()))?;
        device
            .newBufferWithBytes_length_options(
                ptr,
                std::mem::size_of_val(&CUBE_VERTS),
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
                std::mem::size_of_val(&CUBE_INDICES),
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
pub(super) fn build_fog(
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
        settings,
        pipeline,
        froxel_pipeline,
        froxel_volume,
    })
}

// Particle compute + render pipelines, plus one persistent GPU pool per
// emitter. Pools are zero-initialized so every slot starts dead; the
// compute kernel spawns into them on its first dispatch.
pub(super) fn build_particles(
    gpu: &InitGpu<'_>,
    particles: Vec<ParticleEmitterRecord>,
) -> RenderResult<ParticleState> {
    let device = &*gpu.hw.device;
    let (pipelines, emitter_state) = if !particles.is_empty() {
        let pipelines = build_particle_pipelines(device, gpu.hot_reload)?;
        let mut states = Vec::with_capacity(particles.len());
        for rec in &particles {
            states.push(build_emitter_gpu_state(device, rec, gpu.frames_in_flight)?);
        }
        (Some(pipelines), states)
    } else {
        (None, Vec::new())
    };
    Ok(ParticleState {
        records: particles.into_iter().map(Some).collect(),
        emitter_state: emitter_state.into_iter().map(Some).collect(),
        free_slots: Vec::new(),
        pipelines,
        last_elapsed: 0.0,
        frame_index: 0,
        counter_slot: 0,
    })
}

// Planar reflection slots: group every flat reflector (water surfaces + glass
// panes) into a bounded number of distinct planes, one mirror render each.
// Water planes are listed first so they take slots before glass when the budget
// is tight. Each reflector records the slot it samples; planes past the budget
// get no slot and keep the box-projected probe cube (warned here, not silently
// dropped).
pub(super) fn plan_planar(
    water_surfaces: &[WaterSurface],
    glass_panels: &[GlassPanel],
    planar_planes: usize,
) -> PlanarAssignment {
    let mut planes: Vec<[f32; 4]> = Vec::new();
    for s in water_surfaces {
        // Horizontal plane at the surface base height, normal +y.
        planes.push([0.0, 1.0, 0.0, -s.center[1]]);
    }
    for g in glass_panels {
        // The pane plane: normal (unit from `from_args`) through center,
        // so `n . p + d = 0` on the pane.
        let n = g.normal;
        let d = -(n[0] * g.center[0] + n[1] * g.center[1] + n[2] * g.center[2]);
        planes.push([n[0], n[1], n[2], d]);
    }
    // The budget is capped at the capacity ceiling the mirror targets + ICB
    // slots are sized to, so a stale/over-large preset value can never
    // over-allocate.
    let planar_budget = planar_planes.min(MAX_PLANAR_PLANES);
    let assignment = planar_reflection::assign_planar_slots(&planes, planar_budget);
    let overflow = assignment.slots.iter().filter(|s| s.is_none()).count();
    if overflow > 0 {
        tracing::warn!(
            "planar reflection: {} reflector plane(s) exceed the budget of {} \
             and fall back to the box-projected probe cube",
            overflow,
            planar_budget
        );
    }
    assignment
}

// Transparent water surfaces. Built only when the world declared
// ≥1 `WaterSurface`; the transparent-pass executor stays a no-op
// otherwise. Per-surface tessellated grids upload once at init, each recording
// the planar slot `plan_planar` gave it.
pub(super) fn build_water(
    gpu: &InitGpu<'_>,
    water_surfaces: &[WaterSurface],
    planar_slots: &[Option<usize>],
) -> RenderResult<WaterState> {
    let device = &*gpu.hw.device;
    let hot_reload = gpu.hot_reload;
    let (pipeline, pipeline_rt, pipeline_rt_textured, surfaces) = if water_surfaces.is_empty() {
        (None, None, None, Vec::new())
    } else {
        let ps = water::build_water_pipeline(device, hot_reload)?;
        // The ray-traced variants are built whenever the device can ray
        // trace (regardless of whether RT is on at launch), so a live RT
        // toggle can select them without a pipeline rebuild. The shader
        // uses `metal_raytracing`, so it must not be compiled on a non-RT
        // device. The textured variant additionally needs a bindless world
        // at draw time; it is selected over the flat variant then.
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
    Ok(WaterState {
        pipeline,
        pipeline_rt,
        pipeline_rt_textured,
        surfaces,
    })
}

// Translucent glass: the `GlassPanel` producer and the see-through mesh path.
pub(super) fn build_glass(
    gpu: &InitGpu<'_>,
    glass_panels: &[GlassPanel],
    planar_slots: &[Option<usize>],
    draw_objects: &[DrawObject],
) -> RenderResult<GlassState> {
    let device = &*gpu.hw.device;
    let hot_reload = gpu.hot_reload;

    // Glass panels. Built only when the world declared ≥1 `GlassPanel`; rides
    // the same transparent pass as water. Per-panel world-space quads upload
    // once at init, each recording the planar slot `plan_planar` gave it.
    let (pipeline, pipeline_rt, pipeline_rt_textured, panels) = if glass_panels.is_empty() {
        (None, None, None, Vec::new())
    } else {
        let ps = glass::build_glass_pipeline(device, hot_reload)?;
        // The ray-traced variants are built whenever the device can ray
        // trace (regardless of whether RT is on at launch), so a live RT
        // toggle can select them without a pipeline rebuild. The shader
        // uses `metal_raytracing`, so it must not be compiled on a non-RT
        // device. The textured variant additionally needs a bindless world
        // at draw time; it is selected over the flat variant then.
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

    // Transparent glass MESH pipelines (Layer 2): built whenever the device can
    // ray trace, INDEPENDENT of any `GlassPanel` -- the transparent material
    // lives on imported meshes, not panels, and a live RT toggle then has them
    // ready. `mesh_pipeline_rt.is_some()` gates the whole transparent-mesh
    // reroute; `seethrough_mesh_indices` marks which `draw_objects` carry it.
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
    // Layer 2 see-through glass is opt-in per `Material` (the `see_through`
    // arg, which implies `transparent`): see-through only looks right when the
    // space behind the glass is modeled. A material that is `transparent` but
    // NOT `see_through` renders as Layer 1 (opaque, low roughness, scene
    // reflections) = tinted reflective glass that hides the interior. This list
    // drives the producer + the opaque-pass skip (`mesh_glass_active`) + the
    // RT-BLAS exclude together.
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
    })
}

// The planar mirror targets, one set per assigned plane at the render
// resolution. Built only when the world has >=1 reflector; the per-frame pass is
// additionally gated on RT being off.
pub(super) fn build_planar_reflection(
    gpu: &InitGpu<'_>,
    planar: &PlanarAssignment,
    features: &Features,
) -> RenderResult<Option<PlanarReflectionSet>> {
    if planar.representatives.is_empty() {
        return Ok(None);
    }
    Ok(Some(create_planar_set(
        &gpu.hw.device,
        features.render.0,
        features.render.1,
        features.hdr_samples,
        &planar.representatives,
    )?))
}

// Raymarched SDF volumes. Each volume builds its own pipelines from the
// field the build compiled into its payload; the proxy-cube buffers are
// allocated once and shared across all volumes. Empty input list means
// both stay None / empty and the raymarch executor short-circuits.
pub(super) fn build_raymarch(
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
