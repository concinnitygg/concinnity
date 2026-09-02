// src/vulkan/water.rs
//
// WaterSurface: one producer of the engine's transparent pass on the Vulkan
// backend (`transparent.rs` owns the render pass itself, the scene snapshot, the
// shared descriptor / pipeline layouts and the combined back-to-front draw
// order; `glass.rs` is the other producer). Each surface is a flat tessellated
// XZ grid built once at init and displaced per frame by the vertex stage's
// Gerstner sum; the fragment refracts the pass's scene snapshot, tints and foams
// it by the water-column thickness the main depth gives, and mixes a reflection
// over it by a Schlick Fresnel term (see shaders/water.slang, the single source
// all three backends compile).
//
// Same uniform layouts, back-to-front ordering and manual depth-occlusion test
// as the DirectX and Metal hosts.

use ash::vk;

use super::allocator::DeviceAllocator;
use crate::components::{MAX_WATER_WAVES, WaterSurface, WaterWave};
use crate::geometry::water_grid::build_water_grid;
use crate::gfx::mesh_payload::Vertex;
use crate::vulkan::slang_builtins::SlangCompile;
use crate::vulkan::transparent::{
    ProducerCtx, RecordUpload, TransparentProducer, TransparentRecord, TransparentVertexInput,
    create_transparent_pipeline,
};

// `WaterParams` / `WaterWaveGpu` (the per-surface UBO and its wave lanes) are
// GPU-free layout structs that live in `core::render`; re-export them so
// `crate::vulkan::water::WaterParams` is unchanged for the `water_params_from`
// path.
pub(in crate::vulkan) use concinnity_core::render::uniforms::{
    WATER_MAX_WAVES, WaterParams, WaterWaveGpu,
};

// The shader-side wave lane for one authored wave. Pure; unit tested.
fn wave_to_gpu(w: &WaterWave) -> WaterWaveGpu {
    WaterWaveGpu {
        dir_amp_wave: [w.direction[0], w.direction[1], w.amplitude, w.wavelength],
        speed_steep_pad: [w.speed, w.steepness, 0.0, 0.0],
    }
}

// Build the per-surface `WaterParams` from an authored surface. `planar` is the
// mirror lane with its ripple offset scaled by the surface's roughness when the
// surface has a planar reflection slot, and zeroed otherwise, which is what
// selects the sharp mirror render over the probe / sky cube. Pure; unit tested. Mirrors `directx::water::water_params_from`.
fn water_params_from(surface: &WaterSurface, planar: bool) -> WaterParams {
    let mut waves = [WaterWaveGpu::default(); WATER_MAX_WAVES];
    for (slot, src) in waves.iter_mut().zip(surface.waves.iter()) {
        *slot = wave_to_gpu(src);
    }
    WaterParams {
        centre: [surface.centre[0], surface.centre[1], surface.centre[2], 0.0],
        deep_colour: [
            surface.deep_colour[0],
            surface.deep_colour[1],
            surface.deep_colour[2],
            0.0,
        ],
        shallow_colour: [
            surface.shallow_colour[0],
            surface.shallow_colour[1],
            surface.shallow_colour[2],
            0.0,
        ],
        depth_falloff: surface.depth_falloff_metres,
        foam_width: surface.foam_width_metres,
        foam_intensity: surface.foam_intensity,
        fresnel_power: surface.fresnel_power,
        roughness: surface.roughness,
        refraction_strength: surface.refraction_strength,
        wave_count: surface.waves.len().min(MAX_WATER_WAVES) as u32,
        _pad: 0.0,
        waves,
        planar: WaterParams::planar_lane(surface.roughness, planar),
    }
}

// Compile the water vertex + fragment shaders, injecting the MSAA define so the
// depth sampler type matches the main-depth resource's sample count.
fn compile_water_shaders(
    hot_reload: bool,
    msaa: bool,
    probe_cube_count: u32,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let ctx = super::builtins::Ctx {
        hot_reload,
        msaa,
        pool_size: 0,
        probe_count: probe_cube_count as usize,
    };
    let vert = super::slang_builtins::WATER_VERT.compile(&ctx)?;
    let frag = super::slang_builtins::WATER_FRAG.compile(&ctx)?;
    Ok((vert, frag))
}

// SPIR-V blobs for the ray-traced water pipelines: the shared vertex stage (the
// same one the base pass uses -- the trace is entirely in the fragment), the
// flat fragment, and the textured fragment (`None` when the bindless pool is
// absent).
struct WaterRtShaders {
    vs: Vec<u8>,
    flat_fs: Vec<u8>,
    textured_fs: Option<Vec<u8>>,
}

// Compile the water vertex shader + the ray-traced water fragment (flat, plus
// the textured variant when `pool_size > 0`). slangc emits `SPV_KHR_ray_query`
// for the traversal, which the device already advertises wherever these
// pipelines are built.
fn compile_water_rt_shaders(
    hot_reload: bool,
    msaa: bool,
    pool_size: usize,
    probe_cube_count: u32,
) -> Result<WaterRtShaders, String> {
    // The pool declaration needs at least one slot even when the bindless pool
    // is absent (the textured variant is then skipped).
    let ctx = super::builtins::Ctx {
        hot_reload,
        msaa,
        pool_size: pool_size.max(1),
        probe_count: probe_cube_count as usize,
    };
    let vs = super::slang_builtins::WATER_VERT.compile(&ctx)?;
    let flat_fs = super::slang_builtins::WATER_FRAG_RT.compile(&ctx)?;
    let textured_fs = if pool_size > 0 {
        Some(super::slang_builtins::WATER_FRAG_RT_TEXTURED.compile(&ctx)?)
    } else {
        None
    };
    Ok(WaterRtShaders {
        vs,
        flat_fs,
        textured_fs,
    })
}

// Upload one surface's tessellated grid VB + IB and its per-surface
// `WaterParams` UBO, then allocate + write the surface's descriptor set.
fn build_surface_record(
    alloc: &DeviceAllocator,
    ctx: &ProducerCtx,
    surface: &WaterSurface,
    planar_slot: Option<usize>,
) -> Result<TransparentRecord, String> {
    let (verts, idxs) =
        build_water_grid(surface.extent[0], surface.extent[1], surface.subdivisions)?;

    // Flatten into the standard engine `Vertex` layout. Tangent and colour are
    // placeholders: the water shader rebuilds its normal frame analytically from
    // the wave derivatives and the fragment ignores per-vertex colour.
    let packed: Vec<Vertex> = verts
        .into_iter()
        .map(|(pos, normal, color, uv)| Vertex {
            pos,
            normal,
            tangent: [1.0, 0.0, 0.0],
            color,
            uv,
        })
        .collect();

    let params = water_params_from(surface, planar_slot.is_some());
    TransparentRecord::upload(
        alloc,
        ctx.record_descriptors(planar_slot),
        RecordUpload {
            vertices: &packed,
            indices: &idxs,
            params: bytemuck::bytes_of(&params),
            visible: surface.visible,
            centre: surface.centre,
            planar_slot,
        },
    )
}

// Build the water pipelines and one record per authored surface. The RT pair is
// built whenever the pass has RT pipeline layouts (regardless of whether RT is on
// at launch, so a live `quality-set ray_traced_reflections` selects it with no
// pipeline rebuild); a compile failure leaves it absent and the base
// probe/planar path runs.
pub(in crate::vulkan) fn build_water_producer(
    ctx: ProducerCtx,
    surfaces: &[WaterSurface],
    // Per-surface planar resolve slot (aligned with `surfaces`); `None` surfaces
    // keep the probe/sky reflection. From `assign_planar_slots`.
    planar_slots: &[Option<usize>],
) -> Result<TransparentProducer, String> {
    let (vert_spv, frag_spv) =
        compile_water_shaders(ctx.hot_reload, ctx.msaa, ctx.probe_cube_count)?;
    let pipeline = create_transparent_pipeline(
        ctx.device,
        ctx.render_pass,
        ctx.layout,
        &vert_spv,
        &frag_spv,
        TransparentVertexInput::Position,
    )?;

    let (flat_rt_pso, textured_rt_pso) = match ctx.rt_layout_flat {
        Some(flat_layout) => match build_water_rt_pipelines(&ctx, flat_layout) {
            Ok(pair) => pair,
            Err(e) => {
                tracing::warn!(
                    "water RT pipelines failed to build ({e}); using the probe / planar water path"
                );
                (None, None)
            }
        },
        None => (None, None),
    };

    let mut records = Vec::with_capacity(surfaces.len());
    for (i, surface) in surfaces.iter().enumerate() {
        let planar_slot = planar_slots.get(i).copied().flatten();
        records.push(build_surface_record(ctx.alloc, &ctx, surface, planar_slot)?);
    }

    Ok(TransparentProducer {
        pipeline,
        flat_rt_pso,
        textured_rt_pso,
        records,
    })
}

// The flat + textured RT water pipelines. The textured one is skipped when the
// bindless pool is absent or the device could not spare a fifth descriptor set,
// leaving the flat trace.
type WaterRtPipelines = (
    Option<super::owned::OwnedPipeline>,
    Option<super::owned::OwnedPipeline>,
);

fn build_water_rt_pipelines(
    ctx: &ProducerCtx,
    flat_layout: vk::PipelineLayout,
) -> Result<WaterRtPipelines, String> {
    let shaders = compile_water_rt_shaders(
        ctx.hot_reload,
        ctx.msaa,
        ctx.bindless_pool_size,
        ctx.probe_cube_count,
    )?;
    let flat = create_transparent_pipeline(
        ctx.device,
        ctx.render_pass,
        flat_layout,
        &shaders.vs,
        &shaders.flat_fs,
        TransparentVertexInput::Position,
    )?;
    let textured = match (ctx.rt_layout_textured, &shaders.textured_fs) {
        (Some(layout), Some(fs)) => Some(create_transparent_pipeline(
            ctx.device,
            ctx.render_pass,
            layout,
            &shaders.vs,
            fs,
            TransparentVertexInput::Position,
        )?),
        _ => None,
    };
    Ok((Some(flat), textured))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The `WaterParams` / `WaterWaveGpu` layout tests live with the structs in
    // `concinnity_core::render::uniforms`, and are checked against the compiled shader
    // by `shader_layout`.

    #[test]
    fn wave_to_gpu_packs_the_lanes() {
        let w = WaterWave {
            amplitude: 0.25,
            wavelength: 3.0,
            speed: 1.5,
            direction: [0.6, -0.8],
            steepness: 0.4,
        };
        let g = wave_to_gpu(&w);
        assert_eq!(g.dir_amp_wave, [0.6, -0.8, 0.25, 3.0]);
        assert_eq!(g.speed_steep_pad, [1.5, 0.4, 0.0, 0.0]);
    }

    #[test]
    fn water_params_from_maps_fields() {
        let surface = WaterSurface {
            centre: [1.0, 2.0, 3.0],
            deep_colour: [0.02, 0.05, 0.12],
            shallow_colour: [0.1, 0.3, 0.4],
            depth_falloff_metres: 3.0,
            foam_width_metres: 0.2,
            foam_intensity: 0.5,
            fresnel_power: 4.0,
            roughness: 0.08,
            refraction_strength: 0.05,
            waves: vec![WaterWave::default(), WaterWave::default()],
            ..Default::default()
        };
        let p = water_params_from(&surface, true);
        assert_eq!(p.centre, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(p.deep_colour, [0.02, 0.05, 0.12, 0.0]);
        assert_eq!(p.shallow_colour, [0.1, 0.3, 0.4, 0.0]);
        assert_eq!(p.depth_falloff, 3.0);
        assert_eq!(p.foam_width, 0.2);
        assert_eq!(p.foam_intensity, 0.5);
        assert_eq!(p.fresnel_power, 4.0);
        assert_eq!(p.roughness, 0.08);
        assert_eq!(p.refraction_strength, 0.05);
        assert_eq!(p.wave_count, 2);
        assert_eq!(p.planar, WaterParams::planar_lane(0.08, true));
        assert!(p.planar[0] > 0.5 && p.planar[1] > 0.0);
        // A slotless surface keeps the probe / sky path.
        assert_eq!(water_params_from(&surface, false).planar, [0.0; 4]);
    }

    // More authored waves than the shader's array can hold must clamp rather
    // than overflow the fixed lane count.
    #[test]
    fn water_params_clamps_the_wave_count() {
        let surface = WaterSurface {
            waves: vec![WaterWave::default(); MAX_WATER_WAVES + 3],
            ..Default::default()
        };
        assert_eq!(
            water_params_from(&surface, false).wave_count,
            MAX_WATER_WAVES as u32
        );
    }

    // Compile the water vertex + fragment shaders (both MSAA variants) so a
    // regression fails the suite without a GPU.
    #[test]
    fn water_shaders_compile() {
        if !crate::slangc_gate::slangc_available() {
            return;
        }
        // Both the ceiling and a device-shortened probe cube array must compile.
        for probes in [1, concinnity_core::render::uniforms::MAX_PROBES as u32] {
            super::compile_water_shaders(false, true, probes).expect("water compiles (msaa)");
            super::compile_water_shaders(false, false, probes).expect("water compiles (no msaa)");
        }
    }

    // Compile the ray-traced water shaders (both MSAA variants, both flat +
    // textured) so a regression in water.slang's `WATER_RT` arm (the shared
    // `{RT_TRACE}` traversal + the probe `{PROBE_COMMON}` injection + the
    // `RT_TEXTURED` split) fails the suite without a GPU.
    #[test]
    fn water_rt_shaders_compile() {
        if !crate::slangc_gate::slangc_available() {
            return;
        }
        for &msaa in &[true, false] {
            let shaders = super::compile_water_rt_shaders(false, msaa, 4, 4)
                .expect("water rt shaders compile");
            assert!(crate::vulkan::pipeline::is_spirv(&shaders.vs));
            assert!(crate::vulkan::pipeline::is_spirv(&shaders.flat_fs));
            assert!(
                shaders.textured_fs.is_some(),
                "pool_size>0 builds the textured variant"
            );
        }
        // pool_size 0 builds only the flat variant.
        let flat_only =
            super::compile_water_rt_shaders(false, false, 0, 4).expect("water rt flat compiles");
        assert!(flat_only.textured_fs.is_none());
    }
}
