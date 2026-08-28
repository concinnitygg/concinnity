// src/directx/water.rs
//
// WaterSurface: one producer of the engine's transparent pass on the D3D12
// backend (`transparent.rs` owns the pass itself, the scene snapshot, the shared
// root signatures and the combined back-to-front draw order; `glass.rs` is the
// other producer). Each surface is a flat tessellated XZ grid built once at init
// and displaced per frame by the vertex stage's Gerstner sum; the fragment
// refracts the pass's scene snapshot, tints and foams it by the water-column
// thickness the main depth gives, and mixes a reflection over it by a Schlick
// Fresnel term.
//
// The shaders are the shared `shaders/water.slang`, compiled through
// `slang_builtins`; the ray-traced fragment needs shader model 6.5 for its
// inline ray query, the base pair 6.0.

use windows::Win32::Graphics::Direct3D12::*;

use super::allocator::DeviceAllocator;
use crate::components::{MAX_WATER_WAVES, WaterSurface, WaterWave};
use crate::directx::context::dump_on_err;
use crate::directx::slang_builtins;
use crate::directx::slang_builtins::SlangCompile;
use crate::directx::transparent::{
    RecordUpload, TransparentProducer, TransparentRecord, create_transparent_pso,
};
use crate::geometry::water_grid::build_water_grid;
use crate::gfx::mesh_payload::Vertex;

// `WaterParams` / `WaterWaveGpu` (the per-surface cbuffer and its wave lanes)
// are GPU-free layout structs that live in concinnity-render; re-export them so
// `crate::directx::water::WaterParams` is unchanged for the
// `water_params_from` path.
pub(in crate::directx) use concinnity_render::uniforms::{
    WATER_MAX_WAVES, WaterParams, WaterWaveGpu,
};

// Wave-normal screen-space distortion scale for the planar reflection sample.
// Small: the planar reflection is a flat-plane render, so the wave normal only
// perturbs the lookup a little to fake ripple displacement. Mirrors
// `metal::water::PLANAR_DISTORTION`.
const PLANAR_DISTORTION: f32 = 0.03;

// The shader-side wave lane for one authored wave. Pure; unit tested.
fn wave_to_gpu(w: &WaterWave) -> WaterWaveGpu {
    WaterWaveGpu {
        dir_amp_wave: [w.direction[0], w.direction[1], w.amplitude, w.wavelength],
        speed_steep_pad: [w.speed, w.steepness, 0.0, 0.0],
    }
}

// Build the per-surface `WaterParams` from an authored surface. `planar` is
// `[1.0, PLANAR_DISTORTION, 0, 0]` when the surface has a planar reflection slot
// and zeroed otherwise, which is what selects the sharp mirror render over the
// probe / sky cube. Pure; unit tested. Mirrors `metal::water::water_params_from`.
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
        planar: if planar {
            [1.0, PLANAR_DISTORTION, 0.0, 0.0]
        } else {
            [0.0; 4]
        },
    }
}

// Compile the water vertex + fragment shaders. The fragment comes in an MSAA
// pair, which keeps its depth SRV declaration in sync with the resource's sample
// count; the vertex reads no depth and serves both pipelines. Used at init and
// by shader hot-reload.
pub(in crate::directx) fn compile_water_shaders(
    msaa_samples: u32,
    hot_reload: bool,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let frag = if msaa_samples > 1 {
        &slang_builtins::WATER_FRAG_MSAA
    } else {
        &slang_builtins::WATER_FRAG
    };
    let vs = slang_builtins::WATER_VERT.compile(hot_reload)?;
    let ps = frag.compile(hot_reload)?;
    Ok((vs, ps))
}

// Rebuild the water PSO against fresh shader source. Called from the DirectX
// shader hot-reload pass; the root signature is reused.
pub(in crate::directx) fn rebuild_water_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    msaa_samples: u32,
    hot_reload: bool,
    info_queue: Option<&ID3D12InfoQueue>,
) -> Result<ID3D12PipelineState, String> {
    let (vs, ps) = compile_water_shaders(msaa_samples, hot_reload)?;
    dump_on_err(
        info_queue,
        create_transparent_pso(device, root_sig, &vs, &ps),
    )
}

// DXIL for the two RT water fragments, plus the vertex stage they share with the
// base pass (both root signatures put the transparent view CBV at b0 and the
// per-record params at b1).
struct WaterRtShaders {
    vs: Vec<u8>,
    flat_ps: Vec<u8>,
    textured_ps: Vec<u8>,
}

// Compile the flat + textured ray-traced fragments (SM 6.5, for the inline ray
// query). The shared source remaps its probe cube array to t20, since the RT
// geometry SRVs claim t4..t10. Returns an `Err` (which the caller turns into a
// None RT pipeline + the base path) when slangc is unavailable or the shader
// fails to compile.
fn compile_water_rt_shaders(msaa_samples: u32, hot_reload: bool) -> Result<WaterRtShaders, String> {
    let msaa = msaa_samples > 1;
    let flat = if msaa {
        &slang_builtins::WATER_RT_FRAG_MSAA
    } else {
        &slang_builtins::WATER_RT_FRAG
    };
    let textured = if msaa {
        &slang_builtins::WATER_RT_FRAG_TEXTURED_MSAA
    } else {
        &slang_builtins::WATER_RT_FRAG_TEXTURED
    };
    Ok(WaterRtShaders {
        vs: slang_builtins::WATER_VERT.compile(hot_reload)?,
        flat_ps: flat.compile(hot_reload)?,
        textured_ps: textured.compile(hot_reload)?,
    })
}

// What building the water producer needs from the pass that owns it: the
// allocator, the two shared root signatures (the RT one is `None` on a non-DXR
// GPU), and the render-state / hot-reload toggles.
#[derive(Clone, Copy)]
pub(in crate::directx) struct WaterBuild<'a> {
    pub alloc: &'a DeviceAllocator,
    pub root_sig: &'a ID3D12RootSignature,
    pub rt_root_sig: Option<&'a ID3D12RootSignature>,
    pub msaa_samples: u32,
    pub hot_reload: bool,
    pub info_queue: Option<&'a ID3D12InfoQueue>,
}

// Build the water pipelines and one record per authored surface. The RT pair is
// built whenever the pass has an RT root signature (regardless of whether RT is
// on at launch, so a live `quality-set ray_traced_reflections` selects it with no
// pipeline rebuild); a compile failure leaves it absent and the base
// probe/planar path runs.
pub(in crate::directx) fn build_water_producer(
    build: WaterBuild,
    surfaces: &[WaterSurface],
    // Per-surface planar resolve slot (aligned with `surfaces`); `None` surfaces
    // keep the probe/sky reflection. From `assign_planar_slots`.
    planar_slots: &[Option<usize>],
) -> Result<TransparentProducer, String> {
    let WaterBuild {
        alloc,
        root_sig,
        rt_root_sig,
        msaa_samples,
        hot_reload,
        info_queue,
    } = build;
    let device = alloc.device();
    let (vs, ps) = compile_water_shaders(msaa_samples, hot_reload)?;
    let pso = dump_on_err(
        info_queue,
        create_transparent_pso(device, root_sig, &vs, &ps),
    )?;

    let (flat_rt_pso, textured_rt_pso) = match rt_root_sig {
        Some(sig) => {
            match build_water_rt_pipelines(device, sig, msaa_samples, hot_reload, info_queue) {
                Ok(pair) => (Some(pair.0), Some(pair.1)),
                Err(e) => {
                    tracing::warn!(
                        "water RT reflection pipeline build failed ({e}); \
                         using the probe/planar water path"
                    );
                    (None, None)
                }
            }
        }
        None => (None, None),
    };

    let mut records = Vec::with_capacity(surfaces.len());
    for (i, surface) in surfaces.iter().enumerate() {
        let planar_slot = planar_slots.get(i).copied().flatten();
        let (verts, idxs) =
            build_water_grid(surface.extent[0], surface.extent[1], surface.subdivisions)?;

        // Flatten into the standard Vertex layout. Tangent and colour are
        // placeholders: the water shader rebuilds its normal frame analytically
        // from the wave derivatives and the fragment ignores per-vertex colour.
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
        records.push(TransparentRecord::upload(
            alloc,
            RecordUpload {
                vertices: &packed,
                indices: &idxs,
                params: bytemuck::bytes_of(&params),
                visible: surface.visible,
                centre: surface.centre,
                planar_slot,
            },
        )?);
    }

    Ok(TransparentProducer {
        pso,
        flat_rt_pso,
        textured_rt_pso,
        records,
    })
}

// Compile and build the flat + textured RT water PSOs against the pass's RT root
// signature. Both use the same render state as the base PSO.
fn build_water_rt_pipelines(
    device: &ID3D12Device,
    rt_root_sig: &ID3D12RootSignature,
    msaa_samples: u32,
    hot_reload: bool,
    info_queue: Option<&ID3D12InfoQueue>,
) -> Result<(ID3D12PipelineState, ID3D12PipelineState), String> {
    let shaders = compile_water_rt_shaders(msaa_samples, hot_reload)?;
    let flat = dump_on_err(
        info_queue,
        create_transparent_pso(device, rt_root_sig, &shaders.vs, &shaders.flat_ps),
    )?;
    let textured = dump_on_err(
        info_queue,
        create_transparent_pso(device, rt_root_sig, &shaders.vs, &shaders.textured_ps),
    )?;
    Ok((flat, textured))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The `WaterParams` / `WaterWaveGpu` layout tests live with the structs in
    // `concinnity_render::uniforms`, and are checked against the compiled shader
    // by `shader_layout`.

    // The water shaders compile at runtime from the shared single source, so a
    // syntax or register error in either MSAA variant would otherwise surface
    // only as an init failure on a GPU host.
    #[test]
    fn water_shaders_compile() {
        if !crate::slangc_gate::slangc_available() {
            return;
        }
        for msaa in [1u32, 4] {
            super::compile_water_shaders(msaa, false)
                .unwrap_or_else(|e| panic!("water shaders (msaa={msaa}) must compile: {e}"));
        }
    }

    // The same for the ray-traced pair, which additionally exercises the shared
    // traversal fragment and the shader model 6.5 the ray query needs.
    #[test]
    fn water_rt_shaders_compile() {
        if !crate::slangc_gate::slangc_available() {
            return;
        }
        for msaa in [1u32, 4] {
            super::compile_water_rt_shaders(msaa, false)
                .unwrap_or_else(|e| panic!("water_rt shaders (msaa={msaa}) must compile: {e}"));
        }
    }

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
        assert_eq!(p.planar, [1.0, PLANAR_DISTORTION, 0.0, 0.0]);
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
}
