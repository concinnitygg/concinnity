//! WaterSurface: one producer of the engine's transparent pass on the D3D12
//! backend (`transparent.rs` owns the pass itself, the scene snapshot, the shared
//! root signatures and the combined back-to-front draw order; `glass.rs` is the
//! other producer). Each surface is a flat tessellated XZ grid built once at init
//! and displaced per frame by the vertex stage's Gerstner sum; the fragment
//! refracts the pass's scene snapshot, tints and foams it by the water-column
//! thickness the main depth gives, and mixes a reflection over it by a Schlick
//! Fresnel term.
//!
//! The shaders are the shared `shaders/water.hlsl`, compiled through
//! `builtin_shaders`; the ray-traced fragment needs shader model 6.5 for its
//! inline ray query, the base pair 6.0.

use concinnity_core::components::WaterSurface;
use concinnity_core::geometry::water_grid::build_water_grid;
use concinnity_core::gfx::mesh_payload::Vertex;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::uniforms::WaterParams;
use windows::Win32::Graphics::Direct3D12::*;

use super::allocator::DeviceAllocator;
use crate::directx::builtin_shaders;
use crate::directx::builtin_shaders::CompileProgram;
use crate::directx::context::dump_on_err;
use crate::directx::transparent::{
    RecordUpload, TransparentProducer, TransparentRecord, create_transparent_pso,
};

// Compile the water vertex + fragment shaders. The fragment comes in an MSAA
// pair, which keeps its depth SRV declaration in sync with the resource's sample
// count; the vertex reads no depth and serves both pipelines. Used at init and
// by shader hot-reload.
pub(in crate::directx) fn compile_water_shaders(
    msaa_samples: u32,
    hot_reload: bool,
) -> RenderResult<(Vec<u8>, Vec<u8>)> {
    let vs = builtin_shaders::WATER_VERT.compile(hot_reload)?;
    let ps = builtin_shaders::WATER_FRAG
        .at(msaa_samples > 1)
        .compile(hot_reload)?;
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
) -> RenderResult<ID3D12PipelineState> {
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
// query). Returns an `Err` (which the caller turns into a None RT pipeline +
// the base path) when dxc is unavailable or the shader fails to compile.
fn compile_water_rt_shaders(msaa_samples: u32, hot_reload: bool) -> RenderResult<WaterRtShaders> {
    let msaa = msaa_samples > 1;
    Ok(WaterRtShaders {
        vs: builtin_shaders::WATER_VERT.compile(hot_reload)?,
        flat_ps: builtin_shaders::WATER_FRAG_RT
            .at(msaa)
            .compile(hot_reload)?,
        textured_ps: builtin_shaders::WATER_FRAG_RT_TEXTURED
            .at(msaa)
            .compile(hot_reload)?,
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
) -> RenderResult<TransparentProducer> {
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
            build_water_grid(surface.extent[0], surface.extent[1], surface.subdivisions)
                .map_err(RenderError::Other)?;

        // Flatten into the standard Vertex layout. Tangent and color are
        // placeholders: the water shader rebuilds its normal frame analytically
        // from the wave derivatives and the fragment ignores per-vertex color.
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

        let params = WaterParams::from_surface(surface, planar_slot.is_some());
        records.push(TransparentRecord::upload(
            alloc,
            RecordUpload {
                vertices: &packed,
                indices: &idxs,
                params: bytemuck::bytes_of(&params),
                visible: surface.visible,
                center: surface.center,
                planar_slot,
            },
        )?);
    }

    Ok(TransparentProducer {
        pso,
        flat_rt_pso,
        textured_rt_pso,
        reflection_flat_pso: None,
        reflection_textured_pso: None,
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
) -> RenderResult<(ID3D12PipelineState, ID3D12PipelineState)> {
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
    // The `WaterParams` / `WaterWaveGpu` layout tests live with the structs in
    // `concinnity_core::render::uniforms`, and are checked against the compiled shader
    // by `shader_layout`.

    // The water shaders compile at runtime from the shared single source, so a
    // syntax or register error in either MSAA variant would otherwise surface
    // only as an init failure on a GPU host.
    #[test]
    fn water_shaders_compile() {
        concinnity_shader::require_dxc!();
        for msaa in [1u32, 4] {
            super::compile_water_shaders(msaa, false)
                .unwrap_or_else(|e| panic!("water shaders (msaa={msaa}) must compile: {e}"));
        }
    }

    // The same for the ray-traced pair, which additionally exercises the shared
    // traversal fragment and the shader model 6.5 the ray query needs.
    #[test]
    fn water_rt_shaders_compile() {
        concinnity_shader::require_dxc!();
        for msaa in [1u32, 4] {
            super::compile_water_rt_shaders(msaa, false)
                .unwrap_or_else(|e| panic!("water_rt shaders (msaa={msaa}) must compile: {e}"));
        }
    }
}
