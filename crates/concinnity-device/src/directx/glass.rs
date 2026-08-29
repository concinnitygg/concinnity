// src/directx/glass.rs
//
// The glass producers of the engine's transparent pass on the D3D12 backend
// (`transparent.rs` owns the pass itself, the scene snapshot, the shared root
// signatures and the combined back-to-front draw order; `water.rs` is the third
// producer). Two live here because they are the same material family:
//
//   * `GlassPanel` -- a flat world-space quad built once at init. The fragment
//     refracts the pass's scene snapshot, tints it, and mixes a reflection over
//     it by a Schlick Fresnel term.
//   * A see-through glass MESH -- an imported `Material` flagged `see_through`,
//     drawn from the shared scene buffers with a per-pixel reflection ray. It is
//     ray-traced only, so it builds nothing without DXR and its meshes then
//     rasterise opaque in the main pass instead.
//
// The shaders are the shared `shaders/glass.slang` and `shaders/glass_mesh.slang`,
// compiled through `slang_builtins`; the ray-traced fragments need shader model
// 6.5 for their inline ray query, the base pair 6.0.

use windows::Win32::Graphics::Direct3D12::*;

use super::allocator::DeviceAllocator;
use crate::components::GlassPanel;
use crate::directx::context::dump_on_err;
use crate::directx::slang_builtins;
use crate::directx::slang_builtins::SlangCompile;
use crate::directx::transparent::{
    GlassMeshProducer, RecordUpload, TransparentProducer, TransparentRecord, create_transparent_pso,
};
use crate::geometry::glass_quad::build_glass_quad;
use crate::gfx::mesh_payload::Vertex;

// `GlassParams` (the per-panel cbuffer) is a GPU-free layout struct that lives
// in `core::render`; re-export it so `crate::directx::glass::GlassParams` is
// unchanged for the `glass_params_from` path.
pub(in crate::directx) use concinnity_core::render::uniforms::GlassParams;

// Build the per-panel `GlassParams` from an authored panel. Pure; unit
// tested. Mirrors `metal::glass::glass_params_from`.
fn glass_params_from(panel: &GlassPanel, planar: f32) -> GlassParams {
    let n = panel.normal; // already unit-length from GlassPanel::from_args
    GlassParams {
        centre: [panel.centre[0], panel.centre[1], panel.centre[2], 0.0],
        normal: [n[0], n[1], n[2], 0.0],
        tint: [panel.tint[0], panel.tint[1], panel.tint[2], 0.0],
        opacity: panel.opacity,
        refraction_strength: panel.refraction_strength,
        fresnel_power: panel.fresnel_power,
        planar,
    }
}

// Compile the glass vertex + fragment shaders. The fragment comes in an MSAA
// pair, which keeps its depth SRV declaration in sync with the resource's
// sample count; the vertex reads no depth and serves both pipelines. Used at
// init and by shader hot-reload.
pub(in crate::directx) fn compile_glass_shaders(
    msaa_samples: u32,
    hot_reload: bool,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let frag = if msaa_samples > 1 {
        &slang_builtins::GLASS_FRAG_MSAA
    } else {
        &slang_builtins::GLASS_FRAG
    };
    let vs = slang_builtins::GLASS_VERT.compile(hot_reload)?;
    let ps = frag.compile(hot_reload)?;
    Ok((vs, ps))
}

// Rebuild the glass PSO against fresh shader source. Called from the DirectX
// shader hot-reload pass; the root signature is reused.
pub(in crate::directx) fn rebuild_glass_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    msaa_samples: u32,
    hot_reload: bool,
    info_queue: Option<&ID3D12InfoQueue>,
) -> Result<ID3D12PipelineState, String> {
    let (vs, ps) = compile_glass_shaders(msaa_samples, hot_reload)?;
    dump_on_err(
        info_queue,
        create_transparent_pso(device, root_sig, &vs, &ps),
    )
}

// DXIL for the two RT glass fragments, plus the vertex stage they share with the
// base pass (both root signatures put the transparent view CBV at b0).
struct GlassRtShaders {
    vs: Vec<u8>,
    flat_ps: Vec<u8>,
    textured_ps: Vec<u8>,
}

// Compile the flat + textured ray-traced fragments (SM 6.5, for the inline ray
// query). The shared source remaps its probe cube array to t20, since the RT
// geometry SRVs claim t4..t10. Returns an `Err` (which the caller turns into a
// None RT pipeline + the base path) when slangc is unavailable or the shader
// fails to compile.
fn compile_glass_rt_shaders(msaa_samples: u32, hot_reload: bool) -> Result<GlassRtShaders, String> {
    let msaa = msaa_samples > 1;
    let flat = if msaa {
        &slang_builtins::GLASS_RT_FRAG_MSAA
    } else {
        &slang_builtins::GLASS_RT_FRAG
    };
    let textured = if msaa {
        &slang_builtins::GLASS_RT_FRAG_TEXTURED_MSAA
    } else {
        &slang_builtins::GLASS_RT_FRAG_TEXTURED
    };
    Ok(GlassRtShaders {
        vs: slang_builtins::GLASS_VERT.compile(hot_reload)?,
        flat_ps: flat.compile(hot_reload)?,
        textured_ps: textured.compile(hot_reload)?,
    })
}

// What building the glass producer needs from the pass that owns it: the
// allocator, the two shared root signatures (the RT one is `None` on a
// non-DXR GPU), and the render-state / hot-reload toggles.
#[derive(Clone, Copy)]
pub(in crate::directx) struct GlassBuild<'a> {
    pub alloc: &'a DeviceAllocator,
    pub root_sig: &'a ID3D12RootSignature,
    pub rt_root_sig: Option<&'a ID3D12RootSignature>,
    pub msaa_samples: u32,
    pub hot_reload: bool,
    pub info_queue: Option<&'a ID3D12InfoQueue>,
}

// Build the glass pipelines and one record per authored panel. The RT pair is
// built whenever the pass has an RT root signature (regardless of whether RT is
// on at launch, so a live `quality-set ray_traced_reflections` selects it with no
// pipeline rebuild); a compile failure leaves it absent and the base
// probe/planar path runs.
pub(in crate::directx) fn build_glass_producer(
    build: GlassBuild,
    panels: &[GlassPanel],
    // Per-pane planar resolve slot (aligned with `panels`); `None` panes keep the
    // probe/sky reflection. From `assign_planar_slots`.
    planar_slots: &[Option<usize>],
) -> Result<TransparentProducer, String> {
    let GlassBuild {
        alloc,
        root_sig,
        rt_root_sig,
        msaa_samples,
        hot_reload,
        info_queue,
    } = build;
    let device = alloc.device();
    let (vs, ps) = compile_glass_shaders(msaa_samples, hot_reload)?;
    let pso = dump_on_err(
        info_queue,
        create_transparent_pso(device, root_sig, &vs, &ps),
    )?;

    let (flat_rt_pso, textured_rt_pso) = match rt_root_sig {
        Some(sig) => {
            match build_glass_rt_pipelines(device, sig, msaa_samples, hot_reload, info_queue) {
                Ok(pair) => (Some(pair.0), Some(pair.1)),
                Err(e) => {
                    tracing::warn!(
                        "glass RT reflection pipeline build failed ({e}); \
                     using the probe/planar glass path"
                    );
                    (None, None)
                }
            }
        }
        None => (None, None),
    };

    let mut records = Vec::with_capacity(panels.len());
    for (i, panel) in panels.iter().enumerate() {
        let planar_slot = planar_slots.get(i).copied().flatten();
        let (verts, idxs) = build_glass_quad(panel.centre, panel.normal, panel.half_size);

        // Flatten into the standard Vertex layout. Tangent is a placeholder (the
        // glass shader rebuilds its frame from the panel normal) and per-vertex
        // colour is unused.
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

        // Bake the planar flag: the pane samples the sharp mirror render only
        // when it was assigned a planar slot.
        let params = glass_params_from(panel, if planar_slot.is_some() { 1.0 } else { 0.0 });
        records.push(TransparentRecord::upload(
            alloc,
            RecordUpload {
                vertices: &packed,
                indices: &idxs,
                params: bytemuck::bytes_of(&params),
                visible: panel.visible,
                centre: panel.centre,
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

// Compile and build the flat + textured RT glass PSOs against the pass's RT root
// signature. Both use the same render state as the base PSO.
fn build_glass_rt_pipelines(
    device: &ID3D12Device,
    rt_root_sig: &ID3D12RootSignature,
    msaa_samples: u32,
    hot_reload: bool,
    info_queue: Option<&ID3D12InfoQueue>,
) -> Result<(ID3D12PipelineState, ID3D12PipelineState), String> {
    let shaders = compile_glass_rt_shaders(msaa_samples, hot_reload)?;
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

// What building the see-through mesh producer needs. There is no base root
// signature here: the producer is ray-traced only, so it is built solely against
// the pass's RT signature.
#[derive(Clone, Copy)]
pub(in crate::directx) struct GlassMeshBuild<'a> {
    pub alloc: &'a DeviceAllocator,
    pub rt_root_sig: &'a ID3D12RootSignature,
    pub msaa_samples: u32,
    pub hot_reload: bool,
    pub info_queue: Option<&'a ID3D12InfoQueue>,
}

// Compile the flat + textured see-through mesh fragments (SM 6.5, for the inline
// ray query) and the vertex stage they share. Unlike the pane family there is no
// non-RT pair: the trace is what makes the mesh see-through.
fn compile_glass_mesh_shaders(
    msaa_samples: u32,
    hot_reload: bool,
) -> Result<GlassRtShaders, String> {
    let msaa = msaa_samples > 1;
    let flat = if msaa {
        &slang_builtins::GLASS_MESH_RT_FRAG_MSAA
    } else {
        &slang_builtins::GLASS_MESH_RT_FRAG
    };
    let textured = if msaa {
        &slang_builtins::GLASS_MESH_RT_FRAG_TEXTURED_MSAA
    } else {
        &slang_builtins::GLASS_MESH_RT_FRAG_TEXTURED
    };
    Ok(GlassRtShaders {
        vs: slang_builtins::GLASS_MESH_VERT.compile(hot_reload)?,
        flat_ps: flat.compile(hot_reload)?,
        textured_ps: textured.compile(hot_reload)?,
    })
}

// Build the see-through mesh pipelines and the per-frame params ring one block
// per mesh deep. The producer holds no records: a mesh's geometry lives in the
// shared scene buffers and its params change per frame, so the encoder rebuilds
// its draw list each frame (see `collect_mesh_draws`).
pub(in crate::directx) fn build_glass_mesh_producer(
    build: GlassMeshBuild,
    object_indices: &[usize],
) -> Result<GlassMeshProducer, String> {
    let GlassMeshBuild {
        alloc,
        rt_root_sig,
        msaa_samples,
        hot_reload,
        info_queue,
    } = build;
    let device = alloc.device();
    let shaders = compile_glass_mesh_shaders(msaa_samples, hot_reload)?;
    let flat_rt_pso = dump_on_err(
        info_queue,
        create_transparent_pso(device, rt_root_sig, &shaders.vs, &shaders.flat_ps),
    )?;
    let textured_rt_pso = dump_on_err(
        info_queue,
        create_transparent_pso(device, rt_root_sig, &shaders.vs, &shaders.textured_ps),
    )?;
    GlassMeshProducer::new(
        alloc,
        flat_rt_pso,
        Some(textured_rt_pso),
        object_indices.to_vec(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // The `TransparentView` / `GlassParams` layout tests live with the structs
    // in `concinnity_core::render::uniforms`, and are checked against the compiled
    // shader by `shader_layout`.

    // The glass shaders compile at runtime from the shared single source, so a
    // syntax or register error in either MSAA variant would otherwise surface
    // only as an init failure on a GPU host. slangc resolves from PATH and may
    // be absent, in which case the runtime path reports its own error and this
    // skips rather than failing.
    #[test]
    fn glass_shaders_compile() {
        if !crate::slangc_gate::slangc_available() {
            return;
        }
        for msaa in [1u32, 4] {
            super::compile_glass_shaders(msaa, false)
                .unwrap_or_else(|e| panic!("glass shaders (msaa={msaa}) must compile: {e}"));
        }
    }

    // The same for the ray-traced pair, which additionally exercises the shared
    // traversal fragment and the shader model 6.5 the ray query needs. Both MSAA
    // variants and both hit-shading variants go through
    // `compile_glass_rt_shaders`.
    #[test]
    fn glass_rt_shaders_compile() {
        if !crate::slangc_gate::slangc_available() {
            return;
        }
        for msaa in [1u32, 4] {
            super::compile_glass_rt_shaders(msaa, false)
                .unwrap_or_else(|e| panic!("glass_rt shaders (msaa={msaa}) must compile: {e}"));
        }
    }

    // And the see-through mesh family, whose vertex stage is a second one (it
    // applies the model matrix) and whose fragments carry the same SM 6.5 trace.
    #[test]
    fn glass_mesh_shaders_compile() {
        if !crate::slangc_gate::slangc_available() {
            return;
        }
        for msaa in [1u32, 4] {
            super::compile_glass_mesh_shaders(msaa, false)
                .unwrap_or_else(|e| panic!("glass_mesh shaders (msaa={msaa}) must compile: {e}"));
        }
    }

    #[test]
    fn glass_params_from_maps_fields() {
        let panel = GlassPanel {
            centre: [1.0, 2.0, 3.0],
            normal: [0.0, 0.0, 1.0],
            tint: [0.6, 0.85, 0.9],
            opacity: 0.45,
            refraction_strength: 0.04,
            fresnel_power: 4.0,
            ..Default::default()
        };
        let p = glass_params_from(&panel, 1.0);
        assert_eq!(p.centre, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(p.normal, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(p.tint, [0.6, 0.85, 0.9, 0.0]);
        assert_eq!(p.opacity, 0.45);
        assert_eq!(p.refraction_strength, 0.04);
        assert_eq!(p.fresnel_power, 4.0);
        assert_eq!(p.planar, 1.0);
        // A slotless pane gets planar = 0.0 (probe/sky fallback path).
        assert_eq!(glass_params_from(&panel, 0.0).planar, 0.0);
    }
}
