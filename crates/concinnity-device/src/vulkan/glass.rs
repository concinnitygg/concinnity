//! GlassPanel: one producer of the engine's transparent pass on the Vulkan
//! backend (`transparent.rs` owns the render pass itself, the scene snapshot, the
//! shared descriptor / pipeline layouts and the combined back-to-front draw
//! order; `water.rs` is the other producer). Each panel is a flat world-space
//! quad, built once at init; the fragment shader refracts the pass's scene
//! snapshot, tints it, and mixes a reflection over it by a Schlick Fresnel term
//! (see shaders/glass.hlsl, the single source all three backends compile).
//!
//! Same uniform layouts, back-to-front ordering and manual depth-occlusion test
//! as the DirectX and Metal hosts.

use ash::vk;
use concinnity_core::components::GlassPanel;
use concinnity_core::geometry::glass_quad::build_glass_quad;
use concinnity_core::gfx::mesh_payload::Vertex;
use concinnity_core::render::error::RenderResult;
// `GlassParams` (the per-panel UBO) is a GPU-free layout struct that lives in
// `core::render`; re-export it so `crate::vulkan::glass::GlassParams` is
// unchanged for the `glass_params_from` path.
pub(in crate::vulkan) use concinnity_core::render::uniforms::GlassParams;

use super::allocator::DeviceAllocator;
use crate::vulkan::builtin_shaders::CompileProgram;
use crate::vulkan::transparent::{
    GlassMeshProducer, ProducerCtx, RecordUpload, TracedGlassPipelines, TransparentProducer,
    TransparentRecord, TransparentVertexInput, create_glass_reflection_pipeline,
    create_transparent_pipeline,
};

// Build the per-panel `GlassParams` from an authored panel. `planar` is 1.0 when
// the pane has a planar reflection slot, else 0.0. Pure; unit tested. Mirrors
// `directx::glass::glass_params_from`.
fn glass_params_from(panel: &GlassPanel, planar: f32) -> GlassParams {
    let n = panel.normal; // already unit-length from GlassPanel::from_args
    GlassParams {
        center: [panel.center[0], panel.center[1], panel.center[2], 0.0],
        normal: [n[0], n[1], n[2], 0.0],
        tint: [panel.tint[0], panel.tint[1], panel.tint[2], 0.0],
        opacity: panel.opacity,
        refraction_strength: panel.refraction_strength,
        fresnel_power: panel.fresnel_power,
        planar,
    }
}

// Compile the glass vertex + fragment shaders, the fragment at the main
// depth's sample count so its depth read matches the resource.
fn compile_glass_shaders(hot_reload: bool, msaa: bool) -> RenderResult<(Vec<u8>, Vec<u8>)> {
    let vert = super::builtin_shaders::GLASS_VERT.compile(hot_reload)?;
    let frag = super::builtin_shaders::GLASS_FRAG
        .at(msaa)
        .compile(hot_reload)?;
    Ok((vert, frag))
}

// SPIR-V blobs for the ray-traced glass pipelines: the shared vertex stage (the
// same one the base pass uses -- the trace is entirely in the fragment), the
// flat fragment, and the textured fragment (`None` when the bindless pool is
// absent). Mirrors `post::rt_reflections::RtShaders`.
struct GlassRtShaders {
    vs: Vec<u8>,
    flat_fs: Vec<u8>,
    textured_fs: Option<Vec<u8>>,
    // The reflection pre-pass fragments, paired the same way.
    reflection_flat_fs: Vec<u8>,
    reflection_textured_fs: Option<Vec<u8>>,
}

// Compile the glass vertex shader + the ray-traced glass fragment (flat, plus
// the textured variant when `pool_size > 0`). dxc emits `SPV_KHR_ray_query`
// for the traversal, which the device already advertises wherever these
// pipelines are built.
fn compile_glass_rt_shaders(
    hot_reload: bool,
    msaa: bool,
    pool_size: usize,
) -> RenderResult<GlassRtShaders> {
    let vs = super::builtin_shaders::GLASS_VERT.compile(hot_reload)?;
    let flat_fs = super::builtin_shaders::GLASS_FRAG_RT
        .at(msaa)
        .compile(hot_reload)?;
    let reflection_flat_fs = super::builtin_shaders::GLASS_REFLECTION_FRAG
        .at(msaa)
        .compile(hot_reload)?;
    let (textured_fs, reflection_textured_fs) = if pool_size > 0 {
        (
            Some(
                super::builtin_shaders::GLASS_FRAG_RT_TEXTURED
                    .at(msaa)
                    .compile(hot_reload)?,
            ),
            Some(
                super::builtin_shaders::GLASS_REFLECTION_FRAG_TEXTURED
                    .at(msaa)
                    .compile(hot_reload)?,
            ),
        )
    } else {
        (None, None)
    };
    Ok(GlassRtShaders {
        vs,
        flat_fs,
        textured_fs,
        reflection_flat_fs,
        reflection_textured_fs,
    })
}

// Upload one panel's static quad VB + IB and its per-panel `GlassParams` UBO,
// then allocate + write the panel's descriptor set.
fn build_panel_record(
    alloc: &DeviceAllocator,
    ctx: &ProducerCtx,
    panel: &GlassPanel,
    planar_slot: Option<usize>,
) -> RenderResult<TransparentRecord> {
    let (verts, idxs) = build_glass_quad(panel.center, panel.normal, panel.half_size);

    // Flatten into the standard engine `Vertex` layout. Tangent is a placeholder
    // (the glass shader rebuilds its frame from the panel normal) and per-vertex
    // color is unused.
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

    let params = glass_params_from(panel, if planar_slot.is_some() { 1.0 } else { 0.0 });
    TransparentRecord::upload(
        alloc,
        ctx.record_descriptors(planar_slot),
        RecordUpload {
            vertices: &packed,
            indices: &idxs,
            params: bytemuck::bytes_of(&params),
            visible: panel.visible,
            center: panel.center,
            planar_slot,
        },
    )
}

// Build the glass pipelines and one record per authored panel. The RT pair is
// built whenever the pass has RT pipeline layouts (regardless of whether RT is on
// at launch, so a live `quality-set ray_traced_reflections` selects it with no
// pipeline rebuild); a compile failure leaves it absent and the base
// probe/planar path runs.
pub(in crate::vulkan) fn build_glass_producer(
    ctx: ProducerCtx,
    panels: &[GlassPanel],
    // Per-pane planar resolve slot (aligned with `panels`); `None` panes keep the
    // probe/sky reflection. From `assign_planar_slots`.
    planar_slots: &[Option<usize>],
) -> RenderResult<TransparentProducer> {
    let (vert_spv, frag_spv) = compile_glass_shaders(ctx.hot_reload, ctx.msaa)?;
    let pipeline = create_transparent_pipeline(
        ctx.device,
        ctx.render_pass,
        ctx.layout,
        &vert_spv,
        &frag_spv,
        TransparentVertexInput::Position,
    )?;

    let traced = match ctx.rt_layout_flat {
        Some(flat_layout) => match build_glass_rt_pipelines(&ctx, flat_layout) {
            Ok(p) => Some(p),
            Err(e) => {
                tracing::warn!(
                    "glass RT pipelines failed to build ({e}); using the probe / planar glass path"
                );
                None
            }
        },
        None => None,
    };
    let (flat_rt_pso, textured_rt_pso, reflection_flat_pso, reflection_textured_pso) = match traced
    {
        Some(p) => (
            Some(p.shade_flat),
            p.shade_textured,
            Some(p.reflection_flat),
            p.reflection_textured,
        ),
        None => (None, None, None, None),
    };

    let mut records = Vec::with_capacity(panels.len());
    for (i, panel) in panels.iter().enumerate() {
        let planar_slot = planar_slots.get(i).copied().flatten();
        records.push(build_panel_record(ctx.alloc, &ctx, panel, planar_slot)?);
    }

    Ok(TransparentProducer {
        pipeline,
        flat_rt_pso,
        textured_rt_pso,
        reflection_flat_pso,
        reflection_textured_pso,
        records,
    })
}

// The traced RT glass pipelines. The textured pair is skipped when the bindless
// pool is absent or the device could not spare a fifth descriptor set, leaving
// the flat trace.
fn build_glass_rt_pipelines(
    ctx: &ProducerCtx,
    flat_layout: vk::PipelineLayout,
) -> RenderResult<TracedGlassPipelines> {
    let shaders = compile_glass_rt_shaders(ctx.hot_reload, ctx.msaa, ctx.bindless_pool_size)?;
    build_traced_pipelines(ctx, flat_layout, &shaders, TransparentVertexInput::Position)
}

// Both traced pairs over one vertex stage: the shading pipelines target the
// transparent pass, the reflection ones the glass reflection pre-pass.
fn build_traced_pipelines(
    ctx: &ProducerCtx,
    flat_layout: vk::PipelineLayout,
    shaders: &GlassRtShaders,
    vertex_input: TransparentVertexInput,
) -> RenderResult<TracedGlassPipelines> {
    let shade = |layout, fs: &[u8]| {
        create_transparent_pipeline(
            ctx.device,
            ctx.render_pass,
            layout,
            &shaders.vs,
            fs,
            vertex_input,
        )
    };
    let reflection = |layout, fs: &[u8]| {
        create_glass_reflection_pipeline(
            ctx.device,
            ctx.reflection_render_pass,
            layout,
            &shaders.vs,
            fs,
            vertex_input,
        )
    };
    let (shade_textured, reflection_textured) = match (
        ctx.rt_layout_textured,
        &shaders.textured_fs,
        &shaders.reflection_textured_fs,
    ) {
        (Some(layout), Some(fs), Some(rfs)) => {
            (Some(shade(layout, fs)?), Some(reflection(layout, rfs)?))
        }
        _ => (None, None),
    };
    Ok(TracedGlassPipelines {
        shade_flat: shade(flat_layout, &shaders.flat_fs)?,
        shade_textured,
        reflection_flat: reflection(flat_layout, &shaders.reflection_flat_fs)?,
        reflection_textured,
    })
}

// Compile the see-through mesh vertex stage + its ray-traced fragments (flat,
// plus the textured variant when the bindless pool is live). No non-RT pair: the
// trace is what makes the mesh see-through.
fn compile_glass_mesh_shaders(
    hot_reload: bool,
    msaa: bool,
    pool_size: usize,
) -> RenderResult<GlassRtShaders> {
    let vs = super::builtin_shaders::GLASS_MESH_VERT.compile(hot_reload)?;
    let flat_fs = super::builtin_shaders::GLASS_MESH_FRAG_RT
        .at(msaa)
        .compile(hot_reload)?;
    let reflection_flat_fs = super::builtin_shaders::GLASS_MESH_REFLECTION_FRAG
        .at(msaa)
        .compile(hot_reload)?;
    let (textured_fs, reflection_textured_fs) = if pool_size > 0 {
        (
            Some(
                super::builtin_shaders::GLASS_MESH_FRAG_RT_TEXTURED
                    .at(msaa)
                    .compile(hot_reload)?,
            ),
            Some(
                super::builtin_shaders::GLASS_MESH_REFLECTION_FRAG_TEXTURED
                    .at(msaa)
                    .compile(hot_reload)?,
            ),
        )
    } else {
        (None, None)
    };
    Ok(GlassRtShaders {
        vs,
        flat_fs,
        textured_fs,
        reflection_flat_fs,
        reflection_textured_fs,
    })
}

// Build the see-through mesh pipelines and the per-frame params ring one block
// per mesh deep. The producer holds no records: a mesh's geometry lives in the
// shared scene buffers and its params change per frame, so the encoder rebuilds
// its draw list each frame (see `collect_mesh_draws`). Only called when the pass
// has RT pipeline layouts.
pub(in crate::vulkan) fn build_glass_mesh_producer(
    ctx: ProducerCtx,
    flat_layout: vk::PipelineLayout,
    object_indices: &[usize],
) -> RenderResult<GlassMeshProducer> {
    let shaders = compile_glass_mesh_shaders(ctx.hot_reload, ctx.msaa, ctx.bindless_pool_size)?;
    let pipelines = build_traced_pipelines(
        &ctx,
        flat_layout,
        &shaders,
        TransparentVertexInput::PositionAndNormal,
    )?;
    GlassMeshProducer::new(&ctx, pipelines, object_indices.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    // The `TransparentView` / `GlassParams` layout tests live with the structs
    // in `concinnity_core::render::uniforms`, and are checked against the compiled
    // shader by `shader_layout`.

    // The see-through mesh family compiles from the shared single source at
    // runtime, so a syntax or binding error would otherwise surface only as an
    // init failure on a GPU host.
    #[test]
    fn glass_mesh_shaders_compile() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        for msaa in [false, true] {
            super::compile_glass_mesh_shaders(false, msaa, 16)
                .unwrap_or_else(|e| panic!("glass_mesh shaders (msaa={msaa}) must compile: {e}"));
        }
    }

    #[test]
    fn glass_params_from_maps_fields() {
        let panel = GlassPanel {
            center: [1.0, 2.0, 3.0],
            normal: [0.0, 0.0, 1.0],
            tint: [0.6, 0.85, 0.9],
            opacity: 0.45,
            refraction_strength: 0.04,
            fresnel_power: 4.0,
            ..Default::default()
        };
        let p = glass_params_from(&panel, 1.0);
        assert_eq!(p.center, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(p.normal, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(p.tint, [0.6, 0.85, 0.9, 0.0]);
        assert_eq!(p.opacity, 0.45);
        assert_eq!(p.refraction_strength, 0.04);
        assert_eq!(p.fresnel_power, 4.0);
        assert_eq!(p.planar, 1.0);
        // A slotless pane gets planar = 0.0 (probe/sky fallback path).
        assert_eq!(glass_params_from(&panel, 0.0).planar, 0.0);
    }

    // Compile the glass vertex + fragment shaders (both MSAA variants) so a
    // regression fails the suite without a GPU. Mirrors the decal / fog compile
    // guards.
    #[test]
    fn glass_shaders_compile() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        super::compile_glass_shaders(false, true).expect("glass compiles (msaa)");
        super::compile_glass_shaders(false, false).expect("glass compiles (no msaa)");
    }

    // Compile the ray-traced glass shaders (both MSAA variants, both flat +
    // textured) so a regression in glass.hlsl's `GLASS_RT` arm (the shared
    // `{RT_TRACE}` traversal + the probe `{PROBE_COMMON}` injection + the
    // `RT_TEXTURED` split) fails the suite without a GPU. Mirrors
    // `rt_reflections_shaders_compile`. The CPU<->GPU `RtParams` / `RtGeomEntry`
    // layouts are guarded by the `rt_params_layout_*` / `rt_geom_entry_*` tests
    // in gfx::render_types.
    #[test]
    fn glass_rt_shaders_compile() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        for &msaa in &[true, false] {
            let shaders =
                super::compile_glass_rt_shaders(false, msaa, 4).expect("glass rt shaders compile");
            assert!(crate::vulkan::pipeline::is_spirv(&shaders.vs));
            assert!(crate::vulkan::pipeline::is_spirv(&shaders.flat_fs));
            assert!(
                shaders.textured_fs.is_some(),
                "pool_size>0 builds the textured variant"
            );
        }
        // pool_size 0 builds only the flat variant.
        let flat_only =
            super::compile_glass_rt_shaders(false, false, 0).expect("glass rt flat compiles");
        assert!(flat_only.textured_fs.is_none());
    }
}
