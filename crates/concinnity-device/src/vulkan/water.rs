//! WaterSurface: one producer of the engine's transparent pass on the Vulkan
//! backend (`transparent.rs` owns the render pass itself, the scene snapshot, the
//! shared descriptor / pipeline layouts and the combined back-to-front draw
//! order; `glass.rs` is the other producer). Each surface is a flat tessellated
//! XZ grid built once at init and displaced per frame by the vertex stage's
//! Gerstner sum; the fragment refracts the pass's scene snapshot, tints and foams
//! it by the water-column thickness the main depth gives, and mixes a reflection
//! over it by a Schlick Fresnel term (see shaders/water.hlsl, the single source
//! all three backends compile).
//!
//! Same uniform layouts, back-to-front ordering and manual depth-occlusion test
//! as the DirectX and Metal hosts.

use ash::vk;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::water::WaterBody;
use concinnity_core::render::water::frame::WaterMotion;

use concinnity_core::render::shader_programs::ShaderProgram;

use crate::vulkan::builtin_shaders::CompileProgram;
use crate::vulkan::transparent::{
    ProducerCtx, RecordUpload, ShapePipelines, TransparentProducer, TransparentRecord,
    TransparentVertexInput, create_transparent_pipeline,
};

// The programs one water shape draws with: a rectangle's, or a planet sea's.
struct WaterPrograms {
    vert: &'static ShaderProgram,
    frag: &'static ShaderProgram,
    frag_rt: &'static ShaderProgram,
    frag_rt_textured: &'static ShaderProgram,
}

const RECTANGLE: WaterPrograms = WaterPrograms {
    vert: &super::builtin_shaders::WATER_VERT,
    frag: &super::builtin_shaders::WATER_FRAG,
    frag_rt: &super::builtin_shaders::WATER_FRAG_RT,
    frag_rt_textured: &super::builtin_shaders::WATER_FRAG_RT_TEXTURED,
};

const SEA: WaterPrograms = WaterPrograms {
    vert: &super::builtin_shaders::WATER_SEA_VERT,
    frag: &super::builtin_shaders::WATER_SEA_FRAG,
    frag_rt: &super::builtin_shaders::WATER_SEA_FRAG_RT,
    frag_rt_textured: &super::builtin_shaders::WATER_SEA_FRAG_RT_TEXTURED,
};

// Compile a shape's vertex + fragment shaders, injecting the MSAA define so
// the depth sampler type matches the main-depth resource's sample count.
fn compile_water_shaders(
    programs: &WaterPrograms,
    hot_reload: bool,
    msaa: bool,
) -> RenderResult<(Vec<u8>, Vec<u8>)> {
    let vert = programs.vert.compile(hot_reload)?;
    let frag = programs.frag.at(msaa).compile(hot_reload)?;
    Ok((vert, frag))
}

// SPIR-V blobs for a shape's ray-traced water pipelines: its vertex stage (the
// same one the base pass uses -- the trace is entirely in the fragment), the
// flat fragment, and the textured fragment (`None` when the bindless pool is
// absent).
struct WaterRtShaders {
    vs: Vec<u8>,
    flat_fs: Vec<u8>,
    textured_fs: Option<Vec<u8>>,
}

// Compile a shape's vertex shader + its ray-traced fragment (flat, plus the
// textured variant when `pool_size > 0`). dxc emits `SPV_KHR_ray_query` for
// the traversal, which the device already advertises wherever these pipelines
// are built.
fn compile_water_rt_shaders(
    programs: &WaterPrograms,
    hot_reload: bool,
    msaa: bool,
    pool_size: usize,
) -> RenderResult<WaterRtShaders> {
    let vs = programs.vert.compile(hot_reload)?;
    let flat_fs = programs.frag_rt.at(msaa).compile(hot_reload)?;
    let textured_fs = if pool_size > 0 {
        Some(programs.frag_rt_textured.at(msaa).compile(hot_reload)?)
    } else {
        None
    };
    Ok(WaterRtShaders {
        vs,
        flat_fs,
        textured_fs,
    })
}

// Upload one surface's tessellated grid VB + IB and its per-frame `WaterParams`
// UBOs, then allocate + write the surface's descriptor sets.
fn build_surface_record(
    ctx: &ProducerCtx,
    body: &WaterBody,
    planar_slot: Option<usize>,
) -> RenderResult<(TransparentRecord, WaterMotion)> {
    let surface = &body.surface;
    let (vertices, indices) = body.mesh().map_err(RenderError::Other)?;
    let motion = WaterMotion::new(body, planar_slot.is_some());
    let record = TransparentRecord::upload(
        &ctx.gpu(),
        ctx.record_descriptors(planar_slot),
        RecordUpload {
            vertices: &vertices,
            indices: &indices,
            params: bytemuck::bytes_of(motion.params()),
            visible: surface.visible,
            center: body.sort_center(),
            planar_slot,
        },
    )?;
    Ok((record, motion))
}

// Build the water pipelines and one record per authored surface. The RT pair is
// built whenever the pass has RT pipeline layouts (regardless of whether RT is on
// at launch, so a live `quality-set ray_traced_reflections` selects it with no
// pipeline rebuild); a compile failure leaves it absent and the base
// probe/planar path runs.
pub(in crate::vulkan) fn build_water_producer(
    ctx: ProducerCtx,
    surfaces: &[WaterBody],
    // Per-surface planar resolve slot (aligned with `surfaces`); `None` surfaces
    // keep the probe/sky reflection. From `assign_planar_slots`.
    planar_slots: &[Option<usize>],
) -> RenderResult<TransparentProducer> {
    let rectangle = build_water_pipelines(&ctx, &RECTANGLE)?;
    let sea = surfaces
        .iter()
        .any(|b| b.sea.is_some())
        .then(|| build_water_pipelines(&ctx, &SEA))
        .transpose()?;

    let mut records = Vec::with_capacity(surfaces.len());
    let mut water = Vec::with_capacity(surfaces.len());
    for (i, surface) in surfaces.iter().enumerate() {
        let planar_slot = planar_slots.get(i).copied().flatten();
        let (record, motion) = build_surface_record(&ctx, surface, planar_slot)?;
        records.push(record);
        water.push(motion);
    }

    Ok(TransparentProducer {
        pipeline: rectangle.pipeline,
        flat_rt_pso: rectangle.flat_rt_pso,
        textured_rt_pso: rectangle.textured_rt_pso,
        reflection_flat_pso: None,
        reflection_textured_pso: None,
        sea,
        records,
        water,
    })
}

// A shape's water pipelines. The RT pair is built whenever the pass has RT
// pipeline layouts; a compile failure leaves it absent and the base probe /
// planar path runs.
fn build_water_pipelines(
    ctx: &ProducerCtx,
    programs: &WaterPrograms,
) -> RenderResult<ShapePipelines> {
    let (vert_spv, frag_spv) = compile_water_shaders(programs, ctx.hot_reload, ctx.msaa)?;
    let pipeline = create_transparent_pipeline(
        ctx.device,
        ctx.render_pass,
        ctx.layout,
        &vert_spv,
        &frag_spv,
        TransparentVertexInput::Position,
    )?;
    let (flat_rt_pso, textured_rt_pso) = match ctx.rt_layout_flat {
        Some(flat_layout) => match build_water_rt_pipelines(ctx, programs, flat_layout) {
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
    Ok(ShapePipelines {
        pipeline,
        flat_rt_pso,
        textured_rt_pso,
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
    programs: &WaterPrograms,
    flat_layout: vk::PipelineLayout,
) -> RenderResult<WaterRtPipelines> {
    let shaders =
        compile_water_rt_shaders(programs, ctx.hot_reload, ctx.msaa, ctx.bindless_pool_size)?;
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
    // The `WaterParams` / `WaterWaveGpu` layout tests live with the structs in
    // `concinnity_core::render::uniforms`, and are checked against the compiled shader
    // by `shader_layout`.

    // Compile the water vertex + fragment shaders (both MSAA variants) so a
    // regression fails the suite without a GPU.
    #[test]
    fn water_shaders_compile() {
        concinnity_shader::require_dxc!();
        for programs in [&super::RECTANGLE, &super::SEA] {
            super::compile_water_shaders(programs, false, true).expect("water compiles (msaa)");
            super::compile_water_shaders(programs, false, false).expect("water compiles");
        }
    }

    // Compile the ray-traced water shaders (both MSAA variants, both flat +
    // textured) so a regression in water.hlsl's `WATER_RT` arm (the shared
    // `{RT_TRACE}` traversal + the probe `{PROBE_COMMON}` injection + the
    // `RT_TEXTURED` split) fails the suite without a GPU.
    #[test]
    fn water_rt_shaders_compile() {
        concinnity_shader::require_dxc!();
        for (programs, msaa) in [
            (&super::RECTANGLE, true),
            (&super::RECTANGLE, false),
            (&super::SEA, true),
            (&super::SEA, false),
        ] {
            let shaders = super::compile_water_rt_shaders(programs, false, msaa, 4)
                .expect("water rt shaders compile");
            assert!(crate::vulkan::pipeline::is_spirv(&shaders.vs));
            assert!(crate::vulkan::pipeline::is_spirv(&shaders.flat_fs));
            assert!(
                shaders.textured_fs.is_some(),
                "pool_size>0 builds the textured variant"
            );
        }
        // pool_size 0 builds only the flat variant.
        let flat_only = super::compile_water_rt_shaders(&super::RECTANGLE, false, false, 0)
            .expect("water rt flat compiles");
        assert!(flat_only.textured_fs.is_none());
    }
}
