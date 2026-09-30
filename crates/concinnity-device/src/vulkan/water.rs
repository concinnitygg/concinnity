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
use concinnity_core::components::WaterSurface;
use concinnity_core::geometry::water_grid::build_water_grid;
use concinnity_core::gfx::mesh_payload::Vertex;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::uniforms::WaterParams;

use super::allocator::DeviceAllocator;
use crate::vulkan::builtin_shaders::CompileProgram;
use crate::vulkan::transparent::{
    ProducerCtx, RecordUpload, TransparentProducer, TransparentRecord, TransparentVertexInput,
    create_transparent_pipeline,
};

// Compile the water vertex + fragment shaders, injecting the MSAA define so the
// depth sampler type matches the main-depth resource's sample count.
fn compile_water_shaders(hot_reload: bool, msaa: bool) -> RenderResult<(Vec<u8>, Vec<u8>)> {
    let vert = super::builtin_shaders::WATER_VERT.compile(hot_reload)?;
    let frag = super::builtin_shaders::WATER_FRAG
        .at(msaa)
        .compile(hot_reload)?;
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
// the textured variant when `pool_size > 0`). dxc emits `SPV_KHR_ray_query`
// for the traversal, which the device already advertises wherever these
// pipelines are built.
fn compile_water_rt_shaders(
    hot_reload: bool,
    msaa: bool,
    pool_size: usize,
) -> RenderResult<WaterRtShaders> {
    let vs = super::builtin_shaders::WATER_VERT.compile(hot_reload)?;
    let flat_fs = super::builtin_shaders::WATER_FRAG_RT
        .at(msaa)
        .compile(hot_reload)?;
    let textured_fs = if pool_size > 0 {
        Some(
            super::builtin_shaders::WATER_FRAG_RT_TEXTURED
                .at(msaa)
                .compile(hot_reload)?,
        )
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
) -> RenderResult<TransparentRecord> {
    let (verts, idxs) =
        build_water_grid(surface.extent[0], surface.extent[1], surface.subdivisions)
            .map_err(RenderError::Other)?;

    // Flatten into the standard engine `Vertex` layout. Tangent and color are
    // placeholders: the water shader rebuilds its normal frame analytically from
    // the wave derivatives and the fragment ignores per-vertex color.
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
    TransparentRecord::upload(
        alloc,
        ctx.record_descriptors(planar_slot),
        RecordUpload {
            vertices: &packed,
            indices: &idxs,
            params: bytemuck::bytes_of(&params),
            visible: surface.visible,
            center: surface.center,
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
) -> RenderResult<TransparentProducer> {
    let (vert_spv, frag_spv) = compile_water_shaders(ctx.hot_reload, ctx.msaa)?;
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
        reflection_flat_pso: None,
        reflection_textured_pso: None,
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
) -> RenderResult<WaterRtPipelines> {
    let shaders = compile_water_rt_shaders(ctx.hot_reload, ctx.msaa, ctx.bindless_pool_size)?;
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
        super::compile_water_shaders(false, true).expect("water compiles (msaa)");
        super::compile_water_shaders(false, false).expect("water compiles (no msaa)");
    }

    // Compile the ray-traced water shaders (both MSAA variants, both flat +
    // textured) so a regression in water.hlsl's `WATER_RT` arm (the shared
    // `{RT_TRACE}` traversal + the probe `{PROBE_COMMON}` injection + the
    // `RT_TEXTURED` split) fails the suite without a GPU.
    #[test]
    fn water_rt_shaders_compile() {
        concinnity_shader::require_dxc!();
        for &msaa in &[true, false] {
            let shaders =
                super::compile_water_rt_shaders(false, msaa, 4).expect("water rt shaders compile");
            assert!(crate::vulkan::pipeline::is_spirv(&shaders.vs));
            assert!(crate::vulkan::pipeline::is_spirv(&shaders.flat_fs));
            assert!(
                shaders.textured_fs.is_some(),
                "pool_size>0 builds the textured variant"
            );
        }
        // pool_size 0 builds only the flat variant.
        let flat_only =
            super::compile_water_rt_shaders(false, false, 0).expect("water rt flat compiles");
        assert!(flat_only.textured_fs.is_none());
    }
}
