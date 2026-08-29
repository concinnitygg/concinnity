// src/metal/post/rt_reflections.rs
//
// Hardware ray-traced reflection pass. A fullscreen-triangle fragment shader
// that, per glossy pixel, rebuilds a world-space surface point + normal from the
// SSR pre-pass G-buffer, traces a reflection ray against the scene acceleration
// structure ([`crate::metal::raytrace`]), shades the hit (sun Lambert + IBL
// ambient * material tint) or the IBL prefilter cube on a miss, and composites
// the result over the scene with the same Fresnel/gloss weighting SSR uses.
//
// It occupies the SSR-resolve slot in the frame graph (reads hdr_resolve, writes
// scene_pre_taa, which aliases `ssr_targets.output`) and is mutually exclusive
// with SSR resolve. Like SSGI it relies on the SSR pre-pass G-buffer, so the
// pre-pass is forced on whenever RT reflections are enabled.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer as _, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType,
    MTLRenderCommandEncoder as _, MTLRenderPassDescriptor, MTLRenderPipelineState, MTLStoreAction,
};

use crate::metal::context::MtlContext;
use crate::metal::encode::RenderEncode;
use crate::metal::post::fullscreen::{
    FullscreenBlend, build_slang_fullscreen_pipeline, set_fragment_sampler_range,
};
use crate::metal::scoped_encoder::ScopedEncoder;
use crate::metal::slang_shaders::SlangLib;

// Fragment sampler index the textured variant reads the bindless pool through.
// slangc splits the combined screen sources into texture + sampler pairs at
// 0..3 and reserves 4..3+MAX_PROBES for the probe cube array, so the pool's own
// sampler lands past both runs; the emitted `[[sampler(12)]]` is what pins it.
const RT_POOL_SAMPLER_INDEX: usize = 12;

// Build one RT-reflection pipeline from the given single-source variant: a
// fullscreen-triangle pass that traces a reflection ray and writes reflected
// radiance + composite weight into a single-sample `RGBA16Float` target (the
// same `ssr_targets.output` SSR resolve would write). Two variants exist:
// `RT_REFLECTIONS_FRAG` (flat tint) and `RT_REFLECTIONS_FRAG_TEXTURED` (samples
// the bindless albedo pool). Built only when RT reflections are enabled and the
// GPU supports ray tracing: the metallib carries a real ray query, which a
// non-RT device cannot load.
pub(crate) fn build_rt_reflection_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    fragment: &SlangLib,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    build_slang_fullscreen_pipeline(
        device,
        fragment,
        MTLPixelFormat::RGBA16Float,
        FullscreenBlend::Replace,
        hot_reload,
    )
}

impl MtlContext {
    // Encode the RT-reflection resolve: a fullscreen pass that traces each
    // glossy pixel's reflection ray against the scene BVH and writes the
    // reflected radiance + composite weight into the reflection target, then
    // runs the shared roughness-aware blur + composite into `ssr_targets.output`.
    // Runs after the main pass in the SSR-resolve slot; only called when RT
    // reflections are on and the acceleration structure + pipeline are live.
    // Returns `Ok(0)` (a no-op) when any required resource is missing: the engine
    // then leaves the base scene untouched, the same defensive pattern
    // `encode_ssgi` uses.
    pub(in crate::metal) fn encode_rt_reflections(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        rt_params: &crate::gfx::render_types::RtParams,
        bindless_tex_args: Option<&Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    ) -> Result<u32, String> {
        let (targets, accel, gb_normal_depth, gb_roughness) = match (
            &self.ssr.targets,
            &self.rt.accel,
            self.gbuffer_normal_depth(),
            self.gbuffer_roughness(),
        ) {
            (Some(t), Some(a), Some(n), Some(r)) => (t, a, n, r),
            // No G-buffer or acceleration structure (unsupported GPU / empty
            // scene): skip, leaving the base scene.
            _ => return Ok(0),
        };

        // The BVH this pass reads (TLAS, skinned BLAS, deformed-vertex buffer)
        // was built earlier this frame in `raytrace::rebuild_skinned`, on command
        // buffers committed (in `rt_dynamic_update`) before this trace's command
        // buffer on the shared queue. That rebuild is fully async (no
        // `waitUntilCompleted`), so same-queue FIFO commit order runs skin → build →
        // trace, the same ordering every cross-pass read relies on. No explicit
        // GPU-side wait is encoded here.
        // Sample the hit's albedo texture from the bindless pool when it is
        // available (the standard GPU-cull path); otherwise fall back to the
        // flat-tint pipeline so non-bindless worlds still get RT reflections.
        let textured = bindless_tex_args.is_some() && self.rt.pipeline_textured.is_some();
        let pipeline = match (textured, &self.rt.pipeline_textured, &self.rt.pipeline) {
            (true, Some(p), _) => p,
            (_, _, Some(p)) => p,
            _ => return Ok(0),
        };

        let desc = MTLRenderPassDescriptor::new();
        // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
        // declares.
        unsafe {
            let ca = desc.colorAttachments().objectAtIndexedSubscript(0);
            ca.setTexture(Some(targets.reflection.as_ref()));
            ca.setLoadAction(MTLLoadAction::DontCare);
            ca.setStoreAction(MTLStoreAction::Store);
        }
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_render(&desc, crate::metal::pass_timing::PassId::RtReflections);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&desc)
                .ok_or("failed to get RT reflections encoder")?,
            "rt reflections",
        );
        enc.set_pipeline(pipeline);
        // Textures + samplers mirror the SSR resolve.
        enc.set_fragment_texture(self.hdr_targets.hdr_resolve.as_ref(), 0);
        enc.set_fragment_texture(gb_normal_depth, 1);
        enc.set_fragment_texture(gb_roughness, 2);
        enc.set_fragment_texture(self.env_map.prefilter.as_ref(), 3);
        // Local reflection-probe cubes at texture(4..4+MAX_PROBES): a missed
        // reflection ray reflects the box-projected scene capture instead of
        // the foreign sky HDR (the source the forward IBL specular term uses).
        // probe_cube_or_sky returns the sky for unbaked slots, so binding all
        // MAX_PROBES is always valid; the ProbeSet's count gates use.
        for i in 0..concinnity_core::render::uniforms::MAX_PROBES {
            enc.set_fragment_texture(self.probe_cube_or_sky(i), 4 + i);
        }
        // The screen sources take the post sampler at 0..2; the prefilter
        // cube and every probe cube take the cube sampler at 3..4+MAX_PROBES,
        // one per texture slangc split the combined declarations into. The
        // textured variant reads the bindless pool through the repeat-address
        // pool sampler, which lands past the probe run.
        set_fragment_sampler_range(&enc, &self.post_sampler, 0, 3);
        set_fragment_sampler_range(
            &enc,
            self.cube_sampler.as_ref(),
            3,
            1 + concinnity_core::render::uniforms::MAX_PROBES,
        );
        set_fragment_sampler_range(&enc, self.sampler.as_ref(), RT_POOL_SAMPLER_INDEX, 1);
        // buffer(0) params; buffers 1..3 the shared geometry the kernel
        // fetches the hit triangle from; the TLAS at buffer(4).
        enc.set_fragment_value(rt_params, 0);
        enc.set_fragment_buffer(self.vertex_buffer.as_ref(), 0, 1);
        enc.set_fragment_buffer(self.index_buffer.as_ref(), 0, 2);
        enc.set_fragment_buffer(accel.geom_table.as_ref(), 0, 3);
        enc.set_fragment_acceleration_structure(accel.tlas.as_ref(), 4);
        // Deformed (posed) skinned vertices + the skinned index buffer
        // the trace fetches for skinned hits. Always bound (1-element
        // dummies when the scene has no skinned geometry) so the shader's
        // bindings are satisfied even though the skinned branch is never
        // taken then. Direct buffer bindings, so Metal makes them resident.
        enc.set_fragment_buffer(accel.deformed_verts.as_ref(), 0, 5);
        enc.set_fragment_buffer(accel.skinned_indices.as_ref(), 0, 6);
        // Reflection-probe set (count + per-probe parallax boxes) at buffer(8);
        // count == 0 keeps the sky miss fallback. (buffer(7) is the bindless
        // texture pool, bound only on the textured path below.)
        enc.set_fragment_value(&self.probe.set, 8);
        // Textured path: bind the bindless albedo pool at buffer(7) (the
        // same index the main pass uses) and declare its textures resident.
        if textured && let Some(tex_args) = bindless_tex_args {
            enc.set_fragment_buffer(
                tex_args.as_ref(),
                0,
                crate::metal::context::BINDLESS_TEXTURE_ARG_BUFFER_INDEX,
            );
            self.use_bindless_textures(&enc);
        }
        // The TLAS references each BLAS indirectly, so the BLASes are not
        // auto-tracked: declare them resident or the trace reads garbage.
        crate::metal::raytrace::use_blas_resident_fragment(&enc, &accel.blas);
        // SAFETY: the fullscreen triangle's three vertices are generated from
        // `[[vertex_id]]` in the shader, so the draw reads no vertex buffer.
        unsafe {
            enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
        // End the trace encoder before opening the composite render pass: only
        // one command encoder may be live on a command buffer at a time.
        drop(enc);
        self.encode_reflection_composite(cmd_buf)?;
        Ok(0)
    }
}
