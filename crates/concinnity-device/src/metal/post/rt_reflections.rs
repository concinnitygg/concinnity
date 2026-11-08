//! Hardware ray-traced reflection pass, screen space first. A classify kernel
//! walks each glossy pixel's reflection ray through SSGI's depth pyramid,
//! writes what the screen and the probe settle into the reflection target, and
//! lists the pixels still owed a hardware ray; a trace kernel dispatched
//! indirectly over that list traces them against the scene acceleration
//! structure ([`crate::metal::raytrace`]) and adds their share. The shared
//! roughness-aware blur + composite then blends the target over the scene.
//!
//! It occupies the SSR-resolve slot in the frame graph (reads `hdr_resolve`, writes
//! `scene_pre_taa`, the reflection composite's output) and is mutually exclusive
//! with SSR resolve. Like SSGI it relies on the SSR pre-pass G-buffer, so the
//! pre-pass is forced on whenever RT reflections are enabled.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::post::rt_reflections::{
    CLASSIFY_GROUP, RAY_ARGS_RESET, TRACE_GROUP, classify_groups, ray_list_len,
};
use concinnity_engine::render::shader::programs::abi::rt_reflections::metal as rt;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLBlitCommandEncoder as _, MTLBuffer, MTLCommandBuffer as _, MTLComputeCommandEncoder as _,
    MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize, MTLTexture as _,
};

use crate::metal::builtin_shaders::{
    RT_REFLECTIONS_CLASSIFY, RT_REFLECTIONS_TRACE, RT_REFLECTIONS_TRACE_TEXTURED, compute_pipeline,
};
use crate::metal::context::MtlContext;
use crate::metal::encode::ComputeEncode;
use crate::metal::error::allocation_failed;
use crate::metal::scoped_encoder::ScopedEncoder;

type ComputePipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

// The three RT reflection kernels: the classify, and the trace with flat-tint
// and textured hit shading. Built only when RT reflections are enabled and the
// GPU supports ray tracing: the trace's metallib carries a real ray query,
// which a non-RT device cannot load.
pub(crate) struct RtReflectionPipelines {
    pub classify: ComputePipeline,
    pub trace: ComputePipeline,
    pub trace_textured: ComputePipeline,
}

impl RtReflectionPipelines {
    pub(crate) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        Ok(Self {
            classify: compute_pipeline(device, &RT_REFLECTIONS_CLASSIFY, hot_reload)?,
            trace: compute_pipeline(device, &RT_REFLECTIONS_TRACE, hot_reload)?,
            trace_textured: compute_pipeline(device, &RT_REFLECTIONS_TRACE_TEXTURED, hot_reload)?,
        })
    }
}

// The pixels the classify owes a ray and the trace's indirect arguments, with
// the arguments every frame starts from. The list holds one entry per texel of
// the reflection target it was fit to.
pub(crate) struct RtRays {
    list: Buffer,
    args: Buffer,
    reset: Buffer,
    entries: u64,
}

impl RtRays {
    fn new(device: &ProtocolObject<dyn MTLDevice>, entries: u64) -> RenderResult<Self> {
        let private = |bytes: u64, what: &str| {
            device
                .newBufferWithLength_options(
                    usize::saturating_from(bytes),
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| allocation_failed(what))
        };
        let reset = crate::metal::raytrace::upload_buffer(device, &RAY_ARGS_RESET, "rt ray reset")?;
        Ok(Self {
            list: private(entries * 4, "rt ray list")?,
            args: private(
                u64::saturating_from(size_of_val(&RAY_ARGS_RESET)),
                "rt ray arguments",
            )?,
            reset,
            entries,
        })
    }

    // Keep `rays` big enough for a `width` x `height` target. Only a larger
    // target reallocates; a frame in flight keeps the old buffers alive through
    // its command buffer's retain.
    pub(crate) fn fit(
        rays: &mut Option<Self>,
        device: &ProtocolObject<dyn MTLDevice>,
        (width, height): (u32, u32),
    ) -> RenderResult<()> {
        let entries = ray_list_len(width, height);
        if rays.as_ref().is_some_and(|r| r.entries >= entries) {
            return Ok(());
        }
        *rays = None;
        *rays = Some(Self::new(device, entries)?);
        Ok(())
    }
}

// The pass's screen-space targets this frame.
struct RtScreen<'a> {
    reflection: &'a ProtocolObject<dyn objc2_metal::MTLTexture>,
    normal_depth: &'a ProtocolObject<dyn objc2_metal::MTLTexture>,
    roughness: &'a ProtocolObject<dyn objc2_metal::MTLTexture>,
    rays: &'a RtRays,
}

impl MtlContext {
    // Encode the RT-reflection resolve: reset the ray arguments, classify every
    // trace texel, trace the listed ones, then run the shared roughness-aware
    // blur + composite into its output. Runs after the main pass in the
    // SSR-resolve slot; only called when RT reflections are on and the
    // acceleration structure + pipelines are live. Returns `Ok(0)` (a no-op)
    // when any required resource is missing: the engine then leaves the base
    // scene untouched, the same defensive pattern `encode_ssgi` uses.
    pub(in crate::metal) fn encode_rt_reflections(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        rt_params: &render_types::RtParams,
        bindless_tex_args: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
    ) -> RenderResult<u32> {
        let (
            Some(reflection),
            Some(accel),
            Some(pipelines),
            Some(rays),
            Some(gb_normal_depth),
            Some(gb_roughness),
        ) = (
            &self.ssr.reflection,
            &self.rt.accel,
            &self.rt.pipelines.reflections,
            &self.rt.rays,
            self.gbuffer_normal_depth(),
            self.gbuffer_roughness(),
        )
        else {
            return Ok(0);
        };
        let screen = RtScreen {
            reflection,
            normal_depth: gb_normal_depth,
            roughness: gb_roughness,
            rays,
        };
        // SSGI's pyramid, which the classify reads only while `rt_params`
        // names its levels; the G-buffer stands in for it otherwise.
        let pyramid = match &self.ssgi.pass {
            Some(pass) if rt_params.cam_pos[3] > 0.0 => pass.pyramid()?.texture(),
            _ => gb_normal_depth,
        };

        let blit = ScopedEncoder::new(
            cmd_buf
                .blitCommandEncoder()
                .ok_or_else(|| RenderError::Other("failed to get RT ray reset encoder".into()))?,
            ns_string!("rt ray reset"),
        );
        // SAFETY: both buffers hold the 16-byte argument block.
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &rays.reset,
                0,
                &rays.args,
                0,
                size_of_val(&RAY_ARGS_RESET),
            );
        }
        drop(blit);

        let desc = objc2_metal::MTLComputePassDescriptor::new();
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_compute_first(&desc, crate::metal::pass_timing::PassId::RtReflections);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .computeCommandEncoderWithDescriptor(&desc)
                .ok_or_else(|| RenderError::Other("failed to get RT classify encoder".into()))?,
            ns_string!("rt reflections classify"),
        );
        enc.set_pipeline(&pipelines.classify);
        self.bind_rt_screen(&enc, rt_params, &screen);
        enc.set_texture(pyramid, rt::PYRAMID.index);
        let (groups_x, groups_y) = classify_groups(
            u32::saturating_from(reflection.width()),
            u32::saturating_from(reflection.height()),
        );
        let classify_tg = usize::saturating_from(CLASSIFY_GROUP);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: usize::saturating_from(groups_x),
                height: usize::saturating_from(groups_y),
                depth: 1,
            },
            MTLSize {
                width: classify_tg,
                height: classify_tg,
                depth: 1,
            },
        );
        drop(enc);

        // Sample the hit's albedo texture from the bindless pool when it is
        // available (the standard GPU-cull path); otherwise the flat-tint
        // trace so non-bindless worlds still get RT reflections.
        let textured = bindless_tex_args.is_some();
        let desc = objc2_metal::MTLComputePassDescriptor::new();
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_compute_last(&desc, crate::metal::pass_timing::PassId::RtReflections);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .computeCommandEncoderWithDescriptor(&desc)
                .ok_or_else(|| RenderError::Other("failed to get RT trace encoder".into()))?,
            ns_string!("rt reflections trace"),
        );
        enc.set_pipeline(if textured {
            &pipelines.trace_textured
        } else {
            &pipelines.trace
        });
        self.bind_rt_screen(&enc, rt_params, &screen);
        // Buffers 1..3 the shared geometry the kernel fetches the hit triangle
        // from; the TLAS at buffer(4); the deformed (posed) skinned vertices +
        // the skinned index buffer at 5 and 6, always bound (1-element
        // dummies when the scene has no skinned geometry).
        enc.set_buffer(self.scene.vertex_buffer.as_ref(), 0, rt::VERTICES.index);
        enc.set_buffer(self.scene.index_buffer.as_ref(), 0, rt::INDICES.index);
        enc.set_buffer(accel.geom_table.as_ref(), 0, rt::GEOMETRY.index);
        enc.set_acceleration_structure(accel.tlas.as_ref(), rt::SCENE_TLAS.index);
        enc.set_buffer(accel.deformed_verts.as_ref(), 0, rt::SKINNED_VERTICES.index);
        enc.set_buffer(accel.skinned_indices.as_ref(), 0, rt::SKINNED_INDICES.index);
        if textured && let Some(tex_args) = bindless_tex_args {
            // The bindless albedo pool, at the pool's offset in the main pass's
            // block, its textures declared resident.
            enc.set_buffer(
                tex_args.as_ref(),
                crate::metal::bindless_args::BINDLESS_POOL_OFFSET,
                rt::TEXTURES.index,
            );
            self.use_bindless_textures_compute(&enc);
            enc.set_sampler(self.scene.sampler.as_ref(), rt::POOL_SAMPLER.index);
            // The canopy block and view tables a hit on ground growing grass
            // shades from, as the main pass binds them.
            enc.set_value(&self.canopy, rt::CANOPY.index);
            if let Some(tables) = self.canopy_tables.buffer() {
                enc.set_buffer(tables, 0, rt::CANOPY_TABLES.index);
            }
        }
        // The TLAS references each BLAS indirectly, so the BLASes are not
        // auto-tracked: declare them resident or the trace reads garbage.
        crate::metal::raytrace::use_blas_resident_compute(&enc, accel.traced_blas());
        // SAFETY: the argument buffer holds the three group counts the classify
        // grew to cover its list, at offset 0, written before this encoder.
        unsafe {
            enc.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                &rays.args,
                0,
                MTLSize {
                    width: usize::saturating_from(TRACE_GROUP),
                    height: 1,
                    depth: 1,
                },
            );
        }
        // End the trace encoder before opening the composite render pass: only
        // one command encoder may be live on a command buffer at a time.
        drop(enc);
        self.encode_reflection_composite(cmd_buf)?;
        Ok(0)
    }

    // What both kernels bind: the params, the screen sources and their
    // samplers, the probe fallback, the output, and the ray list and arguments.
    fn bind_rt_screen(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        rt_params: &render_types::RtParams,
        screen: &RtScreen<'_>,
    ) {
        enc.set_value(rt_params, rt::PARAMS.index);
        enc.set_texture(self.targets.hdr.hdr_resolve.as_ref(), rt::SCENE.index);
        enc.set_texture(screen.normal_depth, rt::GBUFFER.index);
        enc.set_texture(screen.roughness, rt::ROUGHNESS.index);
        enc.set_texture(self.scene.env_map.prefilter.as_ref(), rt::PREFILTER.index);
        // A missed ray reflects the box-projected scene capture (the source the
        // forward IBL specular term uses), blending the probes the main
        // camera's cluster grid bins at the pixel; count == 0 keeps the sky
        // fallback.
        self.probe_bindings().bind_compute(
            enc,
            crate::metal::probe_set::ProbeSlots {
                set: rt::PROBE_SET.index,
                records: rt::PROBE_RECORDS.index,
                cubes: Some(rt::PROBE_CUBES.index),
                cluster: Some((rt::CLUSTER.index, rt::CLUSTER_LIST.index)),
            },
        );
        // The screen sources take the post sampler; the prefilter cube and the
        // probe array's own sampler take the cube sampler.
        for slot in [
            rt::SCENE_SAMPLER,
            rt::GBUFFER_SAMPLER,
            rt::ROUGHNESS_SAMPLER,
        ] {
            enc.set_sampler(&self.composite.sampler, slot.index);
        }
        for slot in [rt::PREFILTER_SAMPLER, rt::PROBE_CUBE_SAMPLER] {
            enc.set_sampler(self.scene.cube_sampler.as_ref(), slot.index);
        }
        enc.set_texture(screen.reflection, rt::OUTPUT.index);
        enc.set_buffer(&screen.rays.list, 0, rt::RAY_LIST.index);
        enc.set_buffer(&screen.rays.args, 0, rt::RAY_ARGS.index);
    }
}
