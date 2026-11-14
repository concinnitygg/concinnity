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
use concinnity_engine::render::post::reflection_composite::DenoiseFrame;
use concinnity_engine::render::post::rt_reflections::{
    MISS_ARGS_RESET, RAY_ARGS_RESET, classify_groups, fit_to_frame, ray_list_len, sky_listed,
};
use concinnity_engine::render::shader::programs::abi::rt_reflections::metal as rt;
use concinnity_engine::render::shader::programs::abi::rt_reflections::{
    CLASSIFY_GROUP, TRACE_GROUP,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLBlitCommandEncoder as _, MTLBuffer, MTLCommandBuffer as _, MTLComputeCommandEncoder as _,
    MTLComputePipelineState, MTLDevice, MTLPixelFormat, MTLResourceOptions, MTLTexture as _,
    MTLTextureUsage,
};

use crate::metal::builtin_shaders::{
    RT_REFLECTIONS_CLASSIFY, RT_REFLECTIONS_TRACE, RT_REFLECTIONS_TRACE_TEXTURED, compute_pipeline,
};
use crate::metal::context::MtlContext;
use crate::metal::descriptors::TextureDesc;
use crate::metal::dispatch::{dispatch_groups, threadgroup};
use crate::metal::encode::ComputeEncode;
use crate::metal::error::allocation_failed;
use crate::metal::post::reflection_composite::MtlReflectionCompositePass;
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
// the arguments every frame starts from, and the hits both kernels record for
// the denoiser. The list holds one entry per texel of the reflection target it
// was fit to; the hits are that target's size.
pub(crate) struct RtRays {
    list: Buffer,
    args: Buffer,
    reset: Buffer,
    // The rays the trace lists for the sky's own pass, one entry per texel at
    // most, and that list's indirect arguments with the block it starts from.
    misses: Buffer,
    miss_args: Buffer,
    miss_reset: Buffer,
    entries: u64,
    hits: Texture,
}

type Texture = Retained<ProtocolObject<dyn objc2_metal::MTLTexture>>;

// The hits target: each texel's reflected distance and deferred sky.
fn hits_target(
    device: &ProtocolObject<dyn MTLDevice>,
    (width, height): (u32, u32),
) -> RenderResult<Texture> {
    let desc = TextureDesc {
        format: MTLPixelFormat::RG16Float,
        width: width.max(1) as usize,
        height: height.max(1) as usize,
        usage: MTLTextureUsage(MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::ShaderWrite.0),
        ..Default::default()
    }
    .build();
    device
        .newTextureWithDescriptor(&desc)
        .ok_or_else(|| allocation_failed("rt reflection hits"))
}

impl RtRays {
    fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        entries: u64,
        extent: (u32, u32),
    ) -> RenderResult<Self> {
        let private = |bytes: u64, what: &str| {
            device
                .newBufferWithLength_options(
                    usize::saturating_from(bytes),
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| allocation_failed(what))
        };
        let reset = crate::metal::raytrace::upload_buffer(device, &RAY_ARGS_RESET, "rt ray reset")?;
        let miss_reset =
            crate::metal::raytrace::upload_buffer(device, &MISS_ARGS_RESET, "rt miss reset")?;
        let args_bytes = u64::saturating_from(size_of_val(&RAY_ARGS_RESET));
        let miss_bytes = u64::saturating_from(size_of::<render_types::ReflectionMissGpu>());
        Ok(Self {
            list: private(entries * 4, "rt ray list")?,
            args: private(args_bytes, "rt ray arguments")?,
            reset,
            misses: private(entries * miss_bytes, "rt miss list")?,
            miss_args: private(args_bytes, "rt miss arguments")?,
            miss_reset,
            entries,
            hits: hits_target(device, extent)?,
        })
    }

    // Keep `rays` big enough for a `width` x `height` target, with hits of
    // exactly its size. Only a larger target reallocates the lists; a frame in
    // flight keeps the old objects alive through its command buffer's retain.
    pub(crate) fn fit(
        rays: &mut Option<Self>,
        device: &ProtocolObject<dyn MTLDevice>,
        extent: (u32, u32),
    ) -> RenderResult<()> {
        let entries = ray_list_len(extent.0, extent.1);
        match rays.as_mut() {
            Some(r) if r.entries >= entries => {
                let size = (
                    u32::saturating_from(r.hits.width()),
                    u32::saturating_from(r.hits.height()),
                );
                if size != extent {
                    r.hits = hits_target(device, extent)?;
                }
            }
            _ => {
                *rays = None;
                *rays = Some(Self::new(device, entries, extent)?);
            }
        }
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
        denoise: Option<&render_types::ReflectionDenoiseParams>,
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
        // The rays the trace lists as leaving the scene take their sky, the
        // clouds marched along each, from the clouds' own kernel; without
        // this frame's clouds the trace settles them from the cube instead.
        // A mirror's sky waits for its turn only while the denoiser carries it.
        let sky = self.reflected_sky().filter(|_| sky_listed(rt_params));
        let velocity = self.gbuffer_velocity();
        let denoised = denoise.is_some()
            && velocity.is_some()
            && self
                .ssr
                .composite
                .as_ref()
                .is_some_and(MtlReflectionCompositePass::denoises);
        let fitted;
        let rt_params = match fit_to_frame(rt_params, sky.is_some(), denoised) {
            Some(p) => {
                fitted = p;
                &fitted
            }
            None => rt_params,
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
        // SAFETY: both buffers hold the 16-byte argument block.
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &rays.miss_reset,
                0,
                &rays.miss_args,
                0,
                size_of_val(&MISS_ARGS_RESET),
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
        dispatch_groups(&enc, [groups_x, groups_y, 1], CLASSIFY_GROUP);
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
        enc.set_buffer(&rays.misses, 0, rt::MISSES_OUT.index);
        enc.set_buffer(&rays.miss_args, 0, rt::MISS_ARGS_OUT.index);
        // The waves a hit on a level shell is shaded from: the simulated
        // cascades, or the stand-in a world without water builds on a device
        // that traces, whose waves are off so the kernel never samples them.
        if let Some(ocean) = &self.water.ocean {
            enc.set_texture(&ocean.arrays.slopes, rt::WAVE_SLOPES.index);
            enc.set_sampler(&ocean.arrays.sampler, rt::WAVE_SAMPLER.index);
        }
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
                threadgroup(TRACE_GROUP),
            );
        }
        if let Some(sky) = &sky {
            self.encode_reflected_sky(&enc, sky, reflection, &rays.misses, &rays.miss_args);
        }
        // End the trace encoder before opening the composite render pass: only
        // one command encoder may be live on a command buffer at a time.
        drop(enc);
        let frame = denoise
            .zip(velocity)
            .map(|(params, velocity)| DenoiseFrame {
                hits: rays.hits.as_ref(),
                velocity,
                params,
            });
        self.encode_reflection_composite(cmd_buf, frame)?;
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
        enc.set_texture(&screen.rays.hits, rt::HITS_OUT.index);
        enc.set_buffer(&screen.rays.list, 0, rt::RAY_LIST.index);
        enc.set_buffer(&screen.rays.args, 0, rt::RAY_ARGS.index);
    }
}
