//! Metal shader hot-reload: `MtlContext::reload_shaders` rebuilds every live
//! built-in pipeline from the checkout's shader sources. The `reload-shaders`
//! debug command sets the shared flag, and the main thread polls it at the top
//! of `draw_frame`.
//!
//! Entirely a dev-loop concern: the flag exists only when `MtlContext::new` is
//! called with `hot_reload = true`. Production `cn run` never sets it.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::gfx::mesh_payload;
use concinnity_core::render::error::RenderResult;
use objc2::rc::Retained;
use objc2_metal::{MTLVertexDescriptor, MTLVertexFormat, MTLVertexStepFunction};
use std::sync::atomic::Ordering;

use super::auto_exposure::build_auto_exposure_pipelines;
use super::context::MtlContext;
use super::cull::{build_cull_pipeline, build_shadow_cull_pipeline};
use super::decal::build_decal_pipeline;
use super::descriptors::{VertexAttr, VertexLayout, vertex_descriptor};
use super::fog::build_fog_pipeline;
use super::hiz::build_hiz_pipelines;
use super::init::pipelines::{
    build_bindless_sampler_args, build_main_pipeline, build_shadow_bindless_pipeline,
    build_shadow_pipeline, make_vertex_descriptor,
};
use super::pipeline::{build_post_pipeline, build_text_pipeline};
use super::post::post_device::MtlPostDevice;
use super::post::{
    build_bloom_pipelines, build_gbuffer_bindless_pipeline, build_reflection_blur_pipeline,
    build_reflection_composite_pipeline, build_rt_reflection_pipeline, build_ssao_pipeline,
};
use super::resources::skinning::{build_skinned_shadow_pipeline, make_skinned_vertex_descriptor};
use crate::metal::builtin_shaders::{SSAO_BLUR, SSAO_KERNEL};

// Rebuild a built-in pipeline only when it is currently live. Expands to
// `if $cond { Some($build?) } else { None }`: the rebuild-then-swap pattern
// `reload_shaders` repeats for every optional pipeline: a `None` field stays
// `None`, and any compile error (the `?`) aborts the whole reload before the
// swap, leaving the live pipelines untouched.
macro_rules! rebuild_if_live {
    ($cond:expr_2021, $build:expr_2021 $(,)?) => {
        if $cond { Some($build?) } else { None }
    };
}

// Static-vertex-layout descriptor used by the velocity / SSAO / SSR pre-pass
// rebuilds during hot-reload. Matches the layout `MtlContext::new` builds at
// init; kept in sync by construction since both touch the 56-byte static
// `Vertex` struct.
fn static_vertex_descriptor() -> Retained<MTLVertexDescriptor> {
    vertex_descriptor(
        &[
            VertexAttr {
                index: 0,
                format: MTLVertexFormat::Float3,
                offset: 0,
                buffer_index: 1,
            },
            VertexAttr {
                index: 1,
                format: MTLVertexFormat::Float3,
                offset: 12,
                buffer_index: 1,
            },
            VertexAttr {
                index: 2,
                format: MTLVertexFormat::Float3,
                offset: 24,
                buffer_index: 1,
            },
            VertexAttr {
                index: 3,
                format: MTLVertexFormat::Float3,
                offset: 36,
                buffer_index: 1,
            },
            VertexAttr {
                index: 4,
                format: MTLVertexFormat::Float2,
                offset: 48,
                buffer_index: 1,
            },
        ],
        &[VertexLayout {
            buffer_index: 1,
            stride: std::mem::size_of::<mesh_payload::Vertex>(),
            step: MTLVertexStepFunction::PerVertex,
        }],
    )
}

impl MtlContext {
    // True when the shared shader-reload flag is set. Cheap atomic load; called
    // at the top of `draw_frame`. Returns false when hot-reload is off so
    // the production path never enters the reload branch.
    pub(super) fn shader_reload_requested(&self) -> bool {
        self.hot_reload
            .reload_pending
            .as_ref()
            .map(|f| f.load(Ordering::SeqCst))
            .unwrap_or(false)
    }

    // Clear the pending-reload flag. Called after `reload_shaders` regardless
    // of outcome so a failed rebuild does not loop forever.
    pub(super) fn clear_shader_reload_flag(&self) {
        if let Some(flag) = &self.hot_reload.reload_pending {
            flag.store(false, Ordering::SeqCst);
        }
    }

    // Rebuild every built-in Metal renderer pipeline from disk-resident source.
    // Each pipeline is constructed into a temporary first; only when every
    // rebuild succeeds does the context atomically swap them in. Any compile
    // or link error logs the underlying message and leaves the live pipelines
    // untouched: a typo in a shader edit won't crash the running session.
    //
    // Covers the main pass and its GPU cull (one builder, since the cull's
    // argument encoder comes from the pipeline it feeds) and the skinned
    // G-buffer pre-pass and skinned shadow variants, which compile from the
    // same single-source files as their static siblings under a different
    // entry. A world Shader's own pair still wins in the main build, so a save
    // to an engine template never swaps a world's program for the engine's.
    pub(super) fn reload_shaders(&mut self) -> RenderResult<()> {
        if !self.hot_reload.enabled {
            return Ok(());
        }
        let device = &self.hw.device;
        let hr = true;

        // Build every replacement into a temporary first. A `?` early-return
        // here means we never overwrite a live pipeline with a failed build:
        // any compile error leaves the running session rendering with the
        // previous shader source.
        let post = build_post_pipeline(device, self.hw.swap_pixel_format, hr)?;
        let bloom = rebuild_if_live!(
            self.bloom_pipelines.is_some(),
            build_bloom_pipelines(device, hr)
        );

        let text = rebuild_if_live!(
            self.text.pipeline_state.is_some(),
            build_text_pipeline(device, self.hw.swap_pixel_format, hr)
        );
        // Pipelines only: nothing here encodes, so the device needs no probe set.
        let post_device = MtlPostDevice {
            device,
            sampler: &self.composite.sampler,
            cube_sampler: &self.scene.cube_sampler,
            probes: None,
            timing: None,
            hot_reload: hr,
        };
        let taa = rebuild_if_live!(
            self.taa.pass.is_some(),
            concinnity_core::render::post::taa::build_pipeline(&post_device)
        );
        // The main pass rebuilds together with the GPU cull and the bindless
        // argument encoders, whose layouts come from the same sources. A world
        // Shader's pair wins here exactly as it does at init, so a save to an
        // engine template does not swap a world's program for the engine's.
        let main = rebuild_if_live!(
            self.cull.main_pipeline.is_some(),
            build_main_pipeline(
                device,
                &make_vertex_descriptor(),
                self.world_shader.as_ref(),
                hr,
                self.targets.hdr.sample_count,
            )
        );
        let main_cull = rebuild_if_live!(main.is_some(), build_cull_pipeline(device, hr));
        // The engine sampler block rides the fresh fragment's encoder.
        let main_sampler_args = rebuild_if_live!(
            main.is_some(),
            build_bindless_sampler_args(
                device,
                hr,
                &self.scene.sampler,
                &self.shadow.sampler,
                &self.scene.cube_sampler,
            )
        );
        // Hi-Z build kernels are engine built-ins (independent of the world
        // shader); rebuild them whenever a Hi-Z resource exists so a saved
        // edit to `hiz_build.hlsl` is picked up. The texture + mip views are
        // kept: only the pipelines swap.
        let hiz_samples = self.targets.hdr.sample_count;
        let hiz = rebuild_if_live!(
            self.cull.hiz.is_some(),
            build_hiz_pipelines(device, hr, hiz_samples)
        );
        let auto_ev = rebuild_if_live!(
            self.auto_exposure.pipelines.is_some(),
            build_auto_exposure_pipelines(device, hr)
        );
        let decal = rebuild_if_live!(
            self.decal.pipeline.is_some(),
            build_decal_pipeline(device, hr)
        );
        let fog = rebuild_if_live!(self.fog.pipeline.is_some(), build_fog_pipeline(device, hr));

        // The shadow pipeline needs the static vertex layout.
        let static_vdesc = static_vertex_descriptor();
        let ssao_kernel = rebuild_if_live!(
            self.ssao.kernel_pipeline.is_some(),
            build_ssao_pipeline(device, &SSAO_KERNEL, hr)
        );
        let ssao_blur = rebuild_if_live!(
            self.ssao.blur_pipeline.is_some(),
            build_ssao_pipeline(device, &SSAO_BLUR, hr)
        );
        // The G-buffer pipeline builds its own two-stream vertex descriptor
        // internally.
        let gbuffer_bindless = rebuild_if_live!(
            self.gbuffer.bindless_pipeline.is_some(),
            build_gbuffer_bindless_pipeline(device, hr)
        );
        let ssr_resolve = rebuild_if_live!(
            self.ssr.resolve.is_some(),
            concinnity_core::render::post::ssr::build_pipeline(&post_device)
        );
        let reflection_composite = rebuild_if_live!(
            self.ssr.composite_pipeline.is_some(),
            build_reflection_composite_pipeline(device, hr)
        );
        let reflection_blur = rebuild_if_live!(
            self.ssr.blur_pipeline.is_some(),
            build_reflection_blur_pipeline(device, hr)
        );
        let ssgi = rebuild_if_live!(
            self.ssgi.pass.is_some(),
            concinnity_core::render::post::ssgi::build_pipelines(&post_device)
        );
        let rt_reflections = rebuild_if_live!(
            self.rt.pipelines.resolve.is_some(),
            build_rt_reflection_pipeline(
                device,
                &crate::metal::builtin_shaders::RT_REFLECTIONS_FRAG,
                hr
            )
        );
        let rt_reflections_textured = rebuild_if_live!(
            self.rt.pipelines.resolve_textured.is_some(),
            build_rt_reflection_pipeline(
                device,
                &crate::metal::builtin_shaders::RT_REFLECTIONS_FRAG_TEXTURED,
                hr
            )
        );

        // The skinned shadow caster rides the 80-byte skinned vertex layout.
        let skinned_vdesc = if self.skinned.shadow_pipeline_state.is_some() {
            Some(make_skinned_vertex_descriptor())
        } else {
            None
        };

        // Shadow pass shaders are engine-internal (compiled from
        // `shadow.metal`), so they rebuild here alongside the other
        // built-ins rather than in `update_default_world_shader`. The static
        // shadow pipeline shares the 56-byte static layout; the skinned one
        // rides the 80-byte skinned layout.
        let shadow = rebuild_if_live!(
            self.shadow.pipeline_state.is_some(),
            build_shadow_pipeline(device, &static_vdesc, hr)
        );
        let skinned_shadow = rebuild_if_live!(
            self.skinned.shadow_pipeline_state.is_some(),
            build_skinned_shadow_pipeline(
                device,
                skinned_vdesc.as_ref().expect("skinned vdesc just built"),
                hr,
            )
        );

        // GPU-driven cascaded-shadow pipelines: the frustum-only shadow
        // decision kernel (from cull.hlsl) + the depth-only bindless shadow
        // render pipeline. Both engine-internal, so they rebuild here. Gated on
        // the live shadow-bindless path.
        let shadow_cull = rebuild_if_live!(
            self.cull.shadow_pipeline.is_some(),
            build_shadow_cull_pipeline(device, hr)
        );
        let shadow_bindless = rebuild_if_live!(
            self.cull.shadow_bindless_pipeline.is_some(),
            build_shadow_bindless_pipeline(device, &static_vdesc, hr)
        );

        // All builds succeeded: swap into the live context. After this
        // point the next frame's draw calls bind the freshly compiled
        // pipelines.
        self.composite.pipeline = post;
        if let Some(b) = bloom {
            self.bloom_pipelines = Some(b);
        }
        if let Some(p) = text {
            self.text.pipeline_state = Some(p);
        }
        if let (Some(p), Some(taa)) = (taa, self.taa.pass.as_mut()) {
            taa.swap_pipeline(p);
        }
        if let (Some(p), Some(cull), Some(sampler_args)) = (main, main_cull, main_sampler_args) {
            self.cull.main_pipeline = Some(p);
            self.arg_buffers.bindless_sampler_args = Some(sampler_args);
            self.cull.pipeline = Some(cull.decide);
            self.cull.pipeline_phase2 = Some(cull.decide_phase2);
            self.cull.encode_pipeline = Some(cull.encode);
            self.cull.icb_arg_encoder = Some(cull.icb_arg_encoder);
            // Force the ICB rebuilds on the next frame so every argument buffer
            // is re-encoded with the arg encoder the new encode kernel produced.
            // The status buffers and the phase-2 ICB are rebuilt by the same
            // `ensure_*_capacity` passes that rebuild the ICBs.
            self.cull.icbs = Vec::new();
            self.cull.icb_arg_buffer = None;
            self.cull.icb_capacity = 0;
            self.cull.icbs_2 = Vec::new();
            self.cull.icb_2_arg_buffer = None;
            self.cull.status_buffer = None;
            self.cull.shadow_icb = None;
            self.cull.shadow_icb_arg_buffer = None;
            self.cull.shadow_status = None;
            self.cull.shadow_icb_capacity = 0;
        }
        if let Some((init_pipeline, downsample_pipeline)) = hiz
            && let Some(h) = self.cull.hiz.as_mut()
        {
            h.swap_pipelines(init_pipeline, downsample_pipeline);
        }
        if let Some(p) = auto_ev {
            self.auto_exposure.pipelines = Some(p);
        }
        if let Some(p) = decal {
            self.decal.pipeline = Some(p);
        }
        if let Some(p) = fog {
            self.fog.pipeline = Some(p);
        }
        if let Some(p) = ssao_kernel {
            self.ssao.kernel_pipeline = Some(p);
        }
        if let Some(p) = ssao_blur {
            self.ssao.blur_pipeline = Some(p);
        }
        if let Some(p) = gbuffer_bindless {
            self.gbuffer.bindless_pipeline = Some(p);
        }
        if let (Some(p), Some(resolve)) = (ssr_resolve, self.ssr.resolve.as_mut()) {
            resolve.swap_pipeline(p);
        }
        if let Some(p) = reflection_composite {
            self.ssr.composite_pipeline = Some(p);
        }
        if let Some(p) = reflection_blur {
            self.ssr.blur_pipeline = Some(p);
        }
        if let (Some(p), Some(pass)) = (ssgi, self.ssgi.pass.as_mut()) {
            pass.swap_pipelines(p);
        }
        if let Some(p) = rt_reflections {
            self.rt.pipelines.resolve = Some(p);
        }
        if let Some(p) = rt_reflections_textured {
            self.rt.pipelines.resolve_textured = Some(p);
        }
        if let Some(p) = shadow {
            self.shadow.pipeline_state = Some(p);
        }
        if let Some(p) = skinned_shadow {
            self.skinned.shadow_pipeline_state = Some(p);
        }
        if let Some(p) = shadow_cull {
            self.cull.shadow_pipeline = Some(p);
        }
        if let Some(p) = shadow_bindless {
            self.cull.shadow_bindless_pipeline = Some(p);
        }
        Ok(())
    }

    // Rebuild the main pipeline from the world default Shader's freshly
    // compiled payload, for [`Self::update_world_shader`] on bucket 0. Mirrors
    // the rebuild-then-swap safety pattern of [`Self::reload_shaders`]: every
    // replacement is constructed into a temporary first, and the swap only
    // runs when every build succeeds, so a typo in a shader edit leaves the
    // live pipelines untouched and the session keeps rendering.
    //
    // Every draw a world Shader reaches goes through the GPU-driven pass, so
    // this is the one pipeline it owns. The shadow and G-buffer pipelines
    // compile from engine-internal source and are covered by
    // [`Self::reload_shaders`].
    pub(super) fn update_default_world_shader(
        &mut self,
        programs: &concinnity_core::components::ShaderPrograms,
    ) -> RenderResult<()> {
        let world = Some(programs);

        // Build everything into temporaries first. Any `?` early-return
        // leaves the live pipelines untouched, mirroring `reload_shaders`.
        // A scene-less world never built a main pipeline; there is nothing
        // for the fresh world-shader programs to replace.
        let vert_desc = make_vertex_descriptor();
        let new_main = if self.cull.main_pipeline.is_some() {
            let hr = self.hot_reload.enabled;
            let device = &self.hw.device;
            let pipeline =
                build_main_pipeline(device, &vert_desc, world, hr, self.targets.hdr.sample_count)?;
            let cull = build_cull_pipeline(device, hr)?;
            // The engine sampler block rides the fresh fragment's encoder; built
            // here (still before the swap) so a failure leaves the live state
            // untouched.
            let sampler_args = build_bindless_sampler_args(
                device,
                hr,
                &self.scene.sampler,
                &self.shadow.sampler,
                &self.scene.cube_sampler,
            )?;
            Some((pipeline, cull, sampler_args))
        } else {
            None
        };

        // All builds succeeded: swap into the live context. After this
        // point the next frame's draw calls bind the freshly compiled
        // pipelines.
        if let Some((pipeline, cull, sampler_args)) = new_main {
            self.cull.main_pipeline = Some(pipeline);
            // Swap the cull state with the pipeline; `two_pass_occlusion` keeps
            // its init-time resolution.
            self.cull.pipeline = Some(cull.decide);
            self.cull.pipeline_phase2 = Some(cull.decide_phase2);
            self.cull.encode_pipeline = Some(cull.encode);
            self.cull.icb_arg_encoder = Some(cull.icb_arg_encoder);
            self.arg_buffers.bindless_sampler_args = Some(sampler_args);
            // Force fresh ICBs on the next frame so every argument buffer is
            // re-encoded with the new encoder. Matches the `cull` swap in
            // `reload_shaders`; the status buffers and phase-2 ICB rebuild
            // alongside.
            self.cull.icbs = Vec::new();
            self.cull.icb_arg_buffer = None;
            self.cull.icb_capacity = 0;
            self.cull.icbs_2 = Vec::new();
            self.cull.icb_2_arg_buffer = None;
            self.cull.status_buffer = None;
            self.cull.shadow_icb = None;
            self.cull.shadow_icb_arg_buffer = None;
            self.cull.shadow_status = None;
            self.cull.shadow_icb_capacity = 0;
        }

        self.world_shader = Some(programs.clone());
        Ok(())
    }
}
