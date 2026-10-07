//! Vulkan shader hot-reload: `VkContext::reload_shaders` rebuilds every live
//! built-in pipeline from the checkout's shader sources. The `reload-shaders`
//! debug command sets the shared flag, and the main thread polls it at the top
//! of `draw_frame`.
//!
//! Entirely a dev-loop concern: the flag exists only when `VkContext::new` is
//! called with `hot_reload = true`. Production `cn run` never sets it. Mirrors
//! `directx/hot_reload.rs` and `metal/hot_reload.rs`.

use ash::vk;
use concinnity_core::render::backend_init;
use concinnity_core::render::error::{RenderError, RenderResult};
use std::sync::atomic::Ordering;

use super::context::VkContext;
use super::pipeline::{
    build_bucket_pipeline, compile_bindless_shaders, compile_composite_shaders,
    compile_cull_shader, compile_cull_shader_phase2, compile_text_shaders,
    create_composite_pipeline, create_text_pipeline,
};
use super::pipeline_desc::compute_pipeline;

// Rebuild a feature's pipeline(s) into a temporary only when `$cond` says the
// feature is live, propagating any compile/create error out of the enclosing
// `reload_shaders`. Expands to `Some(build?)` when live, `None` otherwise. A
// feature whose resources sit behind an `Option` maps over that `Option`
// instead, so its build borrows them directly. Mirrors
// `directx/hot_reload.rs::rebuild_if_live!`.
macro_rules! rebuild_if_live {
    ($cond:expr_2021, $build:expr_2021 $(,)?) => {
        if $cond { Some($build?) } else { None }
    };
}

impl VkContext {
    // True when the shared shader-reload flag is set. Cheap atomic load;
    // called at the top of `draw_frame`. Returns false when hot-reload is
    // off so the production path never enters the reload branch.
    pub(in crate::vulkan) fn shader_reload_requested(&self) -> bool {
        self.hot_reload
            .reload_pending
            .as_ref()
            .map(|f| f.load(Ordering::SeqCst))
            .unwrap_or(false)
    }

    // Clear the pending-reload flag. Called after `reload_shaders`
    // regardless of outcome so a failed rebuild does not loop forever.
    pub(in crate::vulkan) fn clear_shader_reload_flag(&self) {
        if let Some(flag) = &self.hot_reload.reload_pending {
            flag.store(false, Ordering::SeqCst);
        }
    }

    // Rebuild every built-in Vulkan pipeline from disk-resident source.
    // Each pipeline is constructed into a temporary first; only when every
    // rebuild succeeds does the context swap them in (after destroying
    // the displaced ones). Any GLSL compile or pipeline-create failure
    // logs the underlying message and leaves the live pipelines untouched:
    // a typo in a shader edit won't crash the running session.
    //
    // Covers every runtime-bundled pipeline whose source lives in
    // `vulkan/shaders/`: composite, text, bloom (prefilter / downsample /
    // upsample), bindless main (when live), GPU-cull compute, auto-exposure
    // (build + average), projected-decal, volumetric-fog, SSAO (depth copy,
    // kernel, blur), SSR (resolve), the reflection composite (blur, composite),
    // TAA (resolve), the sky, and the G-buffer pre-pass with the sky's motion
    // behind it. The world-loaded main / shadow / instanced / skinned
    // pipelines remain out of scope; same split as DirectX. The caller
    // has already `device_wait_idle`'d so swapping pipelines out from
    // under in-flight command buffers is safe.
    pub(in crate::vulkan) fn reload_shaders(&mut self) -> RenderResult<()> {
        if !self.hot_reload.enabled {
            return Ok(());
        }
        let device = self.hw.device.clone();
        let device = &device;
        let hr = true;

        // Build every replacement into a temporary first. A `?` early-return
        // here means we never overwrite a live pipeline with a failed build:
        // any compile error leaves the running session rendering with the
        // previous shader source.

        // Composite (always live).
        let (composite_vs, composite_ps) = compile_composite_shaders(hr)?;
        let composite_pipeline = create_composite_pipeline(
            device,
            self.composite.render_pass.handle(),
            self.composite.pipeline_layout.handle(),
            &composite_vs,
            &composite_ps,
        )?;

        // Text (only when the world declared text atlases).
        let text_pipeline = rebuild_if_live!(self.text.pipeline.is_some(), {
            let (tv, tf) = compile_text_shaders(hr)?;
            create_text_pipeline(
                device,
                self.composite.render_pass.handle(),
                self.text.pipeline_layout.handle(),
                &tv,
                &tf,
                vk::SampleCountFlags::TYPE_1,
            )
        });

        // Bloom (live from init to teardown, 3 pipelines).
        let bloom_rebuilt = rebuild_if_live!(
            self.bloom.is_some(),
            concinnity_core::render::post::bloom::build_pipelines(&self.post_device(0))
        );

        // Bucket 0 of the GPU-driven main pass, from the engine's freshly
        // compiled pair; a world default Shader's own pair is spliced into the
        // same templates, so it is rebuilt against them too.
        let bindless_main_pipeline = rebuild_if_live!(
            self.cull.bindless_pipeline_layout.is_some() && self.cull.bindless_pipeline.is_some(),
            {
                let engine_pair = compile_bindless_shaders(hr)?;
                let pipeline =
                    self.build_world_main_pipeline(self.world_shader.as_ref(), &engine_pair)?;
                Ok::<_, RenderError>((pipeline, engine_pair))
            }
        );
        let sky_pipeline = crate::vulkan::sky::build_sky_pipeline(
            &self.hw.device,
            self.sky.layout(),
            self.targets.main_render_pass.handle(),
            self.targets.msaa_samples,
            hr,
        )?;
        // The G-buffer pre-pass (when the cull records drive it) and the sky's
        // motion behind it.
        let gbuffer_pipelines = self
            .gbuffer
            .as_ref()
            .map(|gb| {
                let render_pass = gb.prepass_render_pass.handle();
                let prepass = self
                    .cull
                    .gbuffer_bindless_pipeline_layout
                    .as_ref()
                    .filter(|_| self.cull.gbuffer_bindless_pipeline.is_some())
                    .map(|layout| {
                        crate::vulkan::post::gbuffer::build_prepass_pipeline(
                            device,
                            layout.handle(),
                            render_pass,
                            hr,
                        )
                    })
                    .transpose()?;
                let sky = gb.sky.rebuild_pipeline(device, render_pass)?;
                Ok::<_, RenderError>((prepass, sky))
            })
            .transpose()?;
        // The cull kernel, and its phase-2 twin (two-pass occlusion) when built:
        // the same source with the `CULL_PHASE2` define, over the shared layout.
        let cull_pipelines = self
            .cull
            .cull_kernels
            .as_ref()
            .map(|kernels| {
                let layout = kernels.pipeline_layout.handle();
                let pipeline = compute_pipeline(device, layout, &compile_cull_shader(hr)?, "cull")?;
                let phase2 = kernels
                    .pipeline_phase2
                    .as_ref()
                    .map(|_| {
                        compute_pipeline(device, layout, &compile_cull_shader_phase2(hr)?, "cull")
                    })
                    .transpose()?;
                Ok::<_, RenderError>((pipeline, phase2))
            })
            .transpose()?;
        // Hi-Z build kernels (live alongside the cull pipeline).
        let hiz_pipelines = self
            .cull
            .hiz
            .as_ref()
            .map(|hiz| hiz.recompile_pipelines(device, hr))
            .transpose()?;

        // Auto-exposure (gated on the post-process config): the histogram build
        // + average compute pipelines.
        let auto_exposure_pipelines = self
            .auto_exposure
            .resources
            .as_ref()
            .map(|ae| ae.rebuild_pipelines(device, hr))
            .transpose()?;

        // Decal (always built when DecalResources exists, which is
        // unconditional in `VkContext::new`).
        let msaa = self.targets.msaa_samples != vk::SampleCountFlags::TYPE_1;
        let decal_pipeline = self
            .decal
            .resources
            .as_ref()
            .map(|decals| super::decal::rebuild_decal_pipeline(device, decals, msaa, hr))
            .transpose()?;

        // Lines (only once a frame published some and the lazy build ran).
        let line_pipeline = self
            .lines
            .resources
            .as_ref()
            .map(|lines| super::line::rebuild_line_pipeline(device, lines, msaa, hr))
            .transpose()?;

        // Fog (only when the world declared a VolumetricFog). Rebuilds both the
        // fullscreen render pipeline and the froxel-volume compute kernel; the
        // trailing `.map` tuples them into one Result.
        let fog_pipelines = self
            .fog
            .resources
            .as_ref()
            .map(|fog| {
                let render = super::fog::rebuild_fog_pipeline(device, fog, msaa, hr)?;
                super::fog::rebuild_fog_froxel_pipeline(device, fog, hr)
                    .map(|froxel| (render, froxel))
            })
            .transpose()?;

        // SSAO (only when PostProcessConfig opted in). Rebuilds the depth
        // copy, kernel + blur.
        let ssao_rebuilt = rebuild_if_live!(
            self.ssao.is_some(),
            concinnity_core::render::post::ssao::build_pipelines(&self.post_device(0))
        );

        // SSR (only when PostProcessConfig opted in). Rebuilds the resolve.
        let ssr_rebuilt = rebuild_if_live!(
            self.ssr.is_some(),
            concinnity_core::render::post::ssr::build_pipeline(&self.post_device(0))
        );

        // SSGI (only when indirect_lighting: ssgi). Rebuilds its four pipelines.
        let ssgi_rebuilt = rebuild_if_live!(
            self.ssgi.is_some(),
            concinnity_core::render::post::ssgi::build_pipelines(&self.post_device(0))
        );

        // RT reflections (only when the world opted in + the GPU supports it).
        // Rebuilds the flat + textured ray-query pipelines.
        let rt_rebuilt = self
            .rt_reflections
            .as_ref()
            .map(|rt| crate::vulkan::post::rt_reflections::rebuild_rt_pipelines(device, rt, hr))
            .transpose()?;

        // Reflection composite (only when a reflection path owns the scene image).
        // Rebuilds the roughness blur + composite pipelines.
        let reflection_composite_rebuilt = rebuild_if_live!(
            self.reflection_composite.is_some(),
            concinnity_core::render::post::reflection_composite::build_pipelines(
                &self.post_device(0)
            )
        );

        // TAA (only when PostProcessConfig opted in). Rebuilds the resolve
        // pipeline; the velocity channel lives on the unified G-buffer pre-pass.
        let taa_rebuilt = rebuild_if_live!(
            self.taa.is_some(),
            concinnity_core::render::post::taa::build_pipeline(&self.post_device(0))
        );

        // Particles (only when ≥1 emitter is live or has ever been
        // added at runtime). Rebuilds the compute + render pipelines in
        // one shot.
        let particle_rebuilt = self
            .particle
            .resources
            .as_ref()
            .map(|particles| particles.rebuild_pipelines(device, hr))
            .transpose()?;

        // All builds succeeded: swap the freshly compiled pipelines in. Each
        // assignment drops the pipeline it displaces, which retires it through
        // the device's queue rather than destroying it under a submission that
        // may still name it.
        self.composite.pipeline = composite_pipeline;

        if let Some(new_pipeline) = text_pipeline {
            self.text.pipeline = Some(new_pipeline);
        }

        if let (Some(rebuilt), Some(bloom)) = (bloom_rebuilt, self.bloom.as_mut()) {
            bloom.swap_pipelines(rebuilt);
        }

        if let Some((new_pipeline, engine_pair)) = bindless_main_pipeline {
            self.cull.bindless_pipeline = Some(new_pipeline);
            self.cull.bindless_main_spv = engine_pair;
        }
        self.sky.swap_pipeline(sky_pipeline);
        if let (Some((prepass, sky)), Some(gb)) = (gbuffer_pipelines, self.gbuffer.as_mut()) {
            if prepass.is_some() {
                self.cull.gbuffer_bindless_pipeline = prepass;
            }
            gb.sky.swap_pipeline(sky);
        }
        // The wireframe twins were built from the pre-reload shaders; drop them
        // so the next wireframe frame rebuilds against these.
        self.invalidate_wireframe_pipelines();
        if let (Some((pipeline, phase2)), Some(kernels)) =
            (cull_pipelines, self.cull.cull_kernels.as_mut())
        {
            kernels.pipeline = pipeline;
            kernels.pipeline_phase2 = phase2;
        }
        if let (Some((init, downsample)), Some(hiz)) = (hiz_pipelines, self.cull.hiz.as_mut()) {
            hiz.swap_pipelines(init, downsample);
        }

        if let (Some((build, average)), Some(ae)) = (
            auto_exposure_pipelines,
            self.auto_exposure.resources.as_mut(),
        ) {
            ae.swap_pipelines(build, average);
        }

        if let (Some(new_pipeline), Some(decals)) = (decal_pipeline, self.decal.resources.as_mut())
        {
            decals.pipeline = new_pipeline;
        }
        if let (Some(new_pipeline), Some(lines)) = (line_pipeline, self.lines.resources.as_mut()) {
            lines.pipeline = new_pipeline;
        }
        if let (Some((render, froxel)), Some(fog)) = (fog_pipelines, self.fog.resources.as_mut()) {
            fog.pipeline = render;
            fog.froxel_pipeline = froxel;
        }
        if let (Some(rebuilt), Some(ssao)) = (ssao_rebuilt, self.ssao.as_mut()) {
            ssao.swap_pipelines(rebuilt);
        }
        if let (Some(rebuilt), Some(ssr)) = (ssr_rebuilt, self.ssr.as_mut()) {
            ssr.swap_pipeline(rebuilt);
        }
        if let (Some(rebuilt), Some(ssgi)) = (ssgi_rebuilt, self.ssgi.as_mut()) {
            ssgi.swap_pipelines(rebuilt);
        }
        if let (Some(rebuilt), Some(rt)) = (rt_rebuilt, self.rt_reflections.as_mut()) {
            rt.swap_pipelines(rebuilt);
        }
        if let (Some(rebuilt), Some(rc)) = (
            reflection_composite_rebuilt,
            self.reflection_composite.as_mut(),
        ) {
            rc.swap_pipelines(rebuilt);
        }
        if let (Some(rebuilt), Some(taa)) = (taa_rebuilt, self.taa.as_mut()) {
            taa.swap_pipelines(rebuilt);
        }
        if let (Some((cp, rp)), Some(p)) = (particle_rebuilt, self.particle.resources.as_mut()) {
            p.swap_pipelines(cp, rp);
        }
        Ok(())
    }

    // Rebuild bucket 0 of the GPU-driven main pass from the world default
    // Shader's freshly compiled programs and hot-swap it, for
    // `update_world_shader`, or swap in `prepared` when a worker already built
    // it. Mirrors the rebuild-then-swap safety pattern of `reload_shaders`: the
    // replacement is constructed first and the swap only runs when the build
    // succeeds, so a typo in a shader edit leaves the live pipeline untouched
    // and the session keeps rendering.
    pub(in crate::vulkan) fn update_default_world_shader(
        &mut self,
        programs: &concinnity_core::components::ShaderPrograms,
        prepared: Option<crate::vulkan::owned::OwnedPipeline>,
    ) -> RenderResult<()> {
        let new_main = match prepared {
            Some(pipeline) => pipeline,
            None => self.build_world_main_pipeline(Some(programs), &self.cull.bindless_main_spv)?,
        };
        // Drain the GPU before destroying the displaced pipeline so no in-flight
        // command buffer still references it: the debug hot-reload drive does
        // not `wait_idle` for us, unlike the built-in `reload_shaders` path the
        // draw loop guards.
        self.wait_idle();
        self.cull.bindless_pipeline = Some(new_main);
        self.world_shader = Some(programs.clone());
        self.invalidate_wireframe_pipelines();
        Ok(())
    }

    // Bucket 0's pipeline against the live bindless layout: the world default
    // Shader's pair where `world` declares one, `engine_pair` otherwise. Errors
    // when the GPU-driven pass is not live, which means the world has nothing
    // to draw.
    fn build_world_main_pipeline(
        &self,
        world: Option<&concinnity_core::components::ShaderPrograms>,
        engine_pair: &(Vec<u8>, Vec<u8>),
    ) -> RenderResult<crate::vulkan::owned::OwnedPipeline> {
        let targets = self.bucket_pipeline_targets().ok_or_else(|| {
            RenderError::Other("the GPU-driven main pass is not live".to_string())
        })?;
        build_bucket_pipeline(
            &self.hw.device,
            targets,
            0,
            backend_init::WorldShader {
                programs: world,
                deferred: false,
            },
            engine_pair,
        )
    }
}
