//! D3D12 shader hot-reload: `DxContext::reload_shaders` rebuilds every live
//! built-in PSO from the checkout's shader sources. The `reload-shaders` debug
//! command sets the shared flag, and the main thread polls it at the top of
//! `draw_frame`.
//!
//! Entirely a dev-loop concern: the flag exists only when `DxContext::new` is
//! called with `hot_reload = true`. Production `cn run` never sets it. Mirrors
//! `src/metal/hot_reload.rs`.

use concinnity_engine::render::backend::RenderBackend;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::shader::programs::surface::SurfaceVariant;
use concinnity_engine::render::shader::world_pipelines::{
    BucketSource, check_rebuild, replace_bucket,
};

use concinnity_engine::render::post;
use concinnity_engine::render::post::disocclusion::DisocclusionPipelines;
use concinnity_engine::render::post::reflection_composite::ReflectionCompositePipelines;
use concinnity_engine::render::post::ssao::SsaoPipelines;
use concinnity_engine::render::post::ssgi::SsgiPipelines;
use concinnity_engine::render::post::taa::TaaPipelines;

use super::context::DxContext;
use super::init::pipelines::{BucketPipelineTargets, BucketPsos, build_bucket_pipeline};
use super::post::post_device::PostPipeline;
use crate::directx::builtin_shaders::{CULL_EARLY, CULL_LATE, CompileProgram as _};

// Rebuild a feature's PSOs into a temporary when `$cond` says it is live,
// propagating errors: `Some(build?)` when live, `None` otherwise.
macro_rules! rebuild_if_live {
    ($cond:expr_2021, $build:expr_2021 $(,)?) => {
        if $cond { Some($build?) } else { None }
    };
}

// The passes drawn through the shared post-pass seam, each rebuilt only when
// live.
struct SharedPostPipelines {
    ssao: Option<SsaoPipelines<PostPipeline>>,
    ssr: Option<PostPipeline>,
    ssgi: Option<SsgiPipelines<PostPipeline>>,
    disocclusion: Option<DisocclusionPipelines<PostPipeline>>,
    taa: Option<TaaPipelines<PostPipeline>>,
    reflection_composite: Option<ReflectionCompositePipelines<PostPipeline>>,
}

impl DxContext {
    // Rebuild every built-in PSO from source into temporaries, swapping them in
    // only when all succeed; a failure logs and keeps the live pipelines. The
    // shadow PSO is out of scope.
    pub(super) fn reload_shaders(&mut self) -> RenderResult<()> {
        if !self.hot_reload.enabled {
            return Ok(());
        }
        self.hot_reload.generation += 1;
        let device = &self.hw.device;
        let info_queue = self.hw.info_queue.as_ref();
        let hr = true;

        // Build every replacement into a temporary first. A `?` early-return
        // here means we never overwrite a live pipeline with a failed build:
        // any compile error leaves the running session rendering with the
        // previous shader source.

        // Composite (always live).
        let (composite_vs, composite_ps) = super::pipeline::compile_composite_shaders(hr)?;
        let composite_pso = super::context::dump_on_err(
            info_queue,
            super::pipeline::create_composite_pso(
                device,
                &self.composite.root_sig,
                &composite_vs,
                &composite_ps,
                self.swapchain.format,
            ),
        )?;

        // Text (only when the world declared text atlases).
        let text_pso = rebuild_if_live!(self.text.pso.is_some(), {
            let (text_vs, text_ps) = super::pipeline::compile_text_shaders(hr)?;
            super::context::dump_on_err(
                info_queue,
                super::pipeline::create_text_pso(
                    device,
                    &self.text.root_sig,
                    &text_vs,
                    &text_ps,
                    self.swapchain.format,
                    1,
                ),
            )
        });

        // Bloom (live outside a resize).
        let bloom_rebuilt = rebuild_if_live!(
            self.bloom.is_some(),
            post::bloom::build_pipelines(&self.post_device(0))
        );

        // Every shader bucket's main and pre-pass PSOs, rebuilt as pairs so they never
        // compile from different templates.
        let bindless_main_pso = rebuild_if_live!(
            self.cull.main_bindless_root_sig.is_some() && self.cull.main_bindless_pso.is_some(),
            {
                let engine = super::init::pipelines::compile_main_bindless_shaders(hr)?;
                let psos = self.build_world_main_pso(self.world_shader.as_ref(), &engine)?;
                let world_buckets = self.rebuild_world_buckets(&engine)?;
                Ok::<_, RenderError>((psos, world_buckets, engine))
            }
        );
        // The cull PSO, and its late twin (two-pass occlusion) when built,
        // both against the shared root signature.
        let cull_psos = self
            .cull
            .cull_kernels
            .as_ref()
            .map(|kernels| {
                let create = |cs: &[u8]| {
                    super::context::dump_on_err(
                        info_queue,
                        super::cull::create_cull_pso(device, &kernels.root_sig, cs),
                    )
                };
                let pso = create(&CULL_EARLY.compile(hr)?)?;
                let late = kernels
                    .pso_late
                    .as_ref()
                    .map(|_| create(&CULL_LATE.compile(hr)?))
                    .transpose()?;
                Ok::<_, RenderError>((pso, late))
            })
            .transpose()?;

        // Hi-Z (only when cull pipeline is live; same gating condition).
        // Rebuilds all three SPD kernels against the existing root signatures
        // so the live cull root binding stays valid. The tail takes the
        // signature without the depth SRV table.
        let hiz_rebuilt = if let Some(hiz) = self.cull.hiz.as_ref() {
            let (spd_single_cs, spd_msaa_cs, spd_tail_cs) = super::hiz::compile_hiz_shaders(hr)?;
            let spd_single_pso = super::context::dump_on_err(
                info_queue,
                super::pso::compute_pso(device, &hiz.root_sig, &spd_single_cs, "hiz spd_single"),
            )?;
            let spd_msaa_pso = super::context::dump_on_err(
                info_queue,
                super::pso::compute_pso(device, &hiz.root_sig, &spd_msaa_cs, "hiz spd_msaa"),
            )?;
            let spd_tail_pso = super::context::dump_on_err(
                info_queue,
                super::pso::compute_pso(device, &hiz.tail_root_sig, &spd_tail_cs, "hiz spd_tail"),
            )?;
            Some((spd_single_pso, spd_msaa_pso, spd_tail_pso))
        } else {
            None
        };

        // Auto-exposure (gated on the post-process config).
        let (auto_exp_build, auto_exp_average) = if let Some(ae) =
            self.auto_exposure.resources.as_ref()
        {
            let (build_cs, average_cs) = super::auto_exposure::compile_auto_exposure_shaders(hr)?;
            let build_pso = super::context::dump_on_err(
                info_queue,
                super::pso::compute_pso(
                    device,
                    ae.build_root_sig(),
                    &build_cs,
                    "auto-exposure build",
                ),
            )?;
            let average_pso = super::context::dump_on_err(
                info_queue,
                super::pso::compute_pso(
                    device,
                    ae.average_root_sig(),
                    &average_cs,
                    "auto-exposure average",
                ),
            )?;
            (Some(build_pso), Some(average_pso))
        } else {
            (None, None)
        };

        let receiver_psos = self
            .shadow_receivers
            .as_ref()
            .map(|r| super::context::dump_on_err(info_queue, r.rebuild_pipelines(device, hr)))
            .transpose()?;
        let view_samples_pso = self
            .view_samples
            .as_ref()
            .map(|v| super::context::dump_on_err(info_queue, v.rebuild_pipeline(device, hr)))
            .transpose()?;

        // Decal (always built when DecalResources exists, which is unconditional).
        let msaa_samples = self.targets.hdr.msaa_samples;
        let sky_pso = super::context::dump_on_err(
            info_queue,
            super::sky::build_sky_pso(
                device,
                &super::builtin_shaders::SKY_FRAG,
                self.sky.root_sig(),
                msaa_samples,
                hr,
            ),
        )?;
        if let Some(atmosphere) = self.atmosphere.as_mut() {
            atmosphere.rebuild_pipelines(device, msaa_samples)?;
        }
        // The sky's motion behind the G-buffer pre-pass; the pre-pass's own
        // PSOs rebuild with the main pass's above.
        let gbuffer_sky_pso = self
            .gbuffer
            .as_ref()
            .map(|gb| gb.sky.rebuild_pso(device, info_queue))
            .transpose()?;
        let grass = match (
            self.grass.as_ref(),
            self.cull.main_bindless_root_sig.as_ref(),
            self.cull.prepass_root_sig.as_ref(),
        ) {
            (Some(_), Some(main), Some(prepass)) => Some(super::context::dump_on_err(
                info_queue,
                super::grass::GrassPipelines::build(
                    device,
                    super::grass::GrassRootSigs { main, prepass },
                    msaa_samples,
                    hr,
                ),
            )?),
            _ => None,
        };
        let decal_pso = self
            .decal
            .state
            .as_ref()
            .map(|decals| {
                super::decal::rebuild_decal_pso(
                    device,
                    &decals.root_sig,
                    msaa_samples,
                    hr,
                    info_queue,
                )
            })
            .transpose()?;

        // Lines (only once a frame published some and the lazy build ran).
        let line_pso = self
            .lines
            .resources
            .as_ref()
            .map(|lines| super::line::rebuild_line_pso(device, lines, msaa_samples, hr, info_queue))
            .transpose()?;

        // The transparent pass's two producers (each only when the world declared
        // its asset). Both rebuild against the pass's shared root signature.
        let glass_pso = self
            .transparent
            .as_ref()
            .filter(|t| t.has_glass())
            .map(|t| {
                super::glass::rebuild_glass_pso(device, t.root_sig(), msaa_samples, hr, info_queue)
            })
            .transpose()?;
        let water_pso = self
            .transparent
            .as_ref()
            .filter(|t| t.has_water())
            .map(|t| {
                super::water::rebuild_water_pso(device, t.root_sig(), msaa_samples, hr, info_queue)
            })
            .transpose()?;

        // Fog (only when the world declared a VolumetricFog). Both the
        // render PSO (fragment volume sampler) and the compute PSO
        // (froxel-volume kernel) rebuild from the same `fog.metal`-style
        // source pair.
        let fog_psos = self
            .fog
            .resources
            .as_ref()
            .map(|fog| {
                let render = super::fog::rebuild_fog_pso(
                    device,
                    &fog.root_sig,
                    msaa_samples,
                    hr,
                    info_queue,
                )?;
                super::fog::rebuild_fog_froxel_pso(device, &fog.froxel_root_sig, hr, info_queue)
                    .map(|froxel| (render, froxel))
            })
            .transpose()?;

        let shared_post = self.build_shared_post_pipelines()?;

        // RT reflections (only when DXR + DXC compile + accel build all succeeded
        // at init). The shader compiles through DXC (SM 6.5).
        let rt_rebuilt = self
            .rt_reflections
            .as_ref()
            .map(|rt| {
                super::post::rt_reflections::rebuild_rt_reflections_pipelines(
                    device, rt, hr, info_queue,
                )
            })
            .transpose()?;

        // All builds succeeded; swap into the live context. After this
        // point the next frame's draw calls bind the freshly compiled
        // pipelines.
        self.composite.pso = composite_pso;
        if let Some(p) = text_pso {
            self.text.pso = Some(p);
        }
        if let (Some(rebuilt), Some(bloom)) = (bloom_rebuilt, self.bloom.as_mut()) {
            bloom.swap_pipelines(rebuilt);
        }
        if let Some((psos, world_buckets, engine)) = bindless_main_pso {
            self.cull.main_bindless_pso = Some(psos.main);
            self.cull.main_prepass_pso = psos.prepass;
            self.swap_world_buckets(world_buckets);
            self.cull.bindless_main_shaders = engine;
        }
        self.sky.swap_pso(sky_pso);
        if let (Some(sky), Some(gb)) = (gbuffer_sky_pso, self.gbuffer.as_mut()) {
            gb.sky.swap_pso(sky);
        }
        if let (Some(p), Some(grass)) = (grass, self.grass.as_mut()) {
            grass.pipelines = p;
        }
        // The wireframe twins were built from the pre-reload shaders; drop them
        // so the next wireframe frame rebuilds against these.
        self.invalidate_wireframe_pipelines();
        if let (Some((pso, late)), Some(kernels)) = (cull_psos, self.cull.cull_kernels.as_mut()) {
            kernels.pso = pso;
            kernels.pso_late = late;
        }
        if let (Some((single, msaa, tail)), Some(hiz)) = (hiz_rebuilt, self.cull.hiz.as_mut()) {
            hiz.swap_pipelines(single, msaa, tail);
        }
        if let (Some(build), Some(average), Some(ae)) = (
            auto_exp_build,
            auto_exp_average,
            self.auto_exposure.resources.as_mut(),
        ) {
            ae.swap_pipelines(build, average);
        }
        if let (Some(p), Some(r)) = (receiver_psos, self.shadow_receivers.as_mut()) {
            r.swap_pipelines(p);
        }
        if let (Some(p), Some(v)) = (view_samples_pso, self.view_samples.as_mut()) {
            v.swap_pipeline(p);
        }
        if let (Some(pso), Some(decals)) = (decal_pso, self.decal.state.as_mut()) {
            decals.pso = pso;
        }
        if let (Some(pso), Some(lines)) = (line_pso, self.lines.resources.as_mut()) {
            lines.pso = pso;
        }
        if let Some(transparent) = self.transparent.as_mut() {
            transparent.swap_pipelines(glass_pso, water_pso);
        }
        if let (Some((render, froxel)), Some(fog)) = (fog_psos, self.fog.resources.as_mut()) {
            fog.pso = render;
            fog.froxel_pso = froxel;
        }
        self.swap_shared_post_pipelines(shared_post);
        if let (Some(rebuilt), Some(rt)) = (rt_rebuilt, self.rt_reflections.as_mut()) {
            super::post::rt_reflections::swap_rt_reflections_pipelines(rt, rebuilt);
        }
        Ok(())
    }

    // Rebuild every live pass drawn through the shared post-pass seam.
    fn build_shared_post_pipelines(&self) -> RenderResult<SharedPostPipelines> {
        let device = self.post_device(0);
        Ok(SharedPostPipelines {
            ssao: rebuild_if_live!(
                self.ssao.resources.is_some(),
                post::ssao::build_pipelines(&device)
            ),
            // Only when the resolve itself is authored.
            ssr: rebuild_if_live!(
                self.ssr.as_ref().is_some_and(|s| s.resolve.is_some()),
                post::ssr::build_pipeline(&device)
            ),
            ssgi: rebuild_if_live!(self.ssgi.is_some(), post::ssgi::build_pipelines(&device)),
            disocclusion: rebuild_if_live!(
                self.disocclusion.is_some(),
                post::disocclusion::build_pipelines(&device)
            ),
            taa: rebuild_if_live!(self.taa.is_some(), post::taa::build_pipelines(&device)),
            reflection_composite: rebuild_if_live!(
                self.reflection_composite.is_some(),
                post::reflection_composite::build_pipelines(&device)
            ),
        })
    }

    // Swap the rebuilt shared post passes into the live context.
    fn swap_shared_post_pipelines(&mut self, rebuilt: SharedPostPipelines) {
        if let (Some(p), Some(ssao)) = (rebuilt.ssao, self.ssao.resources.as_mut()) {
            ssao.swap_pipelines(p);
        }
        if let (Some(p), Some(ssr)) = (rebuilt.ssr, self.ssr.as_mut()) {
            ssr.swap_pipeline(p);
        }
        if let (Some(p), Some(ssgi)) = (rebuilt.ssgi, self.ssgi.as_mut()) {
            ssgi.swap_pipelines(p);
        }
        if let (Some(p), Some(pass)) = (rebuilt.disocclusion, self.disocclusion.as_mut()) {
            pass.swap_pipelines(p);
        }
        if let (Some(p), Some(rc)) = (
            rebuilt.reflection_composite,
            self.reflection_composite.as_mut(),
        ) {
            rc.swap_pipelines(p);
        }
        if let (Some(p), Some(taa)) = (rebuilt.taa, self.taa.as_mut()) {
            taa.pass.swap_pipelines(p);
        }
    }
}

// World-Shader runtime hot-swap (RenderBackend::update_world_shader)

// cn-debug-only runtime-mutation surface; dead from the FFI lib crate's roots,
// live in the concinnity binary. See the note on the analogous block in
// [directx/particle.rs].
impl DxContext {
    // Rebuild bucket 0's main and pre-pass PSOs from the world default Shader, or
    // install `prepared`. Built first, swapped only on success.
    pub(in crate::directx) fn update_default_world_shader(
        &mut self,
        programs: &concinnity_engine::components::ShaderPrograms,
        prepared: Option<BucketPsos>,
    ) -> RenderResult<()> {
        let new = replace_bucket(
            0,
            self.cull.main_prepass_pso.is_some(),
            prepared,
            |p: &BucketPsos| p.prepass.is_some(),
            || self.build_world_main_pso(Some(programs), &self.cull.bindless_main_shaders),
        )?;
        // Drain the GPU before the swap releases the displaced PSOs: a command
        // list does not keep one alive, and the debug reload drive does not
        // wait for us.
        self.wait_idle();
        self.cull.main_bindless_pso = Some(new.main);
        self.cull.main_prepass_pso = new.prepass;
        self.world_shader = Some(programs.clone());
        self.invalidate_wireframe_pipelines();
        Ok(())
    }

    // Bucket 0's PSOs: the world default Shader's programs, else `engine_default`.
    // Errors when the GPU-driven pass is not live, or the rebuild would drop the
    // live pre-pass.
    pub(super) fn build_world_main_pso(
        &self,
        world: Option<&concinnity_engine::components::ShaderPrograms>,
        engine_default: &super::init::pipelines::BindlessMainShaders,
    ) -> RenderResult<BucketPsos> {
        let targets = self
            .world_pso_targets()
            .ok_or_else(|| RenderError::Other("the GPU-driven main pass is not live".into()))?;
        let psos = build_bucket_pipeline(
            &self.hw.device,
            self.hw.info_queue.as_ref(),
            BucketPipelineTargets {
                root_sigs: &targets.root_sigs,
                msaa_samples: self.targets.hdr.msaa_samples,
                engine_default,
                hot_reload: self.hot_reload.enabled,
            },
            0,
            &BucketSource {
                programs: world,
                deferred: false,
                variant: SurfaceVariant::Plain,
            },
        )?;
        check_rebuild(
            0,
            self.cull.main_prepass_pso.is_some(),
            psos.prepass.is_some(),
        )?;
        Ok(psos)
    }
}
