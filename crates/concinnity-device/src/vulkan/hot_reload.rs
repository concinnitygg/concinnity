//! Vulkan shader hot-reload: `VkContext::reload_shaders` rebuilds every live
//! built-in pipeline from the checkout's shader sources. The `reload-shaders`
//! debug command sets the shared flag, and the main thread polls it at the top
//! of `draw_frame`.
//!
//! Each feature's pipelines are one set below: how they are built from the
//! current sources and where they are installed. A reload builds every live
//! set before installing any, so a compile error leaves the running pipelines
//! in place.

use concinnity_engine::render::backend::RenderBackend;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::post::reload::{self, PostPipelines};
use concinnity_engine::render::shader::pipeline_set::{self, Gated, PipelineSet, Pipelines};
use concinnity_engine::render::shader::programs::surface::SurfaceVariant;
use concinnity_engine::render::shader::world_pipelines::{
    BucketSource, check_rebuild, replace_bucket,
};

use super::atmosphere::{AtmospherePipelines, AtmosphereSky, AtmosphereState};
use super::auto_exposure::AutoExposureResources;
use super::context::{CullKernels, VkContext};
use super::decal::DecalResources;
use super::fog::FogResources;
use super::grass::{GrassDrawPipelines, GrassPipelines, GrassResources};
use super::hiz::HiZResources;
use super::line::LineResources;
use super::owned::OwnedPipeline;
use super::particle::{ParticlePipelines, ParticleResources};
use super::pipeline::{
    BindlessSpv, BucketPipelineTargets, BucketPipelines, build_bucket_pipeline,
    compile_bindless_shaders, compile_composite_shaders, compile_text_shaders,
    create_composite_pipeline, create_text_pipeline,
};
use super::pipeline_desc::compute_pipeline;
use super::post::GbufferResources;
use super::post::RtReflectionsResources;
use super::post::post_device::{PostPipeline, VkPostDevice};
use super::post::rt_reflections::{RebuiltRtPipelines, rebuild_rt_pipelines};
use super::shadow_receivers::ShadowReceiversGpu;
use super::view_samples::ViewSamplesGpu;
use crate::vulkan::builtin_shaders::{CULL_EARLY, CULL_LATE, CompileProgram as _};

// Every feature's pipeline set, in the order a reload builds them.
const PIPELINE_SETS: &[&dyn PipelineSet<VkContext>] = &[
    &COMPOSITE,
    &TEXT,
    &ENGINE_POST,
    &MAIN,
    &SKY,
    &GBUFFER_SKY,
    &CULL,
    &HIZ,
    &AUTO_EXPOSURE,
    &SHADOW_RECEIVERS,
    &VIEW_SAMPLES,
    &DECAL,
    &LINES,
    &FOG,
    &ATMOSPHERE,
    &RT_REFLECTIONS,
    &PARTICLES,
    &GRASS,
];

const COMPOSITE: Pipelines<VkContext, OwnedPipeline> = Pipelines {
    build: |c, hr| {
        Some(compile_composite_shaders(hr).and_then(|(vs, ps)| {
            create_composite_pipeline(
                &c.hw.device,
                c.composite.formats,
                c.composite.pipeline_layout.handle(),
                &vs,
                &ps,
            )
        }))
    },
    install: |c, p| c.composite.pipeline = p,
};

const TEXT: Pipelines<VkContext, OwnedPipeline> = Pipelines {
    build: |c, hr| {
        c.text.pipeline.is_some().then(|| {
            let (vs, fs) = compile_text_shaders(hr)?;
            create_text_pipeline(
                &c.hw.device,
                c.composite.formats,
                c.text.pipeline_layout.handle(),
                &vs,
                &fs,
                ash::vk::SampleCountFlags::TYPE_1,
            )
        })
    },
    install: |c, p| c.text.pipeline = Some(p),
};

// The engine-owned post passes that are live.
const ENGINE_POST: Pipelines<VkContext, Vec<PostPipelines<PostPipeline>>> = Pipelines {
    build: |c, hr| {
        let device = VkPostDevice {
            hot_reload: hr,
            ..c.post_device(0)
        };
        Some(reload::build_live(&device, |pass| c.has_post_pass(pass)))
    },
    install: |c, built| {
        for pipelines in built {
            c.install_post_pipelines(pipelines);
        }
    },
};

// Every shader bucket's main and pre-pass pipelines, rebuilt as pairs so they
// never compile from different templates.
type MainPipelines = (BucketPipelines, Vec<(usize, BucketPipelines)>, BindlessSpv);
const MAIN: Pipelines<VkContext, MainPipelines> = Pipelines {
    build: |c, hr| {
        (c.cull.bindless_pipeline_layout.is_some() && c.cull.bindless_pipeline.is_some()).then(
            || {
                let engine = compile_bindless_shaders(hr)?;
                let main = c.build_world_main_pipeline(c.world_shader.as_ref(), &engine, hr)?;
                let world_buckets = c.rebuild_world_buckets(&engine, hr)?;
                Ok((main, world_buckets, engine))
            },
        )
    },
    install: |c, (main, world_buckets, engine)| {
        c.cull.bindless_pipeline = Some(main.main);
        c.cull.prepass_pipeline = main.prepass;
        for (bucket, fresh) in world_buckets {
            if let Some(live) = c.cull.world_pipelines.get_mut(bucket) {
                *live = fresh;
            }
        }
        c.cull.bindless_main_spv = engine;
        // The wireframe twins were built from the old sources; the next
        // wireframe frame rebuilds them.
        c.invalidate_wireframe_pipelines();
    },
};

const SKY: Pipelines<VkContext, OwnedPipeline> = Pipelines {
    build: |c, hr| {
        Some(super::sky::build_sky_pipeline(
            &c.hw.device,
            &super::builtin_shaders::SKY_FRAG,
            c.sky.layout(),
            c.targets.msaa_samples,
            hr,
        ))
    },
    install: |c, p| c.sky.swap_pipeline(p),
};

// The sky's motion behind the G-buffer pre-pass.
const GBUFFER_SKY: Gated<VkContext, GbufferResources, OwnedPipeline> = Gated {
    state: |c| c.gbuffer.as_ref(),
    state_mut: |c| c.gbuffer.as_mut(),
    build: |c, gb, hr| gb.sky.rebuild_pipeline(&c.hw.device, hr),
    install: |gb, p| gb.sky.swap_pipeline(p),
};

// The cull kernel, and its late twin (two-pass occlusion) when built: the
// same source with the `CULL_LATE` define, over the shared layout.
const CULL: Gated<VkContext, CullKernels, (OwnedPipeline, Option<OwnedPipeline>)> = Gated {
    state: |c| c.cull.cull_kernels.as_ref(),
    state_mut: |c| c.cull.cull_kernels.as_mut(),
    build: |c, kernels, hr| {
        let (device, layout) = (&c.hw.device, kernels.pipeline_layout.handle());
        let pipeline = compute_pipeline(device, layout, &CULL_EARLY.compile(hr)?, "cull")?;
        let late = kernels
            .pipeline_late
            .as_ref()
            .map(|_| compute_pipeline(device, layout, &CULL_LATE.compile(hr)?, "cull"))
            .transpose()?;
        Ok((pipeline, late))
    },
    install: |kernels, (pipeline, late)| {
        kernels.pipeline = pipeline;
        kernels.pipeline_late = late;
    },
};

const HIZ: Gated<VkContext, HiZResources, (OwnedPipeline, OwnedPipeline)> = Gated {
    state: |c| c.cull.hiz.as_ref(),
    state_mut: |c| c.cull.hiz.as_mut(),
    build: |c, hiz, hr| hiz.recompile_pipelines(&c.hw.device, hr),
    install: |hiz, (spd, tail)| hiz.swap_pipelines(spd, tail),
};

const AUTO_EXPOSURE: Gated<VkContext, AutoExposureResources, (OwnedPipeline, OwnedPipeline)> =
    Gated {
        state: |c| c.auto_exposure.resources.as_ref(),
        state_mut: |c| c.auto_exposure.resources.as_mut(),
        build: |c, ae, hr| ae.rebuild_pipelines(&c.hw.device, hr),
        install: |ae, (build, average)| ae.swap_pipelines(build, average),
    };

const SHADOW_RECEIVERS: Gated<VkContext, ShadowReceiversGpu, (OwnedPipeline, OwnedPipeline)> =
    Gated {
        state: |c| c.shadow_receivers.as_ref(),
        state_mut: |c| c.shadow_receivers.as_mut(),
        build: |c, r, hr| r.rebuild_pipelines(&c.hw.device, hr),
        install: |r, p| r.swap_pipelines(p),
    };

const VIEW_SAMPLES: Gated<VkContext, ViewSamplesGpu, OwnedPipeline> = Gated {
    state: |c| c.view_samples.as_ref(),
    state_mut: |c| c.view_samples.as_mut(),
    build: |c, v, hr| v.rebuild_pipeline(&c.hw.device, hr),
    install: |v, p| v.swap_pipeline(p),
};

const DECAL: Gated<VkContext, DecalResources, OwnedPipeline> = Gated {
    state: |c| c.decal.resources.as_ref(),
    state_mut: |c| c.decal.resources.as_mut(),
    build: |c, decals, hr| {
        super::decal::rebuild_decal_pipeline(&c.hw.device, decals, c.main_msaa(), hr)
    },
    install: |decals, p| decals.pipeline = p,
};

// Built once a frame published lines.
const LINES: Gated<VkContext, LineResources, OwnedPipeline> = Gated {
    state: |c| c.lines.resources.as_ref(),
    state_mut: |c| c.lines.resources.as_mut(),
    build: |c, lines, hr| {
        super::line::rebuild_line_pipeline(&c.hw.device, lines, c.main_msaa(), hr)
    },
    install: |lines, p| lines.pipeline = p,
};

// The fullscreen fog draw and the froxel-volume kernel.
const FOG: Gated<VkContext, FogResources, (OwnedPipeline, OwnedPipeline)> = Gated {
    state: |c| c.fog.resources.as_ref(),
    state_mut: |c| c.fog.resources.as_mut(),
    build: |c, fog, hr| {
        let device = &c.hw.device;
        let render = super::fog::rebuild_fog_pipeline(device, fog, c.main_msaa(), hr)?;
        let froxel = super::fog::rebuild_fog_froxel_pipeline(device, fog, hr)?;
        Ok((render, froxel))
    },
    install: |fog, (render, froxel)| {
        fog.pipeline = render;
        fog.froxel_pipeline = froxel;
    },
};

const ATMOSPHERE: Gated<VkContext, AtmosphereState, AtmospherePipelines> = Gated {
    state: |c| c.atmosphere.as_ref(),
    state_mut: |c| c.atmosphere.as_mut(),
    build: |c, atmosphere, hr| {
        let sky = AtmosphereSky {
            global_set_layout: c.descriptors.global_set_layout.handle(),
            samples: c.targets.msaa_samples,
        };
        atmosphere.build_pipelines(&c.hw.device, sky, c.main_msaa(), hr)
    },
    install: AtmosphereState::swap_pipelines,
};

const RT_REFLECTIONS: Gated<VkContext, RtReflectionsResources, RebuiltRtPipelines> = Gated {
    state: |c| c.rt_reflections.as_ref(),
    state_mut: |c| c.rt_reflections.as_mut(),
    build: |c, rt, hr| rebuild_rt_pipelines(&c.hw.device, rt, hr),
    install: RtReflectionsResources::swap_pipelines,
};

const PARTICLES: Gated<VkContext, ParticleResources, ParticlePipelines> = Gated {
    state: |c| c.particle.resources.as_ref(),
    state_mut: |c| c.particle.resources.as_mut(),
    build: |c, particles, hr| particles.rebuild_pipelines(&c.hw.device, hr),
    install: ParticleResources::swap_pipelines,
};

// The blade kernels and draws, and the G-buffer pre-pass draws when the world
// has a G-buffer.
type GrassBuild = (GrassPipelines, Option<GrassDrawPipelines>);
const GRASS: Gated<VkContext, GrassResources, GrassBuild> = Gated {
    state: |c| c.grass.as_ref(),
    state_mut: |c| c.grass.as_mut(),
    build: |c, grass, hr| {
        let device = &c.hw.device;
        let pipelines = grass.rebuild_pipelines(device, c.grass_targets(), hr)?;
        let prepass = c
            .gbuffer
            .is_some()
            .then(|| grass.build_prepass(device, hr))
            .transpose()?;
        Ok((pipelines, prepass))
    },
    install: |grass, (pipelines, prepass)| {
        grass.swap_pipelines(pipelines);
        if let Some(prepass) = prepass {
            grass.swap_prepass(prepass);
        }
    },
};

impl VkContext {
    // Rebuild every live pipeline set from the current shader sources. The
    // device is already idle.
    pub(in crate::vulkan) fn reload_shaders(&mut self) -> RenderResult<()> {
        if !self.hot_reload.enabled {
            return Ok(());
        }
        self.hot_reload.generation += 1;
        pipeline_set::rebuild(self, PIPELINE_SETS, true)
    }

    // Rebuild bucket 0's main and pre-pass pipelines from the world default
    // Shader, or install `prepared`. Built first, swapped only on success.
    pub(in crate::vulkan) fn update_default_world_shader(
        &mut self,
        programs: &concinnity_engine::components::ShaderPrograms,
        prepared: Option<BucketPipelines>,
    ) -> RenderResult<()> {
        let new = replace_bucket(
            0,
            self.cull.prepass_pipeline.is_some(),
            prepared,
            |p: &BucketPipelines| p.prepass.is_some(),
            || {
                let engine = &self.cull.bindless_main_spv;
                self.build_world_main_pipeline(Some(programs), engine, self.hot_reload.enabled)
            },
        )?;
        // Drain the GPU before destroying the displaced pipeline so no in-flight
        // command buffer still references it: the debug hot-reload drive does
        // not `wait_idle` for us, unlike the built-in `reload_shaders` path the
        // draw loop guards.
        self.wait_idle();
        self.cull.bindless_pipeline = Some(new.main);
        self.cull.prepass_pipeline = new.prepass;
        self.world_shader = Some(programs.clone());
        self.invalidate_wireframe_pipelines();
        Ok(())
    }

    // Bucket 0's pipelines: the world default Shader's programs, else `engine`,
    // from the sources `hot_reload` selects. Errors when the GPU-driven pass is
    // not live, or the rebuild would drop the live pre-pass.
    fn build_world_main_pipeline(
        &self,
        world: Option<&concinnity_engine::components::ShaderPrograms>,
        engine: &BindlessSpv,
        hot_reload: bool,
    ) -> RenderResult<BucketPipelines> {
        let targets = self.bucket_pipeline_targets().ok_or_else(|| {
            RenderError::Other("the GPU-driven main pass is not live".to_string())
        })?;
        let targets = BucketPipelineTargets {
            hot_reload,
            ..targets
        };
        let pipelines = build_bucket_pipeline(
            &self.hw.device,
            targets,
            0,
            &BucketSource {
                programs: world,
                deferred: false,
                variant: SurfaceVariant::Plain,
            },
            engine,
        )?;
        let live_prepass = self.cull.prepass_pipeline.is_some();
        check_rebuild(0, live_prepass, pipelines.prepass.is_some())?;
        Ok(pipelines)
    }
}
