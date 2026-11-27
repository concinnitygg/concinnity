//! Metal shader hot-reload: `MtlContext::reload_shaders` rebuilds every live
//! built-in pipeline from the checkout's shader sources. The `reload-shaders`
//! debug command sets the shared flag, and the main thread polls it at the top
//! of `draw_frame`.
//!
//! Each feature's pipelines are one set below: how they are built from the
//! current sources and where they are installed. A reload builds every live
//! set before installing any, so a compile error leaves the running pipelines
//! in place.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::post::reload::{self, PostPipelines};
use concinnity_engine::render::shader::pipeline_set::{self, Gated, PipelineSet, Pipelines};
use concinnity_engine::render::shader::programs::surface::SurfaceVariant;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputePipelineState, MTLPixelFormat, MTLRenderPipelineState};

use super::auto_exposure::{AutoExposurePipelines, build_auto_exposure_pipelines};
use super::context::MtlContext;
use super::cull::{CullPipeline, ShadowViewIcb, build_cull_pipeline, build_shadow_cull_pipeline};
use super::decal::build_decal_pipeline;
use super::fog::build_fog_pipeline;
use super::grass::{GrassPipelines, GrassState};
use super::hiz::{HiZResources, HizPipelines, build_hiz_pipelines};
use super::init::pipelines::{
    BucketBuild, BucketPipelines, build_bindless_sampler_args, build_shadow_bindless_pipeline,
    make_vertex_descriptor, replacement,
};
use super::pipeline::{build_post_pipeline, build_text_pipeline};
use super::post::RtReflectionPipelines;
use super::post::post_device::PostPipeline;
use super::shadow_receivers::{ReceiverPipelines, ShadowReceiversGpu};
use super::view_samples::ViewSamplesGpu;

type RenderState = Retained<ProtocolObject<dyn MTLRenderPipelineState>>;
type ComputeState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

// Every feature's pipeline set, in the order a reload builds them.
const PIPELINE_SETS: &[&dyn PipelineSet<MtlContext>] = &[
    &COMPOSITE,
    &TEXT,
    &ENGINE_POST,
    &MAIN,
    &WORLD_BUCKETS,
    &HIZ,
    &SHADOW_RECEIVERS,
    &VIEW_SAMPLES,
    &AUTO_EXPOSURE,
    &DECAL,
    &FOG,
    &SKY,
    &GRASS,
    &RT_REFLECTIONS,
    &SHADOW_CULL,
    &SHADOW_BINDLESS,
    &FAR_SHADOW_BINDLESS,
];

const COMPOSITE: Pipelines<MtlContext, RenderState> = Pipelines {
    build: |c, hr| {
        Some(build_post_pipeline(
            &c.hw.device,
            c.hw.swap_pixel_format,
            hr,
        ))
    },
    install: |c, p| c.composite.pipeline = p,
};

const TEXT: Pipelines<MtlContext, RenderState> = Pipelines {
    build: |c, hr| {
        c.text
            .pipeline_state
            .is_some()
            .then(|| build_text_pipeline(&c.hw.device, c.hw.swap_pixel_format, hr))
    },
    install: |c, p| c.text.pipeline_state = Some(p),
};

// The engine-owned post passes that are live.
const ENGINE_POST: Pipelines<MtlContext, Vec<PostPipelines<PostPipeline>>> = Pipelines {
    build: |c, hr| {
        Some(reload::build_live(&c.post_pipeline_device(hr), |pass| {
            c.has_post_pass(pass)
        }))
    },
    install: |c, built| {
        for pipelines in built {
            c.install_post_pipelines(pipelines);
        }
    },
};

// The bucket builds from the sources `hot_reload` selects.
fn bucket_build(c: &MtlContext, hot_reload: bool) -> BucketBuild {
    BucketBuild {
        hot_reload,
        ..c.bucket_build()
    }
}

// Bucket 0's main pipelines with the GPU cull and argument encoders, which
// share its sources. A world Shader's pair still wins.
type MainPipelines = (
    BucketPipelines,
    CullPipeline,
    Retained<ProtocolObject<dyn MTLBuffer>>,
);
const MAIN: Pipelines<MtlContext, MainPipelines> = Pipelines {
    build: |c, hr| {
        c.cull.main_pipeline.is_some().then(|| {
            let device = &c.hw.device;
            let main = replacement(
                device,
                0,
                c.world_shader.as_ref(),
                SurfaceVariant::Plain,
                bucket_build(c, hr),
                c.cull.main_pipeline.as_ref(),
                None,
            )?;
            let cull = build_cull_pipeline(device, hr)?;
            // The engine sampler block rides the fresh fragment's encoder.
            let sampler_args = build_bindless_sampler_args(
                device,
                hr,
                &c.scene.sampler,
                &c.shadow.sampler,
                &c.scene.cube_sampler,
            )?;
            Ok((main, cull, sampler_args))
        })
    },
    install: |c, (main, cull, sampler_args)| {
        c.cull.main_pipeline = Some(main);
        c.arg_buffers.bindless_sampler_args = Some(sampler_args);
        c.cull.pipeline = Some(cull.decide);
        c.cull.pipeline_late = Some(cull.decide_late);
        c.cull.scan_pipeline = Some(cull.scan);
        c.cull.scatter_pipeline = Some(cull.scatter);
        c.cull.icb_arg_encoder = Some(cull.icb_arg_encoder);
        // Force the ICB rebuilds on the next frame so every argument buffer is
        // re-encoded with the arg encoder the new scatter produced. The status
        // buffers and the late ICB are rebuilt by the same `ensure_*_capacity`
        // passes that rebuild the ICBs.
        c.cull.icbs = Vec::new();
        c.cull.icb_arg_buffer = None;
        c.cull.icb_capacity = 0;
        c.cull.icbs_late = Vec::new();
        c.cull.icb_late_arg_buffer = None;
        c.cull.status_buffer = None;
        c.cull.shadow_views = ShadowViewIcb::default();
        c.cull.spot_views = ShadowViewIcb::default();
        c.cull.far_view = ShadowViewIcb::default();
    },
};

// Every resident world Shader bucket's pipelines, main and pre-pass together.
const WORLD_BUCKETS: Pipelines<MtlContext, Vec<(usize, BucketPipelines)>> = Pipelines {
    build: |c, hr| Some(c.rebuild_world_buckets(bucket_build(c, hr))),
    install: |c, buckets| {
        for (bucket, pipelines) in buckets {
            if let Some(live) = c.cull.world_pipelines.get_mut(bucket) {
                *live = pipelines;
            }
        }
    },
};

// The Hi-Z build kernels; the pyramid and its views are kept.
const HIZ: Gated<MtlContext, HiZResources, HizPipelines> = Gated {
    state: |c| c.cull.hiz.as_ref(),
    state_mut: |c| c.cull.hiz.as_mut(),
    build: |c, _, hr| build_hiz_pipelines(&c.hw.device, hr, c.targets.hdr.sample_count),
    install: |hiz, (init, downsample)| hiz.swap_pipelines(init, downsample),
};

const SHADOW_RECEIVERS: Gated<MtlContext, ShadowReceiversGpu, ReceiverPipelines> = Gated {
    state: |c| c.shadow_receivers.as_ref(),
    state_mut: |c| c.shadow_receivers.as_mut(),
    build: |c, _, hr| ReceiverPipelines::build(&c.hw.device, hr),
    install: |r, p| r.pipelines = p,
};

const VIEW_SAMPLES: Gated<MtlContext, ViewSamplesGpu, ComputeState> = Gated {
    state: |c| c.view_samples.as_ref(),
    state_mut: |c| c.view_samples.as_mut(),
    build: |c, _, hr| ViewSamplesGpu::build_pipeline(&c.hw.device, hr),
    install: |v, p| v.pipeline = p,
};

const AUTO_EXPOSURE: Pipelines<MtlContext, AutoExposurePipelines> = Pipelines {
    build: |c, hr| {
        c.auto_exposure
            .pipelines
            .is_some()
            .then(|| build_auto_exposure_pipelines(&c.hw.device, hr))
    },
    install: |c, p| c.auto_exposure.pipelines = Some(p),
};

const DECAL: Pipelines<MtlContext, RenderState> = Pipelines {
    build: |c, hr| {
        c.decal
            .pipeline
            .is_some()
            .then(|| build_decal_pipeline(&c.hw.device, hr))
    },
    install: |c, p| c.decal.pipeline = Some(p),
};

const FOG: Pipelines<MtlContext, RenderState> = Pipelines {
    build: |c, hr| {
        c.fog
            .pipeline
            .is_some()
            .then(|| build_fog_pipeline(&c.hw.device, hr))
    },
    install: |c, p| c.fog.pipeline = Some(p),
};

// The background sky and its motion behind the G-buffer pre-pass.
const SKY: Pipelines<MtlContext, (RenderState, RenderState)> = Pipelines {
    build: |c, hr| Some(sky_pipelines(c, hr)),
    install: |c, (sky, velocity)| {
        c.sky.pipeline = sky;
        c.sky.velocity_pipeline = velocity;
    },
};

fn sky_pipelines(c: &MtlContext, hot_reload: bool) -> RenderResult<(RenderState, RenderState)> {
    let device = &c.hw.device;
    let sky = super::sky::build_sky_pipeline(
        device,
        &super::builtin_shaders::SKY_FRAG,
        c.targets.hdr.sample_count,
        hot_reload,
    )?;
    let velocity = super::sky::build_sky_velocity_pipeline(device, hot_reload)?;
    Ok((sky, velocity))
}

const GRASS: Gated<MtlContext, GrassState, GrassPipelines> = Gated {
    state: |c| c.grass.as_ref(),
    state_mut: |c| c.grass.as_mut(),
    build: |c, _, hr| GrassPipelines::build(&c.hw.device, c.targets.hdr.sample_count, hr),
    install: |grass, p| grass.pipelines = p,
};

const RT_REFLECTIONS: Pipelines<MtlContext, RtReflectionPipelines> = Pipelines {
    build: |c, hr| {
        c.rt.pipelines
            .reflections
            .is_some()
            .then(|| RtReflectionPipelines::new(&c.hw.device, hr))
    },
    install: |c, p| c.rt.pipelines.reflections = Some(p),
};

// The GPU-driven shadow path: the frustum-only decision kernel and the
// depth-only bindless draw, for the cascades and the far map.
const SHADOW_CULL: Pipelines<MtlContext, ComputeState> = Pipelines {
    build: |c, hr| {
        c.cull
            .shadow_pipeline
            .is_some()
            .then(|| build_shadow_cull_pipeline(&c.hw.device, hr))
    },
    install: |c, p| c.cull.shadow_pipeline = Some(p),
};

const SHADOW_BINDLESS: Pipelines<MtlContext, RenderState> = Pipelines {
    build: |c, hr| {
        c.cull
            .shadow_bindless_pipeline
            .is_some()
            .then(|| shadow_bindless(c, MTLPixelFormat::Depth32Float, hr))
    },
    install: |c, p| c.cull.shadow_bindless_pipeline = Some(p),
};

const FAR_SHADOW_BINDLESS: Pipelines<MtlContext, RenderState> = Pipelines {
    build: |c, hr| {
        c.cull
            .far_shadow_bindless_pipeline
            .is_some()
            .then(|| shadow_bindless(c, MTLPixelFormat::Depth16Unorm, hr))
    },
    install: |c, p| c.cull.far_shadow_bindless_pipeline = Some(p),
};

// The depth-only bindless shadow draw into `depth_format`, over the static
// vertex layout.
fn shadow_bindless(
    c: &MtlContext,
    depth_format: MTLPixelFormat,
    hot_reload: bool,
) -> RenderResult<RenderState> {
    build_shadow_bindless_pipeline(
        &c.hw.device,
        &make_vertex_descriptor(),
        depth_format,
        hot_reload,
    )
}

impl MtlContext {
    // Rebuild every live pipeline set from the current shader sources.
    pub(super) fn reload_shaders(&mut self) -> RenderResult<()> {
        if !self.hot_reload.enabled {
            return Ok(());
        }
        self.hot_reload.generation += 1;
        pipeline_set::rebuild(self, PIPELINE_SETS, true)
    }

    // Rebuild bucket 0's main and pre-pass pipelines from the world default
    // Shader, or install `prepared`. Built before the swap; the ICBs inherit the
    // pipeline, so they need no re-encode.
    pub(super) fn update_default_world_shader(
        &mut self,
        programs: &concinnity_engine::components::ShaderPrograms,
        prepared: Option<BucketPipelines>,
    ) -> RenderResult<()> {
        // A scene-less world never built a main pipeline; there is nothing for
        // the fresh world-shader programs to replace.
        if self.cull.main_pipeline.is_some() {
            let pipelines = replacement(
                &self.hw.device,
                0,
                Some(programs),
                SurfaceVariant::Plain,
                self.bucket_build(),
                self.cull.main_pipeline.as_ref(),
                prepared,
            )?;
            self.cull.main_pipeline = Some(pipelines);
        }
        self.world_shader = Some(programs.clone());
        Ok(())
    }
}
