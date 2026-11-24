//! Metal shader hot-reload: `MtlContext::reload_shaders` rebuilds every live
//! built-in pipeline from the checkout's shader sources. The `reload-shaders`
//! debug command sets the shared flag, and the main thread polls it at the top
//! of `draw_frame`.
//!
//! Entirely a dev-loop concern: the flag exists only when `MtlContext::new` is
//! called with `hot_reload = true`. Production `cn run` never sets it.
#![deny(unsafe_op_in_unsafe_fn)]

use crate::metal::cull::ShadowViewIcb;
use concinnity_engine::gfx::mesh_payload;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::shader::programs::surface::SurfaceVariant;
use objc2::rc::Retained;
use objc2_metal::{MTLPixelFormat, MTLVertexDescriptor, MTLVertexFormat, MTLVertexStepFunction};

use super::auto_exposure::build_auto_exposure_pipelines;
use super::context::MtlContext;
use super::cull::{build_cull_pipeline, build_shadow_cull_pipeline};
use super::decal::build_decal_pipeline;
use super::descriptors::{VertexAttr, VertexLayout, vertex_descriptor};
use super::fog::build_fog_pipeline;
use super::hiz::build_hiz_pipelines;
use super::init::pipelines::{
    BucketBuild, BucketPipelines, build_bindless_sampler_args, build_shadow_bindless_pipeline,
    replacement,
};
use super::pipeline::{build_post_pipeline, build_text_pipeline};
use super::post::RtReflectionPipelines;
use super::post::post_device::MtlPostDevice;

// Rebuild a pipeline into a temporary when `$cond` says it is live; an error
// aborts the reload before the swap.
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
            stride: size_of::<mesh_payload::Vertex>(),
            step: MTLVertexStepFunction::PerVertex,
        }],
    )
}

impl MtlContext {
    // Rebuild every built-in pipeline from source into temporaries, swapping
    // them in only when all succeed; a failure logs and keeps the live
    // pipelines.
    pub(super) fn reload_shaders(&mut self) -> RenderResult<()> {
        if !self.hot_reload.enabled {
            return Ok(());
        }
        self.hot_reload.generation += 1;
        let device = &self.hw.device;
        let hr = true;

        // Build every replacement into a temporary first. A `?` early-return
        // here means we never overwrite a live pipeline with a failed build:
        // any compile error leaves the running session rendering with the
        // previous shader source.
        let post = build_post_pipeline(device, self.hw.swap_pixel_format, hr)?;
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
            concinnity_engine::render::post::taa::build_pipelines(&post_device)
        );
        let bloom = rebuild_if_live!(
            self.bloom.is_some(),
            concinnity_engine::render::post::bloom::build_pipelines(&post_device)
        );
        // The main pass rebuilds with the GPU cull and argument encoders, which
        // share its sources. A world Shader's pair still wins, and each bucket's
        // main and pre-pass rebuild together.
        let build = BucketBuild {
            hot_reload: hr,
            ..self.bucket_build()
        };
        let main = rebuild_if_live!(
            self.cull.main_pipeline.is_some(),
            replacement(
                device,
                0,
                self.world_shader.as_ref(),
                SurfaceVariant::Plain,
                build,
                self.cull.main_pipeline.as_ref(),
                None,
            )
        );
        let world_buckets = self.rebuild_world_buckets(build)?;
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
        let receivers = rebuild_if_live!(
            self.shadow_receivers.is_some(),
            super::shadow_receivers::ReceiverPipelines::build(device, hr)
        );
        let view_samples = rebuild_if_live!(
            self.view_samples.is_some(),
            super::view_samples::ViewSamplesGpu::build_pipeline(device, hr)
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
        let sky = super::sky::build_sky_pipeline(
            device,
            &super::builtin_shaders::SKY_FRAG,
            self.targets.hdr.sample_count,
            hr,
        )?;
        let sky_velocity = super::sky::build_sky_velocity_pipeline(device, hr)?;
        let grass = rebuild_if_live!(
            self.grass.is_some(),
            super::grass::GrassPipelines::build(device, self.targets.hdr.sample_count, hr)
        );

        // The shadow pipeline needs the static vertex layout.
        let static_vdesc = static_vertex_descriptor();
        let ssao = rebuild_if_live!(
            self.ssao.pass.is_some(),
            concinnity_engine::render::post::ssao::build_pipelines(&post_device)
        );
        let ssr_resolve = rebuild_if_live!(
            self.ssr.resolve.is_some(),
            concinnity_engine::render::post::ssr::build_pipeline(&post_device)
        );
        let reflection_composite = rebuild_if_live!(
            self.ssr.composite.is_some(),
            concinnity_engine::render::post::reflection_composite::build_pipelines(&post_device)
        );
        let ssgi = rebuild_if_live!(
            self.ssgi.pass.is_some(),
            concinnity_engine::render::post::ssgi::build_pipelines(&post_device)
        );
        let disocclusion = rebuild_if_live!(
            self.disocclusion.is_some(),
            concinnity_engine::render::post::disocclusion::build_pipelines(&post_device)
        );
        let rt_reflections = rebuild_if_live!(
            self.rt.pipelines.reflections.is_some(),
            RtReflectionPipelines::new(device, hr)
        );

        // GPU-driven shadow pipelines: the frustum-only shadow
        // decision kernel (from cull.hlsl) + the depth-only bindless shadow
        // render pipeline. Both engine-internal, so they rebuild here. Gated on
        // the live shadow-bindless path.
        let shadow_cull = rebuild_if_live!(
            self.cull.shadow_pipeline.is_some(),
            build_shadow_cull_pipeline(device, hr)
        );
        let shadow_bindless = rebuild_if_live!(
            self.cull.shadow_bindless_pipeline.is_some(),
            build_shadow_bindless_pipeline(device, &static_vdesc, MTLPixelFormat::Depth32Float, hr)
        );
        let far_shadow_bindless = rebuild_if_live!(
            self.cull.far_shadow_bindless_pipeline.is_some(),
            build_shadow_bindless_pipeline(device, &static_vdesc, MTLPixelFormat::Depth16Unorm, hr)
        );

        // All builds succeeded: swap into the live context. After this
        // point the next frame's draw calls bind the freshly compiled
        // pipelines.
        self.composite.pipeline = post;
        if let (Some(p), Some(pass)) = (bloom, self.bloom.as_mut()) {
            pass.swap_pipelines(p);
        }
        if let Some(p) = text {
            self.text.pipeline_state = Some(p);
        }
        if let (Some(p), Some(taa)) = (taa, self.taa.pass.as_mut()) {
            taa.swap_pipelines(p);
        }
        if let (Some(p), Some(cull), Some(sampler_args)) = (main, main_cull, main_sampler_args) {
            self.cull.main_pipeline = Some(p);
            self.arg_buffers.bindless_sampler_args = Some(sampler_args);
            self.cull.pipeline = Some(cull.decide);
            self.cull.pipeline_late = Some(cull.decide_late);
            self.cull.scan_pipeline = Some(cull.scan);
            self.cull.scatter_pipeline = Some(cull.scatter);
            self.cull.icb_arg_encoder = Some(cull.icb_arg_encoder);
            // Force the ICB rebuilds on the next frame so every argument buffer
            // is re-encoded with the arg encoder the new scatter produced.
            // The status buffers and the late ICB are rebuilt by the same
            // `ensure_*_capacity` passes that rebuild the ICBs.
            self.cull.icbs = Vec::new();
            self.cull.icb_arg_buffer = None;
            self.cull.icb_capacity = 0;
            self.cull.icbs_late = Vec::new();
            self.cull.icb_late_arg_buffer = None;
            self.cull.status_buffer = None;
            self.cull.shadow_views = ShadowViewIcb::default();
            self.cull.spot_views = ShadowViewIcb::default();
            self.cull.far_view = ShadowViewIcb::default();
        }
        if let Some((init_pipeline, downsample_pipeline)) = hiz
            && let Some(h) = self.cull.hiz.as_mut()
        {
            h.swap_pipelines(init_pipeline, downsample_pipeline);
        }
        if let Some(p) = auto_ev {
            self.auto_exposure.pipelines = Some(p);
        }
        if let (Some(p), Some(r)) = (receivers, self.shadow_receivers.as_mut()) {
            r.pipelines = p;
        }
        if let (Some(p), Some(v)) = (view_samples, self.view_samples.as_mut()) {
            v.pipeline = p;
        }
        if let Some(p) = decal {
            self.decal.pipeline = Some(p);
        }
        if let Some(p) = fog {
            self.fog.pipeline = Some(p);
        }
        self.sky.pipeline = sky;
        self.sky.velocity_pipeline = sky_velocity;
        if let (Some(p), Some(grass)) = (grass, self.grass.as_mut()) {
            grass.pipelines = p;
        }
        if let (Some(p), Some(pass)) = (ssao, self.ssao.pass.as_mut()) {
            pass.swap_pipelines(p);
        }
        if let (Some(p), Some(resolve)) = (ssr_resolve, self.ssr.resolve.as_mut()) {
            resolve.swap_pipeline(p);
        }
        if let (Some(p), Some(pass)) = (reflection_composite, self.ssr.composite.as_mut()) {
            pass.swap_pipelines(p);
        }
        if let (Some(p), Some(pass)) = (ssgi, self.ssgi.pass.as_mut()) {
            pass.swap_pipelines(p);
        }
        if let (Some(p), Some(pass)) = (disocclusion, self.disocclusion.as_mut()) {
            pass.swap_pipelines(p);
        }
        if let Some(p) = rt_reflections {
            self.rt.pipelines.reflections = Some(p);
        }
        if let Some(p) = shadow_cull {
            self.cull.shadow_pipeline = Some(p);
        }
        if let Some(p) = shadow_bindless {
            self.cull.shadow_bindless_pipeline = Some(p);
        }
        if let Some(p) = far_shadow_bindless {
            self.cull.far_shadow_bindless_pipeline = Some(p);
        }
        for (bucket, pipelines) in world_buckets {
            if let Some(live) = self.cull.world_pipelines.get_mut(bucket) {
                *live = pipelines;
            }
        }
        Ok(())
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
