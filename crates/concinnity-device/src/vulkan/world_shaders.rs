// Runtime residency of the material-referenced world shader pipelines: init
// leaves `None` for Shaders owned by scenes other than the start scene, and the
// streaming pump installs them as their scenes pin. Also issues each bucket's
// list of the GPU-culled command buffer as one indirect draw.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::render::backend::WorldShaders;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::backend::{PipelineBuilder, PipelineSwap, PreparedPipelines};
use concinnity_engine::render::shader::programs::surface::SurfaceVariant;
use concinnity_engine::render::shader::world_pipelines::{
    BucketSource, check_rebuild, replace_bucket,
};

use super::context::VkContext;
use super::pipeline::{
    BindlessSpv, BucketPipelineTargets, BucketPipelines, PrepassTargets, build_bucket_pipeline,
    build_bucket_prepass, build_world_prepass, build_world_shader_pipeline,
};
use crate::vulkan::pipeline_builder::{VkPipelineBuilder, world_shader_for};
use crate::vulkan::record::Recorder;
use std::sync::Arc;

impl VkContext {
    // One shader bucket's pipelines to replace a bucket that has a pre-pass
    // when `live_prepass`: `prepared` when it keeps what the live bucket draws
    // with, else ones built here. Refused when the build would drop a live
    // pre-pass, so the caller keeps the live pair.
    fn bucket_pipelines(
        &self,
        bucket: u32,
        programs: &concinnity_engine::components::ShaderPrograms,
        variant: SurfaceVariant,
        prepared: Option<PreparedPipelines>,
        live_prepass: bool,
    ) -> RenderResult<BucketPipelines> {
        self.cull.world_pipelines.slot(bucket)?;
        let targets = self.bucket_pipeline_targets().ok_or_else(|| {
            RenderError::Other("shader buckets need the bindless main pass".to_string())
        })?;
        replace_bucket(
            bucket as usize,
            live_prepass,
            world_shader_for(prepared, &self.pipeline_gate, targets)
                .filter(|p| p.variant == variant),
            |p: &BucketPipelines| p.prepass.is_some(),
            || {
                build_world_shader_pipeline(
                    &self.hw.device,
                    targets,
                    bucket as usize,
                    programs,
                    variant,
                )
            },
        )
    }

    // The grass variants of `shader` built from its fresh programs, all before any
    // is installed so a failure keeps the live family whole.
    fn build_family(
        &self,
        shader: u32,
        programs: &concinnity_engine::components::ShaderPrograms,
        install: bool,
    ) -> RenderResult<Vec<(u32, BucketPipelines)>> {
        let mut built = Vec::new();
        for bucket in self.cull.world_pipelines.family(shader) {
            let variant = self.cull.world_pipelines.variant(bucket as usize);
            let live = self.cull.world_pipelines.get(bucket as usize);
            if !install && live.is_none() {
                continue;
            }
            let live_prepass = live.is_some_and(|live| live.prepass.is_some());
            built.push((
                bucket,
                self.bucket_pipelines(bucket, programs, variant, None, live_prepass)?,
            ));
        }
        Ok(built)
    }

    // Install a family `build_family` built.
    fn install_family(&mut self, built: Vec<(u32, BucketPipelines)>) -> RenderResult<()> {
        for (bucket, pipelines) in built {
            self.cull.world_pipelines.install(bucket, pipelines)?;
        }
        Ok(())
    }

    // What every bucket's pipelines are built against here, or `None` when the
    // GPU-driven main pass is not live. The pre-pass's half is there only once
    // the world has a G-buffer.
    pub(in crate::vulkan) fn bucket_pipeline_targets(&self) -> Option<BucketPipelineTargets> {
        let layout = self.cull.bindless_pipeline_layout.as_ref()?;
        let prepass = match (self.cull.prepass_layout.as_ref(), self.gbuffer.as_ref()) {
            (Some(prepass), Some(gb)) => Some(PrepassTargets {
                render_pass: gb.prepass_render_pass.handle(),
                layout: prepass.pipeline_layout.handle(),
            }),
            _ => None,
        };
        Some(BucketPipelineTargets {
            render_pass: self.targets.main_render_pass.handle(),
            layout: layout.handle(),
            prepass,
            msaa_samples: self.targets.msaa_samples,
            swapchain_format: self.swapchain.format,
            hot_reload: self.hot_reload.enabled,
            template_generation: self.hot_reload.generation,
        })
    }

    // Build the pre-pass pipeline each bucket is missing while a G-buffer exists.
    // A failure leaves that bucket without G-buffer draws, never without shading.
    pub(in crate::vulkan) fn sync_prepass_pipelines(&mut self) {
        self.sync_grass_prepass(false);
        let Some((targets, prepass)) = self
            .bucket_pipeline_targets()
            .and_then(|targets| Some((targets, targets.prepass?)))
        else {
            return;
        };
        let device = self.hw.device.clone();
        let engine = &self.cull.bindless_main_spv;
        let build =
            |bucket: usize, programs: Option<&concinnity_engine::components::ShaderPrograms>| {
                match programs {
                    Some(programs) => {
                        build_world_prepass(&device, prepass, bucket, programs, targets.hot_reload)
                    }
                    None => build_bucket_prepass(
                        &device,
                        prepass,
                        bucket,
                        &engine.prepass_vert,
                        &engine.prepass_frag,
                    ),
                }
            };
        if self.cull.prepass_pipeline.is_none() {
            let fresh = build(0, self.world_shader.as_ref());
            self.cull.prepass_pipeline = fresh;
        }
        let buckets: Vec<usize> = self.cull.world_pipelines.resident_buckets().collect();
        for bucket in buckets {
            let Some(pipelines) = self.cull.world_pipelines.get(bucket) else {
                continue;
            };
            if pipelines.prepass.is_some() {
                continue;
            }
            let fresh = build(bucket, pipelines.programs.as_ref());
            if let Some(pipelines) = self.cull.world_pipelines.get_mut(bucket) {
                pipelines.prepass = fresh;
            }
        }
    }

    // Every resident material bucket's pipelines rebuilt from the current
    // templates, `engine` standing in for a bucket with no Shader of its own.
    // Fails as a whole when any bucket's main pipeline fails or its rebuild
    // would drop a live pre-pass, so the caller keeps every live pair.
    pub(in crate::vulkan) fn rebuild_world_buckets(
        &self,
        engine: &BindlessSpv,
    ) -> RenderResult<Vec<(usize, BucketPipelines)>> {
        let Some(targets) = self.bucket_pipeline_targets() else {
            return Ok(Vec::new());
        };
        let mut rebuilt = Vec::new();
        for bucket in self.cull.world_pipelines.resident_buckets() {
            let Some(live) = self.cull.world_pipelines.get(bucket) else {
                continue;
            };
            let source = BucketSource {
                programs: live.programs.as_ref(),
                deferred: false,
                variant: live.variant,
            };
            let pipelines =
                build_bucket_pipeline(&self.hw.device, targets, bucket, &source, engine)?;
            check_rebuild(bucket, live.prepass.is_some(), pipelines.prepass.is_some())?;
            rebuilt.push((bucket, pipelines));
        }
        Ok(rebuilt)
    }

    // Issue the bucket 1.. regions of `indirect`, each under its own pipeline;
    // bucket 0 is the caller's. Returns the draws issued.
    pub(in crate::vulkan) fn draw_bucket_regions(
        &self,
        rec: &Recorder<'_>,
        indirect: vk::Buffer,
        frame_idx: usize,
    ) -> u32 {
        let mut issued = 0;
        for bucket in self.drawn_buckets(frame_idx) {
            let Some(pipelines) = self.cull.world_pipelines.get(bucket) else {
                continue;
            };
            // Every bucket shares the bindless layout, so the Wireframe twin
            // stands in for each one while that view mode is on.
            let pipeline = self.wireframe_or(&pipelines.main, self.wireframe.bindless.as_ref());
            rec.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline);
            self.draw_main_stream(rec, indirect, u32::saturating_from(bucket));
            issued += 1;
        }
        issued
    }

    // Issue the bucket 1.. regions under each shader's pre-pass pipeline, skipping
    // the same non-resident buckets the color pass skips. Returns the regions
    // issued.
    pub(in crate::vulkan) fn draw_prepass_bucket_regions(
        &self,
        rec: &Recorder<'_>,
        indirect: vk::Buffer,
        frame_idx: usize,
    ) -> u32 {
        let mut issued = 0;
        for bucket in self.drawn_buckets(frame_idx) {
            let Some(prepass) = self
                .cull
                .world_pipelines
                .get(bucket)
                .and_then(|pipelines| pipelines.prepass.as_ref())
            else {
                continue;
            };
            rec.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, prepass);
            self.draw_main_stream(rec, indirect, u32::saturating_from(bucket));
            issued += 1;
        }
        issued
    }

    // The buckets past the default whose regions draw in frame `frame_idx`:
    // resident, holding records, and pickable in that frame's view.
    fn drawn_buckets(&self, frame_idx: usize) -> impl Iterator<Item = usize> + '_ {
        let live = self
            .cull
            .live_buckets
            .get(frame_idx)
            .map_or(u64::MAX, |l| l.load(std::sync::atomic::Ordering::Relaxed));
        self.cull
            .variants
            .drawn_buckets(live, self.cull.world_pipelines.resident_buckets())
    }

    // Draw stream `stream`'s list of `indirect`, one of the main view's
    // indirect buffers.
    pub(in crate::vulkan) fn draw_main_stream(
        &self,
        rec: &Recorder<'_>,
        indirect: vk::Buffer,
        stream: u32,
    ) {
        self.draw_stream(
            rec,
            self.main_lists(indirect),
            stream,
            self.stream_layout().slots(stream),
        );
    }
}

impl WorldShaders for VkContext {
    // Install one shader bucket's main-pass and pre-pass pipelines: `prepared`
    // when they were built for this context's targets, else ones built here.
    // Replaces whatever the bucket currently holds, so a re-pin after an
    // eviction installs cleanly.
    fn install_world_shader(
        &mut self,
        bucket: u32,
        programs: &concinnity_engine::components::ShaderPrograms,
        prepared: Option<PreparedPipelines>,
    ) -> RenderResult<()> {
        let variant = self.cull.world_pipelines.variant(bucket as usize);
        let pipelines = self.bucket_pipelines(bucket, programs, variant, prepared, false)?;
        let family = self.build_family(bucket, programs, true)?;
        // The displaced pipelines drop into the device's retire queue, which
        // holds them until every frame in flight that recorded against them
        // retires.
        self.cull.world_pipelines.install(bucket, pipelines)?;
        self.install_family(family)
    }

    // Release one bucket's pipelines. They drop into the device's retire queue,
    // which destroys them only once every frame in flight that recorded against
    // them has retired.
    fn evict_world_shader(&mut self, bucket: u32) {
        let family = self.cull.world_pipelines.family(bucket);
        for b in family {
            self.cull.world_pipelines.evict(b);
        }
        self.cull.world_pipelines.evict(bucket);
    }

    // Rebuild one world Shader's pipelines from hot-reloaded programs, or install
    // `prepared`. Both halves build before the old ones retire, so a failure keeps
    // the live pipelines.
    fn update_world_shader(
        &mut self,
        bucket: u32,
        programs: &concinnity_engine::components::ShaderPrograms,
        prepared: Option<PreparedPipelines>,
    ) -> RenderResult<PipelineSwap> {
        if bucket == 0 {
            let prepared = self
                .bucket_pipeline_targets()
                .and_then(|targets| world_shader_for(prepared, &self.pipeline_gate, targets));
            let family = self.build_family(0, programs, false)?;
            self.update_default_world_shader(programs, prepared)?;
            self.install_family(family)?;
            return Ok(PipelineSwap::Swapped);
        }
        self.cull.world_pipelines.slot(bucket)?;
        if !self.cull.world_pipelines.resident(bucket as usize) {
            return Ok(PipelineSwap::NotResident);
        }
        let live_prepass = (self.cull.world_pipelines.get(bucket as usize))
            .is_some_and(|live| live.prepass.is_some());
        let variant = self.cull.world_pipelines.variant(bucket as usize);
        let pipelines = self.bucket_pipelines(bucket, programs, variant, prepared, live_prepass)?;
        let family = self.build_family(bucket, programs, false)?;
        self.cull.world_pipelines.install(bucket, pipelines)?;
        self.install_family(family)?;
        Ok(PipelineSwap::Swapped)
    }

    // A builder for this context's world Shader and volume pipelines, for a
    // streaming or hot-reload worker.
    fn pipeline_builder(&self) -> Option<Arc<dyn PipelineBuilder>> {
        Some(Arc::new(VkPipelineBuilder {
            device: self.hw.device.clone(),
            gate: self.pipeline_gate.clone(),
            world: self.bucket_pipeline_targets(),
            volumes: self.volume_pipeline_targets(),
            variants: self.cull.world_pipelines.variants(),
        }))
    }
}
