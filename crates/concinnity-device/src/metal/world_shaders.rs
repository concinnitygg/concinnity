// Runtime residency of the material-referenced world shader pipelines: init
// leaves `None` for Shaders owned by scenes other than the start scene, and the
// streaming pump installs them as their scenes pin.

use concinnity_engine::render::backend::WorldShaders;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::backend::{PipelineBuilder, PipelineSwap, PreparedPipelines};
use std::sync::Arc;

use super::MtlContext;
use super::bucket_pipelines::{BucketPipelines, build_bucket_pipelines, replacement};
use super::init::pipelines::make_vertex_descriptor;
use super::pipeline_builder::{MtlPipelineBuilder, world_shader_for};

impl MtlContext {
    // The grass variants of `shader` built from its fresh programs, all before
    // any is installed so a failure keeps the live family whole.
    fn build_family(
        &self,
        shader: u32,
        programs: &concinnity_engine::components::ShaderPrograms,
        install: bool,
    ) -> RenderResult<Vec<(u32, BucketPipelines)>> {
        let mut built = Vec::new();
        for bucket in self.cull.world_pipelines.family(shader) {
            if !install && !self.cull.world_pipelines.resident(bucket as usize) {
                continue;
            }
            let pipelines = replacement(
                &self.hw.device,
                bucket as usize,
                Some(programs),
                self.cull.world_pipelines.variant(bucket as usize),
                self.bucket_build(),
                self.cull.world_pipelines.get(bucket as usize),
                None,
            )?;
            built.push((bucket, pipelines));
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
}

impl WorldShaders for MtlContext {
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
        self.cull.world_pipelines.slot(bucket)?;
        let family = self.build_family(bucket, programs, true)?;
        let variant = self.cull.world_pipelines.variant(bucket as usize);
        let prepared =
            world_shader_for(prepared, self.bucket_build()).filter(|p| p.variant == variant);
        let pipelines = match prepared {
            Some(pipelines) => pipelines,
            None => build_bucket_pipelines(
                &self.hw.device,
                &make_vertex_descriptor(),
                bucket as usize,
                Some(programs),
                variant,
                self.bucket_build(),
            )?,
        };
        self.cull.world_pipelines.install(bucket, pipelines)?;
        self.install_family(family)
    }

    // Release one bucket's pipeline. Command buffers retain what they encoded,
    // so this is safe mid-frame; the next main pass skips the bucket.
    fn evict_world_shader(&mut self, bucket: u32) {
        let family = self.cull.world_pipelines.family(bucket);
        for b in family {
            self.cull.world_pipelines.evict(b);
        }
        self.cull.world_pipelines.evict(bucket);
    }

    // Rebuild one world Shader's pipelines from hot-reloaded programs, or
    // install `prepared`. Both halves build before the live pair is replaced.
    fn update_world_shader(
        &mut self,
        bucket: u32,
        programs: &concinnity_engine::components::ShaderPrograms,
        prepared: Option<PreparedPipelines>,
    ) -> RenderResult<PipelineSwap> {
        if bucket == 0 {
            let prepared = world_shader_for(prepared, self.bucket_build());
            let family = self.build_family(0, programs, false)?;
            self.update_default_world_shader(programs, prepared)?;
            self.install_family(family)?;
            return Ok(PipelineSwap::Swapped);
        }
        self.cull.world_pipelines.slot(bucket)?;
        if !self.cull.world_pipelines.resident(bucket as usize) {
            return Ok(PipelineSwap::NotResident);
        }
        let build = self.bucket_build();
        let family = self.build_family(bucket, programs, false)?;
        let pipelines = replacement(
            &self.hw.device,
            bucket as usize,
            Some(programs),
            self.cull.world_pipelines.variant(bucket as usize),
            build,
            self.cull.world_pipelines.get(bucket as usize),
            world_shader_for(prepared, build),
        )?;
        self.cull.world_pipelines.install(bucket, pipelines)?;
        self.install_family(family)?;
        Ok(PipelineSwap::Swapped)
    }

    // A builder for this context's world Shader and volume pipelines, for a
    // streaming or hot-reload worker.
    fn pipeline_builder(&self) -> Option<Arc<dyn PipelineBuilder>> {
        Some(Arc::new(MtlPipelineBuilder {
            device: self.hw.device.clone(),
            targets: self.bucket_build(),
            variants: self.cull.world_pipelines.variants(),
        }))
    }
}
