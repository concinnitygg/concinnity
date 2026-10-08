// Runtime residency of the material-referenced world shader pipelines.
//
// Init builds a pipeline for every world Shader whose payload it decoded and
// leaves a `None` in `world_pipelines` for each one it deferred (a Shader owned
// by a scene other than the start scene). The streaming pump calls in here as
// those scenes pin and unpin, handing over a pipeline its worker already built.

use concinnity_core::render::backend::{PipelineBuilder, PipelineSwap, PreparedPipelines};
use concinnity_core::render::error::RenderResult;
use std::sync::Arc;

use super::MtlContext;
use super::bucket_pipelines::{build_bucket_pipelines, replacement};
use super::init::pipelines::make_vertex_descriptor;
use super::pipeline_builder::{MtlPipelineBuilder, world_shader_for};

impl MtlContext {
    // Install one shader bucket's main-pass and pre-pass pipelines: `prepared`
    // when they were built for this context's targets, else ones built here.
    // Replaces whatever the bucket currently holds, so a re-pin after an
    // eviction installs cleanly.
    pub(super) fn install_world_shader(
        &mut self,
        bucket: u32,
        programs: &concinnity_core::components::ShaderPrograms,
        prepared: Option<PreparedPipelines>,
    ) -> RenderResult<()> {
        self.cull.world_pipelines.slot(bucket)?;
        let pipelines = match world_shader_for(prepared, self.bucket_build()) {
            Some(pipelines) => pipelines,
            None => build_bucket_pipelines(
                &self.hw.device,
                &make_vertex_descriptor(),
                bucket as usize,
                Some(programs),
                self.bucket_build(),
            )?,
        };
        self.cull.world_pipelines.install(bucket, pipelines)?;
        Ok(())
    }

    // Rebuild one world Shader's pipelines from hot-reloaded programs, or swap
    // in `prepared` when they were built for this context's targets. Bucket 0
    // is the main pipeline pair; another bucket is rebuilt only while
    // installed. Both halves are built before they replace the live pair, so
    // a failed build of either leaves the live pipelines bound.
    pub(super) fn update_world_shader(
        &mut self,
        bucket: u32,
        programs: &concinnity_core::components::ShaderPrograms,
        prepared: Option<PreparedPipelines>,
    ) -> RenderResult<PipelineSwap> {
        if bucket == 0 {
            let prepared = world_shader_for(prepared, self.bucket_build());
            self.update_default_world_shader(programs, prepared)?;
            return Ok(PipelineSwap::Swapped);
        }
        self.cull.world_pipelines.slot(bucket)?;
        if !self.cull.world_pipelines.resident(bucket as usize) {
            return Ok(PipelineSwap::NotResident);
        }
        let build = self.bucket_build();
        let pipelines = replacement(
            &self.hw.device,
            bucket as usize,
            Some(programs),
            build,
            self.cull.world_pipelines.get(bucket as usize),
            world_shader_for(prepared, build),
        )?;
        self.cull.world_pipelines.install(bucket, pipelines)?;
        Ok(PipelineSwap::Swapped)
    }

    // A builder for this context's world Shader and volume pipelines, for a
    // streaming or hot-reload worker.
    pub(super) fn pipeline_builder(&self) -> Arc<dyn PipelineBuilder> {
        Arc::new(MtlPipelineBuilder {
            device: self.hw.device.clone(),
            targets: self.bucket_build(),
        })
    }

    // Release one bucket's pipeline. A Metal command buffer retains the
    // pipelines encoded into it, so dropping this reference mid-frame is safe
    // and the next frame's main pass simply skips the bucket. That retention is
    // Metal's alone: DirectX and Vulkan command lists do NOT keep a pipeline
    // alive, so their evict drains the device first.
    pub(super) fn evict_world_shader(&mut self, bucket: u32) {
        self.cull.world_pipelines.evict(bucket);
    }
}
