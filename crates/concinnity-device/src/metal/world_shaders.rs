// Runtime residency of the material-referenced world shader pipelines.
//
// Init builds a pipeline for every world Shader whose payload it decoded and
// leaves a `None` in `world_pipelines` for each one it deferred (a Shader owned
// by a scene other than the start scene). The streaming pump calls in here as
// those scenes pin and unpin, handing over a pipeline its worker already built.

use concinnity_core::render::backend::{PipelineBuilder, PipelineSwap, PreparedPipelines};
use concinnity_core::render::error::{RenderError, RenderResult};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLRenderPipelineState;
use std::sync::Arc;

use super::MtlContext;
use super::init::pipelines::{build_bucket_pipeline, make_vertex_descriptor};
use super::pipeline_builder::{MtlPipelineBuilder, PipelineTargets, world_shader_for};

impl MtlContext {
    // Install one shader bucket's bindless main-pass pipeline: `prepared` when
    // it was built for this context's targets, else one built here. Replaces
    // whatever the bucket currently holds, so a re-pin after an eviction
    // installs cleanly.
    pub(super) fn install_world_shader(
        &mut self,
        bucket: u32,
        programs: &concinnity_core::components::ShaderPrograms,
        prepared: Option<PreparedPipelines>,
    ) -> RenderResult<()> {
        let slot = self.world_pipeline_slot(bucket)?;
        let pso = match world_shader_for(prepared, self.pipeline_targets()) {
            Some(pso) => pso,
            None => build_bucket_pipeline(
                &self.hw.device,
                &make_vertex_descriptor(),
                bucket as usize,
                programs,
                self.hot_reload.enabled,
                self.targets.hdr.sample_count,
            )?,
        };
        self.cull.world_pipelines[slot] = Some(pso);
        Ok(())
    }

    // Rebuild one world Shader's pipeline from hot-reloaded programs, or swap
    // in `prepared` when it was built for this context's targets. Bucket 0 is
    // the main pipeline; another bucket is rebuilt only while installed, and
    // the replacement is built before it replaces, so a failed build leaves the
    // live pipeline bound.
    pub(super) fn update_world_shader(
        &mut self,
        bucket: u32,
        programs: &concinnity_core::components::ShaderPrograms,
        prepared: Option<PreparedPipelines>,
    ) -> RenderResult<PipelineSwap> {
        if bucket == 0 {
            let prepared = world_shader_for(prepared, self.pipeline_targets());
            self.update_default_world_shader(programs, prepared)?;
            return Ok(PipelineSwap::Swapped);
        }
        self.world_pipeline_slot(bucket)?;
        if !self.world_shader_resident(bucket as usize) {
            return Ok(PipelineSwap::NotResident);
        }
        self.install_world_shader(bucket, programs, prepared)?;
        Ok(PipelineSwap::Swapped)
    }

    // What this context's world Shader and volume pipelines are built against.
    pub(super) fn pipeline_targets(&self) -> PipelineTargets {
        PipelineTargets {
            sample_count: self.targets.hdr.sample_count,
            hot_reload: self.hot_reload.enabled,
        }
    }

    // A builder for this context's world Shader and volume pipelines, for a
    // streaming or hot-reload worker.
    pub(super) fn pipeline_builder(&self) -> Arc<dyn PipelineBuilder> {
        Arc::new(MtlPipelineBuilder {
            device: self.hw.device.clone(),
            targets: self.pipeline_targets(),
        })
    }

    // Release one bucket's pipeline. A Metal command buffer retains the
    // pipelines encoded into it, so dropping this reference mid-frame is safe
    // and the next frame's main pass simply skips the bucket. That retention is
    // Metal's alone: DirectX and Vulkan command lists do NOT keep a pipeline
    // alive, so their evict drains the device first.
    pub(super) fn evict_world_shader(&mut self, bucket: u32) {
        if let Ok(slot) = self.world_pipeline_slot(bucket) {
            self.cull.world_pipelines[slot] = None;
        }
    }

    // Whether a bucket's draws can render this frame: bucket 0 is the world
    // default program, every other bucket needs its pipeline installed.
    pub(super) fn world_shader_resident(&self, bucket: usize) -> bool {
        bucket == 0
            || matches!(
                self.cull.world_pipelines.get(bucket.wrapping_sub(1)),
                Some(Some(_))
            )
    }

    pub(super) fn world_pipeline(
        &self,
        bucket: usize,
    ) -> Option<&Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
        self.cull
            .world_pipelines
            .get(bucket.checked_sub(1)?)?
            .as_ref()
    }

    // A bucket outside the world's table is a scene-authoring mistake, not a
    // device failure, so it stays `Other` whatever the pipeline build would say.
    fn world_pipeline_slot(&self, bucket: u32) -> RenderResult<usize> {
        let slot = (bucket as usize).checked_sub(1).ok_or_else(|| {
            RenderError::Other("shader bucket 0 is the world default program".into())
        })?;
        if slot >= self.cull.world_pipelines.len() {
            return Err(RenderError::Other(format!(
                "shader bucket {bucket} is past the world's {} shader pipeline(s)",
                self.cull.world_pipelines.len()
            )));
        }
        Ok(slot)
    }
}
