// Runtime residency of the material-referenced world shader pipelines.
//
// Init builds a pipeline for every world Shader whose payload it decoded and
// leaves a `None` in `world_pipelines` for each one it deferred (a Shader owned
// by a scene other than the start scene). The streaming pump calls in here as
// those scenes pin and unpin, handing over a pipeline its worker already built.
//
// The bucket regions of the GPU-culled command buffer are issued here too: the
// cull kernel wrote every record's command into exactly one region, so each
// region is one `cmd_draw_indexed_indirect` under that bucket's pipeline.
// Mirrors `metal/world_shaders.rs` and `directx/world_shaders.rs`.

use ash::vk;
use concinnity_core::render::backend::{PipelineBuilder, PipelineSwap, PreparedPipelines};
use concinnity_core::render::error::{RenderError, RenderResult};

use super::context::VkContext;
use super::pipeline::{BucketPipelineTargets, build_world_shader_pipeline};
use crate::vulkan::pipeline_builder::{VkPipelineBuilder, world_shader_for};
use std::sync::Arc;

impl VkContext {
    // Install one shader bucket's bindless main-pass pipeline: `prepared` when
    // it was built for this context's targets, else one built here. Replaces
    // whatever the bucket currently holds, so a re-pin after an eviction
    // installs cleanly.
    pub(in crate::vulkan) fn install_world_shader(
        &mut self,
        bucket: u32,
        programs: &concinnity_core::components::ShaderPrograms,
        prepared: Option<PreparedPipelines>,
    ) -> RenderResult<()> {
        self.cull.world_pipelines.slot(bucket)?;
        let targets = self.bucket_pipeline_targets().ok_or_else(|| {
            RenderError::Other("shader buckets need the bindless main pass".to_string())
        })?;
        let pipeline = match world_shader_for(prepared, &self.pipeline_gate, targets) {
            Some(pipeline) => pipeline,
            None => {
                build_world_shader_pipeline(&self.hw.device, targets, bucket as usize, programs)?
            }
        };
        // The displaced pipeline drops into the device's retire queue, which
        // holds it until every frame in flight that recorded against it retires.
        self.cull.world_pipelines.install(bucket, pipeline)?;
        Ok(())
    }

    // Rebuild one world Shader's pipeline from hot-reloaded programs, or swap
    // in `prepared` when it was built for this context's targets. Bucket 0 is
    // the main pass's pipeline; another bucket is rebuilt only while
    // installed, and the replacement is built before the old pipeline retires,
    // so a failed build leaves the live one bound.
    pub(in crate::vulkan) fn update_world_shader(
        &mut self,
        bucket: u32,
        programs: &concinnity_core::components::ShaderPrograms,
        prepared: Option<PreparedPipelines>,
    ) -> RenderResult<PipelineSwap> {
        if bucket == 0 {
            let prepared = self
                .bucket_pipeline_targets()
                .and_then(|targets| world_shader_for(prepared, &self.pipeline_gate, targets));
            self.update_default_world_shader(programs, prepared)?;
            return Ok(PipelineSwap::Swapped);
        }
        self.cull.world_pipelines.slot(bucket)?;
        if !self.cull.world_pipelines.resident(bucket as usize) {
            return Ok(PipelineSwap::NotResident);
        }
        self.install_world_shader(bucket, programs, prepared)?;
        Ok(PipelineSwap::Swapped)
    }

    // What every bucket's pipeline is built against here, or `None` when the
    // GPU-driven main pass is not live.
    pub(in crate::vulkan) fn bucket_pipeline_targets(&self) -> Option<BucketPipelineTargets> {
        let layout = self.cull.bindless_pipeline_layout.as_ref()?;
        Some(BucketPipelineTargets {
            render_pass: self.targets.main_render_pass.handle(),
            layout: layout.handle(),
            msaa_samples: self.targets.msaa_samples,
            swapchain_format: self.swapchain.format,
            hot_reload: self.hot_reload.enabled,
        })
    }

    // A builder for this context's world Shader and volume pipelines, for a
    // streaming or hot-reload worker.
    pub(in crate::vulkan) fn pipeline_builder(&self) -> Arc<dyn PipelineBuilder> {
        Arc::new(VkPipelineBuilder {
            device: self.hw.device.clone(),
            gate: self.pipeline_gate.clone(),
            world: self.bucket_pipeline_targets(),
            volumes: self.volume_pipeline_targets(),
        })
    }

    // Release one bucket's pipeline. It drops into the device's retire queue,
    // which destroys it only once every frame in flight that recorded against
    // it has retired.
    pub(in crate::vulkan) fn evict_world_shader(&mut self, bucket: u32) {
        self.cull.world_pipelines.evict(bucket);
    }

    // Issue the bucket 1.. regions of `indirect`, each under its own material
    // shader's pipeline. Bucket 0's region is issued by the caller (it runs under
    // the pipeline the pass already bound), so this covers only the
    // material-referenced shaders. `draw_count` is the record prefix each region
    // draws, matching bucket 0's. Returns the number of indirect draws issued, and
    // leaves the last bucket's pipeline bound.
    pub(in crate::vulkan) fn draw_bucket_regions(
        &self,
        cmd: vk::CommandBuffer,
        indirect: vk::Buffer,
        draw_count: u32,
    ) -> u32 {
        self.for_each_resident_bucket(|bucket| {
            let Some(pipeline) = self.cull.world_pipelines.get(bucket) else {
                return;
            };
            // Every bucket shares the bindless layout, so the Wireframe twin
            // stands in for each one while that view mode is on.
            let pipeline = self.wireframe_or(pipeline, self.wireframe.bindless.as_ref());
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                self.hw.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.handle(),
                );
            }
            self.draw_bucket_region(cmd, indirect, draw_count, bucket);
        })
    }

    // Issue the bucket 1.. regions of `indirect` under the pipeline the caller
    // already bound. Used by the depth / velocity pre-pass, which shades nothing
    // and so runs every bucket through its own single pipeline -- but still has to
    // skip a non-resident bucket, or the pre-pass would lay down depth and motion
    // for geometry the color pass omits.
    pub(in crate::vulkan) fn draw_bucket_regions_shared_pipeline(
        &self,
        cmd: vk::CommandBuffer,
        indirect: vk::Buffer,
        draw_count: u32,
    ) -> u32 {
        self.for_each_resident_bucket(|bucket| {
            self.draw_bucket_region(cmd, indirect, draw_count, bucket)
        })
    }

    // Run `f` for every bucket past the default whose Shader is resident,
    // returning how many ran. A bucket whose scene has not pinned yet has no
    // pipeline: skip it until warmup builds one rather than drawing it with the
    // wrong program.
    fn for_each_resident_bucket(&self, mut f: impl FnMut(usize)) -> u32 {
        let mut issued = 0;
        for bucket in self.cull.world_pipelines.resident_buckets() {
            f(bucket);
            issued += 1;
        }
        issued
    }

    fn draw_bucket_region(
        &self,
        cmd: vk::CommandBuffer,
        indirect: vk::Buffer,
        draw_count: u32,
        bucket: usize,
    ) {
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            self.hw.device.cmd_draw_indexed_indirect(
                cmd,
                indirect,
                self.bucket_region_offset(bucket),
                draw_count,
                super::cull::INDIRECT_COMMAND_STRIDE,
            );
        }
    }
}
