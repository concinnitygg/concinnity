// Runtime residency of the material-referenced world shader pipelines.
//
// Init builds a pipeline for every world Shader whose payload it decoded and
// leaves a `None` in `world_pipelines` for each one it deferred (a Shader owned
// by a scene other than the start scene). The streaming pump calls in here as
// those scenes pin and unpin, so the pipeline build lands behind the loading
// screen rather than on the frame that first draws the material.
//
// The bucket regions of the GPU-culled command buffer are issued here too: the
// cull kernel wrote every record's command into exactly one region, so each
// region is one `cmd_draw_indexed_indirect` under that bucket's pipeline.
// Mirrors `metal/world_shaders.rs` and `directx/world_shaders.rs`.

use ash::vk;

use crate::vulkan::owned::OwnedPipeline;

use super::context::VkContext;
use super::pipeline::{BucketPipelineTargets, build_bucket_pipeline};

impl VkContext {
    // Build the bindless main-pass pipeline for one shader bucket. Replaces
    // whatever the bucket currently holds, so a re-pin after an eviction
    // rebuilds cleanly.
    pub(in crate::vulkan) fn install_world_shader(
        &mut self,
        bucket: u32,
        shader: crate::gfx::backend_init::ShaderBytes<'_>,
    ) -> Result<(), String> {
        let slot = self.world_pipeline_slot(bucket)?;
        let layout = self
            .cull
            .bindless_pipeline_layout
            .as_ref()
            .ok_or_else(|| "shader buckets need the bindless main pass".to_string())?;
        let pipeline = build_bucket_pipeline(
            &self.device,
            BucketPipelineTargets {
                render_pass: self.main_render_pass.handle(),
                layout: layout.handle(),
                msaa_samples: self.msaa_samples,
                swapchain_format: self.swapchain.format,
            },
            bucket as usize,
            shader,
            &self.cull.bindless_main_spv,
        )?;
        // A re-pin over a slot that still holds a pipeline has the same in-flight
        // hazard as an evict, so retire the old one the same way.
        self.evict_world_shader(bucket);
        self.cull.world_pipelines[slot] = Some(pipeline);
        Ok(())
    }

    // Release one bucket's pipeline. A Vulkan pipeline may not be destroyed while
    // a submitted command buffer still references it, so the in-flight frames have
    // to finish first. An evict only happens when a scene unpins, which is already
    // a loading-screen stall, so waiting here costs nothing a player sees.
    pub(in crate::vulkan) fn evict_world_shader(&mut self, bucket: u32) {
        let Ok(slot) = self.world_pipeline_slot(bucket) else {
            return;
        };
        if self.cull.world_pipelines[slot].is_none() {
            return;
        }
        // Idle first: the pipeline is retired as the slot's `Option` drops, and
        // the retire queue only guarantees the frames-in-flight window, which
        // this out-of-frame path does not tick.
        // SAFETY: a wait on this device's own queues; it takes no borrowed state.
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        self.cull.world_pipelines[slot] = None;
        self.device.reclaim_idle();
    }

    // Whether a bucket's draws can render this frame: bucket 0 is the world
    // default program, every other bucket needs its pipeline installed.
    pub(in crate::vulkan) fn world_shader_resident(&self, bucket: usize) -> bool {
        bucket == 0
            || matches!(
                self.cull.world_pipelines.get(bucket.wrapping_sub(1)),
                Some(Some(_))
            )
    }

    pub(in crate::vulkan) fn world_pipeline(&self, bucket: usize) -> Option<&OwnedPipeline> {
        self.cull
            .world_pipelines
            .get(bucket.checked_sub(1)?)?
            .as_ref()
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
            let Some(pipeline) = self.world_pipeline(bucket) else {
                return;
            };
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                self.device.cmd_bind_pipeline(
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
    // for geometry the colour pass omits.
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
        for bucket in 1..self.shader_bucket_count() {
            if !self.world_shader_resident(bucket) {
                continue;
            }
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
            self.device.cmd_draw_indexed_indirect(
                cmd,
                indirect,
                self.bucket_region_offset(bucket),
                draw_count,
                super::cull::INDIRECT_COMMAND_STRIDE,
            );
        }
    }

    fn world_pipeline_slot(&self, bucket: u32) -> Result<usize, String> {
        let slot = (bucket as usize)
            .checked_sub(1)
            .ok_or_else(|| "shader bucket 0 is the world default program".to_string())?;
        if slot >= self.cull.world_pipelines.len() {
            return Err(format!(
                "shader bucket {bucket} is past the world's {} shader pipeline(s)",
                self.cull.world_pipelines.len()
            ));
        }
        Ok(slot)
    }
}
