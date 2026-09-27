// Replacing a live volume's raymarch pipelines from a hot-reloaded field.
//
// A Vulkan pipeline may not be destroyed while a submitted command buffer still
// references it, so the frames in flight finish before the old pipelines drop.
// A save is a developer's edit, not a frame a player sees, so the wait costs
// nothing that matters.

use concinnity_core::components::sdf_programs::SdfPrograms;
use concinnity_core::render::backend::PipelineSwap;
use concinnity_core::render::error::RenderResult;

use super::{VolumePipelineTargets, VolumePipelines, build_volume_pipelines};
use crate::shader::raymarch_source;
use crate::vulkan::context::VkContext;

impl VkContext {
    // Rebuild every pipeline volume `volume` draws with from `programs`. Both
    // are built before either replaces the live one, so a failed build leaves
    // the volume drawing as it was.
    pub(in crate::vulkan) fn replace_sdf_volume_pipelines(
        &mut self,
        volume: usize,
        programs: &SdfPrograms,
    ) -> RenderResult<PipelineSwap> {
        let Some(rm) = self.raymarch.as_ref() else {
            return Ok(PipelineSwap::NotResident);
        };
        let Some(record) = rm.volumes.get(volume) else {
            return Ok(PipelineSwap::NotResident);
        };
        let VolumePipelines {
            pipeline,
            shadow_pipeline,
        } = build_volume_pipelines(
            &VolumePipelineTargets {
                device: &self.hw.device,
                render_pass: rm.render_pass.handle(),
                layout: rm.pipeline_layout.handle(),
                shadow_render_pass: self.shadow.render_pass.handle(),
                shadow_layout: rm.shadow_pipeline_layout.handle(),
                msaa_samples: self.targets.msaa_samples,
                hot_reload: self.hot_reload.enabled,
            },
            programs,
            record.flags,
            &record.label,
        )?;
        // Idle first: the old pipelines retire as they drop, and the retire
        // queue only covers the frames-in-flight window, which this
        // out-of-frame path does not tick.
        self.wait_idle();
        if let Some(record) = self
            .raymarch
            .as_mut()
            .and_then(|rm| rm.volumes.get_mut(volume))
        {
            record.pipeline = pipeline;
            record.shadow_pipeline = shadow_pipeline;
            record.refractive = raymarch_source::taps_scene(programs);
        }
        self.hw.device.reclaim_idle();
        Ok(PipelineSwap::Swapped)
    }
}
