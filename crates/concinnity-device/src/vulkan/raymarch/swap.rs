// Replacing a live volume's raymarch pipelines from a hot-reloaded field.
//
// A Vulkan pipeline may not be destroyed while a submitted command buffer still
// references it, so the frames in flight finish before the old pipelines drop.
// A save is a developer's edit, not a frame a player sees, so the wait costs
// nothing that matters.

use concinnity_core::components::sdf_programs::SdfPrograms;
use concinnity_core::render::backend::{PipelineSwap, PreparedPipelines};
use concinnity_core::render::error::RenderResult;

use super::{VolumePipelineTargets, VolumePipelines, build_volume_pipelines};
use crate::shader::raymarch_source;
use crate::vulkan::context::VkContext;
use crate::vulkan::pipeline_builder::volume_for;

impl VkContext {
    // Rebuild every pipeline volume `volume` draws with from `programs`, or
    // swap in `prepared` when it was built for this volume against this
    // context's targets. Both are built before either replaces the live one,
    // so a failed build leaves the volume drawing as it was.
    pub(in crate::vulkan) fn replace_sdf_volume_pipelines(
        &mut self,
        volume: usize,
        programs: &SdfPrograms,
        prepared: Option<PreparedPipelines>,
    ) -> RenderResult<PipelineSwap> {
        let Some(targets) = self.volume_pipeline_targets() else {
            return Ok(PipelineSwap::NotResident);
        };
        let Some(record) = self.raymarch.as_ref().and_then(|rm| rm.volumes.get(volume)) else {
            return Ok(PipelineSwap::NotResident);
        };
        let VolumePipelines {
            pipeline,
            front_pipeline,
            shadow_pipeline,
            prepass_pipelines,
        } = match volume_for(prepared, &self.pipeline_gate, targets, record.flags) {
            Some(pipelines) => pipelines,
            None => build_volume_pipelines(
                &self.hw.device,
                &targets,
                programs,
                record.flags,
                &record.label,
            )?,
        };
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
            record.front_pipeline = front_pipeline;
            record.shadow_pipeline = shadow_pipeline;
            record.prepass_pipelines = prepass_pipelines;
            record.refractive = raymarch_source::taps_scene(programs);
        }
        self.hw.device.reclaim_idle();
        Ok(PipelineSwap::Swapped)
    }

    // What a volume's pipelines are built against here, or `None` when the
    // world has no raymarch pass.
    pub(in crate::vulkan) fn volume_pipeline_targets(&self) -> Option<VolumePipelineTargets> {
        let rm = self.raymarch.as_ref()?;
        Some(rm.volume_targets(
            self.shadow.render_pass.handle(),
            self.targets.msaa_samples,
            self.hot_reload.enabled,
        ))
    }
}
