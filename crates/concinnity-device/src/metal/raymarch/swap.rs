// Replacing a live volume's raymarch pipelines from a hot-reloaded field.
//
// A Metal command buffer retains the pipelines encoded into it, so the old
// pipelines can drop the moment the new ones are in place: a frame still in
// flight keeps drawing with what it encoded.

use concinnity_core::components::sdf_programs::SdfPrograms;
use concinnity_core::render::backend::{PipelineSwap, PreparedPipelines};
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::shader_programs::raymarch::VolumeFlags;

use super::super::MtlContext;
use super::super::pipeline_builder::volume_for;
use super::{VolumePipelines, build_volume_pipelines};
use crate::shader::raymarch_source;

impl MtlContext {
    // Rebuild every pipeline volume `volume` draws with from `programs`, or
    // swap in `prepared` when it was built for this volume's flags. Both are
    // built before either replaces the live one, so a failed build leaves the
    // volume drawing as it was.
    pub(in crate::metal) fn replace_sdf_volume_pipelines(
        &mut self,
        volume: usize,
        programs: &SdfPrograms,
        prepared: Option<PreparedPipelines>,
    ) -> RenderResult<PipelineSwap> {
        let Some(record) = self.raymarch.volumes.get(volume) else {
            return Ok(PipelineSwap::NotResident);
        };
        let flags = VolumeFlags {
            volumetric: record.volumetric,
            cast_shadows: record.cast_shadows,
        };
        let hot_reload = self.hot_reload.enabled;
        let VolumePipelines {
            pipeline,
            shadow_pipeline,
        } = match volume_for(prepared, flags, hot_reload) {
            Some(pipelines) => pipelines,
            None => {
                build_volume_pipelines(&self.hw.device, programs, flags, hot_reload, &record.label)?
            }
        };
        let record = &mut self.raymarch.volumes[volume];
        record.pipeline = pipeline;
        record.shadow_pipeline = shadow_pipeline;
        record.refractive = raymarch_source::taps_scene(programs);
        Ok(PipelineSwap::Swapped)
    }
}
