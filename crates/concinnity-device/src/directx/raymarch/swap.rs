// Replacing a live volume's raymarch PSOs from a hot-reloaded field.
//
// D3D12 command lists do not keep a pipeline state alive, so the frames in
// flight that recorded against the old PSOs have to finish before those drop.
// A save is a developer's edit, not a frame a player sees, so draining the
// queue for it costs nothing that matters.

use concinnity_core::components::sdf_programs::SdfPrograms;
use concinnity_core::render::backend::{PipelineSwap, PreparedPipelines};
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::shader_programs::raymarch::VolumeFlags;

use super::{VolumePsoTargets, VolumePsos, build_volume_psos};
use crate::directx::context::DxContext;
use crate::directx::pipeline_builder::volume_for;
use crate::shader::raymarch_source;

impl DxContext {
    // Rebuild every PSO volume `volume` draws with from `programs`, or swap in
    // `prepared` when it was built for this volume against this context's
    // targets. Both are built before either replaces the live one, so a failed
    // build leaves the volume drawing as it was.
    pub(in crate::directx) fn replace_sdf_volume_pipelines(
        &mut self,
        volume: usize,
        programs: &SdfPrograms,
        prepared: Option<PreparedPipelines>,
    ) -> RenderResult<PipelineSwap> {
        let (Some(rm), Some(targets)) = (self.raymarch.as_ref(), self.volume_pso_targets()) else {
            return Ok(PipelineSwap::NotResident);
        };
        let Some(record) = rm.volumes.get(volume) else {
            return Ok(PipelineSwap::NotResident);
        };
        let flags = VolumeFlags {
            volumetric: record.volumetric,
            cast_shadows: record.cast_shadows,
        };
        let VolumePsos {
            pso,
            front_pso,
            shadow_pso,
            prepass_psos,
        } = match volume_for(prepared, &targets, flags) {
            Some(psos) => psos,
            None => build_volume_psos(
                &VolumePsoTargets {
                    device: &self.hw.device,
                    info_queue: self.hw.info_queue.as_ref(),
                    root_sig: &rm.root_sig,
                    shadow_root_sig: &rm.shadow_root_sig,
                    msaa_samples: self.targets.hdr.msaa_samples,
                    hot_reload: self.hot_reload.enabled,
                },
                programs,
                flags,
                &record.label,
            )?,
        };
        self.wait_idle();
        let Some(record) = self
            .raymarch
            .as_mut()
            .and_then(|rm| rm.volumes.get_mut(volume))
        else {
            return Ok(PipelineSwap::NotResident);
        };
        record.pso = pso;
        record.front_pso = front_pso;
        record.shadow_pso = shadow_pso;
        record.prepass_psos = prepass_psos;
        record.refractive = raymarch_source::taps_scene(programs);
        Ok(PipelineSwap::Swapped)
    }
}
