// Replacing a live volume's raymarch pipelines from a hot-reloaded field.
//
// A Metal command buffer retains the pipelines encoded into it, so the old
// pipelines can drop the moment the new ones are in place: a frame still in
// flight keeps drawing with what it encoded.

use concinnity_core::components::sdf_programs::SdfPrograms;
use concinnity_core::platform::Platform;
use concinnity_core::render::backend::PipelineSwap;
use concinnity_core::render::error::RenderResult;

use super::super::MtlContext;
use super::{VolumePipelines, build_volume_pipelines};
use crate::shader::raymarch_source::{self, Request, VolumeFlags};

impl MtlContext {
    // Rebuild every pipeline volume `volume` draws with from `programs`. Both
    // are built before either replaces the live one, so a failed build leaves
    // the volume drawing as it was.
    pub(in crate::metal) fn replace_sdf_volume_pipelines(
        &mut self,
        volume: usize,
        programs: &SdfPrograms,
    ) -> RenderResult<PipelineSwap> {
        let Some(record) = self.raymarch.volumes.get(volume) else {
            return Ok(PipelineSwap::NotResident);
        };
        let flags = VolumeFlags {
            volumetric: record.volumetric,
            cast_shadows: record.cast_shadows,
        };
        let VolumePipelines {
            pipeline,
            shadow_pipeline,
        } = build_volume_pipelines(
            &self.hw.device,
            programs,
            flags,
            self.hot_reload.enabled,
            &record.label,
        )?;
        let record = &mut self.raymarch.volumes[volume];
        record.pipeline = pipeline;
        record.shadow_pipeline = shadow_pipeline;
        record.refractive = raymarch_source::taps_scene(programs);
        Ok(PipelineSwap::Swapped)
    }
}

// Compile the metallibs a volume's pipelines load into the shader cache, so the
// pipeline build that follows loads them rather than compiling on the render
// thread. Needs no device, so it runs on any thread.
pub(crate) fn warm_sdf_field(
    programs: &SdfPrograms,
    flags: VolumeFlags,
    hot_reload: bool,
) -> RenderResult<()> {
    warm_entries(
        programs,
        flags,
        hot_reload,
        super::super::msl_cache::warm_cooked,
    )
}

fn warm_entries(
    programs: &SdfPrograms,
    flags: VolumeFlags,
    hot_reload: bool,
    mut warm: impl FnMut(&[u8], &str) -> RenderResult<()>,
) -> RenderResult<()> {
    let label = format!("SdfVolume field '{}'", programs.field.path);
    for family in flags.families() {
        for program in concinnity_core::render::shader_programs::raymarch::ALL
            .iter()
            .filter(|p| p.family == family)
        {
            let req = Request {
                family,
                platform: Platform::Metal,
                entry: program.entry,
                hot_reload,
                label: &label,
            };
            let msl = raymarch_source::artifact(programs, &req, crate::shader::compile::cooked)?;
            warm(&msl, &label)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::ShaderSource;
    use concinnity_core::components::compiled_programs::CompiledProgram;
    use concinnity_core::render::shader_programs::raymarch;
    use concinnity_core::render::shader_source;

    // A payload holding a stored artifact for every entry `flags` draws with.
    fn stored(flags: VolumeFlags) -> SdfPrograms {
        let mut programs = SdfPrograms {
            field: ShaderSource {
                path: "shaders/blob.hlsl".to_string(),
                text: "// a field".to_string(),
            },
            programs: Vec::new(),
        };
        for program in raymarch::programs(flags.volumetric, flags.cast_shadows) {
            let src = raymarch::source(program.family, Platform::Metal, programs.field.as_file());
            programs.programs.push(CompiledProgram {
                entry: program.entry.to_string(),
                source_digest: shader_source::source_digest(&src),
                artifact: program.entry.as_bytes().to_vec(),
            });
        }
        programs
    }

    // Warming covers exactly the entries the volume's pipelines load, and reads
    // each from the payload rather than compiling.
    #[test]
    fn warming_covers_every_entry_the_volume_draws_with() {
        for (volumetric, cast_shadows, want) in [
            (false, false, &["raymarch_vertex", "raymarch_fragment"][..]),
            (
                false,
                true,
                &[
                    "raymarch_vertex",
                    "raymarch_fragment",
                    "raymarch_shadow_vertex",
                    "raymarch_shadow_fragment",
                ][..],
            ),
            (
                true,
                true,
                &["raymarch_volumetric_vertex", "raymarch_volumetric_fragment"][..],
            ),
        ] {
            let flags = VolumeFlags {
                volumetric,
                cast_shadows,
            };
            let mut warmed = Vec::new();
            warm_entries(&stored(flags), flags, false, |msl, _| {
                warmed.push(String::from_utf8(msl.to_vec()).unwrap());
                Ok(())
            })
            .unwrap();
            assert_eq!(warmed, want, "{flags:?}");
        }
    }
}
