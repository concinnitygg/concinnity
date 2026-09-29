// World Shader and SdfVolume pipelines built on a hot-reload worker.
//
// `MTLDevice` is thread-safe and a pipeline state depends on nothing but the
// device, its functions and the descriptor, so the builder carries the device
// and the two settings the descriptors read. The swap on the frame thread then
// only replaces a reference.

use concinnity_core::components::ShaderPrograms;
use concinnity_core::components::sdf_programs::SdfPrograms;
use concinnity_core::render::backend::{PipelineBuilder, PreparedPipelines};
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::shader_programs::raymarch::VolumeFlags;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLRenderPipelineState};

use super::init::pipelines::{build_bucket_pipeline, build_main_pipeline, make_vertex_descriptor};
use super::raymarch::{VolumePipelines, build_volume_pipelines};

// What a world Shader's or a volume's pipelines are built against, beyond the
// device. A prepared pipeline is swapped in only while the context still
// matches the targets it was built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PipelineTargets {
    pub sample_count: u32,
    pub hot_reload: bool,
}

pub(super) struct MtlPipelineBuilder {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub targets: PipelineTargets,
}

// A world Shader's main-pass pipeline and what it was built for.
pub(super) struct PreparedWorldShader {
    pub targets: PipelineTargets,
    pub pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
}

// A volume's pipelines and what they were built for.
pub(super) struct PreparedVolume {
    pub hot_reload: bool,
    pub flags: VolumeFlags,
    pub pipelines: VolumePipelines,
}

impl PipelineBuilder for MtlPipelineBuilder {
    fn world_shader(
        &self,
        bucket: u32,
        programs: &ShaderPrograms,
    ) -> RenderResult<PreparedPipelines> {
        let PipelineTargets {
            sample_count,
            hot_reload,
        } = self.targets;
        let pipeline = objc2::rc::autoreleasepool(|_| {
            let vert_desc = make_vertex_descriptor();
            if bucket == 0 {
                build_main_pipeline(
                    &self.device,
                    &vert_desc,
                    Some(programs),
                    hot_reload,
                    sample_count,
                )
            } else {
                build_bucket_pipeline(
                    &self.device,
                    &vert_desc,
                    bucket as usize,
                    programs,
                    hot_reload,
                    sample_count,
                )
            }
        })?;
        Ok(PreparedPipelines::new(PreparedWorldShader {
            targets: self.targets,
            pipeline,
        }))
    }

    fn sdf_volume(
        &self,
        programs: &SdfPrograms,
        flags: VolumeFlags,
        label: &str,
    ) -> RenderResult<PreparedPipelines> {
        let hot_reload = self.targets.hot_reload;
        let pipelines = objc2::rc::autoreleasepool(|_| {
            build_volume_pipelines(&self.device, programs, flags, hot_reload, label)
        })?;
        Ok(PreparedPipelines::new(PreparedVolume {
            hot_reload,
            flags,
            pipelines,
        }))
    }
}

// The world Shader pipeline in `prepared` when it was built for `targets`.
pub(super) fn world_shader_for(
    prepared: Option<PreparedPipelines>,
    targets: PipelineTargets,
) -> Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    prepared?
        .downcast::<PreparedWorldShader>()
        .filter(|p| p.targets == targets)
        .map(|p| p.pipeline)
}

// The volume pipelines in `prepared` when they were built for a volume with
// `flags` under `hot_reload`.
pub(super) fn volume_for(
    prepared: Option<PreparedPipelines>,
    flags: VolumeFlags,
    hot_reload: bool,
) -> Option<VolumePipelines> {
    prepared?
        .downcast::<PreparedVolume>()
        .filter(|p| p.flags == flags && p.hot_reload == hot_reload)
        .map(|p| p.pipelines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLPixelFormat};

    const TARGETS: PipelineTargets = PipelineTargets {
        sample_count: 4,
        hot_reload: true,
    };

    const SURFACE: VolumeFlags = VolumeFlags {
        volumetric: false,
        cast_shadows: false,
    };

    // A real pipeline state to stand in for any the builder makes; `None`
    // without a Metal device.
    fn pipeline() -> Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
        let device = MTLCreateSystemDefaultDevice()?;
        Some(
            super::super::pipeline::build_text_pipeline(&device, MTLPixelFormat::BGRA8Unorm, false)
                .unwrap(),
        )
    }

    fn world(
        targets: PipelineTargets,
        pipeline: &Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    ) -> Option<PreparedPipelines> {
        Some(PreparedPipelines::new(PreparedWorldShader {
            targets,
            pipeline: pipeline.clone(),
        }))
    }

    fn volume(
        flags: VolumeFlags,
        hot_reload: bool,
        pipeline: &Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    ) -> Option<PreparedPipelines> {
        Some(PreparedPipelines::new(PreparedVolume {
            hot_reload,
            flags,
            pipelines: VolumePipelines {
                pipeline: pipeline.clone(),
                shadow_pipeline: None,
            },
        }))
    }

    #[test]
    fn a_world_shader_pipeline_is_taken_only_for_the_targets_it_was_built_for() {
        let Some(pso) = pipeline() else { return };
        assert!(world_shader_for(world(TARGETS, &pso), TARGETS).is_some());
        let single_sample = PipelineTargets {
            sample_count: 1,
            ..TARGETS
        };
        assert!(world_shader_for(world(TARGETS, &pso), single_sample).is_none());
        let embedded = PipelineTargets {
            hot_reload: false,
            ..TARGETS
        };
        assert!(world_shader_for(world(TARGETS, &pso), embedded).is_none());
        assert!(world_shader_for(volume(SURFACE, true, &pso), TARGETS).is_none());
        assert!(world_shader_for(None, TARGETS).is_none());
    }

    #[test]
    fn volume_pipelines_are_taken_only_for_the_flags_they_were_built_for() {
        let Some(pso) = pipeline() else { return };
        assert!(volume_for(volume(SURFACE, true, &pso), SURFACE, true).is_some());
        let caster = VolumeFlags {
            cast_shadows: true,
            ..SURFACE
        };
        assert!(volume_for(volume(SURFACE, true, &pso), caster, true).is_none());
        assert!(volume_for(volume(SURFACE, true, &pso), SURFACE, false).is_none());
        assert!(volume_for(world(TARGETS, &pso), SURFACE, true).is_none());
    }
}
