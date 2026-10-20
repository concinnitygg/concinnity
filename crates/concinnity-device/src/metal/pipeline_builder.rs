// World Shader and SdfVolume pipelines built on a hot-reload worker.
//
// `MTLDevice` is thread-safe and a pipeline state depends on nothing but the
// device, its functions and the descriptor, so the builder carries the device
// and the two settings the descriptors read. The swap on the frame thread then
// only replaces a reference.

use concinnity_core::components::ShaderPrograms;
use concinnity_core::components::sdf_programs::SdfPrograms;
use concinnity_core::render::backend::error::RenderResult;
use concinnity_core::render::backend::{PipelineBuilder, PreparedPipelines};
use concinnity_core::render::shader::programs::raymarch::VolumeFlags;
use concinnity_core::render::shader::programs::surface::GrassShading;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use super::bucket_pipelines::{BucketBuild, BucketPipelines, build_bucket_pipelines};
use super::init::pipelines::make_vertex_descriptor;
use super::raymarch::{VolumePipelines, build_volume_pipelines};

// A world Shader's or a volume's pipelines are built against the device and
// `targets`; a prepared pipeline is swapped in only while the context still
// matches the targets it was built for.
pub(super) struct MtlPipelineBuilder {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub targets: BucketBuild,
    // Per bucket: what of the grass its main fragment shades.
    pub grass: Vec<GrassShading>,
}

// A world Shader's main-pass and pre-pass pipelines and what they were built
// for.
pub(super) struct PreparedWorldShader {
    pub targets: BucketBuild,
    pub pipelines: BucketPipelines,
}

// What a volume's pipelines depend on beyond its flags: the shader source path
// and the engine templates they were compiled from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct VolumeBuild {
    pub hot_reload: bool,
    pub template_generation: u64,
}

// A volume's pipelines and what they were built for.
pub(super) struct PreparedVolume {
    pub build: VolumeBuild,
    pub flags: VolumeFlags,
    pub pipelines: VolumePipelines,
}

impl PipelineBuilder for MtlPipelineBuilder {
    fn world_shader(
        &self,
        bucket: u32,
        programs: &ShaderPrograms,
    ) -> RenderResult<PreparedPipelines> {
        let pipelines = objc2::rc::autoreleasepool(|_| {
            build_bucket_pipelines(
                &self.device,
                &make_vertex_descriptor(),
                bucket as usize,
                Some(programs),
                self.grass.get(bucket as usize).copied().unwrap_or_default(),
                self.targets,
            )
        })?;
        Ok(PreparedPipelines::new(PreparedWorldShader {
            targets: self.targets,
            pipelines,
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
            build: VolumeBuild {
                hot_reload,
                template_generation: self.targets.template_generation,
            },
            flags,
            pipelines,
        }))
    }
}

// The world Shader pipelines in `prepared` when they were built for `targets`.
pub(super) fn world_shader_for(
    prepared: Option<PreparedPipelines>,
    targets: BucketBuild,
) -> Option<BucketPipelines> {
    prepared?
        .downcast::<PreparedWorldShader>()
        .filter(|p| p.targets == targets)
        .map(|p| p.pipelines)
}

// The volume pipelines in `prepared` when they were built for a volume with
// `flags` under `build`.
pub(super) fn volume_for(
    prepared: Option<PreparedPipelines>,
    flags: VolumeFlags,
    build: VolumeBuild,
) -> Option<VolumePipelines> {
    prepared?
        .downcast::<PreparedVolume>()
        .filter(|p| p.flags == flags && p.build == build)
        .map(|p| p.pipelines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLPixelFormat, MTLRenderPipelineState};

    const TARGETS: BucketBuild = BucketBuild {
        sample_count: 4,
        hot_reload: true,
        prepass: true,
        template_generation: 1,
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
        targets: BucketBuild,
        pipeline: &Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    ) -> Option<PreparedPipelines> {
        Some(PreparedPipelines::new(PreparedWorldShader {
            targets,
            pipelines: BucketPipelines {
                main: pipeline.clone(),
                prepass: Some(pipeline.clone()),
                programs: None,
                grass: GrassShading::None,
            },
        }))
    }

    const BUILD: VolumeBuild = VolumeBuild {
        hot_reload: true,
        template_generation: 1,
    };

    fn volume(
        flags: VolumeFlags,
        build: VolumeBuild,
        pipeline: &Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    ) -> Option<PreparedPipelines> {
        Some(PreparedPipelines::new(PreparedVolume {
            build,
            flags,
            pipelines: VolumePipelines {
                pipeline: pipeline.clone(),
                front_pipeline: None,
                shadow_pipeline: None,
                prepass_pipelines: None,
            },
        }))
    }

    #[test]
    fn a_world_shader_pipeline_is_taken_only_for_the_targets_it_was_built_for() {
        let Some(pso) = pipeline() else { return };
        assert!(world_shader_for(world(TARGETS, &pso), TARGETS).is_some());
        let single_sample = BucketBuild {
            sample_count: 1,
            ..TARGETS
        };
        assert!(world_shader_for(world(TARGETS, &pso), single_sample).is_none());
        let embedded = BucketBuild {
            hot_reload: false,
            ..TARGETS
        };
        assert!(world_shader_for(world(TARGETS, &pso), embedded).is_none());
        // Built with a pre-pass for a context that has since lost its G-buffer.
        let no_gbuffer = BucketBuild {
            prepass: false,
            ..TARGETS
        };
        assert!(world_shader_for(world(TARGETS, &pso), no_gbuffer).is_none());
        // Built from the templates a reload has since replaced.
        let reloaded = BucketBuild {
            template_generation: 2,
            ..TARGETS
        };
        assert!(world_shader_for(world(TARGETS, &pso), reloaded).is_none());
        assert!(world_shader_for(volume(SURFACE, BUILD, &pso), TARGETS).is_none());
        assert!(world_shader_for(None, TARGETS).is_none());
    }

    #[test]
    fn volume_pipelines_are_taken_only_for_the_flags_they_were_built_for() {
        let Some(pso) = pipeline() else { return };
        assert!(volume_for(volume(SURFACE, BUILD, &pso), SURFACE, BUILD).is_some());
        let caster = VolumeFlags {
            cast_shadows: true,
            ..SURFACE
        };
        assert!(volume_for(volume(SURFACE, BUILD, &pso), caster, BUILD).is_none());
        let embedded = VolumeBuild {
            hot_reload: false,
            ..BUILD
        };
        assert!(volume_for(volume(SURFACE, BUILD, &pso), SURFACE, embedded).is_none());
        // Built from the templates a reload has since replaced.
        let reloaded = VolumeBuild {
            template_generation: 2,
            ..BUILD
        };
        assert!(volume_for(volume(SURFACE, BUILD, &pso), SURFACE, reloaded).is_none());
        assert!(volume_for(world(TARGETS, &pso), SURFACE, BUILD).is_none());
    }
}
