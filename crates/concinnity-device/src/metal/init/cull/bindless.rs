//! The bindless main pass: the main pipeline and the material-referenced world
//! shader pipelines the cull kernel routes draws between.

use super::CullInputs;
use crate::metal::init::InitGpu;
use crate::metal::init::pipelines::{self, BucketPipelines};
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::shader::programs::surface::SurfaceVariant;
use concinnity_engine::render::shader::world_pipelines::{BucketPlan, WorldPipelines};

pub(super) struct BindlessPass {
    pub(super) active: bool,
    pub(super) main_pipeline: Option<BucketPipelines>,
    pub(super) world_pipelines: WorldPipelines<BucketPipelines>,
    pub(super) bucket_count: usize,
}

pub(super) fn build_bindless_pass(
    gpu: &InitGpu<'_>,
    inputs: &CullInputs<'_>,
) -> RenderResult<BindlessPass> {
    let device = &*gpu.hw.device;
    let hot_reload = gpu.hot_reload;
    let world_shaders = inputs.world_shaders;
    let hdr_samples = inputs.features.hdr_samples;

    // A world with no 3D scene content skips the main PBR pipeline and the
    // whole GPU-cull path: the Main pass then survives as a bare clear the
    // composite pass samples (the same shape a world_hidden frame takes).
    let active = inputs.features.scene;
    // Pre-pass pipelines are built only for a world that starts with a
    // G-buffer consumer; a quality change that adds one builds them then.
    let build = pipelines::BucketBuild {
        hot_reload,
        sample_count: hdr_samples,
        prepass: inputs.features.gbuffer_enabled,
        template_generation: 0,
    };
    let main_pipeline = if active {
        Some(pipelines::build_bucket_pipelines(
            device,
            inputs.vert_desc,
            0,
            world_shaders[0].programs,
            SurfaceVariant::Plain,
            build,
        )?)
    } else {
        None
    };

    // Material-referenced shaders (ShaderHandle 1..), and each Shader's grass
    // variants, each get a bindless pipeline; the cull kernel routes their
    // draws into per-bucket ICBs.
    let plan = BucketPlan::new(world_shaders)?;
    let world_pipelines = if inputs.features.scene && !plan.extra().is_empty() {
        if !active {
            return Err(RenderError::Other(
                "material-referenced Shaders need the GPU-driven main pass, which a \
                        world with no 3D scene content does not build"
                    .into(),
            ));
        }
        WorldPipelines::build(&plan, |bucket, source| {
            pipelines::build_bucket_pipelines(
                device,
                inputs.vert_desc,
                bucket,
                source.programs,
                source.variant,
                build,
            )
        })?
    } else {
        WorldPipelines::default()
    };
    let bucket_count = world_pipelines.bucket_count();

    Ok(BindlessPass {
        active,
        main_pipeline,
        world_pipelines,
        bucket_count,
    })
}
