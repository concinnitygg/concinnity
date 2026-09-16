//! The bindless main pass: the main pipeline and the material-referenced world
//! shader pipelines the cull kernel routes draws between.

use concinnity_core::gfx::render_types;
use concinnity_core::render::error::{RenderError, RenderResult};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLRenderPipelineState;

use super::CullInputs;
use crate::metal::init::InitGpu;
use crate::metal::init::pipelines::{self, WorldPipelineTable};

pub(super) struct BindlessPass {
    pub(super) active: bool,
    pub(super) main_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub(super) world_pipelines: WorldPipelineTable,
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
    let main_pipeline = if active {
        Some(pipelines::build_main_pipeline(
            device,
            inputs.vert_desc,
            world_shaders[0].programs,
            hot_reload,
            hdr_samples,
        )?)
    } else {
        None
    };

    // Material-referenced shaders (ShaderHandle 1..) each get a bindless
    // pipeline; the cull kernel routes their draws into per-bucket ICBs.
    let world_pipelines = if inputs.features.scene && world_shaders.len() > 1 {
        let max = render_types::MAX_SHADER_BUCKETS;
        if world_shaders.len() > max {
            return Err(RenderError::Other(format!(
                "world declares {} Shaders but at most {max} are supported",
                world_shaders.len()
            )));
        }
        if !active {
            return Err(RenderError::Other(
                "material-referenced Shaders need the GPU-driven main pass, which a \
                        world with no 3D scene content does not build"
                    .into(),
            ));
        }
        pipelines::build_world_pipeline_table(
            device,
            inputs.vert_desc,
            &world_shaders[1..],
            hot_reload,
            hdr_samples,
        )?
    } else {
        Vec::new()
    };
    let bucket_count = 1 + world_pipelines.len();

    Ok(BindlessPass {
        active,
        main_pipeline,
        world_pipelines,
        bucket_count,
    })
}
