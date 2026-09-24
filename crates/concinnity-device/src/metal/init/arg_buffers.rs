//! The argument buffers the bindless main pass writes through: the engine
//! sampler block, and the gates that decide when a texture block's ring slot is
//! rewritten.

use concinnity_core::render::error::RenderResult;

use super::{InitGpu, pipelines};
use crate::metal::bindless_args::{ResidencySet, SlotGates};
use crate::metal::context::{MtlArgumentBuffers, MtlSceneAssets, ShadowState};
use crate::metal::cull::CullState;

pub(super) fn build_arg_buffers(
    gpu: &InitGpu<'_>,
    cull: &CullState,
    scene: &MtlSceneAssets,
    shadow: &ShadowState,
) -> RenderResult<MtlArgumentBuffers> {
    let device = &*gpu.hw.device;

    // The engine sampler block for the single-source main program, written once
    // from the three sampler states.
    let bindless_sampler_args = if cull.bindless {
        Some(pipelines::build_bindless_sampler_args(
            device,
            gpu.hot_reload,
            &scene.sampler,
            &shadow.sampler,
            &scene.cube_sampler,
        )?)
    } else {
        None
    };

    Ok(MtlArgumentBuffers {
        bindless_tex_gates: SlotGates::new(gpu.frames_in_flight.max(1)),
        bindless_tail_gates: SlotGates::new(gpu.frames_in_flight.max(1)),
        bindless_residency: ResidencySet::new(),
        bindless_sampler_args,
        texture_epoch: 0,
    })
}
