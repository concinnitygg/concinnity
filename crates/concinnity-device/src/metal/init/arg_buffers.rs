//! The argument buffers the bindless main pass and the probe-sampling passes
//! write through: the texture block encoder, the engine sampler block, the probe
//! cube encoder, and the gates that decide when a ring slot re-encodes.

use concinnity_core::render::error::RenderResult;

use super::{InitGpu, pipelines};
use crate::metal::bindless_args::{ResidencySet, SlotGates};
use crate::metal::context::{MtlArgumentBuffers, MtlSceneAssets, ShadowState};
use crate::metal::cull::CullState;
use crate::metal::probe_cubes::probe_cube_arg_encoder;

pub(super) fn build_arg_buffers(
    gpu: &InitGpu<'_>,
    cull: &CullState,
    scene: &MtlSceneAssets,
    shadow: &ShadowState,
) -> RenderResult<MtlArgumentBuffers> {
    let device = &*gpu.hw.device;

    // The bindless texture encoder, and the engine sampler block for the
    // single-source main program, written once from the three sampler states.
    let (bindless_tex_encoder, bindless_sampler_args) = if cull.bindless {
        let encoders = pipelines::build_bindless_arg_encoders(device, gpu.hot_reload)?;
        let sampler_args = pipelines::build_bindless_sampler_args(
            device,
            &encoders.sampler,
            &scene.sampler,
            &shadow.sampler,
            &scene.cube_sampler,
        )?;
        (Some(encoders.texture), Some(sampler_args))
    } else {
        (None, None)
    };

    // The probe cube argument encoder is world-independent: every pass that
    // samples the set declares the same block, and the layout is fixed by
    // MAX_PROBES rather than by world content.
    let probe_cube_encoder = probe_cube_arg_encoder(device, gpu.hot_reload)?;

    Ok(MtlArgumentBuffers {
        bindless_tex_encoder,
        bindless_tex_gates: SlotGates::new(gpu.frames_in_flight.max(1) + 1),
        bindless_tail_gates: SlotGates::new(gpu.frames_in_flight.max(1) + 1),
        bindless_residency: ResidencySet::new(),
        bindless_sampler_args,
        probe_cube_encoder,
        texture_epoch: 0,
    })
}
