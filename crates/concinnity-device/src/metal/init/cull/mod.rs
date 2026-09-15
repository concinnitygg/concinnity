//! The GPU-driven cull state: the bindless main pass, the compute cull with its
//! Hi-Z pyramid, and the GPU-driven cascade shadow, plus the probe state whose
//! bake renders through the cull and the instance records it folds in.

use concinnity_core::gfx::lod;
use concinnity_core::gfx::render_types::{GpuDrawArgs, InstancedCluster, draw_args_flags};
use concinnity_core::render::backend_init::WorldShader;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::reflection_probe;
use concinnity_core::render::uniforms::ProbeSet;
use concinnity_core::transform::IDENTITY;
use objc2_metal::MTLVertexDescriptor;

use super::{Features, InitGpu};
use crate::metal::bindless_args::{ResidencySet, SlotGates};
use crate::metal::context::{InstancedState, ProbeState};
use crate::metal::cull::{CullState, metal_instance_records};
use crate::metal::probe_prefilter::ProbePrefilterPipelines;
use crate::metal::transient::RetirePool;

mod bindless;
mod compute;
mod shadow;

pub(super) struct CullInputs<'a> {
    pub(super) world_shaders: &'a [WorldShader<'a>],
    pub(super) vert_desc: &'a MTLVertexDescriptor,
    pub(super) features: &'a Features,
    // Whether the cascade shadow pipeline exists.
    pub(super) shadow_enabled: bool,
    pub(super) occlusion_two_pass: bool,
}

pub(super) fn build_cull(gpu: &InitGpu<'_>, inputs: CullInputs<'_>) -> RenderResult<CullState> {
    let bindless = bindless::build_bindless_pass(gpu, &inputs)?;
    let compute = compute::build_compute_cull(gpu, &bindless, inputs.features)?;
    let shadow =
        shadow::build_shadow_cull(gpu, &bindless, inputs.vert_desc, inputs.shadow_enabled)?;

    // Two-pass occlusion is only usable on the bindless cull path (the
    // phase-2 pipeline exists exactly then). Gate the request here so the
    // runtime flag is true only when the feature can actually run.
    let two_pass_occlusion = inputs.occlusion_two_pass && compute.pipeline_phase2.is_some();

    Ok(CullState {
        bindless: bindless.active,
        main_pipeline: bindless.main_pipeline,
        world_pipelines: bindless.world_pipelines,
        pipeline: compute.pipeline,
        encode_pipeline: compute.encode_pipeline,
        bucket_count: bindless.bucket_count,
        icbs: Vec::new(),
        icb_arg_encoder: compute.icb_arg_encoder,
        icb_arg_buffer: None,
        icb_capacity: 0,
        pipeline_phase2: compute.pipeline_phase2,
        icbs_2: Vec::new(),
        icb_2_arg_buffer: None,
        status_buffer: None,
        two_pass_occlusion,
        hiz: compute.hiz,
        prev_view_proj: IDENTITY,
        cur_view_proj: IDENTITY,
        hiz_valid: false,
        shadow_pipeline: shadow.pipeline,
        shadow_bindless_pipeline: shadow.bindless_pipeline,
        shadow_icb: None,
        shadow_icb_arg_buffer: None,
        shadow_status: None,
        shadow_icb_capacity: 0,
        mirror_slots: Vec::new(),
        mirror_status: None,
        mirror_icb_capacity: 0,
    })
}

// The reflection-probe state, empty until `set_reflection_probes` supplies
// placements. The convolution kernels share the cull pipeline's gate: a probe
// capture renders through the bindless ICB, so a world without the cull
// pipeline never bakes one and never needs them.
pub(super) fn build_probe(gpu: &InitGpu<'_>, cull: &CullState) -> RenderResult<ProbeState> {
    let prefilter = if cull.pipeline.is_some() {
        Some(ProbePrefilterPipelines::new(
            &gpu.hw.device,
            gpu.hot_reload,
        )?)
    } else {
        None
    };
    Ok(ProbeState {
        placements: Vec::new(),
        maps: Vec::new(),
        bake_queue: reflection_probe::ProbeBakeQueue::new(0),
        set: ProbeSet::EMPTY,
        rendering: None,
        prefiltering: None,
        prefilter,
        retire_pool: RetirePool::new(),
        cube_args: None,
        cube_arg_gates: SlotGates::new(gpu.frames_in_flight.max(1) + 1),
        cube_residency: ResidencySet::new(),
    })
}

// Fold every instanced-cluster instance into the GPU-driven bindless
// main pass: each becomes a `GpuObjectData` record appended after the
// static objects, drawn through the shared cull + indirect path
// (`build_object_buffer` / `build_draw_args_buffer` re-append these every
// frame; see `cull_count`). Built once here against the final bindless
// pool counts (the texture pool, the same count the static fill uses) via
// the Metal-local `metal_instance_records`, which addresses the flat pool
// with Metal's CPU-bias convention (NOT the shared core
// `instance_object_records`, which is the DX/VK raw-index convention):
// instances are placed at world load and never move, so the records are
// static. The draw args carry the cluster base index range;
// `build_draw_args_buffer` patches per-instance LOD over it each frame for
// the clusters that declare alternates.
pub(super) fn build_instanced(
    clusters: Vec<InstancedCluster>,
    albedo_count: usize,
) -> InstancedState {
    let records = metal_instance_records(&clusters, albedo_count);
    let mut draw_args: Vec<GpuDrawArgs> = Vec::with_capacity(records.len());
    for cluster in &clusters {
        for _ in &cluster.instances {
            draw_args.push(GpuDrawArgs {
                index_count: cluster.index_count as u32,
                index_offset: cluster.index_offset as u32,
                base_vertex: 0,
                flags: draw_args_flags(true, true, true),
            });
        }
    }
    InstancedState {
        any_lod: lod::any_cluster_has_lod(&clusters),
        clusters,
        records,
        draw_args,
    }
}
