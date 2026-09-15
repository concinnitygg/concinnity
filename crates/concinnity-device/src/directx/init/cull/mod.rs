//! The GPU-driven cull state: record sizing, the bindless main pass with its
//! shader buckets, the compute cull with its Hi-Z pyramid, and the GPU-driven
//! shadow, G-buffer and two-pass occlusion passes built over them.

use concinnity_core::gfx::render_types::clone_reserve;
use concinnity_core::render::backend_init::{SceneData, WorldShader};
use concinnity_core::render::error::RenderResult;
use concinnity_core::transform::IDENTITY;

use super::InitGpu;
use crate::directx::context::{CullState, DxDescriptors, DxTargets, FRAMES};
use crate::directx::probe_prefilter::ProbePrefilterPipelines;
use bindless::BindlessPass;
use compute::ComputeCull;

mod bindless;
mod compute;
mod gbuffer;
mod shadow;
mod two_pass;

// How many records the GPU-driven cull buffers hold, decided from the world
// before any of them is built.
pub(super) struct CullPlan {
    pub(super) n_instances: usize,
    pub(super) n_cull: usize,
}

pub(super) fn plan_cull(world: &SceneData<'_>) -> CullPlan {
    // Total instances across all clusters, folded into the GPU-driven bindless
    // pass as `GpuObjectData` records after the `n_objects` static objects.
    let n_instances: usize = world
        .instanced_clusters
        .iter()
        .map(|c| c.instances.len())
        .sum();
    // Merged record count: static build-time objects, the instanced-cluster
    // instances, the runtime reserve, then the skinned objects. The per-frame
    // static fills write only the first `n_objects`; the instance records are
    // written once at init; runtime and skinned records are written each frame
    // into their reserved regions. `n_chunk_max` sizes the streamed-chunk window
    // and `clone_reserve` the spawned-clone one; both live in the single
    // runtime reserve between the instances and the skinned tail (see
    // `DrawState::n_runtime`).
    let n_objects = world.draw_objects.len();
    let n_cull =
        n_objects + n_instances + world.n_chunk_max + clone_reserve(n_objects) + world.n_skinned;
    CullPlan {
        n_instances,
        n_cull,
    }
}

pub(super) struct CullInputs<'a> {
    pub(super) world: &'a SceneData<'a>,
    pub(super) world_shaders: &'a [WorldShader<'a>],
    pub(super) plan: &'a CullPlan,
    pub(super) descriptors: &'a DxDescriptors,
    pub(super) targets: &'a DxTargets,
    // Albedo resources at the front of the flat bindless pool.
    pub(super) albedo_count: usize,
    pub(super) shadow_enabled: bool,
    pub(super) gbuffer_enabled: bool,
    pub(super) occlusion_two_pass: bool,
}

pub(super) fn build_cull(gpu: &InitGpu<'_>, inputs: CullInputs<'_>) -> RenderResult<CullState> {
    let CullInputs {
        world,
        world_shaders,
        plan,
        descriptors,
        targets,
        albedo_count,
        shadow_enabled,
        gbuffer_enabled,
        occlusion_two_pass,
    } = inputs;
    let bindless =
        bindless::build_bindless_pass(gpu, world_shaders, plan, targets.hdr.msaa_samples)?;
    let compute = compute::build_compute_cull(
        gpu,
        compute::ComputeInputs {
            bindless: &bindless,
            plan,
            descriptors,
            targets,
        },
    )?;
    let two_pass =
        two_pass::build_two_pass_cull(gpu, &bindless, &compute, plan, occlusion_two_pass)?;
    let shadow_cull = shadow::build_shadow_cull(gpu, &compute, plan, shadow_enabled)?;
    let gbuffer_pass = gbuffer::build_gbuffer_pass(gpu, &compute, plan, gbuffer_enabled)?;
    write_instance_records(world, plan, albedo_count, &bindless, &compute);

    // The bindless texture pool's base, one table handle per frame-in-flight
    // copy; pool index `texture_slot` lands on the albedo SRV and
    // `albedo_count + normal_slot` on the normal SRV. The bindless main pass
    // and the RT hit shader bind the recording frame's copy.
    let bindless_pool_gpu = (0..FRAMES)
        .map(|f| {
            descriptors
                .slot_gpu(descriptors.layout.flat_pool_base_slot + f * descriptors.flat_pool_len)
        })
        .collect();

    Ok(CullState {
        main_bindless_root_sig: Some(bindless.root_sig),
        main_bindless_pso: Some(bindless.pso),
        world_pipelines: bindless.world_pipelines,
        bucket_stride: plan.n_cull,
        bindless_main_shaders: bindless.shaders,
        object_buffer_resources: bindless.object_buffers,
        object_buffer_ptrs: bindless.object_ptrs,
        bindless_pool_gpu,
        cull_root_sig: compute.root_sig,
        cull_pso: compute.pso,
        cull_pso_phase2: two_pass.pso,
        cull_command_signature: compute.command_signature,
        draw_args_buffer_resources: compute.draw_args_buffers,
        draw_args_buffer_ptrs: compute.draw_args_ptrs,
        indirect_cmd_buffers: compute.indirect_buffers,
        cull_status_buffers: compute.status_buffers,
        indirect_cmd_buffers_2: two_pass.indirect_buffers,
        shadow_bindless_root_sig: shadow_cull.bindless_root_sig,
        shadow_bindless_pso: shadow_cull.bindless_pso,
        shadow_bindless_cmd_sig: shadow_cull.cmd_sig,
        cull_pso_shadow: shadow_cull.cull_pso,
        shadow_indirect_buffers: shadow_cull.indirect_buffers,
        shadow_cull_status_buffers: shadow_cull.status_buffers,
        gbuffer_bindless_root_sig: gbuffer_pass.root_sig,
        gbuffer_bindless_pso: gbuffer_pass.pso,
        gbuffer_bindless_cmd_sig: gbuffer_pass.cmd_sig,
        prev_model_buffers: gbuffer_pass.prev_model_buffers,
        model_history_root_sig: gbuffer_pass.model_history_root_sig,
        model_history_pso: gbuffer_pass.model_history_pso,
        model_history_prime: std::sync::atomic::AtomicBool::new(false),
        occlusion_two_pass,
        hiz: compute.hiz,
        prev_view_proj: std::cell::Cell::new(IDENTITY),
        hiz_valid: std::cell::Cell::new(false),
    })
}

// The reflection-probe convolution kernels, under the same gate the bake
// itself needs: a probe capture renders through the bindless GPU cull, so a
// world without the cull PSO never bakes one and never needs them.
//
// The convolution kernels also read their source mip through a UAV, which
// D3D12 allows for this format only under `TypedUAVLoadAdditionalFormats`.
// Without it there is no probe bake and the cube array keeps sampling the sky.
pub(super) fn build_probe_prefilter(
    gpu: &InitGpu<'_>,
    cull: &CullState,
) -> RenderResult<Option<ProbePrefilterPipelines>> {
    let typed_uav_load = crate::directx::probe_prefilter::typed_uav_load_supported(&gpu.hw.device);
    if cull.cull_pso.is_some() && !typed_uav_load {
        tracing::warn!(
            "reflection probes: device lacks TypedUAVLoadAdditionalFormats, skipping probe baking"
        );
    }
    let probe_prefilter = match cull.cull_pso.is_some() && typed_uav_load {
        true => Some(ProbePrefilterPipelines::new(
            &gpu.hw.device,
            gpu.hot_reload,
        )?),
        false => None,
    };
    Ok(probe_prefilter)
}

// GPU-driven instanced merge: write each instance's `GpuObjectData` record
// (+ `GpuDrawArgs`) once into every frame buffer, after the `n_objects`
// static records. Instances are placed at world load and never move, so
// these records are static -- the per-frame static fill (`build_object_buffer`
// / `build_draw_args_buffer`) writes only `[0, n_objects)`, leaving the
// instance tail intact. Only runs when the bindless cull buffers exist (the
// bindless pass is active with build-time geometry) and the world declares
// instanced props.
fn write_instance_records(
    world: &SceneData<'_>,
    plan: &CullPlan,
    albedo_count: usize,
    bindless: &BindlessPass,
    compute: &ComputeCull,
) {
    use concinnity_core::gfx::render_types::{
        GpuDrawArgs, GpuObjectData, draw_args_flags, instance_object_records,
    };
    if plan.n_instances == 0 || bindless.object_ptrs.is_empty() {
        return;
    }
    let n_objects = world.draw_objects.len();
    let records = instance_object_records(&world.instanced_clusters, albedo_count as u32);
    // Cluster base index range (cluster indices are absolute, so
    // base_vertex = 0), which `build_draw_args_buffer` patches per frame
    // for the clusters that declare alternates. Every instance is
    // visible + resident + cullable, so its finite per-instance world AABB
    // is frustum/distance/Hi-Z tested independently by the cull kernel.
    let mut draw_args: Vec<GpuDrawArgs> = Vec::with_capacity(records.len());
    for cluster in &world.instanced_clusters {
        for _ in &cluster.instances {
            draw_args.push(GpuDrawArgs {
                index_count: cluster.index_count as u32,
                index_offset: cluster.index_offset as u32,
                base_vertex: 0,
                flags: draw_args_flags(true, true, true),
            });
        }
    }
    let obj_stride = std::mem::size_of::<GpuObjectData>();
    let da_stride = std::mem::size_of::<GpuDrawArgs>();
    for (obj_ptr, da_ptr) in bindless
        .object_ptrs
        .iter()
        .zip(compute.draw_args_ptrs.iter())
    {
        // SAFETY: the buffers were sized for `n_objects + n_instances`
        // records, so writing `records.len()` past the `n_objects` offset
        // stays in bounds.
        unsafe {
            std::ptr::copy_nonoverlapping(
                records.as_ptr() as *const u8,
                obj_ptr.add(n_objects * obj_stride),
                records.len() * obj_stride,
            );
            std::ptr::copy_nonoverlapping(
                draw_args.as_ptr() as *const u8,
                da_ptr.add(n_objects * da_stride),
                draw_args.len() * da_stride,
            );
        }
    }
}
