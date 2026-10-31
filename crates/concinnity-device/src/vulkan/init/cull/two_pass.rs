//! Two-pass Hi-Z occlusion: the late cull pipeline with its dedicated pool
//! and sets, the second indirect buffers, and the early pass and late main
//! render passes.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types;
use concinnity_engine::render::backend::error::RenderResult;

use super::CullPlan;
use super::bindless::BindlessPass;
use super::compute::{ComputeCull, main_cull_set};
use crate::vulkan::context::HDR_FORMAT;
use crate::vulkan::descriptor_layout::PoolSizes;
use crate::vulkan::init::InitGpu;
use crate::vulkan::owned::{OwnedDescriptorPool, OwnedPipeline, OwnedRenderPass};
use crate::vulkan::pipeline::compile_cull_shader_late;
use crate::vulkan::pipeline_desc::compute_pipeline;
use crate::vulkan::render_pass::create_main_render_pass_two_pass;
use crate::vulkan::resources::alloc_descriptor_sets;
use crate::vulkan::set_writes::SetWrites;

// The two-pass occlusion resources; `None`/empty unless the world requested
// two-pass occlusion and the bindless cull path is active.
pub(super) struct TwoPassCull {
    pub(super) pipeline: Option<OwnedPipeline>,
    pub(super) sets: Vec<vk::DescriptorSet>,
    pub(super) pool: Option<OwnedDescriptorPool>,
    pub(super) indirect_buffers: Vec<crate::vulkan::allocator::PooledBuffer>,
    // Each frame's late packing set, from `pool`.
    pub(super) compact_sets: Vec<vk::DescriptorSet>,
    pub(super) main_render_pass_early: Option<OwnedRenderPass>,
    pub(super) main_render_pass_late: Option<OwnedRenderPass>,
}

// Build the two-pass Hi-Z occlusion resources: the late cull pipeline and
// sets, the second indirect buffers, and the early pass and late render passes.
pub(super) fn build_two_pass_cull(
    gpu: &InitGpu<'_>,
    bindless: &BindlessPass,
    compute: &ComputeCull,
    plan: &CullPlan,
    indirect_commands: usize,
    msaa_samples: vk::SampleCountFlags,
    occlusion_two_pass: bool,
) -> RenderResult<TwoPassCull> {
    let InitGpu {
        hw,
        command_pool,
        frames,
        hot_reload,
    } = *gpu;
    let (device, alloc) = (&hw.device, &hw.alloc);
    let n_frames = u32::saturating_from(frames);
    let (object_buffers, draw_args_buffers, cull_status_buffers) = (
        &bindless.object_buffers,
        &compute.draw_args_buffers,
        &compute.status_buffers,
    );
    // Two-pass Hi-Z occlusion resources. Built only when the world
    // requested `occlusion_two_pass` AND the bindless cull path is active:
    // the late cull pipeline (`CULL_LATE`, same layout as the early pass), a
    // second set of per-frame indirect buffers `CullLate` writes / `MainLate`
    // reads, a dedicated descriptor pool + per-frame late cull sets
    // (bindings 0/1/2/3 = object / draw-args / second-indirect /
    // cull-status), and the early and late main render passes. The Hi-Z
    // late cull-read sets live inside `HiZResources` (built with the cull
    // pass when `occlusion_two_pass`). Mirrors `directx/init/pipelines.rs`.
    type TwoPassCullResources = (
        Option<OwnedPipeline>,
        Vec<vk::DescriptorSet>,
        Option<OwnedDescriptorPool>,
        Vec<crate::vulkan::allocator::PooledBuffer>,
        Vec<vk::DescriptorSet>,
        Option<OwnedRenderPass>,
        Option<OwnedRenderPass>,
    );
    let (
        cull_pipeline_late,
        cull_sets_late,
        two_pass_pool,
        indirect_buffers_late,
        compact_sets_late,
        main_render_pass_early,
        main_render_pass_late,
    ): TwoPassCullResources = if let Some(kernels) = compute.kernels.as_ref()
        && let Some(compact) = compute.compact.as_ref()
        && occlusion_two_pass
    {
        let (set_layout, pipeline_layout) = (&kernels.set_layout, &kernels.pipeline_layout);
        let n = plan.n_cull as u64;
        let object_buffer_size = n * size_of::<render_types::GpuObjectData>() as u64;
        let draw_args_size = n * size_of::<render_types::GpuDrawArgs>() as u64;
        // Laid out exactly like the early buffers: `MainLate` issues the same
        // per-stream lists over this buffer.
        let indirect_size = crate::vulkan::compact::indirect_buffer_bytes(indirect_commands);
        let group_count_size = crate::vulkan::compact::group_count_bytes(plan.n_cull);
        let status_size = n * size_of::<u32>() as u64;

        // Late cull pipeline (`CULL_LATE`, shared layout).
        let cs_late = compile_cull_shader_late(hot_reload)?;
        let late_pipeline =
            compute_pipeline(device, pipeline_layout.handle(), &cs_late, "cull late")?;

        // Second indirect-command buffers (device-local, GPU-written).
        let mut ind2_buffers = Vec::with_capacity(frames);
        for _ in 0..frames {
            ind2_buffers.push(alloc.create_buffer(
                indirect_size,
                crate::vulkan::compact::INDIRECT_USAGE,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?);
        }
        crate::vulkan::compact::zero_indirect_buffers(
            device,
            command_pool,
            hw.graphics_queue,
            &ind2_buffers,
        )?;

        // Dedicated descriptor pool for the per-frame late cull and
        // packing sets, kept off the shared pool's exact sizing.
        let pool_sizes = two_pass_pool_sizes(n_frames).build();
        let pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(2 * n_frames),
            )
            .map_err(|e| crate::vulkan::error::map_vk_result(e, "two-pass cull descriptor pool"))?;
        let late_set_layouts: Vec<_> = (0..frames).map(|_| set_layout.handle()).collect();
        let late_sets = alloc_descriptor_sets(device, pool.handle(), &late_set_layouts)?;
        for (i, &set) in late_sets.iter().enumerate() {
            // Binding 2 is the frame's group counts, which the late pass reuses once
            // the early pass's packing has read them; binding 3 the cull status the
            // early pass wrote.
            SetWrites::new(set)
                .storage_buffer(0, object_buffers[i].buffer(), object_buffer_size)
                .storage_buffer(1, draw_args_buffers[i].buffer(), draw_args_size)
                .storage_buffer(2, compute.group_count_buffers[i].buffer(), group_count_size)
                .storage_buffer(3, cull_status_buffers[i].buffer(), status_size)
                .storage_buffer(
                    4,
                    compute.variant_buffers[i].buffer(),
                    super::compute::VARIANT_BUFFER_BYTES,
                )
                .apply(device);
        }

        let compact_layouts: Vec<_> = (0..frames).map(|_| compact.set_layout.handle()).collect();
        let compact_sets_late = alloc_descriptor_sets(device, pool.handle(), &compact_layouts)?;
        for (i, &set) in compact_sets_late.iter().enumerate() {
            crate::vulkan::compact::write_compact_set(
                device,
                set,
                &crate::vulkan::compact::CompactBuffers {
                    variants: (
                        compute.variant_buffers[i].buffer(),
                        super::compute::VARIANT_BUFFER_BYTES,
                    ),
                    draw_args: (draw_args_buffers[i].buffer(), draw_args_size),
                    status: (cull_status_buffers[i].buffer(), status_size),
                    group_counts: (compute.group_count_buffers[i].buffer(), group_count_size),
                    lists: crate::vulkan::compact::DrawLists {
                        buffer: ind2_buffers[i].buffer(),
                        commands: indirect_commands,
                    },
                },
            );
        }

        // Early (STORE MSAA color) + late (LOAD color + depth) main
        // render passes, both compatible with the existing framebuffers.
        let rp1 = create_main_render_pass_two_pass(device, HDR_FORMAT, msaa_samples, false)?;
        let late_rp = create_main_render_pass_two_pass(device, HDR_FORMAT, msaa_samples, true)?;

        (
            Some(late_pipeline),
            late_sets,
            Some(pool),
            ind2_buffers,
            compact_sets_late,
            Some(rp1),
            Some(late_rp),
        )
    } else {
        (None, Vec::new(), None, Vec::new(), Vec::new(), None, None)
    };
    Ok(TwoPassCull {
        pipeline: cull_pipeline_late,
        sets: cull_sets_late,
        pool: two_pass_pool,
        indirect_buffers: indirect_buffers_late,
        compact_sets: compact_sets_late,
        main_render_pass_early,
        main_render_pass_late,
    })
}

// The two-pass pool: each frame's late cull set and packing set.
fn two_pass_pool_sizes(frames: u32) -> PoolSizes {
    PoolSizes::default()
        .sets(&main_cull_set(), frames)
        .sets(&crate::vulkan::compact::compact_set(), frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_pass_pool_covers_its_sets() {
        assert!(two_pass_pool_sizes(3).covers(&[
            (&main_cull_set(), 3),
            (&crate::vulkan::compact::compact_set(), 3),
        ]));
    }
}
