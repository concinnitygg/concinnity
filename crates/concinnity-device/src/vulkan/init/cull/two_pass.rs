//! Two-pass Hi-Z occlusion: the phase-2 cull pipeline with its dedicated pool
//! and sets, the second indirect buffers, and the phase-1 and phase-2 main
//! render passes.

use ash::vk;
use concinnity_core::gfx::render_types;
use concinnity_core::render::error::RenderResult;

use super::CullPlan;
use super::bindless::BindlessPass;
use super::compute::ComputeCull;
use crate::vulkan::context::HDR_FORMAT;
use crate::vulkan::init::InitGpu;
use crate::vulkan::owned::{OwnedDescriptorPool, OwnedPipeline, OwnedRenderPass};
use crate::vulkan::pipeline::*;
use crate::vulkan::render_pass::create_main_render_pass_two_pass;
use crate::vulkan::resources::alloc_descriptor_sets;

// The two-pass occlusion resources; `None`/empty unless the world requested
// two-pass occlusion and the bindless cull path is active.
pub(super) struct TwoPassCull {
    pub(super) pipeline: Option<OwnedPipeline>,
    pub(super) sets: Vec<vk::DescriptorSet>,
    pub(super) pool: Option<OwnedDescriptorPool>,
    pub(super) indirect_buffers: Vec<crate::vulkan::allocator::PooledBuffer>,
    pub(super) main_render_pass_phase1: Option<OwnedRenderPass>,
    pub(super) main_render_pass_phase2: Option<OwnedRenderPass>,
}

// Build the two-pass Hi-Z occlusion resources: the phase-2 cull pipeline and
// sets, the second indirect buffers, and the phase-1 and phase-2 render passes.
pub(super) fn build_two_pass_cull(
    gpu: &InitGpu<'_>,
    bindless: &BindlessPass,
    compute: &ComputeCull,
    plan: &CullPlan,
    shader_bucket_count: usize,
    msaa_samples: vk::SampleCountFlags,
    occlusion_two_pass: bool,
) -> RenderResult<TwoPassCull> {
    let InitGpu {
        hw,
        frames,
        hot_reload,
        ..
    } = *gpu;
    let (device, alloc) = (&hw.device, &hw.alloc);
    let n_frames = frames as u32;
    let (object_buffers, draw_args_buffers, cull_status_buffers) = (
        &bindless.object_buffers,
        &compute.draw_args_buffers,
        &compute.status_buffers,
    );
    // Two-pass Hi-Z occlusion resources. Built only when the world
    // requested `occlusion_two_pass` AND the bindless cull path is active:
    // the phase-2 cull pipeline (`main_phase2`, same layout as phase 1), a
    // second set of per-frame indirect buffers `Cull2` writes / `Main2`
    // reads, a dedicated descriptor pool + per-frame phase-2 cull sets
    // (bindings 0/1/2/3 = object / draw-args / second-indirect /
    // cull-status), and the phase-1/phase-2 main render passes. The Hi-Z
    // phase-2 cull-read sets live inside `HiZResources` (built with the cull
    // pass when `occlusion_two_pass`). Mirrors `directx/init/pipelines.rs`.
    type TwoPassCullResources = (
        Option<OwnedPipeline>,
        Vec<vk::DescriptorSet>,
        Option<OwnedDescriptorPool>,
        Vec<crate::vulkan::allocator::PooledBuffer>,
        Option<OwnedRenderPass>,
        Option<OwnedRenderPass>,
    );
    let (
        cull_pipeline_phase2,
        cull_sets2,
        two_pass_pool,
        indirect_buffers2,
        main_render_pass_phase1,
        main_render_pass_phase2,
    ): TwoPassCullResources = if let (Some(set_layout), Some(pipeline_layout)) = (
        compute.set_layout.as_ref(),
        compute.pipeline_layout.as_ref(),
    ) && occlusion_two_pass
    {
        let n = plan.n_cull as u64;
        let object_buffer_size = n * std::mem::size_of::<render_types::GpuObjectData>() as u64;
        let draw_args_size = n * std::mem::size_of::<render_types::GpuDrawArgs>() as u64;
        // Bucket-expanded exactly like the phase-1 buffers: `Main2` issues the
        // same per-bucket regions over this buffer.
        let indirect_size = shader_bucket_count as u64
            * n
            * std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u64;
        let status_size = n * std::mem::size_of::<u32>() as u64;

        // Phase-2 cull pipeline (`main_phase2` entry, shared layout).
        let cs2 = compile_cull_shader_phase2(hot_reload)?;
        let pipeline2 = create_cull_pipeline(device, pipeline_layout.handle(), &cs2)?;

        // Second indirect-command buffers (device-local, GPU-written).
        let mut ind2_buffers = Vec::with_capacity(frames);
        for _ in 0..frames {
            ind2_buffers.push(alloc.create_buffer(
                indirect_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?);
        }

        // Dedicated descriptor pool for the per-frame phase-2 cull sets
        // (4 storage buffers each), kept off the shared pool's exact sizing.
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(4 * n_frames);
        let pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(std::slice::from_ref(&pool_size))
                    .max_sets(n_frames),
            )
            .map_err(|e| format!("two-pass cull descriptor pool: {e}"))?;
        let set_layouts2: Vec<_> = (0..frames).map(|_| set_layout.handle()).collect();
        let sets2 = alloc_descriptor_sets(device, pool.handle(), &set_layouts2)?;
        for (i, &set) in sets2.iter().enumerate() {
            let obj_info = vk::DescriptorBufferInfo::default()
                .buffer(object_buffers[i].buffer())
                .offset(0)
                .range(object_buffer_size);
            let arg_info = vk::DescriptorBufferInfo::default()
                .buffer(draw_args_buffers[i].buffer())
                .offset(0)
                .range(draw_args_size);
            // Binding 2: the *second* indirect buffer (Cull2 writes it).
            let cmd_info = vk::DescriptorBufferInfo::default()
                .buffer(ind2_buffers[i].buffer())
                .offset(0)
                .range(indirect_size);
            // Binding 3: the cull-status buffer (phase 1 wrote it; read here).
            let status_info = vk::DescriptorBufferInfo::default()
                .buffer(cull_status_buffers[i].buffer())
                .offset(0)
                .range(status_size);
            let infos = [obj_info, arg_info, cmd_info, status_info];
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(b, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(b as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        // Phase-1 (STORE MSAA color) + phase-2 (LOAD color + depth) main
        // render passes, both compatible with the existing framebuffers.
        let rp1 = create_main_render_pass_two_pass(device, HDR_FORMAT, msaa_samples, false)?;
        let rp2 = create_main_render_pass_two_pass(device, HDR_FORMAT, msaa_samples, true)?;

        (
            Some(pipeline2),
            sets2,
            Some(pool),
            ind2_buffers,
            Some(rp1),
            Some(rp2),
        )
    } else {
        (None, Vec::new(), None, Vec::new(), None, None)
    };
    Ok(TwoPassCull {
        pipeline: cull_pipeline_phase2,
        sets: cull_sets2,
        pool: two_pass_pool,
        indirect_buffers: indirect_buffers2,
        main_render_pass_phase1,
        main_render_pass_phase2,
    })
}
