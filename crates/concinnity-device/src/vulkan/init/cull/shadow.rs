//! The GPU-driven cascade shadow pass: the shadow cull kernel with its
//! per-(frame, cascade) sets and indirect buffers, and the depth-only bindless
//! pipeline.

use ash::vk;
use concinnity_core::gfx::render_types;
use concinnity_core::render::error::RenderResult;

use super::CullPlan;
use super::bindless::BindlessPass;
use super::compute::ComputeCull;
use crate::vulkan::context::{VkDescriptors, VkShadow};
use crate::vulkan::init::InitGpu;
use crate::vulkan::owned::{OwnedPipeline, OwnedPipelineLayout, OwnedSetLayout};
use crate::vulkan::pipeline::*;
use crate::vulkan::resources::alloc_descriptor_sets;

// The GPU-driven shadow pass; `None`/empty unless the bindless cull path is
// active and shadows are enabled.
pub(super) struct ShadowCull {
    pub(super) cull_pipeline: Option<OwnedPipeline>,
    pub(super) cull_pipeline_layout: Option<OwnedPipelineLayout>,
    pub(super) set_layout: Option<OwnedSetLayout>,
    pub(super) sets: Vec<Vec<vk::DescriptorSet>>,
    pub(super) bindless_pipeline: Option<OwnedPipeline>,
    pub(super) bindless_pipeline_layout: Option<OwnedPipelineLayout>,
    pub(super) indirect_buffers: Vec<Vec<crate::vulkan::allocator::PooledBuffer>>,
}

// Build the GPU-driven shadow pass: a frustum + distance cull per cascade and
// the depth-only bindless pipeline that draws its survivors.
pub(super) fn build_shadow_cull(
    gpu: &InitGpu<'_>,
    bindless: &BindlessPass,
    compute: &ComputeCull,
    shadow: &VkShadow,
    descriptors: &VkDescriptors,
    plan: &CullPlan,
) -> RenderResult<ShadowCull> {
    let InitGpu {
        hw,
        frames,
        hot_reload,
        ..
    } = *gpu;
    let (device, alloc) = (&hw.device, &hw.alloc);
    let (object_buffers, draw_args_buffers) =
        (&bindless.object_buffers, &compute.draw_args_buffers);
    // GPU-driven shadow pass resources. Built when the bindless cull path is
    // active AND shadows are enabled: a frustum + distance only cull pipeline
    // (`SHADOW_CULL`, lean 3-SSBO set: objects + draw-args + this cascade's
    // indirect buffer), one indirect buffer + cull set per (frame, cascade),
    // and a depth-only bindless graphics pipeline (shadow-global set 0 + the
    // bindless GpuObjectData set 1 + a cascade-index push constant). Each
    // re-rendered cascade then runs one cull dispatch + one
    // `cmd_draw_indexed_indirect` (static + instance prefix) + one for the
    // skinned tail, replacing the CPU per-object shadow loop.
    type ShadowCullResources = (
        Option<OwnedPipeline>,
        Option<OwnedPipelineLayout>,
        Option<OwnedSetLayout>,
        Vec<Vec<vk::DescriptorSet>>,
        Option<OwnedPipeline>,
        Option<OwnedPipelineLayout>,
        Vec<Vec<crate::vulkan::allocator::PooledBuffer>>,
    );
    let (
        shadow_cull_pipeline,
        shadow_cull_pipeline_layout,
        shadow_cull_set_layout,
        shadow_cull_sets,
        shadow_bindless_pipeline,
        shadow_bindless_pipeline_layout,
        shadow_indirect_buffers,
    ): ShadowCullResources = if plan.bindless_active
        && shadow.pipeline.is_some()
        && let Some(bl_set_layout) = bindless.set_layout.as_ref()
    {
        let cascades = render_types::NUM_SHADOW_CASCADES;
        // Lean shadow cull set layout: objects(0) + draw-args(1) + commands(2).
        let sc_bindings: Vec<_> = (0..3u32)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let sc_set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&sc_bindings),
            )
            .map_err(|e| crate::vulkan::error::map_vk_result(e, "shadow cull set layout"))?;

        let sc_push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(CULL_PUSH_CONSTANT_BYTES);
        let sc_layouts = [sc_set_layout.handle()];
        let sc_pl = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&sc_layouts)
                    .push_constant_ranges(std::slice::from_ref(&sc_push)),
            )
            .map_err(|e| crate::vulkan::error::map_vk_result(e, "shadow cull pipeline layout"))?;
        let sc_spv = compile_shadow_cull_shader(hot_reload)?;
        let sc_pipeline = create_cull_pipeline(device, sc_pl.handle(), &sc_spv)?;

        // Depth-only bindless shadow graphics pipeline: shadow-global set 0 +
        // the bindless GpuObjectData set 1 + a cascade-index push constant.
        let sb_push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(4);
        let shadow_global_set_layout = &shadow.global_set_layout;
        let sb_layouts = [shadow_global_set_layout.handle(), bl_set_layout.handle()];
        let sb_pl = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&sb_layouts)
                    .push_constant_ranges(std::slice::from_ref(&sb_push)),
            )
            .map_err(|e| {
                crate::vulkan::error::map_vk_result(e, "shadow bindless pipeline layout")
            })?;
        let sb_spv = compile_shadow_bindless_vs(hot_reload)?;
        let sb_pipeline =
            create_shadow_pipeline(device, shadow.render_pass.handle(), sb_pl.handle(), &sb_spv)?;

        // Per-(frame, cascade) indirect buffers + cull sets. Each cull set
        // binds this frame's object + draw-args SSBOs and this cascade's
        // indirect buffer; the cull dispatch for cascade `c` binds set
        // `[frame][c]`, and the cascade's draws read buffer `[frame][c]`.
        let n = plan.n_cull as u64;
        let object_buffer_size = n * std::mem::size_of::<render_types::GpuObjectData>() as u64;
        let draw_args_size = n * std::mem::size_of::<render_types::GpuDrawArgs>() as u64;
        let indirect_size = n * std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u64;
        let mut sc_indirect_bufs: Vec<Vec<crate::vulkan::allocator::PooledBuffer>> =
            Vec::with_capacity(frames);
        let mut sc_sets: Vec<Vec<vk::DescriptorSet>> = Vec::with_capacity(frames);
        for f in 0..frames {
            let mut bufs = Vec::with_capacity(cascades);
            for _ in 0..cascades {
                bufs.push(alloc.create_buffer(
                    indirect_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )?);
            }
            let set_layouts: Vec<_> = (0..cascades).map(|_| sc_set_layout.handle()).collect();
            let sets =
                alloc_descriptor_sets(device, descriptors.descriptor_pool.handle(), &set_layouts)?;
            for (c, &set) in sets.iter().enumerate() {
                let obj_info = vk::DescriptorBufferInfo::default()
                    .buffer(object_buffers[f].buffer())
                    .offset(0)
                    .range(object_buffer_size);
                let arg_info = vk::DescriptorBufferInfo::default()
                    .buffer(draw_args_buffers[f].buffer())
                    .offset(0)
                    .range(draw_args_size);
                let cmd_info = vk::DescriptorBufferInfo::default()
                    .buffer(bufs[c].buffer())
                    .offset(0)
                    .range(indirect_size);
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&obj_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&arg_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&cmd_info)),
                ];
                // SAFETY: `writes` and the buffer/image infos it borrows are live for the call,
                // and every set and resource it names belongs to this device.
                unsafe { device.update_descriptor_sets(&writes, &[]) };
            }
            sc_indirect_bufs.push(bufs);
            sc_sets.push(sets);
        }

        (
            Some(sc_pipeline),
            Some(sc_pl),
            Some(sc_set_layout),
            sc_sets,
            Some(sb_pipeline),
            Some(sb_pl),
            sc_indirect_bufs,
        )
    } else {
        (None, None, None, Vec::new(), None, None, Vec::new())
    };
    Ok(ShadowCull {
        cull_pipeline: shadow_cull_pipeline,
        cull_pipeline_layout: shadow_cull_pipeline_layout,
        set_layout: shadow_cull_set_layout,
        sets: shadow_cull_sets,
        bindless_pipeline: shadow_bindless_pipeline,
        bindless_pipeline_layout: shadow_bindless_pipeline_layout,
        indirect_buffers: shadow_indirect_buffers,
    })
}
