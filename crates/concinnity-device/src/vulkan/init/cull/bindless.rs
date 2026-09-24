//! The bindless static main pass: its set and pipeline layouts, bucket 0's
//! pipeline, the per-frame object buffers and bindless sets, and one pipeline
//! per material-referenced shader bucket.

use ash::vk;
use concinnity_core::gfx::render_types;
use concinnity_core::render::backend_init::WorldShader;
use concinnity_core::render::error::{RenderError, RenderResult};

use super::CullPlan;
use crate::vulkan::context::{VkDescriptors, VkSceneAssets, VkTargets};
use crate::vulkan::init::InitGpu;
use crate::vulkan::owned::{OwnedPipeline, OwnedPipelineLayout, OwnedSetLayout};
use crate::vulkan::pipeline::*;
use crate::vulkan::resources::alloc_descriptor_sets;

// The bindless static pass; `None`/empty when the bindless path is inactive.
pub(super) struct BindlessPass {
    pub(super) pipeline: Option<OwnedPipeline>,
    pub(super) pipeline_layout: Option<OwnedPipelineLayout>,
    pub(super) set_layout: Option<OwnedSetLayout>,
    pub(super) sets: Vec<vk::DescriptorSet>,
    pub(super) object_buffers: Vec<crate::vulkan::allocator::PooledBuffer>,
    pub(super) main_spv: (Vec<u8>, Vec<u8>),
}

pub(super) struct BindlessInputs<'a> {
    pub(super) world_shaders: &'a [WorldShader<'a>],
    pub(super) plan: &'a CullPlan,
    pub(super) descriptors: &'a VkDescriptors,
    pub(super) targets: &'a VkTargets,
    pub(super) scene: &'a VkSceneAssets,
    pub(super) swapchain_format: vk::Format,
}

// Build the bindless static pass: bucket 0's pipeline, the per-frame object
// buffers, and one bindless set per frame.
pub(super) fn build_bindless_pass(
    gpu: &InitGpu<'_>,
    inputs: BindlessInputs<'_>,
) -> RenderResult<BindlessPass> {
    let InitGpu {
        hw,
        frames,
        hot_reload,
        ..
    } = *gpu;
    let (device, alloc) = (&hw.device, &hw.alloc);
    let BindlessInputs {
        world_shaders,
        plan,
        descriptors,
        targets,
        scene,
        swapchain_format,
    } = inputs;
    let CullPlan {
        n_cull,
        bindless_active,
        bindless_pool_size,
        bindless_uab,
        ..
    } = *plan;
    // Bindless static pass: bindless static main pass resources. A dedicated
    // set layout (set 1: SSBO + bindless texture pool), pipeline layout,
    // pipeline, per-frame GpuObjectData storage buffers, and one descriptor
    // set per frame. `None`/empty when the bindless pass is inactive.
    let (
        bindless_pipeline,
        bindless_pipeline_layout,
        bindless_set_layout,
        bindless_sets,
        object_buffers,
        bindless_main_spv,
    ) = if bindless_active {
        let set_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(bindless_pool_size as u32)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        // On a budget-constrained device the pool binding is declared
        // update-after-bind so it budgets against the update-after-bind
        // sampled-image limit instead of the plain one. The pool is written once at
        // init, before any frame binds it, so nothing depends on the relaxed
        // update timing itself: this is purely how the layout is budgeted.
        let binding_flags = [
            vk::DescriptorBindingFlags::empty(),
            vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
        ];
        let mut flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
        let mut set_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&set_bindings);
        if bindless_uab {
            set_info = set_info
                .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                .push_next(&mut flags_info);
        }
        let set_layout = device
            .create_descriptor_set_layout(&set_info)
            .map_err(|e| crate::vulkan::error::map_vk_result(e, "bindless set layout"))?;

        let layouts = [descriptors.global_set_layout.handle(), set_layout.handle()];
        let pipeline_layout = device
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts))
            .map_err(|e| crate::vulkan::error::map_vk_result(e, "bindless pipeline layout"))?;

        // The engine's own pair is the program for every bucket that
        // declares no Shader and the source of the Wireframe twin; bucket 0
        // takes the world default Shader's pair where it declares one.
        let engine_pair = compile_bindless_shaders(hot_reload)?;
        let pipeline = build_bucket_pipeline(
            device,
            BucketPipelineTargets {
                render_pass: targets.main_render_pass.handle(),
                layout: pipeline_layout.handle(),
                msaa_samples: targets.msaa_samples,
                swapchain_format,
                hot_reload,
            },
            0,
            world_shaders[0],
            &engine_pair,
        )?;

        // Per-frame GpuObjectData storage buffers, persistently mapped.
        // Sized for `n_cull` so the instanced merge's records fit past the
        // `n_objects` static prefix.
        let object_buffer_size =
            (n_cull * std::mem::size_of::<render_types::GpuObjectData>()) as u64;
        let mut buffers = Vec::with_capacity(frames);
        for _ in 0..frames {
            buffers.push(alloc.create_buffer(
                object_buffer_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?);
        }

        // One bindless set per frame: binding 0 = that frame's SSBO,
        // binding 1 = the shared pool ([albedo views..] ++ [normal..]), read
        // through the global set's linear sampler.
        let set_layouts: Vec<_> = (0..frames).map(|_| set_layout.handle()).collect();
        let sets =
            alloc_descriptor_sets(device, descriptors.descriptor_pool.handle(), &set_layouts)?;
        // Every slot the layout declares has to be written, and at the
        // ceiling there are more of them than the world fills: the world's
        // textures, then the reserved fallbacks (flat-normal, white), then
        // white again across the unused tail so a slot the shader can index
        // still names a live view. Mirrors the Metal pool's fill. Sizing
        // guarantees the world fits, so this only ever pads.
        let mut pool_infos: Vec<vk::DescriptorImageInfo> = scene
            .textures
            .iter()
            .chain(scene.fallback_textures.iter())
            .map(|img| {
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(img.view)
            })
            .collect();
        if let Some(&tail) = pool_infos.last() {
            pool_infos.resize(bindless_pool_size, tail);
        }
        for (i, &set) in sets.iter().enumerate() {
            let buf_info = vk::DescriptorBufferInfo::default()
                .buffer(buffers[i].buffer())
                .offset(0)
                .range(object_buffer_size);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buf_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&pool_infos),
            ];
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        (
            Some(pipeline),
            Some(pipeline_layout),
            Some(set_layout),
            sets,
            buffers,
            engine_pair,
        )
    } else {
        (
            None,
            None,
            None,
            Vec::new(),
            Vec::new(),
            (Vec::new(), Vec::new()),
        )
    };
    Ok(BindlessPass {
        pipeline: bindless_pipeline,
        pipeline_layout: bindless_pipeline_layout,
        set_layout: bindless_set_layout,
        sets: bindless_sets,
        object_buffers,
        main_spv: bindless_main_spv,
    })
}

// Material-referenced shaders (ShaderHandle 1..) each get their own
// bindless main-pass pipeline, so their draws route into their own region
// of the GPU-culled command buffer.
pub(super) fn build_world_pipelines(
    gpu: &InitGpu<'_>,
    bindless: &BindlessPass,
    world_shaders: &[WorldShader<'_>],
    targets: &VkTargets,
    swapchain_format: vk::Format,
) -> RenderResult<Vec<Option<OwnedPipeline>>> {
    let InitGpu { hw, hot_reload, .. } = *gpu;
    let bucket_shaders = world_shaders.get(1..).unwrap_or(&[]);
    Ok(
        match (bindless.pipeline_layout.as_ref(), bucket_shaders.is_empty()) {
            (Some(layout), false) => {
                let max = render_types::MAX_SHADER_BUCKETS;
                if bucket_shaders.len() + 1 > max {
                    return Err(RenderError::Other(format!(
                        "world declares {} Shaders but at most {max} can be routed",
                        bucket_shaders.len() + 1
                    )));
                }
                build_world_pipeline_table(
                    &hw.device,
                    BucketPipelineTargets {
                        render_pass: targets.main_render_pass.handle(),
                        layout: layout.handle(),
                        msaa_samples: targets.msaa_samples,
                        swapchain_format,
                        hot_reload,
                    },
                    bucket_shaders,
                    &bindless.main_spv,
                )?
            }
            _ => Vec::new(),
        },
    )
}
