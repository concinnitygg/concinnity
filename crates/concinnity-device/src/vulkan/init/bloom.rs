//! Bloom: the write and blend render passes, the prefilter, downsample and
//! upsample pipelines, the per-frame mip chain with its framebuffers, and the
//! dedicated pool the input sets are allocated from.

use ash::vk;
use concinnity_core::gfx::render_types::PostProcessParams;
use concinnity_core::render::error::RenderResult;

use super::InitGpu;
use crate::vulkan::context::{BloomState, HDR_FORMAT, VkTargets};
use crate::vulkan::owned::OwnedSampler;
use crate::vulkan::post::bloom::{
    BloomDeviceContext, MAX_BLOOM_MIPS, alloc_bloom_input_sets, compile_bloom_shaders,
    create_bloom_chain, create_bloom_framebuffers, create_bloom_pipeline, rebind_bloom_input0,
};
use crate::vulkan::post::reflection_composite::ReflectionCompositeResources;
use crate::vulkan::render_pass::create_bloom_render_pass;
use crate::vulkan::resources::create_descriptor_set_layout;

pub(super) struct BloomInputs<'a> {
    pub(super) targets: &'a VkTargets,
    pub(super) extent: vk::Extent2D,
    pub(super) sampler: &'a OwnedSampler,
    pub(super) reflection_composite: Option<&'a ReflectionCompositeResources>,
}

pub(super) fn build_bloom(gpu: &InitGpu<'_>, inputs: BloomInputs<'_>) -> RenderResult<BloomState> {
    let InitGpu {
        hw,
        command_pool,
        frames,
        hot_reload,
    } = *gpu;
    let device = &hw.device;
    let BloomInputs {
        targets,
        extent,
        sampler,
        reflection_composite,
    } = inputs;
    let write_pass = create_bloom_render_pass(device, HDR_FORMAT, false)?;
    let blend_pass = create_bloom_render_pass(device, HDR_FORMAT, true)?;
    // Bloom set (set 0 for every bloom pass): the single input image.
    let set_layout = create_descriptor_set_layout(
        device,
        &[(
            0,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            vk::ShaderStageFlags::FRAGMENT,
        )],
    )?;
    // Post-process push constant: the full `PostProcessParams` struct,
    // fragment-stage. Read by the bloom-prefilter shader.
    let post_pc_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(std::mem::size_of::<PostProcessParams>() as u32);
    // Bloom layout: one descriptor set (the input image) + the shared
    // post-process push constant (read only by the prefilter).
    let set_layouts = [set_layout.handle()];
    let pipeline_layout = device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(std::slice::from_ref(&post_pc_range)),
        )
        .map_err(|e| crate::vulkan::error::map_vk_result(e, "bloom pipeline layout"))?;

    // Bloom pipelines: prefilter, downsample and upsample.
    let bs = compile_bloom_shaders(hot_reload)?;
    let pipeline_prefilter = create_bloom_pipeline(
        device,
        write_pass.handle(),
        pipeline_layout.handle(),
        &bs.vert,
        &bs.prefilter,
        false,
    )?;
    let pipeline_downsample = create_bloom_pipeline(
        device,
        write_pass.handle(),
        pipeline_layout.handle(),
        &bs.vert,
        &bs.downsample,
        false,
    )?;
    // The upsample pipeline targets the LOAD blend pass and blends
    // additively onto the mip already there.
    let pipeline_upsample = create_bloom_pipeline(
        device,
        blend_pass.handle(),
        pipeline_layout.handle(),
        &bs.vert,
        &bs.upsample,
        true,
    )?;

    // Mip 0 binds the transient pool's `bloom_top` image when bloom is on.
    let bloom_top_pairs = targets.transient_pool.pairs_for_frames("bloom_top", frames);
    let (mips, mip_extents) = create_bloom_chain(
        &BloomDeviceContext {
            alloc: &hw.alloc,
            device,
            command_pool,
            queue: hw.graphics_queue,
        },
        extent,
        frames,
        &bloom_top_pairs,
    )?;
    let (write_framebuffers, blend_framebuffers) = create_bloom_framebuffers(
        device,
        write_pass.handle(),
        blend_pass.handle(),
        &mips,
        &mip_extents,
    )?;

    // A dedicated, resettable pool isolates bloom's variable set count
    // (the octave count can shift on resize) from the main pool. Sized for
    // the worst case (`MAX_BLOOM_MIPS + 1` sets per frame).
    let bloom_pool_capacity = frames as u32 * (MAX_BLOOM_MIPS + 1);
    let bloom_pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(bloom_pool_capacity);
    let descriptor_pool = device
        .create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(std::slice::from_ref(&bloom_pool_size))
                .max_sets(bloom_pool_capacity),
        )
        .map_err(|e| crate::vulkan::error::map_vk_result(e, "bloom descriptor pool"))?;
    let input_sets = alloc_bloom_input_sets(
        device,
        descriptor_pool.handle(),
        set_layout.handle(),
        sampler.handle(),
        &targets.hdr_resolve_images,
        &mips,
    )?;
    // The reflection composite replaces the bloom prefilter's scene input
    // (input 0) with its output, the same scene image the composite pass
    // samples when a reflection path is active and TAA is off (a SSGI-only
    // build leaves the prefilter on the raw HDR resolve). One shared image, so
    // every frame's prefilter input 0 points at it.
    if let Some(view) = reflection_composite.map(|c| c.output.view) {
        for frame_sets in &input_sets {
            rebind_bloom_input0(device, frame_sets[0], view, sampler.handle());
        }
    }
    Ok(BloomState {
        write_pass,
        blend_pass,
        pipeline_prefilter,
        pipeline_downsample,
        pipeline_upsample,
        pipeline_layout,
        set_layout,
        descriptor_pool,
        mips,
        mip_extents,
        write_framebuffers,
        blend_framebuffers,
        input_sets,
    })
}
