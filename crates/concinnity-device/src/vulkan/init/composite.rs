//! Composite: the swapchain render pass and framebuffers, the tonemap pipeline,
//! and one input set per frame in flight.

use ash::vk;
use concinnity_core::gfx::render_types::CompositeParams;
use concinnity_core::render::error::RenderResult;

use super::InitGpu;
use crate::vulkan::context::{
    BloomState, CompositeState, SwapchainState, VkDescriptors, VkSceneAssets, VkTargets,
};
use crate::vulkan::owned::OwnedSampler;
use crate::vulkan::pipeline::{compile_composite_shaders, create_composite_pipeline};
use crate::vulkan::post::reflection_composite::ReflectionCompositeResources;
use crate::vulkan::render_pass::create_composite_render_pass;
use crate::vulkan::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use crate::vulkan::swapchain::{create_composite_framebuffers, write_composite_set};

pub(super) struct CompositeInputs<'a> {
    pub(super) swapchain: &'a SwapchainState,
    pub(super) descriptors: &'a VkDescriptors,
    pub(super) bloom: &'a BloomState,
    pub(super) scene: &'a VkSceneAssets,
    pub(super) targets: &'a VkTargets,
    pub(super) sampler: &'a OwnedSampler,
    pub(super) reflection_composite: Option<&'a ReflectionCompositeResources>,
}

pub(super) fn build_composite(
    gpu: &InitGpu<'_>,
    inputs: CompositeInputs<'_>,
) -> RenderResult<CompositeState> {
    let InitGpu {
        hw,
        frames,
        hot_reload,
        ..
    } = *gpu;
    let device = &hw.device;
    let CompositeInputs {
        swapchain,
        descriptors,
        bloom,
        scene,
        targets,
        sampler,
        reflection_composite,
    } = inputs;
    let render_pass = create_composite_render_pass(device, swapchain.format)?;
    let framebuffers = create_composite_framebuffers(
        device,
        render_pass.handle(),
        &swapchain.image_views,
        swapchain.extent,
    )?;
    // Composite set (set 0 for composite pass): HDR resolve image at
    // binding 0, bloom mip 0 at binding 1, the 3D color LUT at binding 2,
    // then the G-buffer channels the debug view modes visualize (3 =
    // normal+depth, 4 = roughness, 5 = SSAO occlusion).
    let set_layout = create_descriptor_set_layout(
        device,
        &(0..6)
            .map(|b| {
                (
                    b,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ShaderStageFlags::FRAGMENT,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    // The composite shader reads the post-process tunables plus the scene fade,
    // so its push constant covers the wider `CompositeParams`.
    let composite_pc_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(std::mem::size_of::<CompositeParams>() as u32);
    let set_layouts = [set_layout.handle()];
    let pipeline_layout = device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(std::slice::from_ref(&composite_pc_range)),
        )
        .map_err(|e| format!("composite pipeline layout: {e}"))?;
    let pipeline = {
        let (cv, cf) = compile_composite_shaders(hot_reload)?;
        create_composite_pipeline(
            device,
            render_pass.handle(),
            pipeline_layout.handle(),
            &cv,
            &cf,
        )?
    };

    // Composite sets (one per frame-in-flight slot): binding 0 = the
    // scene image (SSR output when SSR is on, else this slot's HDR
    // resolve), binding 1 = that slot's bloom mip 0, binding 2 = the
    // shared 3D color LUT. The TAA wiring later overrides binding 0 to
    // the TAA output when TAA is on.
    let composite_layouts: Vec<_> = (0..frames).map(|_| set_layout.handle()).collect();
    let sets = alloc_descriptor_sets(
        device,
        descriptors.descriptor_pool.handle(),
        &composite_layouts,
    )?;
    for (i, &set) in sets.iter().enumerate() {
        // Scene image: the reflection composite output (the SSR / RT reflection
        // blended over the scene) when a reflection path is active, else the raw
        // HDR resolve (a SSGI-only build composited its bounce into the latter
        // upstream). The TAA / upscale wiring overrides this later.
        let scene_view = reflection_composite
            .map(|c| c.output.view)
            .unwrap_or(targets.hdr_resolve_images[i].view);
        write_composite_set(
            device,
            set,
            scene_view,
            bloom.mips[i][0].view,
            scene.color_lut.view,
            sampler.handle(),
        );
    }
    Ok(CompositeState {
        render_pass,
        framebuffers,
        pipeline,
        pipeline_layout,
        _set_layout: set_layout,
        sets,
    })
}
