//! Composite: the swapchain formats, the tonemap pipeline,
//! and one input set per frame in flight.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types::CompositeParams;
use concinnity_engine::render::backend::error::RenderResult;

use super::InitGpu;
use crate::vulkan::context::{CompositeState, Descriptors, SceneAssets, SwapchainState, Targets};
use crate::vulkan::owned::OwnedSampler;
use crate::vulkan::pipeline::{compile_composite_shaders, create_composite_pipeline};
use crate::vulkan::post::bloom::composite_bloom_view;
use crate::vulkan::post::reflection_composite::VkReflectionCompositePass;
use crate::vulkan::rendering::PassFormats;
use crate::vulkan::resources::{
    alloc_descriptor_sets, create_descriptor_set_layout, source_set_bindings,
};
use crate::vulkan::set_writes::SetWrites;
use crate::vulkan::swapchain::write_composite_set;

// Sources the composite fragment samples, each through a sampler of its own.
pub(super) const COMPOSITE_SOURCES: u32 = 8;

#[derive(Clone, Copy)]
pub(super) struct CompositeInputs<'a> {
    pub(super) swapchain: &'a SwapchainState,
    pub(super) descriptors: &'a Descriptors,
    pub(super) scene: &'a SceneAssets,
    pub(super) targets: &'a Targets,
    pub(super) sampler: &'a OwnedSampler,
    pub(super) reflection_composite: Option<&'a VkReflectionCompositePass>,
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
        scene,
        targets,
        sampler,
        reflection_composite,
    } = inputs;
    let formats = PassFormats::color(swapchain.format);
    // Composite set 0: scene (0), bloom top octave (1), color LUT (2), the debug
    // view channels (3-7), and each source's sampler at 8 + its index.
    let set_layout = create_descriptor_set_layout(device, &source_set_bindings(COMPOSITE_SOURCES))?;
    // The composite shader reads the post-process tunables plus the scene fade,
    // so its push constant covers the wider `CompositeParams`.
    let composite_pc_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(u32::saturating_from(size_of::<CompositeParams>()));
    let set_layouts = [set_layout.handle()];
    let pipeline_layout = device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(std::slice::from_ref(&composite_pc_range)),
        )
        .map_err(|e| crate::vulkan::error::map_vk_result(e, "composite pipeline layout"))?;
    let pipeline = {
        let (cv, cf) = compile_composite_shaders(hot_reload)?;
        create_composite_pipeline(device, formats, pipeline_layout.handle(), &cv, &cf)?
    };

    // One composite set per frame in flight; TAA later re-points binding 0.
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
        let scene_view =
            reflection_composite.map_or(targets.hdr_resolve_images[i].view, |c| c.output().view());
        write_composite_set(
            device,
            set,
            scene_view,
            composite_bloom_view(&targets.transient_pool, scene.ssao_white.view, i),
            scene.color_lut.view,
        );
        (COMPOSITE_SOURCES..2 * COMPOSITE_SOURCES)
            .fold(SetWrites::new(set), |w, b| w.sampler(b, sampler.handle()))
            .apply(device);
    }
    Ok(CompositeState {
        formats,
        pipeline,
        pipeline_layout,
        _set_layout: set_layout,
        sets,
    })
}
