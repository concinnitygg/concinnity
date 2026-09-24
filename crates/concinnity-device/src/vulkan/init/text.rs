//! Text: the glyph atlases with their sampler, set layout and per-atlas sets,
//! and the text pipeline that draws in the composite pass.

use ash::vk;
use concinnity_core::render::backend_init::MediaPayloads;
use concinnity_core::render::error::RenderResult;

use super::InitGpu;
use crate::vulkan::context::{CompositeState, TextState, VkDescriptors};
use crate::vulkan::pipeline::{compile_text_shaders, create_text_pipeline};
use crate::vulkan::resources::{
    alloc_descriptor_sets, create_descriptor_set_layout, source_set_bindings, write_source_set,
};
use crate::vulkan::texture::{create_sampler_linear_clamp, upload_texture};
use crate::vulkan::upload_ring::UploadRing;

pub(super) fn build_text(
    gpu: &InitGpu<'_>,
    media: &MediaPayloads<'_>,
    composite: &CompositeState,
    descriptors: &VkDescriptors,
) -> RenderResult<TextState> {
    let InitGpu {
        hw,
        frames,
        hot_reload,
        ..
    } = *gpu;
    let device = &hw.device;
    let atlas_textures = media
        .text_atlases
        .iter()
        .enumerate()
        .map(|(i, (w, h, px))| {
            upload_texture(&gpu.upload(), *w, *h, px)
                .map_err(|e| e.context(format_args!("text_atlas[{i}]")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let sampler = create_sampler_linear_clamp(device)?;
    // Text set (set 0 for text pass): the atlas and its sampler.
    let set_layout = create_descriptor_set_layout(device, &source_set_bindings(1))?;
    let text_pc_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(16);
    let text_set_layouts = [set_layout.handle()];
    let pipeline_layout = device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&text_set_layouts)
                .push_constant_ranges(std::slice::from_ref(&text_pc_range)),
        )
        .map_err(|e| crate::vulkan::error::map_vk_result(e, "text pipeline layout"))?;

    // Text renders in the composite pass (post-tonemap, single-sample), so
    // its pipeline targets the composite render pass.
    let pipeline = if !atlas_textures.is_empty() {
        let (tv, tf) = compile_text_shaders(hot_reload)?;
        let tp = create_text_pipeline(
            device,
            composite.render_pass.handle(),
            pipeline_layout.handle(),
            &tv,
            &tf,
            vk::SampleCountFlags::TYPE_1,
        )?;
        Some(tp)
    } else {
        None
    };

    // Text atlas sets.
    let atlas_sets = if atlas_textures.is_empty() {
        vec![]
    } else {
        let text_atlas_layouts: Vec<_> =
            atlas_textures.iter().map(|_| set_layout.handle()).collect();
        let sets = alloc_descriptor_sets(
            device,
            descriptors.descriptor_pool.handle(),
            &text_atlas_layouts,
        )?;
        for (&set, atlas) in sets.iter().zip(atlas_textures.iter()) {
            write_source_set(device, set, &[(atlas.view, sampler.handle())]);
        }
        sets
    };
    Ok(TextState {
        atlas_textures,
        _set_layout: set_layout,
        atlas_sets,
        pipeline,
        pipeline_layout,
        _sampler: sampler,
        upload: UploadRing::new(frames),
    })
}
