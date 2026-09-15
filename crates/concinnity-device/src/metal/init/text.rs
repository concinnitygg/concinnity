//! The HUD text state: the glyph atlases, the text pipeline, its sampler, and
//! the per-frame geometry ring.

use concinnity_core::render::error::RenderResult;
use objc2_metal::{
    MTLDevice as _, MTLSamplerAddressMode, MTLSamplerDescriptor, MTLSamplerMinMagFilter,
};

use super::InitGpu;
use crate::metal::context::TextState;
use crate::metal::pipeline::build_text_pipeline;
use crate::metal::text_upload::TextUploadRing;
use crate::metal::texture::upload_texture;

pub(super) fn build_text(
    gpu: &InitGpu<'_>,
    text_atlases: &[(u32, u32, Vec<u8>)],
) -> RenderResult<TextState> {
    let device = &*gpu.hw.device;
    let (pipeline_state, atlas_textures) = if text_atlases.is_empty() {
        (None, Vec::new())
    } else {
        let text_ps = build_text_pipeline(device, gpu.hw.swap_pixel_format, gpu.hot_reload)?;
        let mut gpu_atlases = Vec::with_capacity(text_atlases.len());
        for (i, (aw, ah, pixels)) in text_atlases.iter().enumerate() {
            let tex = upload_texture(&gpu.hw.allocator, *aw, *ah, pixels)
                .map_err(|e| e.context(format_args!("text_atlas[{i}]")))?;
            gpu_atlases.push(tex);
        }
        (Some(text_ps), gpu_atlases)
    };

    let sampler = {
        let desc = MTLSamplerDescriptor::new();
        desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
        desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
        desc.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
        desc.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
        device
            .newSamplerStateWithDescriptor(&desc)
            .ok_or("failed to create text sampler state")?
    };

    Ok(TextState {
        pipeline_state,
        atlas_textures,
        sampler,
        upload: TextUploadRing::new(gpu.frames_in_flight),
    })
}
