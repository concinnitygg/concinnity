//! Text: the glyph atlases with their views, and the text pipeline that draws
//! in the composite pass.

use concinnity_core::render::backend_init::MediaPayloads;
use concinnity_core::render::error::RenderResult;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;

use super::InitGpu;
use crate::directx::context::{DxDescriptors, FRAMES, TextState, dump_on_err};
use crate::directx::pipeline::{compile_text_shaders, create_text_pso, create_text_root_signature};
use crate::directx::texture::{GpuResource, upload_texture};
use crate::directx::upload_ring::UploadRing;

pub(super) fn build_text(
    gpu: &InitGpu<'_>,
    descriptors: &DxDescriptors,
    media: &MediaPayloads<'_>,
    swap_format: DXGI_FORMAT,
) -> RenderResult<TextState> {
    let hw = gpu.hw;
    // Text atlas textures
    let mut atlas_textures: Vec<GpuResource> = Vec::new();
    let mut atlas_srv_gpus: Vec<D3D12_GPU_DESCRIPTOR_HANDLE> = Vec::new();
    for (i, (w, h, px)) in media.text_atlases.iter().enumerate() {
        let s = descriptors.layout.atlas_base_slot + i;
        let res = upload_texture(
            &hw.alloc,
            *w,
            *h,
            px,
            descriptors.slot_cpu(s),
            descriptors.slot_gpu(s),
        )
        .map_err(|e| e.context(format!("text_atlas[{i}]")))?;
        atlas_srv_gpus.push(descriptors.slot_gpu(s));
        atlas_textures.push(res);
    }

    let (text_vs, text_ps) = compile_text_shaders(gpu.hot_reload)?;
    let (root_sig, pso) = build_text_pipeline(
        &hw.device,
        hw.info_queue.as_ref(),
        &text_vs,
        &text_ps,
        swap_format,
        !atlas_textures.is_empty(),
    )?;
    Ok(TextState {
        root_sig,
        pso,
        upload: UploadRing::new(FRAMES),
        atlas_textures,
        atlas_srv_gpus,
    })
}

fn build_text_pipeline(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    text_vs: &[u8],
    text_ps: &[u8],
    swap_format: DXGI_FORMAT,
    has_atlases: bool,
) -> Result<(ID3D12RootSignature, Option<ID3D12PipelineState>), String> {
    let text_root_sig = dump_on_err(info_queue, create_text_root_signature(device))?;
    // Text renders in the composite pass into the single-sample swapchain
    // backbuffer (post-tonemap), so its PSO targets the swapchain format at
    // sample count 1.
    let text_pso = if has_atlases {
        Some(dump_on_err(
            info_queue,
            create_text_pso(device, &text_root_sig, text_vs, text_ps, swap_format, 1),
        )?)
    } else {
        None
    };
    Ok((text_root_sig, text_pso))
}
