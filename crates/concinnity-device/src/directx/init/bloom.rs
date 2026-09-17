//! Bloom: the mip chain with its RTVs and SRVs, and the prefilter, downsample
//! and upsample pipelines.

use concinnity_core::render::error::{RenderError, RenderResult};
use windows::Win32::Graphics::Direct3D12::*;

use super::InitGpu;
use super::heap_layout::RtvHeapLayout;
use crate::directx::context::{DxDescriptors, DxTargets, SwapchainState, dump_on_err};
use crate::directx::post::bloom::{
    BloomState, compile_bloom_shaders, create_bloom_mips, create_bloom_pso,
    create_bloom_root_signature, write_color_rtv,
};
use crate::directx::texture::{HDR_FORMAT, write_hdr_srv};

pub(super) struct BloomInputs<'a> {
    pub(super) descriptors: &'a DxDescriptors,
    pub(super) swapchain: &'a SwapchainState,
    pub(super) rtv: &'a RtvHeapLayout,
    pub(super) targets: &'a DxTargets,
    // Drawable size, which the mip chain is sized to.
    pub(super) output: (u32, u32),
}

pub(super) fn build_bloom(gpu: &InitGpu<'_>, inputs: BloomInputs<'_>) -> RenderResult<BloomState> {
    let BloomInputs {
        descriptors,
        swapchain,
        rtv,
        targets,
        output: (width, height),
    } = inputs;
    let device = gpu.hw.alloc.device();
    let info_queue = gpu.hw.info_queue.as_ref();
    let bloom_srv_base_slot = descriptors.layout.bloom_srv_base_slot;

    // Bloom mip chain + per-mip RTV/SRV writes. `mips[0]` (`bloom_top`) is the
    // pool's placed resource; the finer mips are committed.
    let bloom_top = targets
        .transient_pool
        .resource_for("bloom_top")
        .ok_or_else(|| RenderError::Other("transient pool missing bloom_top".into()))?
        .clone();
    let (bloom_mips, bloom_mip_extents) = create_bloom_mips(device, width, height, bloom_top)?;
    let mut bloom_mip_rtvs: Vec<D3D12_CPU_DESCRIPTOR_HANDLE> = Vec::with_capacity(bloom_mips.len());
    let mut bloom_mip_srv_gpus: Vec<D3D12_GPU_DESCRIPTOR_HANDLE> =
        Vec::with_capacity(bloom_mips.len());
    for (i, mip) in bloom_mips.iter().enumerate() {
        let mip_rtv = swapchain.rtv(rtv.bloom_base_slot + i);
        write_color_rtv(device, mip, mip_rtv);
        bloom_mip_rtvs.push(mip_rtv);
        write_hdr_srv(device, mip, descriptors.slot_cpu(bloom_srv_base_slot + i));
        bloom_mip_srv_gpus.push(descriptors.slot_gpu(bloom_srv_base_slot + i));
    }

    // Bloom pipelines (prefilter / downsample / upsample). All three write
    // HDR_FORMAT mips; the upsample blends additively so each coarser mip
    // accumulates onto the finer one.
    let bloom_root_sig = dump_on_err(info_queue, create_bloom_root_signature(device))?;
    let bloom_shaders = compile_bloom_shaders(gpu.hot_reload)?;
    let bloom_pso_prefilter = dump_on_err(
        info_queue,
        create_bloom_pso(
            device,
            &bloom_root_sig,
            &bloom_shaders.vs,
            &bloom_shaders.prefilter_ps,
            HDR_FORMAT,
            false,
        ),
    )?;
    let bloom_pso_downsample = dump_on_err(
        info_queue,
        create_bloom_pso(
            device,
            &bloom_root_sig,
            &bloom_shaders.vs,
            &bloom_shaders.downsample_ps,
            HDR_FORMAT,
            false,
        ),
    )?;
    let bloom_pso_upsample = dump_on_err(
        info_queue,
        create_bloom_pso(
            device,
            &bloom_root_sig,
            &bloom_shaders.vs,
            &bloom_shaders.upsample_ps,
            HDR_FORMAT,
            true,
        ),
    )?;

    Ok(BloomState {
        mips: bloom_mips,
        mip_rtvs: bloom_mip_rtvs,
        mip_srv_gpus: bloom_mip_srv_gpus,
        mip_extents: bloom_mip_extents,
        root_sig: bloom_root_sig,
        pso_prefilter: bloom_pso_prefilter,
        pso_downsample: bloom_pso_downsample,
        pso_upsample: bloom_pso_upsample,
    })
}
