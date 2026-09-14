//! Render targets: the DSV heap with the main depth buffer, the HDR scene
//! target and its MSAA resolve, the views the depth- and scene-sampling passes
//! bind, and the render graph's transient pool.

use concinnity_core::render::error::RenderResult;
use windows::Win32::Graphics::Direct3D12::*;

use super::heap_layout::{DSV_MAIN_DEPTH_SLOT, RtvHeapLayout};
use super::{Features, InitGpu, heaps};
use crate::directx::context::{
    DepthState, DxDescriptors, DxTargets, Extents, HdrState, SwapchainState, UpscaleState,
};
use crate::directx::texture::{
    HDR_FORMAT, create_hdr_color_target, create_hdr_resolve_target, create_main_depth_texture,
    write_hdr_srv,
};
use crate::directx::transient_pool::{TransientResourcePool, transient_slots};

pub(super) struct TargetInputs<'a> {
    pub(super) descriptors: &'a DxDescriptors,
    pub(super) swapchain: &'a SwapchainState,
    pub(super) rtv: &'a RtvHeapLayout,
    pub(super) upscale: &'a UpscaleState,
    pub(super) features: &'a Features,
    pub(super) output: (u32, u32),
    pub(super) clear_color: [f32; 4],
}

pub(super) fn build_targets(
    gpu: &InitGpu<'_>,
    inputs: TargetInputs<'_>,
) -> RenderResult<DxTargets> {
    let TargetInputs {
        descriptors,
        swapchain,
        rtv,
        upscale,
        features,
        output: (width, height),
        clear_color,
    } = inputs;
    let hw = gpu.hw;
    let layout = &descriptors.layout;
    let msaa_samples = features.msaa_samples;
    // Off-screen scene render resolution. The active backend reports the
    // resolved render dims (clamped to the backend's supported ratio
    // range); a missing / failed upscaler leaves the scene at full output.
    let (render_w, render_h) = match &upscale.backend {
        Some(u) => u.render_dims(),
        None => (width, height),
    };

    let dsv_heap = heaps::create_dsv_heap(&hw.device)?;
    let dsv_descriptor_size = heaps::descriptor_size(&hw.device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    let main_dsv_cpu = heaps::cpu_handle(&dsv_heap, dsv_descriptor_size, DSV_MAIN_DEPTH_SLOT);

    // Main depth buffer. Allowed as SRV so the projected-decal pass can
    // sample it to reconstruct world positions; runtime `add_decal`
    // needs this even when no decals were declared at init.
    let depth_resource = create_main_depth_texture(
        &hw.device,
        render_w,
        render_h,
        main_dsv_cpu,
        msaa_samples,
        true,
    )?;

    // Off-screen HDR scene target
    // The main + instanced passes render linear-light HDR into this; the
    // composite pass tonemaps it onto the swapchain. RTV heap slot [FRAMES]
    // (after the back-buffer RTVs) holds its render-target view.
    let hdr_color_rtv = swapchain.rtv(rtv.hdr_slot);
    let hdr_color = create_hdr_color_target(
        &hw.device,
        render_w,
        render_h,
        msaa_samples,
        hdr_color_rtv,
        clear_color,
    )?;
    // The resolved sample count decides which of two shapes the frame has:
    // with MSAA the main pass resolves `hdr_color` into a separate
    // single-sample spine (and the render graph carries both as resources),
    // without it `hdr_color` is the spine and there is no resolve step at
    // all. Log it so a verification run can say which shape it exercised.
    tracing::info!("d3d12 HDR target: {msaa_samples}x MSAA");
    let hdr_resolve = if msaa_samples > 1 {
        Some(create_hdr_resolve_target(&hw.device, render_w, render_h)?)
    } else {
        None
    };
    // RTV for `hdr_resolve`: the projected-decal pass renders into the
    // resolved scene target, so it needs a render-target view. Sits in
    // the RTV heap right after the SSAO RTVs. Only created when MSAA is
    // on (MSAA off uses the existing `hdr_color_rtv`).
    let hdr_resolve_rtv = if let Some(resolve) = &hdr_resolve {
        let rtv_handle = swapchain.rtv(rtv.decal_resolve_slot);
        // SAFETY: the view descriptor and the resource it names are live for the call, and the
        // destination handle addresses a slot this context reserved for the view in a heap it
        // owns.
        unsafe {
            let rtv_desc = D3D12_RENDER_TARGET_VIEW_DESC {
                Format: HDR_FORMAT,
                ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                ..Default::default()
            };
            hw.device
                .CreateRenderTargetView(resolve, Some(&rtv_desc), rtv_handle);
        }
        Some(rtv_handle)
    } else {
        None
    };
    // The composite pass samples the resolved target (MSAA on) or the
    // directly-rendered HDR target (MSAA off).
    write_hdr_srv(
        &hw.device,
        hdr_resolve.as_ref().unwrap_or(&hdr_color),
        descriptors.slot_cpu(layout.hdr_srv_slot),
    );

    // Main-depth SRV, shared by every depth-sampling decoration pass (decal,
    // glass, lines) at their own t0. The DSV-only flag was dropped above so
    // this is valid.
    crate::directx::decal::write_main_depth_srv(
        &hw.device,
        &depth_resource,
        descriptors.slot_cpu(layout.decal_depth_srv_slot),
        msaa_samples,
    );

    // Transient pool: the graph-owned transient render targets. `bloom_top`
    // (bloom mip 0) is always managed; `ao_output` is placed only when SSAO is
    // on (else `resource_for` returns None and the main pass binding 6 falls
    // back to `ssao_white`). Built before the bloom chain, SSAO and the
    // G-buffer, which read their placed resources back by label.
    let transient_pool = TransientResourcePool::build(
        hw.alloc.device(),
        hw.alloc.queue(),
        &transient_slots(
            features.ssao_enabled,
            features.gbuffer_enabled,
            (render_w, render_h),
            (width, height),
        )?,
    )?;

    Ok(DxTargets {
        hdr: HdrState {
            color: hdr_color,
            color_rtv: hdr_color_rtv,
            resolve: hdr_resolve,
            resolve_rtv: hdr_resolve_rtv,
            srv_gpu: descriptors.slot_gpu(layout.hdr_srv_slot),
            msaa_samples,
        },
        depth: DepthState {
            dsv: main_dsv_cpu,
            resource: depth_resource,
            heap: dsv_heap,
        },
        main_depth_srv_gpu: descriptors.slot_gpu(layout.decal_depth_srv_slot),
        extent: Extents {
            render_width: render_w,
            render_height: render_h,
            output_width: width,
            output_height: height,
        },
        transient_pool,
    })
}
