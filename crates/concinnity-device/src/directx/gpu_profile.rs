//! Pre-backend GPU performance and capability probe for DirectX. Selects the DXGI
//! adapter the renderer would (through the same selection device creation
//! makes) and classifies it WITHOUT a swapchain or pipelines, so the quality
//! ceiling and the capability clamp resolve before the backend (and its render
//! targets) are built. Mirrors `DxContext::gpu_profile` exactly (vendor id +
//! dedicated VRAM). Returns `None` when no adapter can run the renderer, which
//! resolves the run to headless.

use concinnity_core::render::backend::{
    DeviceCapabilities, DeviceProbe, GpuClassInput, GpuProfile, GpuVendor, classify_tier,
};
use windows::Win32::Graphics::Dxgi::*;

// What a device that can (or cannot) run DXR 1.1 supports. The built context
// reports the same answer.
pub(super) fn capabilities(ray_tracing: bool) -> DeviceCapabilities {
    DeviceCapabilities {
        ray_tracing,
        ray_tracing_with_xess: true,
        selectable_upscaler: true,
        upscalers: super::post::upscale::availability(),
        // The cull BVH + RT tables key fixed build-time slot indices and
        // cannot refit; only the runtime-append region recycles.
        reuses_build_slots: false,
        // Per-object material state is baked into the GPU records at build
        // time, so a slot rewrite in `SceneState` would not reach the frame.
        rewrites_draws: false,
    }
}

// Pick the adapter the renderer will (the same selection device creation
// makes, software adapter last), classify it, and ask its device for the DXR
// tier. The device is dropped on return. Built on a throwaway DXGI factory with
// no debug flag, so it never needs the "Graphics Tools" optional feature.
pub(crate) fn probe_device() -> Option<DeviceProbe> {
    let factory: IDXGIFactory4 =
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }.ok()?;
    let selection = super::init::select_adapter(&factory).ok()?;
    // SAFETY: a property query on a live COM object; it only reads.
    let desc = unsafe { selection.adapter.GetDesc1() }.ok()?;
    Some(DeviceProbe {
        profile: adapter_profile(&desc),
        capabilities: capabilities(super::raytrace::raytracing_supported(&selection.device)),
    })
}

// Classify an adapter from its description: the vendor and its dedicated VRAM.
pub(super) fn adapter_profile(desc: &DXGI_ADAPTER_DESC1) -> GpuProfile {
    let vendor = GpuVendor::from_pci_id(desc.VendorId);
    let dedicated = desc.DedicatedVideoMemory as u64;
    // A discrete GPU has dedicated VRAM; an integrated part reports little or
    // none. A small floor keeps a few MB of carve-out from reading as discrete.
    let discrete = dedicated >= (256u64 << 20);
    let tier = classify_tier(&GpuClassInput {
        vendor,
        memory_budget_bytes: dedicated,
        discrete,
        apple_family: 0,
    });
    GpuProfile {
        vendor,
        tier,
        memory_budget_bytes: dedicated,
        unified_memory: !discrete,
        discrete,
    }
}
