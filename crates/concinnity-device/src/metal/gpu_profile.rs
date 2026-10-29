//! Coarse GPU performance profile from the live `MTLDevice`, for default-quality
//! selection. Every signal is read straight off the device (no extra GPU work):
//! the unified-memory flag, the recommended working-set as the memory budget, and
//! the highest supported Apple GPU family as the generation. The shared
//! `classify_tier` (core's render/backend/probe.rs) maps these to a tier so every
//! backend classifies the same way.

use concinnity_engine::render::backend::{
    DeviceCapabilities, DeviceProbe, GpuClassInput, GpuProfile, GpuVendor, classify_tier,
};
use concinnity_engine::render::output::upscaler::UpscalerAvailability;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily};

// Probe the system default GPU's profile and capabilities without building the
// renderer, for the quality ceiling and the capability clamp that resolve before
// the backend (and its render targets) are created. Creates only the cheap
// default-device handle, which is dropped immediately. `None` when the machine
// exposes no Metal device at all, which is what resolves a run to headless.
pub(crate) fn probe_device() -> Option<DeviceProbe> {
    MTLCreateSystemDefaultDevice().map(|device| {
        let capabilities = device_capabilities(&device);
        if device.supportsRaytracing() && !capabilities.ray_tracing {
            tracing::info!(
                "ray tracing needs macOS 14 / iOS 17 or later; reflections fall back to screen space"
            );
        }
        DeviceProbe {
            profile: device_profile(&device),
            capabilities,
        }
    })
}

// What `device` can run. The built context reports the same answer.
pub(crate) fn device_capabilities(device: &ProtocolObject<dyn MTLDevice>) -> DeviceCapabilities {
    DeviceCapabilities {
        ray_tracing: super::raytrace::raytracing_supported(device),
        // No upscaler but MetalFX, so XeSS never shares the device.
        ray_tracing_with_xess: true,
        // Upscaling always goes through MetalFX; there is no selector.
        selectable_upscaler: false,
        upscalers: UpscalerAvailability::NONE,
        // The per-frame RT topology refresh re-admits recycled build-time
        // slots, so every retired slot may be reused.
        reuses_build_slots: true,
        // The per-object buffer is rebuilt from `draw.objects` every frame,
        // so a slot's rewritten material draws on the next one.
        rewrites_draws: true,
    }
}

// Highest Apple GPU family the device supports, as a generation rank (7 = M1 ..
// 10 = M4), or 0 when the device reports no Apple family (e.g. an AMD dGPU on an
// Intel Mac). Probed top-down so a newer device returns its own generation.
fn apple_gpu_family(device: &ProtocolObject<dyn MTLDevice>) -> u8 {
    for (family, rank) in [
        (MTLGPUFamily::Apple10, 10u8),
        (MTLGPUFamily::Apple9, 9),
        (MTLGPUFamily::Apple8, 8),
        (MTLGPUFamily::Apple7, 7),
    ] {
        if device.supportsFamily(family) {
            return rank;
        }
    }
    0
}

pub(crate) fn device_profile(device: &ProtocolObject<dyn MTLDevice>) -> GpuProfile {
    let unified = device.hasUnifiedMemory();
    let budget = device.recommendedMaxWorkingSetSize();
    let family = apple_gpu_family(device);
    // Apple silicon (or any device reporting an Apple GPU family) is Apple;
    // otherwise an AMD / Intel dGPU on an Intel Mac, classified by VRAM.
    let vendor = if family >= 7 || unified {
        GpuVendor::Apple
    } else {
        GpuVendor::Other
    };
    // A true discrete GPU: not unified-memory (Apple silicon) and not low-power
    // (an Intel-Mac iGPU). Leaves the Intel-Mac AMD dGPU as the only discrete
    // case; Apple silicon reports unified, not discrete.
    let discrete = !unified && !device.isLowPower();
    let tier = classify_tier(&GpuClassInput {
        vendor,
        memory_budget_bytes: budget,
        discrete,
        apple_family: family,
    });
    GpuProfile {
        vendor,
        tier,
        memory_budget_bytes: budget,
        unified_memory: unified,
        discrete,
    }
}
