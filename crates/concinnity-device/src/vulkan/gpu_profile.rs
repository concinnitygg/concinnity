// src/vulkan/gpu_profile.rs
//
// Pre-backend GPU performance probe for Vulkan. Loads the Vulkan entry, creates a
// throwaway surface-free / extension-free / layer-free instance, picks a physical
// device (preferring a discrete GPU), classifies it, and destroys the instance --
// all WITHOUT a surface, swapchain, logical device, or validation layer, so the
// auto-config quality ceiling can be resolved before the backend (and its render
// targets) are built. Mirrors `VkContext::gpu_profile` exactly (vendor id, device
// type, summed DEVICE_LOCAL heaps) and the standalone `metal/gpu_profile.rs`
// pattern. Returns `UNKNOWN` on any failure (no loader, instance-create fails, no
// physical device), which the resolver treats as "no clamp".

use ash::vk;

use crate::gfx::backend::{
    GpuClassInput, GpuProfile, GpuVendor, apple_family_from_device_name, classify_tier,
};

pub(crate) fn probe_gpu_profile() -> GpuProfile {
    probe_device().unwrap_or(GpuProfile::UNKNOWN)
}

fn probe_device() -> Option<GpuProfile> {
    // Dynamically load the Vulkan loader; `None` when it is absent.
    let entry = super::loader::load_entry().ok()?;
    // A minimal instance: no window/surface extensions, no validation layer, no
    // debug-utils messenger. Enumerating + querying physical devices needs none
    // of those, so the probe stays cheap and avoids loader-layer init. The one
    // exception is `VK_KHR_portability_enumeration`, without which the loader
    // hides a portability driver (MoltenVK) and the probe sees no device at all.
    let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_0);
    // SAFETY: an enumeration query on a live instance handle; it only reads, and ash sizes the
    // output vector from the count the driver reports.
    let available = unsafe { entry.enumerate_instance_extension_properties(None) }.ok()?;
    let portability = super::instance_exts::select(
        &super::instance_exts::names_of(&available),
        // The probe never presents, so it has no use for the HDR colour spaces.
        false,
    );
    let ext_names: Vec<*const std::os::raw::c_char> = portability
        .names()
        .iter()
        .map(|n| n.as_ptr())
        .collect::<Vec<_>>();
    let instance_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .flags(portability.flags())
        .enabled_extension_names(&ext_names);
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    let instance = unsafe { entry.create_instance(&instance_info, None) }.ok()?;
    let profile = pick_and_classify(&instance);
    // The probe owns the instance (no VkContext drops it), so destroy it before
    // returning. The device props/memory queried below are plain copies with no
    // lifetime tie to the instance.
    // SAFETY: `instance` was created here and nothing derived from it outlives this call.
    unsafe { instance.destroy_instance(None) };
    profile
}

// Pick the same physical device the renderer will, so the probed tier describes
// the GPU that actually renders (it matters on a multi-GPU host). `device::
// pick_physical_device` takes the first enumerated device with a graphics queue
// family, a present queue, and `VK_KHR_swapchain`; the probe mirrors that filter
// minus the present check (which needs a surface the probe deliberately has none
// of -- a device with a graphics queue + swapchain almost always supports
// present). Falls back to the first enumerated device when none qualifies (the
// renderer would then fail to init, so the tier is moot, but a best-effort
// classification is harmless).
fn pick_and_classify(instance: &ash::Instance) -> Option<GpuProfile> {
    // SAFETY: an enumeration query on a live instance handle; it only reads, and ash sizes the
    // output vector from the count the driver reports.
    let devices = unsafe { instance.enumerate_physical_devices() }.ok()?;
    let chosen = devices
        .iter()
        .copied()
        .find(|&pd| renderer_eligible(instance, pd))
        .or_else(|| devices.first().copied())?;
    Some(device_profile(instance, chosen))
}

// Whether a device has a graphics queue family and exposes `VK_KHR_swapchain`,
// mirroring `device::pick_physical_device`'s filter without the surface-bound
// present-support check.
fn renderer_eligible(instance: &ash::Instance, pd: vk::PhysicalDevice) -> bool {
    // SAFETY: a property query on a live handle; it only reads.
    let has_graphics = unsafe { instance.get_physical_device_queue_family_properties(pd) }
        .iter()
        .any(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS));
    if !has_graphics {
        return false;
    }
    // SAFETY: an enumeration query on a live instance handle; it only reads, and ash sizes the
    // output vector from the count the driver reports.
    let exts = unsafe { instance.enumerate_device_extension_properties(pd) }.unwrap_or_default();
    exts.iter().any(|e| {
        // SAFETY: Vulkan fills `extension_name` with a NUL-terminated string, and the borrow does
        // not outlive the properties entry it points into.
        let name = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
        name.to_bytes() == b"VK_KHR_swapchain"
    })
}

// The device's reported name. Apple silicon's GPU generation is only reachable
// through it under MoltenVK, so `VkContext::gpu_profile` shares this helper.
pub(super) fn device_name(props: &vk::PhysicalDeviceProperties) -> String {
    // SAFETY: Vulkan fills `extension_name` with a NUL-terminated string, and the borrow does not
    // outlive the properties entry it points into.
    unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

// Classify a chosen physical device, mirroring `VkContext::gpu_profile`.
fn device_profile(instance: &ash::Instance, pd: vk::PhysicalDevice) -> GpuProfile {
    // SAFETY: a property query on a live handle; it only reads.
    let props = unsafe { instance.get_physical_device_properties(pd) };
    let vendor = match props.vendor_id {
        0x10DE => GpuVendor::Nvidia,
        0x1002 => GpuVendor::Amd,
        0x8086 => GpuVendor::Intel,
        0x106B => GpuVendor::Apple, // Apple / MoltenVK
        _ => GpuVendor::Other,
    };
    let discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
    let unified = props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU;
    // SAFETY: a property query on a live handle; it only reads.
    let mem = unsafe { instance.get_physical_device_memory_properties(pd) };
    let budget: u64 = (0..mem.memory_heap_count as usize)
        .filter(|&i| {
            mem.memory_heaps[i]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        })
        .map(|i| mem.memory_heaps[i].size)
        .sum();
    let tier = classify_tier(&GpuClassInput {
        vendor,
        memory_budget_bytes: budget,
        discrete,
        apple_family: apple_family_from_device_name(&device_name(&props)),
    });
    GpuProfile {
        vendor,
        tier,
        memory_budget_bytes: budget,
        unified_memory: unified,
        discrete,
    }
}
