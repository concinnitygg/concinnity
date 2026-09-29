//! A headless Vulkan device for GPU-gated unit tests. Each test skips when no
//! driver is present, so CI without a GPU passes vacuously.

use ash::vk;

// A headless instance and device for GPU-gated tests, or None where no
// Vulkan driver is present (CI). Field order is drop order for the explicit
// teardown in `Drop`.
pub(in crate::vulkan) struct TestGpu {
    pub device: ash::Device,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    _entry: ash::Entry,
}

impl Drop for TestGpu {
    fn drop(&mut self) {
        // SAFETY: the handle was created from this device and is destroyed exactly once; the
        // caller has already waited for the device to go idle, so no submission still
        // references it.
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

pub(in crate::vulkan) fn test_gpu() -> Option<TestGpu> {
    let gpu = test_gpu_impl(false);
    if gpu.is_none() {
        eprintln!("skipped: no Vulkan driver");
    }
    gpu
}

// A device with `bufferDeviceAddress` enabled (core 1.2), for the
// device-address pool tests, or None where the driver cannot provide one.
pub(in crate::vulkan) fn test_gpu_with_device_address() -> Option<TestGpu> {
    let gpu = test_gpu_impl(true);
    if gpu.is_none() {
        eprintln!("skipped: no bufferDeviceAddress-capable Vulkan driver");
    }
    gpu
}

fn test_gpu_impl(device_address: bool) -> Option<TestGpu> {
    let entry = crate::vulkan::loader::load_entry().ok()?;
    let app = vk::ApplicationInfo::default().api_version(if device_address {
        vk::API_VERSION_1_2
    } else {
        vk::API_VERSION_1_0
    });
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
    // it names belongs to this device.
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app),
            None,
        )
    }
    .ok()?;
    let destroy_instance = |instance: ash::Instance| {
        // SAFETY: `instance` was created here and nothing derived from it outlives this call.
        unsafe { instance.destroy_instance(None) };
        None
    };
    // SAFETY: an enumeration query on a live instance handle; it only reads, and ash sizes the
    // output vector from the count the driver reports.
    let physical_device = match unsafe { instance.enumerate_physical_devices() } {
        Ok(devices) if !devices.is_empty() => devices[0],
        _ => return destroy_instance(instance),
    };
    let mut enable = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
    if device_address {
        // SAFETY: a property query on a live handle; it only reads.
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        if props.api_version < vk::API_VERSION_1_2 {
            return destroy_instance(instance);
        }
        let mut bda = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
        let mut feats = vk::PhysicalDeviceFeatures2::default().push_next(&mut bda);
        // SAFETY: a property query on a live handle; it only reads.
        unsafe { instance.get_physical_device_features2(physical_device, &mut feats) };
        if bda.buffer_device_address == 0 {
            return destroy_instance(instance);
        }
        enable = enable.buffer_device_address(true);
    }
    let queue_infos = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(0)
        .queue_priorities(&[1.0])];
    let mut device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_infos);
    if device_address {
        device_info = device_info.push_next(&mut enable);
    }
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
    // it names belongs to this device.
    let device = match unsafe { instance.create_device(physical_device, &device_info, None) } {
        Ok(device) => device,
        Err(_) => return destroy_instance(instance),
    };
    Some(TestGpu {
        device,
        instance,
        physical_device,
        _entry: entry,
    })
}
