//! The shared hardware a Vulkan context builds on: the platform window,
//! instance, debug messenger, surface, device, swapchain and allocator a fresh
//! launch acquires, and the bundle a live world reload inherits instead.

use ash::vk;
use concinnity_core::components;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::hdr_output;
use std::ffi::{CStr, CString, c_char};

use crate::vulkan::device::*;
use crate::vulkan::swapchain::*;

// The backend inputs a fresh hardware acquisition reads.
pub(super) struct HardwareRequest<'a> {
    pub(super) title: &'a str,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) title_bar: bool,
    pub(super) validation: bool,
    pub(super) frames: usize,
    pub(super) vsync: bool,
    pub(super) hdr_display: bool,
    pub(super) hdr_pq: bool,
    pub(super) temporal_upscaling: bool,
    pub(super) upscale_backend: components::UpscalerBackend,
}

// Acquire the shared hardware for a fresh launch. `upscale_sdk` stays alive
// through device creation, since its instance-extension pointers and XeSS
// feature chain are read there.
pub(super) fn acquire_hardware(req: HardwareRequest<'_>) -> RenderResult<SharedHardware> {
    let HardwareRequest {
        title,
        width,
        height,
        title_bar,
        validation,
        frames,
        vsync,
        hdr_display,
        hdr_pq,
        temporal_upscaling,
        upscale_backend,
    } = req;
    // Platform window: native Win32 on Windows, AppKit on macOS, GLFW on Linux.
    let mut window = crate::vulkan::PlatformWindow::new(
        title,
        width,
        height,
        &components::WindowMode::Windowed,
        true,
        title_bar,
    )?;

    let entry = crate::vulkan::loader::load_entry()?;

    // Resolve which (if any) upscaler SDK needs Vulkan instance / device
    // extensions enabled at creation time (DLSS / XeSS). Queried before
    // instance creation (it needs at most the loaded SDK), then threaded
    // into `create_logical_device` for the device extensions / features.
    // Inert (`choice == Native`) when upscaling is off or the backend needs
    // nothing; held in scope until after device creation so its
    // instance-ext pointers + XeSS feature chain stay valid. Resolved before
    // `app_info` so its `min_api_version` can raise the instance apiVersion.
    let upscale_sdk = crate::vulkan::post::UpscaleSdk::prepare(temporal_upscaling, upscale_backend);

    let app_name = CString::new(title).unwrap_or_default();
    let engine_name = CString::new("Concinnity").unwrap();
    // Vulkan 1.2 baseline: FidelityFX FSR's precompiled shaders are SPIR-V
    // 1.5, valid only under a 1.2+ instance. XeSS 3.x raises the floor to
    // 1.3 (its shaders use SPV_KHR_integer_dot_product, a 1.3 capability),
    // reported via `min_api_version`. Take the max, clamped to what the
    // loader actually supports so an unsupported request can't fail instance
    // creation (the backend then falls back). The engine's own shaders are
    // unaffected by the bump.
    // SAFETY: an enumeration query on a live instance handle; it only reads, and ash
    // sizes the output vector from the count the driver reports.
    let loader_version = unsafe { entry.try_enumerate_instance_version() }
        .ok()
        .flatten()
        .unwrap_or(vk::API_VERSION_1_2);
    let api_version = vk::API_VERSION_1_2
        .max(upscale_sdk.min_api_version())
        .min(loader_version);
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(api_version);

    // Hold the windowing extension name CStrings in scope so their pointers
    // stay valid through instance creation, then drop with the rest of init
    // (mirrors `device.rs`'s `enabled`/`ext_names` pairing). The later
    // pushes are all `'static` NAME pointers, so they need no backing store.
    let instance_ext_cstrings: Vec<CString> = window
        .required_instance_extensions()
        .iter()
        .map(|s| CString::new(s.as_str()).unwrap())
        .collect();
    let mut ext_names_raw: Vec<*const c_char> =
        instance_ext_cstrings.iter().map(|c| c.as_ptr()).collect();

    let debug_ext = ash::ext::debug_utils::NAME.as_ptr();
    if validation {
        ext_names_raw.push(debug_ext);
    }

    // The optional instance extensions the loader actually exposes:
    // `VK_EXT_swapchain_colorspace` for the extended-range surface
    // formats HDR output needs, and `VK_KHR_portability_enumeration`
    // so a portability driver (MoltenVK) is enumerable at all. A
    // missing one degrades rather than failing instance creation.
    let available_ext_props =
        // SAFETY: an enumeration query on a live instance handle; it only reads, and
        // ash sizes the output vector from the count the driver reports.
        unsafe { entry.enumerate_instance_extension_properties(None) }
            .unwrap_or_default();
    let optional_exts = crate::vulkan::instance_exts::select(
        &crate::vulkan::instance_exts::names_of(&available_ext_props),
        hdr_display,
    );
    let swapchain_colorspace_ext_available = optional_exts.swapchain_colorspace;
    if hdr_display && !swapchain_colorspace_ext_available {
        tracing::warn!(
            "HDR display requested but VK_EXT_swapchain_colorspace is not exposed by the \
     Vulkan loader; falling back to SDR (BGRA8 sRGB) output"
        );
    }
    ext_names_raw.extend(optional_exts.names().iter().map(|n| n.as_ptr()));

    // Instance extensions the chosen upscaler SDK requires (DLSS / XeSS).
    // The pointers borrow from `upscale_sdk`, which outlives this scope.
    for ptr in upscale_sdk.instance_extension_ptrs() {
        ext_names_raw.push(ptr);
    }

    let layer_names_raw: Vec<*const c_char> = if validation {
        // Leaked: the instance borrows the name for its whole lifetime.
        let layer = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
        vec![layer.into_raw().cast_const()]
    } else {
        vec![]
    };

    let instance_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .flags(optional_exts.flags())
        .enabled_extension_names(&ext_names_raw)
        .enabled_layer_names(&layer_names_raw);

    // SAFETY: the create-info and every slice it borrows are live for the call, and
    // each handle it names belongs to this device.
    let instance = unsafe { entry.create_instance(&instance_info, None) }
        .map_err(|e| format!("create instance: {e}"))?;
    // A run with no layer messages looks exactly like a run the layer
    // found nothing wrong with, so say which one happened. Reaching
    // here with the layer requested means it loaded: a missing
    // `VK_LAYER_KHRONOS_validation` fails instance creation above.
    if validation {
        tracing::info!("vulkan validation layer: enabled");
    }

    // Budget the messenger callback consumes to drop benign DLSS first-frame
    // layout errors; set after `build_upscaler` resolves to DLSS. Heap-boxed
    // so its address stays stable, and handed to the owning device
    // handle alongside the messenger: the callback reads it for as long
    // as the messenger can fire, which is past the device teardown.
    // `None` when validation (the messenger) is off.
    let debug_filter: Option<Box<std::sync::atomic::AtomicU32>> =
        validation.then(|| Box::new(std::sync::atomic::AtomicU32::new(0)));
    let (debug_utils, debug_messenger) = if validation {
        let du = ash::ext::debug_utils::Instance::new(&entry, &instance);
        let user_data = debug_filter
            .as_ref()
            .map(|b| &**b as *const std::sync::atomic::AtomicU32 as *mut std::ffi::c_void)
            .unwrap_or(std::ptr::null_mut());
        let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_callback))
            .user_data(user_data);
        // SAFETY: the create-info and every slice it borrows are live for the call, and
        // each handle it names belongs to this device.
        let messenger = unsafe { du.create_debug_utils_messenger(&info, None) }
            .map_err(|e| format!("debug messenger: {e}"))?;
        (Some(du), Some(messenger))
    } else {
        (None, None)
    };

    let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
    let surface = window.create_surface(&entry, &instance)?;

    let (physical_device, graphics_family, present_family) =
        pick_physical_device(&instance, &surface_loader, surface)?;

    // Logical device. `rt_capable` comes back true when the device exposes
    // the ray-query extension set (and XeSS is not the active backend); the
    // RT extensions are enabled whenever capable so a live RT toggle works,
    // independent of whether the world wants RT at launch. The
    // acceleration-structure build + RT pass below are gated on
    // `rt_settings.is_some() && rt_capable` (everything falls back to SSR
    // when RT is off or the device is incapable).
    let crate::vulkan::device::LogicalDevice {
        device,
        memory_budget: memory_budget_supported,
        rt_capable,
        depth_bias_clamp,
        update_after_bind,
    } = create_logical_device(
        &instance,
        physical_device,
        graphics_family,
        present_family,
        validation,
        &upscale_sdk,
    )?;
    // Hand the raw device to the owning wrapper straight away: from
    // here on the device, the instance and the entry are destroyed
    // by the last handle to them, and every Vulkan object the
    // backend owns retires through this device's queue.
    let device = crate::vulkan::owned::VkDevice::new(
        entry.clone(),
        instance.clone(),
        device,
        frames,
        crate::vulkan::owned::DebugMessenger {
            utils: debug_utils,
            messenger: debug_messenger,
            filter: debug_filter,
        },
        depth_bias_clamp,
    );

    // SAFETY: a property query on a live handle; it only reads.
    let graphics_queue = unsafe { device.get_device_queue(graphics_family, 0) };
    // SAFETY: a property query on a live handle; it only reads.
    let present_queue = unsafe { device.get_device_queue(present_family, 0) };

    // Timestamp support: the per-frame GPU-time chip uses a query pool
    // with `2 * frames` slots, a pair per in-flight frame. `timestamp_period`
    // is nanoseconds-per-tick; `timestamp_valid_bits` on the graphics queue
    // family must be non-zero for `cmd_write_timestamp` to be valid. Without
    // either the renderer leaves `gpu_frame_us` at zero. Mirrors
    // `directx::build_timestamp_resources`.
    let device_props =
        // SAFETY: a property query on a live handle; it only reads.
        unsafe { instance.get_physical_device_properties(physical_device) };

    // Persisted pipeline cache: seeded from disk when a blob for
    // this device exists, handed to every pipeline creation below.
    crate::vulkan::pipeline_cache::install(&device, &device_props);

    // SAFETY: a property query on a live handle; it only reads.
    let queue_family_props =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let timestamp_period = device_props.limits.timestamp_period;
    let timestamp_valid_bits = queue_family_props
        .get(graphics_family as usize)
        .map(|f| f.timestamp_valid_bits)
        .unwrap_or(0);
    let timestamps_supported = timestamp_period > 0.0 && timestamp_valid_bits > 0;
    let timestamp_query_pool = if timestamps_supported {
        // One per-frame block of `SLOTS_PER_FRAME` slots (whole-frame pair +
        // one pair per render pass) per frame in flight.
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count((crate::vulkan::pass_timing::SLOTS_PER_FRAME * frames) as u32);
        // SAFETY: the create-info and every slice it borrows are live for the call, and
        // each handle it names belongs to this device.
        match unsafe { device.create_query_pool(&info, None) } {
            Ok(p) => Some(p),
            Err(e) => {
                tracing::warn!("timestamp query pool create failed: {e}");
                None
            }
        }
    } else {
        None
    };

    // Device-local heap indices for the VRAM-residency chip. Sums
    // `heap_usage` on every DEVICE_LOCAL heap when `VK_EXT_memory_budget`
    // is supported; otherwise the field stays empty and the chip reports
    // zero (matching DirectX's adapter-without-QueryVideoMemoryInfo
    // fallback).
    let memory_props =
        // SAFETY: a property query on a live handle; it only reads.
        unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let device_local_heaps: Vec<u32> = if memory_budget_supported {
        (0..memory_props.memory_heap_count as usize)
            .filter(|i| {
                memory_props.memory_heaps[*i]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            })
            .map(|i| i as u32)
            .collect()
    } else {
        Vec::new()
    };

    let max_msaa_samples = get_max_usable_sample_count(&instance, physical_device);

    // HDR-output resolve. The world's `hdr_display` toggle is the
    // gate; even on a capable display, no HDR unless the asset opts
    // in. The reverse (`hdr_display = true` on an SDR-only surface,
    // or with the color-space loader extension missing) falls back
    // to SDR with a logged warning. Vulkan has no portable max-EDR
    // query: when the surface advertises the scRGB-linear color
    // space we synthesize a placeholder `max_edr = 2.0` (the
    // HDR400-class minimum) so the shared `HdrOutputMode::resolve`
    // logic stays uniform across backends.
    // Probe which HDR color-space pairs the surface advertises. An
    // advertised HDR color space is Vulkan's "HDR available" signal (there
    // is no portable max-EDR query), so we synthesize the placeholder
    // `max_edr` from it. scRGB-linear drives the extended-linear path; an
    // `HDR10_ST2084_EXT` pair (float or 10-bit packed) drives the PQ path.
    // SAFETY: a property query on a live handle; it only reads.
    let surface_formats =
        unsafe { surface_loader.get_physical_device_surface_formats(physical_device, surface) }
            .unwrap_or_default();
    let advertises = |fmt: vk::Format, cs: vk::ColorSpaceKHR| {
        surface_formats
            .iter()
            .any(|f| f.format == fmt && f.color_space == cs)
    };
    let scrgb_advertises = swapchain_colorspace_ext_available
        && advertises(
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
        );
    let pq_advertises = swapchain_colorspace_ext_available
        && (advertises(
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ColorSpaceKHR::HDR10_ST2084_EXT,
        ) || advertises(
            vk::Format::A2B10G10R10_UNORM_PACK32,
            vk::ColorSpaceKHR::HDR10_ST2084_EXT,
        ));
    // PQ needs the HDR10 color space. When `hdr_pq` is requested but only
    // scRGB is advertised, fall back to the extended-linear path so the
    // shader encode and the swapchain color space never diverge (sending
    // PQ-encoded values to an scRGB-linear swapchain would look wrong).
    let pq_capable = hdr_pq && pq_advertises;
    if hdr_display && hdr_pq && !pq_advertises {
        tracing::warn!(
            "HDR display + hdr_pq:true requested but no surface format advertises HDR10 PQ \
     (RGBA16F / A2B10G10R10_UNORM_PACK32 + HDR10_ST2084_EXT); falling back to \
     scRGB-linear extended-range output"
        );
    }
    let max_edr = if scrgb_advertises || pq_advertises {
        2.0
    } else {
        1.0
    };
    let hdr_mode = hdr_output::HdrOutputMode::resolve(hdr_display, pq_capable, max_edr);
    if hdr_display && !hdr_mode.is_hdr() {
        tracing::warn!(
            "HDR display requested but no surface format advertises an HDR color space \
     (scRGB linear or HDR10 PQ): falling back to SDR (BGRA8 sRGB) output"
        );
    } else if hdr_mode.pq_flag() > 0.5 {
        tracing::info!("HDR display output enabled: HDR10 PQ swapchain (SMPTE ST 2084)");
    } else if hdr_mode.is_hdr() {
        tracing::info!(
            "HDR display output enabled: scRGB-linear swapchain (RGBA16F + \
     EXTENDED_SRGB_LINEAR_EXT)"
        );
    }
    let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
    let (swapchain, swapchain_images, swapchain_format, swapchain_extent) = create_swapchain_inner(
        &SwapchainSurface {
            instance: &instance,
            device: &device,
            pd: physical_device,
            surface_loader: &surface_loader,
            surface,
            swapchain_loader: &swapchain_loader,
        },
        SwapchainQueueFamilies {
            graphics_family,
            present_family,
        },
        SwapchainConfig {
            width,
            height,
            old_swapchain: vk::SwapchainKHR::null(),
            hdr_mode,
            vsync,
        },
    )?;
    let swapchain_image_views =
        create_swapchain_image_views(&device, &swapchain_images, swapchain_format)?;

    // The device allocator every pooled buffer / image is placed
    // through, built before any resource creation so init-time
    // resources can pool. A reload inherits the outgoing
    // context's instead (the other match arm), so the rebuilt
    // world places into the blocks the old world releases.
    let alloc =
        crate::vulkan::allocator::DeviceAllocator::new(&instance, physical_device, &device, frames);

    Ok(SharedHardware {
        window,
        entry,
        instance,
        device,
        physical_device,
        surface,
        surface_loader,
        graphics_queue,
        present_queue,
        graphics_family,
        swapchain_loader,
        swapchain,
        swapchain_images,
        swapchain_format,
        swapchain_extent,
        swapchain_image_views,
        max_msaa_samples,
        hdr_mode,
        memory_budget_supported,
        rt_capable,
        update_after_bind,
        device_local_heaps,
        timestamp_query_pool,
        timestamp_period,
        alloc,
    })
}

// The shared hardware `VkContext::build` acquires (fresh launch) or inherits
// (live editor reload) before it builds any per-world resource. Destructured
// right after so the rest of `build` is identical on both paths.
pub(super) struct SharedHardware {
    pub(super) window: crate::vulkan::PlatformWindow,
    pub(super) entry: ash::Entry,
    pub(super) instance: ash::Instance,
    pub(super) device: crate::vulkan::owned::VkDevice,
    pub(super) physical_device: vk::PhysicalDevice,
    pub(super) surface: vk::SurfaceKHR,
    pub(super) surface_loader: ash::khr::surface::Instance,
    pub(super) graphics_queue: vk::Queue,
    pub(super) present_queue: vk::Queue,
    pub(super) graphics_family: u32,
    pub(super) swapchain_loader: ash::khr::swapchain::Device,
    pub(super) swapchain: vk::SwapchainKHR,
    pub(super) swapchain_images: Vec<vk::Image>,
    pub(super) swapchain_format: vk::Format,
    pub(super) swapchain_extent: vk::Extent2D,
    pub(super) swapchain_image_views: Vec<vk::ImageView>,
    // This device's ceiling for the HDR format, not the count the world runs
    // at: `resolve_sample_count` clamps the world's request against it.
    pub(super) max_msaa_samples: vk::SampleCountFlags,
    pub(super) hdr_mode: hdr_output::HdrOutputMode,
    pub(super) memory_budget_supported: bool,
    pub(super) rt_capable: bool,
    pub(super) update_after_bind: bool,
    pub(super) device_local_heaps: Vec<u32>,
    pub(super) timestamp_query_pool: Option<vk::QueryPool>,
    pub(super) timestamp_period: f32,
    // Fresh on a launch; the outgoing context's on a reload, so the rebuilt
    // world places into the blocks the old world's leases released.
    pub(super) alloc: crate::vulkan::allocator::DeviceAllocator,
}

// The shared hardware an outgoing context hands to its successor on a live
// editor `reload_world` (see `VkContext::apply_world_reload`). The loaders and
// `ash::{Entry,Instance,Device}` are cheap dispatch-table clones over the same
// underlying objects; the raw `vk::*` handles are `Copy`; the window and the
// (already-`Option`) debug + timestamp handles are moved out of the outgoing
// context so its `Drop` leaves them alone; the device allocator is a shared
// handle (clones share one pool). Vulkan handles are not refcounted, so the
// outgoing `Drop` also skips destroying the shared instance / device /
// surface / swapchain (gated on `reused_by_successor`).
pub(super) struct VkReuse {
    pub(super) window: crate::vulkan::PlatformWindow,
    pub(super) entry: ash::Entry,
    pub(super) instance: ash::Instance,
    pub(super) device: crate::vulkan::owned::VkDevice,
    pub(super) physical_device: vk::PhysicalDevice,
    pub(super) surface: vk::SurfaceKHR,
    pub(super) surface_loader: ash::khr::surface::Instance,
    pub(super) graphics_queue: vk::Queue,
    pub(super) present_queue: vk::Queue,
    pub(super) graphics_family: u32,
    pub(super) swapchain_loader: ash::khr::swapchain::Device,
    pub(super) swapchain: vk::SwapchainKHR,
    pub(super) swapchain_images: Vec<vk::Image>,
    pub(super) swapchain_format: vk::Format,
    pub(super) swapchain_extent: vk::Extent2D,
    pub(super) hdr_mode: hdr_output::HdrOutputMode,
    pub(super) memory_budget_supported: bool,
    pub(super) rt_capable: bool,
    pub(super) update_after_bind: bool,
    pub(super) device_local_heaps: Vec<u32>,
    pub(super) timestamp_query_pool: Option<vk::QueryPool>,
    pub(super) timestamp_period: f32,
    pub(super) alloc: crate::vulkan::allocator::DeviceAllocator,
}

impl VkReuse {
    // Turn the inherited hardware into a `SharedHardware`, recreating the only
    // per-context object among it: fresh swapchain image views over the reused
    // swapchain's images (the outgoing context frees its own views in Drop).
    pub(super) fn into_shared(self) -> Result<SharedHardware, String> {
        let swapchain_image_views = create_swapchain_image_views(
            &self.device,
            &self.swapchain_images,
            self.swapchain_format,
        )?;
        // Re-queried rather than carried: the outgoing context holds the count
        // its world resolved to, and the incoming world's AA mode may differ.
        let max_msaa_samples = get_max_usable_sample_count(&self.instance, self.physical_device);
        Ok(SharedHardware {
            window: self.window,
            entry: self.entry,
            instance: self.instance,
            device: self.device,
            physical_device: self.physical_device,
            surface: self.surface,
            surface_loader: self.surface_loader,
            graphics_queue: self.graphics_queue,
            present_queue: self.present_queue,
            graphics_family: self.graphics_family,
            swapchain_loader: self.swapchain_loader,
            swapchain: self.swapchain,
            swapchain_images: self.swapchain_images,
            swapchain_format: self.swapchain_format,
            swapchain_extent: self.swapchain_extent,
            swapchain_image_views,
            max_msaa_samples,
            hdr_mode: self.hdr_mode,
            memory_budget_supported: self.memory_budget_supported,
            rt_capable: self.rt_capable,
            update_after_bind: self.update_after_bind,
            device_local_heaps: self.device_local_heaps,
            timestamp_query_pool: self.timestamp_query_pool,
            timestamp_period: self.timestamp_period,
            alloc: self.alloc,
        })
    }
}

// Validation layer debug callback: logs validation errors and warnings.
// DLSS's first EvaluateFeature samples two NGX-internal resources it leaves in
// UNDEFINED, tripping VUID-vkCmdDraw-None-09600 exactly twice per feature
// creation. They are internal to nvngx_dlss.dll (not bindable through the NGX
// parameter API, confirmed by supplying our own exposure input, which did not
// displace them) and benign (the upscale output is correct). The debug messenger
// drops this many such messages while DLSS is the active upscaler. D3D12 never
// surfaces them (it has no image-layout validation model).
pub(in crate::vulkan) const DLSS_FIRST_FRAME_LAYOUT_SUPPRESS: u32 = 2;

// Decide whether to drop a validation message rather than log it: true only for
// the benign DLSS first-frame layout VUID while `budget` is positive (consuming
// one unit of it). Every other VUID, and an exhausted budget, returns false so
// the message still surfaces. Split out from `debug_callback` so the suppression
// logic is unit testable without a live Vulkan instance.
fn drop_benign_dlss_layout_error(message_id: &[u8], budget: &std::sync::atomic::AtomicU32) -> bool {
    if message_id != b"VUID-vkCmdDraw-None-09600" {
        return false;
    }
    budget
        .fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |n| (n > 0).then(|| n - 1),
        )
        .is_ok()
}

// Validation messages route through here (installed only when validation is on).
// `user` is a `*const AtomicU32`: a budget of benign DLSS first-frame layout
// errors to drop, set after `build_upscaler` resolves to DLSS (and reset on
// resize, which re-creates the feature). Null when no budget is wired.
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if data.is_null() {
        return vk::FALSE;
    }
    // SAFETY: the null check above passed, and Vulkan guarantees the callback data outlives the
    // callback.
    let data = unsafe { &*data };

    // Drop the benign DLSS first-frame layout errors (see the helper); any other
    // VUID, or an exhausted budget, still logs.
    if !user.is_null() && !data.p_message_id_name.is_null() {
        // SAFETY: Vulkan fills `extension_name` with a NUL-terminated string, and the borrow does
        // not outlive the properties entry it points into.
        let vuid = unsafe { CStr::from_ptr(data.p_message_id_name) };
        // SAFETY: `user` is the `AtomicU32` budget pointer this messenger was registered with; it
        // is non-null per the check above and outlives the messenger.
        let budget = unsafe { &*(user as *const std::sync::atomic::AtomicU32) };
        if drop_benign_dlss_layout_error(vuid.to_bytes(), budget) {
            return vk::FALSE;
        }
    }

    // SAFETY: Vulkan fills `p_message` with a NUL-terminated string that lives for the duration of
    // the callback.
    let msg = unsafe { CStr::from_ptr(data.p_message) }.to_string_lossy();
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        tracing::error!("[Vulkan] {}", msg);
    } else {
        tracing::warn!("[Vulkan] {}", msg);
    }
    vk::FALSE
}

#[cfg(test)]
mod tests {
    use super::{DLSS_FIRST_FRAME_LAYOUT_SUPPRESS, drop_benign_dlss_layout_error};
    use std::sync::atomic::{AtomicU32, Ordering};

    const LAYOUT_VUID: &[u8] = b"VUID-vkCmdDraw-None-09600";

    #[test]
    fn drops_exactly_the_budgeted_layout_errors_then_logs() {
        let budget = AtomicU32::new(DLSS_FIRST_FRAME_LAYOUT_SUPPRESS);
        for _ in 0..DLSS_FIRST_FRAME_LAYOUT_SUPPRESS {
            assert!(drop_benign_dlss_layout_error(LAYOUT_VUID, &budget));
        }
        // Budget spent: a further occurrence logs, so a real bug would surface.
        assert!(!drop_benign_dlss_layout_error(LAYOUT_VUID, &budget));
        assert_eq!(budget.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn never_drops_other_vuids_or_touches_budget() {
        let budget = AtomicU32::new(DLSS_FIRST_FRAME_LAYOUT_SUPPRESS);
        assert!(!drop_benign_dlss_layout_error(
            b"VUID-vkCmdDraw-None-02699",
            &budget
        ));
        assert!(!drop_benign_dlss_layout_error(b"", &budget));
        assert_eq!(
            budget.load(Ordering::Relaxed),
            DLSS_FIRST_FRAME_LAYOUT_SUPPRESS
        );
    }

    #[test]
    fn drops_nothing_when_budget_is_zero() {
        let budget = AtomicU32::new(0);
        assert!(!drop_benign_dlss_layout_error(LAYOUT_VUID, &budget));
    }
}
