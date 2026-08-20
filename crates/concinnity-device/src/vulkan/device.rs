// src/vulkan/device.rs
//
// Vulkan physical/logical device selection and queue-family queries.
use std::ffi::{CStr, CString, c_void};

use ash::{Device, vk};

use crate::vulkan::post::{ResolvedBackend, UpscaleSdk};

pub(super) fn pick_physical_device(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32, u32), String> {
    // SAFETY: an enumeration query on a live instance handle; it only reads, and ash sizes the
    // output vector from the count the driver reports.
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| format!("enumerate physical devices: {e}"))?;
    for pd in devices {
        if let Ok((gf, pf)) = query_queue_families(instance, pd, surface_loader, surface) {
            let extensions =
                // SAFETY: an enumeration query on a live instance handle; it only reads, and ash
                // sizes the output vector from the count the driver reports.
                unsafe { instance.enumerate_device_extension_properties(pd) }.unwrap_or_default();
            let has_swapchain = extensions.iter().any(|e| {
                // SAFETY: Vulkan fills `extension_name` with a NUL-terminated string, and the
                // borrow does not outlive the properties entry it points into.
                let name = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
                name.to_bytes() == b"VK_KHR_swapchain"
            });
            if has_swapchain {
                return Ok((pd, gf, pf));
            }
        }
    }
    Err("no suitable Vulkan physical device found".to_string())
}

pub(super) fn query_queue_families(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(u32, u32), String> {
    // SAFETY: a property query on a live handle; it only reads.
    let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
    let mut graphics = None;
    let mut present = None;
    for (i, f) in families.iter().enumerate() {
        if f.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            graphics = Some(i as u32);
        }
        // SAFETY: a property query on a live handle; it only reads.
        if unsafe { surface_loader.get_physical_device_surface_support(pd, i as u32, surface) }
            .unwrap_or(false)
        {
            present = Some(i as u32);
        }
    }
    match (graphics, present) {
        (Some(g), Some(p)) => Ok((g, p)),
        _ => Err("no suitable queue families".to_string()),
    }
}

// A built logical device and the optional capabilities it was created with.
pub(super) struct LogicalDevice {
    pub device: Device,
    // `VK_EXT_memory_budget` was enabled, for the VRAM-residency chip.
    pub memory_budget: bool,
    // The hardware ray-query path is CAPABLE (the device exposes
    // `VK_KHR_acceleration_structure` + `VK_KHR_ray_query` +
    // `VK_KHR_deferred_host_operations` + the matching features AND XeSS does
    // not own the feature chain). Capability is independent of whether the world
    // requested RT: the RT extensions + feature structs are enabled whenever the
    // device is capable so a live `apply_quality_settings` toggle can bring RT up
    // at runtime (a device extension cannot be enabled after `create_device`).
    // The caller decides whether to BUILD RT at launch (`rt_settings.is_some() &&
    // rt_capable`); a capable-but-unused device just carries the inert extension
    // enables. False on an RT-incapable GPU or under XeSS, and the renderer stays
    // on SSR.
    pub rt_capable: bool,
    // `descriptorBindingSampledImageUpdateAfterBind` was enabled, so the bindless
    // texture pool's set layout may opt into
    // `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT` and budget
    // against the far larger update-after-bind sampler limit. Only enabled on a
    // device whose plain per-stage sampler budget is too tight to seat the pool
    // (`descriptor_layout::sampler_budget_is_constrained`), which is MoltenVK and
    // no desktop driver.
    pub update_after_bind: bool,
}

pub(super) fn create_logical_device(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    graphics_family: u32,
    present_family: u32,
    validation: bool,
    upscaler_sdk: &UpscaleSdk,
) -> Result<LogicalDevice, String> {
    let priority = [1.0f32];
    let mut queue_infos = vec![
        vk::DeviceQueueCreateInfo::default()
            .queue_family_index(graphics_family)
            .queue_priorities(&priority),
    ];
    if present_family != graphics_family {
        queue_infos.push(
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(present_family)
                .queue_priorities(&priority),
        );
    }

    // VK_EXT_memory_budget is optional, used by the profiler overlay to
    // report current VRAM residency. Falls back to a zero reading when
    // unavailable; matches DirectX's zero-fallback on adapters without
    // QueryVideoMemoryInfo support.
    // SAFETY: an enumeration query on a live instance handle; it only reads, and ash sizes the
    // output vector from the count the driver reports.
    let exts = unsafe { instance.enumerate_device_extension_properties(pd) }.unwrap_or_default();
    let has_memory_budget = exts.iter().any(|e| {
        // SAFETY: Vulkan fills `extension_name` with a NUL-terminated string, and the borrow does
        // not outlive the properties entry it points into.
        let name = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
        name.to_bytes() == b"VK_EXT_memory_budget"
    });

    // FidelityFX FSR (the Vulkan temporal upscaler, `vulkan/post/upscale.rs`)
    // builds FP16 + extended-subgroup shader permutations whenever the *physical
    // device* reports support (it inspects the enumerated extension list, not the
    // enabled one), and those pipelines require the matching features ENABLED at
    // device creation. Enable them when present so the FFX context + pipelines are
    // valid; the engine's own shaders don't use them, so this is otherwise inert.
    // When a feature is unsupported, FFX also detects it as absent and falls back
    // to its FP32 path, keeping the two in sync.
    let has_ext = |needle: &[u8]| {
        exts.iter().any(|e| {
            // SAFETY: Vulkan fills `extension_name` with a NUL-terminated string, and the borrow
            // does not outlive the properties entry it points into.
            let name = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
            name.to_bytes() == needle
        })
    };
    let has_f16 = has_ext(b"VK_KHR_shader_float16_int8");
    let has_16bit = has_ext(b"VK_KHR_16bit_storage");
    let has_subgroup_ext = has_ext(b"VK_KHR_shader_subgroup_extended_types");
    // FFX FSR's VK backend loads the KHR-suffixed `vkGetBufferMemoryRequirements2KHR`
    // (and uses the dedicated-allocation path); those entry points only resolve
    // when the extensions are ENABLED, even though both are core in Vulkan 1.1.
    // Without them FFX calls a null pointer during context creation (AV at 0x0).
    let has_mem_reqs2 = has_ext(b"VK_KHR_get_memory_requirements2");
    let has_dedicated = has_ext(b"VK_KHR_dedicated_allocation");

    // Hardware ray-traced reflections trace inline `rayQueryEXT` against a
    // scene acceleration structure, which needs VK_KHR_acceleration_structure
    // (+ its VK_KHR_deferred_host_operations dependency) and VK_KHR_ray_query,
    // plus the `accelerationStructure` / `rayQuery` / `bufferDeviceAddress`
    // features enabled. All four are RDNA2 / Turing-and-up; on an older GPU the
    // probe fails and the caller falls back to SSR. XeSS owns the device-feature
    // chain (it appends a Vulkan12Features), so RT is not co-enabled there.
    let has_accel_struct = has_ext(b"VK_KHR_acceleration_structure");
    let has_ray_query = has_ext(b"VK_KHR_ray_query");
    let has_deferred_host = has_ext(b"VK_KHR_deferred_host_operations");
    let rt_exts_present = has_accel_struct && has_ray_query && has_deferred_host;

    // A portability driver (MoltenVK) exposes `VK_KHR_portability_subset`, and
    // Vulkan requires an implementation that exposes it to have it enabled, so
    // `create_device` is invalid without this.
    let has_portability_subset = has_ext(b"VK_KHR_portability_subset");

    // Probe which of those features the device actually supports.
    let mut f16_probe = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let mut s16_probe = vk::PhysicalDevice16BitStorageFeatures::default();
    let mut sub_probe = vk::PhysicalDeviceShaderSubgroupExtendedTypesFeatures::default();
    let mut accel_probe = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
    let mut rq_probe = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
    let mut rt_bda_probe = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
    // Descriptor indexing serves two independent needs, so it is probed
    // unconditionally: every bindless shader indexes the texture pool with a
    // non-uniform index via `nonuniformEXT`
    // (`shaderSampledImageArrayNonUniformIndexing`), and a sampler-constrained
    // device declares that pool update-after-bind
    // (`descriptorBindingSampledImageUpdateAfterBind`).
    let mut di_probe = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
    // Present only on a portability driver (MoltenVK). Probed here and chained
    // back verbatim at device creation, which enables exactly the subset the
    // driver supports.
    let mut portability_probe = vk::PhysicalDevicePortabilitySubsetFeaturesKHR::default();
    {
        let mut probe = vk::PhysicalDeviceFeatures2::default();
        if has_f16 {
            probe = probe.push_next(&mut f16_probe);
        }
        if has_16bit {
            probe = probe.push_next(&mut s16_probe);
        }
        if has_subgroup_ext {
            probe = probe.push_next(&mut sub_probe);
        }
        if has_portability_subset {
            probe = probe.push_next(&mut portability_probe);
        }
        probe = probe.push_next(&mut di_probe);
        if rt_exts_present {
            probe = probe
                .push_next(&mut accel_probe)
                .push_next(&mut rq_probe)
                .push_next(&mut rt_bda_probe);
        }
        // SAFETY: a property query on a live handle; it only reads.
        unsafe { instance.get_physical_device_features2(pd, &mut probe) };
    }
    let want_f16 = has_f16 && f16_probe.shader_float16 != 0;
    let want_16bit = has_16bit && s16_probe.storage_buffer16_bit_access != 0;
    let want_subgroup_ext = has_subgroup_ext && sub_probe.shader_subgroup_extended_types != 0;
    // RT is enabled only when requested, every extension is present, every
    // feature is supported, and XeSS is not the active backend (it forbids a
    // second feature chain). Resolved finally below; this captures the device's
    // capability half.
    let rt_device_capable = rt_exts_present
        && accel_probe.acceleration_structure != 0
        && rq_probe.ray_query != 0
        && rt_bda_probe.buffer_device_address != 0
        && di_probe.shader_sampled_image_array_non_uniform_indexing != 0;

    // Owned extension names, kept alive for the `create_device` call (the raw
    // ptr array below borrows from this).
    let mut enabled: Vec<CString> = vec![CString::new("VK_KHR_swapchain").unwrap()];
    if has_memory_budget {
        enabled.push(CString::new("VK_EXT_memory_budget").unwrap());
    }
    if want_f16 {
        enabled.push(CString::new("VK_KHR_shader_float16_int8").unwrap());
    }
    if want_16bit {
        enabled.push(CString::new("VK_KHR_16bit_storage").unwrap());
    }
    if want_subgroup_ext {
        enabled.push(CString::new("VK_KHR_shader_subgroup_extended_types").unwrap());
    }
    if has_mem_reqs2 {
        enabled.push(CString::new("VK_KHR_get_memory_requirements2").unwrap());
    }
    if has_dedicated {
        enabled.push(CString::new("VK_KHR_dedicated_allocation").unwrap());
    }
    if has_portability_subset {
        enabled.push(CString::new("VK_KHR_portability_subset").unwrap());
    }

    // RT capability gate: the device is capable AND XeSS is not the active
    // backend (XeSS appends its own Vulkan12Features chain, which can't coexist
    // with the RT feature structs + a non-null pEnabledFeatures). The extensions
    // are enabled whenever capable, NOT only when the world wants RT at launch:
    // a Vulkan device extension cannot be turned on after `create_device`, so a
    // session that launched with RT off could never toggle it on otherwise. The
    // enables are inert when RT is never built. Under XeSS or on an RT-incapable
    // GPU the renderer stays on SSR. `buffer_device_address` is core in 1.2, so
    // the three RT extensions are all that's added here.
    let rt_capable = rt_device_capable && upscaler_sdk.choice != ResolvedBackend::Xess;

    // Non-uniform indexing gate for the bindless texture pool. The main scene
    // pass indexes it per fragment, so this rides device support alone rather
    // than RT capability: an RT-incapable GPU still runs that pass, and without
    // the feature a divergent index is undefined. Never under XeSS, which
    // forbids the descriptor-indexing struct alongside the `Vulkan12Features`
    // it appends.
    let want_nonuniform_indexing = di_probe.shader_sampled_image_array_non_uniform_indexing != 0
        && upscaler_sdk.choice != ResolvedBackend::Xess;

    // Update-after-bind gate for the bindless texture pool. Enabled only on a
    // device whose plain per-stage sampler budget cannot seat the pool, which
    // keeps every desktop driver on the untouched feature chain, and never under
    // XeSS, which forbids the descriptor-indexing struct alongside the
    // `Vulkan12Features` it appends.
    let want_update_after_bind = di_probe.descriptor_binding_sampled_image_update_after_bind != 0
        && upscaler_sdk.choice != ResolvedBackend::Xess
        && super::descriptor_layout::sampler_budget_is_constrained(
            // SAFETY: a property query on a live handle; it only reads.
            unsafe { instance.get_physical_device_properties(pd) }
                .limits
                .max_per_stage_descriptor_samplers,
        );

    if rt_capable {
        enabled.push(CString::new("VK_KHR_acceleration_structure").unwrap());
        enabled.push(CString::new("VK_KHR_ray_query").unwrap());
        enabled.push(CString::new("VK_KHR_deferred_host_operations").unwrap());
    }

    // DLSS / XeSS device extensions, queried from the SDK (filtered to what the
    // physical device exposes and not already enabled above). Empty for FSR /
    // native / when upscaling is off. Logged so a missing one is visible.
    let upscale_dev_exts = upscaler_sdk.device_extensions(instance, pd, &enabled);
    if !upscale_dev_exts.is_empty() {
        tracing::info!(
            "Vulkan device extensions for {:?} upscaler: {:?}",
            upscaler_sdk.choice,
            upscale_dev_exts
        );
        enabled.extend(upscale_dev_exts);
    }

    let ext_names: Vec<*const std::os::raw::c_char> = enabled.iter().map(|c| c.as_ptr()).collect();

    // Device-level validation layers are inferred from the instance in modern Vulkan;
    // VkDeviceCreateInfo::ppEnabledLayerNames is deprecated and ignored.
    let _ = validation;

    // `shader_sampled_image_array_dynamic_indexing` lets the bindless static
    // pass index its `sampler2D tex_pool[N]` array by a dynamically-uniform
    // index. `multi_draw_indirect` lets the compute-cull-driven main pass
    // issue every build-time object's draw with one
    // `cmd_draw_indexed_indirect` (`draw_count > 1`). Both are Vulkan 1.0 core
    // features, near-universally supported.
    //
    // `shader_int16` + `shader_storage_image_{read,write}_without_format` are
    // the base-feature bits FFX FSR's compute shaders declare (the `Int16` and
    // `StorageImage*WithoutFormat` SPIR-V capabilities). They're enabled when
    // the device supports them so the upscaler's pipelines validate cleanly;
    // inert for the engine's own shaders.
    // SAFETY: a property query on a live handle; it only reads.
    let base_supported = unsafe { instance.get_physical_device_features(pd) };
    let features = vk::PhysicalDeviceFeatures::default()
        .shader_sampled_image_array_dynamic_indexing(true)
        .multi_draw_indirect(true)
        // Anisotropic filtering for the scene albedo / normal sampler now that
        // those textures carry a mip chain. Inert when the device lacks it.
        .sampler_anisotropy(base_supported.sampler_anisotropy != 0)
        // BC1/BC3/BC5/BC7 sampling for KTX2 / DDS block-compressed textures.
        .texture_compression_bc(base_supported.texture_compression_bc != 0)
        // `VK_POLYGON_MODE_LINE` for the wireframe view mode's main-pass
        // pipeline variants. Inert (and the mode falls back to solid fill) on a
        // device without it.
        .fill_mode_non_solid(base_supported.fill_mode_non_solid != 0)
        .shader_int16(base_supported.shader_int16 != 0)
        .shader_storage_image_write_without_format(
            base_supported.shader_storage_image_write_without_format != 0,
        )
        .shader_storage_image_read_without_format(
            base_supported.shader_storage_image_read_without_format != 0,
        );

    // FFX FP16 enable structs, chained into device creation alongside the basic
    // `enabled_features` (allowed: a `VkPhysicalDeviceFeatures2` in pNext is not,
    // but the individual feature structs are). Each bit mirrors what the probe
    // found supported.
    let mut f16_enable = vk::PhysicalDeviceShaderFloat16Int8Features::default()
        .shader_float16(f16_probe.shader_float16 != 0);
    let mut s16_enable = vk::PhysicalDevice16BitStorageFeatures::default()
        .storage_buffer16_bit_access(s16_probe.storage_buffer16_bit_access != 0)
        .uniform_and_storage_buffer16_bit_access(
            s16_probe.uniform_and_storage_buffer16_bit_access != 0,
        );
    let mut sub_enable = vk::PhysicalDeviceShaderSubgroupExtendedTypesFeatures::default()
        .shader_subgroup_extended_types(sub_probe.shader_subgroup_extended_types != 0);

    // The single-source bindless vertex reads the BaseInstance builtin (Slang
    // lowers SV_StartInstanceLocation to it), which needs shaderDrawParameters
    // enabled. Core since Vulkan 1.1 and universally supported on desktop
    // drivers (MoltenVK included); probed anyway so an exotic device degrades
    // to a validation error rather than a crash.
    let mut draw_params_probe = vk::PhysicalDeviceShaderDrawParametersFeatures::default();
    {
        let mut probe = vk::PhysicalDeviceFeatures2::default().push_next(&mut draw_params_probe);
        // SAFETY: a property query on a live handle; it only reads.
        unsafe { instance.get_physical_device_features2(pd, &mut probe) };
    }
    let want_draw_params = draw_params_probe.shader_draw_parameters != 0;
    let mut draw_params_enable = vk::PhysicalDeviceShaderDrawParametersFeatures::default()
        .shader_draw_parameters(want_draw_params);

    // DLSS (NGX) needs the `bufferDeviceAddress` *feature* enabled, not just the
    // `VK_EXT_buffer_device_address` extension NGX lists: NGX calls
    // `vkGetBufferDeviceAddress` + allocates memory with
    // `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`, both of which the validation layer
    // rejects unless the feature bit is on. Enable it when DLSS is the chosen
    // backend and the device supports it (probed); otherwise inert.
    // `bufferDeviceAddress` is needed by DLSS (NGX) and by the RT path (the
    // acceleration-structure build inputs + scratch are addressed by device
    // address). RT already probed it above (`rt_bda_probe`). Enabled whenever RT
    // is capable so a live RT toggle has it available, even if RT is off at
    // launch (the only added cost on a capable-but-RT-off device).
    let want_bda = rt_capable
        || (upscaler_sdk.choice == ResolvedBackend::Dlss && {
            let mut bda_probe = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
            {
                let mut probe = vk::PhysicalDeviceFeatures2::default().push_next(&mut bda_probe);
                // SAFETY: a property query on a live handle; it only reads.
                unsafe { instance.get_physical_device_features2(pd, &mut probe) };
            }
            bda_probe.buffer_device_address != 0
        });
    let mut bda_enable =
        vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(want_bda);
    // RT feature enablers, chained into device creation when RT is on.
    let mut accel_enable =
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
    let mut rq_enable = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
    let mut di_enable = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
        .shader_sampled_image_array_non_uniform_indexing(want_nonuniform_indexing)
        .descriptor_binding_sampled_image_update_after_bind(want_update_after_bind);
    // The probed portability subset, reused as the enable struct so every
    // feature the driver supports is on. Its `p_next` still points into the
    // probe chain, so clear it before it is pushed onto a different chain.
    let mut portability_enable = portability_probe;
    portability_enable.p_next = std::ptr::null_mut();

    tracing::info!(
        "Vulkan device features: fp16={want_f16}, 16bit_storage={want_16bit}, \
         subgroup_extended_types={want_subgroup_ext}, buffer_device_address={want_bda}, \
         ray_query={rt_capable}, nonuniform_indexing={want_nonuniform_indexing}, \
         update_after_bind={want_update_after_bind} (upscaler + RT enablers)"
    );

    // XeSS patches a device-feature `pNext` chain (it adds a
    // `VkPhysicalDeviceVulkan12Features`), which Vulkan forbids alongside a
    // non-null `pEnabledFeatures`. So for XeSS the base features ride in a
    // `VkPhysicalDeviceFeatures2` chain (no `enabled_features`) that XeSS then
    // appends to; every other path keeps the simpler `enabled_features` form.
    // The FFX FP16 / 16-bit / subgroup-extended-types enabler structs are NOT
    // chained here: they were promoted into Vulkan 1.2, so they are subsumed by
    // the `Vulkan12Features` XeSS adds (chaining both is a validation error), and
    // FSR is not the active backend under XeSS so its enablers are unneeded.
    if upscaler_sdk.choice == ResolvedBackend::Xess {
        // ShaderDrawParameters is its own feature struct (not part of the
        // `Vulkan12Features` XeSS appends), so chaining it here is valid.
        let mut features2 = vk::PhysicalDeviceFeatures2::default().features(features);
        if want_draw_params {
            features2 = features2.push_next(&mut draw_params_enable);
        }
        // Hand XeSS our chain head; it patches required features + appends its
        // own structs (SDK-owned memory, valid while `upscaler_sdk` lives) and
        // returns the head to use as `VkDeviceCreateInfo.pNext`.
        let head = upscaler_sdk.xess_device_features(
            instance,
            pd,
            &mut features2 as *mut _ as *mut c_void,
        );
        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&ext_names);
        // SAFETY: `head` is a valid feature `pNext` chain (our `features2`, with
        // XeSS structs appended), alive until `create_device` returns.
        device_info.p_next = head as *const c_void;
        // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
        // it names belongs to this device.
        let device = unsafe { instance.create_device(pd, &device_info, None) }
            .map_err(|e| format!("create device (xess features): {e}"))?;
        // Neither RT nor update-after-bind is ever co-enabled with XeSS (see
        // `rt_capable` / `want_update_after_bind` above).
        return Ok(LogicalDevice {
            device,
            memory_budget: has_memory_budget,
            rt_capable: false,
            update_after_bind: false,
        });
    }

    let mut device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&ext_names)
        .enabled_features(&features);
    if want_f16 {
        device_info = device_info.push_next(&mut f16_enable);
    }
    if want_16bit {
        device_info = device_info.push_next(&mut s16_enable);
    }
    if want_subgroup_ext {
        device_info = device_info.push_next(&mut sub_enable);
    }
    if want_bda {
        device_info = device_info.push_next(&mut bda_enable);
    }
    if want_draw_params {
        device_info = device_info.push_next(&mut draw_params_enable);
    }
    if rt_capable {
        device_info = device_info
            .push_next(&mut accel_enable)
            .push_next(&mut rq_enable);
    }
    if want_nonuniform_indexing || want_update_after_bind {
        device_info = device_info.push_next(&mut di_enable);
    }
    if has_portability_subset {
        device_info = device_info.push_next(&mut portability_enable);
    }

    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    let device = unsafe { instance.create_device(pd, &device_info, None) }
        .map_err(|e| format!("create device: {e}"))?;
    Ok(LogicalDevice {
        device,
        memory_budget: has_memory_budget,
        rt_capable,
        update_after_bind: want_update_after_bind,
    })
}

pub(super) fn get_max_usable_sample_count(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
) -> vk::SampleCountFlags {
    // SAFETY: a property query on a live handle; it only reads.
    let props = unsafe { instance.get_physical_device_properties(pd) };
    let counts =
        props.limits.framebuffer_color_sample_counts & props.limits.framebuffer_depth_sample_counts;
    for &candidate in &[vk::SampleCountFlags::TYPE_4, vk::SampleCountFlags::TYPE_2] {
        if counts.contains(candidate) {
            return candidate;
        }
    }
    vk::SampleCountFlags::TYPE_1
}
