// src/factory.rs
//
// The backend factory: route the assembled inputs to the backend selected at
// compile time. The three backend_* cfgs are mutually exclusive, so exactly one
// arm compiles. This is the single construction choke point - the client holds
// only a `Box<dyn RenderBackend>` and never names a concrete backend context.

// Probe a cheap throwaway device handle to classify the GPU, so the auto-config
// quality ceiling can influence the render targets / effect pipelines the backend
// sizes at init. Each backend creates only the cheap handle it needs and
// classifies it: Metal the default-device handle, DirectX the DXGI adapter (no
// device / swapchain), Vulkan a surface-free instance (destroyed immediately).
// The fallback returns UNKNOWN (which the resolver treats as "no ceiling") only
// when no backend is configured.
pub fn probe_gpu_profile() -> crate::gfx::backend::GpuProfile {
    #[cfg(backend_dx)]
    {
        crate::directx::probe_gpu_profile()
    }
    #[cfg(backend_vk)]
    {
        crate::vulkan::probe_gpu_profile()
    }
    #[cfg(backend_metal)]
    {
        crate::metal::probe_gpu_profile()
    }
    #[cfg(not(any(backend_dx, backend_vk, backend_metal)))]
    {
        crate::gfx::backend::GpuProfile::UNKNOWN
    }
}

// Route the assembled `BackendInit` to the backend selected at compile time.
// Construction inputs are documented on `BackendInit` itself.
pub fn init_backend(
    init: crate::gfx::backend_init::BackendInit<'_>,
) -> Option<Box<dyn crate::gfx::backend::RenderBackend>> {
    #[cfg(backend_dx)]
    {
        match crate::directx::DxContext::new(init) {
            Ok(dx) => Some(Box::new(dx)),
            Err(e) => {
                tracing::error!("GraphicsSystem: D3D12 init failed: {}", e);
                None
            }
        }
    }

    #[cfg(backend_vk)]
    {
        match crate::vulkan::VkContext::new(init) {
            Ok(vk) => Some(Box::new(vk)),
            Err(e) => {
                tracing::error!("GraphicsSystem: Vulkan init failed: {}", e);
                None
            }
        }
    }

    #[cfg(backend_metal)]
    {
        match crate::metal::MtlContext::new(init) {
            Ok(mtl) => Some(Box::new(mtl)),
            Err(e) => {
                tracing::error!("GraphicsSystem: Metal init failed: {}", e);
                None
            }
        }
    }
}
