//! The backend factory: route the assembled inputs to the backend selected at
//! compile time. The three backend_* cfgs are mutually exclusive, so at most one
//! arm compiles; a build with no backend feature compiles none and reports
//! `Unsupported`. This is the single construction choke point - the client holds
//! only a `Box<dyn RenderBackend>` and never names a concrete backend context.

use concinnity_core::render::backend;
use concinnity_core::render::backend_init;
use concinnity_core::render::error::RenderResult;

/// Probe a cheap throwaway device handle to classify the GPU, so the auto-config
/// quality ceiling can influence the render targets / effect pipelines the backend
/// sizes at init. Each backend creates only the cheap handle it needs and
/// classifies it: Metal the default-device handle, DirectX the DXGI adapter (no
/// device / swapchain), Vulkan a surface-free instance (destroyed immediately).
///
/// `None` means the machine exposes no usable GPU at all. That is a different
/// answer from a GPU this code cannot classify (`Some(GpuProfile::UNKNOWN)`):
/// the first can only run headless, the second renders at an unclamped
/// quality.
pub fn probe_gpu_profile() -> Option<backend::GpuProfile> {
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
        None
    }
}

/// Route the assembled `BackendInit` to the backend selected at compile time.
/// Construction inputs are documented on `BackendInit` itself.
pub fn init_backend(
    init: backend_init::BackendInit<'_>,
) -> RenderResult<Box<dyn backend::RenderBackend>> {
    #[cfg(backend_dx)]
    {
        crate::directx::DxContext::new(init)
            .map(|dx| Box::new(dx) as Box<dyn backend::RenderBackend>)
    }

    #[cfg(backend_vk)]
    {
        crate::vulkan::VkContext::new(init)
            .map(|vk| Box::new(vk) as Box<dyn backend::RenderBackend>)
    }

    #[cfg(backend_metal)]
    {
        crate::metal::MtlContext::new(init)
            .map(|mtl| Box::new(mtl) as Box<dyn backend::RenderBackend>)
    }

    #[cfg(not(any(backend_dx, backend_vk, backend_metal)))]
    {
        let _ = init;
        Err(concinnity_core::render::error::RenderError::Unsupported { op: "init_backend" })
    }
}
