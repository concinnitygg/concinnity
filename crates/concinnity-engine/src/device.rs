//! The rendering backend this build links, behind two entry points that exist
//! whether or not one is linked.
//!
//! A build with no backend feature has no `concinnity-device` in its graph at
//! all. Both answers the callers need are already part of the contract: backend
//! construction reports failure by yielding no backend, and an unclassified GPU
//! is what quality auto-config falls back to.

use crate::gfx::backend::{GpuProfile, RenderBackend};
use crate::gfx::backend_init::BackendInit;

/// Whether a rendering backend compiles into this build. False leaves the
/// headless loop as the only one that can run a world.
pub const AVAILABLE: bool = cfg!(any(backend_metal, backend_dx, backend_vk));

/// Classify the GPU for quality auto-config.
pub(crate) fn probe_gpu_profile() -> GpuProfile {
    #[cfg(any(backend_metal, backend_dx, backend_vk))]
    {
        concinnity_device::probe_gpu_profile()
    }
    #[cfg(not(any(backend_metal, backend_dx, backend_vk)))]
    {
        GpuProfile::UNKNOWN
    }
}

/// Build the backend the client draws through, or report that there is none.
pub(crate) fn init_backend(init: BackendInit<'_>) -> Option<Box<dyn RenderBackend>> {
    #[cfg(any(backend_metal, backend_dx, backend_vk))]
    {
        concinnity_device::init_backend(init)
    }
    #[cfg(not(any(backend_metal, backend_dx, backend_vk)))]
    {
        let _ = init;
        None
    }
}
