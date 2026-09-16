//! The rendering backend this build links, behind two entry points that exist
//! whether or not one is linked.
//!
//! A build with no backend feature has no `concinnity-device` in its graph at
//! all. Both answers the callers need are already part of the contract: backend
//! construction reports `Unsupported`, and an unclassified GPU is what quality
//! auto-config falls back to.

use concinnity_core::render::backend::{GpuProfile, RenderBackend};
use concinnity_core::render::backend_init::BackendInit;
use concinnity_core::render::error::RenderResult;

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

/// Build the backend the client draws through, or report why there is none.
pub(crate) fn init_backend(init: BackendInit<'_>) -> RenderResult<Box<dyn RenderBackend>> {
    #[cfg(any(backend_metal, backend_dx, backend_vk))]
    {
        concinnity_device::init_backend(init)
    }
    #[cfg(not(any(backend_metal, backend_dx, backend_vk)))]
    {
        let _ = init;
        Err(concinnity_core::render::error::RenderError::Unsupported { op: "init_backend" })
    }
}

// Only a build with no backend compiles the `Unsupported` arm, so only it can
// test what that arm reports.
#[cfg(all(test, not(any(backend_metal, backend_dx, backend_vk))))]
mod tests {
    use super::*;
    use concinnity_core::components::Window;
    use concinnity_core::render::error::RenderError;

    #[test]
    fn a_build_with_no_backend_reports_unsupported() {
        let window = Window::default();
        let init = BackendInit::minimal(&window, Vec::new());
        assert_eq!(
            init_backend(init).err(),
            Some(RenderError::Unsupported { op: "init_backend" })
        );
    }
}
