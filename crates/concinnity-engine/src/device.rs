//! The rendering backend this build links, behind two entry points that exist
//! whether or not one is linked.
//!
//! A build with no backend feature has no `concinnity-device` in its graph at
//! all. Both answers the callers need are already part of the contract: backend
//! construction reports `Unsupported`, and a machine with no GPU is what the
//! probe reports.

use concinnity_core::render::backend::{DeviceProbe, RenderBackend};
use concinnity_core::render::backend_init::BackendInit;
use concinnity_core::render::error::RenderResult;

/// Whether a rendering backend compiles into this build. False leaves the
/// headless loop as the only one that can run a world.
pub const AVAILABLE: bool = cfg!(any(backend_metal, backend_dx, backend_vk));

/// Classify the GPU the renderer would draw on and read what it can run,
/// without building the renderer.
///
/// `None` means the machine exposes no usable GPU, which is the one hardware
/// condition that resolves a run to headless. A GPU that is present but
/// unclassifiable comes back with `GpuProfile::UNKNOWN`, which quality
/// auto-config reads as "no clamp".
pub(crate) fn probe_device() -> Option<DeviceProbe> {
    // Run-mode resolution, the quality ceiling and the capability clamp all
    // ask, and a machine's GPU does not come and go mid-process, so the
    // throwaway device handle is built once.
    static PROBED: std::sync::OnceLock<Option<DeviceProbe>> = std::sync::OnceLock::new();
    *PROBED.get_or_init(|| {
        #[cfg(any(backend_metal, backend_dx, backend_vk))]
        {
            concinnity_device::probe_device()
        }
        #[cfg(not(any(backend_metal, backend_dx, backend_vk)))]
        {
            None
        }
    })
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
