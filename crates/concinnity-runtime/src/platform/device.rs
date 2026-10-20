//! The rendering backends this build links, behind entry points that exist
//! whether or not one is linked.
//!
//! A build with no backend feature has no `concinnity-device` in its graph at
//! all. Both answers the callers need are already part of the contract: backend
//! construction reports `Unsupported`, and a machine with no GPU is what the
//! probe reports.

use concinnity_core::platform::Platform;
use concinnity_core::render::backend::error::RenderResult;
use concinnity_core::render::backend::init::BackendInit;
use concinnity_core::render::backend::{DeviceProbe, RenderBackend};
use std::sync::OnceLock;

/// Whether a rendering backend compiles into this build. False leaves the
/// headless loop as the only one that can run a world.
pub(crate) const AVAILABLE: bool = cfg!(any(backend_metal, backend_dx, backend_vk));

/// Classify the GPU the renderer would draw on and read what it can run,
/// without building the renderer.
///
/// `None` means the machine exposes no usable GPU, which is the one hardware
/// condition that resolves a run to headless. A GPU that is present but
/// unclassifiable comes back with `GpuProfile::UNKNOWN`, which quality
/// auto-config reads as "no clamp".
pub(crate) fn probe_device() -> Option<DeviceProbe> {
    probe_device_on(crate::platform::current())
}

/// [`probe_device`] through `platform`'s backend rather than the selected one.
pub(crate) fn probe_device_on(platform: Platform) -> Option<DeviceProbe> {
    // Selection, run-mode resolution, the quality ceiling and the capability
    // clamp all ask, and a machine's GPU does not come and go mid-process, so
    // each API's throwaway device handle is built once.
    static PROBED: [OnceLock<Option<DeviceProbe>>; Platform::ALL.len()] =
        [const { OnceLock::new() }; Platform::ALL.len()];
    let slot = Platform::ALL
        .iter()
        .position(|p| *p == platform)
        .unwrap_or(0);
    *PROBED[slot].get_or_init(|| {
        #[cfg(any(backend_metal, backend_dx, backend_vk))]
        {
            concinnity_device::probe_device(platform)
        }
        #[cfg(not(any(backend_metal, backend_dx, backend_vk)))]
        {
            None
        }
    })
}

/// Build the backend the runtime draws through, or report why there is none.
pub(crate) fn init_backend(init: BackendInit<'_>) -> RenderResult<Box<dyn RenderBackend>> {
    crate::crash::note("backend", init.platform.key());
    #[cfg(any(backend_metal, backend_dx, backend_vk))]
    {
        concinnity_device::init_backend(init)
    }
    #[cfg(not(any(backend_metal, backend_dx, backend_vk)))]
    {
        let _ = init;
        Err(
            concinnity_core::render::backend::error::RenderError::Unsupported {
                op: "init_backend",
            },
        )
    }
}

// Only a build with no backend compiles the `Unsupported` arm, so only it can
// test what that arm reports.
#[cfg(all(test, not(any(backend_metal, backend_dx, backend_vk))))]
mod tests {
    use super::*;
    use concinnity_core::components::Window;
    use concinnity_core::render::backend::error::RenderError;

    #[test]
    fn a_build_with_no_backend_reports_unsupported() {
        let window = Window::default();
        let init = BackendInit::minimal(crate::platform::current(), &window, Vec::new());
        assert_eq!(
            init_backend(init).err(),
            Some(RenderError::Unsupported { op: "init_backend" })
        );
    }
}
