//! The backend factory: route the assembled inputs to the backend the run
//! renders with. A build compiles in the target's native backend and, with the
//! `vulkan` feature, Vulkan beside it; a platform the build did not compile in,
//! and every platform on a build with no backend, reports `Unsupported`. This is the single construction choke point - the
//! client holds only a `Box<dyn RenderBackend>` and never names a concrete
//! backend context.

use concinnity_engine::platform::Platform;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::{backend, backend::init};

/// The graphics APIs this build can render with, the target's native one
/// first. Empty on a target with no window layer.
pub const COMPILED_PLATFORMS: &[Platform] = &[
    #[cfg(backend_metal)]
    Platform::Metal,
    #[cfg(backend_dx)]
    Platform::DirectX,
    #[cfg(backend_vk)]
    Platform::Vulkan,
];

/// Probe a cheap throwaway `platform` device handle to classify the GPU and
/// read what it can run, so the quality ceiling and the capability clamp can
/// shape the render targets and effect pipelines the backend sizes at init. Each backend creates
/// only the handle it needs: Metal the default-device handle, DirectX the
/// adapter it would select (with the device that selection makes), Vulkan a
/// surface-free instance (destroyed immediately).
///
/// `None` means the machine exposes no GPU `platform` can use, including a
/// platform this build did not compile in. That is a different answer from a
/// GPU this code cannot classify (`GpuProfile::UNKNOWN`): the first can only
/// run headless, the second renders at an unclamped quality.
pub fn probe_device(platform: Platform) -> Option<backend::DeviceProbe> {
    match platform {
        #[cfg(backend_dx)]
        Platform::DirectX => crate::directx::probe_device(),
        #[cfg(backend_vk)]
        Platform::Vulkan => crate::vulkan::probe_device(),
        #[cfg(backend_metal)]
        Platform::Metal => crate::metal::probe_device(),
        _ => None,
    }
}

/// Route the assembled `BackendInit` to the backend its `platform` names.
/// Construction inputs are documented on `BackendInit` itself.
///
/// # Errors
///
/// Returns an error if this build has no backend for the platform, or that
/// backend fails to initialize.
pub fn init_backend(init: init::BackendInit<'_>) -> RenderResult<Box<dyn backend::RenderBackend>> {
    match init.platform {
        #[cfg(backend_dx)]
        Platform::DirectX => crate::directx::DxContext::new(init)
            .map(|dx| Box::new(dx) as Box<dyn backend::RenderBackend>),
        #[cfg(backend_vk)]
        Platform::Vulkan => crate::vulkan::VkContext::new(init)
            .map(|vk| Box::new(vk) as Box<dyn backend::RenderBackend>),
        #[cfg(backend_metal)]
        Platform::Metal => crate::metal::MtlContext::new(init)
            .map(|mtl| Box::new(mtl) as Box<dyn backend::RenderBackend>),
        _ => Err(RenderError::Unsupported { op: "init_backend" }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_native_backend_is_listed_first() {
        let native = if cfg!(target_os = "macos") {
            Platform::Metal
        } else if cfg!(target_os = "windows") {
            Platform::DirectX
        } else {
            Platform::Vulkan
        };
        assert_eq!(COMPILED_PLATFORMS.first(), Some(&native));
    }

    #[test]
    fn a_platform_the_build_left_out_probes_no_device() {
        for platform in Platform::ALL {
            if !COMPILED_PLATFORMS.contains(&platform) {
                assert!(probe_device(platform).is_none(), "{platform:?}");
            }
        }
    }

    #[test]
    fn every_compiled_platform_is_listed_once() {
        for platform in Platform::ALL {
            let listed = COMPILED_PLATFORMS
                .iter()
                .filter(|p| **p == platform)
                .count();
            assert!(listed <= 1, "{platform:?}");
        }
        assert_eq!(
            COMPILED_PLATFORMS.contains(&Platform::Vulkan),
            cfg!(backend_vk)
        );
    }
}
