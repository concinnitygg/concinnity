//! Which shader compile target this build's rendering backend consumes.
//!
//! The backend cfg is resolved once in build.rs; this is the one place that
//! reads it as a value, so the runtime, the editor, and the cook all name the
//! same platform without each resolving it again.

use concinnity_core::platform::Platform;

/// The shader platform this build's rendering backend consumes. Resolved from
/// the backend cfg rather than the target OS, so a Windows Vulkan build
/// correctly reports SPIR-V rather than DXIL.
///
/// A build with no backend runs nothing that consumes bytecode, but the cook
/// still needs a language to produce; the one this target renders with is the
/// useful answer, and no backend cfg contradicts it.
pub fn current() -> Platform {
    #[cfg(backend_metal)]
    {
        Platform::Metal
    }
    #[cfg(backend_dx)]
    {
        Platform::DirectX
    }
    #[cfg(backend_vk)]
    {
        Platform::Vulkan
    }
    #[cfg(not(any(backend_metal, backend_dx, backend_vk)))]
    {
        native_platform(std::env::consts::OS)
    }
}

// What a target renders with when no backend is compiled in, mirroring how the
// `native` feature resolves.
#[cfg(any(test, not(any(backend_metal, backend_dx, backend_vk))))]
fn native_platform(target_os: &str) -> Platform {
    match target_os {
        "macos" | "ios" => Platform::Metal,
        "windows" => Platform::DirectX,
        _ => Platform::Vulkan,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_target_has_a_native_platform() {
        assert_eq!(native_platform("macos"), Platform::Metal);
        assert_eq!(native_platform("windows"), Platform::DirectX);
        assert_eq!(native_platform("linux"), Platform::Vulkan);
        assert_eq!(native_platform("ios"), Platform::Metal);
        assert_eq!(native_platform("android"), Platform::Vulkan);
    }
}
