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
        Platform::Hlsl
    }
    #[cfg(backend_vk)]
    {
        Platform::Glsl
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
        "macos" => Platform::Metal,
        "windows" => Platform::Hlsl,
        _ => Platform::Glsl,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // At most one backend cfg is on, so the resolved platform is one of the
    // three.
    #[test]
    fn the_backend_resolves_to_one_platform() {
        let platform = current();
        assert!(["metal", "hlsl", "glsl"].contains(&platform.key()));
    }

    #[test]
    fn every_target_has_a_native_platform() {
        assert_eq!(native_platform("macos"), Platform::Metal);
        assert_eq!(native_platform("windows"), Platform::Hlsl);
        assert_eq!(native_platform("linux"), Platform::Glsl);
    }
}
