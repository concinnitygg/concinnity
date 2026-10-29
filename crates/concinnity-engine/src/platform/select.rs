//! Which graphics API a process renders with, given the request, the build and
//! the machine.

use super::Platform;

/// What an operating system (a `std::env::consts::OS` name) renders with when
/// no backend is compiled in.
pub fn native_platform(os: &str) -> Platform {
    match os {
        "macos" | "ios" => Platform::Metal,
        "windows" => Platform::DirectX,
        _ => Platform::Vulkan,
    }
}

/// The API a request resolves to, and why it was not the one asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Selection {
    /// The API the process renders with.
    pub platform: Platform,
    /// Why it is not the requested one, when it is not.
    pub fallback: Option<Fallback>,
}

/// Why a requested API gave way to the native one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fallback {
    /// The build did not compile the API in.
    NotCompiled(Platform),
    /// The machine has no usable device for it.
    NoDevice(Platform),
}

impl core::fmt::Display for Fallback {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotCompiled(p) => write!(f, "graphics API {} is not in this build", p.key()),
            Self::NoDevice(p) => write!(f, "graphics API {} found no usable device", p.key()),
        }
    }
}

/// Resolve `requested` against what the build compiled in and what `usable`
/// says the machine runs. The native API is never probed: it is the fallback,
/// and a machine with no GPU for it resolves to headless later on its own.
pub fn resolve(
    requested: Option<Platform>,
    native: Platform,
    compiled: &[Platform],
    usable: impl FnOnce(Platform) -> bool,
) -> Selection {
    let fallback = match requested {
        None => None,
        Some(platform) if platform == native => None,
        Some(platform) if !compiled.contains(&platform) => Some(Fallback::NotCompiled(platform)),
        Some(platform) if !usable(platform) => Some(Fallback::NoDevice(platform)),
        Some(platform) => {
            return Selection {
                platform,
                fallback: None,
            };
        }
    };
    Selection {
        platform: native,
        fallback,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DX_VK: &[Platform] = &[Platform::DirectX, Platform::Vulkan];

    fn unreachable_probe(_: Platform) -> bool {
        panic!("the native API is never probed")
    }

    #[test]
    fn every_target_has_a_native_platform() {
        assert_eq!(native_platform("macos"), Platform::Metal);
        assert_eq!(native_platform("windows"), Platform::DirectX);
        assert_eq!(native_platform("linux"), Platform::Vulkan);
        assert_eq!(native_platform("ios"), Platform::Metal);
        assert_eq!(native_platform("android"), Platform::Vulkan);
    }

    #[test]
    fn no_request_or_the_native_one_renders_native_without_probing() {
        for requested in [None, Some(Platform::DirectX)] {
            let got = resolve(requested, Platform::DirectX, DX_VK, unreachable_probe);
            assert_eq!(got.platform, Platform::DirectX);
            assert_eq!(got.fallback, None);
        }
    }

    #[test]
    fn a_compiled_api_with_a_device_is_taken() {
        let got = resolve(Some(Platform::Vulkan), Platform::DirectX, DX_VK, |_| true);
        assert_eq!(got.platform, Platform::Vulkan);
        assert_eq!(got.fallback, None);
    }

    #[test]
    fn an_api_the_build_left_out_falls_back_native() {
        let got = resolve(
            Some(Platform::Metal),
            Platform::DirectX,
            DX_VK,
            unreachable_probe,
        );
        assert_eq!(got.platform, Platform::DirectX);
        assert_eq!(got.fallback, Some(Fallback::NotCompiled(Platform::Metal)));
    }

    #[test]
    fn an_api_with_no_device_falls_back_native() {
        let got = resolve(Some(Platform::Vulkan), Platform::DirectX, DX_VK, |_| false);
        assert_eq!(got.platform, Platform::DirectX);
        assert_eq!(got.fallback, Some(Fallback::NoDevice(Platform::Vulkan)));
    }
}
