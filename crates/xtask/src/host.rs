//! The `<os>-<arch>` a vendored release is built for.

/// A machine a component may have a release for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Host {
    WindowsX64,
    LinuxX64,
    MacosArm64,
}

impl Host {
    /// Every host anything is vendored for.
    pub(crate) const ALL: [Self; 3] = [Self::WindowsX64, Self::LinuxX64, Self::MacosArm64];

    /// The machine this runs on, or `None` where nothing is vendored for it.
    pub(crate) fn current() -> Option<Self> {
        Self::from_consts(std::env::consts::OS, std::env::consts::ARCH)
    }

    fn from_consts(os: &str, arch: &str) -> Option<Self> {
        match (os, arch) {
            ("windows", "x86_64") => Some(Self::WindowsX64),
            ("linux", "x86_64") => Some(Self::LinuxX64),
            ("macos", "aarch64") => Some(Self::MacosArm64),
            _ => None,
        }
    }

    /// The suffix a vendored directory carries, matching what the build scripts
    /// scan `vendor/` for.
    pub(crate) fn slug(self) -> &'static str {
        match self {
            Self::WindowsX64 => "windows-x86_64",
            Self::LinuxX64 => "linux-x86_64",
            Self::MacosArm64 => "macos-aarch64",
        }
    }

    pub(crate) fn is_windows(self) -> bool {
        self == Self::WindowsX64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosts_resolve_from_rust_target_names() {
        assert_eq!(
            Host::from_consts("macos", "aarch64"),
            Some(Host::MacosArm64)
        );
        assert_eq!(
            Host::from_consts("windows", "x86_64"),
            Some(Host::WindowsX64)
        );
        assert_eq!(Host::from_consts("linux", "x86_64"), Some(Host::LinuxX64));
        assert_eq!(Host::from_consts("macos", "x86_64"), None);
    }
}
