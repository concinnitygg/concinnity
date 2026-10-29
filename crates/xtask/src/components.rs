//! What gets vendored: each component's pinned release, where every host gets
//! it from, and the file whose presence means it is installed.

use std::path::{Path, PathBuf};

use crate::host::Host;
use crate::source_build;
use crate::{Result, fail, vendor_dir};

/// One third-party binary the build reads from `vendor/`.
pub(crate) struct Component {
    pub(crate) name: &'static str,
    pub(crate) release: &'static str,
    pub(crate) summary: &'static str,
    /// The hosts it has a release for.
    pub(crate) hosts: &'static [Host],
    /// How a host gets it.
    pub(crate) source: fn(Host) -> Source,
    /// The file whose presence means it is installed, relative to its directory.
    pub(crate) payload: fn(Host) -> &'static str,
    /// The sha256 of each host's download. `None` leaves every download
    /// unpinned; a component that pins any host must pin every one it downloads.
    pub(crate) sha256: Option<&'static [(Host, &'static str)]>,
    /// The build scripts that resolve it. They watch only the files they
    /// resolved, so touching them is what picks up a new install.
    pub(crate) watchers: &'static [&'static str],
    /// Whether the payload is a tool that reports its own version, and that a
    /// build falls back to finding on PATH when nothing is vendored.
    pub(crate) tool: bool,
    /// Whether a bare `vendor` leaves it out unless named or given `--all`.
    pub(crate) optional: bool,
}

/// Where a host gets a component from.
pub(crate) enum Source {
    /// An upstream archive, unpacked whole or cut to the members `keep` names.
    Download {
        url: String,
        keep: Option<&'static [&'static str]>,
    },
    /// Built here, from what no upstream publishes for the host.
    Build(fn(&Component, Host, &source_build::Options) -> Result),
}

impl Component {
    /// The host to act for, `None` when the component has no release for this
    /// machine. A bare verb applies to every component and most hosts can use
    /// only some, so that is a skip; naming the component makes it a failure.
    pub(crate) fn host_for(&self, current: Option<Host>, named: bool) -> Result<Option<Host>> {
        let host = current.filter(|host| self.hosts.contains(host));
        if host.is_none() && named {
            return fail(format!(
                "{} publishes no release for {}/{}",
                self.name,
                std::env::consts::OS,
                std::env::consts::ARCH
            ));
        }
        Ok(host)
    }

    pub(crate) fn install_dir(&self, host: Host) -> PathBuf {
        vendor_dir().join(format!("{}-{}-{}", self.name, self.release, host.slug()))
    }

    pub(crate) fn is_built(&self, host: Host) -> bool {
        matches!((self.source)(host), Source::Build(_))
    }

    pub(crate) fn pinned_digest(&self, host: Host) -> Option<&'static str> {
        self.sha256?
            .iter()
            .find(|(pinned, _)| *pinned == host)
            .map(|(_, digest)| *digest)
    }

    /// Every vendored release of this component for `host` under `vendor`,
    /// installed or not, sorted by name.
    pub(crate) fn vendored_in(&self, vendor: &Path, host: Host) -> Vec<PathBuf> {
        let Ok(entries) = std::fs::read_dir(vendor) else {
            return Vec::new();
        };
        let mut found: Vec<PathBuf> = entries
            .flatten()
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| self.is_release_dir(name, host))
            })
            .map(|entry| entry.path())
            .filter(|path| path.join((self.payload)(host)).is_file())
            .collect();
        found.sort();
        found
    }

    // Whether `dir` is `<name>-<version>-<slug>` with a dotted numeric version,
    // which keeps `fidelityfx` from claiming `fidelityfx-vk` releases.
    fn is_release_dir(&self, dir: &str, host: Host) -> bool {
        dir.strip_prefix(self.name)
            .and_then(|rest| rest.strip_prefix('-'))
            .and_then(|rest| rest.strip_suffix(host.slug()))
            .and_then(|rest| rest.strip_suffix('-'))
            .is_some_and(|version| {
                !version.is_empty()
                    && version
                        .split('.')
                        .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
            })
    }
}

/// Every component, in the order a bare verb visits them.
pub(crate) const COMPONENTS: [&Component; 6] = [
    &DXC,
    &AGILITY,
    &FIDELITYFX,
    &FIDELITYFX_VK,
    &XESS,
    &STREAMLINE,
];

/// Every component's name, for the command line to offer.
pub(crate) const NAMES: [&str; 6] = [
    DXC.name,
    AGILITY.name,
    FIDELITYFX.name,
    FIDELITYFX_VK.name,
    XESS.name,
    STREAMLINE.name,
];

pub(crate) fn find(name: &str) -> Option<&'static Component> {
    COMPONENTS
        .into_iter()
        .find(|component| component.name == name)
}

// The graphics SDKs below are Windows-only, so a bare `fetch` on another host
// skips them rather than failing: nothing on a Mac or a Linux box has any use
// for them. None ships a tool that reports its own version, so the payload
// being where the build script looks is the whole check.
const WINDOWS: &[Host] = &[Host::WindowsX64];
const SDK_WATCHERS: &[&str] = &[
    "build.rs",
    "crates/concinnity-device/build.rs",
    "crates/concinnity-runtime/build.rs",
    "crates/concinnity-dev/build.rs",
];

fn download(url: String) -> Source {
    Source::Download { url, keep: None }
}

/// The DirectX Shader Compiler every engine shader compiles through.
///
/// One release for every host, since the SPIR-V, DXIL and MSL a build embeds
/// and the runtime shader cache's key are all functions of it. Windows and
/// Linux take Microsoft's release archive, cut to the compiler, the two
/// libraries it loads and the license files. Microsoft publishes no macOS
/// build, so a Mac builds the same tagged commit from source (see
/// `source_build::dxc`).
///
/// Every copy reports the same commit, which is what the shader cache keys on.
/// The build counter beside it does not agree even between Microsoft's own
/// archives: the Windows one counts the full history, the Linux one a shallow
/// clone, as a source build here does too.
pub(crate) const DXC: Component = Component {
    name: "dxc",
    release: "1.9.2607",
    summary: "Microsoft's DirectX Shader Compiler, which compiles every engine shader",
    hosts: &Host::ALL,
    source: dxc_source,
    payload: |host| {
        if host.is_windows() {
            "bin/x64/dxc.exe"
        } else {
            "bin/dxc"
        }
    },
    sha256: Some(&[
        (
            Host::WindowsX64,
            "a1dfb116ba3eeae6a1582291b53a8e7bf65ad760676bd3194685c8f7367cd241",
        ),
        (
            Host::LinuxX64,
            "55665c87824051ed4774ff3280a79ccbbb7d39243b9736ca5e98222134112d54",
        ),
    ]),
    watchers: &["crates/concinnity-shader/build.rs"],
    tool: true,
    optional: false,
};

/// The commit `DXC.release` tags, which a source build checks it cloned.
pub(crate) const DXC_COMMIT: &str = "0d3ee6b551b8fa768fbf825300ebab81047ef6a8";
pub(crate) const DXC_REPO: &str = "https://github.com/microsoft/DirectXShaderCompiler";

// Upstream spells one of them LICENCE.
const DXC_LICENSES: &str = "LICEN[CS]E*";

fn dxc_source(host: Host) -> Source {
    // Upstream's asset names follow no scheme from one release to the next.
    // The Linux archive also carries LLVM's tools and static libraries, 1 GB
    // unpacked, so both are cut to the compiler, the libraries it loads, and
    // the licenses.
    let (asset, keep): (&str, &'static [&'static str]) = match host {
        Host::WindowsX64 => (
            "dxc_2026_07_29.zip",
            &[
                "bin/x64/dxc.exe",
                "bin/x64/dxcompiler.dll",
                "bin/x64/dxil.dll",
                DXC_LICENSES,
            ],
        ),
        Host::LinuxX64 => (
            "linux_dxc_2026_07_29.x86_x64.tar.gz",
            &[
                "bin/dxc",
                "lib/libdxcompiler.so",
                "lib/libdxil.so",
                DXC_LICENSES,
            ],
        ),
        Host::MacosArm64 => return Source::Build(source_build::dxc),
    };
    Source::Download {
        url: format!("{DXC_REPO}/releases/download/v{}/{asset}", DXC.release),
        keep: Some(keep),
    }
}

/// Microsoft's D3D12 Agility SDK, a package from Microsoft's own feed: a zip
/// under another name, which its flat container serves without an API key.
pub(crate) const AGILITY: Component = Component {
    name: "agility",
    release: "1.619.3",
    summary: "Microsoft's D3D12 Agility SDK, which FSR 3 needs",
    hosts: WINDOWS,
    source: |_| {
        let release = AGILITY.release;
        download(format!(
            "https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.d3d12/{release}/microsoft.direct3d.d3d12.{release}.nupkg"
        ))
    },
    payload: |_| "build/native/bin/x64/D3D12Core.dll",
    sha256: None,
    watchers: SDK_WATCHERS,
    tool: false,
    optional: false,
};

pub(crate) const FIDELITYFX: Component = Component {
    name: "fidelityfx",
    release: "1.1.4",
    summary: "AMD FidelityFX, for FSR 3 on DirectX",
    hosts: WINDOWS,
    source: |_| {
        let release = FIDELITYFX.release;
        download(format!(
            "https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases/download/v{release}/FidelityFX-SDK-v{release}.zip"
        ))
    },
    payload: |_| "bin/amd_fidelityfx_dx12.dll",
    sha256: None,
    watchers: SDK_WATCHERS,
    tool: false,
    optional: false,
};

/// AMD's Vulkan FSR runtime, rebuilt from the vendored SDK with one fix.
///
/// v1.1.4 declares the FSR3 upscaler's `rw_luma_history` storage image `rgba8`
/// in its GLSL callback while the C++ creates the resource as
/// `R16G16B16A16_FLOAT`, so every Vulkan FSR dispatch trips a validation-layer
/// format mismatch and the reads and stores are undefined
/// (<https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/issues/161>).
/// Declaring `rgba16f` matches the view, which is what
/// `patches/fsr3upscaler_luma_history_rgba16f.patch` records and what
/// `source_build::fidelityfx_vk` applies. Only the shader changes, so the
/// rebuilt DLL is ABI-identical to the stock one.
///
/// Its release is not `FIDELITYFX.release` but the last one with this Vulkan
/// backend, which is a different fact that happens to agree today: SDK v2
/// dropped Vulkan and deleted the GLSL callbacks this patches. The build
/// refuses once the two part.
pub(crate) const FIDELITYFX_VK: Component = Component {
    name: "fidelityfx-vk",
    release: "1.1.4",
    summary: "AMD FidelityFX Vulkan runtime, rebuilt with the FSR3 luma_history fix",
    hosts: WINDOWS,
    source: |_| Source::Build(source_build::fidelityfx_vk),
    payload: |_| "bin/amd_fidelityfx_vk.dll",
    sha256: None,
    watchers: SDK_WATCHERS,
    tool: false,
    // The build needs CMake, Visual Studio and the Vulkan SDK, and the stock
    // DLL works without it.
    optional: true,
};

pub(crate) const XESS: Component = Component {
    name: "xess",
    release: "3.0.1",
    summary: "Intel XeSS",
    hosts: WINDOWS,
    source: |_| {
        let release = XESS.release;
        download(format!(
            "https://github.com/intel/xess/releases/download/v{release}/XeSS_SDK_{release}.zip"
        ))
    },
    payload: |_| "bin/libxess.dll",
    sha256: None,
    watchers: SDK_WATCHERS,
    tool: false,
    optional: false,
};

pub(crate) const STREAMLINE: Component = Component {
    name: "streamline",
    release: "2.14.1",
    summary: "NVIDIA Streamline, for DLSS",
    hosts: WINDOWS,
    source: |_| {
        let release = STREAMLINE.release;
        download(format!(
            "https://github.com/NVIDIA-RTX/Streamline/releases/download/v{release}/streamline-sdk-v{release}.zip"
        ))
    },
    payload: |_| "bin/x64/nvngx_dlss.dll",
    sha256: Some(&[(
        Host::WindowsX64,
        "92c4d954631a1710da86ca3fa8d5034f2b9503838c95fc4ae977ae149319781b",
    )]),
    watchers: SDK_WATCHERS,
    tool: false,
    optional: false,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_follow_the_table() {
        let names: Vec<&str> = COMPONENTS.iter().map(|c| c.name).collect();
        assert_eq!(names, NAMES);
        assert!(find("dxc").is_some());
        assert!(find("slang").is_none());
    }

    #[test]
    fn a_pinned_component_pins_every_download() {
        for component in COMPONENTS {
            if component.sha256.is_none() {
                continue;
            }
            for &host in component.hosts {
                if !component.is_built(host) {
                    assert!(
                        component.pinned_digest(host).is_some(),
                        "{}/{} downloads with no pinned sha256",
                        component.name,
                        host.slug()
                    );
                }
            }
        }
    }

    #[test]
    fn each_host_downloads_or_builds_what_it_should() {
        let cases = [
            (&DXC, Host::WindowsX64, false),
            (&DXC, Host::LinuxX64, false),
            (&DXC, Host::MacosArm64, true),
            (&AGILITY, Host::WindowsX64, false),
            (&FIDELITYFX_VK, Host::WindowsX64, true),
            (&STREAMLINE, Host::WindowsX64, false),
        ];
        for (component, host, built) in cases {
            assert_eq!(
                component.is_built(host),
                built,
                "{}/{}",
                component.name,
                host.slug()
            );
        }
    }

    #[test]
    fn the_vulkan_runtime_tracks_the_sdk_it_is_built_from() {
        assert_eq!(FIDELITYFX_VK.release, FIDELITYFX.release);
    }

    #[test]
    fn a_host_without_a_release_is_skipped_unless_named() {
        assert_eq!(
            XESS.host_for(Some(Host::MacosArm64), false).ok(),
            Some(None)
        );
        assert!(XESS.host_for(Some(Host::MacosArm64), true).is_err());
        assert_eq!(
            XESS.host_for(Some(Host::WindowsX64), true).ok(),
            Some(Some(Host::WindowsX64))
        );
        assert!(DXC.host_for(None, true).is_err());
    }

    #[test]
    fn release_dirs_match_by_name_version_and_slug() {
        let host = Host::WindowsX64;
        assert!(FIDELITYFX.is_release_dir("fidelityfx-1.1.4-windows-x86_64", host));
        assert!(!FIDELITYFX.is_release_dir("fidelityfx-vk-1.1.4-windows-x86_64", host));
        assert!(FIDELITYFX_VK.is_release_dir("fidelityfx-vk-1.1.4-windows-x86_64", host));
        assert!(!FIDELITYFX.is_release_dir("fidelityfx-1.1.4-linux-x86_64", host));
        assert!(!FIDELITYFX.is_release_dir("fidelityfx--windows-x86_64", host));
        assert!(!FIDELITYFX.is_release_dir(".fidelityfx-1.1.4-windows-x86_64.incoming", host));
    }

    #[test]
    fn only_installed_releases_count_as_vendored() {
        let vendor = tempfile::tempdir().unwrap();
        let host = Host::LinuxX64;
        let installed = vendor.path().join("dxc-1.9.2607-linux-x86_64");
        std::fs::create_dir_all(installed.join("bin")).unwrap();
        std::fs::write(installed.join("bin/dxc"), b"").unwrap();
        std::fs::create_dir_all(vendor.path().join("dxc-1.8.0-linux-x86_64")).unwrap();
        assert_eq!(DXC.vendored_in(vendor.path(), host), vec![installed]);
        assert!(
            DXC.vendored_in(&vendor.path().join("absent"), host)
                .is_empty()
        );
    }
}
