//! The tools a development run and an export compile with: the shader compiler
//! every engine shader goes through.

use super::component::{Component, Source};
use super::host::Host;

/// The DirectX Shader Compiler every engine shader compiles through.
///
/// One release for every host, since the SPIR-V, DXIL and MSL the engine
/// compiles and the runtime shader cache's key are all functions of it. Windows
/// and Linux take Microsoft's release archive, cut to the compiler, the two
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
    tool: true,
    runtime: &[],
    notices: &[],
    licenses: &[],
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
        Host::MacosArm64 => return Source::Built,
    };
    Source::Download {
        url: format!("{DXC_REPO}/releases/download/v{}/{asset}", DXC.release),
        keep: Some(keep),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dxc_pins_every_download() {
        for &host in DXC.hosts {
            if !DXC.is_built(host) {
                assert!(DXC.pinned_digest(host).is_some(), "{}", host.slug());
            }
        }
    }

    #[test]
    fn only_a_mac_builds_dxc() {
        assert!(!DXC.is_built(Host::WindowsX64));
        assert!(!DXC.is_built(Host::LinuxX64));
        assert!(DXC.is_built(Host::MacosArm64));
    }

    #[test]
    fn a_vendored_dxc_is_the_one_the_shader_compiler_finds() {
        let Some(host) = Host::current() else {
            return;
        };
        let vendor = tempfile::tempdir().unwrap();
        let exe = DXC
            .install_dir(vendor.path(), host)
            .join((DXC.payload)(host));
        std::fs::create_dir_all(exe.parent().unwrap()).unwrap();
        std::fs::write(&exe, b"").unwrap();
        assert_eq!(concinnity_shader::vendored_dxc(vendor.path()), vec![exe]);
    }
}
