//! What `cargo xtask vendor` vendors: the shader compiler the build itself
//! needs. The runtime SDKs an app loads are vendored by `concinnity vendor`
//! instead, which the engine crates can depend on.

use concinnity_vendor::{Component, Host, Source};

/// Every component, in the order a bare verb visits them.
pub(crate) const COMPONENTS: [&Component; 1] = [&DXC];

/// Every component's name, for the command line to offer.
pub(crate) const NAMES: [&str; 1] = [DXC.name];

pub(crate) fn find(name: &str) -> Option<&'static Component> {
    COMPONENTS
        .into_iter()
        .find(|component| component.name == name)
}

/// The build scripts that resolve `component`. They watch only the files they
/// resolved, so touching them is what picks up a new install.
pub(crate) fn watchers(component: &Component) -> &'static [&'static str] {
    match component.name {
        "dxc" => &["crates/concinnity-shader/build.rs"],
        _ => &[],
    }
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
    fn names_follow_the_table() {
        let names: Vec<&str> = COMPONENTS.iter().map(|c| c.name).collect();
        assert_eq!(names, NAMES);
        assert!(find("dxc").is_some());
        assert!(find("xess").is_none());
    }

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
    fn only_installed_releases_count_as_vendored() {
        let vendor = tempfile::tempdir().unwrap();
        let host = Host::LinuxX64;
        let installed = vendor.path().join("dxc-1.9.2607-linux-x86_64");
        std::fs::create_dir_all(installed.join("bin")).unwrap();
        std::fs::write(installed.join("bin/dxc"), b"").unwrap();
        std::fs::create_dir_all(vendor.path().join("dxc-1.8.0-linux-x86_64")).unwrap();
        assert_eq!(DXC.vendored_in(vendor.path(), host), vec![installed]);
    }
}
