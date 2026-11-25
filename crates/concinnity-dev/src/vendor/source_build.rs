//! Building a component no upstream publishes for this host.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use super::component::Component;
use super::files::{install, remove_dir_quietly, staging_dir};
use super::host::Host;
use super::process::{output, require, run};
use super::tools::{DXC, DXC_COMMIT, DXC_REPO};
use super::{Context, Result, fail};

/// Build `component` for `host` into `vendor`, returning its directory.
pub(crate) fn build(component: &Component, host: Host, vendor: &Path) -> Result<PathBuf> {
    if component.name != DXC.name {
        return fail(format!("{} has no source build", component.name));
    }
    println!(
        "{}: building {} from source",
        component.name, component.release
    );
    dxc(component, host, vendor)?;
    let target = component.install_dir(vendor, host);
    println!("{}: built {}", component.name, target.display());
    Ok(target)
}

// The submodules a compiler build needs. The tests' googletest is the fourth,
// and nothing here builds a test.
const DXC_SUBMODULES: [&str; 3] = [
    "external/SPIRV-Headers",
    "external/SPIRV-Tools",
    "external/DirectX-Headers",
];
const DXC_SOURCE_LICENSES: [&str; 2] = ["LICENSE.TXT", "ThirdPartyNotices.txt"];

// Build dxc from the commit its release tags: a shallow clone of it and three
// submodules, then `dxc`, `dxcompiler` and `dxildll` in Release. That costs a
// compile measured in minutes (3.5 on a 12-core Apple silicon Mac), once per
// pin; the tree it builds in is deleted afterwards and only the installed
// release stays.
fn dxc(component: &Component, host: Host, vendor: &Path) -> Result {
    for tool in ["git", "cmake", "ninja"] {
        require(tool)?;
    }
    let work = vendor
        .join(".build")
        .join(format!("{}-{}", component.name, component.release));
    let (src, tree) = (work.join("src"), work.join("build"));
    let started = Instant::now();

    if !src.join(".git").is_dir() {
        remove_dir_quietly(&src);
        let tag = format!("v{}", component.release);
        run(Command::new("git")
            .args(["clone", "--depth", "1", "--branch", &tag, DXC_REPO])
            .arg(&src))?;
    }
    let head = output(
        Command::new("git")
            .arg("-C")
            .arg(&src)
            .args(["rev-parse", "HEAD"]),
    )?;
    if head != DXC_COMMIT {
        return fail(format!(
            "v{} resolved to {head}, not the pinned {DXC_COMMIT}",
            component.release
        ));
    }
    run(Command::new("git")
        .arg("-C")
        .arg(&src)
        .args(["submodule", "update", "--init", "--depth", "1", "--"])
        .args(DXC_SUBMODULES))?;

    // Explicit values before `-C`: the cache script never overrides one.
    run(Command::new("cmake")
        .args(["-G", "Ninja", "-S"])
        .arg(&src)
        .arg("-B")
        .arg(&tree)
        .args([
            "-DCMAKE_BUILD_TYPE=Release",
            "-DHLSL_INCLUDE_TESTS=OFF",
            "-DSPIRV_BUILD_TESTS=OFF",
            "-DLLVM_INCLUDE_TESTS=OFF",
            "-DCLANG_INCLUDE_TESTS=OFF",
            "-C",
        ])
        .arg(src.join("cmake/caches/PredefinedParams.cmake")))?;
    run(Command::new("cmake")
        .arg("--build")
        .arg(&tree)
        .args(["--target", "dxc", "dxildll"]))?;

    let target = component.install_dir(vendor, host);
    let staging = staging_dir(&target);
    remove_dir_quietly(&staging);
    let suffix = if host == Host::MacosArm64 {
        ".dylib"
    } else {
        ".so"
    };
    install(
        &tree.join("bin/dxc"),
        &staging.join((component.payload)(host)),
    )?;
    for library in ["libdxcompiler", "libdxil"] {
        let file = format!("{library}{suffix}");
        // The real file under the name the executable loads, not a symlink to it.
        let built = tree.join("lib").join(&file);
        let real = fs::canonicalize(&built).context(built.display())?;
        install(&real, &staging.join("lib").join(&file))?;
    }
    for name in DXC_SOURCE_LICENSES {
        install(&src.join(name), &staging.join(name))?;
    }
    remove_dir_quietly(&target);
    fs::rename(&staging, &target).context(target.display())?;

    fs::remove_dir_all(&work).context(work.display())?;
    if let Some(parent) = work.parent() {
        // Only an empty `.build/` goes; another component may be building in it.
        let _removed = fs::remove_dir(parent);
    }
    println!(
        "  built in {:.1} min",
        started.elapsed().as_secs_f64() / 60.0
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vendor::sdks::XESS;

    #[test]
    fn only_dxc_has_a_source_build() {
        let vendor = tempfile::tempdir().unwrap();
        let Err(failure) = build(&XESS, Host::WindowsX64, vendor.path()) else {
            panic!("xess built from source");
        };
        assert!(failure.0.contains("no source build"));
    }
}
