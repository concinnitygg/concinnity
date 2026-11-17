//! `vendor build`: produce a component the build resolves but no upstream
//! publishes for this host.

use std::fs;
use std::process::Command;
use std::time::Instant;

use crate::components::{Component, DXC_COMMIT, DXC_REPO, Source};
use crate::fetch::file_name;
use crate::host::Host;
use crate::tools::{self, install, remove_dir_quietly, staging_dir};
use crate::{Context, Result, fail, status, vendor_dir, workspace_root};

/// How a source build runs.
#[derive(clap::Args)]
pub(crate) struct Options {
    /// Redo it over an existing copy
    #[arg(long)]
    pub(crate) force: bool,
}

pub(crate) fn build(
    component: &Component,
    current: Option<Host>,
    named: bool,
    options: &Options,
) -> Result {
    let name = component.name;
    let Some(host) = component.host_for(current, named)? else {
        println!("{name}: no build for this host, skipped");
        return Ok(());
    };
    let Source::Build(build) = (component.source)(host) else {
        if named {
            return fail(format!(
                "{name} is downloaded here, not built -- run `cargo xtask vendor fetch {name}`"
            ));
        }
        println!("{name}: downloaded here, skipped");
        return Ok(());
    };
    let target = component.install_dir(host);
    if target.join((component.payload)(host)).is_file() && !options.force {
        println!("{name}: already built, {}", file_name(&target));
        return Ok(());
    }

    println!("{name}: building {}", component.release);
    build(component, host, options)?;
    tools::touch_watchers(component);
    let shown = target.strip_prefix(workspace_root()).unwrap_or(&target);
    println!("{name}: built {}", shown.display());
    status::print_report(component, &target, host);
    Ok(())
}

// The submodules a compiler build needs. The tests' googletest is the fourth,
// and nothing here builds a test.
const DXC_SUBMODULES: [&str; 3] = [
    "external/SPIRV-Headers",
    "external/SPIRV-Tools",
    "external/DirectX-Headers",
];
const DXC_SOURCE_LICENSES: [&str; 2] = ["LICENSE.TXT", "ThirdPartyNotices.txt"];

/// Build dxc from the commit its release tags: a shallow clone of it and three
/// submodules, then `dxc`, `dxcompiler` and `dxildll` in Release. That costs a
/// compile measured in minutes (3.5 on a 12-core Apple silicon Mac), once per
/// pin; the tree it builds in is deleted afterwards and only the installed
/// release stays under `vendor/`.
pub(crate) fn dxc(component: &Component, host: Host, _options: &Options) -> Result {
    for tool in ["git", "cmake", "ninja"] {
        tools::require(tool)?;
    }
    let work = vendor_dir()
        .join(".build")
        .join(format!("{}-{}", component.name, component.release));
    let (src, tree) = (work.join("src"), work.join("build"));
    let started = Instant::now();

    if !src.join(".git").is_dir() {
        remove_dir_quietly(&src);
        let tag = format!("v{}", component.release);
        tools::run(
            Command::new("git")
                .args(["clone", "--depth", "1", "--branch", &tag, DXC_REPO])
                .arg(&src),
        )?;
    }
    let head = tools::output(
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
    tools::run(
        Command::new("git")
            .arg("-C")
            .arg(&src)
            .args(["submodule", "update", "--init", "--depth", "1", "--"])
            .args(DXC_SUBMODULES),
    )?;

    // Explicit values before `-C`: the cache script never overrides one.
    tools::run(
        Command::new("cmake")
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
            .arg(src.join("cmake/caches/PredefinedParams.cmake")),
    )?;
    tools::run(
        Command::new("cmake")
            .arg("--build")
            .arg(&tree)
            .args(["--target", "dxc", "dxildll"]),
    )?;

    let target = component.install_dir(host);
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
