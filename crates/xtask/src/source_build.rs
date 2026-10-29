//! `vendor build`: produce a component the build resolves but no upstream
//! publishes for this host.

use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use crate::components::{Component, DXC_COMMIT, DXC_REPO, FIDELITYFX, Source};
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
    /// Generator for cmake to build with
    #[arg(long, default_value = "Visual Studio 18 2026")]
    pub(crate) generator: String,
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

/// Where FidelityFX v1.1.4 declares the image the luma history fix retypes.
const FSR3_SHADER: &str =
    "sdk/include/FidelityFX/gpu/fsr3upscaler/ffx_fsr3upscaler_callbacks_glsl.h";

/// Rebuild the FidelityFX Vulkan runtime from the vendored SDK with the luma
/// history fix applied (see `components::FIDELITYFX_VK`).
pub(crate) fn fidelityfx_vk(component: &Component, host: Host, options: &Options) -> Result {
    if FIDELITYFX.release != component.release {
        return fail(format!(
            "{} is pinned to {}, and {} is the last release to build a Vulkan runtime from",
            FIDELITYFX.name, FIDELITYFX.release, component.release
        ));
    }
    let sdk = FIDELITYFX.install_dir(host);
    let shader = sdk.join(FSR3_SHADER);
    if !shader.is_file() {
        return fail(format!(
            "{} is not vendored at {}; run `cargo xtask vendor fetch {}` first",
            FIDELITYFX.name,
            file_name(&sdk),
            FIDELITYFX.name
        ));
    }
    tools::require("cmake")?;
    if !vulkan_sdk_set() {
        println!("  warning: VULKAN_SDK is unset, so the shader build may find no glslc");
    }

    fix_shader(&shader)?;
    // ffx-api pulls in the `sdk` subproject, which is what recompiles the
    // shader permutations the fix reaches.
    let api = sdk.join("ffx-api");
    let tree = api.join("build-vk");
    tools::run(
        Command::new("cmake")
            .arg("-S")
            .arg(&api)
            .arg("-B")
            .arg(&tree)
            .args([
                "-G",
                &options.generator,
                "-A",
                "x64",
                "-DFFX_API_BACKEND=VK_X64",
            ]),
    )?;
    tools::run(Command::new("cmake").arg("--build").arg(&tree).args([
        "--config",
        "Release",
        "--parallel",
    ]))?;

    // Release carries no postfix.
    let built = api.join("bin/amd_fidelityfx_vk.dll");
    if !built.is_file() {
        return fail(format!(
            "the build left no amd_fidelityfx_vk.dll at {}",
            built.display()
        ));
    }
    install(
        &built,
        &component.install_dir(host).join((component.payload)(host)),
    )
}

#[expect(
    clippy::disallowed_methods,
    reason = "the Vulkan SDK installer publishes its location only as VULKAN_SDK"
)]
fn vulkan_sdk_set() -> bool {
    std::env::var_os("VULKAN_SDK").is_some_and(|sdk| !sdk.is_empty())
}

fn fix_shader(shader: &Path) -> Result {
    let source = fs::read_to_string(shader).context(shader.display())?;
    match patch_luma_history(&source) {
        Ok(None) => println!("  shader already declares rgba16f"),
        Ok(Some(patched)) => {
            fs::write(shader, patched).context(shader.display())?;
            println!("  patched rw_luma_history: rgba8 -> rgba16f");
        }
        Err(count) => {
            return fail(format!(
                "{} holds {count} rgba8 rw_luma_history declarations to fix, not one; \
                 the SDK layout differs from v{}",
                shader.display(),
                FIDELITYFX.release
            ));
        }
    }
    Ok(())
}

const LUMA_HISTORY_BINDING: &str = "binding = FSR3UPSCALER_BIND_UAV_LUMA_HISTORY, ";

/// `source` with `rw_luma_history` declared `rgba16f`, `None` if it already
/// is, or how many `rgba8` declarations it found when that is not exactly one.
/// `patches/fsr3upscaler_luma_history_rgba16f.patch` records the same change;
/// matching the declaration rather than applying the diff survives line drift.
fn patch_luma_history(source: &str) -> std::result::Result<Option<String>, usize> {
    if !luma_history_declarations(source, "rgba16f").is_empty() {
        return Ok(None);
    }
    let found: Vec<usize> = luma_history_declarations(source, "rgba8")
        .into_iter()
        .filter(|&at| source[..at].ends_with(LUMA_HISTORY_BINDING))
        .collect();
    let [at] = found[..] else {
        return Err(found.len());
    };
    Ok(Some(format!(
        "{}rgba16f{}",
        &source[..at],
        &source[at + "rgba8".len()..]
    )))
}

// The offset of each `<format>) uniform image2D <whitespace> rw_luma_history`.
fn luma_history_declarations(source: &str, format: &str) -> Vec<usize> {
    let head = format!("{format}) uniform image2D");
    source
        .match_indices(&head)
        .map(|(at, _)| at)
        .filter(|&at| {
            let rest = &source[at + head.len()..];
            let name = rest.trim_start();
            name.len() < rest.len() && name.starts_with("rw_luma_history")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const STOCK: &str = "layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_LUMA_HISTORY, rgba8) uniform image2D  rw_luma_history;\n";

    #[test]
    fn the_stock_declaration_is_retyped() {
        let source = format!("// before\n{STOCK}// after\n");
        let patched = patch_luma_history(&source).unwrap().unwrap();
        assert_eq!(patched, source.replace("rgba8", "rgba16f"));
        assert_eq!(patch_luma_history(&patched), Ok(None));
    }

    #[test]
    fn the_recorded_patch_makes_the_same_change() {
        let patch = include_str!("../patches/fsr3upscaler_luma_history_rgba16f.patch");
        let removed = patch
            .lines()
            .find_map(|l| l.strip_prefix("-layout"))
            .unwrap();
        let added = patch
            .lines()
            .find_map(|l| l.strip_prefix("+layout"))
            .unwrap();
        let before = format!("layout{removed}\n");
        assert_eq!(before, STOCK);
        assert_eq!(
            patch_luma_history(&before),
            Ok(Some(format!("layout{added}\n")))
        );
    }

    #[test]
    fn a_declaration_on_another_binding_is_not_touched() {
        let source = "layout (binding = OTHER, rgba8) uniform image2D rw_luma_history;\n";
        assert_eq!(patch_luma_history(source), Err(0));
    }

    #[test]
    fn a_shader_without_the_declaration_is_refused() {
        assert_eq!(patch_luma_history("// nothing here\n"), Err(0));
        assert_eq!(patch_luma_history(&STOCK.repeat(2)), Err(2));
    }

    #[test]
    fn the_name_must_follow_whitespace() {
        let glued = STOCK.replace("image2D  rw", "image2Drw");
        assert_eq!(patch_luma_history(&glued), Err(0));
    }
}
