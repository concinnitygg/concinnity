// Graphics SDK probing, DLL bundling, and `cargo::` directive generation.
//
// Nothing in this module reads the process environment or prints to stdout:
// inputs arrive in `SdkEnv` (populated from the real build-script environment
// by the wrappers in lib.rs) and directives are returned as strings for the
// caller to print. File-system access is limited to the paths named in
// `SdkEnv`, so tests can drive the full probe/copy logic against fabricated
// SDK trees.

use std::path::{Path, PathBuf};

use crate::{Backend, BinaryTargets};

#[cfg(test)]
mod tests;

// Environment inputs for the SDK setup: the values the build script reads from
// Cargo's environment plus the SDK install roots (env-var override already
// applied over the hardcoded default). The `*_enabled` flags carry the
// `CN_ENABLE_*` opt-outs (true unless the variable is set to "0").
#[derive(Clone, Debug)]
pub(crate) struct SdkEnv {
    pub(crate) target_os: String,
    pub(crate) out_dir: Option<PathBuf>,
    // `None` where neither the variable nor `vendor/` named a root. There is no
    // fall back to a guessed install path: every one of these unpacks wherever
    // its user put it, so a hardcoded location was only ever right by accident.
    pub(crate) agility_root: Option<PathBuf>,
    pub(crate) fidelityfx_root: Option<PathBuf>,
    pub(crate) fidelityfx_vk_root: Option<PathBuf>,
    pub(crate) xess_root: Option<PathBuf>,
    pub(crate) streamline_root: Option<PathBuf>,
    pub(crate) dxc_root: Option<PathBuf>,
    pub(crate) windows_sdk_bin: Option<PathBuf>,
    pub(crate) agility_enabled: bool,
    pub(crate) ffx_enabled: bool,
    pub(crate) xess_enabled: bool,
    pub(crate) dlss_enabled: bool,
    pub(crate) dxc_enabled: bool,
}

pub(crate) fn check_cfg_directives() -> Vec<String> {
    [
        "backend_metal",
        "backend_dx",
        "backend_vk",
        "agility_sdk_configured",
        "ffx_sdk_bundled",
        "xess_sdk_bundled",
        "ngx_sdk_bundled",
        "dxc_bundled",
    ]
    .iter()
    .map(|cfg| format!("cargo::rustc-check-cfg=cfg({cfg})"))
    .collect()
}

pub(crate) fn backend_cfg_directive(backend: Backend) -> String {
    rustc_cfg(backend.cfg_name())
}

// The full SDK setup for one backend across every kind of final binary the
// calling package builds. Each kind takes its own linker-argument key and its
// own directory for the bundled DLLs, so the setup runs once per kind; the
// directives both kinds produce -- every cfg, every warning, the unscoped NGX
// link -- are emitted once.
pub(crate) fn graphics_sdk_directives(
    backend: Backend,
    targets: &[BinaryTargets],
    env: &SdkEnv,
) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for &target in targets {
        for line in directives_for_kind(backend, target, env) {
            if !out.contains(&line) {
                out.push(line);
            }
        }
    }
    out
}

// On a non-Windows target (or the Metal backend) this returns nothing: none of
// these SDKs apply.
fn directives_for_kind(backend: Backend, targets: BinaryTargets, env: &SdkEnv) -> Vec<String> {
    let mut out = Vec::new();
    match backend {
        Backend::Dx => {
            agility_directives(env, targets, &mut out);
            fidelityfx_dx_directives(env, targets, &mut out);
            xess_directives(env, targets, &mut out);
            dlss_directives(env, targets, &mut out);
            if targets.bundles() {
                dxc_directives(env, targets, &mut out);
            }
        }
        Backend::Vk if env.target_os == "windows" => {
            // DLSS (NGX) and XeSS expose Vulkan entry points from the same
            // binaries the DirectX backend uses, so the setup helpers are
            // backend-agnostic and reused here. Windowing comes from the
            // shared native Win32 layer (no GLFW DLL to bundle).
            fidelityfx_vk_directives(env, targets, &mut out);
            dlss_directives(env, targets, &mut out);
            xess_directives(env, targets, &mut out);
        }
        _ => {}
    }
    out
}

// Microsoft's Agility SDK D3D12 runtime, bundled only when the build asks for
// it with `CN_ENABLE_AGILITY_SDK=1`. Unlike the four `LoadLibrary` SDKs around
// it, bundling this one binds the executable to the `D3D12/` directory staged
// beside it -- `d3d12.dll` reads the linked exports before any engine code runs
// and fails device creation outright when the directory is absent -- so it is
// off unless the artifact is being built to travel with that directory. Nothing
// is warned about when it is off: that is the ordinary build, and the FSR3
// upscaler reports its own loss when something actually asks for it.
//
// The DLL copy and the exports are only emitted when bundling for a final
// binary; the `agility_sdk_configured` cfg is emitted whenever the opt-in found
// the SDK, so the runtime FSR3 gate matches what the binary actually carries.
// A file inside an SDK, when a root resolved at all. Split from the existence
// check so an unresolved root and a root missing the file are one branch: both
// mean the SDK is unusable, and the warning says which it was.
fn sdk_file(root: Option<&PathBuf>, parts: &[&str]) -> Option<PathBuf> {
    let mut path = root?.clone();
    path.extend(parts);
    path.exists().then_some(path)
}

// Why an SDK is unusable, in the terms whoever has to fix it needs: the path
// that was searched, or that nothing named one.
fn missing_sdk(sdk: &str, var: &str, root: Option<&PathBuf>, parts: &[&str]) -> String {
    match root {
        Some(root) => {
            let mut path = root.clone();
            path.extend(parts);
            format!("{sdk} not found at {}", path.display())
        }
        None => format!("{sdk} not vendored and {var} is not set"),
    }
}

fn agility_directives(env: &SdkEnv, targets: BinaryTargets, out: &mut Vec<String>) {
    out.push(rerun_env("CN_ENABLE_AGILITY_SDK"));
    if !env.agility_enabled {
        return;
    }
    out.push(rerun_env("CN_AGILITY_SDK"));

    const AGILITY_CORE: &[&str] = &["build", "native", "bin", "x64", "D3D12Core.dll"];
    let Some(core_dll) = sdk_file(env.agility_root.as_ref(), AGILITY_CORE) else {
        if targets.bundles() {
            out.push(warning(&format!(
                "CN_ENABLE_AGILITY_SDK is set but the {} - vendor it or set \
                 CN_AGILITY_SDK. FidelityFX FSR3 will be unavailable (the binary \
                 falls back to the OS-bundled D3D12 runtime).",
                missing_sdk(
                    "Agility SDK",
                    "CN_AGILITY_SDK",
                    env.agility_root.as_ref(),
                    AGILITY_CORE
                )
            )));
        }
        return;
    };
    // The directory the runtime DLLs are copied from, which the file above
    // proves exists.
    let sdk_bin = core_dll
        .parent()
        .expect("a file has a parent")
        .to_path_buf();

    if targets.bundles() {
        // `D3D12SDKPath = ".\\D3D12\\"` resolves relative to the .exe, so the
        // DLLs must live in a `D3D12/` beside the binary -- which for an
        // example is `<target>/<profile>/examples/`, not the profile directory.
        let Some(exe_dir) = exe_dir(env, targets) else {
            return;
        };
        let d3d12_dir = exe_dir.join("D3D12");
        if let Err(e) = std::fs::create_dir_all(&d3d12_dir) {
            out.push(warning(&format!(
                "Agility SDK: could not create {}: {e}",
                d3d12_dir.display()
            )));
            return;
        }
        for dll in ["D3D12Core.dll", "d3d12SDKLayers.dll"] {
            let src = sdk_bin.join(dll);
            let dst = d3d12_dir.join(dll);
            if let Err(e) = std::fs::copy(&src, &dst) {
                out.push(warning(&format!(
                    "Agility SDK: could not copy {} -> {}: {e}",
                    src.display(),
                    dst.display()
                )));
                return;
            }
            out.push(rerun_path(&src));
        }

        // Export the two symbols `d3d12.dll` reads at process start. `,DATA` is
        // critical: without it the linker inserts a code thunk that `d3d12.dll`
        // would dereference as a pointer. `/EXPORT` also demands the symbol
        // resolve, so every target the key covers has to define the statics.
        if let Some(key) = targets.link_arg_key() {
            out.push(format!("{key}=/EXPORT:D3D12SDKVersion,DATA"));
            out.push(format!("{key}=/EXPORT:D3D12SDKPath,DATA"));
        }
    }

    out.push(rustc_cfg("agility_sdk_configured"));
}

// AMD FidelityFX DX12 upscaler runtime. The renderer loads the DLL with
// `LoadLibrary` at runtime, so bundling only copies it next to the .exe; the
// `ffx_sdk_bundled` cfg is emitted when the SDK is present regardless.
fn fidelityfx_dx_directives(env: &SdkEnv, targets: BinaryTargets, out: &mut Vec<String>) {
    out.push(rerun_env("CN_ENABLE_FFX_FSR3"));
    if !env.ffx_enabled {
        if targets.bundles() {
            out.push(warning(
                "FidelityFX SDK bundling skipped (CN_ENABLE_FFX_FSR3=0); \
                 temporal upscaling will be unavailable unless amd_fidelityfx_dx12.dll \
                 is on PATH at runtime",
            ));
        }
        return;
    }
    out.push(rerun_env("CN_FIDELITYFX_SDK"));

    const FFX_DX: &[&str] = &["bin", "amd_fidelityfx_dx12.dll"];
    let Some(dll_src) = sdk_file(env.fidelityfx_root.as_ref(), FFX_DX) else {
        if targets.bundles() {
            out.push(warning(&format!(
                "{} - vendor it or set CN_FIDELITYFX_SDK. Temporal upscaling will \
                 be unavailable unless amd_fidelityfx_dx12.dll is on PATH at runtime.",
                missing_sdk(
                    "FidelityFX SDK",
                    "CN_FIDELITYFX_SDK",
                    env.fidelityfx_root.as_ref(),
                    FFX_DX
                )
            )));
        }
        return;
    };

    if targets.bundles()
        && !copy_next_to_exe(env, targets, &dll_src, "amd_fidelityfx_dx12.dll", out)
    {
        return;
    }
    out.push(rustc_cfg("ffx_sdk_bundled"));
}

// AMD FidelityFX Vulkan upscaler runtime. Prefers the rebuilt DLL under
// `vendor/fidelityfx-vk-*` (carries the FSR3 rw_luma_history format fix),
// falling back to the stock SDK copy, which warns every dispatch.
fn fidelityfx_vk_directives(env: &SdkEnv, targets: BinaryTargets, out: &mut Vec<String>) {
    out.push(rerun_env("CN_ENABLE_FFX_FSR3"));
    if !env.ffx_enabled {
        if targets.bundles() {
            out.push(warning(
                "FidelityFX SDK bundling skipped (CN_ENABLE_FFX_FSR3=0); \
                 Vulkan temporal upscaling will be unavailable unless amd_fidelityfx_vk.dll \
                 is on PATH at runtime",
            ));
        }
        return;
    }
    out.push(rerun_env("CN_FIDELITYFX_SDK"));

    // Both roots hold the runtime at the same place, so the rebuilt copy is
    // reached by preferring its root rather than by naming a second path.
    const FFX_VK: &[&str] = &["bin", "amd_fidelityfx_vk.dll"];
    let dll_src = sdk_file(env.fidelityfx_vk_root.as_ref(), FFX_VK)
        .or_else(|| sdk_file(env.fidelityfx_root.as_ref(), FFX_VK));
    let Some(dll_src) = dll_src else {
        if targets.bundles() {
            out.push(warning(&format!(
                "FidelityFX VK runtime not found: nothing built by \
                 `vendor.py build fidelityfx-vk`, and {}. Build the patched \
                 runtime, vendor the SDK, set CN_FIDELITYFX_SDK, or put \
                 amd_fidelityfx_vk.dll on PATH at runtime; Vulkan temporal \
                 upscaling will fall back to native resolution.",
                missing_sdk(
                    "the SDK's own copy is",
                    "CN_FIDELITYFX_SDK",
                    env.fidelityfx_root.as_ref(),
                    FFX_VK
                )
            )));
        }
        return;
    };

    if targets.bundles() && !copy_next_to_exe(env, targets, &dll_src, "amd_fidelityfx_vk.dll", out)
    {
        return;
    }
    out.push(rustc_cfg("ffx_sdk_bundled"));
}

// Intel XeSS upscaler runtime. Pure `LoadLibrary` at runtime, so bundling only
// copies the DLL; the cfg gates the copy and a log.
fn xess_directives(env: &SdkEnv, targets: BinaryTargets, out: &mut Vec<String>) {
    out.push(rerun_env("CN_ENABLE_XESS"));
    if !env.xess_enabled {
        if targets.bundles() {
            out.push(warning(
                "XeSS SDK bundling skipped (CN_ENABLE_XESS=0); the XeSS \
                 upscaler will be unavailable unless libxess.dll is on PATH at runtime",
            ));
        }
        return;
    }
    out.push(rerun_env("CN_XESS_SDK"));

    const XESS_DLL: &[&str] = &["bin", "libxess.dll"];
    let Some(dll_src) = sdk_file(env.xess_root.as_ref(), XESS_DLL) else {
        if targets.bundles() {
            out.push(warning(&format!(
                "{} - vendor it or set CN_XESS_SDK. The XeSS upscaler backend will \
                 be unavailable unless libxess.dll is on PATH at runtime.",
                missing_sdk("XeSS SDK", "CN_XESS_SDK", env.xess_root.as_ref(), XESS_DLL)
            )));
        }
        return;
    };

    if targets.bundles() && !copy_next_to_exe(env, targets, &dll_src, "libxess.dll", out) {
        return;
    }
    out.push(rustc_cfg("xess_sdk_bundled"));
}

// DLSS via raw NGX. The import lib is always linked when present (the DLSS code
// compiled into the runtime rlib references its symbols, so every final binary
// and the rlib's own tests must resolve them). When bundling for a final binary
// the feature DLL `nvngx_dlss.dll` is also copied next to the .exe.
fn dlss_directives(env: &SdkEnv, targets: BinaryTargets, out: &mut Vec<String>) {
    out.push(rerun_env("CN_ENABLE_DLSS"));
    if !env.dlss_enabled {
        if targets.bundles() {
            out.push(warning(
                "DLSS (NGX) setup skipped (CN_ENABLE_DLSS=0); the DLSS \
                 upscaler backend will be unavailable",
            ));
        }
        return;
    }
    out.push(rerun_env("CN_STREAMLINE_SDK"));

    const NGX_LIB: &[&str] = &[
        "external",
        "ngx-sdk",
        "lib",
        "Windows_x86_64",
        "nvsdk_ngx_d.lib",
    ];
    let Some(ngx_lib) = sdk_file(env.streamline_root.as_ref(), NGX_LIB) else {
        if targets.bundles() {
            out.push(warning(&format!(
                "{} - vendor it or set CN_STREAMLINE_SDK. The DLSS upscaler \
                 backend will be unavailable.",
                missing_sdk(
                    "NGX import lib",
                    "CN_STREAMLINE_SDK",
                    env.streamline_root.as_ref(),
                    NGX_LIB
                )
            )));
        }
        return;
    };

    // Pass the NGX static import lib straight to the linker for the final
    // artifact (a build-script `rustc-link-lib` does not reliably propagate, and
    // `rustc-link-arg` is scoped to the calling package's own targets, so each
    // package that links the DLSS code must emit this itself).
    out.push(format!("cargo::rustc-link-arg={}", ngx_lib.display()));
    out.push(rerun_path(&ngx_lib));
    out.push(rustc_cfg("ngx_sdk_bundled"));

    if targets.bundles() {
        const NGX_DLL: &[&str] = &["bin", "x64", "nvngx_dlss.dll"];
        let Some(dll_src) = sdk_file(env.streamline_root.as_ref(), NGX_DLL) else {
            out.push(warning(&format!(
                "{} - DLSS will fail to create its feature at runtime.",
                missing_sdk(
                    "NGX feature DLL",
                    "CN_STREAMLINE_SDK",
                    env.streamline_root.as_ref(),
                    NGX_DLL
                )
            )));
            return;
        };
        copy_next_to_exe(env, targets, &dll_src, "nvngx_dlss.dll", out);
    }
}

// DirectX Shader Compiler (`dxcompiler.dll` + `dxil.dll`) for the runtime DXC
// path that compiles the inline ray-tracing reflection shader. Copy-only, so
// only relevant when bundling for a final binary.
fn dxc_directives(env: &SdkEnv, targets: BinaryTargets, out: &mut Vec<String>) {
    out.push(rerun_env("CN_ENABLE_DXC"));
    if !env.dxc_enabled {
        out.push(warning(
            "DXC bundling skipped (CN_ENABLE_DXC=0); hardware \
             ray-traced reflections will be unavailable unless dxcompiler.dll + \
             dxil.dll are on PATH at runtime (the renderer falls back to SSR)",
        ));
        return;
    }
    out.push(rerun_env("CN_DXC_SDK"));

    let Some(dxc_dir) = find_dxc_dir(env) else {
        out.push(warning(
            "dxcompiler.dll + dxil.dll not found - set CN_DXC_SDK \
             to a directory containing them, or install the Windows SDK. Hardware \
             ray-traced reflections will be unavailable (the renderer falls back \
             to SSR).",
        ));
        return;
    };

    for dll in ["dxcompiler.dll", "dxil.dll"] {
        let src = dxc_dir.join(dll);
        if !copy_next_to_exe(env, targets, &src, dll, out) {
            return;
        }
    }
    out.push(rustc_cfg("dxc_bundled"));
}

// Copy `src` into the directory holding the package's binaries so
// `LoadLibrary` (which searches the .exe directory first) finds it. Skips the
// copy when the destination is already current, which avoids a redundant
// overwrite (and the Windows sharing violation it can raise while the DLL is
// loaded). Returns false on failure after recording a `cargo::warning`.
fn copy_next_to_exe(
    env: &SdkEnv,
    targets: BinaryTargets,
    src: &Path,
    file_name: &str,
    out: &mut Vec<String>,
) -> bool {
    let Some(exe_dir) = exe_dir(env, targets) else {
        return false;
    };
    // On a clean tree the build script can run before Cargo has laid out the
    // directory it will link the binaries into, so create it rather than
    // warning a DLL away.
    if let Err(e) = std::fs::create_dir_all(&exe_dir) {
        out.push(warning(&format!(
            "could not create {}: {e}",
            exe_dir.display()
        )));
        return false;
    }
    let dst = exe_dir.join(file_name);

    // Watch the source regardless of the copy below, so a newer SDK DLL
    // retriggers the build script even when this run is skipped as up to date.
    out.push(rerun_path(src));

    if up_to_date(src, &dst) {
        return true;
    }
    if let Err(e) = std::fs::copy(src, &dst) {
        out.push(warning(&format!(
            "could not copy {} -> {}: {e}",
            src.display(),
            dst.display()
        )));
        return false;
    }
    true
}

// A prior copy is current when the destination exists, matches the source
// size, and is no older than the source. `fs::copy` stamps the destination
// with a fresh mtime, so an unchanged source stays up to date until it is
// replaced by a newer SDK DLL (make-style staleness). Any metadata error is
// treated as stale so the copy is attempted.
fn up_to_date(src: &Path, dst: &Path) -> bool {
    let (Ok(s), Ok(d)) = (std::fs::metadata(src), std::fs::metadata(dst)) else {
        return false;
    };
    if s.len() != d.len() {
        return false;
    }
    matches!((s.modified(), d.modified()), (Ok(sm), Ok(dm)) if dm >= sm)
}

// The directory Cargo writes `targets` into, which is where the bundled DLLs
// have to land: `<target>/<profile>/` for bins, `<target>/<profile>/examples/`
// for examples.
fn exe_dir(env: &SdkEnv, targets: BinaryTargets) -> Option<PathBuf> {
    let dir = profile_dir(env)?;
    Some(match targets.exe_subdir() {
        Some(sub) => dir.join(sub),
        None => dir,
    })
}

// `<target>/<profile>/`, the root Cargo lays every build artifact out under.
fn profile_dir(env: &SdkEnv) -> Option<PathBuf> {
    profile_dir_from_out_dir(env.out_dir.as_deref()?).map(Path::to_path_buf)
}

// Walk up to `<target>/<profile>/` from an `OUT_DIR`
// (`<target>/<profile>/build/<pkg>-<hash>/out/`).
fn profile_dir_from_out_dir(out_dir: &Path) -> Option<&Path> {
    out_dir.ancestors().nth(3)
}

// Locate a directory holding both `dxcompiler.dll` and `dxil.dll`: the
// `CN_DXC_SDK` override, else the highest-versioned Windows SDK `x64` bin
// that carries both.
fn find_dxc_dir(env: &SdkEnv) -> Option<PathBuf> {
    let has_both = |d: &Path| d.join("dxcompiler.dll").exists() && d.join("dxil.dll").exists();

    if let Some(dir) = &env.dxc_root
        && has_both(dir)
    {
        return Some(dir.clone());
    }

    let versions = sorted_version_dirs(
        std::fs::read_dir(env.windows_sdk_bin.as_ref()?)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect(),
    );
    versions
        .into_iter()
        .rev()
        .map(|ver| ver.join("x64"))
        .find(|x64| has_both(x64))
}

// Sort version directories ascending so the newest is last. Lexicographic is
// adequate because every Windows SDK entry is `10.0.NNNNN.0` (equal width).
fn sorted_version_dirs(mut dirs: Vec<PathBuf>) -> Vec<PathBuf> {
    dirs.sort();
    dirs
}

fn warning(msg: &str) -> String {
    format!("cargo::warning={msg}")
}

fn rerun_env(var: &str) -> String {
    format!("cargo::rerun-if-env-changed={var}")
}

fn rerun_path(path: &Path) -> String {
    format!("cargo::rerun-if-changed={}", path.display())
}

fn rustc_cfg(cfg: &str) -> String {
    format!("cargo::rustc-cfg={cfg}")
}
