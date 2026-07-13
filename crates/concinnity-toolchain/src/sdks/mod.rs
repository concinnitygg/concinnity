// Graphics SDK probing, DLL bundling, and `cargo::` directive generation.
//
// Nothing in this module reads the process environment or prints to stdout:
// inputs arrive in `SdkEnv` (populated from the real build-script environment
// by the wrappers in lib.rs) and directives are returned as strings for the
// caller to print. File-system access is limited to the paths named in
// `SdkEnv`, so tests can drive the full probe/copy logic against fabricated
// SDK trees.

use std::path::{Path, PathBuf};

use crate::{Backend, SdkOptions};

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
    pub(crate) workspace_root: Option<PathBuf>,
    pub(crate) agility_root: PathBuf,
    pub(crate) fidelityfx_root: PathBuf,
    pub(crate) xess_root: PathBuf,
    pub(crate) streamline_root: PathBuf,
    pub(crate) dxc_root: Option<PathBuf>,
    pub(crate) windows_sdk_bin: PathBuf,
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

// The full SDK setup for one backend. On a non-Windows target (or the Metal
// backend) this returns nothing: none of these SDKs apply.
pub(crate) fn graphics_sdk_directives(
    backend: Backend,
    opts: SdkOptions,
    env: &SdkEnv,
) -> Vec<String> {
    let mut out = Vec::new();
    match backend {
        Backend::Dx => {
            agility_directives(env, opts.bundle_dlls, &mut out);
            fidelityfx_dx_directives(env, opts.bundle_dlls, &mut out);
            xess_directives(env, opts.bundle_dlls, &mut out);
            dlss_directives(env, opts.bundle_dlls, &mut out);
            if opts.bundle_dlls {
                dxc_directives(env, &mut out);
            }
        }
        Backend::Vk if env.target_os == "windows" => {
            // DLSS (NGX) and XeSS expose Vulkan entry points from the same
            // binaries the DirectX backend uses, so the setup helpers are
            // backend-agnostic and reused here. Windowing comes from the
            // shared native Win32 layer (no GLFW DLL to bundle).
            fidelityfx_vk_directives(env, opts.bundle_dlls, &mut out);
            dlss_directives(env, opts.bundle_dlls, &mut out);
            xess_directives(env, opts.bundle_dlls, &mut out);
        }
        _ => {}
    }
    out
}

// Microsoft's Agility SDK D3D12 runtime. The DLL copy and the binary's
// `D3D12SDKVersion`/`D3D12SDKPath` exports are only emitted when bundling for a
// final binary; the `agility_sdk_configured` cfg is always emitted when the SDK
// is present so the runtime FSR3 gate matches what the binary actually carries.
fn agility_directives(env: &SdkEnv, bundle_dlls: bool, out: &mut Vec<String>) {
    out.push(rerun_env("CN_ENABLE_AGILITY_SDK"));
    if !env.agility_enabled {
        if bundle_dlls {
            out.push(warning(
                "Agility SDK setup skipped (CN_ENABLE_AGILITY_SDK=0); \
                 binary will use the OS-bundled D3D12 runtime",
            ));
        }
        return;
    }
    out.push(rerun_env("AGILITY_SDK_ROOT"));

    let sdk_bin = env
        .agility_root
        .join("build")
        .join("native")
        .join("bin")
        .join("x64");
    let core_dll = sdk_bin.join("D3D12Core.dll");

    if !core_dll.exists() {
        if bundle_dlls {
            out.push(warning(&format!(
                "Agility SDK not found at {} - set AGILITY_SDK_ROOT \
                 or install the `microsoft.direct3d.d3d12` NuGet package. FidelityFX \
                 FSR3 will be unavailable (the binary falls back to the OS-bundled \
                 D3D12 runtime).",
                sdk_bin.display()
            )));
        }
        return;
    }

    if bundle_dlls {
        // `D3D12SDKPath = ".\\D3D12\\"` resolves relative to the .exe, so the
        // DLLs must live in `<target>/<profile>/D3D12/`.
        let Some(profile_dir) = profile_dir(env) else {
            return;
        };
        let d3d12_dir = profile_dir.join("D3D12");
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
        // would dereference as a pointer. The symbols themselves are defined as
        // `#[used]` statics in the binary crate's source.
        out.push("cargo::rustc-link-arg-bins=/EXPORT:D3D12SDKVersion,DATA".to_string());
        out.push("cargo::rustc-link-arg-bins=/EXPORT:D3D12SDKPath,DATA".to_string());
    }

    out.push(rustc_cfg("agility_sdk_configured"));
}

// AMD FidelityFX DX12 upscaler runtime. The renderer loads the DLL with
// `LoadLibrary` at runtime, so bundling only copies it next to the .exe; the
// `ffx_sdk_bundled` cfg is emitted when the SDK is present regardless.
fn fidelityfx_dx_directives(env: &SdkEnv, bundle_dlls: bool, out: &mut Vec<String>) {
    out.push(rerun_env("CN_ENABLE_FFX_FSR3"));
    if !env.ffx_enabled {
        if bundle_dlls {
            out.push(warning(
                "FidelityFX SDK bundling skipped (CN_ENABLE_FFX_FSR3=0); \
                 temporal upscaling will be unavailable unless amd_fidelityfx_dx12.dll \
                 is on PATH at runtime",
            ));
        }
        return;
    }
    out.push(rerun_env("FIDELITYFX_SDK_ROOT"));

    let dll_src = env
        .fidelityfx_root
        .join("bin")
        .join("amd_fidelityfx_dx12.dll");
    if !dll_src.exists() {
        if bundle_dlls {
            out.push(warning(&format!(
                "FidelityFX SDK not found at {} - set FIDELITYFX_SDK_ROOT \
                 or install the SDK. Temporal upscaling will be unavailable unless \
                 amd_fidelityfx_dx12.dll is on PATH at runtime.",
                dll_src.display()
            )));
        }
        return;
    }

    if bundle_dlls && !copy_next_to_exe(env, &dll_src, "amd_fidelityfx_dx12.dll", out) {
        return;
    }
    out.push(rustc_cfg("ffx_sdk_bundled"));
}

// AMD FidelityFX Vulkan upscaler runtime. Prefers the in-repo patched DLL under
// `crates/concinnity-engine/third_party/ffx/` (carries the FSR3 rw_luma_history
// format fix), falling back to the stock SDK copy.
fn fidelityfx_vk_directives(env: &SdkEnv, bundle_dlls: bool, out: &mut Vec<String>) {
    out.push(rerun_env("CN_ENABLE_FFX_FSR3"));
    if !env.ffx_enabled {
        if bundle_dlls {
            out.push(warning(
                "FidelityFX SDK bundling skipped (CN_ENABLE_FFX_FSR3=0); \
                 Vulkan temporal upscaling will be unavailable unless amd_fidelityfx_vk.dll \
                 is on PATH at runtime",
            ));
        }
        return;
    }
    out.push(rerun_env("FIDELITYFX_SDK_ROOT"));

    // The vendored DLL lives at a fixed location relative to the workspace
    // root, so this works no matter which package's build script called in.
    let vendored = env.workspace_root.as_ref().map(|root| {
        root.join("crates")
            .join("concinnity-engine")
            .join("third_party")
            .join("ffx")
            .join("amd_fidelityfx_vk.dll")
    });
    let sdk_dll = env
        .fidelityfx_root
        .join("bin")
        .join("amd_fidelityfx_vk.dll");

    let dll_src = match vendored {
        Some(v) if v.exists() => v,
        _ => sdk_dll,
    };
    if !dll_src.exists() {
        if bundle_dlls {
            out.push(warning(&format!(
                "FidelityFX VK runtime not found ({}). Set FIDELITYFX_SDK_ROOT, \
                 run scripts/setup_ffx_vk_dll.ps1, or put amd_fidelityfx_vk.dll on PATH at \
                 runtime; Vulkan temporal upscaling will fall back to native resolution.",
                dll_src.display()
            )));
        }
        return;
    }

    if bundle_dlls && !copy_next_to_exe(env, &dll_src, "amd_fidelityfx_vk.dll", out) {
        return;
    }
    out.push(rustc_cfg("ffx_sdk_bundled"));
}

// Intel XeSS upscaler runtime. Pure `LoadLibrary` at runtime, so bundling only
// copies the DLL; the cfg gates the copy and a log.
fn xess_directives(env: &SdkEnv, bundle_dlls: bool, out: &mut Vec<String>) {
    out.push(rerun_env("CN_ENABLE_XESS"));
    if !env.xess_enabled {
        if bundle_dlls {
            out.push(warning(
                "XeSS SDK bundling skipped (CN_ENABLE_XESS=0); the XeSS \
                 upscaler will be unavailable unless libxess.dll is on PATH at runtime",
            ));
        }
        return;
    }
    out.push(rerun_env("XESS_SDK_ROOT"));

    let dll_src = env.xess_root.join("bin").join("libxess.dll");
    if !dll_src.exists() {
        if bundle_dlls {
            out.push(warning(&format!(
                "XeSS SDK not found at {} - set XESS_SDK_ROOT or install \
                 the SDK. The XeSS upscaler backend will be unavailable unless \
                 libxess.dll is on PATH at runtime.",
                dll_src.display()
            )));
        }
        return;
    }

    if bundle_dlls && !copy_next_to_exe(env, &dll_src, "libxess.dll", out) {
        return;
    }
    out.push(rustc_cfg("xess_sdk_bundled"));
}

// DLSS via raw NGX. The import lib is always linked when present (the DLSS code
// compiled into the runtime rlib references its symbols, so every final binary
// and the rlib's own tests must resolve them). When bundling for a final binary
// the feature DLL `nvngx_dlss.dll` is also copied next to the .exe.
fn dlss_directives(env: &SdkEnv, bundle_dlls: bool, out: &mut Vec<String>) {
    out.push(rerun_env("CN_ENABLE_DLSS"));
    if !env.dlss_enabled {
        if bundle_dlls {
            out.push(warning(
                "DLSS (NGX) setup skipped (CN_ENABLE_DLSS=0); the DLSS \
                 upscaler backend will be unavailable",
            ));
        }
        return;
    }
    out.push(rerun_env("STREAMLINE_SDK_ROOT"));

    let ngx_lib = env
        .streamline_root
        .join("external")
        .join("ngx-sdk")
        .join("lib")
        .join("Windows_x86_64")
        .join("nvsdk_ngx_d.lib");
    if !ngx_lib.exists() {
        if bundle_dlls {
            out.push(warning(&format!(
                "NGX import lib not found at {} - set STREAMLINE_SDK_ROOT. \
                 The DLSS upscaler backend will be unavailable.",
                ngx_lib.display()
            )));
        }
        return;
    }

    // Pass the NGX static import lib straight to the linker for the final
    // artifact (a build-script `rustc-link-lib` does not reliably propagate, and
    // `rustc-link-arg` is scoped to the calling package's own targets, so each
    // package that links the DLSS code must emit this itself).
    out.push(format!("cargo::rustc-link-arg={}", ngx_lib.display()));
    out.push(rerun_path(&ngx_lib));
    out.push(rustc_cfg("ngx_sdk_bundled"));

    if bundle_dlls {
        let dll_src = env
            .streamline_root
            .join("bin")
            .join("x64")
            .join("nvngx_dlss.dll");
        if !dll_src.exists() {
            out.push(warning(&format!(
                "NGX feature DLL not found at {} - DLSS will fail to \
                 create its feature at runtime.",
                dll_src.display()
            )));
            return;
        }
        copy_next_to_exe(env, &dll_src, "nvngx_dlss.dll", out);
    }
}

// DirectX Shader Compiler (`dxcompiler.dll` + `dxil.dll`) for the runtime DXC
// path that compiles the inline ray-tracing reflection shader. Copy-only, so
// only relevant when bundling for a final binary.
fn dxc_directives(env: &SdkEnv, out: &mut Vec<String>) {
    out.push(rerun_env("CN_ENABLE_DXC"));
    if !env.dxc_enabled {
        out.push(warning(
            "DXC bundling skipped (CN_ENABLE_DXC=0); hardware \
             ray-traced reflections will be unavailable unless dxcompiler.dll + \
             dxil.dll are on PATH at runtime (the renderer falls back to SSR)",
        ));
        return;
    }
    out.push(rerun_env("DXC_SDK_ROOT"));

    let Some(dxc_dir) = find_dxc_dir(env) else {
        out.push(warning(
            "dxcompiler.dll + dxil.dll not found - set DXC_SDK_ROOT \
             to a directory containing them, or install the Windows SDK. Hardware \
             ray-traced reflections will be unavailable (the renderer falls back \
             to SSR).",
        ));
        return;
    };

    for dll in ["dxcompiler.dll", "dxil.dll"] {
        let src = dxc_dir.join(dll);
        if !copy_next_to_exe(env, &src, dll, out) {
            return;
        }
    }
    out.push(rustc_cfg("dxc_bundled"));
}

// Copy `src` to `<target>/<profile>/<file_name>` so `LoadLibrary` (which
// searches the .exe directory first) finds it. Returns false on failure after
// recording a `cargo::warning`.
fn copy_next_to_exe(env: &SdkEnv, src: &Path, file_name: &str, out: &mut Vec<String>) -> bool {
    let Some(profile_dir) = profile_dir(env) else {
        return false;
    };
    let dst = profile_dir.join(file_name);
    if let Err(e) = std::fs::copy(src, &dst) {
        out.push(warning(&format!(
            "could not copy {} -> {}: {e}",
            src.display(),
            dst.display()
        )));
        return false;
    }
    out.push(rerun_path(src));
    true
}

// `<target>/<profile>/`, where the final binaries and bundled DLLs land.
fn profile_dir(env: &SdkEnv) -> Option<PathBuf> {
    profile_dir_from_out_dir(env.out_dir.as_deref()?).map(Path::to_path_buf)
}

// Walk up to `<target>/<profile>/` from an `OUT_DIR`
// (`<target>/<profile>/build/<pkg>-<hash>/out/`).
fn profile_dir_from_out_dir(out_dir: &Path) -> Option<&Path> {
    out_dir.ancestors().nth(3)
}

// Locate a directory holding both `dxcompiler.dll` and `dxil.dll`: the
// `DXC_SDK_ROOT` override, else the highest-versioned Windows SDK `x64` bin
// that carries both.
fn find_dxc_dir(env: &SdkEnv) -> Option<PathBuf> {
    let has_both = |d: &Path| d.join("dxcompiler.dll").exists() && d.join("dxil.dll").exists();

    if let Some(dir) = &env.dxc_root
        && has_both(dir)
    {
        return Some(dir.clone());
    }

    let versions = sorted_version_dirs(
        std::fs::read_dir(&env.windows_sdk_bin)
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
