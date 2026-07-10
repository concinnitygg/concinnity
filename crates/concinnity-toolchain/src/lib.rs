// Shared build-script support for the workspace.
//
// Two responsibilities, both previously copy-pasted between the runtime crate's
// build script and the editor crate's build script (and missing entirely from
// the example binaries, which is why they failed to link against the runtime's
// DLSS code on Windows):
//
// 1. Resolve the rendering backend once and emit it as a single cfg
//    (`backend_metal` / `backend_dx` / `backend_vk`) the source gates on.
//
// 2. Detect the optional graphics SDKs and emit the cfgs the renderer gates on
//    (`agility_sdk_configured`, `ffx_sdk_bundled`, `xess_sdk_bundled`,
//    `ngx_sdk_bundled`, `dxc_bundled`). For a package that produces a final
//    binary (the editor, an example) this also copies the runtime DLLs next to
//    the .exe and links the NGX import lib; for the runtime rlib's own test
//    binaries only the NGX link is needed (no DLL copy), selected with
//    `SdkOptions { bundle_dlls: false }`.
//
// The public entry points emit `cargo::` directives on stdout, which Cargo
// attributes to the build script of whichever package called in. That is what
// lets an example binary's build script pick up the same NGX link and DLL
// bundling the editor's does, without duplicating any of this logic.
//
// This file is the thin environment-reading layer: it snapshots everything the
// setup needs from the process environment into an `SdkEnv` and prints the
// directives. The probe/copy/directive logic itself lives in the `sdks`
// module, which never touches the environment or stdout.

use std::path::{Path, PathBuf};

mod sdks;

use sdks::SdkEnv;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Metal,
    Dx,
    Vk,
}

impl Backend {
    pub(crate) fn cfg_name(self) -> &'static str {
        match self {
            Backend::Metal => "backend_metal",
            Backend::Dx => "backend_dx",
            Backend::Vk => "backend_vk",
        }
    }
}

// Options for the SDK setup. `bundle_dlls` distinguishes a package that produces
// a final binary (true: copy runtime DLLs next to the .exe, emit the Agility
// linker exports) from the runtime rlib's own test binaries (false: link the
// NGX import lib and emit the gating cfgs, but place no DLLs).
#[derive(Clone, Copy, Debug)]
pub struct SdkOptions {
    pub bundle_dlls: bool,
}

// Resolve the backend from the target OS and whether the `vulkan` feature is on.
// macOS is always Metal; Windows defaults to DirectX and opts into Vulkan with
// the feature; everything else (Linux) is Vulkan.
pub fn resolve_backend(target_os: &str, vulkan: bool) -> Backend {
    match (target_os, vulkan) {
        ("macos", _) => Backend::Metal,
        ("windows", false) => Backend::Dx,
        _ => Backend::Vk,
    }
}

// Declare every cfg the renderer source gates on so `--check-cfg` does not warn.
// A package only needs this if its own source references one of these cfgs.
pub fn emit_check_cfgs() {
    for line in sdks::check_cfg_directives() {
        println!("{line}");
    }
}

// Resolve the backend from the Cargo-provided environment and emit the
// `rustc-cfg` for it, returning the choice so the caller can branch.
pub fn emit_backend_cfg() -> Backend {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let vulkan = std::env::var("CARGO_FEATURE_VULKAN").is_ok();
    let backend = resolve_backend(&target_os, vulkan);
    println!("{}", sdks::backend_cfg_directive(backend));
    backend
}

// Set up the optional graphics SDKs for the given backend. On a non-Windows
// target (or the Metal backend) this is a no-op: none of these SDKs apply.
pub fn setup_graphics_sdks(backend: Backend, opts: SdkOptions) {
    let env = sdk_env_from_cargo();
    for line in sdks::graphics_sdk_directives(backend, opts, &env) {
        println!("{line}");
    }
}

// Default SDK install roots, used when the matching env var is unset.
const DEFAULT_AGILITY_SDK_ROOT: &str = "C:\\microsoft.direct3d.d3d12.1.619.3";
const DEFAULT_FIDELITYFX_SDK_ROOT: &str = "C:\\FidelityFX-SDK-v1.1.4";
const DEFAULT_XESS_SDK_ROOT: &str = "C:\\XeSS_SDK_3.0.1";
const DEFAULT_STREAMLINE_SDK_ROOT: &str = "C:\\streamline-sdk-v2.11.1";
const DEFAULT_WINDOWS_SDK_BIN: &str = "C:\\Program Files (x86)\\Windows Kits\\10\\bin";

// Snapshot every environment input the SDK setup reads. This is the only place
// the setup consults the process environment; everything downstream works on
// the returned struct.
fn sdk_env_from_cargo() -> SdkEnv {
    let var = |name: &str| std::env::var(name).ok();
    let root = |name: &str, default: &str| {
        var(name)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(default))
    };
    // Each SDK probe defaults to ON and is opted out of with `<VAR>=0`.
    let enabled = |name: &str| var(name).as_deref() != Some("0");
    SdkEnv {
        target_os: var("CARGO_CFG_TARGET_OS").unwrap_or_default(),
        out_dir: var("OUT_DIR").map(PathBuf::from),
        workspace_root: workspace_root(),
        agility_root: root("AGILITY_SDK_ROOT", DEFAULT_AGILITY_SDK_ROOT),
        fidelityfx_root: root("FIDELITYFX_SDK_ROOT", DEFAULT_FIDELITYFX_SDK_ROOT),
        xess_root: root("XESS_SDK_ROOT", DEFAULT_XESS_SDK_ROOT),
        streamline_root: root("STREAMLINE_SDK_ROOT", DEFAULT_STREAMLINE_SDK_ROOT),
        dxc_root: var("DXC_SDK_ROOT").map(PathBuf::from),
        windows_sdk_bin: PathBuf::from(DEFAULT_WINDOWS_SDK_BIN),
        agility_enabled: enabled("CN_ENABLE_AGILITY_SDK"),
        ffx_enabled: enabled("CN_ENABLE_FFX_FSR3"),
        xess_enabled: enabled("CN_ENABLE_XESS"),
        dlss_enabled: enabled("CN_ENABLE_DLSS"),
        dxc_enabled: enabled("CN_ENABLE_DXC"),
    }
}

// Locate the workspace root by walking up from the caller's manifest until a
// `Cargo.toml` declaring `[workspace]` is found.
fn workspace_root() -> Option<PathBuf> {
    let start = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    find_ancestor_with(Path::new(&start), |dir| {
        std::fs::read_to_string(dir.join("Cargo.toml"))
            .map(|c| c.contains("[workspace]"))
            .unwrap_or(false)
    })
}

// Walk `start` and its ancestors, returning the first that satisfies `pred`.
fn find_ancestor_with(start: &Path, pred: impl Fn(&Path) -> bool) -> Option<PathBuf> {
    let mut dir = start.to_path_buf();
    loop {
        if pred(&dir) {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_resolution_covers_every_target() {
        assert_eq!(resolve_backend("macos", false), Backend::Metal);
        assert_eq!(resolve_backend("macos", true), Backend::Metal);
        assert_eq!(resolve_backend("windows", false), Backend::Dx);
        assert_eq!(resolve_backend("windows", true), Backend::Vk);
        assert_eq!(resolve_backend("linux", false), Backend::Vk);
        assert_eq!(resolve_backend("linux", true), Backend::Vk);
    }

    #[test]
    fn backend_cfg_names_are_stable() {
        assert_eq!(Backend::Metal.cfg_name(), "backend_metal");
        assert_eq!(Backend::Dx.cfg_name(), "backend_dx");
        assert_eq!(Backend::Vk.cfg_name(), "backend_vk");
    }

    #[test]
    fn ancestor_search_finds_marked_dir() {
        let start = Path::new("/a/b/c/d");
        let hit = find_ancestor_with(start, |p| p == Path::new("/a/b"));
        assert_eq!(hit, Some(PathBuf::from("/a/b")));

        let miss = find_ancestor_with(start, |p| p == Path::new("/x"));
        assert_eq!(miss, None);
    }

    #[test]
    fn workspace_root_finds_the_workspace_manifest() {
        // Cargo sets CARGO_MANIFEST_DIR for test binaries, so the walk starts
        // at this crate and must land on the workspace's own Cargo.toml.
        let root = workspace_root().expect("workspace root");
        let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("read manifest");
        assert!(manifest.contains("[workspace]"));
    }

    #[test]
    fn sdk_env_snapshot_defaults_probes_on() {
        let env = sdk_env_from_cargo();
        // Probes default ON when their opt-out variable is unset. Only assert
        // for variables the surrounding environment leaves unset, so a local
        // `<VAR>=0` opt-out does not fail the test.
        for (var, flag) in [
            ("CN_ENABLE_AGILITY_SDK", env.agility_enabled),
            ("CN_ENABLE_FFX_FSR3", env.ffx_enabled),
            ("CN_ENABLE_XESS", env.xess_enabled),
            ("CN_ENABLE_DLSS", env.dlss_enabled),
            ("CN_ENABLE_DXC", env.dxc_enabled),
        ] {
            if std::env::var(var).is_err() {
                assert!(flag, "{var} should default on");
            }
        }
        // Roots fall back to the hardcoded defaults when unset.
        if std::env::var("XESS_SDK_ROOT").is_err() {
            assert_eq!(env.xess_root, PathBuf::from(DEFAULT_XESS_SDK_ROOT));
        }
    }

    #[test]
    fn graphics_sdk_setup_is_a_noop_off_windows_targets() {
        // Metal never has SDKs to set up, and the Vulkan arm is gated on a
        // Windows target OS (CARGO_CFG_TARGET_OS is unset outside build
        // scripts), so neither requires any SDK to be present.
        for bundle_dlls in [false, true] {
            setup_graphics_sdks(Backend::Metal, SdkOptions { bundle_dlls });
            setup_graphics_sdks(Backend::Vk, SdkOptions { bundle_dlls });
        }
        // The check-cfg list is emitted unconditionally and must not panic.
        emit_check_cfgs();
    }
}
