//! Shared build-script support for the workspace.
//!
//! Besides the Metal shader precompilation in `metal_shaders` and the source
//! hashing in `source_hash`, two responsibilities, both previously copy-pasted
//! between the runtime crate's build script and the editor crate's build script
//! (and missing entirely from the example binaries, which is why they failed to
//! link against the runtime's DLSS code on Windows):
//!
//! 1. Resolve the rendering backend once and emit it as a single cfg
//!    (`backend_metal` / `backend_dx` / `backend_vk`) the source gates on.
//!
//! 2. Detect the optional graphics SDKs and emit the cfgs the renderer gates on
//!    (`agility_sdk_configured`, `ffx_sdk_bundled`, `xess_sdk_bundled`,
//!    `ngx_sdk_bundled`, `dxc_bundled`). For a package that produces final
//!    binaries this also copies the runtime DLLs next to the .exe and links the
//!    NGX import lib; for a package that produces only an rlib and its own test
//!    binaries just the NGX link is needed. Which of the two, and where the
//!    binaries land, is `BinaryTargets`.
//!
//! The public entry points emit `cargo::` directives on stdout, which Cargo
//! attributes to the build script of whichever package called in. That is what
//! lets an example binary's build script pick up the same NGX link and DLL
//! bundling the CLI's does, without duplicating any of this logic.
//!
//! This file is the thin environment-reading layer: it snapshots everything the
//! setup needs from the process environment into an `SdkEnv` and prints the
//! directives. The probe/copy/directive logic itself lives in the `sdks`
//! module, which never touches the environment or stdout.

#[cfg(feature = "fetch")]
pub mod fetch;

use std::path::{Path, PathBuf};

mod metal_shaders;
mod sdks;
mod source_hash;

pub use metal_shaders::{SlangLibSpec, SlangShaders, precompile_metal_shaders};
use sdks::SdkEnv;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// The graphics backend a build targets.
pub enum Backend {
    /// Metal, on macOS.
    Metal,
    /// DirectX 12, on Windows.
    Dx,
    /// Vulkan, on Windows and Linux.
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

/// Which of the calling package's targets are the final binaries the graphics
/// SDKs serve. Cargo scopes a linker argument by target kind and places each kind
/// in its own directory, so this picks both the `cargo::rustc-link-arg-*` key the
/// Agility exports go out under and the directory the runtime DLLs are copied
/// into -- which has to be the one holding the .exe, since that is where Windows
/// looks for them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryTargets {
    /// The package links no binary of its own. Its test and bench executables
    /// still resolve the NGX symbols through the plain `rustc-link-arg`, but
    /// nothing is placed beside them.
    None,
    /// `src/main.rs` and `src/bin/`, which land in `<target>/<profile>/`.
    Bins,
    /// `examples/`, which land in `<target>/<profile>/examples/`.
    Examples,
}

impl BinaryTargets {
    pub(crate) fn bundles(self) -> bool {
        self != BinaryTargets::None
    }

    // The `cargo::rustc-link-arg-*` key covering these targets. Cargo rejects
    // `rustc-link-arg-bins` outright from a package with no bin target, and has
    // no per-example form at all, so an argument emitted for `Examples` reaches
    // every example the package builds.
    pub(crate) fn link_arg_key(self) -> Option<&'static str> {
        match self {
            BinaryTargets::None => None,
            BinaryTargets::Bins => Some("cargo::rustc-link-arg-bins"),
            BinaryTargets::Examples => Some("cargo::rustc-link-arg-examples"),
        }
    }

    // Subdirectory of `<target>/<profile>/` Cargo writes these binaries to.
    pub(crate) fn exe_subdir(self) -> Option<&'static str> {
        matches!(self, BinaryTargets::Examples).then_some("examples")
    }
}

// Resolve the backend from the target OS and whether the `vulkan` feature is on.
// macOS defaults to Metal and Windows to DirectX; both opt into Vulkan with the
// feature. Everything else (Linux) is Vulkan regardless. macOS Vulkan runs over
// MoltenVK and exists for cross-backend testing, not for shipping.
pub(crate) fn resolve_backend(target_os: &str, vulkan: bool) -> Backend {
    match (target_os, vulkan) {
        ("macos", false) => Backend::Metal,
        ("windows", false) => Backend::Dx,
        _ => Backend::Vk,
    }
}

/// Declare every cfg the renderer source gates on so `--check-cfg` does not warn.
/// A package only needs this if its own source references one of these cfgs.
pub fn emit_check_cfgs() {
    for line in sdks::check_cfg_directives() {
        println!("{line}");
    }
}

/// Resolve the backend from the Cargo-provided environment, emitting nothing.
/// For a package that needs the backend only to pick its SDK setup and never
/// gates its own source on one, so has no reason to carry the cfg.
pub fn backend_from_cargo() -> Backend {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let vulkan = std::env::var("CARGO_FEATURE_VULKAN").is_ok();
    resolve_backend(&target_os, vulkan)
}

/// Resolve the backend from the Cargo-provided environment and emit the
/// `rustc-cfg` for it, returning the choice so the caller can branch.
pub fn emit_backend_cfg() -> Backend {
    let backend = backend_from_cargo();
    println!("{}", sdks::backend_cfg_directive(backend));
    backend
}

/// Set up the optional graphics SDKs for the given backend. On a non-Windows
/// target (or the Metal backend) this is a no-op: none of these SDKs apply.
pub fn setup_graphics_sdks(backend: Backend, targets: BinaryTargets) {
    let env = sdk_env_from_cargo();
    for line in sdks::graphics_sdk_directives(backend, targets, &env) {
        println!("{line}");
    }
}

/// Hash the Rust sources under `roots`, and emit the rerun directives that
/// re-run the calling build script when any of them change. Each root is either
/// a directory tree (every `.rs` under it participates) or a single file.
///
/// The hash is what a content-addressed cache folds in so that a change to the
/// code producing its stored bytes evicts entries whose other inputs did not
/// move. See `source_hash` for the shape of the guarantee.
pub fn hash_sources(roots: &[PathBuf]) -> u32 {
    let workspace = workspace_root().expect("build script runs inside the workspace");
    let workspace = workspace.canonicalize().unwrap_or(workspace);
    let mut named = Vec::new();
    for root in roots {
        // Directory-level rerun directives catch added and removed files.
        println!("cargo:rerun-if-changed={}", root.display());
        let mut files = Vec::new();
        source_hash::collect(root, &mut files);
        named.extend(
            files
                .into_iter()
                .map(|file| (source_hash::relative_name(&workspace, &file), file)),
        );
    }
    source_hash::hash_named(&mut named)
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
        assert_eq!(resolve_backend("macos", true), Backend::Vk);
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
        for targets in [
            BinaryTargets::None,
            BinaryTargets::Bins,
            BinaryTargets::Examples,
        ] {
            setup_graphics_sdks(Backend::Metal, targets);
            setup_graphics_sdks(Backend::Vk, targets);
        }
        // The check-cfg list is emitted unconditionally and must not panic.
        emit_check_cfgs();
    }
}
