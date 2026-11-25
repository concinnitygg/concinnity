//! Shared build-script support for the workspace.
//!
//! Besides the source hashing in `source_hash` and the version stamp, one
//! responsibility every build script that compiles the renderer needs: resolve
//! the rendering backends the build compiles in and emit one cfg per backend
//! (`backend_metal` / `backend_dx` / `backend_vk`) the source gates on, behind
//! one call, [`setup_graphics_backends`].
//!
//! Nothing here compiles a shader or reads a graphics SDK: the engine's shaders
//! compile when an app first runs (or when an export ships them), and the
//! upscalers and the Agility runtime are loaded when an app runs, so a build
//! needs neither a shader compiler nor any SDK.

use std::path::PathBuf;

mod source_hash;
mod version_stamp;

pub use version_stamp::emit_version_stamp;

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
    /// The shader platform this backend consumes.
    pub fn platform(self) -> concinnity_core::platform::Platform {
        use concinnity_core::platform::Platform;
        match self {
            Self::Metal => Platform::Metal,
            Self::Dx => Platform::DirectX,
            Self::Vk => Platform::Vulkan,
        }
    }

    pub(crate) fn cfg_name(self) -> &'static str {
        match self {
            Self::Metal => "backend_metal",
            Self::Dx => "backend_dx",
            Self::Vk => "backend_vk",
        }
    }
}

// Whether the calling package's build script runs with its `vulkan` feature on.
#[expect(
    clippy::disallowed_methods,
    reason = "cargo passes build-script inputs through the environment"
)]
fn vulkan_from_cargo() -> bool {
    std::env::var("CARGO_FEATURE_VULKAN").is_ok()
}

// Resolve the backends from the target OS: the one each target renders with,
// native first, plus Vulkan beside it when asked for. Linux renders with Vulkan
// alone. iOS and Android have no window layer yet, so they build no backend and
// run headless. macOS Vulkan runs over MoltenVK and exists for cross-backend
// testing, not for shipping.
pub(crate) fn resolve_backends(target_os: &str, vulkan: bool) -> Vec<Backend> {
    let native = match target_os {
        "ios" | "android" => return Vec::new(),
        "macos" => Some(Backend::Metal),
        "windows" => Some(Backend::Dx),
        _ => None,
    };
    let vulkan = (vulkan || native.is_none()).then_some(Backend::Vk);
    native.into_iter().chain(vulkan).collect()
}

// Declare every backend cfg the renderer source gates on, so `--check-cfg` does
// not warn about the ones a build leaves unset.
fn emit_check_cfgs() {
    for backend in [Backend::Metal, Backend::Dx, Backend::Vk] {
        println!("cargo::rustc-check-cfg=cfg({})", backend.cfg_name());
    }
}

// The whole setup: declare the cfgs, then emit one for each backend.
#[expect(
    clippy::disallowed_methods,
    reason = "cargo passes build-script inputs through the environment"
)]
fn setup_for(vulkan: bool) -> Vec<Backend> {
    emit_check_cfgs();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let backends = resolve_backends(&target_os, vulkan);
    for &backend in &backends {
        println!("cargo::rustc-cfg={}", backend.cfg_name());
    }
    backends
}

/// The whole build-script graphics setup for a package that compiles the
/// renderer: declares the cfgs its source gates on, resolves the backends from
/// the calling package's own features and emits a cfg for each
/// (`backend_metal` / `backend_dx` / `backend_vk`). The target's native backend
/// is always built; the `vulkan` feature adds Vulkan beside it. Returns the
/// backends, native first, so a build script with steps of its own can branch.
///
/// A package that took `concinnity`'s `vulkan` feature carries a `vulkan`
/// feature of its own, since a build script only sees the features of the
/// package it belongs to.
///
/// An empty list is a target with no window layer yet (iOS, Android), which
/// builds no GPU backend. That emits no backend cfg, and the check-cfg list
/// still declares all three, so the source gating on them compiles.
///
/// ```no_run
/// concinnity_toolchain::setup_graphics_backends();
/// ```
pub fn setup_graphics_backends() -> Vec<Backend> {
    setup_for(vulkan_from_cargo())
}

/// Hash the Rust sources under `roots`, and emit the rerun directives that
/// re-run the calling build script when any of them change. Each root is either
/// a directory tree (every `.rs` under it participates) or a single file.
///
/// The hash is what a content-addressed cache folds in so that a change to the
/// code producing its stored bytes evicts entries whose other inputs did not
/// move. See `source_hash` for the shape of the guarantee.
///
/// # Panics
///
/// Panics outside a build script, where cargo sets no `CARGO_MANIFEST_DIR`.
#[expect(
    clippy::disallowed_methods,
    reason = "cargo passes build-script inputs through the environment"
)]
pub fn hash_sources(roots: &[PathBuf]) -> u32 {
    let package =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("build script runs under cargo"));
    let package = package.canonicalize().unwrap_or(package);
    let mut named = Vec::new();
    for root in roots {
        // Directory-level rerun directives catch added and removed files.
        println!("cargo:rerun-if-changed={}", root.display());
        let mut files = Vec::new();
        source_hash::collect(root, &mut files);
        named.extend(
            files
                .into_iter()
                .map(|file| (source_hash::relative_name(&package, &file), file)),
        );
    }
    source_hash::hash_named(&mut named)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_desktop_target_resolves_to_what_it_renders_with() {
        assert_eq!(resolve_backends("macos", false), [Backend::Metal]);
        assert_eq!(resolve_backends("windows", false), [Backend::Dx]);
        assert_eq!(resolve_backends("linux", false), [Backend::Vk]);
    }

    #[test]
    fn vulkan_joins_the_native_backend_after_it() {
        assert_eq!(
            resolve_backends("macos", true),
            [Backend::Metal, Backend::Vk]
        );
        assert_eq!(
            resolve_backends("windows", true),
            [Backend::Dx, Backend::Vk]
        );
        assert_eq!(resolve_backends("linux", true), [Backend::Vk]);
    }

    #[test]
    fn targets_without_a_window_layer_build_no_backend() {
        for target_os in ["ios", "android"] {
            assert!(resolve_backends(target_os, false).is_empty());
            assert!(resolve_backends(target_os, true).is_empty());
        }
    }

    #[test]
    fn each_backend_consumes_its_own_platform() {
        use concinnity_core::platform::Platform;
        assert_eq!(Backend::Metal.platform(), Platform::Metal);
        assert_eq!(Backend::Dx.platform(), Platform::DirectX);
        assert_eq!(Backend::Vk.platform(), Platform::Vulkan);
    }

    #[test]
    fn backend_cfg_names_are_stable() {
        assert_eq!(Backend::Metal.cfg_name(), "backend_metal");
        assert_eq!(Backend::Dx.cfg_name(), "backend_dx");
        assert_eq!(Backend::Vk.cfg_name(), "backend_vk");
    }

    #[expect(
        clippy::disallowed_methods,
        reason = "the expected backends follow the target cargo builds for"
    )]
    #[test]
    fn the_setup_returns_the_backends_it_resolved() {
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        for vulkan in [false, true] {
            assert_eq!(
                setup_for(vulkan),
                resolve_backends(&target_os, vulkan),
                "vulkan: {vulkan}"
            );
        }
    }
}
