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
//!    binaries land, is read off the calling package by `targets` -- a build
//!    script declares nothing about its own target list.
//!
//! The public entry points emit `cargo::` directives on stdout, which Cargo
//! attributes to the build script of whichever package called in. That is what
//! lets an example binary's build script pick up the same NGX link and DLL
//! bundling the CLI's does, without duplicating any of this logic. It is also
//! what a package outside this workspace needs, since the NGX link reaches only
//! the package that emits it: `setup_graphics_sdks_for_consumer` is the whole
//! setup behind one call for a build script that has no backend features of its
//! own to read.
//!
//! This file is the thin environment-reading layer: it snapshots everything the
//! setup needs from the process environment into an `SdkEnv` and prints the
//! directives. The probe/copy/directive logic itself lives in the `sdks`
//! module, which never touches the environment or stdout.

use std::path::{Path, PathBuf};

mod metal_shaders;
mod sdks;
mod slang_artifacts;
mod source_hash;
mod targets;
mod vendored;
mod version_stamp;

pub use metal_shaders::{SlangLibSpec, precompile_metal_shaders};
use sdks::SdkEnv;
pub use slang_artifacts::{SlangArtifact, precompile_slang_artifacts};
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
    pub(crate) fn cfg_name(self) -> &'static str {
        match self {
            Backend::Metal => "backend_metal",
            Backend::Dx => "backend_dx",
            Backend::Vk => "backend_vk",
        }
    }
}

// Which of the calling package's targets are the final binaries the graphics
// SDKs serve. Cargo scopes a linker argument by target kind and places each kind
// in its own directory, so this picks both the `cargo::rustc-link-arg-*` key the
// Agility exports go out under and the directory the runtime DLLs are copied
// into -- which has to be the one holding the .exe, since that is where Windows
// looks for them. Discovered per package by `targets`, never named by a caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BinaryTargets {
    // The package links no binary of its own. Its test and bench executables
    // still resolve the NGX symbols through the plain `rustc-link-arg`, but
    // nothing is placed beside them.
    None,
    // `src/main.rs` and `src/bin/`, which land in `<target>/<profile>/`.
    Bins,
    // `examples/`, which land in `<target>/<profile>/examples/`.
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

/// The backend features a package was built with. `native` names all three and
/// resolves to the one the target renders with; the other three name a backend
/// outright.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackendFeatures {
    /// Whatever backend the target renders with.
    pub native: bool,
    /// Metal, which only macOS has.
    pub metal: bool,
    /// DirectX 12, which only Windows has.
    pub directx: bool,
    /// Vulkan, which Windows and Linux have, and macOS has over MoltenVK.
    pub vulkan: bool,
}

impl BackendFeatures {
    /// Read the four features from the Cargo-provided environment of whichever
    /// package's build script is running.
    pub fn from_cargo() -> Self {
        Self {
            native: feature_on("NATIVE"),
            metal: feature_on("METAL"),
            directx: feature_on("DIRECTX"),
            vulkan: feature_on("VULKAN"),
        }
    }
}

fn feature_on(name: &str) -> bool {
    std::env::var(format!("CARGO_FEATURE_{name}")).is_ok()
}

// Resolve the backend from the target OS and the backend features. A feature
// naming a backend the target does not have is inert, so `--features directx`
// on macOS builds no backend rather than failing; that keeps the features
// additive, which is what lets `native` name all three at once. Vulkan wins
// where both apply, so `--all-features` resolves rather than conflicts. macOS
// Vulkan runs over MoltenVK and exists for cross-backend testing, not for
// shipping.
pub(crate) fn resolve_backend(target_os: &str, features: BackendFeatures) -> Option<Backend> {
    if features.vulkan {
        return Some(Backend::Vk);
    }
    match target_os {
        "macos" => (features.metal || features.native).then_some(Backend::Metal),
        "windows" => (features.directx || features.native).then_some(Backend::Dx),
        _ => features.native.then_some(Backend::Vk),
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
/// gates its own source on one, so has no reason to carry the cfg. `None` is a
/// build with no backend: a CPU-only runtime with no GPU code in it.
pub fn backend_from_cargo() -> Option<Backend> {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    resolve_backend(&target_os, BackendFeatures::from_cargo())
}

/// Resolve the backend from the Cargo-provided environment and emit the
/// `rustc-cfg` for it, returning the choice so the caller can branch. A build
/// with no backend emits no cfg; `emit_check_cfgs` still declares all three, so
/// the source gating on them compiles.
pub fn emit_backend_cfg() -> Option<Backend> {
    let backend = backend_from_cargo();
    if let Some(backend) = backend {
        println!("{}", sdks::backend_cfg_directive(backend));
    }
    backend
}

/// Set up the optional graphics SDKs for the given backend. On a non-Windows
/// target (or the Metal backend) this is a no-op: none of these SDKs apply.
///
/// Which kinds of final binary the calling package builds is read from that
/// package, not passed in: a package can build both bins and examples, and each
/// kind takes its own linker-argument key and its own directory for the bundled
/// DLLs, so the setup runs once per kind. A directive both kinds produce --
/// every cfg, every warning -- is emitted once.
pub fn setup_graphics_sdks(backend: Option<Backend>) {
    let env = sdk_env_from_cargo();
    if let Some(dir) = manifest_dir() {
        for path in targets::watched_inputs(&dir) {
            println!("cargo::rerun-if-changed={}", path.display());
        }
    }
    let Some(backend) = backend else {
        return;
    };
    for line in sdks::graphics_sdk_directives(backend, &binary_targets_from_cargo(), &env) {
        println!("{line}");
    }
}

/// The whole build-script setup for a package outside this workspace that links
/// the engine: declares the cfgs, emits the backend cfg, and runs
/// [`setup_graphics_sdks`], returning the backend it resolved.
///
/// A package depending on `concinnity` needs this because the NGX link
/// directive is scoped to the package that emits it. When the runtime is built
/// with DLSS available its upscaler compiles into the rlib, and the binary
/// linking that rlib has to resolve the NGX symbols itself; without this its
/// link fails on `NVSDK_NGX_*`. It is also what puts the SDK runtime DLLs
/// beside that binary, where `LoadLibrary` finds them.
///
/// One call from the consuming package's `build.rs` is the whole setup:
///
/// ```no_run
/// concinnity_toolchain::setup_graphics_sdks_for_consumer();
/// ```
///
/// The backend is resolved as if `native` were on, which is what `concinnity`'s
/// own defaults give for the target. A build script cannot see the features its
/// dependencies were built with -- `CARGO_FEATURE_*` names the calling
/// package's own -- so a package that took `concinnity`'s `vulkan` feature says
/// so by carrying a `vulkan` feature of its own; that is the one backend which
/// differs from `native` on a target that has it.
///
/// A missing SDK is reported as a `cargo::warning` naming the root it was
/// looked for under, and costs that upscaler rather than the build.
pub fn setup_graphics_sdks_for_consumer() -> Option<Backend> {
    emit_check_cfgs();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let backend = resolve_backend(&target_os, consumer_features(BackendFeatures::from_cargo()));
    if let Some(backend) = backend {
        println!("{}", sdks::backend_cfg_directive(backend));
    }
    setup_graphics_sdks(backend);
    backend
}

// What a consuming package's backend features mean: `native` on top of whatever
// it named itself, since a package that names no backend gets `concinnity`'s
// own default. `native` covers every target, so the backend a name adds over it
// is Vulkan -- the one a target can have besides the one it renders with.
pub(crate) fn consumer_features(named: BackendFeatures) -> BackendFeatures {
    BackendFeatures {
        native: true,
        ..named
    }
}

// The calling package's binary kinds, from the manifest Cargo is building it
// from. A package Cargo hands no manifest (nothing does outside a build script)
// falls back to the kind that scopes nothing.
fn binary_targets_from_cargo() -> Vec<BinaryTargets> {
    let Some(dir) = manifest_dir() else {
        return vec![BinaryTargets::None];
    };
    let manifest = std::fs::read_to_string(dir.join("Cargo.toml")).unwrap_or_default();
    targets::binary_targets(&manifest, &dir)
}

fn manifest_dir() -> Option<PathBuf> {
    std::env::var("CARGO_MANIFEST_DIR").ok().map(PathBuf::from)
}

/// Hash the Rust sources under `roots`, and emit the rerun directives that
/// re-run the calling build script when any of them change. Each root is either
/// a directory tree (every `.rs` under it participates) or a single file.
///
/// The hash is what a content-addressed cache folds in so that a change to the
/// code producing its stored bytes evicts entries whose other inputs did not
/// move. See `source_hash` for the shape of the guarantee.
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

// Default SDK install roots, used when the matching env var is unset.
// The Windows SDK's tool directory, under whichever Program Files the OS
// reports. Windows always sets the variable, and off Windows there is no
// Windows SDK to find, so an absent one is not a fallback but an answer.
fn windows_sdk_bin() -> Option<PathBuf> {
    let program_files = std::env::var("ProgramFiles(x86)").ok()?;
    Some(
        Path::new(&program_files)
            .join("Windows Kits")
            .join("10")
            .join("bin"),
    )
}

// Whether an opt-in variable's value asks for the feature, for the one SDK that
// is off by default. Bundling Agility links `D3D12SDKVersion` / `D3D12SDKPath`
// into the binary, and `d3d12.dll` reads those before any engine code runs: a
// binary carrying them starts only where the staged `D3D12/` directory sits
// beside it, so an executable copied anywhere else -- every `cargo install` --
// reaches no adapter at all. That makes bundling a decision about how the
// artifact is distributed rather than about which SDKs the build machine
// happens to have, so it has to be asked for. `0` keeps meaning off, as it
// always has for these variables.
fn opted_in(value: Option<&str>) -> bool {
    matches!(value, Some("1" | "true" | "TRUE"))
}

// Snapshot every environment input the SDK setup reads. This is the only place
// the setup consults the process environment; everything downstream works on
// the returned struct.
fn sdk_env_from_cargo() -> SdkEnv {
    let var = |name: &str| std::env::var(name).ok();
    let workspace = workspace_root();
    let root = |name: &str, component: &str| {
        sdk_root(var(name), vendored::newest(workspace.as_deref(), component))
    };
    // The `LoadLibrary`-at-runtime SDKs default to ON and are opted out of with
    // `<VAR>=0`: an absent DLL costs the feature and nothing else.
    let enabled = |name: &str| var(name).as_deref() != Some("0");
    // Agility is the exception and defaults to OFF; see `opted_in`.
    let opted_in = |name: &str| self::opted_in(var(name).as_deref());
    SdkEnv {
        target_os: var("CARGO_CFG_TARGET_OS").unwrap_or_default(),
        out_dir: var("OUT_DIR").map(PathBuf::from),
        agility_root: root("CN_AGILITY_SDK", "agility"),
        fidelityfx_root: root("CN_FIDELITYFX_SDK", "fidelityfx"),
        // No variable of its own: the patched runtime is rebuilt into
        // `vendor/` and exists nowhere else, so a root to point at would name a
        // directory only that rebuild produces. An explicit `CN_FIDELITYFX_SDK`
        // still supplies the stock DLL it falls back to.
        fidelityfx_vk_root: vendored::newest(workspace.as_deref(), "fidelityfx-vk"),
        xess_root: root("CN_XESS_SDK", "xess"),
        streamline_root: root("CN_STREAMLINE_SDK", "streamline"),
        dxc_root: var("CN_DXC_SDK").map(PathBuf::from),
        windows_sdk_bin: windows_sdk_bin(),
        agility_enabled: opted_in("CN_ENABLE_AGILITY_SDK"),
        ffx_enabled: enabled("CN_ENABLE_FFX_FSR3"),
        xess_enabled: enabled("CN_ENABLE_XESS"),
        dlss_enabled: enabled("CN_ENABLE_DLSS"),
        dxc_enabled: enabled("CN_ENABLE_DXC"),
    }
}

// Where an SDK is: the explicit variable, else whatever `vendor/` holds. There
// is no third guess -- each of these unpacks wherever its user put it, so a
// hardcoded install path was only ever right by accident, and being wrong it
// reported a location nobody had rather than that nothing was found. The
// variable leads because an explicit answer beats a discovered one, including
// when it is wrong: a mistyped path fails instead of silently resolving
// elsewhere.
fn sdk_root(explicit: Option<String>, vendored: Option<PathBuf>) -> Option<PathBuf> {
    explicit.map(PathBuf::from).or(vendored)
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

    fn features(names: &[&str]) -> BackendFeatures {
        BackendFeatures {
            native: names.contains(&"native"),
            metal: names.contains(&"metal"),
            directx: names.contains(&"directx"),
            vulkan: names.contains(&"vulkan"),
        }
    }

    #[test]
    fn native_resolves_to_what_each_target_renders_with() {
        assert_eq!(
            resolve_backend("macos", features(&["native"])),
            Some(Backend::Metal)
        );
        assert_eq!(
            resolve_backend("windows", features(&["native"])),
            Some(Backend::Dx)
        );
        assert_eq!(
            resolve_backend("linux", features(&["native"])),
            Some(Backend::Vk)
        );
    }

    #[test]
    fn a_named_backend_selects_it_where_the_target_has_it() {
        assert_eq!(
            resolve_backend("macos", features(&["metal"])),
            Some(Backend::Metal)
        );
        assert_eq!(
            resolve_backend("windows", features(&["directx"])),
            Some(Backend::Dx)
        );
        assert_eq!(
            resolve_backend("windows", features(&["vulkan"])),
            Some(Backend::Vk)
        );
        assert_eq!(
            resolve_backend("macos", features(&["vulkan"])),
            Some(Backend::Vk)
        );
        assert_eq!(
            resolve_backend("linux", features(&["vulkan"])),
            Some(Backend::Vk)
        );
    }

    #[test]
    fn a_backend_the_target_does_not_have_is_inert() {
        assert_eq!(resolve_backend("macos", features(&["directx"])), None);
        assert_eq!(resolve_backend("windows", features(&["metal"])), None);
        assert_eq!(resolve_backend("linux", features(&["metal"])), None);
        assert_eq!(resolve_backend("linux", features(&["directx"])), None);
    }

    #[test]
    fn no_backend_feature_resolves_to_no_backend() {
        for target_os in ["macos", "windows", "linux"] {
            assert_eq!(resolve_backend(target_os, features(&[])), None);
        }
    }

    #[test]
    fn vulkan_wins_where_more_than_one_backend_applies() {
        let all = features(&["native", "metal", "directx", "vulkan"]);
        for target_os in ["macos", "windows", "linux"] {
            assert_eq!(resolve_backend(target_os, all), Some(Backend::Vk));
        }
    }

    #[test]
    fn a_consumer_naming_no_backend_gets_the_one_its_target_renders_with() {
        for (target_os, backend) in [
            ("macos", Backend::Metal),
            ("windows", Backend::Dx),
            ("linux", Backend::Vk),
        ] {
            assert_eq!(
                resolve_backend(target_os, consumer_features(features(&[]))),
                Some(backend)
            );
        }
    }

    #[test]
    fn a_consumer_naming_vulkan_gets_vulkan() {
        for target_os in ["macos", "windows", "linux"] {
            assert_eq!(
                resolve_backend(target_os, consumer_features(features(&["vulkan"]))),
                Some(Backend::Vk)
            );
        }
    }

    #[test]
    fn a_consumer_naming_the_backend_its_target_already_renders_with_changes_nothing() {
        for (target_os, named) in [("macos", "metal"), ("windows", "directx")] {
            assert_eq!(
                resolve_backend(target_os, consumer_features(features(&[named]))),
                resolve_backend(target_os, consumer_features(features(&[])))
            );
        }
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
    fn sdk_env_snapshot_defaults_the_loadlibrary_probes_on() {
        let env = sdk_env_from_cargo();
        // The `LoadLibrary` probes default ON when their opt-out variable is
        // unset. Only assert for variables the surrounding environment leaves
        // unset, so a local `<VAR>=0` opt-out does not fail the test.
        for (var, flag) in [
            ("CN_ENABLE_FFX_FSR3", env.ffx_enabled),
            ("CN_ENABLE_XESS", env.xess_enabled),
            ("CN_ENABLE_DLSS", env.dlss_enabled),
            ("CN_ENABLE_DXC", env.dxc_enabled),
        ] {
            if std::env::var(var).is_err() {
                assert!(flag, "{var} should default on");
            }
        }
    }

    // The order is the contract, and it is not observable from a snapshot: a
    // host that has vendored the SDK resolves it there with nothing set, which
    // is correct and which asserting the default would call a failure.
    #[test]
    fn an_sdk_root_prefers_the_variable_then_vendor_then_nothing() {
        let vendored = PathBuf::from("/checkout/vendor/xess-3.0.1-windows-x86_64");
        let cases = [
            (Some("/explicit"), Some(vendored.clone()), Some("/explicit")),
            (Some("/explicit"), None, Some("/explicit")),
            (
                None,
                Some(vendored.clone()),
                Some("/checkout/vendor/xess-3.0.1-windows-x86_64"),
            ),
            // Nothing named a root, which is an answer rather than a reason to
            // guess: the warning then says the SDK is not vendored and the
            // variable is unset, not that it was missing from a path nobody has.
            (None, None, None),
        ];
        for (explicit, found, want) in cases {
            let got = sdk_root(explicit.map(str::to_string), found.clone());
            assert_eq!(got, want.map(PathBuf::from), "{explicit:?} + {found:?}");
        }
    }

    // Agility is the one that binds the executable to a directory beside it, so
    // an unset variable must leave it OFF: the default has to be the artifact
    // that runs anywhere, not the one that runs only where it was built.
    #[test]
    fn sdk_env_snapshot_defaults_agility_off() {
        if std::env::var("CN_ENABLE_AGILITY_SDK").is_err() {
            assert!(!sdk_env_from_cargo().agility_enabled);
        }
    }

    // Only an affirmative value opts in. `0` has always meant off and keeps
    // meaning off, so an environment carrying the old opt-out is unaffected.
    #[test]
    fn only_an_affirmative_value_opts_into_agility() {
        for on in ["1", "true", "TRUE"] {
            assert!(opted_in(Some(on)), "{on}");
        }
        for off in ["0", "", "no", "yes", "2", "false"] {
            assert!(!opted_in(Some(off)), "{off}");
        }
        assert!(!opted_in(None));
    }

    #[test]
    fn graphics_sdk_setup_is_a_noop_off_windows_targets() {
        // Metal never has SDKs to set up, and the Vulkan arm is gated on a
        // Windows target OS (CARGO_CFG_TARGET_OS is unset outside build
        // scripts), so neither requires any SDK to be present.
        let env = sdk_env_from_cargo();
        for targets in [
            &[BinaryTargets::None][..],
            &[BinaryTargets::Bins],
            &[BinaryTargets::Examples],
            &[BinaryTargets::Bins, BinaryTargets::Examples],
        ] {
            for backend in [Backend::Metal, Backend::Vk] {
                assert!(sdks::graphics_sdk_directives(backend, targets, &env).is_empty());
            }
        }
        setup_graphics_sdks(Some(Backend::Metal));
        setup_graphics_sdks(Some(Backend::Vk));
        // A build with no backend has no SDK setup to do at all.
        setup_graphics_sdks(None);
        // The check-cfg list is emitted unconditionally and must not panic.
        emit_check_cfgs();
    }

    #[test]
    fn this_crate_builds_no_final_binary() {
        // concinnity-toolchain is a lib with no bin and no example, so the
        // discovery run from its own test binary has to say so.
        assert_eq!(binary_targets_from_cargo(), vec![BinaryTargets::None]);
    }
}
