//! concinnity-run: the shipped app player, built by the `player` feature.
//!
//! A minimal standalone binary that plays a world's pre-compiled blobs.
//!
//! The state root (holding the world's data, plus the `saves/` and `settings`
//! the app writes at runtime) is anchored to the executable, not the launch
//! working directory, so the app finds its data whether it is double-clicked,
//! launched from a shell, or run from inside a macOS `.app` bundle.
//!
//! Inside that directory the world is either a file named `data` (one
//! self-contained blob, what `cn export` ships for a small game) or a directory
//! named `data` holding blob `0` and its overflow siblings. A single positional
//! argument overrides both with a blob file or a directory of blobs; it moves
//! only what is read, never where the app writes.
//!
//! One file, owning what a process owns -- the tracking allocator, the backend
//! stamp `cn export` reads back, and the per-user base the writable state falls
//! back to. Resolving the layout and loading the world live in concinnity-host
//! and concinnity-engine.

use std::path::{Path, PathBuf};

concinnity_core::install_global_allocator!();

// Backend stamp read back by `cn export`.
//
// A shipped player consumes shaders in exactly one format, fixed by the backend
// it was compiled for: Metal `.metallib`, DirectX DXBC, or Vulkan SPIR-V. `cn`
// and this runtime compile independently, so a DX-built `cn` could sit beside a
// Vulkan-built runtime and export a player that fails to load every shader at
// launch. To catch that, the runtime bakes a fixed marker plus its shader
// platform key into its binary; `cn export` scans these bytes and refuses a
// mismatch. The token after the `cn-runtime-platform:` prefix is the
// shader-platform key (`metal` / `hlsl` / `glsl`), matching
// `concinnity_core::platform::Platform::key`. `main` takes the static's
// address through a `black_box` so no linker dead-strips it.
// A build with no backend consumes no shaders, so it carries no stamp and the
// export treats it the way it treats any unstamped runtime.
#[cfg(backend_metal)]
#[used]
static CN_RUNTIME_PLATFORM: [u8; 26] = *b"cn-runtime-platform:metal\0";
#[cfg(backend_dx)]
#[used]
static CN_RUNTIME_PLATFORM: [u8; 25] = *b"cn-runtime-platform:hlsl\0";
#[cfg(backend_vk)]
#[used]
static CN_RUNTIME_PLATFORM: [u8; 25] = *b"cn-runtime-platform:glsl\0";

fn main() -> std::io::Result<()> {
    // Keep the backend stamp in the linked binary (its bytes are what `cn
    // export` scans); taking its address defeats any linker dead-stripping.
    #[cfg(any(backend_metal, backend_dx, backend_vk))]
    std::hint::black_box(&CN_RUNTIME_PLATFORM);

    let exe = std::env::current_exe()?;
    let exe_dir = exe.parent().unwrap_or_else(|| Path::new("."));
    let tree = concinnity_engine::paths::tree_for_exe(&exe, exe_dir, per_user_base().as_deref());

    // Before anything else can fault: a report written from here on lands
    // beside the app rather than nowhere.
    concinnity_engine::crash::install(Some(&tree.crashes_dir()));

    // One positional argument names the world instead of the bundled `data`.
    // It moves only what is read: `saves/` and `settings` stay in the tree
    // resolved above, so pointing the player at a blob in a read-only place
    // never relocates a player's saves.
    let requested = std::env::args_os()
        .nth(1)
        .map_or_else(|| tree.data_dir(), PathBuf::from);
    let blob = concinnity_engine::blob_source(&requested).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no world data at {}", requested.display()),
        )
    })?;

    concinnity_engine::run_from(&tree, blob.as_source())
}

// The platform base for per-user application state, from the environment the
// process was launched with. `None` when it cannot be resolved, in which case
// the writable state stays beside the data.
#[cfg(windows)]
fn per_user_base() -> Option<PathBuf> {
    // %LOCALAPPDATA% (e.g. C:\Users\<user>\AppData\Local), falling back to the
    // roaming %APPDATA% if the local one is somehow unset.
    non_empty_env("LOCALAPPDATA")
        .or_else(|| non_empty_env("APPDATA"))
        .map(PathBuf::from)
}

#[cfg(target_os = "macos")]
fn per_user_base() -> Option<PathBuf> {
    non_empty_env("HOME").map(|h| PathBuf::from(h).join("Library").join("Application Support"))
}

#[cfg(all(unix, not(target_os = "macos")))]
fn per_user_base() -> Option<PathBuf> {
    // The XDG base-directory spec: $XDG_DATA_HOME, else ~/.local/share.
    non_empty_env("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| non_empty_env("HOME").map(|h| PathBuf::from(h).join(".local").join("share")))
}

// An environment variable's value when set and non-empty. Keeps the base
// resolvers from returning a base rooted at "" (which would place per-user
// state at the filesystem root).
#[cfg(any(windows, unix))]
fn non_empty_env(key: &str) -> Option<std::ffi::OsString> {
    std::env::var_os(key).filter(|v| !v.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    // The host always has a resolvable base (HOME / LOCALAPPDATA), so a
    // read-only install on it always has somewhere to redirect its writes.
    #[test]
    fn the_per_user_base_resolves_to_an_absolute_directory() {
        let base = per_user_base().expect("a per-user base on the test host");
        assert!(base.is_absolute(), "{}", base.display());
    }

    // The shipped player counts its own heap. Nothing forces the declaration
    // at the top of this file to exist, so this is what catches its removal:
    // without it the player would run correctly while reporting no memory at
    // all, and crash reports would ship without heap figures.
    #[test]
    fn the_shipped_player_tracks_its_own_heap() {
        const MIB: usize = 1 << 20;

        let before = concinnity_core::memory::stats()
            .expect("this binary declares the tracking allocator")
            .alloc_count;
        let held: Vec<u8> = core::hint::black_box(vec![0; MIB]);
        let after = concinnity_core::memory::stats().expect("the allocator stays installed");

        assert!(
            after.alloc_count > before,
            "allocation count did not move ({before} -> {}) across a megabyte",
            after.alloc_count
        );
        assert!(after.peak_bytes >= after.live_bytes);
        drop(core::hint::black_box(held));
    }
}
