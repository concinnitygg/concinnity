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

use std::path::{Path, PathBuf};

mod blob;
mod state;

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
    concinnity_engine::crash::install();

    // Keep the backend stamp in the linked binary (its bytes are what `cn
    // export` scans); taking its address defeats any linker dead-stripping.
    std::hint::black_box(&CN_RUNTIME_PLATFORM);

    let exe = std::env::current_exe()?;
    let exe_dir = exe.parent().unwrap_or_else(|| Path::new("."));
    let content_dir = state::state_dir_for_exe(exe_dir);

    // Redirect runtime-writable state (`saves/` + `settings`) to a per-user
    // directory when the content dir cannot be written -- a read-only install
    // such as Program Files. In the portable case (content dir writable) both
    // stay beside the data, preserving the single-folder layout. The world's
    // own `AppConfig.home`, applied once the blob is read, overrides either.
    if !state::dir_is_writable(&content_dir)
        && let Some(writable) = state::per_user_state_dir(&state::app_name_from_exe(&exe))
    {
        concinnity_engine::set_writable_state_dir(writable);
    }

    // One positional argument names the world instead of the bundled `data`.
    // It moves only what is read: `saves/` and `settings` stay anchored above,
    // so pointing the player at a blob in a read-only place never relocates a
    // player's saves.
    let requested = std::env::args_os()
        .nth(1)
        .map_or_else(|| content_dir.join("data"), PathBuf::from);
    let blob = blob::blob_source(&requested).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no world data at {}", requested.display()),
        )
    })?;

    concinnity_engine::run_from(&content_dir, blob.as_source())
}

#[cfg(test)]
mod tests {
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
