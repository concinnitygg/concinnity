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
//! One file, owning what a process owns -- the tracking allocator and the
//! backend stamp `cn export` reads back. Resolving the layout and loading the
//! world live in concinnity-host and concinnity-runtime.

use std::path::{Path, PathBuf};

concinnity_core::install_global_allocator!();

// Backend stamp read back by `cn export`.
//
// A shipped player consumes shaders in the formats of the backends it carries:
// Metal `.metallib`, DirectX DXIL, Vulkan SPIR-V. `cn` and this runtime compile
// independently, so the runtime bakes a fixed marker plus the shader-platform
// keys of every backend it can render with (`metal` / `directx` / `vulkan`,
// comma-separated, matching `concinnity_engine::platform::Platform::key`); `cn
// export` scans these bytes, refuses a player it cannot cook for, and cooks the
// blobs for exactly those platforms. `main` takes the static's address through
// a `black_box` so no linker dead-strips it. A build with no backend stamps an
// empty list, which the export treats as an unstamped runtime.
#[used]
static CN_RUNTIME_PLATFORM: &[u8] =
    concat!("cn-runtime-platform:", env!("CN_RUNTIME_PLATFORMS"), "\0").as_bytes();

fn main() -> std::io::Result<()> {
    // Keep the backend stamp in the linked binary (its bytes are what `cn
    // export` scans); taking its address defeats any linker dead-stripping.
    std::hint::black_box(&CN_RUNTIME_PLATFORM);

    let exe = std::env::current_exe()?;
    let exe_dir = exe.parent().unwrap_or_else(|| Path::new("."));
    let base = concinnity_host::store::paths::per_user_base();
    let tree = concinnity_host::store::paths::tree_for_exe(&exe, exe_dir, base.as_deref());

    // Before anything else can fault: a report written from here on lands
    // beside the app rather than nowhere.
    concinnity_runtime::crash::install(Some(&tree.crashes_dir()));

    // One positional argument names the world instead of the bundled `data`.
    // It moves only what is read: `saves/` and `settings` stay in the tree
    // resolved above, so pointing the player at a blob in a read-only place
    // never relocates a player's saves.
    let requested = std::env::args_os()
        .nth(1)
        .map_or_else(|| tree.data_dir(), PathBuf::from);
    let blob = concinnity_runtime::BlobSource::classify(&requested).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no world data at {}", requested.display()),
        )
    })?;

    concinnity_runtime::run_from(&tree, &blob)
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
