//! concinnity-player: the shipped app player, built by the `std` feature.
//!
//! A minimal standalone binary that plays a world's pre-compiled blobs.
//!
//! The state root (holding the world's data, plus the `saves/` and `settings`
//! the app writes at runtime) is anchored to the executable, not the launch
//! working directory, so the app finds its data whether it is double-clicked,
//! launched from a shell, or run from inside a macOS `.app` bundle.
//!
//! Inside that directory the world is the `data/` directory `cn export` ships,
//! holding blob `0`, its overflow siblings, and the engine shaders. A single
//! positional argument overrides it with a blob file or a directory of blobs;
//! it moves only what is read, never where the app writes.
//!
//! One file, owning what a process owns: the backend and shader stamps `cn
//! export` reads back. Resolving the layout and loading the world come from
//! `concinnity::player`.

use std::path::{Path, PathBuf};

use concinnity::player;

// Backend stamp read back by `cn export`.
//
// A shipped player consumes shaders in the formats of the backends it carries:
// Metal `.metallib`, DirectX DXIL, Vulkan SPIR-V. `cn` and this runtime compile
// independently, so the runtime bakes a fixed marker plus the shader-platform
// keys of every backend it can render with (`metal` / `directx` / `vulkan`,
// comma-separated, matching `concinnity_core::platform::Platform::key`); `cn
// export` scans these bytes, refuses a player it cannot cook for, and cooks the
// blobs for exactly those platforms. `main` takes the static's address through
// a `black_box` so no linker dead-strips it. A build with no backend stamps an
// empty list, which the export treats as an unstamped runtime.
#[used]
static CN_RUNTIME_PLATFORM: &[u8] =
    concat!("cn-runtime-platform:", env!("CN_RUNTIME_PLATFORMS"), "\0").as_bytes();

// Shader stamp read back by `cn export`: the stamp of the shaders this player
// accepts as shipped, as 16 hex digits after a fixed marker, so an export
// compiled by a `cn` built from other shader sources or flags is refused before
// it ships rather than when the player cannot find a compiler.
#[used]
static CN_SHADER_STAMP: [u8; 33] = shader_stamp(player::SHIPPED_SHADER_STAMP);

const fn shader_stamp(stamp: u64) -> [u8; 33] {
    const MARKER: &[u8; 16] = b"cn-shader-stamp:";
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = [0u8; 33];
    let mut i = 0;
    while i < MARKER.len() {
        out[i] = MARKER[i];
        i += 1;
    }
    let mut digit = 0;
    while digit < 16 {
        out[MARKER.len() + digit] = HEX[((stamp >> (60 - 4 * digit)) & 0xf) as usize];
        digit += 1;
    }
    out
}

fn main() -> std::io::Result<()> {
    // Keep the stamps in the linked binary (their bytes are what `cn export`
    // scans); taking their addresses defeats any linker dead-stripping.
    std::hint::black_box(&CN_RUNTIME_PLATFORM);
    std::hint::black_box(&CN_SHADER_STAMP);

    let exe = std::env::current_exe()?;
    let exe_dir = exe.parent().unwrap_or_else(|| Path::new("."));
    let base = player::per_user_base();
    let tree = player::tree_for_exe(&exe, exe_dir, base.as_deref());

    // Before anything else can fault: a report written from here on lands
    // beside the app rather than nowhere.
    player::install_crash_hooks(Some(&tree.crash_log_path()));

    // One positional argument names the world instead of the bundled `data`.
    // It moves only what is read: `saves/` and `settings` stay in the tree
    // resolved above, so pointing the player at a blob in a read-only place
    // never relocates a player's saves.
    let requested = std::env::args_os()
        .nth(1)
        .map_or_else(|| tree.data_dir(), PathBuf::from);
    let blob = player::BlobSource::classify(&requested).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no world data at {}", requested.display()),
        )
    })?;

    player::run_from(&tree, &blob)
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_shader_stamp_is_the_marker_then_hex_digits() {
        let stamp = super::shader_stamp(0x0123_4567_89ab_cdef);
        assert_eq!(&stamp[..32], b"cn-shader-stamp:0123456789abcdef");
        assert_eq!(stamp[32], 0);
    }

    // The shipped player counts its own heap through the library's
    // counting allocator. Without it the player would run correctly
    // while reporting no memory at all, and crash reports would ship without
    // heap figures.
    #[test]
    fn the_shipped_player_tracks_its_own_heap() {
        const MIB: usize = 1 << 20;

        let before = concinnity::memory::stats()
            .expect("this binary declares the tracking allocator")
            .alloc_count;
        let held: Vec<u8> = core::hint::black_box(vec![0; MIB]);
        let after = concinnity::memory::stats().expect("the allocator stays installed");

        assert!(
            after.alloc_count > before,
            "allocation count did not move ({before} -> {}) across a megabyte",
            after.alloc_count
        );
        assert!(after.peak_bytes >= after.live_bytes);
        drop(core::hint::black_box(held));
    }
}
