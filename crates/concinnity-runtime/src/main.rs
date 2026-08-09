// concinnity-runtime: the shipped app player.
//
// A minimal standalone binary that plays a world's pre-compiled blobs.
//
// The state root (holding `data/`, plus the `saves/` and `settings` the app
// writes at runtime) is anchored to the executable, not the launch working
// directory, so the app finds its data whether it is double-clicked, launched
// from a shell, or run from inside a macOS `.app` bundle.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

// Windows' system `d3d12.dll` reads these two symbols from the host EXE's PE
// export table at process start to load the bundled Agility SDK D3D12 runtime
// in place of the older OS copy (modern FidelityFX FSR3 needs it). The build
// script copies the Agility DLLs into `<exe dir>/D3D12/` and emits the matching
// linker exports. Mirrors examples/bistro and concinnity-editor; keep the
// version in sync when bumping the Agility SDK.
#[cfg(backend_dx)]
#[unsafe(no_mangle)]
#[used]
pub static D3D12SDKVersion: u32 = 619;

#[cfg(backend_dx)]
#[unsafe(no_mangle)]
#[used]
pub static D3D12SDKPath: &[u8; 9] = b".\\D3D12\\\0";

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
    let content_dir = state_dir_for_exe(exe_dir);

    // Redirect runtime-writable state (`saves/` + `settings`) to a per-user
    // directory when the content dir cannot be written -- a read-only install
    // such as Program Files. In the portable case (content dir writable) both
    // stay beside the data, preserving the single-folder layout.
    if !dir_is_writable(&content_dir)
        && let Some(writable) = per_user_state_dir(&app_name_from_exe(&exe))
    {
        concinnity_engine::set_writable_state_dir(writable);
    }

    concinnity_engine::run_from(&content_dir)
}

// Resolve the flat state root that holds the world's `data/` blobs (and, unless
// redirected, the `saves/` + `settings` written at runtime) from the
// executable's directory. Inside a macOS `.app` the executable sits at
// `Contents/MacOS/<exe>` and the data lives in `Contents/Resources/`;
// everywhere else the data sits directly beside the executable.
fn state_dir_for_exe(exe_dir: &Path) -> PathBuf {
    let in_app_bundle = exe_dir.file_name() == Some(OsStr::new("MacOS"))
        && exe_dir.parent().and_then(Path::file_name) == Some(OsStr::new("Contents"));
    match exe_dir.parent() {
        Some(contents) if in_app_bundle => contents.join("Resources"),
        _ => exe_dir.to_path_buf(),
    }
}

// The application name used to key the per-user writable directory: the
// executable's file stem (the export slug), falling back to a generic name.
fn app_name_from_exe(exe: &Path) -> String {
    exe.file_stem()
        .and_then(OsStr::to_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("concinnity")
        .to_string()
}

// Whether `dir` accepts new files. Probes by creating (and removing) a uniquely
// named file; a read-only install (Program Files) fails here. A missing dir is
// treated as writable -- the runtime creates `saves/` under it on first save.
fn dir_is_writable(dir: &Path) -> bool {
    if !dir.exists() {
        return true;
    }
    let probe = dir.join(format!(".cn-write-probe-{}", std::process::id()));
    match std::fs::File::create(&probe) {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

// A per-user, always-writable directory for `saves/` + `settings`, keyed by the
// app name. `None` only when the platform's base directory cannot be resolved
// from the environment, in which case the caller leaves writable state beside
// the data.
fn per_user_state_dir(app: &str) -> Option<PathBuf> {
    per_user_base().map(|base| base.join(app))
}

// The platform base for per-user application state.
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

    // The shipped player counts its own heap. The allocator arrives by linking
    // the engine (`concinnity_engine::heap`) rather than being declared here,
    // so this asserts the property survives all the way out to the binary that
    // actually ships -- which no test inside a library can do for it.
    #[test]
    fn the_shipped_player_tracks_its_own_heap() {
        const MIB: usize = 1 << 20;

        let before = concinnity_memory::stats()
            .expect("linking the engine installs the allocator")
            .alloc_count;
        let held: Vec<u8> = core::hint::black_box(vec![0; MIB]);
        let after = concinnity_memory::stats().expect("the allocator stays installed");

        assert!(
            after.alloc_count > before,
            "allocation count did not move ({before} -> {}) across a megabyte",
            after.alloc_count
        );
        assert!(after.peak_bytes >= after.live_bytes);
        drop(core::hint::black_box(held));
    }

    #[test]
    fn plain_layout_uses_the_executable_directory() {
        // A portable folder (Windows/Linux, or a bare macOS binary): the state
        // tree sits directly beside the executable.
        let dir = Path::new("/apps/MyApp");
        assert_eq!(state_dir_for_exe(dir), Path::new("/apps/MyApp"));
    }

    #[test]
    fn macos_app_bundle_uses_resources() {
        // Contents/MacOS/<exe> -> data under Contents/Resources.
        let dir = Path::new("/Applications/MyGame.app/Contents/MacOS");
        assert_eq!(
            state_dir_for_exe(dir),
            Path::new("/Applications/MyGame.app/Contents/Resources")
        );
    }

    #[test]
    fn macos_like_path_not_in_bundle_stays_beside_exe() {
        // A `MacOS` directory that is not under `Contents` is not a bundle.
        let dir = Path::new("/home/user/MacOS");
        assert_eq!(state_dir_for_exe(dir), Path::new("/home/user/MacOS"));
    }

    #[test]
    fn app_name_falls_back_when_stem_missing() {
        assert_eq!(app_name_from_exe(Path::new("/apps/MyGame")), "MyGame");
        assert_eq!(app_name_from_exe(Path::new("MyGame.exe")), "MyGame");
        // No file name at all: the generic fallback keeps the path well-formed.
        assert_eq!(app_name_from_exe(Path::new("/")), "concinnity");
    }

    #[test]
    fn a_writable_dir_probes_true_and_leaves_nothing_behind() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(dir_is_writable(tmp.path()));
        // The probe file is cleaned up.
        assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 0);
    }

    #[test]
    fn a_missing_dir_is_treated_as_writable() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("not-created-yet");
        assert!(dir_is_writable(&missing));
    }

    #[test]
    fn per_user_dir_appends_the_app_name_under_a_base() {
        // The host always has a resolvable base (HOME / LOCALAPPDATA), so the
        // per-user dir is Some and ends with the app name.
        let dir = per_user_state_dir("MyGame").expect("a per-user base on the test host");
        assert_eq!(dir.file_name().and_then(OsStr::to_str), Some("MyGame"));
        assert!(dir.is_absolute());
    }
}
