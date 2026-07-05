// concinnity-runtime: the shipped app player.
//
// A minimal standalone binary that plays a world's pre-compiled blobs. It links
// only the runtime crate (concinnity-client) and the shared data types it
// re-exports (concinnity-core), never the cook/build pipeline or the editor, so
// a shipped app carries no compiler. `cn export` ships this prebuilt beside the
// `cn` toolchain and copies it (renamed) into each app bundle beside the world's
// `data/` blobs.
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

fn main() -> std::io::Result<()> {
    let exe = std::env::current_exe()?;
    let exe_dir = exe.parent().unwrap_or_else(|| Path::new("."));
    concinnity_client::app::run::run_from(&state_dir_for_exe(exe_dir))
}

// Resolve the flat state root that holds the world's `data/` blobs (and the
// `saves/` + `settings` written at runtime) from the executable's directory.
// Inside a macOS `.app` the executable sits at `Contents/MacOS/<exe>` and the
// data lives in `Contents/Resources/`; everywhere else the data sits directly
// beside the executable.
fn state_dir_for_exe(exe_dir: &Path) -> PathBuf {
    let in_app_bundle = exe_dir.file_name() == Some(OsStr::new("MacOS"))
        && exe_dir.parent().and_then(Path::file_name) == Some(OsStr::new("Contents"));
    match exe_dir.parent() {
        Some(contents) if in_app_bundle => contents.join("Resources"),
        _ => exe_dir.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
