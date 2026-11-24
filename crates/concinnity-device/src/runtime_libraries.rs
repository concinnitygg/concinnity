//! Where the renderer finds the vendor runtime libraries it loads (the
//! upscalers, Streamline, and the Agility D3D12 runtime) besides the
//! executable's own directory: one directory the host names before the first
//! backend starts, as a development run names the libraries it vendored. A
//! shipped app keeps them beside its executable and names none.

use std::path::Path;
#[cfg(windows)]
use std::path::PathBuf;
#[cfg(windows)]
use std::sync::OnceLock;

#[cfg(windows)]
static DIRECTORY: OnceLock<PathBuf> = OnceLock::new();

/// Load the vendor runtime libraries from `dir` as well as from beside the
/// executable. Takes effect for the life of the process from the first call,
/// which has to come before a backend or device probe starts; later calls are
/// ignored. The libraries are Windows-only, so elsewhere this does nothing.
pub fn use_runtime_libraries(dir: &Path) {
    #[cfg(windows)]
    {
        if DIRECTORY.set(dir.to_path_buf()).is_err() {
            return;
        }
        add_to_search_path(dir);
        tracing::info!("runtime libraries: also loading from {}", dir.display());
    }
    #[cfg(not(windows))]
    let _ = dir;
}

/// The directory [`use_runtime_libraries`] named, if any.
#[cfg(windows)]
pub(crate) fn directory() -> Option<&'static Path> {
    DIRECTORY.get().map(PathBuf::as_path)
}

/// Where a vendor library is looked for, in order: beside the executable,
/// then the named directory.
#[cfg(windows)]
pub(crate) fn search_dirs() -> impl Iterator<Item = PathBuf> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    exe_dir
        .into_iter()
        .chain(directory().map(Path::to_path_buf))
}

// Let a library loaded by name, and the libraries it loads in turn by name,
// resolve from `dir` after the executable's directory.
#[cfg(windows)]
fn add_to_search_path(dir: &Path) {
    use std::os::windows::ffi::OsStrExt;
    use windows::Win32::System::LibraryLoader::SetDllDirectoryW;
    use windows::core::PCWSTR;

    let wide: Vec<u16> = dir.as_os_str().encode_wide().chain([0]).collect();
    // SAFETY: `wide` is a NUL-terminated path live for the call, which copies it.
    if let Err(e) = unsafe { SetDllDirectoryW(PCWSTR(wide.as_ptr())) } {
        tracing::warn!(
            "runtime libraries: could not add {} to the search path: {e}",
            dir.display()
        );
    }
}
