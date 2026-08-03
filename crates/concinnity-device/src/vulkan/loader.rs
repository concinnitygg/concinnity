// src/vulkan/loader.rs
//
// Acquiring the Vulkan loader library. `ash::Entry::load` resolves the loader by
// its bare file name through the platform's dynamic linker, which finds it on
// Windows and Linux. It does not on macOS: the LunarG SDK installs
// `libvulkan.dylib` under /usr/local/lib, and dyld searches neither that nor
// Homebrew's prefix for a leaf name. So the plain load is followed by the known
// install paths, tried in order.

use std::path::Path;

// Absolute paths to try after `ash::Entry::load` fails, in preference order.
// Empty where the dynamic linker already finds the loader on its own.
#[cfg(target_os = "macos")]
const FALLBACK_PATHS: &[&str] = &[
    // LunarG SDK, installed system-wide.
    "/usr/local/lib/libvulkan.dylib",
    // Homebrew's vulkan-loader formula, Apple Silicon then Intel prefix.
    "/opt/homebrew/lib/libvulkan.dylib",
    "/usr/local/opt/vulkan-loader/lib/libvulkan.dylib",
];
#[cfg(not(target_os = "macos"))]
const FALLBACK_PATHS: &[&str] = &[];

// Load the Vulkan loader, falling back to the platform's known install paths.
// The error carries the dynamic linker's own message plus the paths tried, so a
// missing SDK is diagnosable from the log alone.
pub(super) fn load_entry() -> Result<ash::Entry, String> {
    let err = match unsafe { ash::Entry::load() } {
        Ok(entry) => return Ok(entry),
        Err(e) => e,
    };
    for path in existing(FALLBACK_PATHS, |p| Path::new(p).exists()) {
        match unsafe { ash::Entry::load_from(path) } {
            Ok(entry) => {
                tracing::info!("Vulkan loader loaded from {path}");
                return Ok(entry);
            }
            Err(e) => tracing::warn!("Vulkan loader at {path} failed to load: {e}"),
        }
    }
    Err(match FALLBACK_PATHS {
        [] => format!("load vulkan: {err}"),
        paths => format!("load vulkan: {err}; also tried {}", paths.join(", ")),
    })
}

// The candidates that are present on disk, so a missing path is skipped without
// a load attempt (and its warning). `exists` is injected for testing.
fn existing<'a>(paths: &[&'a str], exists: impl Fn(&str) -> bool) -> Vec<&'a str> {
    paths.iter().copied().filter(|p| exists(p)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallbacks_are_macos_only() {
        // Every other platform's dynamic linker resolves the loader by name, so
        // an empty list keeps the error message free of irrelevant paths.
        assert_eq!(FALLBACK_PATHS.is_empty(), !cfg!(target_os = "macos"));
    }

    #[test]
    fn fallback_paths_are_absolute() {
        // A relative candidate would resolve against the process working
        // directory, which is wherever the user launched from.
        for path in FALLBACK_PATHS {
            assert!(Path::new(path).is_absolute(), "{path} is not absolute");
        }
    }

    #[test]
    fn only_present_candidates_are_tried() {
        let paths = ["/a/libvulkan.dylib", "/b/libvulkan.dylib"];
        assert_eq!(
            existing(&paths, |p| p.starts_with("/b")),
            ["/b/libvulkan.dylib"]
        );
        assert!(existing(&paths, |_| false).is_empty());
        assert_eq!(existing(&paths, |_| true), paths);
    }
}
