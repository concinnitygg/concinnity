//! The prebuilt player a bundle ships: where it is, which platforms it was built for, and the libraries it needs beside it.

use concinnity_cook::platform::{Platform, PlatformSet};
use std::path::{Path, PathBuf};
use std::{fs, io};

use super::files::copy_tree;

// Reject a target that is not the platform this `cn` was built for. Cross-
// platform export is not supported yet: the runtime player is a single compiled
// binary and cooked blobs embed platform-native shaders, so a bundle can only
// target the host.
pub(super) fn check_target_platform(platform: Option<&str>) -> io::Result<()> {
    let Some(requested) = platform else {
        return Ok(());
    };
    let host = std::env::consts::OS;
    if normalize_platform(requested) == host {
        return Ok(());
    }
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        format!(
            "cross-platform export is not supported yet: this `concinnity` targets '{host}'. \
             Run `concinnity export` on a {requested} machine to produce a {requested} build."
        ),
    ))
}

pub(super) fn normalize_platform(p: &str) -> &str {
    match p.to_lowercase().as_str() {
        "mac" | "macos" | "osx" | "darwin" => "macos",
        "win" | "windows" => "windows",
        "linux" => "linux",
        // Unknown values fall through and simply won't match the host.
        _ => "",
    }
}

// Locate the runtime player that ships beside this executable.
pub(super) fn runtime_binary_path() -> io::Result<PathBuf> {
    let cn = std::env::current_exe()?;
    let dir = cn
        .parent()
        .ok_or_else(|| io::Error::other("cannot locate the concinnity executable's directory"))?;
    let path = dir.join(exe_file_name("concinnity-player"));
    if path.exists() {
        Ok(path)
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "runtime player not found at {} -- the `concinnity-player` binary must sit \
                 beside the `concinnity` executable (in a dev checkout, build it with \
                 `cargo build`)",
                path.display()
            ),
        ))
    }
}

// The fixed prefix of the backend stamp baked into `concinnity-player` (see its
// definition in that binary's main.rs); the comma-separated shader-platform
// keys follow it.
pub(super) const RUNTIME_PLATFORM_MARKER: &[u8] = b"cn-runtime-platform:";

// Read the runtime player's stamped shader platforms by scanning its binary for
// the backend marker. Returns None for an unstamped runtime, warning once so the
// skipped backend check is visible.
pub(super) fn read_runtime_platforms(runtime: &Path) -> io::Result<Option<PlatformSet>> {
    let bytes = fs::read(runtime)?;
    let Some(token) = find_platform_stamp(&bytes) else {
        eprintln!(
            "warning: no backend stamp found in {}; skipping the runtime/cook \
             backend check (rebuild `concinnity-player` to enable it)",
            runtime.display()
        );
        return Ok(None);
    };
    parse_runtime_platforms(&token).map(Some)
}

// The platforms a stamp token names, refusing a key this `cn` does not know.
pub(super) fn parse_runtime_platforms(token: &str) -> io::Result<PlatformSet> {
    token
        .split(',')
        .map(|key| {
            Platform::from_key(key).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("the `concinnity-player` binary names an unknown backend `{key}`"),
                )
            })
        })
        .collect()
}

// The platforms the export cooks for: the player's, when this `cn` can cook
// every one of them. `cn` and `concinnity-player` compile independently, so a `cn`
// built without Vulkan beside a player that renders with it would otherwise ship
// blobs missing the SPIR-V that player loads. `found` is None for an unstamped
// runtime (already warned about), which gets everything `cookable` covers.
pub(super) fn verify_runtime_backend(
    found: Option<PlatformSet>,
    cookable: PlatformSet,
) -> io::Result<PlatformSet> {
    let Some(found) = found else {
        return Ok(cookable);
    };
    if found.is_subset(cookable) {
        return Ok(found);
    }
    let missing: Vec<&str> = found
        .iter()
        .filter(|p| !cookable.contains(*p))
        .map(|p| backend_label(p.key()))
        .collect();
    let hint = if missing.len() == 1 && !cookable.contains(Platform::Vulkan) {
        " Rebuild `concinnity` with the `vulkan` feature (`cargo build --features vulkan`)."
    } else {
        ""
    };
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
            "runtime/cook backend mismatch: the `concinnity-player` binary beside this `concinnity` \
             renders with {}, which this `concinnity` cannot cook blobs for.{hint}",
            missing.join(" and "),
        ),
    ))
}

// Extract the shader-platform token from a runtime binary's backend stamp: the
// run of lowercase ASCII and commas immediately after the marker prefix. Returns None when
// the marker is absent (an older, unstamped runtime).
pub(super) fn find_platform_stamp(bytes: &[u8]) -> Option<String> {
    let start = find_subslice(bytes, RUNTIME_PLATFORM_MARKER)? + RUNTIME_PLATFORM_MARKER.len();
    let rest = &bytes[start..];
    let end = rest
        .iter()
        .position(|b| !(b.is_ascii_lowercase() || *b == b','))
        .unwrap_or(rest.len());
    let token = &rest[..end];
    if token.is_empty() {
        None
    } else {
        std::str::from_utf8(token).ok().map(str::to_string)
    }
}

// Index of the first occurrence of `needle` in `haystack`, or None.
pub(super) fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

// A human-facing label for a shader-platform key.
pub(super) fn backend_label(platform_key: &str) -> &str {
    match platform_key {
        "metal" => "Metal (MSL)",
        "directx" => "DirectX (DXIL)",
        "vulkan" => "Vulkan (SPIR-V)",
        other => other,
    }
}

// Platform executable file name: append `.exe` on Windows.
pub(super) fn exe_file_name(stem: &str) -> String {
    if cfg!(windows) {
        format!("{stem}.exe")
    } else {
        stem.to_string()
    }
}

// Copy the runtime player's sibling native libraries into the bundle beside the
// exe, so the bundle carries everything the player needs to launch. On Windows
// the graphics-SDK runtime DLLs (FidelityFX / XeSS / DLSS / DXC) sit beside the
// built runtime binary, and the Agility SDK D3D12 runtime lives in a `D3D12/`
// subdir that the exe's `D3D12SDKPath` export points at (`.\D3D12\`). A strict
// no-op on macOS and Linux, where the runtime links only system frameworks /
// libraries (no sibling `.dll`, no `D3D12/`).
//
// `runtime_platforms` are the player's stamped shader platforms. The `D3D12/` dir
// is meaningful only to a DirectX player -- its `D3D12SDKPath` export is gated
// on the DX backend, so a Vulkan or Metal player never references it. It is
// copied only for a DX (or unstamped, so unknown) runtime, keeping a Vulkan
// bundle from carrying an inert Agility runtime when built in a target dir
// shared with a DX build.
pub(super) fn copy_runtime_sidecars(
    runtime: &Path,
    runtime_platforms: Option<PlatformSet>,
    dest_dir: &Path,
) -> io::Result<()> {
    let Some(src_dir) = runtime.parent() else {
        return Ok(());
    };
    // Sibling DLLs (the Windows graphics-SDK runtimes). Matched by extension so
    // the set can change without export knowing backend specifics.
    for entry in fs::read_dir(src_dir)? {
        let path = entry?.path();
        let is_dll = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("dll"));
        if is_dll
            && path.is_file()
            && let Some(name) = path.file_name()
        {
            fs::copy(&path, dest_dir.join(name))?;
        }
    }
    // The Agility SDK D3D12 runtime, resolved by the exe relative to itself.
    // Only a DX player looks for it (see above).
    if runtime_wants_d3d12(runtime_platforms) {
        let d3d12 = src_dir.join("D3D12");
        if d3d12.is_dir() {
            copy_tree(&d3d12, &dest_dir.join("D3D12"))?;
        }
    }
    Ok(())
}

// Whether a player with the given stamped platforms references the `D3D12/`
// Agility runtime: one carrying DirectX does, a Vulkan or Metal player never
// does, and an unstamped (unknown) runtime is treated as maybe-DX so nothing it
// needs is dropped.
pub(super) fn runtime_wants_d3d12(runtime_platforms: Option<PlatformSet>) -> bool {
    runtime_platforms.is_none_or(|set| set.contains(Platform::DirectX))
}
