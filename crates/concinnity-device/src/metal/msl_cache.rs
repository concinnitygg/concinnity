// Disk-cached compilation for the MSL sources Metal must assemble at runtime:
// the raymarch libraries, whose user-authored SdfVolume fragment is spliced
// between engine templates and so cannot precompile into the build-time
// metallib. A cold build shells out to the same `xcrun metal` toolchain the
// build script uses and stores the metallib bytes content-addressed in the
// shader cache; a warm launch loads the bytes straight into
// `newLibraryWithData`, skipping the multi-hundred-millisecond in-process
// source compile. Without the toolchain (a machine with no Xcode) every path
// falls back to `newLibraryWithSource`, the previous behavior.
//
// The toolchain's release is part of the key. `shader_cache::verify_toolchain`
// discards the segment when slangc changes, but the Metal toolchain upgrades
// independently of it -- since Xcode 16 it is a separately versioned
// downloadable component, so it moves without a byte of slangc, of source, or
// of Xcode itself changing -- and a metallib from a superseded one loads
// perfectly well, which is exactly what makes replaying it invisible.

use concinnity_host::scratch::Scratch;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLLibrary};
use std::process::Command;
use std::sync::OnceLock;

// Produce the MTLLibrary for `source`, preferring a cached metallib. `label`
// names the shader in cache-miss logs and compile errors.
pub(super) fn compiled_library(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
    label: &str,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
    // No toolchain is both the reason a compile would fail and the reason its
    // output could not be keyed, so the cache is not consulted at all rather
    // than looked up under a key that names no release.
    let Some(compiler) = toolchain_id() else {
        tracing::debug!("{label}: no Metal toolchain, compiling from source");
        return source_library(device, source);
    };
    let key = crate::shader_cache::Key {
        compiler,
        source,
        entry: "main",
        target: "metallib",
        options: 0,
    };
    match crate::shader_cache::cached(&key, label, || compile_to_metallib(source, label)) {
        Ok(bytes) => match super::pipeline::load_library(device, &bytes) {
            Ok(library) => Ok(library),
            Err(e) => {
                tracing::warn!("{label}: cached metallib rejected ({e}), compiling from source");
                source_library(device, source)
            }
        },
        Err(e) => {
            // The compile failed. The in-process one below either succeeds
            // (odd, but its result stands) or surfaces the error the caller
            // expects.
            tracing::debug!("{label}: metallib cache unavailable ({e}), compiling from source");
            source_library(device, source)
        }
    }
}

fn source_library(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
    let options = objc2_metal::MTLCompileOptions::new();
    device
        .newLibraryWithSource_options_error(
            &objc2_foundation::NSString::from_str(source),
            Some(&options),
        )
        .map_err(|e| format!("{e:?}"))
}

// The Metal toolchain's release, as the cache's `compiler` field, or `None`
// when no toolchain resolves. Costs one `xcrun` per process, which is why it is
// a `OnceLock`: a cold one takes a few hundred milliseconds to mount the
// toolchain, against the multi-hundred milliseconds per shader the cache it
// keys exists to save.
fn toolchain_id() -> Option<&'static str> {
    static ID: OnceLock<Option<String>> = OnceLock::new();
    ID.get_or_init(|| {
        let out = Command::new("xcrun")
            .args(["--sdk", "macosx", "metal", "--version"])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        toolchain_id_from(&String::from_utf8_lossy(&out.stdout))
    })
    .as_deref()
}

// The release line of `metal --version`, which reads
// `Apple metal version 32023.864 (metalfe-32023.864)`. The whole line is kept
// rather than a version parsed out of it: it is what identifies the toolchain,
// and a compiler field that failed to parse would silently stop separating the
// releases it exists to separate.
fn toolchain_id_from(version_output: &str) -> Option<String> {
    let line = version_output
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())?;
    Some(format!("metal {line}"))
}

// Compile `source` to metallib bytes with `xcrun metal` / `xcrun metallib`,
// the same two-step pipeline the build script runs for the built-in shaders.
// Each scratch file removes itself, so a failed step leaves nothing behind.
fn compile_to_metallib(source: &str, label: &str) -> Result<Vec<u8>, String> {
    let msl = Scratch::file("msl.metal");
    let air = Scratch::file("msl.air");
    let metallib = Scratch::file("msl.metallib");

    std::fs::write(msl.path(), source)
        .map_err(|e| format!("write {}: {e}", msl.path().display()))?;
    run_step(
        Command::new("xcrun")
            .args(["--sdk", "macosx", "metal", "-c"])
            .arg(msl.path())
            .arg("-o")
            .arg(air.path()),
        label,
        "xcrun metal",
    )?;
    run_step(
        Command::new("xcrun")
            .args(["--sdk", "macosx", "metallib"])
            .arg(air.path())
            .arg("-o")
            .arg(metallib.path()),
        label,
        "xcrun metallib",
    )?;
    std::fs::read(metallib.path()).map_err(|e| format!("read compiled metallib: {e}"))
}

fn run_step(cmd: &mut Command, label: &str, what: &str) -> Result<(), String> {
    let output = cmd
        .output()
        .map_err(|e| format!("{what} failed to launch for {label}: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "{what} failed for {label}:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Since Xcode 16 the Metal toolchain is a separately downloadable
    // component, so it can move under a checkout that changed nothing else.
    // Its output for identical source is not guaranteed identical across
    // releases, and a stale metallib still loads -- so the release has to reach
    // the key, or an upgrade never takes effect.
    #[test]
    fn two_toolchain_releases_key_differently() {
        let older = toolchain_id_from(
            "Apple metal version 32023.404 (metalfe-32023.404)\n\
             Target: air64-apple-darwin25.6.0\n",
        );
        let newer = toolchain_id_from(
            "Apple metal version 32023.864 (metalfe-32023.864)\n\
             Target: air64-apple-darwin25.6.0\n",
        );
        assert!(older.is_some() && newer.is_some(), "{older:?} {newer:?}");
        assert_ne!(older, newer);
    }

    // The field separates toolchains, so the Metal entries must not collide
    // with what another backend's compiler writes under the same source.
    #[test]
    fn the_key_names_the_metal_toolchain() {
        let id = toolchain_id_from("Apple metal version 32023.864 (metalfe-32023.864)\n")
            .expect("a version line yields an id");
        assert!(id.starts_with("metal "), "{id}");
        assert!(id.contains("32023.864"), "{id}");
    }

    // `metal --version` leads with the release, but a toolchain that answered
    // with nothing usable must not collapse every release onto one key.
    #[test]
    fn an_empty_version_report_yields_no_id() {
        assert_eq!(toolchain_id_from(""), None);
        assert_eq!(toolchain_id_from("\n  \n\t\n"), None);
    }

    // Leading blank lines are not a version, and trailing whitespace on the one
    // that is would key two runs of the same toolchain apart.
    #[test]
    fn the_release_line_is_taken_trimmed() {
        assert_eq!(
            toolchain_id_from("\n\n  Apple metal version 32023.864  \nTarget: air64\n"),
            Some("metal Apple metal version 32023.864".to_string())
        );
    }
}
