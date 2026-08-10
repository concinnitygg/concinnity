// Disk-cached compilation for the MSL sources Metal must assemble at runtime:
// the raymarch libraries, whose user-authored SdfVolume fragment is spliced
// between engine templates and so cannot precompile into the build-time
// metallib. A cold build shells out to the same `xcrun metal` toolchain the
// build script uses and stores the metallib bytes content-addressed in the
// shader cache; a warm launch loads the bytes straight into
// `newLibraryWithData`, skipping the multi-hundred-millisecond in-process
// source compile. Without the toolchain (a machine with no Xcode) every path
// falls back to `newLibraryWithSource`, the previous behavior.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLLibrary};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

// Produce the MTLLibrary for `source`, preferring a cached metallib. `label`
// names the shader in cache-miss logs and compile errors.
pub(super) fn compiled_library(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
    label: &str,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
    let key = crate::shader_cache::Key {
        compiler: "metal",
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
            // No toolchain, or the compile failed. The in-process compile
            // below either succeeds (odd, but its result stands) or surfaces
            // the error the caller expects.
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

// Compile `source` to metallib bytes with `xcrun metal` / `xcrun metallib`,
// the same two-step pipeline the build script runs for the built-in shaders.
// Scratch files live beside the shader cache (under the writable state dir)
// and are removed on every exit path.
fn compile_to_metallib(source: &str, label: &str) -> Result<Vec<u8>, String> {
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let dir = concinnity_store::paths::shader_cache_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let stem = dir.join(format!(
        "msl.{}.{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let scratch = Scratch {
        source: stem.with_extension("metal"),
        air: stem.with_extension("air"),
        metallib: stem.with_extension("metallib"),
    };
    std::fs::write(&scratch.source, source)
        .map_err(|e| format!("write {}: {e}", scratch.source.display()))?;
    run_step(
        Command::new("xcrun")
            .args(["--sdk", "macosx", "metal", "-c"])
            .arg(&scratch.source)
            .arg("-o")
            .arg(&scratch.air),
        label,
        "xcrun metal",
    )?;
    run_step(
        Command::new("xcrun")
            .args(["--sdk", "macosx", "metallib"])
            .arg(&scratch.air)
            .arg("-o")
            .arg(&scratch.metallib),
        label,
        "xcrun metallib",
    )?;
    std::fs::read(&scratch.metallib).map_err(|e| format!("read compiled metallib: {e}"))
}

struct Scratch {
    source: PathBuf,
    air: PathBuf,
    metallib: PathBuf,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        for path in [&self.source, &self.air, &self.metallib] {
            let _ = std::fs::remove_file(path);
        }
    }
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
