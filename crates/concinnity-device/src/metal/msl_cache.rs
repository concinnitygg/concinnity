// Disk-cached compilation for MSL sources: a world's Shader and SdfVolume
// programs, whose authored text is spliced into engine templates and so cannot
// precompile into the build-time metallib, and the hand-written cull encode
// kernel when its source no longer matches the embedded copy. A cold build
// compiles through `concinnity_shader::metallib`, the same path the build
// script takes for the built-in shaders, and stores the metallib bytes
// content-addressed in the shader cache; a warm launch loads the bytes straight
// into `newLibraryWithData`, skipping the multi-hundred-millisecond in-process
// source compile. Without the toolchain (a machine with no Xcode) every path
// falls back to `newLibraryWithSource`.
//
// The toolchain's id is part of the key. `shader::cache::verify_toolchain`
// discards the segment when dxc changes, but the Metal toolchain upgrades
// independently of it, and a metallib from a superseded one loads perfectly
// well, which is exactly what makes replaying it invisible.

use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_shader::HlslTarget;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLDevice, MTLFunction, MTLLibrary};

use crate::shader::cache::Key;

// The cache key for a metallib compiled from `source`. Keyed on the Metal
// toolchain's release; `None` without a toolchain, whose output nothing could
// key.
pub(super) fn metallib_key<'a>(source: &'a str, entry: &'a str) -> Option<Key<'a>> {
    concinnity_shader::metallib::toolchain_id().map(|compiler| Key {
        compiler,
        source,
        entry,
        target: HlslTarget::Metallib.name(),
    })
}

// Produce the MTLLibrary for `source`: `embedded` when it was built from this
// exact text, else a cached or fresh metallib, else an in-process source
// compile. `label` names the shader in cache-miss logs and compile errors.
pub(super) fn compiled_library(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
    label: &str,
    embedded: Option<(u64, &'static [u8])>,
) -> RenderResult<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let key = metallib_key(source, "main");
    let compile = || match key {
        Some(_) => compile_to_metallib(source, label),
        None => Err(RenderError::ShaderCompile("no Metal toolchain".into())),
    };
    match crate::shader::builtin::fetch(label, source, embedded, key.as_ref(), compile) {
        Ok(bytes) => match super::pipeline::load_library(device, &bytes) {
            Ok(library) => Ok(library),
            Err(e) => {
                tracing::warn!("{label}: metallib rejected ({e}), compiling from source");
                source_library(device, source)
            }
        },
        Err(e) => {
            // The compile failed or had no toolchain. The in-process one below
            // either succeeds or surfaces the error the caller expects.
            tracing::debug!("{label}: no metallib ({e}), compiling from source");
            source_library(device, source)
        }
    }
}

// `entry` out of the library a cooked MSL artifact builds. `label` names the
// artifact's owner in errors.
pub(super) fn cooked_function(
    device: &ProtocolObject<dyn MTLDevice>,
    msl: &[u8],
    entry: &str,
    label: &str,
) -> RenderResult<Retained<ProtocolObject<dyn MTLFunction>>> {
    let text = std::str::from_utf8(msl).map_err(|e| {
        RenderError::ShaderCompile(format!("{label}: artifact is not MSL text: {e}"))
    })?;
    let library = compiled_library(device, text, label, None)?;
    library
        .newFunctionWithName(&NSString::from_str(entry))
        .ok_or_else(|| RenderError::ShaderCompile(format!("{label}: {entry} not found")))
}

fn source_library(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
) -> RenderResult<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let options = objc2_metal::MTLCompileOptions::new();
    device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
        .map_err(|e| RenderError::ShaderCompile(format!("{e:?}")))
}

// Compile `source` to metallib bytes in a work directory removed afterwards.
fn compile_to_metallib(source: &str, label: &str) -> RenderResult<Vec<u8>> {
    let work = crate::shader::compiler_work::dir().map_err(RenderError::Other)?;
    concinnity_shader::metallib::compile(source, work.path())
        .map_err(|e| RenderError::ShaderCompile(format!("{label}: {e}")))
}
