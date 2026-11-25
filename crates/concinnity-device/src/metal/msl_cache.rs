// Disk-cached compilation for MSL: world Shader and SdfVolume programs, and the
// cull encode kernel when no export shipped it. A cold launch compiles through
// `concinnity_shader::metallib` and stores the metallib content-addressed; a
// warm launch loads it directly.
// Without the toolchain every path uses `newLibraryWithSource`.
//
// The toolchain's id is part of the key, since a metallib from a superseded
// toolchain still loads.

use concinnity_engine::render::backend::error::{RenderError, RenderResult};
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

// Produce the MTLLibrary for `source`: `shipped` when it was compiled from this
// exact text, else a cached or fresh metallib, else an in-process source
// compile. `label` names the shader in cache-miss logs and compile errors.
pub(super) fn compiled_library(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
    label: &str,
    shipped: Option<crate::shader::shipped::Shipped>,
) -> RenderResult<Retained<ProtocolObject<dyn MTLLibrary>>> {
    match metallib_bytes(source, label, shipped) {
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

// The metallib for `source`: `shipped` when it was compiled from this exact
// text, else the cached bytes, else a fresh compile stored into the cache.
fn metallib_bytes(
    source: &str,
    label: &str,
    shipped: Option<crate::shader::shipped::Shipped>,
) -> RenderResult<std::borrow::Cow<'static, [u8]>> {
    let key = metallib_key(source, "main");
    let compile = || match key {
        Some(_) => compile_to_metallib(source, label),
        None => Err(RenderError::ShaderCompile("no Metal toolchain".into())),
    };
    crate::shader::builtin::fetch(label, source, shipped, key.as_ref(), compile)
}

fn msl_text<'a>(msl: &'a [u8], label: &str) -> RenderResult<&'a str> {
    std::str::from_utf8(msl)
        .map_err(|e| RenderError::ShaderCompile(format!("{label}: artifact is not MSL text: {e}")))
}

// `entry` out of the library a cooked MSL artifact builds. `label` names the
// artifact's owner in errors.
pub(super) fn cooked_function(
    device: &ProtocolObject<dyn MTLDevice>,
    msl: &[u8],
    entry: &str,
    label: &str,
) -> RenderResult<Retained<ProtocolObject<dyn MTLFunction>>> {
    let library = compiled_library(device, msl_text(msl, label)?, label, None)?;
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
pub(crate) fn compile_to_metallib(source: &str, label: &str) -> RenderResult<Vec<u8>> {
    let work = crate::shader::compiler_work::dir().map_err(RenderError::Other)?;
    concinnity_shader::metallib::compile(source, work.path())
        .map_err(|e| RenderError::ShaderCompile(format!("{label}: {e}")))
}
