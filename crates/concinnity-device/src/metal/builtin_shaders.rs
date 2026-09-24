// The Metal half of the single-source shader compile: everything the
// declarations in `concinnity_core::render::shader_programs::metal::TABLE` need
// a compiler, a content-addressed cache, or a filesystem for, over the
// declarations it brings into the backend's scope.
//
// The fast path loads the metallib the build script precompiled; source
// compilation remains for hot-reload (disk edits must win) and for binaries
// built on a host without the shader toolchain, whose embedded lookup misses
// these names. The runtime compile caches the metallib in the content-addressed
// shader cache, so a given source text compiles at most once per machine.
#![deny(unsafe_op_in_unsafe_fn)]

use std::borrow::Cow;

use concinnity_core::platform::Platform;
use concinnity_core::render::error::{RenderError, RenderResult};
pub(super) use concinnity_core::render::shader_programs::ShaderProgram;
use concinnity_core::render::shader_programs::Variant;
pub(super) use concinnity_core::render::shader_programs::metal::{
    HIZ_DOWNSAMPLE, HIZ_INIT_MSAA, HIZ_INIT_SINGLE,
};
pub(super) use concinnity_core::render::shader_programs::shared::*;
use concinnity_shader::HlslTarget;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLFunction, MTLLibrary};

use super::pipeline::{load_library, ns_str};

// The one variant of each program this backend compiles: every depth reader
// reads the resolved single-sample depth here, so each compiles once, at the
// single sample count.
fn variant(program: &ShaderProgram) -> Variant<'_> {
    program.at(false)
}

// The exact source text `program` compiles, assembled the way every backend
// assembles it.
fn source(program: &ShaderProgram, hot_reload: bool) -> String {
    crate::shader::source::assemble_variant(hot_reload, Platform::Metal, variant(program))
}

// `program`'s MTLLibrary, from the metallib embedded, cached or compiled (see
// `shader::builtin`).
fn library(
    device: &ProtocolObject<dyn MTLDevice>,
    program: &ShaderProgram,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let source = source(program, hot_reload);
    let name = variant(program).artifact_name();
    let key = super::msl_cache::metallib_key(&source, program.entry);
    let bytes = crate::shader::builtin::fetch(
        &name,
        &source,
        super::metallib::embedded_metallib(&name),
        key.as_ref(),
        || {
            crate::shader::compile::compile(
                program.file,
                program.entry,
                &source,
                HlslTarget::Metallib,
            )
        },
    )?;
    let origin = match bytes {
        Cow::Borrowed(_) => "precompiled metallib",
        Cow::Owned(_) => "compiled metallib",
    };
    load_library(device, &bytes).map_err(|e| e.context(format_args!("{name}: {origin}")))
}

// A single-entry variant's function, ready for a pipeline descriptor. Every
// variant compiles on its own, so a two-stage pipeline takes its vertex and its
// fragment from separate libraries and the two pair by semantic.
pub(super) fn entry_function(
    device: &ProtocolObject<dyn MTLDevice>,
    program: &ShaderProgram,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLFunction>>> {
    let library = library(device, program, hot_reload)?;
    library
        .newFunctionWithName(&ns_str(program.entry))
        .ok_or_else(|| {
            RenderError::ShaderCompile(format!("{} not found in {}", program.entry, program.label))
        })
}

// A compute program's pipeline state.
pub(super) fn compute_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    program: &ShaderProgram,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    let function = entry_function(device, program, hot_reload)?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| RenderError::ShaderCompile(format!("{} pipeline: {e:?}", program.label)))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every variant assembles non-empty source with its defines up front in
    // both hot-reload modes, so embedded and disk agree on existence.
    #[test]
    fn variants_assemble_source_with_their_defines() {
        use concinnity_core::render::shader_programs::metal::TABLE;
        assert_eq!(
            TABLE.msaa,
            [false],
            "the build script embeds what `variant` asks for"
        );
        for program in TABLE.programs() {
            for hot_reload in [false, true] {
                let src = source(program, hot_reload);
                assert!(!src.trim().is_empty(), "{}: empty source", program.label);
                for (k, v) in variant(program).defines() {
                    assert!(
                        src.starts_with('#') && src.contains(&format!("#define {k} {v}\n")),
                        "{}: missing injected define {k}",
                        program.label
                    );
                }
            }
        }
    }
}
