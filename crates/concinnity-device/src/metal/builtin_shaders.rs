// The Metal half of the single-source shader compile. The fast path loads the
// metallib an export shipped; otherwise a compile, cached per machine.
#![deny(unsafe_op_in_unsafe_fn)]

use std::borrow::Cow;

use concinnity_core::platform::Platform;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
pub(super) use concinnity_engine::render::shader::programs::ShaderProgram;
use concinnity_engine::render::shader::programs::Variant;
pub(super) use concinnity_engine::render::shader::programs::metal::{
    HIZ_DOWNSAMPLE, HIZ_INIT_MSAA, HIZ_INIT_SINGLE,
};
pub(super) use concinnity_engine::render::shader::programs::shared::*;
use concinnity_shader::HlslTarget;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLFunction, MTLLibrary};

use super::pipeline::{load_library, ns_str};
use crate::shader::shipped::{Compile, Planned};

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

// `program` as a player fetches it: its artifact name, the source it compiles
// and how.
fn planned(program: &ShaderProgram, hot_reload: bool) -> Planned {
    Planned {
        name: variant(program).artifact_name(),
        source: source(program, hot_reload),
        compile: Compile::Hlsl {
            file: program.file,
            entry: program.entry,
            target: HlslTarget::Metallib,
        },
    }
}

// `program`'s MTLLibrary, from the metallib shipped, cached or compiled (see
// `shader::builtin`).
fn library(
    device: &ProtocolObject<dyn MTLDevice>,
    program: &ShaderProgram,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let planned = planned(program, hot_reload);
    let key = super::msl_cache::metallib_key(&planned.source, program.entry);
    let bytes = crate::shader::builtin::fetch(
        &planned.name,
        &planned.source,
        planned.shipped(Platform::Metal),
        key.as_ref(),
        || planned.compile(),
    )?;
    let origin = match bytes {
        Cow::Borrowed(_) => "shipped metallib",
        Cow::Owned(_) => "compiled metallib",
    };
    load_library(device, &bytes).map_err(|e| e.context(format_args!("{}: {origin}", planned.name)))
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

// Every Metal program and the cull encode kernel as a player fetches them, for
// an export to ship.
pub(crate) fn shipped_plan() -> Vec<Planned> {
    use concinnity_engine::render::shader::programs::metal::TABLE;
    TABLE
        .programs()
        .map(|program| planned(program, false))
        .chain(std::iter::once(super::pipeline::cull_encode_planned(false)))
        .collect()
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

    // The plan an export compiles is every program at the variant a player
    // fetches, plus the cull encode kernel, under the names and sources the
    // runtime looks them up by.
    #[test]
    fn the_shipped_plan_is_what_a_player_fetches() {
        use concinnity_engine::render::shader::programs::metal::TABLE;
        use concinnity_engine::render::shader::source::source_digest;
        let plan: Vec<(String, u64)> = shipped_plan()
            .iter()
            .map(|p| (p.name.clone(), source_digest(&p.source)))
            .collect();
        let mut expected: Vec<(String, u64)> = TABLE
            .programs()
            .map(|p| (variant(p).artifact_name(), source_digest(&source(p, false))))
            .collect();
        expected.push((
            super::super::pipeline::CULL_ENCODE.to_string(),
            source_digest(&super::super::pipeline::cull_encode_source(false)),
        ));
        assert_eq!(plan, expected);
        let mut names: Vec<&str> = plan.iter().map(|(name, _)| name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), plan.len(), "two artifacts share a name");
    }

    // Every variant assembles non-empty source with its defines up front in
    // both hot-reload modes, so embedded and disk agree on existence.
    #[test]
    fn variants_assemble_source_with_their_defines() {
        use concinnity_engine::render::shader::programs::metal::TABLE;
        assert_eq!(
            TABLE.msaa,
            [false],
            "an export ships what `variant` asks for"
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
