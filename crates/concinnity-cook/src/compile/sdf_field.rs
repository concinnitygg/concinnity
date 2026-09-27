//! Compiling an `SdfVolume`'s authored distance field into every entry the
//! volume draws with.
//!
//! Every other engine shader is a build-time artifact of the engine itself.
//! This one is only complete once a world authors the field that goes in the
//! middle of the template, so it compiles here, where a shipped player never
//! needs a compiler of its own. Each family's source is assembled with the
//! field spliced at its marker and compiled for the host's target (see
//! `program`).
//!
//! Shared by the payload build and by `cn debug`'s hot reload, so an edit
//! recompiles through exactly the path the cook took.

use concinnity_core::components::ShaderSource;
use concinnity_core::components::sdf_programs::SdfPrograms;
use concinnity_core::platform::Platform;
use concinnity_core::render::shader_programs::raymarch::{self, Family};
use concinnity_core::render::shader_source::SourceFile;
use concinnity_shader::diagnostics::Diagnostic;

use crate::compile::program::{self, ProgramError};

/// A field's compiled programs, and what the compiler warned of.
#[derive(Debug)]
pub struct CompiledField {
    /// The payload the renderer loads.
    pub programs: SdfPrograms,
    /// Every located warning, each once, in the order first reported.
    pub warnings: Vec<Diagnostic>,
}

/// Compile every entry a volume with these flags draws with, from `field`.
///
/// A failure names the volume and the entries and carries the compiler's own
/// diagnostics: a field with a syntax error, or one missing a function the
/// template calls, has to fail the build here, where the message can point at
/// it, rather than at a renderer's init on someone else's machine.
///
/// The field's text is fenced by `#line` directives naming its
/// [`SourceFile::path`], so a diagnostic in the field carries that path
/// verbatim and one in the engine's template carries `raymarch.hlsl`. A
/// function the template calls with the wrong signature errors in the
/// template, and the note under it names the field's own line. The path is
/// part of the assembled text and so of every artifact's source digest.
pub fn compile_sdf_field(
    name: &str,
    field: SourceFile<'_>,
    platform: Platform,
    volumetric: bool,
    cast_shadows: bool,
) -> Result<CompiledField, ProgramError> {
    let sources: Vec<(Family, String)> = raymarch::families(volumetric, cast_shadows)
        .map(|family| (family, raymarch::source(family, platform, field)))
        .collect();
    let jobs: Vec<program::Job<'_>> = sources
        .iter()
        .flat_map(|(family, source)| {
            raymarch::ALL
                .iter()
                .filter(move |p| p.family == *family)
                .map(move |p| program::Job {
                    file: raymarch::FILE,
                    entry: p.entry,
                    source,
                })
        })
        .collect();
    let compiled = program::compile_all(
        &format!("SdfVolume '{name}'"),
        &format!("sdf-{name}"),
        &jobs,
        platform,
        field_hint,
    )?;
    Ok(CompiledField {
        programs: SdfPrograms {
            field: ShaderSource {
                path: field.path.to_string(),
                text: field.text.to_string(),
            },
            programs: compiled.programs,
        },
        warnings: compiled.warnings,
    })
}

// A function the field never defined is a call to an undefined function; every
// other failure (a syntax error, no compiler) carries its own remedy.
fn field_hint(diagnostic: &str) -> &'static str {
    if diagnostic.contains("found undefined function") {
        "\nAn SdfVolume's field must define `float map(float3 p, SdfParams params, \
             float time)` and `SdfSurface shade(float3 p, float3 normal, SdfParams \
             params, float time, float2 frag_uv)`, or `VolumeSample sampleVolume(float3 \
             p, SdfParams params, float time)` for a volumetric volume."
    } else {
        ""
    }
}

#[cfg(test)]
mod tests;
