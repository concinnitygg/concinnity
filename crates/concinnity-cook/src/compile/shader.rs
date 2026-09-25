//! Compiling a world `Shader`'s two files into every program the backend it
//! cooks for consumes.
//!
//! A world Shader defines hooks the engine's own main-pass entries call, so it
//! compiles as those programs do: each entry in
//! `concinnity_core::render::shader_programs::surface` is assembled from the
//! engine's template with the world's files spliced at the hook markers and
//! compiled for the host's target (see `program`).
//!
//! Shared by the payload build and by `cn debug`'s hot reload, so an edit
//! recompiles through exactly the path the cook took.

use concinnity_core::components::{ShaderPrograms, ShaderSource};
use concinnity_core::platform::Platform;
use concinnity_core::render::shader_programs::surface::{self, SourceFile, Sources};
use concinnity_shader::diagnostics::Diagnostic;

use crate::compile::program::{self, ProgramError};

/// A Shader's compiled programs, and what the compiler warned of.
#[derive(Debug)]
pub struct CompiledShader {
    /// The payload the renderer loads.
    pub programs: ShaderPrograms,
    /// Every located warning, each once, in the order first reported.
    pub warnings: Vec<Diagnostic>,
}

/// Compile every program `platform` consumes from a Shader's files.
///
/// A failure names the Shader and the entries and carries the compiler's own
/// diagnostics: a file with a syntax error, or one missing its hook, has to
/// fail the build here, where the message can point at it, rather than at a
/// renderer's init on someone else's machine.
///
/// Each file's text is fenced by `#line` directives naming its
/// [`SourceFile::path`], so a diagnostic in a Shader's own file carries that
/// path verbatim, and one in the engine's template carries the template's file
/// name (`main_bindless.hlsl`). A caller matches a diagnostic to a file by
/// comparing it with the path it passed here. The path is part of the
/// assembled text and so of every artifact's source digest.
pub fn compile_world_shader(
    name: &str,
    sources: &Sources<'_>,
    platform: Platform,
) -> Result<CompiledShader, ProgramError> {
    let assembled: Vec<String> = surface::ALL
        .iter()
        .map(|program| surface::source(program, platform, sources))
        .collect();
    let jobs: Vec<program::Job<'_>> = surface::ALL
        .iter()
        .zip(&assembled)
        .map(|(program, source)| program::Job {
            file: program.file,
            entry: program.entry,
            source,
        })
        .collect();
    let compiled = program::compile_all(
        &format!("Shader '{name}'"),
        &format!("shader-{name}"),
        &jobs,
        platform,
        hook_hint,
    )?;
    let owned = |file: SourceFile<'_>| ShaderSource {
        path: file.path.to_string(),
        text: file.text.to_string(),
    };
    Ok(CompiledShader {
        programs: ShaderPrograms {
            name: name.to_string(),
            vertex: sources.vertex.map(owned),
            fragment: owned(sources.fragment),
            programs: compiled.programs,
        },
        warnings: compiled.warnings,
    })
}

// A hook the file never defined is a call to an undefined function; every
// other failure (a syntax error, no compiler) carries its own remedy.
fn hook_hint(diagnostic: &str) -> &'static str {
    if diagnostic.contains("found undefined function") {
        "\nA Shader's `fragment` file must define `float4 shade(VertexOut v, \
             GpuObjectData od)` and its `vertex` file, when declared, `VertexOut \
             transform(float4x4 model, float3 pos, float3 normal, float3 tangent, \
             float3 color, float2 uv)`."
    } else {
        ""
    }
}

/// Read one of a Shader's files, naming the path on failure.
pub fn read_shader_source(source_path: &str) -> Result<String, std::io::Error> {
    std::fs::read_to_string(source_path).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("Failed to read shader source '{}': {}", source_path, e),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::render::shader_source;

    const SHADE: &str =
        "float4 shade(VertexOut v, GpuObjectData od) { return float4(1.0, 0.0, 1.0, 1.0); }\n";
    const TRANSFORM: &str = "VertexOut transform(float4x4 model, float3 pos, float3 normal, \
        float3 tangent, float3 color, float2 uv) { return project_vertex(model, pos + \
        float3(0.0, 0.1, 0.0), normal, tangent, color, uv); }\n";

    #[test]
    fn a_read_error_reports_the_path_and_keeps_the_io_error_kind() {
        let err = read_shader_source("/no/such/user.hlsl").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(
            err.to_string()
                .starts_with("Failed to read shader source '/no/such/user.hlsl'"),
            "got: {err}"
        );
    }

    const FRAGMENT_PATH: &str = "shaders/magenta.hlsl";
    const VERTEX_PATH: &str = "shaders/sway.hlsl";

    fn fragment(text: &str) -> SourceFile<'_> {
        SourceFile {
            path: FRAGMENT_PATH,
            text,
        }
    }

    fn fragment_only(text: &str) -> Sources<'_> {
        Sources {
            vertex: None,
            fragment: fragment(text),
        }
    }

    fn with_vertex<'a>(vertex: &'a str, frag: &'a str) -> Sources<'a> {
        Sources {
            vertex: Some(SourceFile {
                path: VERTEX_PATH,
                text: vertex,
            }),
            fragment: fragment(frag),
        }
    }

    fn compile_failure(err: ProgramError) -> program::CompileFailure {
        match err {
            ProgramError::Compile(failed) => failed,
            other => panic!("not a compile failure: {other}"),
        }
    }

    // A fragment-only Shader compiles every program its host consumes, each
    // findable by entry under the digest the renderer will compute, and takes
    // the engine's own projection for the vertex hook.
    #[test]
    fn a_fragment_only_shader_compiles_every_program_of_its_host() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let sources = fragment_only(SHADE);
        for platform in Platform::ALL {
            let compiled = compile_world_shader("magenta", &sources, platform)
                .unwrap_or_else(|e| panic!("{platform:?}: {e}"));
            let programs = compiled.programs;
            assert_eq!(programs.fragment.text, SHADE);
            assert_eq!(programs.fragment.path, FRAGMENT_PATH);
            assert!(programs.vertex.is_none());
            assert!(compiled.warnings.is_empty());
            for program in surface::ALL {
                let source = surface::source(program, platform, &programs.sources());
                let digest = shader_source::source_digest(&source);
                let bytes = programs
                    .artifact(program.entry, digest)
                    .unwrap_or_else(|| panic!("{platform:?}: no artifact for {}", program.entry));
                assert!(!bytes.is_empty());
            }
        }
    }

    // A vertex file replaces the engine's projection in every vertex variant.
    #[test]
    fn a_vertex_file_compiles_into_every_vertex_variant() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let sources = with_vertex(TRANSFORM, SHADE);
        for platform in Platform::ALL {
            let programs = compile_world_shader("sway", &sources, platform)
                .unwrap_or_else(|e| panic!("{platform:?}: {e}"))
                .programs;
            let vertex = programs
                .vertex
                .as_ref()
                .expect("the vertex file rides along");
            assert_eq!(
                (vertex.path.as_str(), vertex.text.as_str()),
                (VERTEX_PATH, TRANSFORM)
            );
            assert_eq!(programs.programs.len(), surface::ALL.len());
        }
    }

    // Each compile runs in its own scratch directory, so a payload that differs
    // between two compiles of the same files has stamped that directory in.
    #[test]
    fn a_shader_payload_is_the_same_on_every_compile() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let sources = with_vertex(TRANSFORM, SHADE);
        for platform in Platform::ALL {
            let compile = || {
                compile_world_shader("sway", &sources, platform)
                    .unwrap_or_else(|e| panic!("{platform:?}: {e}"))
                    .programs
            };
            assert_eq!(compile(), compile(), "{platform:?}");
        }
    }

    // The path names the file in diagnostics and nowhere else: the same text
    // under another path compiles to the same bytes.
    #[test]
    fn the_path_does_not_reach_the_artifact() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let elsewhere = Sources {
            vertex: None,
            fragment: SourceFile {
                path: r"C:\worlds\lake\shaders\magenta.hlsl",
                text: SHADE,
            },
        };
        for platform in Platform::ALL {
            let artifacts = |sources: &Sources<'_>| -> Vec<Vec<u8>> {
                compile_world_shader("magenta", sources, platform)
                    .unwrap_or_else(|e| panic!("{platform:?}: {e}"))
                    .programs
                    .programs
                    .into_iter()
                    .map(|p| p.artifact)
                    .collect()
            };
            assert_eq!(
                artifacts(&fragment_only(SHADE)),
                artifacts(&elsewhere),
                "{platform:?}"
            );
        }
    }

    // An error on a known line of the fragment file is reported at that line
    // of that file, once, although every entry compiled it.
    #[test]
    fn an_error_names_the_authors_file_and_line() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let broken = "// a comment\n\
                      float4 shade(VertexOut v, GpuObjectData od)\n\
                      {\n\
                      \x20   return float4(undeclared_tint, 1.0);\n\
                      }\n";
        let failed = compile_failure(
            compile_world_shader("broken", &fragment_only(broken), Platform::Vulkan).unwrap_err(),
        );
        let errors: Vec<_> = failed.errors().collect();
        assert_eq!(errors.len(), 1, "{failed}");
        assert_eq!(
            (errors[0].path.as_str(), errors[0].line, errors[0].column),
            (FRAGMENT_PATH, 4, 19),
            "{failed}"
        );
        assert!(errors[0].message.contains("undeclared_tint"), "{failed}");
        assert_eq!(failed.failures.len(), surface::ALL.len());
        assert!(
            failed
                .to_string()
                .contains("shaders/magenta.hlsl:4:19: error:"),
            "{failed}"
        );
    }

    // A vertex file's error names the vertex file; the fragment's lines
    // number independently of it.
    #[test]
    fn an_error_in_the_vertex_file_names_the_vertex_file() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let vertex = "VertexOut transform(float4x4 model, float3 pos, float3 normal,\n\
                      \x20   float3 tangent, float3 color, float2 uv)\n\
                      {\n\
                      \x20   return project_vertex(model, pos, normal, tangent, color, uv)\n\
                      }\n";
        let failed = compile_failure(
            compile_world_shader("sway", &with_vertex(vertex, SHADE), Platform::DirectX)
                .unwrap_err(),
        );
        let located: Vec<(&str, u32)> =
            failed.errors().map(|d| (d.path.as_str(), d.line)).collect();
        assert_eq!(located, [(VERTEX_PATH, 4)], "{failed}");
    }

    // A warning fails nothing and comes back located in the author's file,
    // once.
    #[test]
    fn a_warning_is_returned_at_the_authors_line() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let warns = "float4 shade(VertexOut v, GpuObjectData od)\n\
                     {\n\
                     \x20   int truncated = 3.5;\n\
                     \x20   return float4(1.0, 0.0, 1.0, 1.0);\n\
                     }\n";
        let compiled = compile_world_shader("warns", &fragment_only(warns), Platform::Metal)
            .unwrap_or_else(|e| panic!("{e}"));
        let located: Vec<(&str, u32, u32)> = compiled
            .warnings
            .iter()
            .map(|d| (d.path.as_str(), d.line, d.column))
            .collect();
        assert_eq!(located, [(FRAGMENT_PATH, 3, 21)]);
    }

    // A fragment file without `shade` fails naming the Shader and the hook,
    // at build time rather than at a renderer's init.
    #[test]
    fn a_fragment_without_the_hook_fails_naming_it() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let err = compile_world_shader(
            "empty",
            &fragment_only("// nothing here\n"),
            Platform::Vulkan,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.starts_with("Shader 'empty': compiling"), "got: {msg}");
        assert!(msg.contains("shade"), "names the hook: {msg}");
        assert!(msg.contains("must define"), "carries the hook hint: {msg}");
    }

    // A hook with the wrong signature is not an overload of the engine's
    // declaration; the compile fails and the message says which function.
    #[test]
    fn a_hook_with_the_wrong_signature_fails() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let sources = fragment_only("float4 shade(VertexOut v) { return float4(1.0); }\n");
        let err = compile_world_shader("wrong", &sources, Platform::Metal).unwrap_err();
        assert!(err.to_string().contains("shade"), "got: {err}");
    }
}
