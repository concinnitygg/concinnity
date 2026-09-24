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

use concinnity_core::components::ShaderPrograms;
use concinnity_core::platform::Platform;
use concinnity_core::render::shader_programs::surface::{self, Sources};

use crate::compile::program;

/// Compile every program `platform` consumes from a Shader's files.
///
/// A failure names the Shader and the entry and carries the compiler's own
/// diagnostic: a file with a syntax error, or one missing its hook, has to
/// fail the build here, where the message can point at it, rather than at a
/// renderer's init on someone else's machine.
pub fn compile_world_shader(
    name: &str,
    sources: &Sources<'_>,
    platform: Platform,
) -> std::io::Result<ShaderPrograms> {
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
    let programs = program::compile_all(
        &format!("Shader '{name}'"),
        &format!("shader-{name}"),
        &jobs,
        platform,
        hook_hint,
    )?;
    Ok(ShaderPrograms {
        name: name.to_string(),
        vertex: sources.vertex.map(str::to_string),
        fragment: sources.fragment.to_string(),
        programs,
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

    // A fragment-only Shader compiles every program its host consumes, each
    // findable by entry under the digest the renderer will compute, and takes
    // the engine's own projection for the vertex hook.
    #[test]
    fn a_fragment_only_shader_compiles_every_program_of_its_host() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let sources = Sources {
            vertex: None,
            fragment: SHADE,
        };
        for platform in Platform::ALL {
            let programs = compile_world_shader("magenta", &sources, platform)
                .unwrap_or_else(|e| panic!("{platform:?}: {e}"));
            assert_eq!(programs.fragment, SHADE);
            assert!(programs.vertex.is_none());
            for program in surface::ALL {
                let source = surface::source(program, platform, &sources);
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
        let sources = Sources {
            vertex: Some(TRANSFORM),
            fragment: SHADE,
        };
        for platform in Platform::ALL {
            let programs = compile_world_shader("sway", &sources, platform)
                .unwrap_or_else(|e| panic!("{platform:?}: {e}"));
            assert_eq!(programs.vertex.as_deref(), Some(TRANSFORM));
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
        let sources = Sources {
            vertex: Some(TRANSFORM),
            fragment: SHADE,
        };
        for platform in Platform::ALL {
            let compile = || {
                compile_world_shader("sway", &sources, platform)
                    .unwrap_or_else(|e| panic!("{platform:?}: {e}"))
            };
            assert_eq!(compile(), compile(), "{platform:?}");
        }
    }

    // A fragment file without `shade` fails naming the Shader and the hook,
    // at build time rather than at a renderer's init.
    #[test]
    fn a_fragment_without_the_hook_fails_naming_it() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let sources = Sources {
            vertex: None,
            fragment: "// nothing here\n",
        };
        let err = compile_world_shader("empty", &sources, Platform::Vulkan).unwrap_err();
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
        let sources = Sources {
            vertex: None,
            fragment: "float4 shade(VertexOut v) { return float4(1.0); }\n",
        };
        let err = compile_world_shader("wrong", &sources, Platform::Metal).unwrap_err();
        assert!(err.to_string().contains("shade"), "got: {err}");
    }
}
