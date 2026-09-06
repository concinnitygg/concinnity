//! Compiling a world `Shader`'s two files into every program the backend it
//! cooks for consumes.
//!
//! A world Shader defines hooks the engine's own main-pass entries call, so it
//! compiles as those programs do: each entry in
//! `concinnity_core::render::slang_programs::surface` is assembled from the
//! engine's template with the world's files spliced at the hook markers and
//! handed to slangc for the host's target. What is emitted is what that
//! backend's renderer loads directly: SPIR-V for Vulkan, a signed DXIL
//! container for D3D12, and MSL text for Metal, whose `newLibraryWithSource`
//! is an always-present OS API. Each artifact carries the digest of the source
//! it came from, so a renderer whose template has moved compiles instead of
//! loading something stale.
//!
//! Shared by the payload build and by `cn debug`'s hot reload, so an edit
//! recompiles through exactly the path the cook took.

use concinnity_core::components::ShaderPrograms;
use concinnity_core::components::compiled_programs::CompiledProgram;
use concinnity_core::platform::Platform;
use concinnity_core::render::slang_programs::surface::{self, Sources, Stage};
use concinnity_core::render::slang_source;
use concinnity_core::render::uniforms::{BINDLESS_POOL_SIZE, MAX_PROBES};
use concinnity_slang::{SlangJob, SlangTarget};

// What slangc emits for a host, and for a stage where the target needs one.
// DXIL is the only one that takes a profile: it sets the container's feature
// floor, which no target flag implies.
fn target(platform: Platform, stage: Stage) -> SlangTarget {
    match platform {
        Platform::Metal => SlangTarget::Metal,
        Platform::Glsl => SlangTarget::Spirv,
        Platform::Hlsl => SlangTarget::Dxil(match stage {
            Stage::Vertex => "vs_6_0",
            Stage::Fragment => "ps_6_0",
        }),
    }
}

/// Compile every program `platform` consumes from a Shader's files.
///
/// A failure names the Shader and the entry and carries slangc's own
/// diagnostic: a file with a syntax error, or one missing its hook, has to
/// fail the build here, where the message can point at it, rather than at a
/// renderer's init on someone else's machine.
pub fn compile_world_shader(
    name: &str,
    sources: &Sources<'_>,
    platform: Platform,
) -> std::io::Result<ShaderPrograms> {
    let work = concinnity_host::scratch::Scratch::dir(&format!("shader-{name}"))?;
    let mut programs = Vec::new();
    for group in surface::groups(platform) {
        let source = surface::source(group[0], platform, BINDLESS_POOL_SIZE, MAX_PROBES, sources);
        let entries: Vec<&str> = group.iter().map(|p| p.entry).collect();
        let job = SlangJob {
            source: &source,
            file_name: group[0].file,
            entries: &entries,
            target: target(platform, group[0].stage),
        };
        let artifact = concinnity_slang::compile(&job, work.path()).map_err(|e| {
            // A hook the file never defined fails at link; every other failure
            // (a syntax error, no slangc) carries its own remedy.
            let hint = if e.contains("unresolved external symbol") {
                "\nA Shader's `fragment` file must define `float4 shade(VertexOut in, \
                 GpuObjectData od)` and its `vertex` file, when declared, `VertexOut \
                 transform(float4x4 model, float3 pos, float3 normal, float3 tangent, \
                 float3 color, float2 uv)`."
            } else {
                ""
            };
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Shader '{name}': compiling '{}': {e}{hint}",
                    entries.join(", ")
                ),
            )
        })?;
        programs.push(CompiledProgram {
            entries: entries.iter().map(|e| e.to_string()).collect(),
            source_digest: slang_source::source_digest(&source),
            artifact,
        });
    }
    Ok(ShaderPrograms {
        name: name.to_string(),
        vertex: sources.vertex.map(str::to_string),
        fragment: sources.fragment.to_string(),
        programs,
    })
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

// The entry an artifact of `program`'s group is looked up under: the group's
// first entry, which is every entry on the hosts that split them.
#[cfg(test)]
fn first_entry(platform: Platform, program: &surface::Program) -> &'static str {
    surface::groups(platform)
        .into_iter()
        .find(|g| g.iter().any(|p| p.entry == program.entry))
        .map(|g| g[0].entry)
        .expect("every program is in one group")
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHADE: &str =
        "float4 shade(VertexOut in, GpuObjectData od) { return float4(1.0, 0.0, 1.0, 1.0); }\n";
    const TRANSFORM: &str = "VertexOut transform(float4x4 model, float3 pos, float3 normal, \
        float3 tangent, float3 color, float2 uv) { return project_vertex(model, pos + \
        float3(0.0, 0.1, 0.0), normal, tangent, color, uv); }\n";

    // The hosts a compile test can run on here: SPIR-V and MSL text need only
    // slangc, a DXIL container needs dxcompiler beside it, which the vendored
    // Windows release carries and the others do not.
    fn hosts() -> Vec<Platform> {
        let mut hosts = vec![Platform::Metal, Platform::Glsl];
        if cfg!(windows) {
            hosts.push(Platform::Hlsl);
        }
        hosts
    }

    #[test]
    fn each_host_emits_what_its_renderer_consumes() {
        assert_eq!(target(Platform::Metal, Stage::Fragment), SlangTarget::Metal);
        assert_eq!(target(Platform::Glsl, Stage::Vertex), SlangTarget::Spirv);
        assert_eq!(
            target(Platform::Hlsl, Stage::Vertex),
            SlangTarget::Dxil("vs_6_0")
        );
        assert_eq!(
            target(Platform::Hlsl, Stage::Fragment),
            SlangTarget::Dxil("ps_6_0")
        );
    }

    #[test]
    fn a_read_error_reports_the_path_and_keeps_the_io_error_kind() {
        let err = read_shader_source("/no/such/user.slang").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(
            err.to_string()
                .starts_with("Failed to read shader source '/no/such/user.slang'"),
            "got: {err}"
        );
    }

    // A fragment-only Shader compiles every program its host consumes, each
    // findable by entry under the digest the renderer will compute, and takes
    // the engine's own projection for the vertex hook.
    #[test]
    fn a_fragment_only_shader_compiles_every_program_of_its_host() {
        if !crate::slangc_gate::slangc_available() {
            return;
        }
        let sources = Sources {
            vertex: None,
            fragment: SHADE,
        };
        for platform in hosts() {
            let programs = compile_world_shader("magenta", &sources, platform)
                .unwrap_or_else(|e| panic!("{platform:?}: {e}"));
            assert_eq!(programs.fragment, SHADE);
            assert!(programs.vertex.is_none());
            for program in surface::programs(platform) {
                let source =
                    surface::source(program, platform, BINDLESS_POOL_SIZE, MAX_PROBES, &sources);
                let digest = slang_source::source_digest(&source);
                let bytes = programs
                    .artifact(program.entry, digest)
                    .unwrap_or_else(|| panic!("{platform:?}: no artifact for {}", program.entry));
                assert!(!bytes.is_empty());
                assert_eq!(
                    programs.artifact(first_entry(platform, program), digest),
                    Some(bytes)
                );
            }
        }
    }

    // A vertex file replaces the engine's projection in every vertex variant.
    #[test]
    fn a_vertex_file_compiles_into_every_vertex_variant() {
        if !crate::slangc_gate::slangc_available() {
            return;
        }
        let sources = Sources {
            vertex: Some(TRANSFORM),
            fragment: SHADE,
        };
        for platform in hosts() {
            let programs = compile_world_shader("sway", &sources, platform)
                .unwrap_or_else(|e| panic!("{platform:?}: {e}"));
            assert_eq!(programs.vertex.as_deref(), Some(TRANSFORM));
            assert_eq!(programs.programs.len(), surface::groups(platform).len());
        }
    }

    // A fragment file without `shade` fails naming the Shader and the hook,
    // at build time rather than at a renderer's init.
    #[test]
    fn a_fragment_without_the_hook_fails_naming_it() {
        if !crate::slangc_gate::slangc_available() {
            return;
        }
        let sources = Sources {
            vertex: None,
            fragment: "// nothing here\n",
        };
        let err = compile_world_shader("empty", &sources, Platform::Glsl).unwrap_err();
        let msg = err.to_string();
        assert!(msg.starts_with("Shader 'empty': compiling"), "got: {msg}");
        assert!(msg.contains("shade"), "names the hook: {msg}");
    }

    // A hook with the wrong signature is not an overload of the engine's
    // declaration; the compile fails and the message says which function.
    #[test]
    fn a_hook_with_the_wrong_signature_fails() {
        if !crate::slangc_gate::slangc_available() {
            return;
        }
        let sources = Sources {
            vertex: None,
            fragment: "float4 shade(VertexOut in) { return float4(1.0); }\n",
        };
        let err = compile_world_shader("wrong", &sources, Platform::Metal).unwrap_err();
        assert!(err.to_string().contains("shade"), "got: {err}");
    }
}
