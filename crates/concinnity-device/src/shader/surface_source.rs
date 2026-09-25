// Where a world Shader's compiled program comes from.
//
// The cook compiles a Shader's two files into every main-pass entry the host
// consumes and stores what the compiler emitted, so a shipped player needs no
// shader compiler. This resolves that: the stored artifact when the engine
// template it was built against still matches, and a compile here when it does
// not.
//
// The mismatch case is not an error path. It is what makes editing
// `main_shading.hlsl` with a world Shader loaded possible at all: a hot-reload
// build assembles from the checkout, digests differently, and recompiles. A
// machine with no compiler says so, naming the Shader, rather than drawing
// nothing. The sibling of `raymarch_source`.

use concinnity_core::components::ShaderPrograms;
use concinnity_core::platform::Platform;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::shader_programs::surface::{self, Sources};
use concinnity_core::render::shader_source;
use std::borrow::Cow;

/// Which host asks, and whether it assembles from the checkout.
#[derive(Clone, Copy)]
pub(crate) struct Request {
    pub platform: Platform,
    pub hot_reload: bool,
}

/// The artifact holding `entry`: the cook's when the engine template it was
/// built against still matches, and `compile` of the assembled source when it
/// does not.
///
/// `compile` takes the host, the file, the entry and the assembled source, and
/// must emit what the cook emitted for this host, or a fallback compile would
/// produce something the renderer cannot load. Each backend passes
/// `shader::compile::cooked`, which picks the target the way the cook does.
pub(crate) fn artifact<'a>(
    programs: &'a ShaderPrograms,
    entry: &str,
    req: &Request,
    compile: impl FnOnce(Platform, &str, &str, &str) -> RenderResult<Vec<u8>>,
) -> RenderResult<Cow<'a, [u8]>> {
    let label = programs.name.as_str();
    let program = surface::program(entry).ok_or_else(|| {
        RenderError::ShaderCompile(format!(
            "Shader '{label}': no main-pass entry named '{entry}'"
        ))
    })?;
    let source = source(program, req, &programs.sources());
    let digest = shader_source::source_digest(&source);
    if let Some(bytes) = programs.artifact(entry, digest) {
        return Ok(Cow::Borrowed(bytes));
    }
    tracing::debug!("Shader '{label}': {entry} predates the engine template, compiling");
    compile(req.platform, program.file, program.entry, &source)
        .map(Cow::Owned)
        .map_err(|e| e.context(format_args!("Shader '{label}': compiling '{entry}'")))
}

// The source text this host expects for one entry, preferring the checkout's
// templates under hot-reload exactly as every other single-source shader does.
fn source(program: &surface::Program, req: &Request, sources: &Sources<'_>) -> String {
    if !req.hot_reload {
        return surface::source(program, req.platform, sources);
    }
    surface::source_with(
        program,
        req.platform,
        sources,
        crate::shader::source::from_checkout,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::ShaderSource;
    use concinnity_core::components::compiled_programs::CompiledProgram;

    const SHADE: &str = "float4 shade(VertexOut v, GpuObjectData od) { return (float4)(1.0); }";

    fn request(platform: Platform) -> Request {
        Request {
            platform,
            hot_reload: false,
        }
    }

    fn no_compile(_: Platform, _: &str, _: &str, _: &str) -> RenderResult<Vec<u8>> {
        panic!("a matching artifact reached the compiler")
    }

    fn stored(platform: Platform, entry: &str, bytes: &[u8]) -> ShaderPrograms {
        let mut programs = ShaderPrograms {
            name: "wall".to_string(),
            vertex: None,
            fragment: ShaderSource {
                path: "shaders/wall.hlsl".to_string(),
                text: SHADE.to_string(),
            },
            programs: Vec::new(),
        };
        let program = surface::program(entry).unwrap();
        let src = surface::source(program, platform, &programs.sources());
        programs.programs.push(CompiledProgram {
            entry: entry.to_string(),
            source_digest: shader_source::source_digest(&src),
            artifact: bytes.to_vec(),
        });
        programs
    }

    // The stored artifact is taken whenever the template still matches, which
    // is the shipped path and the one that must never reach a compiler.
    #[test]
    fn a_matching_artifact_is_taken_without_compiling() {
        for entry in ["vertex_main_bindless", "fragment_main_bindless"] {
            let programs = stored(Platform::Metal, entry, b"stored bytes");
            let got =
                artifact(&programs, entry, &request(Platform::Metal), no_compile).expect("stored");
            assert_eq!(got.as_ref(), b"stored bytes");
            assert!(matches!(got, Cow::Borrowed(_)), "no compile was needed");
        }
    }

    // An artifact built for another host is not this one's: the digest covers
    // the backend define every source leads with.
    #[test]
    fn an_artifact_from_another_host_does_not_match() {
        let metal = stored(Platform::Metal, "fragment_main_bindless", b"stored bytes");
        let program = surface::program("fragment_main_bindless").unwrap();
        let other_host = surface::source(program, Platform::Vulkan, &metal.sources());
        assert!(
            metal
                .artifact(
                    "fragment_main_bindless",
                    shader_source::source_digest(&other_host)
                )
                .is_none()
        );
    }

    // An entry the table does not name is a caller bug, reported by name
    // rather than compiled into nothing.
    #[test]
    fn an_unknown_entry_is_an_error() {
        let programs = stored(Platform::Metal, "fragment_main_bindless", b"x");
        let err = artifact(
            &programs,
            "no_such_entry",
            &request(Platform::Metal),
            no_compile,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no_such_entry"), "got: {err}");
    }

    // A miss compiles the entry's own program from the source this host
    // assembles, and a failure names the Shader and the entry.
    #[test]
    fn a_stale_artifact_compiles_its_entry_and_names_the_shader() {
        let mut programs = stored(Platform::Metal, "fragment_main_bindless", b"stale");
        programs.programs[0].source_digest ^= 1;
        let program = surface::program("fragment_main_bindless").unwrap();
        let want = surface::source(program, Platform::Metal, &programs.sources());
        let req = request(Platform::Metal);
        let got = artifact(
            &programs,
            "fragment_main_bindless",
            &req,
            |platform, file, entry, src| {
                assert_eq!(platform, Platform::Metal);
                assert_eq!(
                    (file, entry),
                    ("main_bindless.hlsl", "fragment_main_bindless")
                );
                assert_eq!(src, want);
                Ok(b"fresh".to_vec())
            },
        )
        .expect("compiled");
        assert_eq!(got.as_ref(), b"fresh");

        let err = artifact(&programs, "fragment_main_bindless", &req, |_, _, _, _| {
            Err(RenderError::ShaderCompile("no compiler".to_string()))
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("Shader 'wall'"), "got: {err}");
        assert!(err.contains("fragment_main_bindless"), "got: {err}");
        assert!(err.contains("no compiler"), "got: {err}");
    }
}
