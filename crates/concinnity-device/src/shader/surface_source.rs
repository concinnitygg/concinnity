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
use concinnity_core::render::backend::error::RenderResult;
use concinnity_core::render::shader::programs::surface::{self, Sources};
use concinnity_core::render::shader::source;
use std::borrow::Cow;

/// Which host asks, and whether it assembles from the checkout.
#[derive(Clone, Copy)]
pub(crate) struct Request {
    pub platform: Platform,
    pub hot_reload: bool,
}

/// The artifact holding `program`: the cook's when the engine template it was
/// built against still matches, and `compile` of the assembled source when it
/// does not.
///
/// `compile` takes the host, the file, the entry and the assembled source, and
/// must emit what the cook emitted for this host, or a fallback compile would
/// produce something the renderer cannot load. Each backend passes
/// `shader::compile::cooked`, which picks the target the way the cook does.
pub(crate) fn artifact<'a>(
    programs: &'a ShaderPrograms,
    program: &surface::Program,
    req: Request,
    compile: impl FnOnce(Platform, &str, &str, &str) -> RenderResult<Vec<u8>>,
) -> RenderResult<Cow<'a, [u8]>> {
    let label = programs.name.as_str();
    let entry = program.entry;
    let source = source(program, req, &programs.sources());
    let digest = source::source_digest(&source);
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
fn source(program: &surface::Program, req: Request, sources: &Sources<'_>) -> String {
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
    use concinnity_core::components::assets::compiled_programs::CompiledProgram;
    use concinnity_core::render::backend::error::RenderError;

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

    fn stored(platform: Platform, program: &surface::Program, bytes: &[u8]) -> ShaderPrograms {
        let mut programs = ShaderPrograms {
            name: "wall".to_string(),
            vertex: None,
            fragment: ShaderSource {
                path: "shaders/wall.hlsl".to_string(),
                text: SHADE.to_string(),
            },
            programs: Vec::new(),
        };
        let src = surface::source(program, platform, &programs.sources());
        programs.programs.push(CompiledProgram {
            entry: program.entry.to_string(),
            source_digest: source::source_digest(&src),
            artifact: bytes.to_vec(),
        });
        programs
    }

    // The stored artifact is taken whenever the template still matches, which
    // is the shipped path and the one that must never reach a compiler.
    #[test]
    fn a_matching_artifact_is_taken_without_compiling() {
        for program in surface::ALL {
            let programs = stored(Platform::Metal, program, b"stored bytes");
            let got =
                artifact(&programs, program, request(Platform::Metal), no_compile).expect("stored");
            assert_eq!(got.as_ref(), b"stored bytes");
            assert!(matches!(got, Cow::Borrowed(_)), "no compile was needed");
        }
    }

    // An artifact built for another host is not this one's: the digest covers
    // the backend define every source leads with.
    #[test]
    fn an_artifact_from_another_host_does_not_match() {
        let metal = stored(Platform::Metal, &surface::MAIN_FRAGMENT, b"stored bytes");
        let other_host =
            surface::source(&surface::MAIN_FRAGMENT, Platform::Vulkan, &metal.sources());
        assert!(
            metal
                .artifact("fragment_main_bindless", source::source_digest(&other_host))
                .is_none()
        );
    }

    // The canopy fragment shares the plain fragment's entry name; its gate
    // gives it a digest of its own, so neither takes the other's artifact.
    #[test]
    fn the_canopy_fragment_never_takes_the_plain_artifact() {
        let programs = stored(Platform::Metal, &surface::MAIN_FRAGMENT, b"plain");
        let req = request(Platform::Metal);
        let got = artifact(
            &programs,
            &surface::MAIN_FRAGMENT_CANOPY,
            req,
            |_, _, _, src| {
                assert!(src.contains("#define SURFACE_CANOPY 1"));
                Ok(b"canopy".to_vec())
            },
        )
        .expect("compiled");
        assert_eq!(got.as_ref(), b"canopy");
        let plain = artifact(&programs, &surface::MAIN_FRAGMENT, req, no_compile).expect("stored");
        assert_eq!(plain.as_ref(), b"plain");
    }

    // A miss compiles the entry's own program from the source this host
    // assembles, and a failure names the Shader and the entry.
    #[test]
    fn a_stale_artifact_compiles_its_entry_and_names_the_shader() {
        let mut programs = stored(Platform::Metal, &surface::MAIN_FRAGMENT, b"stale");
        programs.programs[0].source_digest ^= 1;
        let want = surface::source(
            &surface::MAIN_FRAGMENT,
            Platform::Metal,
            &programs.sources(),
        );
        let req = request(Platform::Metal);
        let got = artifact(
            &programs,
            &surface::MAIN_FRAGMENT,
            req,
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

        let err = artifact(&programs, &surface::MAIN_FRAGMENT, req, |_, _, _, _| {
            Err(RenderError::ShaderCompile("no compiler".to_string()))
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("Shader 'wall'"), "got: {err}");
        assert!(err.contains("fragment_main_bindless"), "got: {err}");
        assert!(err.contains("no compiler"), "got: {err}");
    }
}
