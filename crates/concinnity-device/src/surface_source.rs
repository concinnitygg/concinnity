// Where a world Shader's compiled program comes from.
//
// The cook compiles a Shader's two files into every main-pass entry the host
// consumes and stores what slangc emitted, so a shipped player needs no shader
// compiler. This resolves that: the stored artifact when the engine template it
// was built against still matches, and a compile here when it does not.
//
// The mismatch case is not an error path. It is what makes editing
// `main_shading.slang` with a world Shader loaded possible at all: a hot-reload
// build assembles from the checkout, digests differently, and recompiles. A
// machine with no slangc says so, naming the Shader, rather than drawing
// nothing. The sibling of `raymarch_source`.

use std::borrow::Cow;

use concinnity_core::components::ShaderPrograms;
use concinnity_core::platform::Platform;
use concinnity_core::render::slang_programs::surface::{self, Sources, Stage};
use concinnity_core::render::slang_source;
use concinnity_slang::{SlangJob, SlangTarget};

/// Which host asks, and under which capacities.
#[derive(Clone, Copy)]
pub(crate) struct Request {
    pub platform: Platform,
    /// The bindless texture-pool length and probe cube array length the host
    /// declares. Only the Vulkan bindless pair reads them; the cook bakes the
    /// ceilings, and a device that cannot seat them digests differently and
    /// compiles here.
    pub pool_size: usize,
    pub probe_count: usize,
    pub hot_reload: bool,
}

// What slangc emits for a host, and for a stage where the target needs one.
// Must match what the cook emitted, or a fallback compile would produce
// something the renderer cannot load.
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

/// The artifact holding `entry`: the cook's when the engine template it was
/// built against still matches, and a compile here when it does not.
pub(crate) fn artifact<'a>(
    programs: &'a ShaderPrograms,
    entry: &str,
    req: &Request,
) -> Result<Cow<'a, [u8]>, String> {
    let label = programs.name.as_str();
    let program = surface::program(entry)
        .ok_or_else(|| format!("Shader '{label}': no main-pass entry named '{entry}'"))?;
    let sources = Sources {
        vertex: programs.vertex.as_deref(),
        fragment: &programs.fragment,
    };
    let source = source(program, req, &sources);
    let digest = slang_source::source_digest(&source);
    if let Some(bytes) = programs.artifact(entry, digest) {
        return Ok(Cow::Borrowed(bytes));
    }
    tracing::debug!("Shader '{label}': {entry} predates the engine template, compiling");
    // The cook groups entries into artifacts per host; a fallback compile
    // holds the same set so a library found under one entry serves its pair.
    let group = surface::groups(req.platform)
        .into_iter()
        .find(|g| g.iter().any(|p| p.entry == entry))
        .unwrap_or_else(|| vec![program]);
    let entries: Vec<&str> = group.iter().map(|p| p.entry).collect();
    let job = SlangJob {
        source: &source,
        file_name: program.file,
        entries: &entries,
        target: target(req.platform, program.stage),
    };
    let work = crate::compiler_work::dir()?;
    concinnity_slang::compile(&job, work.path())
        .map(Cow::Owned)
        .map_err(|e| format!("Shader '{label}': compiling '{entry}': {e}"))
}

// The source text this host expects for one entry, preferring the checkout's
// templates under hot-reload exactly as every other single-source shader does.
fn source(program: &surface::Program, req: &Request, sources: &Sources<'_>) -> String {
    if !req.hot_reload {
        return surface::source(
            program,
            req.platform,
            req.pool_size,
            req.probe_count,
            sources,
        );
    }
    surface::source_with(
        program,
        req.platform,
        req.pool_size,
        req.probe_count,
        sources,
        crate::slang_source::from_checkout,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::compiled_programs::CompiledProgram;
    use concinnity_core::render::uniforms::{BINDLESS_POOL_SIZE, MAX_PROBES};

    const SHADE: &str = "float4 shade(VertexOut in, GpuObjectData od) { return float4(1.0); }";

    fn request(platform: Platform) -> Request {
        Request {
            platform,
            pool_size: BINDLESS_POOL_SIZE,
            probe_count: MAX_PROBES,
            hot_reload: false,
        }
    }

    fn stored(platform: Platform, entries: &[&str], bytes: &[u8]) -> ShaderPrograms {
        let sources = Sources {
            vertex: None,
            fragment: SHADE,
        };
        let program = surface::program(entries[0]).unwrap();
        let src = surface::source(program, platform, BINDLESS_POOL_SIZE, MAX_PROBES, &sources);
        ShaderPrograms {
            name: "wall".to_string(),
            vertex: None,
            fragment: SHADE.to_string(),
            programs: vec![CompiledProgram {
                entries: entries.iter().map(|e| e.to_string()).collect(),
                source_digest: slang_source::source_digest(&src),
                artifact: bytes.to_vec(),
            }],
        }
    }

    // The stored artifact is taken whenever the template still matches, which
    // is the shipped path and the one that must never reach a compiler.
    #[test]
    fn a_matching_artifact_is_taken_without_compiling() {
        let programs = stored(
            Platform::Metal,
            &["vertex_main_bindless", "fragment_main_bindless"],
            b"stored bytes",
        );
        for entry in ["vertex_main_bindless", "fragment_main_bindless"] {
            let got = artifact(&programs, entry, &request(Platform::Metal)).expect("stored");
            assert_eq!(got.as_ref(), b"stored bytes");
            assert!(matches!(got, Cow::Borrowed(_)), "no compile was needed");
        }
    }

    // An artifact built for another host, or under another pool size, is not
    // this one's: the digest covers the defines.
    #[test]
    fn an_artifact_from_another_host_or_pool_size_does_not_match() {
        let metal = stored(
            Platform::Metal,
            &["fragment_main_bindless"],
            b"stored bytes",
        );
        let sources = Sources {
            vertex: None,
            fragment: SHADE,
        };
        let program = surface::program("fragment_main_bindless").unwrap();
        let other_host = surface::source(
            program,
            Platform::Glsl,
            BINDLESS_POOL_SIZE,
            MAX_PROBES,
            &sources,
        );
        assert!(
            metal
                .artifact(
                    "fragment_main_bindless",
                    slang_source::source_digest(&other_host)
                )
                .is_none()
        );
        let vulkan = stored(Platform::Glsl, &["fragment_main_bindless"], b"stored bytes");
        let smaller = surface::source(program, Platform::Glsl, 37, MAX_PROBES, &sources);
        assert!(
            vulkan
                .artifact(
                    "fragment_main_bindless",
                    slang_source::source_digest(&smaller)
                )
                .is_none(),
            "a constrained device's pool size digests differently"
        );
    }

    // An entry the table does not name is a caller bug, reported by name
    // rather than compiled into nothing.
    #[test]
    fn an_unknown_entry_is_an_error() {
        let programs = stored(Platform::Metal, &["fragment_main_bindless"], b"x");
        let err = artifact(&programs, "no_such_entry", &request(Platform::Metal)).unwrap_err();
        assert!(err.contains("no_such_entry"), "got: {err}");
    }

    // A stale artifact compiles in place, holding the same entries the cook
    // would have grouped, so the result loads the way the stored one would.
    #[test]
    fn a_stale_artifact_compiles_its_whole_group() {
        if !crate::slangc_gate::slangc_available() {
            return;
        }
        let mut programs = stored(Platform::Metal, &["fragment_main_bindless"], b"stale");
        programs.programs[0].source_digest ^= 1;
        let got = artifact(
            &programs,
            "fragment_main_bindless",
            &request(Platform::Metal),
        )
        .expect("compiled");
        assert!(matches!(got, Cow::Owned(_)));
        let msl = std::str::from_utf8(&got).expect("MSL text");
        assert!(msl.contains("vertex_main_bindless") && msl.contains("fragment_main_bindless"));
    }
}
