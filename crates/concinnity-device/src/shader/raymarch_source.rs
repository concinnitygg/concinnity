// Where a raymarched volume's compiled shader comes from.
//
// The cook compiles a world's distance field and stores what the compiler
// emitted, so a shipped player needs no shader compiler for the one asset whose
// source is only complete once a world is loaded. This resolves that: the
// stored artifact when the engine template it was built against still matches,
// and a compile here when it does not.
//
// The mismatch case is not an error path. It is what makes editing
// `raymarch.hlsl` possible at all: a hot-reload build assembles from the
// checkout, digests differently, and recompiles. A machine with no compiler
// says so, naming the volume, rather than drawing nothing.

use concinnity_core::components::sdf_programs::SdfPrograms;
use concinnity_core::platform::Platform;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::shader_programs::raymarch::{self, Family};
use concinnity_core::render::shader_source;
use std::borrow::Cow;

/// Decode a volume's payload. A payload that does not decode is a build the
/// renderer cannot use, and saying which volume is the whole of the fix.
pub(crate) fn decode(payload: &[u8], label: &str) -> Result<SdfPrograms, String> {
    SdfPrograms::decode(payload)
        .map_err(|e| format!("SdfVolume '{label}': compiled field does not decode: {e}"))
}

/// Whether a volume's authored field reads the scene behind its surface.
///
/// Every backend carries this on its per-volume record and gates the frame's
/// scene-color copy on some visible volume answering `true`. The copy is a
/// full read plus a full write of the HDR target, so a world whose volumes are
/// all opaque skips an encoder and its barriers outright.
pub(crate) fn taps_scene(programs: &SdfPrograms) -> bool {
    raymarch::field_taps_scene(&programs.field)
}

/// Which artifact a host wants. Every artifact holds one entry point, which is
/// also its lookup key.
pub(crate) struct Request<'a> {
    pub family: Family,
    pub platform: Platform,
    pub entry: &'a str,
    pub hot_reload: bool,
    pub label: &'a str,
}

/// The artifact the request names: the cook's when the engine template it was
/// built against still matches, and `compile` of the assembled source when it
/// does not.
///
/// `compile` takes the host, the file, the entry and the assembled source, and
/// must emit what the cook emitted for this host, or a fallback compile would
/// produce something the renderer cannot load. Each backend passes
/// `shader::compile::cooked`, which picks the target the way the cook does.
pub(crate) fn artifact<'a>(
    programs: &'a SdfPrograms,
    req: &Request<'_>,
    compile: impl FnOnce(Platform, &str, &str, &str) -> RenderResult<Vec<u8>>,
) -> RenderResult<Cow<'a, [u8]>> {
    let Request { label, entry, .. } = *req;
    let source = source(req.family, req.platform, &programs.field, req.hot_reload);
    let digest = shader_source::source_digest(&source);
    if let Some(bytes) = programs.artifact(entry, digest) {
        return Ok(Cow::Borrowed(bytes));
    }
    tracing::debug!("SdfVolume '{label}': {entry} predates the engine template, compiling");
    compile(req.platform, raymarch::FILE, entry, &source)
        .map(Cow::Owned)
        .map_err(|e| e.context(format_args!("SdfVolume '{label}': compiling '{entry}'")))
}

/// One family's artifacts on `platform`, as (vertex, fragment). The cook stores
/// each stage as its own artifact, since a DXIL container or a Vulkan module
/// binds one entry; a template edit makes both miss and compile here.
#[cfg(any(backend_dx, backend_vk))]
pub(crate) fn family_artifacts(
    programs: &SdfPrograms,
    family: Family,
    platform: Platform,
    hot_reload: bool,
    label: &str,
) -> RenderResult<(Vec<u8>, Vec<u8>)> {
    let mut stages = raymarch::ALL.iter().filter(|p| p.family == family);
    let mut stage = |which: &str| -> RenderResult<Vec<u8>> {
        let entry = stages
            .next()
            .unwrap_or_else(|| panic!("a family declares a {which} entry"))
            .entry;
        let req = Request {
            family,
            platform,
            entry,
            hot_reload,
            label,
        };
        artifact(programs, &req, crate::shader::compile::cooked).map(Cow::into_owned)
    };
    let vertex = stage("vertex")?;
    let fragment = stage("fragment")?;
    Ok((vertex, fragment))
}

// The source text this host expects for one family, preferring the checkout's
// templates under hot-reload exactly as every other single-source shader does.
fn source(family: Family, platform: Platform, field: &str, hot_reload: bool) -> String {
    if !hot_reload {
        return raymarch::source(family, platform, field);
    }
    raymarch::source_with(
        family,
        platform,
        field,
        crate::shader::source::from_checkout,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::compiled_programs::CompiledProgram;
    use concinnity_core::render::error::RenderError;

    const FIELD: &str = "// a field";

    fn stored(family: Family, platform: Platform, entry: &str, bytes: &[u8]) -> SdfPrograms {
        let src = raymarch::source(family, platform, FIELD);
        SdfPrograms {
            field: FIELD.to_string(),
            programs: vec![CompiledProgram {
                entry: entry.to_string(),
                source_digest: shader_source::source_digest(&src),
                artifact: bytes.to_vec(),
            }],
        }
    }

    // The stored artifact is taken whenever the template still matches, which
    // is the shipped path and the one that must never reach a compiler.
    #[test]
    fn a_matching_artifact_is_taken_without_compiling() {
        let programs = stored(
            Family::Surface,
            Platform::Metal,
            "raymarch_vertex",
            b"stored bytes",
        );
        let got = artifact(&programs, &request("raymarch_vertex"), |_, _, _, _| {
            panic!("a matching artifact reached the compiler")
        })
        .expect("stored artifact");
        assert_eq!(got.as_ref(), b"stored bytes");
        assert!(matches!(got, Cow::Borrowed(_)), "no compile was needed");
    }

    // A miss compiles the source this host assembles, and a failure names the
    // volume and the entry, which is what makes it actionable in a world of many.
    #[test]
    fn a_stale_artifact_compiles_the_assembled_source_and_names_the_volume() {
        let mut programs = stored(
            Family::Surface,
            Platform::Metal,
            "raymarch_vertex",
            b"stored bytes",
        );
        programs.programs[0].source_digest ^= 1;
        let want = raymarch::source(Family::Surface, Platform::Metal, FIELD);
        let got = artifact(
            &programs,
            &request("raymarch_vertex"),
            |platform, file, entry, src| {
                assert_eq!(platform, Platform::Metal);
                assert_eq!((file, entry), (raymarch::FILE, "raymarch_vertex"));
                assert_eq!(src, want);
                Ok(b"fresh".to_vec())
            },
        )
        .expect("compiled");
        assert_eq!(got.as_ref(), b"fresh");

        let err = artifact(&programs, &request("raymarch_vertex"), |_, _, _, _| {
            Err(RenderError::ShaderCompile("no compiler".to_string()))
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("SdfVolume 'blob'"), "got: {err}");
        assert!(err.contains("raymarch_vertex"), "got: {err}");
        assert!(err.contains("no compiler"), "got: {err}");
    }

    fn request(entry: &str) -> Request<'_> {
        Request {
            family: Family::Surface,
            platform: Platform::Metal,
            entry,
            hot_reload: false,
            label: "blob",
        }
    }

    // An artifact built for another host, or for another family, is not this
    // one's: the digest covers the backend define and the family define alike.
    #[test]
    fn an_artifact_from_another_host_or_family_does_not_match() {
        let metal_surface = stored(
            Family::Surface,
            Platform::Metal,
            "raymarch_vertex",
            b"stored bytes",
        );
        let src_other_host = raymarch::source(Family::Surface, Platform::DirectX, FIELD);
        assert!(
            metal_surface
                .artifact(
                    "raymarch_vertex",
                    shader_source::source_digest(&src_other_host)
                )
                .is_none()
        );
        let src_other_family = raymarch::source(Family::Shadow, Platform::Metal, FIELD);
        assert!(
            metal_surface
                .artifact(
                    "raymarch_vertex",
                    shader_source::source_digest(&src_other_family)
                )
                .is_none()
        );
    }

    // A field the cook never compiled this entry for reads as absent rather
    // than as some other entry's bytes.
    #[test]
    fn an_entry_the_cook_did_not_emit_is_absent() {
        let programs = stored(
            Family::Surface,
            Platform::Metal,
            "raymarch_vertex",
            b"stored bytes",
        );
        let src = raymarch::source(Family::Surface, Platform::Metal, FIELD);
        let digest = shader_source::source_digest(&src);
        assert!(
            programs
                .artifact("raymarch_shadow_vertex", digest)
                .is_none()
        );
    }

    // The flag every backend gates its scene copy on reads the authored field,
    // not the artifact: a volume whose field never calls the tap costs no copy.
    #[test]
    fn only_a_volume_whose_field_taps_the_scene_reads_as_refractive() {
        let mut programs = stored(
            Family::Surface,
            Platform::Metal,
            "raymarch_vertex",
            b"stored bytes",
        );
        assert!(!taps_scene(&programs), "'{FIELD}' calls nothing");
        programs.field = SURFACE_FIELD.to_string();
        assert!(taps_scene(&programs), "the surface field calls the tap");
    }

    // A payload that does not decode names the volume, which is the only thing
    // that makes it actionable in a world of many.
    #[test]
    fn a_corrupt_payload_names_the_volume() {
        let err = decode(&[0xff, 0xff, 0xff, 0xff], "chrome_blob").unwrap_err();
        assert!(err.starts_with("SdfVolume 'chrome_blob':"), "got: {err}");
    }

    // A surface field and a volumetric one, in source. These stand in for a
    // world's own: a test may not read one, and the point of the guard is the
    // engine template around them rather than the field itself.
    const SURFACE_FIELD: &str = r#"
float map(float3 p, SdfParams params, float time)
{
    float3 rp = p + float3(sdf_param(params, 0u), 0.0, 0.0) * time;
    return opSmoothUnion(sdSphere(rp, 0.5), sdTorus(rp, float2(0.6, 0.2)), 0.25);
}
SdfSurface shade(float3 p, float3 normal, SdfParams params, float time, float2 frag_uv)
{
    SdfSurface s;
    s.albedo = float3(0.85, 0.86, 0.88);
    s.roughness = clamp(sdf_param(params, 3u), 0.02, 1.0);
    s.metallic = 1.0;
    s.emissive = float3(0.0, 0.0, 0.0);
    // The scene tap, so the guard covers the one declaration an authored field
    // can pull in that nothing else references.
    s.transmitted = sampleSceneRefracted(frag_uv, normal, 0.05);
    return s;
}
"#;

    const VOLUMETRIC_FIELD: &str = r#"
VolumeSample sampleVolume(float3 p, SdfParams params, float time)
{
    VolumeSample vs;
    vs.density = max(0.0, sdf_param(params, 4u) * (0.5 + 0.5 * sin(p.x + time)));
    vs.scattering = float3(0.8, 0.8, 0.85);
    vs.emission = float3(0.0, 0.0, 0.0);
    return vs;
}
"#;

    // Every entry of every family compiles, on every backend, from the same
    // source the renderer assembles.
    //
    // This is the whole compile coverage for the pass. It replaces the
    // per-backend guards the hand-written templates had, and it covers strictly
    // more: those compiled one backend's copy, this one compiles the shared
    // source for all three, so a spelling that only one target rejects fails
    // here rather than when that renderer boots. A volumetric field defines no
    // `map` and a surface field no `sampleVolume`, which is the other thing it
    // proves: each variant reaches only the entries its family declares.
    #[test]
    fn every_raymarch_entry_compiles_on_every_backend() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let work = concinnity_host::scratch::Scratch::dir("raymarch-compile-guard")
            .expect("scratch directory");
        for platform in Platform::ALL {
            let target = concinnity_shader::HlslTarget::cooked(platform);
            for family in [Family::Surface, Family::Volumetric, Family::Shadow] {
                let field = if family == Family::Volumetric {
                    VOLUMETRIC_FIELD
                } else {
                    SURFACE_FIELD
                };
                let source = raymarch::source(family, platform, field);
                for program in raymarch::ALL.iter().filter(|p| p.family == family) {
                    let job = concinnity_shader::HlslJob {
                        source: &source,
                        file_name: raymarch::FILE,
                        entry: program.entry,
                        target,
                    };
                    concinnity_shader::compile(&job, work.path()).unwrap_or_else(|e| {
                        panic!("{:?}/{:?} {}: {e}", platform, family, program.entry)
                    });
                }
            }
        }
    }
}
