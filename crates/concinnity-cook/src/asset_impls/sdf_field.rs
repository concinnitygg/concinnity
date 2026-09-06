// Compiling an SdfVolume's authored distance field.
//
// Every other engine shader is a build-time artifact. This one was not, because
// the world authors the field that completes the source, so all three backends
// compiled it at world load through whatever compiler their platform ships.
// slangc is not one of those: it is a subprocess, and no exported bundle
// carries it. Compiling here is what keeps a player from needing one.
//
// What is emitted is what that backend's renderer consumes directly: SPIR-V for
// Vulkan, a signed DXIL container for D3D12, and MSL text for Metal, whose
// `newLibraryWithSource` is an always-present OS API and so needs no metallib.
// Each artifact carries the digest of the source it came from, so a renderer
// whose template has moved compiles instead of loading something stale.

use concinnity_core::components::compiled_programs::CompiledProgram;
use concinnity_core::components::sdf_programs::SdfPrograms;
use concinnity_core::platform::Platform;
use concinnity_core::render::slang_programs::raymarch;
use concinnity_core::render::slang_source;
use concinnity_slang::{SlangJob, SlangTarget};

// What slangc emits for a host, and for a stage where the target needs one.
// DXIL is the only one that takes a profile: it sets the container's feature
// floor, which no target flag implies.
fn target(platform: Platform, stage: raymarch::Stage) -> SlangTarget {
    match platform {
        Platform::Metal => SlangTarget::Metal,
        Platform::Glsl => SlangTarget::Spirv,
        Platform::Hlsl => SlangTarget::Dxil(match stage {
            raymarch::Stage::Vertex => "vs_6_0",
            raymarch::Stage::Fragment => "ps_6_0",
        }),
    }
}

/// Compile every entry a volume with these flags draws with, from `field`.
///
/// A failure names the entry and carries slangc's own diagnostic: an authored
/// field with a syntax error has to fail the build here, where the message can
/// point at it, rather than at a renderer's init on someone else's machine.
pub(super) fn compile(
    name: &str,
    field: &str,
    platform: Platform,
    volumetric: bool,
    cast_shadows: bool,
) -> std::io::Result<SdfPrograms> {
    let work = concinnity_host::scratch::Scratch::dir(&format!("sdf-{name}"))?;
    let mut programs = Vec::new();
    for family in raymarch::families(volumetric, cast_shadows) {
        let source = raymarch::source(family, platform, field);
        let digest = slang_source::source_digest(&source);
        for group in entry_groups(family, platform) {
            let entries: Vec<&str> = group.iter().map(|p| p.entry).collect();
            let job = SlangJob {
                source: &source,
                file_name: raymarch::FILE,
                entries: &entries,
                target: target(platform, group[0].stage),
            };
            let artifact = concinnity_slang::compile(&job, work.path()).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "SdfVolume '{name}': compiling '{}': {e}",
                        entries.join(", ")
                    ),
                )
            })?;
            programs.push(CompiledProgram {
                entries: entries.iter().map(|e| e.to_string()).collect(),
                source_digest: digest,
                artifact,
            });
        }
    }
    Ok(SdfPrograms {
        field: field.to_string(),
        programs,
    })
}

// How a family's entries are grouped into artifacts.
//
// Metal takes both stages at once: slangc emits one MSL translation unit and
// the runtime wants one library to pull both functions out of, so splitting
// them would mean compiling the same unit twice at every world load. The other
// two take one entry each, which is what a DXIL container is and what each of
// those renderers already binds.
fn entry_groups(
    family: raymarch::Family,
    platform: Platform,
) -> Vec<Vec<&'static raymarch::Program>> {
    let of_family: Vec<&raymarch::Program> = raymarch::ALL
        .iter()
        .filter(|p| p.family == family)
        .collect();
    match platform {
        Platform::Metal => vec![of_family],
        _ => of_family.into_iter().map(|p| vec![p]).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every host takes the target its renderer can load without a toolchain of
    // its own: Metal source text for `newLibraryWithSource`, and a container
    // for the two that consume bytecode.
    #[test]
    fn each_host_emits_what_its_renderer_consumes() {
        assert_eq!(
            target(Platform::Metal, raymarch::Stage::Fragment),
            SlangTarget::Metal
        );
        assert_eq!(
            target(Platform::Glsl, raymarch::Stage::Vertex),
            SlangTarget::Spirv
        );
    }

    // DXIL is the one target whose profile is not implied, and it is per stage:
    // a container built at the wrong one is rejected by the PSO, not by slangc.
    #[test]
    fn the_dxil_profile_follows_the_stage() {
        assert_eq!(
            target(Platform::Hlsl, raymarch::Stage::Vertex),
            SlangTarget::Dxil("vs_6_0")
        );
        assert_eq!(
            target(Platform::Hlsl, raymarch::Stage::Fragment),
            SlangTarget::Dxil("ps_6_0")
        );
    }

    // Metal takes a family's two stages as one artifact and the other two take
    // one each, which is what keeps a Metal world load to one source compile
    // per family instead of two.
    #[test]
    fn metal_groups_a_family_into_one_artifact_and_the_others_split_it() {
        let metal = entry_groups(raymarch::Family::Surface, Platform::Metal);
        assert_eq!(metal.len(), 1);
        assert_eq!(
            metal[0].iter().map(|p| p.entry).collect::<Vec<_>>(),
            ["raymarch_vertex", "raymarch_fragment"]
        );

        for platform in [Platform::Hlsl, Platform::Glsl] {
            let split = entry_groups(raymarch::Family::Shadow, platform);
            assert_eq!(split.len(), 2, "{platform:?}");
            assert!(split.iter().all(|g| g.len() == 1), "{platform:?}");
        }
    }

    // Every grouping covers its family's entries exactly once, whichever way it
    // splits: a dropped entry is a pipeline that cannot be built at load.
    #[test]
    fn a_grouping_covers_every_entry_of_its_family_once() {
        for family in [
            raymarch::Family::Surface,
            raymarch::Family::Volumetric,
            raymarch::Family::Shadow,
        ] {
            let expected: Vec<&str> = raymarch::ALL
                .iter()
                .filter(|p| p.family == family)
                .map(|p| p.entry)
                .collect();
            for platform in [Platform::Metal, Platform::Hlsl, Platform::Glsl] {
                let mut got: Vec<&str> = entry_groups(family, platform)
                    .iter()
                    .flatten()
                    .map(|p| p.entry)
                    .collect();
                got.sort_unstable();
                let mut want = expected.clone();
                want.sort_unstable();
                assert_eq!(got, want, "{family:?} on {platform:?}");
            }
        }
    }
}
