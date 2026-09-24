// Compiling an SdfVolume's authored distance field.
//
// Every other engine shader is a build-time artifact. This one was not, because
// the world authors the field that completes the source, so all three backends
// compiled it at world load through whatever compiler their platform ships.
// Neither compiler is one of those: both are subprocesses, and no exported
// bundle carries one. Compiling here is what keeps a player from needing one.
// What is emitted is described in `compile::program`.

use concinnity_core::components::sdf_programs::SdfPrograms;
use concinnity_core::platform::Platform;
use concinnity_core::render::shader_programs::raymarch;

use crate::compile::program;

/// Compile every entry a volume with these flags draws with, from `field`.
///
/// A failure names the entry and carries the compiler's own diagnostic: an
/// authored field with a syntax error has to fail the build here, where the
/// message can point at it, rather than at a renderer's init on someone else's
/// machine.
pub(super) fn compile(
    name: &str,
    field: &str,
    platform: Platform,
    volumetric: bool,
    cast_shadows: bool,
) -> std::io::Result<SdfPrograms> {
    compile_with(
        name,
        field,
        platform,
        volumetric,
        cast_shadows,
        concinnity_shader::dxc_available(),
    )
}

// [`compile`] with the host's compiler availability supplied, so the
// no-compiler path is reachable without uninstalling one.
fn compile_with(
    name: &str,
    field: &str,
    platform: Platform,
    volumetric: bool,
    cast_shadows: bool,
    have_compiler: bool,
) -> std::io::Result<SdfPrograms> {
    let owner = format!("SdfVolume '{name}'");
    program::require_compiler(&owner, have_compiler)?;
    let sources: Vec<(raymarch::Family, String)> = raymarch::families(volumetric, cast_shadows)
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
    let programs = program::compile_all(&owner, &format!("sdf-{name}"), &jobs, platform, |_| "")?;
    Ok(SdfPrograms {
        field: field.to_string(),
        programs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal surface field, so a failing compile is the template's fault.
    const SURFACE: &str = "float map(float3 p, SdfParams q, float t) { return sdSphere(p, 0.5); }\n\
        SdfSurface shade(float3 p, float3 n, SdfParams q, float t, float2 uv) {\n\
            SdfSurface s; s.albedo = float3(1.0, 1.0, 1.0); s.roughness = 0.5;\n\
            s.metallic = 0.0; s.emissive = float3(0.0, 0.0, 0.0);\n\
            s.transmitted = float3(0.0, 0.0, 0.0); return s; }\n";

    // A volume whose field has to be compiled on a host with no compiler is an
    // error naming the volume, not a payload that quietly draws nothing.
    #[test]
    fn a_volume_needing_a_compiler_fails_when_there_is_none() {
        let field = "float map(float3 p) { return length(p) - 1.0; }";
        let err = compile_with("blob", field, Platform::Metal, false, true, false)
            .expect_err("no compiler is an error");

        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        let message = err.to_string();
        assert!(message.contains("blob"), "{message}");
        assert!(
            message.contains("compiled payload") || message.contains("dxc"),
            "{message}"
        );
    }

    // Each compile runs in its own scratch directory, so a payload that differs
    // between two compiles of the same field has stamped that directory in.
    #[test]
    fn a_volume_payload_is_the_same_on_every_compile() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        for platform in [Platform::Metal, Platform::Vulkan] {
            let compile = || {
                compile_with("blob", SURFACE, platform, false, true, true)
                    .unwrap_or_else(|e| panic!("{platform:?}: {e}"))
            };
            assert_eq!(compile(), compile(), "{platform:?}");
        }
    }

    // Every artifact is one entry point, which is what a DXIL container is,
    // what a Vulkan pipeline binds, and what one MSL translation carries.
    // Metal's is MSL text, emitted on every host: a world cooked on Windows or
    // Linux still gives a Metal player its field without a compiler.
    #[test]
    fn every_host_cooks_one_artifact_per_entry_for_every_target() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        for platform in Platform::ALL {
            let programs = compile_with("blob", SURFACE, platform, false, true, true)
                .unwrap_or_else(|e| panic!("{platform:?}: {e}"));
            let entries: Vec<&str> = programs.programs.iter().map(|p| p.entry.as_str()).collect();
            assert_eq!(
                entries,
                [
                    "raymarch_vertex",
                    "raymarch_fragment",
                    "raymarch_shadow_vertex",
                    "raymarch_shadow_fragment"
                ],
                "{platform:?}"
            );
            if platform == Platform::Metal {
                for p in &programs.programs {
                    let text = std::str::from_utf8(&p.artifact).expect("MSL is text");
                    assert!(text.contains(&p.entry), "{}", p.entry);
                }
            }
        }
    }
}
