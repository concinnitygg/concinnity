// The DirectX half of the single-source shader compile: everything the
// declarations in `concinnity_core::render::slang_programs::dx` need a compiler, a
// content-addressed cache, or a filesystem for. The declarations themselves are
// re-exported here, so every call site still names them through this module.

use concinnity_slang as slang;

pub(super) use concinnity_core::render::slang_programs::dx::*;

// What a declaration can do once a compiler and a cache are in reach. A trait
// rather than an inherent impl because `SlangProgram` is defined in
// `core::render`, which is `no_std` and knows nothing about either; bringing
// this into scope is what keeps `PROGRAM.compile(..)` reading the same at every
// call site it did before the declarations moved.
pub(crate) trait SlangCompile {
    fn source(&self, hot_reload: bool) -> String;
    fn target(&self) -> slang::SlangTarget;
    fn cache_key<'a>(&self, source: &'a str) -> crate::shader_cache::Key<'a>;
    fn compile(&self, hot_reload: bool) -> Result<Vec<u8>, String>;
}

impl SlangCompile for SlangProgram {
    // Assemble the exact source text this program compiles.
    fn source(&self, hot_reload: bool) -> String {
        crate::slang_source::assemble(hot_reload, self.file, self.defines)
    }

    fn target(&self) -> slang::SlangTarget {
        slang::SlangTarget::Dxil(self.profile)
    }

    // The shader-cache key for `source`. Shared by the runtime compile path
    // and the export-time precompile so the two can never key differently.
    fn cache_key<'a>(&self, source: &'a str) -> crate::shader_cache::Key<'a> {
        crate::shader_cache::Key {
            compiler: "slang",
            source,
            entry: self.entry,
            target: self.profile,
            options: 0,
        }
    }

    // The DXIL for this program: the copy the build script embedded when it was
    // built from this exact source, else a compile (reusing a cached artifact
    // when this source has been compiled before).
    //
    // The match is on the source digest rather than on hot-reload being off,
    // and that distinction is the whole point. `cn debug` and `cn editor` both
    // run with hot-reload on, so a mode check would have left the two binaries a
    // developer actually runs compiling every shader at startup -- and needing
    // slangc installed to do it. Comparing digests instead means an unedited
    // shader takes the embedded artifact in every build, and an edited one is
    // recompiled in all of them.
    fn compile(&self, hot_reload: bool) -> Result<Vec<u8>, String> {
        let source = self.source(hot_reload);
        if let Some((digest, bytes)) = embedded_dxil(self.label)
            && digest == concinnity_core::render::slang_source::source_digest(&source)
        {
            return Ok(bytes.to_vec());
        }
        let key = self.cache_key(&source);
        crate::shader_cache::cached(&key, self.label, || compile_uncached(self, &source))
    }
}

// The build script's precompiled DXIL, keyed by each program's `label`. Every
// name misses on a host that had no slangc at build time, and the renderer
// compiles instead -- which is the only path that then needs slangc at runtime.
//
// `label` is the key rather than the entry point because one entry compiles
// into several programs: `fog_fragment` alone yields an MSAA and a non-MSAA
// DXIL from the same file at the same profile, and the ray-traced glass entry
// yields four. Keying on anything the variant defines do not reach would hand
// one variant's bytes to another -- a wrong render, not a failed one.
// `every_program_has_a_distinct_label` in `core::render` locks it.
include!(concat!(env!("OUT_DIR"), "/engine_dxil.rs"));

pub(super) fn compile_uncached(program: &SlangProgram, source: &str) -> Result<Vec<u8>, String> {
    let job = slang::SlangJob {
        source,
        file_name: program.file,
        entries: &[program.entry],
        target: program.target(),
    };
    let work = crate::compiler_work::dir()?;
    slang::compile(&job, work.path())
}

// Compile every declared program into `bundle`, reusing local cache artifacts
// where present. Called by the export-time precompile alongside the HLSL table.
pub(crate) fn precompile(
    bundle: &mut concinnity_host::store::cache::Segment,
    report: &mut crate::precompile::Report,
) {
    for program in ALL {
        let source = program.source(false);
        let key = program.cache_key(&source);
        report.record(
            &format!("{} {}", program.entry, program.profile),
            crate::shader_cache::ensure_in(bundle, &key, || compile_uncached(program, &source)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_gates_and_abi_define_lead_the_source() {
        assert!(
            MAIN_BINDLESS_FRAG
                .source(false)
                .starts_with("#define DXIL_ABI 1\n#define MAX_PROBES 8\n")
        );
        assert!(
            HIZ_INIT_MSAA
                .source(false)
                .starts_with("#define HIZ_INIT_MSAA 1\n")
        );
        // The SSR resolve reads the probe array but none of the texture pool.
        assert!(
            SSR_RESOLVE
                .source(false)
                .starts_with("#define MAX_PROBES 8\n")
        );
        assert!(!LIGHT_CULL.source(false).starts_with("#define"));
    }

    // An unreplaced `{...}` fragment marker would reach slangc as a syntax
    // error at renderer init; catch a missing splice here instead.
    #[test]
    fn every_program_assembles_with_its_fragments_spliced() {
        for p in ALL {
            let src = p.source(false);
            for marker in [
                "{POST_COMMON}",
                "{OBJECT_COMMON}",
                "{PROBE_TYPES}",
                "{PROBE_COMMON}",
                "{RT_TYPES}",
                "{RT_TRACE}",
                "{PARTICLE_TYPES}",
            ] {
                assert!(
                    !src.contains(marker),
                    "{}: unspliced fragment marker {marker}",
                    p.label
                );
            }
        }
    }

    // The pre-pass rasterises the same visible set the main pass does, sky
    // shell included, and the shell's corners fall outside the far plane, so
    // an unpinned sky vert clips and the G-buffer loses coverage the main pass
    // has. Two entries can carry skybox geometry -- the per-draw static one and
    // the GPU-driven one -- and both must call the pin. The instanced and
    // skinned entries are excluded: neither ever carries skybox geometry.
    #[test]
    fn the_prepass_pins_sky_to_the_far_plane() {
        assert!(concinnity_core::render::shaders::GBUFFER_PREPASS.contains("color.b > 1.5"));
        assert!(
            concinnity_core::render::shaders::GBUFFER_PREPASS.contains("position.z = position.w")
        );
        assert_eq!(
            concinnity_core::render::shaders::GBUFFER_PREPASS
                .matches("gb_sky_pin(")
                .count(),
            3
        );
    }

    // Two programs collide when they would compile identical source with the
    // same entry and profile; the table must not declare the same artifact twice.
    #[test]
    fn table_has_no_duplicate_programs() {
        let mut seen = std::collections::HashSet::new();
        for p in ALL {
            assert!(
                seen.insert((p.source(false), p.entry, p.profile)),
                "duplicate program: {}",
                p.label
            );
        }
    }

    // The key must carry the fields that separate one program's artifact from
    // another's: the slang toolchain, the entry, and the stage profile.
    #[test]
    fn the_cache_key_tracks_the_program() {
        let src = MAIN_BINDLESS_VERT.source(false);
        let key = MAIN_BINDLESS_VERT.cache_key(&src);
        assert_eq!(key.compiler, "slang");
        assert_eq!(key.entry, "vertex_main_bindless");
        assert_eq!(key.target, "vs_6_0");
        assert_eq!(MAIN_BINDLESS_FRAG.cache_key(&src).target, "ps_6_0");
    }

    // Every probe cube array must be exactly as long as the descriptor table
    // its root signature binds, or a probe sample reads past it. Scanned over
    // the whole table rather than a listed few, so a new program that bakes the
    // count in cannot bake in the wrong one.
    #[test]
    fn probe_cube_count_matches_the_host_constant() {
        let want = concinnity_core::render::uniforms::MAX_PROBES.to_string();
        let mut sized = 0usize;
        for p in ALL {
            for (key, value) in p.defines {
                if *key == "MAX_PROBES" {
                    assert_eq!(*value, want.as_str(), "{}", p.label);
                    sized += 1;
                }
            }
        }
        assert!(sized > 0, "no program bakes MAX_PROBES in");
    }

    // The reflection-cut constant in every single-source fragment that gates on
    // it must track the canonical Rust value: the SSR resolve, the composite
    // blur, and the forward pass all decide "does this surface reflect" from it
    // and cannot disagree. The Metal table asserts the same thing, but that
    // module does not build on a DirectX host.
    #[test]
    fn reflection_roughness_cut_matches_canonical() {
        let expected = format!(
            "static const float REFLECTION_ROUGHNESS_CUT = {:?};",
            crate::gfx::ssr::REFLECTION_ROUGHNESS_CUT
        );
        for (name, src) in [
            (
                "main_bindless.slang",
                concinnity_core::render::shaders::MAIN_BINDLESS,
            ),
            ("ssr.slang", concinnity_core::render::shaders::SSR),
            (
                "reflection.slang",
                concinnity_core::render::shaders::REFLECTION,
            ),
            (
                "rt_reflections.slang",
                concinnity_core::render::shaders::RT_REFLECTIONS,
            ),
        ] {
            assert!(
                src.contains(&expected),
                "{name} REFLECTION_ROUGHNESS_CUT drifted from \
                 concinnity_core::gfx::ssr::REFLECTION_ROUGHNESS_CUT"
            );
        }
    }

    // The single-source files carry cluster constants that must track the Rust
    // values the CPU sizes buffers with.
    #[test]
    fn cluster_constants_match_render_types() {
        use crate::gfx::render_types::{CLUSTER_LIGHT_LIST_STRIDE, MAX_LIGHTS_PER_CLUSTER};
        for src in [
            concinnity_core::render::shaders::LIGHT_CULL,
            concinnity_core::render::shaders::MAIN_BINDLESS,
        ] {
            assert!(src.contains(&format!(
                "CLUSTER_LIGHT_LIST_STRIDE = {CLUSTER_LIGHT_LIST_STRIDE}u"
            )));
        }
        assert!(
            concinnity_core::render::shaders::LIGHT_CULL.contains(&format!(
                "MAX_LIGHTS_PER_CLUSTER = {MAX_LIGHTS_PER_CLUSTER}u"
            ))
        );
    }

    // The sky shell's half-extent tracks the camera far plane, so its corners
    // always fall outside it: the bindless vertex path must pin sky verts to the
    // far plane or those corners clip and the clear colour shows through.
    #[test]
    fn the_bindless_vertex_path_pins_sky_to_the_far_plane() {
        assert!(concinnity_core::render::shaders::MAIN_BINDLESS.contains("v.color.b > 1.5"));
        assert!(
            concinnity_core::render::shaders::MAIN_BINDLESS.contains("o.position.z = o.position.w")
        );
    }
}
