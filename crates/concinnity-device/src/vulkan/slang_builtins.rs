// The Vulkan half of the single-source shader compile: everything the
// declarations in `concinnity_core::render::slang_programs::vk` need a compiler, a
// content-addressed cache, or a filesystem for, over the declarations it
// brings into the backend's scope.

use concinnity_core::render::error::{RenderError, RenderResult};
pub(super) use concinnity_core::render::slang_programs::vk::*;
use concinnity_slang as slang;

// Inputs a call site supplies to assemble a program's source.
pub(crate) struct Ctx {
    pub hot_reload: bool,
    pub msaa: bool,
    // Reflection-probe cube-array length for `{MAX_PROBES}` programs; ignored by
    // the rest. Callers pass the descriptor count the global set layout was
    // built with (`descriptor_layout::probe_cube_array_count`), so the
    // `{MAX_PROBES}` define and the layout binding always agree.
    pub probe_count: usize,
}

impl Ctx {
    // For programs whose assembly needs no MSAA state or probe count.
    pub(crate) fn plain(hot_reload: bool) -> Self {
        Self {
            hot_reload,
            msaa: false,
            probe_count: 0,
        }
    }
}

// What a declaration can do once a compiler and a cache are in reach. A trait
// because `SlangProgram` is defined in `core::render`, which is `no_std` and
// knows nothing about either.
pub(crate) trait SlangCompile {
    fn source(&self, ctx: &Ctx) -> String;
    fn cache_key<'a>(&self, source: &'a str) -> crate::shader::cache::Key<'a>;
    fn compile(&self, ctx: &Ctx) -> RenderResult<Vec<u8>>;
}

impl SlangCompile for SlangProgram {
    // Assemble the exact source text this program compiles under `ctx`.
    fn source(&self, ctx: &Ctx) -> String {
        let probes = ctx.probe_count.to_string();
        let mut defines: Vec<(&str, &str)> = self.gates.iter().map(|g| (*g, "1")).collect();
        if self.msaa {
            defines.push(("USE_MSAA", if ctx.msaa { "1" } else { "0" }));
        }
        if self.sizes != Sizes::None {
            debug_assert!(
                ctx.probe_count > 0,
                "{}: sized program assembled with no probe count",
                self.label
            );
            defines.push(("MAX_PROBES", probes.as_str()));
        }
        crate::shader::slang_source::assemble(ctx.hot_reload, self.file, &defines, &[])
    }

    // The shader-cache key for `source`. Shared by the runtime compile path
    // and the export-time precompile so the two can never key differently.
    fn cache_key<'a>(&self, source: &'a str) -> crate::shader::cache::Key<'a> {
        crate::shader::cache::Key {
            compiler: "slang",
            source,
            entry: self.entry,
            target: "spirv",
            options: 0,
        }
    }

    // The SPIR-V for this program: the copy the build script embedded when it
    // was built from this exact source, else a compile (reusing a cached
    // artifact when this source has been compiled before).
    //
    // Matching on the source digest rather than on hot-reload being off is what
    // makes this reach the binaries a developer runs: `cn debug` and `cn editor`
    // both enable hot-reload, and a mode check would leave them compiling every
    // shader at startup and needing slangc to do it. It is also what lets a
    // device that sizes its pool or probe array differently from the build fall
    // through to a compile without any special case for it.
    fn compile(&self, ctx: &Ctx) -> RenderResult<Vec<u8>> {
        let source = self.source(ctx);
        // Only a program that reads the sample count has two artifacts; keying
        // the rest on it would miss the single one they do have.
        let name = spirv_artifact_name(self.label, self.msaa && ctx.msaa);
        if let Some((digest, bytes)) = embedded_spirv(&name)
            && digest == concinnity_core::render::slang_source::source_digest(&source)
        {
            return Ok(bytes.to_vec());
        }
        let key = self.cache_key(&source);
        crate::shader::cache::cached(&key, self.label, || compile_uncached(self, &source))
            .map_err(|e| e.context(self.label))
    }
}

// The build script's precompiled SPIR-V. Every name misses on a host that had
// no slangc at build time, and the renderer compiles instead -- the only path
// that then needs slangc at runtime.
include!(concat!(env!("OUT_DIR"), "/engine_spirv.rs"));

// The lookup key for one program's artifact. Mirrors `spirv_artifact_name` in
// build.rs. The MSAA flag is part of it because a program that reads the main
// pass's depth compiles into two artifacts from one label, and handing one to
// the other would sample the wrong depth image.
fn spirv_artifact_name(label: &str, msaa: bool) -> String {
    if msaa {
        format!("{label}.msaa")
    } else {
        label.to_string()
    }
}

pub(super) fn compile_uncached(program: &SlangProgram, source: &str) -> RenderResult<Vec<u8>> {
    let job = slang::SlangJob {
        source,
        file_name: program.file,
        entries: &[program.entry],
        target: slang::SlangTarget::Spirv,
    };
    let work = crate::shader::compiler_work::dir().map_err(RenderError::Other)?;
    slang::compile(&job, work.path()).map_err(RenderError::ShaderCompile)
}

// Compile every declared program into `bundle`, reusing local cache artifacts
// where present.
//
// The probe cube-array length is a property of the device the bundle eventually
// runs on rather than of the world, so it is baked at the ceiling every desktop
// driver affords. A device that reports less headroom than that declares fewer
// and compiles these at first launch.
pub(crate) fn precompile(
    bundle: &mut concinnity_host::store::cache::Segment,
    report: &mut crate::shader::precompile::Report,
) {
    // A program whose source reads the main pass's sample count gets both
    // variants: which one a device runs is a property of its MSAA mode, not of
    // the bundle.
    for program in ALL {
        let msaa_variants: &[bool] = if program.msaa {
            &[false, true]
        } else {
            &[false]
        };
        for &msaa in msaa_variants {
            let ctx = Ctx {
                hot_reload: false,
                msaa,
                probe_count: concinnity_core::render::uniforms::MAX_PROBES,
            };
            let source = program.source(&ctx);
            let key = program.cache_key(&source);
            report.record(
                program.label,
                crate::shader::cache::ensure_in(bundle, &key, || {
                    compile_uncached(program, &source)
                }),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(probe_count: usize) -> Ctx {
        Ctx {
            hot_reload: false,
            msaa: false,
            probe_count,
        }
    }

    #[test]
    fn sized_programs_inject_their_counts_and_gates_lead() {
        // The probe count is the only capacity a Vulkan program bakes; the
        // texture pool is declared unsized and takes no define.
        let src = MAIN_BINDLESS_FRAG.source(&ctx(5));
        assert!(src.starts_with("#define MAX_PROBES 5\n"));
        let src = SSR_RESOLVE.source(&ctx(5));
        assert!(src.starts_with("#define MAX_PROBES 5\n"));
        let src = HIZ_SPD_MSAA.source(&ctx(0));
        assert!(src.starts_with("#define HIZ_SPD_MSAA 1\n"));
        let src = LIGHT_CULL.source(&ctx(0));
        assert!(!src.starts_with("#define"));
    }

    // An unreplaced `{...}` fragment marker would reach slangc as a syntax
    // error at renderer init; catch a missing splice here instead.
    #[test]
    fn every_program_assembles_with_its_fragments_spliced() {
        for p in ALL {
            let src = p.source(&ctx(4));
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

    // Reading the main pass's depth is what makes a program's assembly depend on
    // the host's sample count, and the export-time precompile enumerates both
    // variants for exactly those. A program that gained or lost the dependency
    // silently would leave a bundle cold for one MSAA mode.
    #[test]
    fn only_the_depth_reading_programs_take_the_sample_count() {
        let mut sampled: Vec<&str> = ALL.iter().filter(|p| p.msaa).map(|p| p.label).collect();
        sampled.sort_unstable();
        assert_eq!(
            sampled,
            [
                "decal_frag.slang",
                "fog_frag.slang",
                "glass_frag.slang",
                "glass_frag_rt.slang",
                "glass_frag_rt_textured.slang",
                "glass_mesh_frag_rt.slang",
                "glass_mesh_frag_rt_textured.slang",
                "glass_mesh_reflection_frag.slang",
                "glass_mesh_reflection_frag_textured.slang",
                "glass_mesh_vert.slang",
                "glass_reflection_frag.slang",
                "glass_reflection_frag_textured.slang",
                "glass_vert.slang",
                "line_frag.slang",
                "particle_frag.slang",
                "water_frag.slang",
                "water_frag_rt.slang",
                "water_frag_rt_textured.slang",
                "water_vert.slang",
            ]
        );
    }

    // Two programs collide when they would compile identical source with the
    // same entry; the table must not declare the same artifact twice.
    #[test]
    fn table_has_no_duplicate_programs() {
        let mut seen = std::collections::HashSet::new();
        for p in ALL {
            let src = p.source(&ctx(4));
            assert!(
                seen.insert((src, p.entry)),
                "duplicate program: {}",
                p.label
            );
        }
    }

    #[test]
    fn every_key_field_tracks_the_program() {
        let a = MAIN_BINDLESS_VERT.source(&ctx(4));
        let b = MAIN_BINDLESS_VERT.source(&ctx(8));
        assert_ne!(a, b, "probe count must change the assembled source");
        let key = MAIN_BINDLESS_VERT.cache_key(&a);
        assert_eq!(key.compiler, "slang");
        assert_eq!(key.target, "spirv");
        assert_eq!(key.entry, "vertex_main_bindless");
    }

    // The single-source files carry cluster constants that must track the
    // Rust values the CPU sizes buffers with.
    #[test]
    fn cluster_constants_match_render_types() {
        use concinnity_core::gfx::render_types::{
            CLUSTER_LIGHT_LIST_STRIDE, MAX_LIGHTS_PER_CLUSTER,
        };
        for src in [
            concinnity_core::render::shaders::LIGHT_CULL,
            concinnity_core::render::shaders::MAIN_SHADING,
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

    // The pool length reaches the descriptor layout and nothing else. A program
    // assembled for two different worlds is one text, which is what lets the
    // build script compile it ahead of any world.
    #[test]
    fn the_pool_length_never_reaches_the_source() {
        let probes = concinnity_core::render::uniforms::MAX_PROBES;
        for program in ALL {
            let source = program.source(&ctx(probes));
            // The capacities are prepended as `#define` lines ahead of the file
            // body, which mentions POOL_SIZE in the Metal branch it never takes.
            let injected: Vec<&str> = source
                .lines()
                .take_while(|l| l.starts_with("#define "))
                .collect();
            assert!(
                !injected.iter().any(|l| l.contains("POOL_SIZE")),
                "{}: pool length injected as {injected:?}",
                program.label
            );
        }
    }
}
