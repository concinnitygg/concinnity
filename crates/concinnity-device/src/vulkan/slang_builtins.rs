// The Vulkan half of the single-source shader compile: everything the
// declarations in `concinnity_core::render::slang_programs::vk` need a compiler, a
// content-addressed cache, or a filesystem for. The declarations themselves are
// re-exported here, so every call site still names them through this module.

use concinnity_slang as slang;

use super::builtins::Ctx;

pub(super) use concinnity_core::render::slang_programs::vk::*;

// What a declaration can do once a compiler and a cache are in reach. A trait
// rather than an inherent impl because `SlangProgram` is defined in
// `core::render`, which is `no_std` and knows nothing about either; bringing
// this into scope keeps `PROGRAM.compile(&ctx)` reading as it did before the
// declarations moved.
pub(crate) trait SlangCompile {
    fn source(&self, ctx: &Ctx) -> String;
    fn cache_key<'a>(&self, source: &'a str) -> crate::shader_cache::Key<'a>;
    fn compile(&self, ctx: &Ctx) -> Result<Vec<u8>, String>;
}

impl SlangCompile for SlangProgram {
    // Assemble the exact source text this program compiles under `ctx`.
    fn source(&self, ctx: &Ctx) -> String {
        let pool = ctx.pool_size.to_string();
        let probes = ctx.probe_count.to_string();
        let mut defines: Vec<(&str, &str)> = self.gates.iter().map(|g| (*g, "1")).collect();
        if self.msaa {
            defines.push(("USE_MSAA", if ctx.msaa { "1" } else { "0" }));
        }
        if self.sizes == Sizes::PoolAndProbes {
            debug_assert!(
                ctx.pool_size > 0,
                "{}: sized program assembled with no pool count",
                self.label
            );
            defines.push(("POOL_SIZE", pool.as_str()));
        }
        if self.sizes != Sizes::None {
            debug_assert!(
                ctx.probe_count > 0,
                "{}: sized program assembled with no probe count",
                self.label
            );
            defines.push(("MAX_PROBES", probes.as_str()));
        }
        crate::slang_source::assemble(ctx.hot_reload, self.file, &defines)
    }

    // The shader-cache key for `source`. Shared by the runtime compile path
    // and the export-time precompile so the two can never key differently.
    fn cache_key<'a>(&self, source: &'a str) -> crate::shader_cache::Key<'a> {
        crate::shader_cache::Key {
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
    fn compile(&self, ctx: &Ctx) -> Result<Vec<u8>, String> {
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
        crate::shader_cache::cached(&key, self.label, || compile_uncached(self, &source))
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

pub(super) fn compile_uncached(program: &SlangProgram, source: &str) -> Result<Vec<u8>, String> {
    let job = slang::SlangJob {
        source,
        file_name: program.file,
        entries: &[program.entry],
        target: slang::SlangTarget::Spirv,
    };
    slang::compile(&job, &crate::shader_cache::slang_work_dir())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(pool_size: usize, probe_count: usize) -> Ctx {
        Ctx {
            hot_reload: false,
            msaa: false,
            pool_size,
            probe_count,
        }
    }

    #[test]
    fn sized_programs_inject_their_counts_and_gates_lead() {
        let src = MAIN_BINDLESS_FRAG.source(&ctx(17, 5));
        assert!(src.starts_with("#define POOL_SIZE 17\n#define MAX_PROBES 5\n"));
        // The SSR resolve reads the probe array but none of the texture pool,
        // so it takes the probe count alone.
        let src = SSR_RESOLVE.source(&ctx(17, 5));
        assert!(src.starts_with("#define MAX_PROBES 5\n"));
        let src = HIZ_INIT_MSAA.source(&ctx(0, 0));
        assert!(src.starts_with("#define HIZ_INIT_MSAA 1\n"));
        let src = LIGHT_CULL.source(&ctx(0, 0));
        assert!(!src.starts_with("#define"));
    }

    // An unreplaced `{...}` fragment marker would reach slangc as a syntax
    // error at renderer init; catch a missing splice here instead.
    #[test]
    fn every_program_assembles_with_its_fragments_spliced() {
        for p in ALL {
            let src = p.source(&ctx(4, 4));
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
                "glass_mesh_vert.slang",
                "glass_vert.slang",
                "line_frag.slang",
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
            let src = p.source(&ctx(4, 4));
            assert!(
                seen.insert((src, p.entry)),
                "duplicate program: {}",
                p.label
            );
        }
    }

    #[test]
    fn every_key_field_tracks_the_program() {
        let a = MAIN_BINDLESS_VERT.source(&ctx(4, 4));
        let b = MAIN_BINDLESS_VERT.source(&ctx(5, 4));
        assert_ne!(a, b, "pool size must change the assembled source");
        let key = MAIN_BINDLESS_VERT.cache_key(&a);
        assert_eq!(key.compiler, "slang");
        assert_eq!(key.target, "spirv");
        assert_eq!(key.entry, "vertex_main_bindless");
    }

    // The single-source files carry cluster constants that must track the
    // Rust values the CPU sizes buffers with.
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
}
