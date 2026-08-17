// Single-source engine shader programs for the Vulkan backend.
//
// Each program compiles a `.slang` file under `src/shaders/` (the
// backend-neutral single-source directory) to SPIR-V at renderer init by
// invoking slangc through `concinnity-slang`, cached in the content-addressed
// shader cache exactly like the GLSL programs in
// `builtins`. The runtime `POOL_SIZE` / `MAX_PROBES` values are injected as
// `#define` lines into the source text, so the cache keys them the same way
// the GLSL `{POOL_SIZE}` substitution always has.
//
// The `[[vk::binding]]` annotations (and ParameterBlock member order) in the
// sources reproduce the engine's existing descriptor-set layouts, so these
// programs are drop-in replacements for the GLSL they superseded. slangc
// names every single-entry SPIR-V entry point `main`, so pipeline stage
// creation is unchanged too.

use concinnity_slang as slang;

use super::builtins::Ctx;

pub(crate) struct SlangProgram {
    // File name under `src/shaders/` for the `cn debug` disk-first resolve;
    // also the embedded fallback's origin.
    pub file: &'static str,
    pub embedded: &'static str,
    pub entry: &'static str,
    // Diagnostic label (compile errors + cache miss logs + export report).
    pub label: &'static str,
    // Fixed variant gate (e.g. HIZ_INIT_MSAA), injected as `#define <gate> 1`.
    pub gate: Option<&'static str>,
    // Inject `POOL_SIZE` / `MAX_PROBES` from the context (the bindless main
    // pair); the rest compile with no size defines.
    pub sized: bool,
}

const MAIN_BINDLESS_SLANG: &str = include_str!("../shaders/main_bindless.slang");
const LIGHT_CULL_SLANG: &str = include_str!("../shaders/light_cull.slang");
const HIZ_BUILD_SLANG: &str = include_str!("../shaders/hiz_build.slang");

pub(super) static MAIN_BINDLESS_VERT: SlangProgram = SlangProgram {
    file: "main_bindless.slang",
    embedded: MAIN_BINDLESS_SLANG,
    entry: "vertex_main_bindless",
    label: "vert_bindless.slang",
    gate: None,
    sized: true,
};
pub(super) static MAIN_BINDLESS_FRAG: SlangProgram = SlangProgram {
    file: "main_bindless.slang",
    embedded: MAIN_BINDLESS_SLANG,
    entry: "fragment_main_bindless",
    label: "frag_bindless.slang",
    gate: None,
    sized: true,
};
pub(super) static LIGHT_CULL: SlangProgram = SlangProgram {
    file: "light_cull.slang",
    embedded: LIGHT_CULL_SLANG,
    entry: "light_cull_kernel",
    label: "light_cull.slang",
    gate: None,
    sized: false,
};
pub(super) static HIZ_INIT_MSAA: SlangProgram = SlangProgram {
    file: "hiz_build.slang",
    embedded: HIZ_BUILD_SLANG,
    entry: "hiz_init_msaa",
    label: "hiz_init_msaa.slang",
    gate: Some("HIZ_INIT_MSAA"),
    sized: false,
};
pub(super) static HIZ_INIT_SINGLE: SlangProgram = SlangProgram {
    file: "hiz_build.slang",
    embedded: HIZ_BUILD_SLANG,
    entry: "hiz_init_single",
    label: "hiz_init_single.slang",
    gate: Some("HIZ_INIT_SINGLE"),
    sized: false,
};
pub(super) static HIZ_DOWNSAMPLE: SlangProgram = SlangProgram {
    file: "hiz_build.slang",
    embedded: HIZ_BUILD_SLANG,
    entry: "hiz_downsample",
    label: "hiz_downsample.slang",
    gate: Some("HIZ_DOWNSAMPLE"),
    sized: false,
};

// Every declared program, iterated by the export-time precompile. Both Hi-Z
// init variants are enumerated: which one a device runs depends on its MSAA
// mode, and a bundle should be warm for either.
pub(crate) static ALL: &[&SlangProgram] = &[
    &MAIN_BINDLESS_VERT,
    &MAIN_BINDLESS_FRAG,
    &LIGHT_CULL,
    &HIZ_INIT_MSAA,
    &HIZ_INIT_SINGLE,
    &HIZ_DOWNSAMPLE,
];

impl SlangProgram {
    // Assemble the exact source text this program compiles under `ctx`.
    pub fn source(&self, ctx: &Ctx) -> String {
        let pool = ctx.pool_size.to_string();
        let probes = ctx.probe_count.to_string();
        let mut defines: Vec<(&str, &str)> = Vec::new();
        if let Some(gate) = self.gate {
            defines.push((gate, "1"));
        }
        if self.sized {
            debug_assert!(
                ctx.pool_size > 0 && ctx.probe_count > 0,
                "{}: sized program assembled with no pool / probe counts",
                self.label
            );
            defines.push(("POOL_SIZE", pool.as_str()));
            defines.push(("MAX_PROBES", probes.as_str()));
        }
        crate::slang_source::assemble(ctx.hot_reload, self.file, self.embedded, &defines)
    }

    // The shader-cache key for `source`. Shared by the runtime compile path
    // and the export-time precompile so the two can never key differently.
    pub(crate) fn cache_key<'a>(&self, source: &'a str) -> crate::shader_cache::Key<'a> {
        crate::shader_cache::Key {
            compiler: "slang",
            source,
            entry: self.entry,
            target: "spirv",
            options: 0,
        }
    }

    // Compile to SPIR-V, reusing a cached artifact when this exact assembled
    // source has been compiled before.
    pub fn compile(&self, ctx: &Ctx) -> Result<Vec<u8>, String> {
        let source = self.source(ctx);
        let key = self.cache_key(&source);
        crate::shader_cache::cached(&key, self.label, || compile_uncached(self, &source))
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
        let src = HIZ_INIT_MSAA.source(&ctx(0, 0));
        assert!(src.starts_with("#define HIZ_INIT_MSAA 1\n"));
        let src = LIGHT_CULL.source(&ctx(0, 0));
        assert!(!src.starts_with("#define"));
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
        for src in [LIGHT_CULL_SLANG, MAIN_BINDLESS_SLANG] {
            assert!(src.contains(&format!(
                "CLUSTER_LIGHT_LIST_STRIDE = {CLUSTER_LIGHT_LIST_STRIDE}u"
            )));
        }
        assert!(LIGHT_CULL_SLANG.contains(&format!(
            "MAX_LIGHTS_PER_CLUSTER = {MAX_LIGHTS_PER_CLUSTER}u"
        )));
    }
}
