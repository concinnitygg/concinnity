// Single-source engine shader programs for the DirectX backend.
//
// Each program compiles a `.slang` file under `src/shaders/` (the
// backend-neutral single-source directory) to a signed DXIL container at
// renderer init by invoking slangc, cached in the content-addressed shader
// cache exactly like the HLSL programs in `builtins`. slangc resolves dxil.dll
// itself, so the containers it emits are already signed for D3D12.
//
// The bindless main pair compiles the source's `DXIL_ABI` block, whose
// register() annotations reproduce the bindless main root signature in
// `init/pipelines.rs` slot for slot: that layout is a contract, since a world
// Shader asset builds its own PSO against the same root signature (see
// world_shaders.rs). `assert_slang_dxil_abi` in build.rs locks it. The two
// compute kernels need no ABI block: slangc assigns b0/t0/u0 from declaration
// order, which is what their root signatures already bind.
//
// These are shader model 6.0 rather than the FXC path's 5.1: the bindless pool
// index is non-uniform across a fragment wave, and `NonUniformResourceIndex`
// is an SM 6.0 construct.

use concinnity_slang as slang;

pub(crate) struct SlangProgram {
    // File name under `src/shaders/` for the hot-reload disk-first resolve;
    // also the embedded fallback's origin and the name slangc diagnostics use.
    pub file: &'static str,
    pub embedded: &'static str,
    pub entry: &'static str,
    // Shader-model profile for the DXIL container (stage + feature floor).
    pub profile: &'static str,
    // Diagnostic label (compile errors + cache miss logs + export report).
    pub label: &'static str,
    // Variant defines injected as `#define` lines ahead of the source.
    pub defines: &'static [(&'static str, &'static str)],
}

const MAIN_BINDLESS_SLANG: &str = include_str!("../shaders/main_bindless.slang");
const LIGHT_CULL_SLANG: &str = include_str!("../shaders/light_cull.slang");
const HIZ_BUILD_SLANG: &str = include_str!("../shaders/hiz_build.slang");

// The bindless main pair's variant defines. `MAX_PROBES` sizes the probe cube
// array the root signature's descriptor table covers; `probe_cube_count_matches`
// locks it to the host constant. `POOL_SIZE` is deliberately absent: the DXIL
// pool is an unbounded array, so the shader never over-declares the per-frame
// descriptor region the host actually wrote.
const MAIN_DEFINES: &[(&str, &str)] = &[("DXIL_ABI", "1"), ("MAX_PROBES", "8")];

pub(super) static MAIN_BINDLESS_VERT: SlangProgram = SlangProgram {
    file: "main_bindless.slang",
    embedded: MAIN_BINDLESS_SLANG,
    entry: "vertex_main_bindless",
    profile: "vs_6_0",
    label: "vert_bindless.slang",
    defines: MAIN_DEFINES,
};
pub(super) static MAIN_BINDLESS_FRAG: SlangProgram = SlangProgram {
    file: "main_bindless.slang",
    embedded: MAIN_BINDLESS_SLANG,
    entry: "fragment_main_bindless",
    profile: "ps_6_0",
    label: "frag_bindless.slang",
    defines: MAIN_DEFINES,
};
pub(super) static LIGHT_CULL: SlangProgram = SlangProgram {
    file: "light_cull.slang",
    embedded: LIGHT_CULL_SLANG,
    entry: "light_cull_kernel",
    profile: "cs_6_0",
    label: "light_cull.slang",
    defines: &[],
};
pub(super) static HIZ_INIT_SINGLE: SlangProgram = SlangProgram {
    file: "hiz_build.slang",
    embedded: HIZ_BUILD_SLANG,
    entry: "hiz_init_single",
    profile: "cs_6_0",
    label: "hiz_init_single.slang",
    defines: &[("HIZ_INIT_SINGLE", "1")],
};
pub(super) static HIZ_INIT_MSAA: SlangProgram = SlangProgram {
    file: "hiz_build.slang",
    embedded: HIZ_BUILD_SLANG,
    entry: "hiz_init_msaa",
    profile: "cs_6_0",
    label: "hiz_init_msaa.slang",
    defines: &[("HIZ_INIT_MSAA", "1")],
};
pub(super) static HIZ_DOWNSAMPLE: SlangProgram = SlangProgram {
    file: "hiz_build.slang",
    embedded: HIZ_BUILD_SLANG,
    entry: "hiz_downsample",
    profile: "cs_6_0",
    label: "hiz_downsample.slang",
    defines: &[("HIZ_DOWNSAMPLE", "1")],
};

// Every declared program, iterated by the export-time precompile. Both Hi-Z
// init variants are enumerated: which one a device runs depends on its MSAA
// mode, and a bundle should be warm for either.
pub(crate) static ALL: &[&SlangProgram] = &[
    &MAIN_BINDLESS_VERT,
    &MAIN_BINDLESS_FRAG,
    &LIGHT_CULL,
    &HIZ_INIT_SINGLE,
    &HIZ_INIT_MSAA,
    &HIZ_DOWNSAMPLE,
];

impl SlangProgram {
    // Assemble the exact source text this program compiles.
    pub fn source(&self, hot_reload: bool) -> String {
        crate::slang_source::assemble(hot_reload, self.file, self.embedded, self.defines)
    }

    fn target(&self) -> slang::SlangTarget {
        slang::SlangTarget::Dxil(self.profile)
    }

    // The shader-cache key for `source`. Shared by the runtime compile path
    // and the export-time precompile so the two can never key differently.
    pub(crate) fn cache_key<'a>(&self, source: &'a str) -> crate::shader_cache::Key<'a> {
        crate::shader_cache::Key {
            compiler: "slang",
            source,
            entry: self.entry,
            target: self.profile,
            options: 0,
        }
    }

    // Compile to DXIL, reusing a cached artifact when this exact assembled
    // source has been compiled before.
    pub fn compile(&self, hot_reload: bool) -> Result<Vec<u8>, String> {
        let source = self.source(hot_reload);
        let key = self.cache_key(&source);
        crate::shader_cache::cached(&key, self.label, || compile_uncached(self, &source))
    }
}

pub(super) fn compile_uncached(program: &SlangProgram, source: &str) -> Result<Vec<u8>, String> {
    let job = slang::SlangJob {
        source,
        file_name: program.file,
        entries: &[program.entry],
        target: program.target(),
    };
    slang::compile(&job, &crate::shader_cache::slang_work_dir())
}

// Compile every declared program into `out_dir`, reusing local cache artifacts
// where present. Called by the export-time precompile alongside the HLSL table.
pub(crate) fn precompile(out_dir: &std::path::Path, report: &mut crate::precompile::Report) {
    for program in ALL {
        let source = program.source(false);
        let key = program.cache_key(&source);
        report.record(
            &format!("{} {}", program.entry, program.profile),
            crate::shader_cache::ensure_in(out_dir, &key, || compile_uncached(program, &source)),
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
        assert!(!LIGHT_CULL.source(false).starts_with("#define"));
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

    // The DXIL_ABI probe cube array must be exactly as long as the descriptor
    // table the root signature binds, or a probe sample reads past it.
    #[test]
    fn probe_cube_count_matches_the_host_constant() {
        let want = crate::directx::probe_uniforms::MAX_PROBES.to_string();
        let value = MAIN_DEFINES
            .iter()
            .find(|(k, _)| *k == "MAX_PROBES")
            .map(|(_, v)| *v);
        assert_eq!(value, Some(want.as_str()));
    }

    // The single-source files carry cluster constants that must track the Rust
    // values the CPU sizes buffers with.
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

    // The sky shell's half-extent tracks the camera far plane, so its corners
    // always fall outside it: the bindless vertex path must pin sky verts to the
    // far plane or those corners clip and the clear colour shows through.
    #[test]
    fn the_bindless_vertex_path_pins_sky_to_the_far_plane() {
        assert!(MAIN_BINDLESS_SLANG.contains("v.color.b > 1.5"));
        assert!(MAIN_BINDLESS_SLANG.contains("o.position.z = o.position.w"));
    }
}
