// Single-source engine shader libraries for the Metal backend.
//
// Each entry names one metallib variant compiled from a `.slang` file under
// `src/shaders/` (the backend-neutral single-source directory). The fast path
// loads the metallib the build script precompiled; source compilation remains
// for hot-reload (disk edits must win) and for binaries built on a host
// without slangc, whose embedded lookup misses these names. The runtime
// compile assembles its source through `crate::slang_source`, invokes slangc
// through `concinnity-slang`, and caches the metallib in the content-addressed
// shader cache, so a given source text compiles at most once per machine.
//
// The `.slang` declarations reproduce the engine's Metal binding layout
// exactly (see main_bindless.slang's METAL_ABI block); the build script's
// `assert_slang_metal_abi` locks the emitted slot assignment.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLLibrary;

use concinnity_slang as slang;

use super::pipeline::load_library;

// The Metal texture-pool capacity and probe-array length the build script
// baked into the precompiled metallibs. Locked to the crate's own constants
// by `defines_match_the_crate_constants` below.
#[cfg(test)]
mod build_defines {
    include!(concat!(env!("OUT_DIR"), "/slang_metal_defines.rs"));
}

// One single-source metallib variant: the lookup `name` the build script
// registered, its `.slang` file, entry points, and variant defines.
pub(super) struct SlangLib {
    pub name: &'static str,
    pub file: &'static str,
    pub embedded: &'static str,
    pub entries: &'static [&'static str],
    pub defines: &'static [(&'static str, &'static str)],
}

const MAIN_DEFINES: &[(&str, &str)] = &[
    ("METAL_ABI", "1"),
    ("POOL_SIZE", "1024"),
    ("MAX_PROBES", "8"),
];

const MAIN_BINDLESS_SLANG: &str = include_str!("../shaders/main_bindless.slang");
const LIGHT_CULL_SLANG: &str = include_str!("../shaders/light_cull.slang");
const HIZ_BUILD_SLANG: &str = include_str!("../shaders/hiz_build.slang");
const FULLSCREEN_SLANG: &str = include_str!("../shaders/fullscreen.slang");
const TAA_SLANG: &str = include_str!("../shaders/taa.slang");
const BLOOM_SLANG: &str = include_str!("../shaders/bloom.slang");
const COMPOSITE_SLANG: &str = include_str!("../shaders/composite.slang");
const SSAO_SLANG: &str = include_str!("../shaders/ssao.slang");
const SSR_SLANG: &str = include_str!("../shaders/ssr.slang");
const SSGI_SLANG: &str = include_str!("../shaders/ssgi.slang");
const REFLECTION_SLANG: &str = include_str!("../shaders/reflection.slang");

// The SSR resolve reads the reflection-probe array, so it bakes in the same
// probe count the main pass does.
const PROBE_DEFINES: &[(&str, &str)] = &[("MAX_PROBES", "8")];

pub(super) static MAIN_BINDLESS_VERT: SlangLib = SlangLib {
    name: "main_bindless_vert.slang",
    file: "main_bindless.slang",
    embedded: MAIN_BINDLESS_SLANG,
    entries: &["vertex_main_bindless"],
    defines: MAIN_DEFINES,
};
pub(super) static MAIN_BINDLESS_FRAG: SlangLib = SlangLib {
    name: "main_bindless_frag.slang",
    file: "main_bindless.slang",
    embedded: MAIN_BINDLESS_SLANG,
    entries: &["fragment_main_bindless"],
    defines: MAIN_DEFINES,
};
pub(super) static LIGHT_CULL: SlangLib = SlangLib {
    name: "light_cull.slang",
    file: "light_cull.slang",
    embedded: LIGHT_CULL_SLANG,
    entries: &["light_cull_kernel"],
    defines: &[],
};
pub(super) static HIZ_INIT_MSAA: SlangLib = SlangLib {
    name: "hiz_init_msaa.slang",
    file: "hiz_build.slang",
    embedded: HIZ_BUILD_SLANG,
    entries: &["hiz_init_msaa"],
    defines: &[("HIZ_INIT_MSAA", "1")],
};
pub(super) static HIZ_DOWNSAMPLE: SlangLib = SlangLib {
    name: "hiz_downsample.slang",
    file: "hiz_build.slang",
    embedded: HIZ_BUILD_SLANG,
    entries: &["hiz_downsample"],
    defines: &[("HIZ_DOWNSAMPLE", "1")],
};

// The fullscreen-triangle vertex stage every ported post pass pairs with; one
// library serves them all, so their fragment metallibs carry no vertex entry.
pub(super) static FULLSCREEN_VERT: SlangLib = SlangLib {
    name: "fullscreen_vert.slang",
    file: "fullscreen.slang",
    embedded: FULLSCREEN_SLANG,
    entries: &["fullscreen_vertex"],
    defines: &[],
};
pub(super) static TAA_FRAG: SlangLib = SlangLib {
    name: "taa_frag.slang",
    file: "taa.slang",
    embedded: TAA_SLANG,
    entries: &["taa_fragment_main"],
    defines: &[],
};
pub(super) static BLOOM_PREFILTER: SlangLib = SlangLib {
    name: "bloom_prefilter.slang",
    file: "bloom.slang",
    embedded: BLOOM_SLANG,
    entries: &["bloom_prefilter_fragment"],
    defines: &[("BLOOM_PREFILTER", "1")],
};
pub(super) static BLOOM_DOWNSAMPLE: SlangLib = SlangLib {
    name: "bloom_downsample.slang",
    file: "bloom.slang",
    embedded: BLOOM_SLANG,
    entries: &["bloom_downsample_fragment"],
    defines: &[("BLOOM_DOWNSAMPLE", "1")],
};
pub(super) static BLOOM_UPSAMPLE: SlangLib = SlangLib {
    name: "bloom_upsample.slang",
    file: "bloom.slang",
    embedded: BLOOM_SLANG,
    entries: &["bloom_upsample_fragment"],
    defines: &[("BLOOM_UPSAMPLE", "1")],
};
pub(super) static COMPOSITE_FRAG: SlangLib = SlangLib {
    name: "composite_frag.slang",
    file: "composite.slang",
    embedded: COMPOSITE_SLANG,
    entries: &["composite_fragment"],
    defines: &[],
};
pub(super) static SSAO_KERNEL: SlangLib = SlangLib {
    name: "ssao_kernel.slang",
    file: "ssao.slang",
    embedded: SSAO_SLANG,
    entries: &["ssao_kernel_fragment"],
    defines: &[("SSAO_KERNEL", "1")],
};
pub(super) static SSAO_BLUR: SlangLib = SlangLib {
    name: "ssao_blur.slang",
    file: "ssao.slang",
    embedded: SSAO_SLANG,
    entries: &["ssao_blur_fragment"],
    defines: &[("SSAO_BLUR", "1")],
};
pub(super) static SSR_RESOLVE: SlangLib = SlangLib {
    name: "ssr_resolve.slang",
    file: "ssr.slang",
    embedded: SSR_SLANG,
    entries: &["ssr_resolve_fragment"],
    defines: PROBE_DEFINES,
};
pub(super) static SSGI_GATHER: SlangLib = SlangLib {
    name: "ssgi_gather.slang",
    file: "ssgi.slang",
    embedded: SSGI_SLANG,
    entries: &["ssgi_gather_fragment"],
    defines: &[("SSGI_GATHER", "1")],
};
pub(super) static SSGI_COMPOSITE: SlangLib = SlangLib {
    name: "ssgi_composite.slang",
    file: "ssgi.slang",
    embedded: SSGI_SLANG,
    entries: &["ssgi_composite_fragment"],
    defines: &[("SSGI_COMPOSITE", "1")],
};
pub(super) static REFLECTION_BLUR: SlangLib = SlangLib {
    name: "reflection_blur.slang",
    file: "reflection.slang",
    embedded: REFLECTION_SLANG,
    entries: &["reflection_blur_fragment"],
    defines: &[("REFLECTION_BLUR", "1")],
};
pub(super) static REFLECTION_COMPOSITE: SlangLib = SlangLib {
    name: "reflection_composite.slang",
    file: "reflection.slang",
    embedded: REFLECTION_SLANG,
    entries: &["reflection_composite_fragment"],
    defines: &[("REFLECTION_COMPOSITE", "1")],
};

// Every registered variant, for the coverage test in `metallib.rs`.
#[cfg(test)]
pub(super) static ALL: &[&SlangLib] = &[
    &MAIN_BINDLESS_VERT,
    &MAIN_BINDLESS_FRAG,
    &LIGHT_CULL,
    &HIZ_INIT_MSAA,
    &HIZ_DOWNSAMPLE,
    &FULLSCREEN_VERT,
    &TAA_FRAG,
    &BLOOM_PREFILTER,
    &BLOOM_DOWNSAMPLE,
    &BLOOM_UPSAMPLE,
    &COMPOSITE_FRAG,
    &SSAO_KERNEL,
    &SSAO_BLUR,
    &SSR_RESOLVE,
    &SSGI_GATHER,
    &SSGI_COMPOSITE,
    &REFLECTION_BLUR,
    &REFLECTION_COMPOSITE,
];

impl SlangLib {
    // The exact source text this variant compiles, assembled the way every
    // backend assembles it.
    fn source(&self, hot_reload: bool) -> String {
        crate::slang_source::assemble(hot_reload, self.file, self.embedded, self.defines)
    }

    // Produce this variant's MTLLibrary. Fast path: the metallib the build
    // script precompiled and embedded. Fallback (hot-reload, or a build host
    // without slangc): compile at runtime through the shader cache.
    pub(super) fn library(
        &self,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        hot_reload: bool,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
        if !hot_reload && let Some(bytes) = super::metallib::embedded_metallib(self.name) {
            return load_library(device, bytes)
                .map_err(|e| format!("{}: failed to load precompiled metallib: {e}", self.name));
        }
        let source = self.source(hot_reload);
        let entry = self.entries.join("+");
        let key = crate::shader_cache::Key {
            compiler: "slang",
            source: &source,
            entry: &entry,
            target: "metallib",
            options: 0,
        };
        let bytes = crate::shader_cache::cached(&key, self.name, || {
            let job = slang::SlangJob {
                source: &source,
                file_name: self.name,
                entries: self.entries,
                target: slang::SlangTarget::Metallib,
            };
            slang::compile(&job, &crate::shader_cache::slang_work_dir())
        })?;
        load_library(device, &bytes)
            .map_err(|e| format!("{}: metallib load failed: {e}", self.name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The values the build script baked into the precompiled metallibs must
    // match the capacities the host binds, or the shader indexes past what
    // the argument encoder wrote.
    #[test]
    fn defines_match_the_crate_constants() {
        use super::build_defines::{SLANG_METAL_MAX_PROBES, SLANG_METAL_POOL_SIZE};
        assert_eq!(
            SLANG_METAL_POOL_SIZE,
            super::super::context::BINDLESS_TEXTURE_COUNT
        );
        assert_eq!(SLANG_METAL_MAX_PROBES, super::super::uniforms::MAX_PROBES);
        let want_pool = SLANG_METAL_POOL_SIZE.to_string();
        let want_probes = SLANG_METAL_MAX_PROBES.to_string();
        for lib in [&MAIN_BINDLESS_VERT, &MAIN_BINDLESS_FRAG] {
            let value = |key: &str| lib.defines.iter().find(|(k, _)| *k == key).map(|(_, v)| *v);
            assert_eq!(value("POOL_SIZE"), Some(want_pool.as_str()));
            assert_eq!(value("MAX_PROBES"), Some(want_probes.as_str()));
        }
    }

    // Every variant assembles non-empty source with its defines up front, in
    // both hot-reload modes (embedded and disk must agree on existence), and
    // with every `{...}` fragment marker spliced -- an unreplaced marker would
    // reach slangc as a syntax error at renderer init rather than here.
    #[test]
    fn variants_assemble_source_with_their_defines() {
        for lib in ALL {
            for hot_reload in [false, true] {
                let src = lib.source(hot_reload);
                assert!(!src.trim().is_empty(), "{}: empty source", lib.name);
                assert!(
                    !src.contains("{POST_COMMON}"),
                    "{}: unspliced fragment marker",
                    lib.name
                );
                for (k, v) in lib.defines {
                    assert!(
                        src.starts_with('#') && src.contains(&format!("#define {k} {v}\n")),
                        "{}: missing injected define {k}",
                        lib.name
                    );
                }
            }
        }
    }

    // The reflection-cut constant in every single-source fragment that gates on
    // it must track the canonical Rust value, like the equivalent locks on the
    // MSL sources: the resolve, the composite blur, and the forward pass all
    // decide "does this surface reflect" from it and cannot disagree.
    #[test]
    fn reflection_roughness_cut_matches_canonical() {
        let expected = format!(
            "static const float REFLECTION_ROUGHNESS_CUT = {:?};",
            crate::gfx::ssr::REFLECTION_ROUGHNESS_CUT
        );
        for (name, src) in [
            ("main_bindless.slang", MAIN_BINDLESS_SLANG),
            ("ssr.slang", SSR_SLANG),
            ("reflection.slang", REFLECTION_SLANG),
        ] {
            assert!(
                src.contains(&expected),
                "{name} REFLECTION_ROUGHNESS_CUT drifted from \
                 concinnity_core::gfx::ssr::REFLECTION_ROUGHNESS_CUT"
            );
        }
    }

    // Every entry-point name the renderer looks up must exist in the source it
    // is declared against, so a rename on one side fails a test rather than a
    // pipeline build.
    #[test]
    fn entries_exist_in_their_sources() {
        for lib in ALL {
            for entry in lib.entries {
                assert!(
                    lib.embedded.contains(&format!(" {entry}(")),
                    "{}: entry {entry} not found in {}",
                    lib.name,
                    lib.file
                );
            }
        }
    }
}
