// src/vulkan/builtins.rs
//
// The declarative table of every built-in GLSL program the Vulkan backend
// compiles at runtime. Each program is declared exactly once: its source file,
// shader kind, target (default vs ray-query), and how the compile source is
// assembled (injected defines, probe_common / {MAX_PROBES} / {POOL_SIZE}
// substitution). Renderer init and hot-reload compile through
// `GlslProgram::compile`, and the export-time precompile iterates `ALL` to
// populate a bundle's shader cache from the very same declarations, so the two
// can never drift.
//
// Not declared here: the SdfVolume raymarch fragment shaders (raymarch.rs),
// whose source embeds world-authored shader text and therefore cannot be
// enumerated ahead of a world; they compile at init through the same cache.

use super::pipeline::{
    PROBE_COMMON_GLSL, compile_glsl, compile_glsl_rt, inject_define, shader_source,
};

// Size of the bindless texture pool for a world with `texture_count` entries
// in its texture table: one image per table slot (a single fallback when the
// table is empty) plus the flat-normal fallback. Baked into the pool-sized
// shaders via `{POOL_SIZE}`; init and the export-time precompile both derive
// the value here so a bundle's precompiled artifacts match its first launch.
pub(crate) fn bindless_pool_size(texture_count: usize) -> usize {
    texture_count.max(1) + 1
}

// How a program's compile source is assembled from its shader file. Defines
// inject immediately after `#version` (later injections land closer to it);
// substitutions then replace the body's `{...}` markers.
pub(crate) struct Assembly {
    // Inject `#define USE_MSAA {0|1}` from `Ctx::msaa`.
    pub msaa: bool,
    // Inject a fixed define line (kernel variants such as CULL_PHASE2).
    pub define: Option<&'static str>,
    // Substitute `{PROBE_COMMON}` / `{MAX_PROBES}` / `{PROBE_DESC_SET}`, with
    // this descriptor-set index string.
    pub probe_desc_set: Option<&'static str>,
    // Substitute `{POOL_SIZE}` from `Ctx::pool_size`.
    pub pool_size: bool,
}

const PLAIN: Assembly = Assembly {
    msaa: false,
    define: None,
    probe_desc_set: None,
    pool_size: false,
};
const MSAA: Assembly = Assembly {
    msaa: true,
    ..PLAIN
};

impl Assembly {
    // MSAA values this assembly can produce, for the export-time enumeration.
    fn msaa_variants(&self) -> &'static [bool] {
        if self.msaa { &[false, true] } else { &[false] }
    }
}

// Inputs a call site supplies to assemble a program's source.
pub(crate) struct Ctx {
    pub hot_reload: bool,
    pub msaa: bool,
    // Bindless texture-pool length for `{POOL_SIZE}` programs; ignored by the
    // rest. Callers pass the live pool size (see `bindless_pool_size`).
    pub pool_size: usize,
    // Reflection-probe cube-array length for `{MAX_PROBES}` programs; ignored by
    // the rest. Callers pass the descriptor count the global set layout was
    // built with (`descriptor_layout::probe_cube_array_count`), so the GLSL array
    // and the layout binding always agree.
    pub probe_count: usize,
}

impl Ctx {
    // For programs whose assembly needs no MSAA state, pool size, or probe count.
    pub fn plain(hot_reload: bool) -> Self {
        Self {
            hot_reload,
            msaa: false,
            pool_size: 0,
            probe_count: 0,
        }
    }
}

pub(crate) struct GlslProgram {
    // File name under `src/vulkan/shaders/` for the `cn debug` disk-first
    // resolve; also the embedded fallback's origin.
    pub file: &'static str,
    pub embedded: &'static str,
    pub kind: shaderc::ShaderKind,
    // Diagnostic label passed to shaderc (compile errors + cache miss logs).
    pub label: &'static str,
    // Compile with the ray-query target (`compile_glsl_rt`, Vulkan 1.2 /
    // SPIR-V 1.4) instead of the default Vulkan 1.0 target.
    pub rt: bool,
    pub assembly: Assembly,
}

impl GlslProgram {
    // Assemble the exact source text this program compiles under `ctx`.
    pub fn source(&self, ctx: &Ctx) -> String {
        let mut src = shader_source(ctx.hot_reload, self.file, self.embedded).into_owned();
        if self.assembly.msaa {
            src = inject_define(&src, msaa_define(ctx.msaa));
        }
        if let Some(define) = self.assembly.define {
            src = inject_define(&src, define);
        }
        if let Some(set) = self.assembly.probe_desc_set {
            debug_assert!(
                ctx.probe_count > 0,
                "{}: probe program assembled with no probe count",
                self.label
            );
            let probe_common =
                shader_source(ctx.hot_reload, "probe_common.glsl", PROBE_COMMON_GLSL);
            src = src
                .replace("{PROBE_COMMON}", &probe_common)
                .replace("{MAX_PROBES}", &ctx.probe_count.to_string())
                .replace("{PROBE_DESC_SET}", set);
        }
        if self.assembly.pool_size {
            src = src.replace("{POOL_SIZE}", &ctx.pool_size.to_string());
        }
        src
    }

    pub fn compile(&self, ctx: &Ctx) -> Result<Vec<u8>, String> {
        let source = self.source(ctx);
        if self.rt {
            compile_glsl_rt(&source, self.kind, self.label)
        } else {
            compile_glsl(&source, self.kind, self.label)
        }
    }
}

fn msaa_define(msaa: bool) -> &'static str {
    if msaa {
        "#define USE_MSAA 1\n"
    } else {
        "#define USE_MSAA 0\n"
    }
}

// Compile every declared program (all enumerable variants) into `out_dir`,
// reusing local cache artifacts where present. `texture_count` sizes the
// `{POOL_SIZE}` substitution exactly as the exported world's first launch will.
// The probe cube-array length is a property of the device the bundle eventually
// runs on, so the probe programs are baked at the `MAX_PROBES` ceiling every
// desktop driver affords; a device that reports less per-stage sampler headroom
// (MoltenVK) simply misses these entries and compiles them at first launch.
pub(crate) fn precompile(
    out_dir: &std::path::Path,
    texture_count: usize,
    report: &mut crate::precompile::Report,
) {
    let pool_size = bindless_pool_size(texture_count);
    for program in ALL {
        for &msaa in program.assembly.msaa_variants() {
            let ctx = Ctx {
                hot_reload: false,
                msaa,
                pool_size,
                probe_count: super::probe_uniforms::MAX_PROBES,
            };
            let source = program.source(&ctx);
            let key = if program.rt {
                super::pipeline::glsl_rt_cache_key(&source, program.kind)
            } else {
                super::pipeline::glsl_cache_key(&source, program.kind)
            };
            let compile = || {
                if program.rt {
                    compile_glsl_rt(&source, program.kind, program.label)
                } else {
                    compile_glsl(&source, program.kind, program.label)
                }
            };
            report.record(
                program.label,
                crate::shader_cache::ensure_in(out_dir, &key, compile),
            );
        }
    }
}

// Declaration shorthand: default target, no assembly.
const fn glsl(
    file: &'static str,
    embedded: &'static str,
    kind: shaderc::ShaderKind,
    label: &'static str,
) -> GlslProgram {
    GlslProgram {
        file,
        embedded,
        kind,
        label,
        rt: false,
        assembly: PLAIN,
    }
}

use shaderc::ShaderKind::{Compute, Fragment, Vertex};

// Embedded sources shared by several programs.
const CULL_COMPUTE_GLSL: &str = include_str!("shaders/cull.comp");
const GLASS_VERT_GLSL: &str = include_str!("shaders/glass.vert");

pub(super) static MAIN_VERT: GlslProgram = glsl(
    "main.vert",
    include_str!("shaders/main.vert"),
    Vertex,
    "vert.glsl",
);
pub(super) static MAIN_FRAG: GlslProgram = glsl(
    "main.frag",
    include_str!("shaders/main.frag"),
    Fragment,
    "frag.glsl",
);
pub(super) static MAIN_BINDLESS_VERT: GlslProgram = glsl(
    "main_bindless.vert",
    include_str!("shaders/main_bindless.vert"),
    Vertex,
    "vert_bindless.glsl",
);
pub(super) static MAIN_BINDLESS_FRAG: GlslProgram = GlslProgram {
    assembly: Assembly {
        probe_desc_set: Some("0"),
        pool_size: true,
        ..PLAIN
    },
    ..glsl(
        "main_bindless.frag",
        include_str!("shaders/main_bindless.frag"),
        Fragment,
        "frag_bindless.glsl",
    )
};
pub(super) static MAIN_VERT_INSTANCED: GlslProgram = glsl(
    "instanced.vert",
    include_str!("shaders/instanced.vert"),
    Vertex,
    "vert_instanced.glsl",
);
pub(super) static SHADOW_VERT: GlslProgram = glsl(
    "shadow.vert",
    include_str!("shaders/shadow.vert"),
    Vertex,
    "shadow_vert.glsl",
);
pub(super) static SHADOW_BINDLESS_VERT: GlslProgram = glsl(
    "shadow_bindless.vert",
    include_str!("shaders/shadow_bindless.vert"),
    Vertex,
    "shadow_bindless.vert",
);
pub(super) static SKINNED_VERT: GlslProgram = glsl(
    "skinned.vert",
    include_str!("shaders/skinned.vert"),
    Vertex,
    "skinned_vert.glsl",
);
pub(super) static SKINNED_SHADOW_VERT: GlslProgram = glsl(
    "skinned_shadow.vert",
    include_str!("shaders/skinned_shadow.vert"),
    Vertex,
    "skinned_shadow_vert.glsl",
);

pub(super) static CULL: GlslProgram =
    glsl("cull.comp", CULL_COMPUTE_GLSL, Compute, "cull_compute.glsl");
pub(super) static CULL_PHASE2: GlslProgram = GlslProgram {
    assembly: Assembly {
        define: Some("#define CULL_PHASE2 1\n"),
        ..PLAIN
    },
    ..glsl(
        "cull.comp",
        CULL_COMPUTE_GLSL,
        Compute,
        "cull_compute_phase2.glsl",
    )
};
pub(super) static CULL_SHADOW: GlslProgram = GlslProgram {
    assembly: Assembly {
        define: Some("#define SHADOW_CULL 1\n"),
        ..PLAIN
    },
    ..glsl(
        "cull.comp",
        CULL_COMPUTE_GLSL,
        Compute,
        "cull_compute_shadow.glsl",
    )
};

pub(super) static TEXT_VERT: GlslProgram = glsl(
    "text.vert",
    include_str!("shaders/text.vert"),
    Vertex,
    "text_vert.glsl",
);
pub(super) static TEXT_FRAG: GlslProgram = glsl(
    "text.frag",
    include_str!("shaders/text.frag"),
    Fragment,
    "text_frag.glsl",
);
pub(super) static COMPOSITE_VERT: GlslProgram = glsl(
    "composite.vert",
    include_str!("shaders/composite.vert"),
    Vertex,
    "composite_vert.glsl",
);
pub(super) static COMPOSITE_FRAG: GlslProgram = glsl(
    "composite.frag",
    include_str!("shaders/composite.frag"),
    Fragment,
    "composite_frag.glsl",
);

pub(super) static GBUFFER_PREPASS_VERT: GlslProgram = glsl(
    "gbuffer_prepass.vert",
    include_str!("shaders/gbuffer_prepass.vert"),
    Vertex,
    "gbuffer_prepass.vert",
);
pub(super) static GBUFFER_PREPASS_VERT_INSTANCED: GlslProgram = glsl(
    "gbuffer_prepass_instanced.vert",
    include_str!("shaders/gbuffer_prepass_instanced.vert"),
    Vertex,
    "gbuffer_prepass_instanced.vert",
);
pub(super) static GBUFFER_PREPASS_VERT_SKINNED: GlslProgram = glsl(
    "gbuffer_prepass_skinned.vert",
    include_str!("shaders/gbuffer_prepass_skinned.vert"),
    Vertex,
    "gbuffer_prepass_skinned.vert",
);
pub(super) static GBUFFER_PREPASS_FRAG: GlslProgram = glsl(
    "gbuffer_prepass.frag",
    include_str!("shaders/gbuffer_prepass.frag"),
    Fragment,
    "gbuffer_prepass.frag",
);
pub(super) static GBUFFER_BINDLESS_VERT: GlslProgram = glsl(
    "gbuffer_bindless.vert",
    include_str!("shaders/gbuffer_bindless.vert"),
    Vertex,
    "gbuffer_bindless.vert",
);
pub(super) static GBUFFER_BINDLESS_FRAG: GlslProgram = glsl(
    "gbuffer_bindless.frag",
    include_str!("shaders/gbuffer_bindless.frag"),
    Fragment,
    "gbuffer_bindless.frag",
);

pub(super) static SSAO_FULLSCREEN_VERT: GlslProgram = glsl(
    "ssao_fullscreen.vert",
    include_str!("shaders/ssao_fullscreen.vert"),
    Vertex,
    "ssao_fullscreen.vert",
);
pub(super) static SSAO_KERNEL_FRAG: GlslProgram = glsl(
    "ssao_kernel.frag",
    include_str!("shaders/ssao_kernel.frag"),
    Fragment,
    "ssao_kernel.frag",
);
pub(super) static SSAO_BLUR_FRAG: GlslProgram = glsl(
    "ssao_blur.frag",
    include_str!("shaders/ssao_blur.frag"),
    Fragment,
    "ssao_blur.frag",
);

pub(super) static SSGI_FULLSCREEN_VERT: GlslProgram = glsl(
    "ssgi_fullscreen.vert",
    include_str!("shaders/ssgi_fullscreen.vert"),
    Vertex,
    "ssgi_fullscreen.vert",
);
pub(super) static SSGI_GATHER_FRAG: GlslProgram = glsl(
    "ssgi_gather.frag",
    include_str!("shaders/ssgi_gather.frag"),
    Fragment,
    "ssgi_gather.frag",
);
pub(super) static SSGI_COMPOSITE_FRAG: GlslProgram = glsl(
    "ssgi_composite.frag",
    include_str!("shaders/ssgi_composite.frag"),
    Fragment,
    "ssgi_composite.frag",
);

pub(super) static SSR_FULLSCREEN_VERT: GlslProgram = glsl(
    "ssr_fullscreen.vert",
    include_str!("shaders/ssr_fullscreen.vert"),
    Vertex,
    "ssr_fullscreen.vert",
);
pub(super) static SSR_RESOLVE_FRAG: GlslProgram = GlslProgram {
    assembly: Assembly {
        probe_desc_set: Some("1"),
        ..PLAIN
    },
    ..glsl(
        "ssr_resolve.frag",
        include_str!("shaders/ssr_resolve.frag"),
        Fragment,
        "ssr_resolve.frag",
    )
};

pub(super) static BLOOM_PREFILTER: GlslProgram = glsl(
    "bloom_prefilter.frag",
    include_str!("shaders/bloom_prefilter.frag"),
    Fragment,
    "bloom_prefilter.glsl",
);
pub(super) static BLOOM_DOWNSAMPLE: GlslProgram = glsl(
    "bloom_downsample.frag",
    include_str!("shaders/bloom_downsample.frag"),
    Fragment,
    "bloom_downsample.glsl",
);
pub(super) static BLOOM_UPSAMPLE: GlslProgram = glsl(
    "bloom_upsample.frag",
    include_str!("shaders/bloom_upsample.frag"),
    Fragment,
    "bloom_upsample.glsl",
);

pub(super) static TAA_FRAG: GlslProgram = glsl(
    "taa.frag",
    include_str!("shaders/taa.frag"),
    Fragment,
    "taa_frag.glsl",
);

pub(super) static REFLECTION_BLUR_FRAG: GlslProgram = glsl(
    "reflection_blur.frag",
    include_str!("shaders/reflection_blur.frag"),
    Fragment,
    "reflection_blur.frag",
);
pub(super) static REFLECTION_COMPOSITE_FRAG: GlslProgram = glsl(
    "reflection_composite.frag",
    include_str!("shaders/reflection_composite.frag"),
    Fragment,
    "reflection_composite.frag",
);

pub(super) static PARTICLE_SIMULATE: GlslProgram = glsl(
    "particle_simulate.comp",
    include_str!("shaders/particle_simulate.comp"),
    Compute,
    "particle_simulate.comp",
);
pub(super) static PARTICLE_VERT: GlslProgram = glsl(
    "particle.vert",
    include_str!("shaders/particle.vert"),
    Vertex,
    "particle.vert",
);
pub(super) static PARTICLE_FRAG: GlslProgram = glsl(
    "particle.frag",
    include_str!("shaders/particle.frag"),
    Fragment,
    "particle.frag",
);

pub(super) static AUTO_EXPOSURE_BUILD: GlslProgram = glsl(
    "auto_exposure_build.comp",
    include_str!("shaders/auto_exposure_build.comp"),
    Compute,
    "auto_exposure_build.comp",
);
pub(super) static AUTO_EXPOSURE_AVERAGE: GlslProgram = glsl(
    "auto_exposure_average.comp",
    include_str!("shaders/auto_exposure_average.comp"),
    Compute,
    "auto_exposure_average.comp",
);

pub(super) static HIZ_INIT: GlslProgram = GlslProgram {
    assembly: MSAA,
    ..glsl(
        "hiz_init.comp",
        include_str!("shaders/hiz_init.comp"),
        Compute,
        "hiz_init.glsl",
    )
};
pub(super) static HIZ_DOWNSAMPLE: GlslProgram = glsl(
    "hiz_downsample.comp",
    include_str!("shaders/hiz_downsample.comp"),
    Compute,
    "hiz_downsample.glsl",
);

pub(super) static FOG_VERT: GlslProgram = GlslProgram {
    assembly: MSAA,
    ..glsl(
        "fog.vert",
        include_str!("shaders/fog.vert"),
        Vertex,
        "fog.vert",
    )
};
pub(super) static FOG_FRAG: GlslProgram = GlslProgram {
    assembly: MSAA,
    ..glsl(
        "fog.frag",
        include_str!("shaders/fog.frag"),
        Fragment,
        "fog.frag",
    )
};
pub(super) static FOG_FROXEL: GlslProgram = glsl(
    "fog_froxel.comp",
    include_str!("shaders/fog_froxel.comp"),
    Compute,
    "fog_froxel.comp",
);

pub(super) static DECAL_VERT: GlslProgram = GlslProgram {
    assembly: MSAA,
    ..glsl(
        "decal.vert",
        include_str!("shaders/decal.vert"),
        Vertex,
        "decal.vert",
    )
};
pub(super) static DECAL_FRAG: GlslProgram = GlslProgram {
    assembly: MSAA,
    ..glsl(
        "decal.frag",
        include_str!("shaders/decal.frag"),
        Fragment,
        "decal.frag",
    )
};

pub(super) static LINE_VERT: GlslProgram = glsl(
    "line.vert",
    include_str!("shaders/line.vert"),
    Vertex,
    "line.vert",
);
pub(super) static LINE_FRAG: GlslProgram = GlslProgram {
    assembly: MSAA,
    ..glsl(
        "line.frag",
        include_str!("shaders/line.frag"),
        Fragment,
        "line.frag",
    )
};

pub(super) static LIGHT_CULL: GlslProgram = glsl(
    "light_cull.comp",
    include_str!("shaders/light_cull.comp"),
    Compute,
    "light_cull.comp",
);

pub(super) static GLASS_VERT: GlslProgram = GlslProgram {
    assembly: MSAA,
    ..glsl("glass.vert", GLASS_VERT_GLSL, Vertex, "glass.vert")
};
pub(super) static GLASS_FRAG: GlslProgram = GlslProgram {
    assembly: Assembly {
        msaa: true,
        probe_desc_set: Some("2"),
        ..PLAIN
    },
    ..glsl(
        "glass.frag",
        include_str!("shaders/glass.frag"),
        Fragment,
        "glass.frag",
    )
};

// Ray-query (Vulkan 1.2 / SPIR-V 1.4) programs.
pub(super) static GLASS_RT_VERT: GlslProgram = GlslProgram {
    rt: true,
    assembly: MSAA,
    ..glsl("glass.vert", GLASS_VERT_GLSL, Vertex, "glass.vert")
};
pub(super) static GLASS_RT_FRAG: GlslProgram = GlslProgram {
    rt: true,
    assembly: Assembly {
        msaa: true,
        probe_desc_set: Some("2"),
        pool_size: true,
        ..PLAIN
    },
    ..glsl(
        "glass_rt.frag",
        include_str!("shaders/glass_rt.frag"),
        Fragment,
        "glass_rt.frag",
    )
};
pub(super) static GLASS_RT_FRAG_TEXTURED: GlslProgram = GlslProgram {
    rt: true,
    assembly: Assembly {
        msaa: true,
        define: Some("#define RT_TEXTURED 1\n"),
        probe_desc_set: Some("2"),
        pool_size: true,
    },
    ..glsl(
        "glass_rt.frag",
        include_str!("shaders/glass_rt.frag"),
        Fragment,
        "glass_rt_textured.frag",
    )
};
pub(super) static RT_FULLSCREEN_VERT: GlslProgram = GlslProgram {
    rt: true,
    ..glsl(
        "rt_reflections.vert",
        include_str!("shaders/rt_reflections.vert"),
        Vertex,
        "rt_reflections.vert",
    )
};
pub(super) static RT_REFLECTIONS_FRAG: GlslProgram = GlslProgram {
    rt: true,
    assembly: Assembly {
        probe_desc_set: Some("1"),
        pool_size: true,
        ..PLAIN
    },
    ..glsl(
        "rt_reflections.frag",
        include_str!("shaders/rt_reflections.frag"),
        Fragment,
        "rt_reflections.frag",
    )
};
pub(super) static RT_REFLECTIONS_FRAG_TEXTURED: GlslProgram = GlslProgram {
    rt: true,
    assembly: Assembly {
        define: Some("#define RT_TEXTURED 1\n"),
        probe_desc_set: Some("1"),
        pool_size: true,
        ..PLAIN
    },
    ..glsl(
        "rt_reflections.frag",
        include_str!("shaders/rt_reflections.frag"),
        Fragment,
        "rt_reflections_textured.frag",
    )
};
pub(super) static RT_SKIN: GlslProgram = GlslProgram {
    rt: true,
    ..glsl(
        "rt_skin.comp",
        include_str!("shaders/rt_skin.comp"),
        Compute,
        "rt_skin.comp",
    )
};

// The SdfVolume proxy vertex shaders are pure engine text (only the fragment
// side embeds user source), so they are enumerable.
pub(super) static RAYMARCH_PROXY_VERT: GlslProgram = glsl(
    "raymarch_proxy.vert",
    include_str!("shaders/raymarch_proxy.vert"),
    Vertex,
    "raymarch_proxy.vert",
);
pub(super) static RAYMARCH_SHADOW_PROXY_VERT: GlslProgram = glsl(
    "raymarch_shadow_proxy.vert",
    include_str!("shaders/raymarch_shadow_proxy.vert"),
    Vertex,
    "raymarch_shadow_proxy.vert",
);

// Every declared program, iterated by the export-time precompile.
pub(crate) static ALL: &[&GlslProgram] = &[
    &MAIN_VERT,
    &MAIN_FRAG,
    &MAIN_BINDLESS_VERT,
    &MAIN_BINDLESS_FRAG,
    &MAIN_VERT_INSTANCED,
    &SHADOW_VERT,
    &SHADOW_BINDLESS_VERT,
    &SKINNED_VERT,
    &SKINNED_SHADOW_VERT,
    &CULL,
    &CULL_PHASE2,
    &CULL_SHADOW,
    &TEXT_VERT,
    &TEXT_FRAG,
    &COMPOSITE_VERT,
    &COMPOSITE_FRAG,
    &GBUFFER_PREPASS_VERT,
    &GBUFFER_PREPASS_VERT_INSTANCED,
    &GBUFFER_PREPASS_VERT_SKINNED,
    &GBUFFER_PREPASS_FRAG,
    &GBUFFER_BINDLESS_VERT,
    &GBUFFER_BINDLESS_FRAG,
    &SSAO_FULLSCREEN_VERT,
    &SSAO_KERNEL_FRAG,
    &SSAO_BLUR_FRAG,
    &SSGI_FULLSCREEN_VERT,
    &SSGI_GATHER_FRAG,
    &SSGI_COMPOSITE_FRAG,
    &SSR_FULLSCREEN_VERT,
    &SSR_RESOLVE_FRAG,
    &BLOOM_PREFILTER,
    &BLOOM_DOWNSAMPLE,
    &BLOOM_UPSAMPLE,
    &TAA_FRAG,
    &REFLECTION_BLUR_FRAG,
    &REFLECTION_COMPOSITE_FRAG,
    &PARTICLE_SIMULATE,
    &PARTICLE_VERT,
    &PARTICLE_FRAG,
    &AUTO_EXPOSURE_BUILD,
    &AUTO_EXPOSURE_AVERAGE,
    &HIZ_INIT,
    &HIZ_DOWNSAMPLE,
    &FOG_VERT,
    &FOG_FRAG,
    &FOG_FROXEL,
    &DECAL_VERT,
    &DECAL_FRAG,
    &LINE_VERT,
    &LINE_FRAG,
    &LIGHT_CULL,
    &GLASS_VERT,
    &GLASS_FRAG,
    &GLASS_RT_VERT,
    &GLASS_RT_FRAG,
    &GLASS_RT_FRAG_TEXTURED,
    &RT_FULLSCREEN_VERT,
    &RT_REFLECTIONS_FRAG,
    &RT_REFLECTIONS_FRAG_TEXTURED,
    &RT_SKIN,
    &RAYMARCH_PROXY_VERT,
    &RAYMARCH_SHADOW_PROXY_VERT,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_size_counts_fallbacks() {
        // Empty table: the fallback white texture + the flat-normal fallback.
        assert_eq!(bindless_pool_size(0), 2);
        assert_eq!(bindless_pool_size(1), 2);
        assert_eq!(bindless_pool_size(7), 8);
    }

    // Two programs collide when they would compile identical source text with
    // the same kind and target; the table must not declare the same slot twice.
    #[test]
    fn table_has_no_duplicate_programs() {
        let mut seen = std::collections::HashSet::new();
        for p in ALL {
            let ctx = Ctx {
                hot_reload: false,
                msaa: false,
                pool_size: 4,
                probe_count: 4,
            };
            assert!(
                seen.insert((p.source(&ctx), p.kind as u32, p.rt)),
                "duplicate program: {}",
                p.label
            );
        }
    }

    #[test]
    fn defines_inject_after_the_version_directive() {
        let ctx = Ctx {
            hot_reload: false,
            msaa: true,
            pool_size: 3,
            probe_count: 3,
        };
        let src = HIZ_INIT.source(&ctx);
        let mut lines = src.lines();
        assert!(lines.next().unwrap().starts_with("#version"));
        assert_eq!(lines.next().unwrap(), "#define USE_MSAA 1");

        // A fixed define lands closer to #version than the MSAA define,
        // matching the historical assembly order.
        let src = GLASS_RT_FRAG_TEXTURED.source(&ctx);
        let mut lines = src.lines();
        assert!(lines.next().unwrap().starts_with("#version"));
        assert_eq!(lines.next().unwrap(), "#define RT_TEXTURED 1");
        assert_eq!(lines.next().unwrap(), "#define USE_MSAA 1");
    }

    #[test]
    fn substitutions_resolve_every_marker() {
        let ctx = Ctx {
            hot_reload: false,
            msaa: false,
            pool_size: 5,
            probe_count: 5,
        };
        for p in [&MAIN_BINDLESS_FRAG, &SSR_RESOLVE_FRAG, &GLASS_RT_FRAG] {
            let src = p.source(&ctx);
            for marker in [
                "{PROBE_COMMON}",
                "{MAX_PROBES}",
                "{PROBE_DESC_SET}",
                "{POOL_SIZE}",
            ] {
                assert!(!src.contains(marker), "{} left {}", p.label, marker);
            }
        }
    }

    // Every probe program sizes its cube array from the context, so the GLSL
    // array length tracks whatever descriptor count the global set layout was
    // built with instead of a compile-time constant the device may not afford.
    #[test]
    fn probe_programs_size_their_cube_array_from_the_context() {
        for probe_count in [1usize, 7, super::super::probe_uniforms::MAX_PROBES] {
            let ctx = Ctx {
                hot_reload: false,
                msaa: false,
                pool_size: 4,
                probe_count,
            };
            for p in ALL.iter().filter(|p| p.assembly.probe_desc_set.is_some()) {
                let src = p.source(&ctx);
                assert!(
                    src.contains(&format!("probe_cubes[{probe_count}]")),
                    "{} did not size its cube array at {probe_count}",
                    p.label
                );
            }
        }
    }

    #[test]
    fn msaa_programs_enumerate_both_variants() {
        assert_eq!(FOG_VERT.assembly.msaa_variants(), &[false, true]);
        assert_eq!(GLASS_RT_FRAG.assembly.msaa_variants(), &[false, true]);
        assert_eq!(CULL_PHASE2.assembly.msaa_variants(), &[false]);
        assert_eq!(RT_REFLECTIONS_FRAG.assembly.msaa_variants(), &[false]);
    }
}
