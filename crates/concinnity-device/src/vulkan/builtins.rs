// src/vulkan/builtins.rs
//
// The declarative table of every built-in GLSL program the Vulkan backend
// compiles at runtime. Each program is declared exactly once: its source file,
// shader kind, target (default vs ray-query), and how the compile source is
// assembled (injected defines, probe_common / object_common / {MAX_PROBES} /
// {POOL_SIZE} substitution). Renderer init and hot-reload compile through
// `GlslProgram::compile`, and the export-time precompile iterates `ALL` to
// populate a bundle's shader cache from the very same declarations, so the two
// can never drift.
//
// Not declared here: the SdfVolume raymarch fragment shaders (raymarch.rs),
// whose source embeds world-authored shader text and therefore cannot be
// enumerated ahead of a world; they compile at init through the same cache.

use super::pipeline::{
    OBJECT_COMMON_GLSL, compile_glsl, compile_glsl_rt, inject_define, shader_source,
};

// Slots a world with `texture_count` table entries needs: one image per slot (a
// single fallback when the table is empty) plus the reserved fallbacks,
// flat-normal and white. What the pool was always sized to, and what a device
// that cannot seat the ceiling still falls back to.
pub(crate) fn world_pool_size(texture_count: usize) -> usize {
    texture_count.max(1) + crate::gfx::render_types::FALLBACK_TEXTURE_COUNT
}

// The pool length a device declares, baked into the pool-sized shaders via
// `{POOL_SIZE}`.
//
// The ceiling wherever it fits, which is every desktop driver, so the shader
// text is a property of the build rather than of the world -- which is what
// lets the build script compile these programs ahead of any world. Where it
// does not fit the pool is sized to the world exactly as it always was, and the
// differing `POOL_SIZE` makes that source miss the precompiled artifacts and
// compile, through the same content check every other artifact goes through.
//
// `ceiling_fits` is the device's answer (see `vulkan::init`); the export-time
// precompile has no device and bakes the ceiling, which is what a bundle's
// eventual host will use unless it is one of the constrained ones.
pub(crate) fn bindless_pool_size(texture_count: usize, ceiling_fits: bool) -> usize {
    if ceiling_fits {
        concinnity_core::render::uniforms::BINDLESS_POOL_SIZE
    } else {
        world_pool_size(texture_count)
    }
}

// How a program's compile source is assembled from its shader file. Defines
// inject immediately after `#version` (later injections land closer to it);
// substitutions then replace the body's `{...}` markers.
pub(crate) struct Assembly {
    // Inject a fixed define line (kernel variants such as CULL_PHASE2).
    pub define: Option<&'static str>,
    // Substitute `{POOL_SIZE}` from `Ctx::pool_size`.
    pub pool_size: bool,
    // Substitute `{OBJECT_DATA}` with the shared `GpuObjectData` declaration.
    pub object_data: bool,
}

const PLAIN: Assembly = Assembly {
    define: None,
    pool_size: false,
    object_data: false,
};

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
    pub(crate) fn plain(hot_reload: bool) -> Self {
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
    pub(crate) fn source(&self, ctx: &Ctx) -> String {
        let mut src = shader_source(ctx.hot_reload, self.file, self.embedded).into_owned();
        if let Some(define) = self.assembly.define {
            src = inject_define(&src, define);
        }
        if self.assembly.pool_size {
            src = src.replace("{POOL_SIZE}", &ctx.pool_size.to_string());
        }
        if self.assembly.object_data {
            let object_common =
                shader_source(ctx.hot_reload, "object_common.glsl", OBJECT_COMMON_GLSL);
            src = src.replace("{OBJECT_DATA}", &object_common);
        }
        src
    }

    pub(crate) fn compile(&self, ctx: &Ctx) -> Result<Vec<u8>, String> {
        let source = self.source(ctx);
        if self.rt {
            compile_glsl_rt(&source, self.kind, self.label)
        } else {
            compile_glsl(&source, self.kind, self.label)
        }
    }
}

// Compile every declared program (all enumerable variants) into `bundle`,
// reusing local cache artifacts where present.
//
// Both the pool length and the probe cube-array length are properties of the
// device the bundle eventually runs on rather than of the world, so both are
// baked at the ceiling every desktop driver affords. A device that reports less
// headroom than that (MoltenVK) simply misses these entries and compiles them
// at first launch.
pub(crate) fn precompile(
    bundle: &mut concinnity_host::store::cache::Segment,
    report: &mut crate::precompile::Report,
) {
    let pool_size = concinnity_core::render::uniforms::BINDLESS_POOL_SIZE;
    for program in ALL {
        let ctx = Ctx {
            hot_reload: false,
            msaa: false,
            pool_size,
            probe_count: concinnity_core::render::uniforms::MAX_PROBES,
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
            crate::shader_cache::ensure_in(bundle, &key, compile),
        );
    }

    // The single-source programs precompile through the same cache under their
    // own compiler id, so a bundle is warm for them too. A program whose source
    // reads the main pass's sample count gets both variants, for the same
    // reason the GLSL loop above enumerates them: which one a device runs is a
    // property of its MSAA mode, not of the bundle.
    for program in super::slang_builtins::ALL {
        let msaa_variants: &[bool] = if program.msaa {
            &[false, true]
        } else {
            &[false]
        };
        for &msaa in msaa_variants {
            let ctx = Ctx {
                hot_reload: false,
                msaa,
                pool_size,
                probe_count: concinnity_core::render::uniforms::MAX_PROBES,
            };
            let source = program.source(&ctx);
            let key = program.cache_key(&source);
            report.record(
                program.label,
                crate::shader_cache::ensure_in(bundle, &key, || {
                    super::slang_builtins::compile_uncached(program, &source)
                }),
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

use crate::vulkan::slang_builtins::SlangCompile;
use shaderc::ShaderKind::{Compute, Fragment, Vertex};

// Embedded sources shared by several programs.
const CULL_COMPUTE_GLSL: &str = include_str!("shaders/cull.comp");

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
pub(super) static MAIN_VERT_INSTANCED: GlslProgram = glsl(
    "instanced.vert",
    include_str!("shaders/instanced.vert"),
    Vertex,
    "vert_instanced.glsl",
);
pub(super) static SKINNED_VERT: GlslProgram = glsl(
    "skinned.vert",
    include_str!("shaders/skinned.vert"),
    Vertex,
    "skinned_vert.glsl",
);

pub(super) static CULL: GlslProgram = GlslProgram {
    assembly: Assembly {
        object_data: true,
        ..PLAIN
    },
    ..glsl("cull.comp", CULL_COMPUTE_GLSL, Compute, "cull_compute.glsl")
};
pub(super) static CULL_PHASE2: GlslProgram = GlslProgram {
    assembly: Assembly {
        define: Some("#define CULL_PHASE2 1\n"),
        object_data: true,
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
        object_data: true,
        ..PLAIN
    },
    ..glsl(
        "cull.comp",
        CULL_COMPUTE_GLSL,
        Compute,
        "cull_compute_shadow.glsl",
    )
};

// Ray-query (Vulkan 1.2 / SPIR-V 1.4) programs.

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
    &MAIN_VERT_INSTANCED,
    &SKINNED_VERT,
    &CULL,
    &CULL_PHASE2,
    &CULL_SHADOW,
    &RAYMARCH_PROXY_VERT,
    &RAYMARCH_SHADOW_PROXY_VERT,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_size_counts_fallbacks() {
        // One slot per table entry (an empty table still pads to one) plus the
        // two reserved fallbacks, flat-normal then white.
        assert_eq!(world_pool_size(0), 3);
        assert_eq!(world_pool_size(1), 3);
        assert_eq!(world_pool_size(7), 9);
    }

    // The uploaded image vectors reproduce the world-sized pool exactly: init
    // pads an empty texture table to one image and always uploads the reserved
    // fallbacks alongside it. A raw texture count is never a valid pool length,
    // so a compile handed one silently drops the last slots.
    #[test]
    fn world_pool_size_matches_the_uploaded_image_counts() {
        for texture_count in [0usize, 1, 7, 64] {
            let gpu_textures = texture_count.max(1);
            assert_eq!(
                world_pool_size(texture_count),
                gpu_textures + crate::gfx::render_types::FALLBACK_TEXTURE_COUNT
            );
            assert!(world_pool_size(texture_count) > texture_count);
        }
    }

    // The ceiling is a constant, so a shader compiled against it is the same
    // text for every world -- which is the whole reason the build script can
    // compile these ahead of time. Without it the length tracks the world and
    // the text does too.
    #[test]
    fn the_ceiling_makes_the_pool_length_independent_of_the_world() {
        let ceiling = concinnity_core::render::uniforms::BINDLESS_POOL_SIZE;
        for texture_count in [0usize, 1, 7, 64] {
            assert_eq!(bindless_pool_size(texture_count, true), ceiling);
            assert_eq!(
                bindless_pool_size(texture_count, false),
                world_pool_size(texture_count)
            );
        }
    }

    // Every slot the ceiling declares is written at init, and the world's own
    // images have to fit inside it for that fill to be a pad rather than a
    // truncation -- which is what `vulkan::init` checks before choosing it.
    #[test]
    fn a_world_that_fits_the_ceiling_leaves_room_to_pad() {
        let ceiling = concinnity_core::render::uniforms::BINDLESS_POOL_SIZE;
        assert!(world_pool_size(ceiling - 3) <= ceiling);
        assert!(world_pool_size(ceiling) > ceiling);
    }

    // Every pool-sized program declares its texture array at the length the
    // caller passed, so a pipeline rebuilt from the layout's stored pool size
    // indexes the same slots the descriptor set was written with.
    #[test]
    fn pool_programs_size_their_texture_array_from_the_context() {
        for pool_size in [2usize, 8, 65] {
            let ctx = Ctx {
                hot_reload: false,
                msaa: false,
                pool_size,
                probe_count: 4,
            };
            for p in ALL.iter().filter(|p| p.assembly.pool_size) {
                let src = p.source(&ctx);
                assert!(
                    src.contains(&format!("tex_pool[{pool_size}]")),
                    "{} did not size its texture pool at {pool_size}",
                    p.label
                );
            }
        }
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
            msaa: false,
            pool_size: 3,
            probe_count: 3,
        };
        let src = CULL_PHASE2.source(&ctx);
        let mut lines = src.lines();
        assert!(lines.next().unwrap().starts_with("#version"));
        assert_eq!(lines.next().unwrap(), "#define CULL_PHASE2 1");
    }

    #[test]
    fn substitutions_resolve_every_marker() {
        let ctx = Ctx {
            hot_reload: false,
            msaa: false,
            pool_size: 5,
            probe_count: 5,
        };
        for p in ALL {
            let src = p.source(&ctx);
            for marker in [
                "{PROBE_COMMON}",
                "{MAX_PROBES}",
                "{PROBE_DESC_SET}",
                "{POOL_SIZE}",
                "{OBJECT_DATA}",
            ] {
                assert!(!src.contains(marker), "{} left {}", p.label, marker);
            }
        }
    }

    // Every program that strides the per-frame object SSBO gets the shared
    // record spliced in, and no other program carries a stray declaration: the
    // whole point of the fragment is that `GpuObjectData` exists exactly once.
    #[test]
    fn object_data_programs_splice_the_shared_record() {
        let ctx = Ctx {
            hot_reload: false,
            msaa: false,
            pool_size: 4,
            probe_count: 4,
        };
        let mut spliced = 0usize;
        for p in ALL {
            let declares = p.source(&ctx).contains("struct GpuObjectData");
            assert_eq!(
                declares, p.assembly.object_data,
                "{}: declares GpuObjectData = {declares}, object_data = {}",
                p.label, p.assembly.object_data
            );
            spliced += usize::from(declares);
        }
        assert_eq!(spliced, 3, "object-data program count changed");
    }
}
