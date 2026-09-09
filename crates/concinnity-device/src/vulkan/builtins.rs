// src/vulkan/builtins.rs
//
// The Vulkan backend's shader-capacity policy and its export-time precompile.
//
// There is no GLSL left to declare: this was the table of hand-written GLSL
// programs, and the raymarch proxy vertex shaders were the last two entries on
// it. They compile from `raymarch.slang` now, alongside every other program
// this backend runs, so what remains here is the pool sizing the single-source
// shaders bake in and the loop that warms a bundle's cache with them.

// Slots a world with `texture_count` table entries needs: one image per slot (a
// single fallback when the table is empty) plus the reserved fallbacks,
// flat-normal and white. The pool's descriptor count, and the only length it
// has: the shaders declare the array unsized and read whatever the set layout
// was built with, so this never reaches the source text.
pub(crate) fn world_pool_size(texture_count: usize) -> usize {
    texture_count.max(1) + crate::gfx::render_types::FALLBACK_TEXTURE_COUNT
}

// Inputs a call site supplies to assemble a program's source.
pub(crate) struct Ctx {
    pub hot_reload: bool,
    pub msaa: bool,
    // Reflection-probe cube-array length for `{MAX_PROBES}` programs; ignored by
    // the rest. Callers pass the descriptor count the global set layout was
    // built with (`descriptor_layout::probe_cube_array_count`), so the GLSL array
    // and the layout binding always agree.
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

// Compile every declared program into `bundle`, reusing local cache artifacts
// where present.
//
// The probe cube-array length is a property of the device the bundle eventually
// runs on rather than of the world, so it is baked at the ceiling every desktop
// driver affords. A device that reports less headroom than that declares fewer
// and compiles these at first launch.
pub(crate) fn precompile(
    bundle: &mut concinnity_host::store::cache::Segment,
    report: &mut crate::precompile::Report,
) {
    // A program whose source reads the main pass's sample count gets both
    // variants: which one a device runs is a property of its MSAA mode, not of
    // the bundle.
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

use crate::vulkan::slang_builtins::SlangCompile;

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

    // The pool length reaches the descriptor layout and nothing else. A program
    // assembled for two different worlds is one text, which is what lets the
    // build script compile it ahead of any world.
    #[test]
    fn the_pool_length_never_reaches_the_source() {
        let probes = concinnity_core::render::uniforms::MAX_PROBES;
        let ctx = |probe_count| Ctx {
            hot_reload: false,
            msaa: false,
            probe_count,
        };
        for program in super::super::slang_builtins::ALL {
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
