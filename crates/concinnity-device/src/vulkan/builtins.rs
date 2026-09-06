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

// Compile every declared program into `bundle`, reusing local cache artifacts
// where present.
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
}
