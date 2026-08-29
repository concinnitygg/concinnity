//! The bindless texture pool's capacity, shared by every backend that declares
//! the pool as a fixed-size array.
//!
//! Metal has always sized its pool this way (`BINDLESS_TEXTURE_COUNT`), and
//! DirectX declares an unbounded array so it needs no ceiling at all. Vulkan
//! used to size the pool to each world's texture table, which made the shader's
//! `POOL_SIZE` a property of the world rather than of the build -- and a shader
//! whose text depends on the world cannot be compiled ahead of that world. This
//! constant is what lets the Vulkan programs be compiled at build time like
//! every other one.
//!
//! A device that cannot seat the ceiling falls back to sizing the pool to the
//! world, exactly as before. That shader text then differs from the one the
//! build script compiled, so the precompiled artifact simply does not match and
//! the renderer compiles -- which is the same content check that governs every
//! other artifact, with no special case for it.

/// Slots in the bindless texture pool. The shader's `POOL_SIZE` define is
/// injected from this value, so the array and the descriptor binding cannot
/// drift.
///
/// Matches Metal's `BINDLESS_TEXTURE_COUNT`. A world with more textures than
/// this has its pool indices clamped into range, so an over-cap index samples a
/// valid texture rather than reading past the array.
pub const BINDLESS_POOL_SIZE: usize = 1024;

// The ceiling has to leave room for the reserved fallbacks (flat-normal and
// white) every world appends past its own textures.
const _: () = assert!(BINDLESS_POOL_SIZE > crate::gfx::render_types::FALLBACK_TEXTURE_COUNT);

#[cfg(test)]
mod tests {
    use super::*;

    // Matching Metal's pool keeps one texture budget across the backends, so a
    // world that fits on one fits on the others.
    #[test]
    fn the_ceiling_matches_the_metal_pool() {
        assert_eq!(BINDLESS_POOL_SIZE, 1024);
    }
}
