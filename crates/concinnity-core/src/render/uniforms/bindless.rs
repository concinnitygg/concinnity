//! The bindless texture pool's capacity: how many slots each backend's host
//! writes. Every shader declares the pool unsized, so the ceiling never reaches
//! shader text and one compiled program serves any world. A Vulkan device that
//! cannot seat the ceiling sizes its descriptor binding to the world, which the
//! shader never sees.

/// Slots in the bindless texture pool the hosts write.
///
/// One budget across the backends, so a world that fits on one fits on the
/// others. A world with more textures than this has its pool indices clamped
/// into range, so an over-cap index samples a valid texture rather than reading
/// past what the host wrote.
pub const BINDLESS_POOL_SIZE: usize = 1024;

// The ceiling has to leave room for the reserved fallbacks (flat-normal and
// white) every world appends past its own textures.
const _: () = assert!(BINDLESS_POOL_SIZE > crate::gfx::render_types::FALLBACK_TEXTURE_COUNT);
