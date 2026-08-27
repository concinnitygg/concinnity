//! The compiled font payload: a power-of-two RGBA atlas of signed-distance-field
//! glyphs plus their metrics, which GraphicsSystem uploads and draws from.
//!
//! [`compile`] rasterises the printable ASCII glyphs of a TTF face with fontdue
//! and encodes them; [`deserialise`] reads the encoding back. Both halves live
//! here so the asset build, the engine's own build script, and the running
//! engine share one definition of the format: the engine embeds a baked atlas
//! for its startup error screen, which must render without any compiled world
//! data.
//!
//! Each atlas texel stores a normalised SDF value in [0, 1] where 0.5 = the glyph
//! outline. Values > 0.5 are inside; values < 0.5 are outside. The fragment shader
//! uses smoothstep + fwidth to reconstruct crisp, scale-independent alpha.

mod compile;
mod decode;
mod payload;
mod sdf;

pub use compile::{BUILTIN_FONT_BYTES, BUILTIN_FONT_FILE, compile};
pub use decode::deserialise;
