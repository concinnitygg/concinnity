//! The images a live backend samples: the handle-indexed texture pool, the
//! color-grading LUT and the IBL environment map.
//!
//! Albedo and normal maps share one handle-indexed texture pool, so every
//! streamed texture flows through the same slot pair whatever its role. The
//! image carries its GPU format and mip chain: RGBA8 regenerates mips on
//! upload, block-compressed formats upload their chain verbatim.

use crate::gfx::render_types::TextureSlot;
use crate::render::backend::error::{RenderError, RenderResult};

/// Texture-pool residency and the whole-image replacements a hot reload makes.
///
/// The pool slots are required: a backend that cannot fill a slot cannot draw
/// a streamed world at all. The LUT and environment-map replacements default
/// to [`RenderError::Unsupported`], so a reload is never counted as applied
/// when nothing changed.
pub trait TextureUploads {
    /// Release a texture slot's image, leaving a 1x1 placeholder in the slot so
    /// a draw still holding the handle has something to sample.
    ///
    /// # Errors
    ///
    /// Returns an error if the slot is out of range.
    fn evict_texture_slot(&mut self, slot: TextureSlot) -> RenderResult<()>;
    /// Replace a texture slot's image after a streaming upload.
    ///
    /// # Errors
    ///
    /// Returns an error if the slot is out of range or the image cannot be
    /// uploaded.
    fn update_texture_slot(
        &mut self,
        slot: TextureSlot,
        image: &crate::bake::texture::TextureImage,
    ) -> RenderResult<()>;

    /// Replace the live color-grading LUT with a fresh `size³` RGBA8 payload.
    /// Driven by asset hot-reload (`cn run -f` only). A request, not a feed:
    /// the reload reports whether the replacement landed.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] on a backend without LUT reload, or
    /// an error if the payload does not fit a LUT or cannot be uploaded.
    fn update_color_lut(&mut self, size: u32, data: &[u8]) -> RenderResult<()> {
        let _ = (size, data);
        Err(RenderError::Unsupported {
            op: "update_color_lut",
        })
    }

    /// Replace the live IBL environment map with a freshly precomputed payload.
    /// `payload` is the serialized byte format emitted by
    /// `crate::bake::environment_map::compile_environment_map_payload`
    /// (header + irradiance cube + prefilter mip chain), so init and hot-reload
    /// share a single byte format. Driven by asset hot-reload (`cn run -f`
    /// only). A request, not a feed: the reload reports whether the
    /// replacement landed.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] on a backend without
    /// environment-map reload, or an error if the payload does not decode or
    /// cannot be uploaded.
    fn update_environment_map(&mut self, payload: &[u8]) -> RenderResult<()> {
        let _ = payload;
        Err(RenderError::Unsupported {
            op: "update_environment_map",
        })
    }
}
