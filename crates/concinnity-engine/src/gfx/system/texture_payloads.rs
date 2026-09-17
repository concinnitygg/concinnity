// Texture pool decode: one image per texture slot, plus the raw payloads the
// streamer re-decodes after init.

use std::collections::HashSet;

use concinnity_core::bake::texture::{self, TextureImage};
use concinnity_core::ecs::{PayloadLocator, PipelineContext};

// The decoded texture pool, one image per slot.
pub(super) struct TexturePayloads {
    pub(super) images: Vec<TextureImage>,
    // Raw compiled payloads, kept past blob release so the streamer can
    // re-decode them off the main thread. Empty when the blobs are disk-backed:
    // the streamer then re-reads each payload from its blob file.
    pub(super) payloads: Vec<Vec<u8>>,
}

// Decode every texture slot. A deferred slot (owned by a scene other than the
// start scene) enters the pool as a 1x1 placeholder, and the streamer decodes
// it once its scene pins. `None` when a payload cannot be read or decoded.
pub(super) fn decode_texture_payloads(
    ctx: &mut PipelineContext,
    locators: &[PayloadLocator],
    deferred_slots: &HashSet<usize>,
    blob_disk_backed: bool,
) -> Option<TexturePayloads> {
    let mut images = Vec::with_capacity(locators.len());
    let mut payloads = Vec::new();
    for (slot, locator) in locators.iter().enumerate() {
        let deferred = deferred_slots.contains(&slot);
        let needs_bytes = !deferred || !blob_disk_backed;
        let bytes = if needs_bytes {
            match ctx.read_payload(locator) {
                Ok(b) => Some(b.to_vec()),
                Err(e) => {
                    tracing::error!("GraphicsSystem: failed to read texture payload: {}", e);
                    return None;
                }
            }
        } else {
            None
        };
        if deferred {
            images.push(TextureImage::rgba8(1, 1, vec![0, 0, 0, 255]));
        } else {
            match texture::deserialize(bytes.as_deref().unwrap_or_default()) {
                Ok(t) => images.push(t),
                Err(e) => {
                    tracing::error!("GraphicsSystem: malformed texture payload: {}", e);
                    return None;
                }
            }
        }
        if !blob_disk_backed && let Some(bytes) = bytes {
            payloads.push(bytes);
        }
    }
    if !deferred_slots.is_empty() {
        tracing::info!(
            "GraphicsSystem: deferred {} scene-owned texture payload(s) past init",
            deferred_slots.len()
        );
    }
    Some(TexturePayloads { images, payloads })
}
