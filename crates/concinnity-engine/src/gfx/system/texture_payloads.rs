// Texture pool decode: one image per texture slot, plus the raw payloads the
// streamer re-decodes after init.

use std::collections::{BTreeSet, HashSet};

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

// The slots that enter the pool as placeholders rather than their images.
pub(super) struct TextureSlotSkips<'a> {
    // Owned by a scene other than the start scene: the streamer decodes them
    // once their scene pins.
    pub(super) deferred: &'a HashSet<usize>,
    // Read only by the cook, so nothing samples them and their payloads are
    // never read.
    pub(super) cook_only: &'a BTreeSet<usize>,
}

// Decode every texture slot. A skipped slot enters the pool as a 1x1
// placeholder. `None` when a payload cannot be read or decoded.
pub(super) fn decode_texture_payloads(
    ctx: &mut PipelineContext,
    locators: &[PayloadLocator],
    skips: TextureSlotSkips<'_>,
    blob_disk_backed: bool,
) -> Option<TexturePayloads> {
    let mut images = Vec::with_capacity(locators.len());
    let mut payloads = Vec::new();
    for (slot, locator) in locators.iter().enumerate() {
        if skips.cook_only.contains(&slot) {
            images.push(TextureImage::rgba8(1, 1, vec![0, 0, 0, 255]));
            if !blob_disk_backed {
                payloads.push(Vec::new());
            }
            continue;
        }
        let deferred = skips.deferred.contains(&slot);
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
    if !skips.deferred.is_empty() {
        tracing::info!(
            "GraphicsSystem: deferred {} scene-owned texture payload(s) past init",
            skips.deferred.len()
        );
    }
    Some(TexturePayloads { images, payloads })
}
