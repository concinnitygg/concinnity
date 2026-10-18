//! Texture pool decode: one image per texture slot, plus the raw payloads the
//! streamer re-decodes after init.

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use hashbrown::HashSet;

use crate::bake::texture::{self, TextureImage};
use crate::ecs::{PayloadLocator, PipelineContext};

/// The decoded texture pool, one image per slot.
pub struct TexturePayloads {
    /// The decoded image per slot.
    pub images: Vec<TextureImage>,
    /// Raw compiled payloads, kept past blob release so the streamer can
    /// re-decode them off the main thread. Empty when the blobs are disk-backed:
    /// the streamer then re-reads each payload from its blob file.
    pub payloads: Vec<Vec<u8>>,
}

/// The slots that enter the pool as placeholders rather than their images.
pub struct TextureSlotSkips<'a> {
    /// Owned by a scene other than the start scene: the streamer decodes them
    /// once their scene pins.
    pub deferred: &'a HashSet<usize>,
    /// Read only by the cook, so nothing samples them and their payloads are
    /// never read.
    pub cook_only: &'a BTreeSet<usize>,
}

/// Decode every texture slot. A skipped slot enters the pool as a 1x1
/// placeholder. `None` when a payload cannot be read or decoded.
pub fn decode_texture_payloads(
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
                    tracing::error!("failed to read texture payload: {}", e);
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
                    tracing::error!("malformed texture payload: {}", e);
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
            "deferred {} scene-owned texture payload(s) past init",
            skips.deferred.len()
        );
    }
    Some(TexturePayloads { images, payloads })
}
