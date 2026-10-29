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
#[derive(Clone, Copy)]
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
    ctx: &mut PipelineContext<'_>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::World;
    use crate::test_support::OneSection;
    use alloc::boxed::Box;

    // A w x h mid-gray image in the compiled texture payload format.
    fn payload(w: u32, h: u32) -> Vec<u8> {
        texture::serialize(&TextureImage::rgba8(w, h, vec![0x7F; (w * h * 4) as usize]))
    }

    // A world whose blob holds a 2x2 then a 4x4 texture, and their locators.
    fn two_textures() -> (World, Vec<PayloadLocator>) {
        let mut section = OneSection::default();
        let locators = vec![
            section.payload(&payload(2, 2)),
            section.payload(&payload(4, 4)),
        ];
        (World::from_payloads(Box::new(section)), locators)
    }

    fn size(image: &TextureImage) -> (u32, u32) {
        (image.width(), image.height())
    }

    // A deferred slot enters the pool as a 1x1 placeholder. Disk-backed, it
    // keeps no payload bytes (the streamer re-reads the blob file); RAM-backed,
    // every slot's bytes are kept for the streamer to re-decode.
    #[test]
    fn a_deferred_slot_decodes_to_a_placeholder() {
        let (mut world, locators) = two_textures();
        let deferred = HashSet::from([1]);
        let cook_only = BTreeSet::new();
        let skips = || TextureSlotSkips {
            deferred: &deferred,
            cook_only: &cook_only,
        };

        let disk = decode_texture_payloads(&mut world.context(), &locators, skips(), true)
            .expect("decodes");
        assert_eq!(disk.images.len(), 2);
        assert_eq!(size(&disk.images[0]), (2, 2));
        assert_eq!(size(&disk.images[1]), (1, 1));
        assert!(disk.payloads.is_empty());

        let ram = decode_texture_payloads(&mut world.context(), &locators, skips(), false)
            .expect("decodes");
        assert_eq!(size(&ram.images[1]), (1, 1));
        assert_eq!(ram.payloads, vec![payload(2, 2), payload(4, 4)]);
    }

    // A texture only the cook reads enters the pool as a 1x1 placeholder and
    // its payload is never read: RAM-backed, its slot keeps an empty entry so
    // the streamer's slots stay dense.
    #[test]
    fn a_cook_only_slot_stays_a_placeholder() {
        let (mut world, locators) = two_textures();
        let cook_only = BTreeSet::from([0]);
        let skips = TextureSlotSkips {
            deferred: &HashSet::new(),
            cook_only: &cook_only,
        };
        let ram = decode_texture_payloads(&mut world.context(), &locators, skips, false)
            .expect("decodes");
        assert_eq!(size(&ram.images[0]), (1, 1));
        assert_eq!(size(&ram.images[1]), (4, 4));
        assert_eq!(ram.payloads, vec![Vec::new(), payload(4, 4)]);
    }
}
