//! Where streamed texture payloads come from, and the thread that decodes them
//! for core's texture pool.
//!
//! `PayloadSource` is the seam. `MemPayloadSource` serves compiled texture
//! payloads already resident in RAM (used by `cn debug`, which builds payloads
//! in memory); `DiskPayloadSource` re-reads them from their blob files on disk
//! (used by `cn run`, so the bytes never stay RAM-resident).

use concinnity_engine::bake::texture::{self, TextureImage};
use concinnity_engine::render::streaming::pools::ItemLoad;
use std::sync::Arc;

use concinnity_host::thread::worker::Worker;

// Fetches and decodes a streamable texture payload by item id. `Send + Sync`
// so the load thread can own one; called off the main thread.
pub(crate) trait PayloadSource: Send + Sync {
    // Decode item `id` into GPU-ready pixels, or a human-readable error.
    fn fetch(&self, id: usize) -> Result<TextureImage, String>;
}

// Compiled texture payloads kept resident in RAM. `cn debug` builds payloads in
// memory with no blob files on disk, so this streams the GPU upload only.
pub(crate) struct MemPayloadSource {
    payloads: Vec<Vec<u8>>,
}

impl MemPayloadSource {
    // `payloads[id]` is the compiled texture payload for streamable item `id`.
    pub(crate) fn new(payloads: Vec<Vec<u8>>) -> Self {
        Self { payloads }
    }
}

impl PayloadSource for MemPayloadSource {
    fn fetch(&self, id: usize) -> Result<TextureImage, String> {
        let bytes = self
            .payloads
            .get(id)
            .ok_or_else(|| format!("no payload for streamed texture {id}"))?;
        texture::deserialize(bytes)
    }
}

// Locates one streamed texture payload inside a blob file on disk.
// `file_offset` is absolute into the file: the payload-section start is
// already folded in, so the load thread only seeks and reads.
#[derive(Clone)]
pub(crate) struct DiskTextureLocator {
    pub path: String,
    pub(crate) file_offset: u64,
    pub len: u64,
}

// Re-reads each compiled texture payload from its blob file on disk, so the
// bytes never stay RAM-resident (`cn run`).
pub(crate) struct DiskPayloadSource {
    // locators[id] points streamed item `id` at its bytes in a blob file.
    locators: Vec<DiskTextureLocator>,
}

impl DiskPayloadSource {
    // `locators[id]` locates the compiled payload for streamable item `id`.
    pub(crate) fn new(locators: Vec<DiskTextureLocator>) -> Self {
        Self { locators }
    }
}

impl PayloadSource for DiskPayloadSource {
    fn fetch(&self, id: usize) -> Result<TextureImage, String> {
        let loc = self
            .locators
            .get(id)
            .ok_or_else(|| format!("no disk locator for streamed texture {id}"))?;
        let bytes = super::file_range::read_at(&loc.path, loc.file_offset, loc.len)?;
        texture::deserialize(&bytes)
    }
}

// The thread that answers the texture pool's loads from `source`.
pub(crate) fn spawn_worker(
    source: Arc<dyn PayloadSource>,
) -> std::io::Result<Worker<usize, ItemLoad<TextureImage>>> {
    Worker::spawn("cn-texture-stream", 1, move |id| (id, source.fetch(id)))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a minimal compiled RGBA8 texture payload via the shared serializer.
    fn make_payload(w: u32, h: u32, fill: u8) -> Vec<u8> {
        let pixels = std::iter::repeat_n(fill, (w * h * 4) as usize).collect();
        texture::serialize(&TextureImage::rgba8(w, h, pixels))
    }

    #[test]
    fn mem_payload_source_decodes_a_payload() {
        let source = MemPayloadSource::new(vec![make_payload(2, 1, 0xAB)]);
        let image = source.fetch(0).expect("fetch ok");
        assert_eq!((image.width(), image.height()), (2, 1));
        assert_eq!(image.mips[0].data.len(), 2 * 4);
        assert!(image.mips[0].data.iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn mem_payload_source_errors_on_unknown_id() {
        let source = MemPayloadSource::new(vec![make_payload(1, 1, 0)]);
        assert!(source.fetch(9).is_err());
    }

    #[test]
    fn disk_payload_source_reads_a_payload_at_offset() {
        let tree = concinnity_testing::TempTree::new();
        let payload = make_payload(2, 1, 0xCD);
        // arbitrary leading bytes standing in for a blob header + defs section
        let prefix = vec![0u8; 37];
        let mut bytes = prefix.clone();
        bytes.extend_from_slice(&payload);
        let path = tree.write("payload.bin", &bytes);
        let source = DiskPayloadSource::new(vec![DiskTextureLocator {
            path: path.to_string_lossy().into_owned(),
            file_offset: prefix.len() as u64,
            len: payload.len() as u64,
        }]);
        let image = source.fetch(0).expect("fetch ok");
        assert_eq!((image.width(), image.height()), (2, 1));
        assert!(image.mips[0].data.iter().all(|&b| b == 0xCD));
    }

    #[test]
    fn disk_payload_source_errors_on_unknown_id() {
        let source = DiskPayloadSource::new(vec![]);
        assert!(source.fetch(0).is_err());
    }

    #[test]
    fn disk_payload_source_errors_on_missing_file() {
        let source = DiskPayloadSource::new(vec![DiskTextureLocator {
            path: "/nonexistent/cn_disk_payload_missing.bin".to_string(),
            file_offset: 0,
            len: 4,
        }]);
        assert!(source.fetch(0).is_err());
    }
}
