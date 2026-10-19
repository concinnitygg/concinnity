// Streamed texture and deferred mesh payload sources.

use hashbrown::HashMap;

use concinnity_core::ecs::PayloadLocator;
use concinnity_host::store::blob::blob_path;
use concinnity_host::store::blob::payload_section_start;

use concinnity_core::render::stream_pools::DeferredMeshPayload;

// Blob `blob_index`'s file and where its payload section starts in it, so a
// locator's payload-section offset becomes an absolute file offset.
pub(super) fn blob_payload_section(blob_index: u32) -> Result<(String, u64), String> {
    let path = blob_path(blob_index)
        .ok_or_else(|| format!("blob {blob_index}: no blob layout installed"))?;
    let start = payload_section_start(&path).map_err(|e| format!("blob {blob_index}: {e}"))?;
    Ok((path, start))
}

// Build the payload source for a streamed texture pool (albedo or normal-map).
//
// When `disk_backed`, each locator's payload-section offset is turned into an
// absolute file offset (the blob file's payload section starts past its header
// and defs) so the streamer can re-read payloads from disk without a RAM copy.
// Otherwise the retained `payloads` are wrapped RAM-resident. Used by the
// Metal, Vulkan, and DirectX texture-streaming paths.
pub(super) fn build_texture_payload_source(
    payloads: Vec<Vec<u8>>,
    locators: &[PayloadLocator],
    disk_backed: bool,
) -> Result<std::sync::Arc<dyn crate::gfx::streaming::texture::PayloadSource>, String> {
    if !disk_backed {
        return Ok(std::sync::Arc::new(
            crate::gfx::streaming::texture::MemPayloadSource::new(payloads),
        ));
    }
    let mut sections: HashMap<u32, (String, u64)> = HashMap::new();
    let mut disk_locators = Vec::with_capacity(locators.len());
    for loc in locators {
        let (path, start) = match sections.get(&loc.blob_index) {
            Some(section) => section.clone(),
            None => {
                let section = blob_payload_section(loc.blob_index)?;
                sections.insert(loc.blob_index, section.clone());
                section
            }
        };
        disk_locators.push(crate::gfx::streaming::texture::DiskTextureLocator {
            path,
            file_offset: start + loc.offset,
            len: loc.len,
        });
    }
    Ok(std::sync::Arc::new(
        crate::gfx::streaming::texture::DiskPayloadSource::new(disk_locators),
    ))
}

// A deferred mesh payload's absolute byte range in its blob file, or None
// (logged) when the blob has no layout or its header cannot be read.
pub(super) fn disk_mesh_payload(locator: &PayloadLocator) -> Option<DeferredMeshPayload> {
    match blob_payload_section(locator.blob_index) {
        Ok((path, start)) => Some(DeferredMeshPayload::Disk {
            path,
            offset: start + locator.offset,
            len: locator.len,
        }),
        Err(e) => {
            tracing::warn!("GraphicsSystem: deferred mesh {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::bake::texture;

    // The RAM-resident source decodes a compiled texture payload on fetch.
    #[test]
    fn build_texture_payload_source_mem_backed_decodes_payload() {
        // 1x1 RGBA tagged payload via the shared serializer.
        let payload = texture::serialize(&texture::TextureImage::rgba8(
            1,
            1,
            vec![0x11, 0x22, 0x33, 0xFF],
        ));

        let src = build_texture_payload_source(vec![payload], &[], false).expect("mem source");
        let decoded = src.fetch(0).expect("decodes item 0");
        assert_eq!((decoded.width(), decoded.height()), (1, 1));
        assert_eq!(decoded.mips[0].data, vec![0x11, 0x22, 0x33, 0xFF]);
        // Out-of-range item id surfaces an error rather than panicking.
        assert!(src.fetch(1).is_err());
    }
}
