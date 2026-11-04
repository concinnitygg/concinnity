// Streamed texture and deferred mesh payload sources.

use hashbrown::HashMap;

use concinnity_engine::ecs::PayloadLocator;
use concinnity_host::store::blob::{BlobLoadError, blob_path, payload_section_start};

use concinnity_engine::render::streaming::pools::DeferredMeshPayload;

// Why a blob's payload section could not be located: the blob, and the load
// failure, whose message carries its cause.
#[derive(Debug, thiserror::Error)]
#[error("blob {blob_index}: {source}")]
pub(crate) struct PayloadSectionError {
    blob_index: u32,
    #[source]
    source: BlobLoadError,
}

// Blob `blob_index`'s file and where its payload section starts in it, so a
// locator's payload-section offset becomes an absolute file offset.
pub(crate) fn blob_payload_section(blob_index: u32) -> Result<(String, u64), PayloadSectionError> {
    let error = |source| PayloadSectionError { blob_index, source };
    let path = blob_path(blob_index).ok_or_else(|| error(BlobLoadError::NoStateRoot))?;
    let start = payload_section_start(&path).map_err(error)?;
    Ok((path, start))
}

// Build the payload source for a streamed texture pool (albedo or normal-map).
//
// When `disk_backed`, each locator's payload-section offset is turned into an
// absolute file offset (the blob file's payload section starts past its header
// and defs) so the streamer can re-read payloads from disk without a RAM copy.
// Otherwise the retained `payloads` are wrapped RAM-resident. Used by the
// Metal, Vulkan, and DirectX texture-streaming paths.
pub(crate) fn build_texture_payload_source(
    payloads: Vec<Vec<u8>>,
    locators: &[PayloadLocator],
    disk_backed: bool,
) -> Result<std::sync::Arc<dyn super::texture::PayloadSource>, PayloadSectionError> {
    if !disk_backed {
        return Ok(std::sync::Arc::new(super::texture::MemPayloadSource::new(
            payloads,
        )));
    }
    let mut sections: HashMap<u32, (String, u64)> = HashMap::new();
    let mut disk_locators = Vec::with_capacity(locators.len());
    for loc in locators {
        let (path, start) = if let Some(section) = sections.get(&loc.blob_index) {
            section.clone()
        } else {
            let section = blob_payload_section(loc.blob_index)?;
            sections.insert(loc.blob_index, section.clone());
            section
        };
        disk_locators.push(super::texture::DiskTextureLocator {
            path,
            file_offset: start + loc.offset,
            len: loc.len,
        });
    }
    Ok(std::sync::Arc::new(super::texture::DiskPayloadSource::new(
        disk_locators,
    )))
}

// A deferred mesh payload's absolute byte range in its blob file, or None
// (logged) when the blob has no layout or its header cannot be read.
pub(crate) fn disk_mesh_payload(locator: &PayloadLocator) -> Option<DeferredMeshPayload> {
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
    use concinnity_engine::bake::texture;

    // A log line names the blob and every cause, down to the read's own error.
    #[test]
    fn a_payload_section_error_names_the_blob_and_its_causes() {
        let error = PayloadSectionError {
            blob_index: 3,
            source: BlobLoadError::Io {
                path: "data/3".to_string(),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "gone"),
            },
        };
        assert_eq!(error.to_string(), "blob 3: reading data/3 failed: gone");

        let error = PayloadSectionError {
            blob_index: 1,
            source: BlobLoadError::Format {
                path: "data/1".to_string(),
                source: concinnity_engine::blob::BlobError::ValidityMismatch(7),
            },
        };
        let text = error.to_string();
        assert!(
            text.starts_with("blob 1: data/1 is not usable world data: "),
            "{text}"
        );
        assert_eq!(text.matches("different version").count(), 1, "{text}");
    }

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
