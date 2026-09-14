// Streamed texture payload sources and the voxel block palette conversion.

use concinnity_core::components::BlockType;
use concinnity_core::ecs::PayloadLocator;
use concinnity_core::geometry::ChunkBlockType;
use concinnity_host::store::blob::blob_path;
use concinnity_host::store::blob::payload_section_start;

// Resolve a `BlockType` asset into the chunk mesher's palette entry. Per-face
// UV overrides fall back to the uv_min/uv_max rectangle, mirroring the
// build-time `geometry::resolve_block_type`. Used by every backend's
// chunk-streaming setup.
pub(super) fn block_type_to_chunk(bt: &BlockType) -> ChunkBlockType {
    let default_rect = [bt.uv_min[0], bt.uv_min[1], bt.uv_max[0], bt.uv_max[1]];
    ChunkBlockType {
        solid: bt.solid,
        uv_top: bt.uv_top.unwrap_or(default_rect),
        uv_bottom: bt.uv_bottom.unwrap_or(default_rect),
        uv_side: bt.uv_side.unwrap_or(default_rect),
    }
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
    let mut section_starts: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    let mut disk_locators = Vec::with_capacity(locators.len());
    for loc in locators {
        let path = blob_path(loc.blob_index)
            .ok_or_else(|| format!("blob {}: no blob layout installed", loc.blob_index))?;
        let start = match section_starts.get(&loc.blob_index) {
            Some(&s) => s,
            None => {
                let s = payload_section_start(&path)
                    .map_err(|e| format!("blob {}: {:?}", loc.blob_index, e))?;
                section_starts.insert(loc.blob_index, s);
                s
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

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::bake::texture;

    // A per-face override wins where set; the remaining faces fall back to the
    // uv_min/uv_max rectangle. Mirrors build-time resolve_block_type.
    #[test]
    fn block_type_to_chunk_overrides_then_falls_back() {
        let bt = BlockType {
            solid: true,
            uv_min: [0.1, 0.2],
            uv_max: [0.6, 0.7],
            uv_top: Some([0.0, 0.0, 0.25, 0.25]),
            uv_bottom: None,
            uv_side: None,
            ..Default::default()
        };
        let chunk = block_type_to_chunk(&bt);
        assert!(chunk.solid);
        assert_eq!(chunk.uv_top, [0.0, 0.0, 0.25, 0.25], "override kept");
        let fallback = [0.1, 0.2, 0.6, 0.7];
        assert_eq!(
            chunk.uv_bottom, fallback,
            "unset face -> uv_min/uv_max rect"
        );
        assert_eq!(chunk.uv_side, fallback);
    }

    #[test]
    fn block_type_to_chunk_air_is_not_solid() {
        let bt = BlockType {
            solid: false,
            ..Default::default()
        };
        assert!(!block_type_to_chunk(&bt).solid);
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
        assert_eq!((decoded.image.width(), decoded.image.height()), (1, 1));
        assert_eq!(decoded.image.mips[0].data, vec![0x11, 0x22, 0x33, 0xFF]);
        // Out-of-range item id surfaces an error rather than panicking.
        assert!(src.fetch(1).is_err());
    }
}
