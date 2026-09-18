// Streamed texture and deferred mesh payload sources, and the voxel block
// palette conversion.

use std::collections::HashMap;

use concinnity_core::components::BlockType;
use concinnity_core::ecs::PayloadLocator;
use concinnity_core::geometry::ChunkBlockType;
use concinnity_host::store::blob::blob_path;
use concinnity_host::store::blob::payload_section_start;

use crate::gfx::draw_list::DeferredMeshSeed;
use crate::gfx::streaming::mesh::DeferredMeshPayload;

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
                    .map_err(|e| format!("blob {}: {}", loc.blob_index, e))?;
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

// Per-stream-id payload refs for the streamed draws built from a deferred mesh.
// A RAM-backed seed carries its bytes; otherwise `resolve_disk` turns the locator
// into a blob file range, and a stream it cannot resolve is left out.
pub(super) fn deferred_mesh_payloads(
    seeds: &HashMap<usize, DeferredMeshSeed>,
    draw_to_handle: &HashMap<usize, usize>,
    stream_draw_indices: &[usize],
    resolve_disk: impl Fn(&PayloadLocator) -> Option<DeferredMeshPayload>,
) -> HashMap<usize, DeferredMeshPayload> {
    let mut payloads = HashMap::new();
    for (stream_id, draw_idx) in stream_draw_indices.iter().enumerate() {
        let Some(seed) = draw_to_handle.get(draw_idx).and_then(|h| seeds.get(h)) else {
            continue;
        };
        let payload = match &seed.bytes {
            Some(bytes) => DeferredMeshPayload::Bytes(bytes.clone()),
            None => match resolve_disk(&seed.locator) {
                Some(payload) => payload,
                None => continue,
            },
        };
        payloads.insert(stream_id, payload);
    }
    payloads
}

// A deferred mesh payload's absolute byte range in its blob file, or None
// (logged) when the blob has no layout or its header cannot be read.
pub(super) fn disk_mesh_payload(locator: &PayloadLocator) -> Option<DeferredMeshPayload> {
    let Some(path) = blob_path(locator.blob_index) else {
        tracing::warn!(
            "GraphicsSystem: deferred mesh blob {} has no layout to read from",
            locator.blob_index
        );
        return None;
    };
    match payload_section_start(&path) {
        Ok(start) => Some(DeferredMeshPayload::Disk {
            path,
            offset: start + locator.offset,
            len: locator.len,
        }),
        Err(e) => {
            tracing::warn!(
                "GraphicsSystem: deferred mesh blob {} unreadable: {}",
                locator.blob_index,
                e
            );
            None
        }
    }
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

    fn seed(blob_index: u32, bytes: Option<Vec<u8>>) -> DeferredMeshSeed {
        DeferredMeshSeed {
            locator: PayloadLocator {
                blob_index,
                offset: 16,
                len: 32,
            },
            bytes,
        }
    }

    // Resolves blob 1 to a fixed file range and fails every other blob.
    fn fake_disk(locator: &PayloadLocator) -> Option<DeferredMeshPayload> {
        (locator.blob_index == 1).then(|| DeferredMeshPayload::Disk {
            path: "blob1".to_string(),
            offset: 100 + locator.offset,
            len: locator.len,
        })
    }

    #[test]
    fn deferred_mesh_payloads_key_each_deferred_draw_by_its_stream_id() {
        // Handle 4 is disk-backed (two draws), handle 5 is RAM-backed, handle 6
        // is not deferred.
        let seeds = HashMap::from([(4, seed(1, None)), (5, seed(1, Some(vec![7, 8])))]);
        let draw_to_handle = HashMap::from([(10, 6), (11, 4), (12, 5), (13, 4)]);
        let payloads =
            deferred_mesh_payloads(&seeds, &draw_to_handle, &[10, 11, 12, 13], fake_disk);
        assert_eq!(payloads.len(), 3);
        assert!(
            !payloads.contains_key(&0),
            "a draw of a non-deferred mesh has no payload"
        );
        for stream_id in [1, 3] {
            assert!(matches!(
                &payloads[&stream_id],
                DeferredMeshPayload::Disk { path, offset: 116, len: 32 } if path == "blob1"
            ));
        }
        assert!(matches!(&payloads[&2], DeferredMeshPayload::Bytes(b) if b == &[7, 8]));
    }

    #[test]
    fn deferred_mesh_payloads_skip_an_unresolvable_blob() {
        let seeds = HashMap::from([(4, seed(2, None)), (5, seed(1, None))]);
        let draw_to_handle = HashMap::from([(0, 4), (1, 5)]);
        let payloads = deferred_mesh_payloads(&seeds, &draw_to_handle, &[0, 1], fake_disk);
        assert_eq!(payloads.len(), 1);
        assert!(payloads.contains_key(&1));
    }
}
