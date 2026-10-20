//! Where each deferred streamed mesh's payload is read from.

use hashbrown::HashMap;

use crate::ecs::PayloadLocator;
use crate::gfx::render_types::DrawIndex;
use crate::render::draw_list::DeferredMeshSeed;
use crate::render::streaming::pools::DeferredMeshPayload;

/// Per-stream-id payload refs for the streamed draws built from a deferred
/// mesh. A RAM-backed seed carries its bytes; otherwise `resolve_disk` turns
/// the locator into a blob file range, and a stream it cannot resolve is left
/// out.
pub fn deferred_mesh_payloads(
    seeds: &HashMap<usize, DeferredMeshSeed>,
    draw_to_handle: &HashMap<DrawIndex, usize>,
    stream_draw_indices: &[DrawIndex],
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;

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
        let d = DrawIndex;
        let draw_to_handle = HashMap::from([(d(10), 6), (d(11), 4), (d(12), 5), (d(13), 4)]);
        let payloads = deferred_mesh_payloads(
            &seeds,
            &draw_to_handle,
            &[d(10), d(11), d(12), d(13)],
            fake_disk,
        );
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
        let draw_to_handle = HashMap::from([(DrawIndex(0), 4), (DrawIndex(1), 5)]);
        let payloads = deferred_mesh_payloads(
            &seeds,
            &draw_to_handle,
            &[DrawIndex(0), DrawIndex(1)],
            fake_disk,
        );
        assert_eq!(payloads.len(), 1);
        assert!(payloads.contains_key(&1));
    }
}
