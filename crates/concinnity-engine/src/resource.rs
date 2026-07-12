// src/resource.rs
//
// Runtime resource tables: per-kind, handle-indexed views of the blob's resource
// stream. A resource (an audio clip today; meshes/textures/materials on the
// Windows follow-up) is compiled by cook and addressed at runtime by its dense
// per-kind handle. The owning system reads the table by that handle instead of
// querying an ECS column or scanning names, so a resource lives in a table it
// owns rather than as a component.

use std::collections::HashSet;

use crate::ecs::PayloadLocator;
use concinnity_core::ecs::{ResourceKind, ResourceRecord};

// One loaded resource's runtime form. A payload resource (audio clip, and later
// meshes / textures) carries a `PayloadLocator` into the blob payload section; a
// data resource (a baked Material) carries its runtime bytes in `data_bytes`.
// AudioClip uses the payload branch; `data_bytes` is present so the shape already
// exists for the data-resource kinds that follow.
#[derive(Debug, Clone, Default)]
pub struct ResourceEntry {
    pub payload: Option<PayloadLocator>,
    #[allow(dead_code)]
    pub data_bytes: Vec<u8>,
}

// Build a dense per-kind table from the blob's resource stream, indexed by
// handle: `table[handle]` is that resource's entry. Records of other kinds are
// ignored; a missing handle yields a default entry so indexing by any handle in
// range never panics.
pub fn resource_table(records: &[ResourceRecord], kind: ResourceKind) -> Vec<ResourceEntry> {
    let tag = kind as u8;
    let Some(max_handle) = records
        .iter()
        .filter(|r| r.resource_kind == tag)
        .map(|r| r.handle)
        .max()
    else {
        return Vec::new();
    };
    let mut table = vec![ResourceEntry::default(); max_handle as usize + 1];
    for record in records.iter().filter(|r| r.resource_kind == tag) {
        table[record.handle as usize] = ResourceEntry {
            payload: record.payload.clone(),
            data_bytes: record.data_bytes.clone(),
        };
    }
    table
}

// The audio clips loaded from the blob, indexed by `AudioClipHandle`. A plain ECS
// resource `AudioSystem` reads at init; this table is what AudioClip became when
// it left the component registry -- the replacement for the `query::<AudioClip>()`
// column plus its per-clip `PayloadLocator`.
#[derive(Debug, Clone, Default)]
pub struct AudioClipTable(pub Vec<ResourceEntry>);

impl AudioClipTable {
    // Build the table from the blob's resource stream.
    pub fn from_records(records: &[ResourceRecord]) -> Self {
        Self(resource_table(records, ResourceKind::AudioClip))
    }

    // The payload locator for a clip handle, if the handle is in range and the
    // clip has a compiled payload.
    pub fn locator(&self, handle: usize) -> Option<PayloadLocator> {
        self.0.get(handle).and_then(|e| e.payload.clone())
    }

    // Iterate every clip's locator in handle order (index == the clip's
    // `AudioClipHandle`), skipping clips with no payload.
    pub fn locators(&self) -> impl Iterator<Item = (usize, PayloadLocator)> + '_ {
        self.0
            .iter()
            .enumerate()
            .filter_map(|(i, e)| e.payload.clone().map(|l| (i, l)))
    }

    // Blob indices that hold an audio-clip payload. The graphics systems consult
    // this so they leave the audio blobs resident for `AudioSystem`, which inits
    // after them. Replaces the old core `audio_clip_blob_indices` component scan.
    pub fn blob_indices(&self) -> HashSet<u32> {
        self.0
            .iter()
            .filter_map(|e| e.payload.as_ref().map(|l| l.blob_index))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(kind: ResourceKind, handle: u32, blob_index: u32) -> ResourceRecord {
        ResourceRecord {
            resource_kind: kind as u8,
            handle,
            payload: Some(PayloadLocator {
                blob_index,
                offset: 0,
                len: 1,
            }),
            data_bytes: Vec::new(),
        }
    }

    #[test]
    fn table_is_dense_by_handle_and_ignores_other_kinds() {
        // Handles out of order, interleaved with another kind; the table places
        // each clip at its handle index and drops the mesh record.
        let records = vec![
            rec(ResourceKind::AudioClip, 1, 0),
            rec(ResourceKind::Mesh, 0, 5),
            rec(ResourceKind::AudioClip, 0, 0),
        ];
        let table = AudioClipTable::from_records(&records);
        assert_eq!(table.0.len(), 2);
        assert!(table.locator(0).is_some());
        assert!(table.locator(1).is_some());
        // A handle past the end resolves to None rather than panicking.
        assert!(table.locator(2).is_none());
    }

    #[test]
    fn empty_stream_yields_an_empty_table() {
        let table = AudioClipTable::from_records(&[]);
        assert!(table.0.is_empty());
        assert!(table.blob_indices().is_empty());
        assert_eq!(table.locators().count(), 0);
    }

    #[test]
    fn blob_indices_collects_every_payload_blob() {
        let records = vec![
            rec(ResourceKind::AudioClip, 0, 0),
            rec(ResourceKind::AudioClip, 1, 3),
        ];
        let table = AudioClipTable::from_records(&records);
        let mut indices: Vec<u32> = table.blob_indices().into_iter().collect();
        indices.sort_unstable();
        assert_eq!(indices, vec![0, 3]);
    }
}
