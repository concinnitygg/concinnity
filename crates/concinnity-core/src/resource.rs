// src/resource.rs
//
// Runtime resource tables: per-kind, handle-indexed views of a compiled blob's
// resource stream. A resource (an audio clip, a texture, a mesh, a material, ...)
// is compiled by cook and addressed at runtime by its dense per-kind handle. The
// owning system reads the table by that handle instead of querying an ECS column
// or scanning names, so a resource lives in a table it owns rather than as a
// component. Renderer-free (the tables are plain handle-indexed data), so they
// live here where the physics / audio subsystem crates can reach them; the
// client re-exports them under `crate::resource::*`, alongside the engine-side
// `install_resource_tables` that inserts them as World resources.

use std::collections::HashSet;

use crate::ecs::{PayloadLocator, ResourceKind, ResourceRecord};

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

// The textures loaded from the blob, indexed by `TextureHandle`. The renderer
// reads this at init to build its shared texture pool, in place of draining a
// `Texture` component column and scanning names to slots. Every texture (file or
// procedural) has a compiled payload, so an entry's `payload` is normally `Some`.
#[derive(Debug, Clone, Default)]
pub struct TextureTable(pub Vec<ResourceEntry>);

impl TextureTable {
    // Build the table from the blob's resource stream.
    pub fn from_records(records: &[ResourceRecord]) -> Self {
        Self(resource_table(records, ResourceKind::Texture))
    }

    // Number of textures (the shared pool size); a `TextureHandle` is in range
    // when its index is below this.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    // The payload locator for a texture handle, if in range and compiled.
    pub fn locator(&self, handle: usize) -> Option<PayloadLocator> {
        self.0.get(handle).and_then(|e| e.payload.clone())
    }

    // Blob indices that hold a texture payload, so the graphics systems can keep
    // the texture blobs resident. Mirrors `AudioClipTable::blob_indices`.
    pub fn blob_indices(&self) -> HashSet<u32> {
        self.0
            .iter()
            .filter_map(|e| e.payload.as_ref().map(|l| l.blob_index))
            .collect()
    }
}

// The color-grading LUTs loaded from the blob, indexed by `ColorLutHandle`. The
// renderer uses only the first (handle 0); a world declares at most one. Replaces
// the `query::<ColorLut>()` drain -- the LUT payload is read once at init.
#[derive(Debug, Clone, Default)]
pub struct ColorLutTable(pub Vec<ResourceEntry>);

impl ColorLutTable {
    pub fn from_records(records: &[ResourceRecord]) -> Self {
        Self(resource_table(records, ResourceKind::ColorLut))
    }

    // Number of declared LUTs; the renderer warns and uses only handle 0 when >1.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    // The payload locator for a LUT handle, if in range and compiled.
    pub fn locator(&self, handle: usize) -> Option<PayloadLocator> {
        self.0.get(handle).and_then(|e| e.payload.clone())
    }
}

// The IBL environment maps loaded from the blob, indexed by `EnvironmentMapHandle`.
// The renderer uses only the first (handle 0); a world declares at most one.
// Replaces the `query::<EnvironmentMap>()` drain.
#[derive(Debug, Clone, Default)]
pub struct EnvironmentMapTable(pub Vec<ResourceEntry>);

impl EnvironmentMapTable {
    pub fn from_records(records: &[ResourceRecord]) -> Self {
        Self(resource_table(records, ResourceKind::EnvironmentMap))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn locator(&self, handle: usize) -> Option<PayloadLocator> {
        self.0.get(handle).and_then(|e| e.payload.clone())
    }
}

// The fonts loaded from the blob, indexed by `FontHandle`. The renderer reads this
// at init to build its glyph atlases + metrics, in place of draining a `Font`
// component column. Every font (built-in or file-backed) has a compiled SDF-atlas
// payload, so an entry's `payload` is normally `Some`.
#[derive(Debug, Clone, Default)]
pub struct FontTable(pub Vec<ResourceEntry>);

impl FontTable {
    pub fn from_records(records: &[ResourceRecord]) -> Self {
        Self(resource_table(records, ResourceKind::Font))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    // Iterate each font's payload locator in handle order (index == the font's
    // `FontHandle`), skipping any font with no payload.
    pub fn locators(&self) -> impl Iterator<Item = (usize, PayloadLocator)> + '_ {
        self.0
            .iter()
            .enumerate()
            .filter_map(|(i, e)| e.payload.clone().map(|l| (i, l)))
    }

    // Blob indices that hold a font payload, so the graphics systems keep the font
    // blobs resident. Mirrors `TextureTable::blob_indices`.
    pub fn blob_indices(&self) -> HashSet<u32> {
        self.0
            .iter()
            .filter_map(|e| e.payload.as_ref().map(|l| l.blob_index))
            .collect()
    }
}

// The static meshes loaded from the blob's resource stream, indexed by
// `MeshHandle`. Mesh shares its handle space with the still-component geometry
// producers (ProceduralMesh, VoxelChunk, mesh-kind File): the Mesh block leads
// that space, so this table covers handles `0..len` and the runtime appends the
// component-produced geometry after it in the same block order cook assigned.
#[derive(Debug, Clone, Default)]
pub struct MeshTable(pub Vec<ResourceEntry>);

impl MeshTable {
    pub fn from_records(records: &[ResourceRecord]) -> Self {
        Self(resource_table(records, ResourceKind::Mesh))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    // Iterate each mesh's payload locator in handle order (index == the mesh's
    // `MeshHandle`), skipping any mesh with no payload.
    pub fn locators(&self) -> impl Iterator<Item = (usize, PayloadLocator)> + '_ {
        self.0
            .iter()
            .enumerate()
            .filter_map(|(i, e)| e.payload.clone().map(|l| (i, l)))
    }
}

// The skinned meshes loaded from the blob's resource stream, indexed by
// `SkinnedMeshHandle`. A hybrid entry: `payload` locates the compiled geometry
// (vertices + indices + skeleton) while `data_bytes` carries the baked runtime
// fields (placement, material/texture handles, capsule, spawn reserve) as a
// `(name_id, SkinnedMesh)` postcard tuple -- `asset_id` is serde-skipped on the
// schema struct, so the interned name travels beside it for the runtime's
// spawn-by-name registration.
#[derive(Debug, Clone, Default)]
pub struct SkinnedMeshTable(pub Vec<ResourceEntry>);

impl SkinnedMeshTable {
    pub fn from_records(records: &[ResourceRecord]) -> Self {
        Self(resource_table(records, ResourceKind::SkinnedMesh))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    // Whether any skinned mesh declares a character capsule; gates whether the
    // world needs a PhysicsSystem.
    pub fn has_capsule(&self) -> bool {
        self.0.iter().any(|e| {
            postcard::from_bytes::<(u32, crate::assets::SkinnedMesh)>(&e.data_bytes)
                .is_ok_and(|(_, sm)| sm.capsule.is_some())
        })
    }
}

// The materials loaded from the blob's resource stream, indexed by
// `MaterialHandle`. Unlike the payload-backed tables above, a Material is a DATA
// resource: cook bakes its validated args into the record's `data_bytes` (no blob
// payload), so `data_bytes(handle)` returns the serialized `Material` the renderer
// deserializes to build its material map.
#[derive(Debug, Clone, Default)]
pub struct MaterialTable(pub Vec<ResourceEntry>);

impl MaterialTable {
    pub fn from_records(records: &[ResourceRecord]) -> Self {
        Self(resource_table(records, ResourceKind::Material))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    // The baked material bytes for a handle, if the handle is in range.
    pub fn data_bytes(&self, handle: usize) -> Option<&[u8]> {
        self.0.get(handle).map(|e| e.data_bytes.as_slice())
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

    #[test]
    fn texture_table_is_dense_by_handle_and_ignores_other_kinds() {
        // A texture record and an audio record interleaved; the texture table
        // keeps only the textures, placed at their handle index.
        let records = vec![
            rec(ResourceKind::Texture, 1, 2),
            rec(ResourceKind::AudioClip, 0, 9),
            rec(ResourceKind::Texture, 0, 1),
        ];
        let table = TextureTable::from_records(&records);
        assert_eq!(table.len(), 2);
        assert!(table.locator(0).is_some());
        assert!(table.locator(1).is_some());
        assert!(table.locator(2).is_none());
        let mut indices: Vec<u32> = table.blob_indices().into_iter().collect();
        indices.sort_unstable();
        assert_eq!(indices, vec![1, 2]);
        assert!(TextureTable::from_records(&[]).is_empty());
    }
}
