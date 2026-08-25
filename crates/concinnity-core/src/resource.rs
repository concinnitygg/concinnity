//! Runtime resource tables: per-kind, handle-indexed views of a compiled blob's
//! resource stream. A resource (an audio clip, a texture, a mesh, a material, ...)
//! is compiled by cook and addressed at runtime by its dense per-kind handle. The
//! owning system reads the table by that handle instead of querying an ECS column
//! or scanning names, so a resource lives in a table it owns rather than as a
//! component. Renderer-free (the tables are plain handle-indexed data), so they
//! live here where the physics / audio subsystem crates can reach them; the
//! client re-exports them under `crate::resource::*`, alongside the engine-side
//! `install_resource_tables` that inserts them as World resources.

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use crate::ecs::{PayloadLocator, ResourceKind, ResourceRecord};

/// One loaded resource's runtime form. A payload resource (audio clip, and later
/// meshes / textures) carries a `PayloadLocator` into the blob payload section; a
/// data resource (a baked Material) carries its runtime bytes in `data_bytes`.
#[derive(Debug, Clone, Default)]
pub struct ResourceEntry {
    /// Where the compiled payload lives, for a payload resource.
    pub payload: Option<PayloadLocator>,
    /// The runtime bytes, for a data resource.
    pub data_bytes: Vec<u8>,
}

// Build a dense per-kind table from the blob's resource stream, indexed by
// handle: `table[handle]` is that resource's entry. Records of other kinds are
// ignored; a missing handle yields a default entry so indexing by any handle in
// range never panics. Each record's data bytes are MOVED into its entry (a
// record belongs to exactly one kind, so the per-kind builders never contend);
// the records are spent scaffolding once every table is built.
pub(crate) fn resource_table(
    records: &mut [ResourceRecord],
    kind: ResourceKind,
) -> Vec<ResourceEntry> {
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
    for record in records.iter_mut().filter(|r| r.resource_kind == tag) {
        table[record.handle as usize] = ResourceEntry {
            payload: record.payload.clone(),
            data_bytes: core::mem::take(&mut record.data_bytes),
        };
    }
    table
}

// Declare the per-kind tables. Every table is the same handle-indexed newtype
// over `ResourceEntry` with the same accessors, so they are generated from one
// list rather than hand-copied; a table whose kind carries no blob payload
// simply reports no locators. Bespoke accessors live in a plain `impl` below.
macro_rules! resource_tables {
    ($($name:ident => $kind:ident),* $(,)?) => {
        $(
            #[derive(Debug, Clone, Default)]
            /// A handle-indexed table of one resource kind.
            pub struct $name(pub Vec<ResourceEntry>);

            impl $name {
                /// Build the table from the blob's resource stream.
                pub fn from_records(records: &mut [ResourceRecord]) -> Self {
                    Self(resource_table(records, ResourceKind::$kind))
                }

                /// Number of resources of this kind; a handle is in range when
                /// its index is below this.
                pub fn len(&self) -> usize {
                    self.0.len()
                }

                /// Whether the table holds no resources.
                pub fn is_empty(&self) -> bool {
                    self.0.is_empty()
                }

                /// The payload locator for a handle, if the handle is in range
                /// and the resource has a compiled payload.
                pub fn locator(&self, handle: usize) -> Option<PayloadLocator> {
                    self.0.get(handle).and_then(|e| e.payload.clone())
                }

                /// Every locator in handle order (index == the resource's
                /// handle), skipping entries with no payload.
                pub fn locators(&self) -> impl Iterator<Item = (usize, PayloadLocator)> + '_ {
                    self.0
                        .iter()
                        .enumerate()
                        .filter_map(|(i, e)| e.payload.clone().map(|l| (i, l)))
                }

                /// Blob indices holding a payload of this kind. The graphics
                /// systems consult this to keep those blobs resident for the
                /// system that inits after them.
                pub fn blob_indices(&self) -> BTreeSet<u32> {
                    self.0
                        .iter()
                        .filter_map(|e| e.payload.as_ref().map(|l| l.blob_index))
                        .collect()
                }
            }
        )*
    };
}

resource_tables! {
    // Audio clips, indexed by `AudioClipHandle`. `AudioSystem` reads this at init.
    AudioClipTable => AudioClip,
    // Textures, indexed by `TextureHandle`. The renderer reads this at init to
    // build its shared texture pool. Every texture (file or procedural) has a
    // compiled payload, so an entry's `payload` is normally `Some`.
    TextureTable => Texture,
    // Color-grading LUTs, indexed by `ColorLutHandle`. The renderer uses only
    // the first (handle 0) and warns when a world declares more.
    ColorLutTable => ColorLut,
    // IBL environment maps, indexed by `EnvironmentMapHandle`. The renderer uses
    // only the first (handle 0); a world declares at most one.
    EnvironmentMapTable => EnvironmentMap,
    // Fonts, indexed by `FontHandle`. The renderer reads this at init to build
    // its glyph atlases + metrics; every font has a compiled SDF-atlas payload.
    FontTable => Font,
    // Static meshes, indexed by `MeshHandle`. Mesh shares its handle space with
    // the still-component geometry producers (ProceduralMesh, VoxelChunk,
    // mesh-kind File): the Mesh block leads that space, so this table covers
    // handles `0..len` and the runtime appends the component-produced geometry
    // after it in the same block order cook assigned.
    MeshTable => Mesh,
    // Skinned meshes, indexed by `SkinnedMeshHandle`. A hybrid entry: `payload`
    // locates the compiled geometry (vertices + indices + skeleton) while
    // `data_bytes` carries the baked runtime fields (placement, material/texture
    // handles, capsule, spawn reserve) as a `(name_id, SkinnedMesh)` postcard
    // tuple -- `asset_id` is serde-skipped on the schema struct, so the interned
    // name travels beside it for the runtime's spawn-by-name registration.
    SkinnedMeshTable => SkinnedMesh,
    // Materials, indexed by `MaterialHandle`. Unlike the payload-backed tables,
    // a Material is a DATA resource: cook bakes its validated args into the
    // record's `data_bytes` (no blob payload), so `data_bytes(handle)` returns
    // the serialized `Material` the renderer deserializes to build its map.
    MaterialTable => Material,
}

impl SkinnedMeshTable {
    /// Whether any skinned mesh declares a character capsule; gates whether the
    /// world needs a PhysicsSystem.
    pub fn has_capsule(&self) -> bool {
        self.0.iter().any(|e| {
            postcard::from_bytes::<(u32, crate::components::SkinnedMesh)>(&e.data_bytes)
                .is_ok_and(|(_, sm)| sm.capsule.is_some())
        })
    }
}

impl MaterialTable {
    /// The baked material bytes for a handle, if the handle is in range.
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
        let mut records = vec![
            rec(ResourceKind::AudioClip, 1, 0),
            rec(ResourceKind::Mesh, 0, 5),
            rec(ResourceKind::AudioClip, 0, 0),
        ];
        let table = AudioClipTable::from_records(&mut records);
        assert_eq!(table.0.len(), 2);
        assert!(table.locator(0).is_some());
        assert!(table.locator(1).is_some());
        // A handle past the end resolves to None rather than panicking.
        assert!(table.locator(2).is_none());
    }

    #[test]
    fn empty_stream_yields_an_empty_table() {
        let table = AudioClipTable::from_records(&mut []);
        assert!(table.0.is_empty());
        assert!(table.blob_indices().is_empty());
        assert_eq!(table.locators().count(), 0);
    }

    #[test]
    fn blob_indices_collects_every_payload_blob() {
        let mut records = vec![
            rec(ResourceKind::AudioClip, 0, 0),
            rec(ResourceKind::AudioClip, 1, 3),
        ];
        let table = AudioClipTable::from_records(&mut records);
        let mut indices: Vec<u32> = table.blob_indices().into_iter().collect();
        indices.sort_unstable();
        assert_eq!(indices, vec![0, 3]);
    }

    #[test]
    fn texture_table_is_dense_by_handle_and_ignores_other_kinds() {
        // A texture record and an audio record interleaved; the texture table
        // keeps only the textures, placed at their handle index.
        let mut records = vec![
            rec(ResourceKind::Texture, 1, 2),
            rec(ResourceKind::AudioClip, 0, 9),
            rec(ResourceKind::Texture, 0, 1),
        ];
        let table = TextureTable::from_records(&mut records);
        assert_eq!(table.len(), 2);
        assert!(table.locator(0).is_some());
        assert!(table.locator(1).is_some());
        assert!(table.locator(2).is_none());
        let mut indices: Vec<u32> = table.blob_indices().into_iter().collect();
        indices.sort_unstable();
        assert_eq!(indices, vec![1, 2]);
        assert!(TextureTable::from_records(&mut []).is_empty());
    }

    #[test]
    fn every_table_reads_only_its_own_kind() {
        // One record per kind in one stream; each table sees exactly its own.
        let mut records = vec![
            rec(ResourceKind::AudioClip, 0, 0),
            rec(ResourceKind::Texture, 0, 1),
            rec(ResourceKind::ColorLut, 0, 2),
            rec(ResourceKind::EnvironmentMap, 0, 3),
            rec(ResourceKind::Font, 0, 4),
            rec(ResourceKind::Mesh, 0, 5),
            rec(ResourceKind::SkinnedMesh, 0, 6),
            rec(ResourceKind::Material, 0, 7),
        ];
        assert_eq!(AudioClipTable::from_records(&mut records).len(), 1);
        assert_eq!(TextureTable::from_records(&mut records).len(), 1);
        assert_eq!(ColorLutTable::from_records(&mut records).len(), 1);
        assert_eq!(EnvironmentMapTable::from_records(&mut records).len(), 1);
        assert_eq!(FontTable::from_records(&mut records).len(), 1);
        assert_eq!(MeshTable::from_records(&mut records).len(), 1);
        assert_eq!(SkinnedMeshTable::from_records(&mut records).len(), 1);
        assert_eq!(MaterialTable::from_records(&mut records).len(), 1);
    }
}
