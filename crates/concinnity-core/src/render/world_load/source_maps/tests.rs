use super::*;
use crate::resource::{TextureSource, TextureSources};
use alloc::string::ToString;
use alloc::vec;

// A fresh map is empty, and pushing an entry is what gives it a length.
#[test]
fn texture_map_starts_empty_and_counts_pushed_entries() {
    let mut map = TextureSourceMap::new();
    assert!(map.is_empty());
    assert_eq!(map.len(), 0);

    map.push_texture("assets/wall.png".to_string(), 0, 4);
    map.push_texture("assets/scene.glb".to_string(), 2, 7);

    assert!(!map.is_empty());
    assert_eq!(map.len(), 2);
    // Each entry keeps the slot + image index it was pushed with, so a reload
    // rewrites the right pool slot.
    assert_eq!((map.entries[0].slot, map.entries[0].image_index), (4, 0));
    assert_eq!((map.entries[1].slot, map.entries[1].image_index), (7, 2));
    assert_eq!(map.entries[1].source(), "assets/scene.glb");
}

// Every slot inside the pool is named; only file-backed ones are reloadable,
// and the catalog past the pool is ignored.
#[test]
fn the_texture_catalog_names_every_slot_and_keeps_file_backed_ones() {
    let source = |name_id, source: &str| TextureSource {
        name_id,
        source: source.to_string(),
        image_index: 0,
    };
    let sources = TextureSources(vec![
        source(10, "a.png"),
        source(11, ""),
        source(12, "c.png"),
    ]);
    let (map, names) = texture_catalog(Some(&sources), 2);
    assert_eq!(map.len(), 1);
    assert_eq!(map.entries[0].slot, 0);
    assert_eq!(names.get(&AssetId(11)), Some(&1));
    assert!(!names.contains_key(&AssetId(12)));
    assert!(texture_catalog(None, 2).0.is_empty());
}

fn meta(source: &str) -> MeshSourceMeta {
    MeshSourceMeta {
        source: source.to_string(),
        primitive_index: 1,
        lod_levels: 2,
        lod_distances: vec![10.0],
    }
}

// A mesh handle with no draw entry, or an empty one, has nowhere to reload
// into; one Mesh can own several draw slots (a mesh shared by many Props).
#[test]
fn mesh_source_map_keeps_every_draw_and_omits_handles_without_draws() {
    let sources = HashMap::from([
        (0, meta("assets/drawn.glb")),
        (1, meta("assets/unreferenced.glb")),
        (2, meta("assets/emptied.glb")),
    ]);
    let draws = HashMap::from([(0, vec![DrawIndex(3), DrawIndex(4)]), (2, Vec::new())]);
    let map = mesh_source_map(&sources, &draws);
    assert_eq!(map.len(), 1);
    let entry = &map.entries[0];
    assert_eq!(entry.source(), "assets/drawn.glb");
    assert_eq!(entry.draw_indices, vec![DrawIndex(3), DrawIndex(4)]);
    assert_eq!((entry.primitive_index, entry.lod_levels), (1, 2));
}

// A procedural mesh with no handle, no draw entry, or an empty one is omitted.
#[test]
fn procedural_mesh_source_map_omits_meshes_without_draws() {
    let args = ProceduralMesh::default;
    let snapshot = HashMap::from([
        (AssetId(1), ("drawn".to_string(), args())),
        (AssetId(2), ("no_handle".to_string(), args())),
        (AssetId(3), ("unreferenced".to_string(), args())),
        (AssetId(4), ("emptied".to_string(), args())),
    ]);
    let handles = HashMap::from([(AssetId(1), 0), (AssetId(3), 1), (AssetId(4), 2)]);
    let draws = HashMap::from([(0, vec![DrawIndex(5)]), (2, Vec::new())]);
    let map = procedural_mesh_source_map(&snapshot, &handles, &draws);
    assert_eq!(map.len(), 1);
    assert_eq!(map.entries[0].name, "drawn");
    assert_eq!(map.entries[0].draw_indices, vec![DrawIndex(5)]);
}

#[test]
fn a_skinned_entry_reads_its_source() {
    let entry = SkinnedMeshSourceEntry {
        source: "assets/fox.glb".to_string(),
        skin_index: 0,
        skinned_index: SkinnedIndex(0),
        vertex_base: 0,
        vertex_count: 3,
        index_count: 3,
        joint_count: 1,
    };
    assert_eq!(entry.source(), "assets/fox.glb");
}
