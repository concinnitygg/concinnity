//! What the renderer publishes about each loaded skinned mesh, indexed by its
//! handle: its interned name, its source-file skin selector, and its
//! morph-target names.
//!
//! The animation correlation web (`SkeletonPose.mesh_id`, `CharacterRig.target`,
//! `AnimationParams.target`, `GroundProbes.target`, `CameraProbe.target`) is
//! keyed by the mesh's dense `SkinnedMeshHandle`. A consumer that starts from a
//! name, such as a typed animation command, goes through [`SkinnedMeshNameIndex`].

use alloc::string::String;
use alloc::vec::Vec;

use hashbrown::HashMap;

use crate::ecs::SkinnedMeshHandle;
use crate::ecs::asset_id::AssetId;

/// Interned-name to handle index for the skinned meshes.
#[derive(Debug, Default, Clone)]
pub struct SkinnedMeshNameIndex(pub HashMap<AssetId, SkinnedMeshHandle>);

impl SkinnedMeshNameIndex {
    /// The handle for an interned mesh name. An id the index does not hold
    /// reads as the handle of the same value, which is how a world built
    /// without the renderer correlates its interned `target` names.
    pub fn get(&self, name_id: AssetId) -> SkinnedMeshHandle {
        self.0
            .get(&name_id)
            .copied()
            .unwrap_or(SkinnedMeshHandle::new(name_id.0))
    }
}

/// Each skinned mesh's source-file skin selector, indexed by handle. A clip
/// re-imported from the mesh's file resolves against the same skin.
#[derive(Debug, Default, Clone)]
pub struct SkinnedMeshSkinIndex(pub Vec<u32>);

impl SkinnedMeshSkinIndex {
    /// The selector for a mesh handle; 0 (the file's first skinned mesh) when
    /// the handle is unknown, matching the asset default.
    pub fn get(&self, handle: SkinnedMeshHandle) -> u32 {
        self.0.get(handle.index()).copied().unwrap_or(0)
    }
}

/// Each skinned mesh's morph-target names, indexed by handle.
#[derive(Debug, Default, Clone)]
pub struct SkinnedMeshMorphNames(pub Vec<Vec<String>>);

impl SkinnedMeshMorphNames {
    /// The morph-target names of the mesh behind `handle`; empty when unknown.
    pub fn of(&self, handle: SkinnedMeshHandle) -> &[String] {
        self.0.get(handle.index()).map_or(&[], Vec::as_slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    #[test]
    fn known_name_returns_its_handle() {
        let mut map = HashMap::new();
        map.insert(AssetId(10), SkinnedMeshHandle::new(0));
        map.insert(AssetId(20), SkinnedMeshHandle::new(1));
        let index = SkinnedMeshNameIndex(map);
        assert_eq!(index.get(AssetId(20)), SkinnedMeshHandle::new(1));
    }

    #[test]
    fn unknown_name_falls_back_to_the_id_value() {
        let index = SkinnedMeshNameIndex::default();
        assert_eq!(index.get(AssetId(42)), SkinnedMeshHandle::new(42));
    }

    #[test]
    fn unknown_handles_read_the_defaults() {
        let skins = SkinnedMeshSkinIndex(vec![3]);
        assert_eq!(skins.get(SkinnedMeshHandle::new(0)), 3);
        assert_eq!(skins.get(SkinnedMeshHandle::new(5)), 0);
        let names = SkinnedMeshMorphNames(vec![vec!["jaw".to_string()]]);
        assert_eq!(names.of(SkinnedMeshHandle::new(0)), ["jaw"]);
        assert!(names.of(SkinnedMeshHandle::new(1)).is_empty());
    }
}
