// SkinnedMesh name -> handle index.
//
// The animation correlation web (`SkeletonPose.mesh_id`, `CharacterRig.target`,
// `AnimParams.target`, `GroundProbes.target`, `CameraProbe.target`) is keyed by
// the mesh's dense `SkinnedMeshHandle`, matching the authored references
// (`Animation.target`, `AnimGraph.target`, `FollowController.target`) directly.
// The one consumer that still starts from a NAME is the debug WebSocket's
// animation commands (`anim-crossfade` / `anim-param` / `anim-state`), which
// resolve the user-typed name to its interned id: this index, published by
// GraphicsSystem from the baked SkinnedMesh data (which carries each mesh's
// interned name id), translates that id to the handle keying the web.

use std::collections::HashMap;

use crate::ecs::SkinnedMeshHandle;
use crate::ecs::asset_id::AssetId;

// Interned-name -> handle index for the skinned meshes, published as a world
// resource by GraphicsSystem while it loads the SkinnedMesh resource table.
#[derive(Debug, Default, Clone)]
pub(crate) struct SkinnedMeshNameIndex(pub HashMap<AssetId, SkinnedMeshHandle>);

// Each skinned mesh's source-file skin selector, indexed by handle and
// published alongside the name index. An animation clip re-imported at
// hot-reload must resolve against the same skin its target mesh was cooked
// from; the reload catalogue reads the selector from here at init.
#[derive(Debug, Default, Clone)]
pub(crate) struct SkinnedMeshSkinIndex(pub Vec<u32>);

impl SkinnedMeshSkinIndex {
    // The selector for a mesh handle; 0 (the file's first skinned mesh) when
    // the handle is unknown, matching the asset default.
    pub(crate) fn get(&self, handle: SkinnedMeshHandle) -> u32 {
        self.0.get(handle.index()).copied().unwrap_or(0)
    }
}

impl SkinnedMeshNameIndex {
    // The handle for an interned mesh name. Falls back to reinterpreting the
    // id value as a handle when absent: a unit test that exercises the
    // animation systems without the renderer deserializes its `target` names
    // through the resolver's interner fallback, so both sides of the
    // correlation carry the interned id and the identity mapping matches them.
    // In a real build every name a debug command can address is in the index.
    pub(crate) fn get(&self, name_id: AssetId) -> SkinnedMeshHandle {
        self.0
            .get(&name_id)
            .copied()
            .unwrap_or(SkinnedMeshHandle(name_id.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_name_returns_its_handle() {
        let mut map = HashMap::new();
        map.insert(AssetId(10), SkinnedMeshHandle(0));
        map.insert(AssetId(20), SkinnedMeshHandle(1));
        let index = SkinnedMeshNameIndex(map);
        assert_eq!(index.get(AssetId(20)), SkinnedMeshHandle(1));
    }

    #[test]
    fn unknown_name_falls_back_to_the_id_value() {
        let index = SkinnedMeshNameIndex::default();
        assert_eq!(index.get(AssetId(42)), SkinnedMeshHandle(42));
    }
}
