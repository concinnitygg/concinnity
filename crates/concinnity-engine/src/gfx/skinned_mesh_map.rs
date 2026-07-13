// SkinnedMesh handle -> asset id bridge.
//
// A SkinnedMesh's authored references (`Animation.target`, `AnimGraph.target`,
// `FollowController.target`) bake to the mesh's dense `SkinnedMeshHandle`, while
// the runtime animation web that correlates against them (`SkeletonPose.mesh_id`,
// `CharacterRig.target`, `AnimParams.target`, `GroundProbes.target`,
// `CameraProbe.target`) is keyed by the mesh's `AssetId`. GraphicsSystem drains
// the SkinnedMesh column first (handle == drain index == cook declaration order)
// and publishes this map; the animation and third-person systems, which init
// after it, resolve each authored handle back to the mesh's asset id through it so
// the correlation web stays comparable.

use crate::ecs::SkinnedMeshHandle;
use crate::ecs::asset_id::AssetId;

// Handle-indexed table of each SkinnedMesh's runtime asset id, published as a
// world resource by GraphicsSystem during its SkinnedMesh drain.
#[derive(Debug, Default, Clone)]
pub struct SkinnedMeshHandleMap(pub Vec<AssetId>);

impl SkinnedMeshHandleMap {
    // The asset id a SkinnedMesh handle addresses. An in-range handle returns the
    // drained mesh's id. An out-of-range handle falls back to reinterpreting the
    // handle value as an id: in a valid build a `target` always names a real
    // SkinnedMesh (cook's cross-reference check rejects the rest) so this never
    // fires there, and it is the path a unit test that exercises the animation
    // systems without the renderer takes -- the map is then empty and the handle
    // carries the interned id from the resolver's validation fallback, so it maps
    // to itself, matching the pre-handle behaviour.
    pub fn get(&self, handle: SkinnedMeshHandle) -> AssetId {
        self.0
            .get(handle.index())
            .copied()
            .unwrap_or(AssetId(handle.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_range_handle_returns_the_drained_id() {
        let map = SkinnedMeshHandleMap(vec![AssetId(10), AssetId(20), AssetId(30)]);
        assert_eq!(map.get(SkinnedMeshHandle(0)), AssetId(10));
        assert_eq!(map.get(SkinnedMeshHandle(2)), AssetId(30));
    }

    #[test]
    fn out_of_range_and_empty_map_fall_back_to_the_handle_value() {
        // The unit-test path: no renderer published a map, so the handle (which
        // carries the interned id from the resolver fallback) maps to itself.
        let empty = SkinnedMeshHandleMap::default();
        assert_eq!(empty.get(SkinnedMeshHandle(42)), AssetId(42));
        let short = SkinnedMeshHandleMap(vec![AssetId(10)]);
        assert_eq!(short.get(SkinnedMeshHandle(5)), AssetId(5));
    }
}
