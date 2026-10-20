//! Live `CharacterShape` re-resolution. Every shape resolves once at init
//! ([`crate::animation::character_shape`]); an editor dragging a slider needs
//! the same resolution against the running world's `SkeletonPose` each frame,
//! without rebuilding the world. This is that seam: what a mesh exposes to a
//! shape (its morph-target and joint names) and a re-seed of its pose through
//! an edited shape.

use alloc::string::String;
use alloc::vec::Vec;

use crate::animation::character_shape;
use crate::animation::skinned_mesh_map::{SkinnedMeshMorphNames, SkinnedMeshNameIndex};
use crate::components::{CharacterCapsule, CharacterRig, CharacterShape, SkeletonPose};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{SkinnedMeshHandle, World};

/// The names a shape targeting one mesh can reference.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShapeTarget {
    /// The mesh's morph targets, in target order.
    pub morph_names: Vec<String>,
    /// The mesh's skeleton joints, in joint order.
    pub joint_names: Vec<String>,
}

/// The handle of the skinned mesh interned as `name_id`, if the running world
/// loaded one under that name.
pub fn mesh_handle(world: &World, name_id: AssetId) -> Option<SkinnedMeshHandle> {
    world
        .resource::<SkinnedMeshNameIndex>()?
        .0
        .get(&name_id)
        .copied()
}

/// What the mesh behind `handle` exposes to a shape. `None` until the mesh
/// has a pose in the world.
pub fn target(world: &World, handle: SkinnedMeshHandle) -> Option<ShapeTarget> {
    let pose = world
        .query::<SkeletonPose>()
        .find(|p| p.mesh_id == handle)?;
    Some(ShapeTarget {
        morph_names: morph_names(world, handle),
        joint_names: pose
            .skeleton
            .joints()
            .iter()
            .map(|j| j.name.clone())
            .collect(),
    })
}

/// Re-seed every pose of `shape.target` through `shape`, and size the mesh's
/// rig capsule from `capsule` (the authored dimensions) through the new
/// proportions. Returns whether a pose was found. The rig's physics body is
/// not resized here; the rebuild that follows a committed edit does that.
pub fn apply(
    world: &mut World,
    shape: &CharacterShape,
    capsule: Option<&CharacterCapsule>,
) -> bool {
    let Some(handle) = shape.target else {
        return false;
    };
    let morph_names = morph_names(world, handle);
    let mut dims = None;
    let mut found = false;
    for pose in world
        .query_mut::<SkeletonPose>()
        .filter(|p| p.mesh_id == handle)
    {
        let layers = character_shape::layers(shape, &pose.skeleton, &morph_names);
        if let Some(c) = capsule {
            dims = Some(character_shape::proportioned_capsule(
                c,
                &pose.skeleton,
                &layers.proportions,
            ));
        }
        pose.set_shape(layers.morph_base, layers.proportions);
        found = true;
    }
    if let Some((half_height, radius)) = dims {
        for rig in world
            .query_mut::<CharacterRig>()
            .filter(|r| r.target == handle)
        {
            rig.half_height = half_height.max(0.05);
            rig.radius = radius.max(0.05);
        }
    }
    found
}

fn morph_names(world: &World, handle: SkinnedMeshHandle) -> Vec<String> {
    world
        .resource::<SkinnedMeshMorphNames>()
        .map(|m| m.of(handle).to_vec())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;

    use super::*;
    use crate::animation::character_shape::tests::chain;
    use crate::components::{JointProportion, ShapeSlider};
    use crate::gfx::render_types::SkinnedIndex;

    fn world_with_pose() -> World {
        let mut world = World::new();
        world.add_component(SkeletonPose::new(
            SkinnedMeshHandle::new(0),
            SkinnedIndex(0),
            chain(),
        ));
        world.insert_resource(SkinnedMeshMorphNames(vec![vec![
            "jaw+".to_string(),
            "jaw-".to_string(),
        ]]));
        world
    }

    #[test]
    fn target_lists_the_pose_joints_and_published_morph_names() {
        let world = world_with_pose();
        let t = target(&world, SkinnedMeshHandle::new(0)).expect("pose present");
        assert_eq!(t.morph_names, ["jaw+", "jaw-"]);
        assert_eq!(t.joint_names, ["root", "head"]);
        assert!(target(&world, SkinnedMeshHandle::new(1)).is_none());
    }

    #[test]
    fn apply_reseeds_the_pose_and_reports_a_missing_target() {
        let mut world = world_with_pose();
        let shape = CharacterShape {
            target: Some(SkinnedMeshHandle::new(0)),
            sliders: vec![ShapeSlider {
                name: "jaw".into(),
                value: -0.5,
            }],
            proportions: vec![JointProportion {
                joint: "root".into(),
                scale: 2.0,
                length: 0.0,
            }],
            ..Default::default()
        };
        assert!(apply(&mut world, &shape, None));
        let pose = world.query::<SkeletonPose>().next().unwrap();
        assert_eq!(pose.morph_base, [0.0, 0.5]);
        assert_eq!(pose.joint_matrices[0][0][0], 2.0);
        assert!(pose.updated);
        let none = CharacterShape::default();
        assert!(!apply(&mut world, &none, None));
    }
}
