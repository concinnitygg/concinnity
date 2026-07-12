// src/assets/runtime_component.rs
//
// Machinery for the RuntimeOnly components whose `Component` impls are generated
// centrally (see `cn_impl_components!`). A RuntimeOnly component is never
// authored in a world and never round-trips through a blob: real instances are
// built by the systems that own them (GraphicsSystem, the animation and
// third-person controllers, the load-time Prop decomposition). Their `Args` type
// therefore carries no authored fields, so they all share this one empty struct
// instead of a distinct per-type args struct.

/// The authored-args placeholder for every RuntimeOnly component. It has no
/// fields because these components are runtime-built, not declared.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct RuntimeArgs {}

// Placeholder constructors for the few RuntimeOnly components that do not derive
// `Default`. The generated `from_args` calls these, but it is never actually
// invoked (the components are built by their owning systems); the value only
// has to be a valid instance so the impl compiles.

use crate::assets::{AnimParams, CharacterRig, SkeletonPose};
use crate::ecs::asset_id::AssetId;
use crate::gfx::skinning;

pub(crate) fn skeleton_pose() -> SkeletonPose {
    SkeletonPose::new(AssetId::default(), 0, skinning::Skeleton::new(Vec::new()))
}

pub(crate) fn character_rig() -> CharacterRig {
    CharacterRig::new(AssetId::default(), 0, skinning::IDENTITY, 0.5, 0.3)
}

pub(crate) fn anim_params() -> AnimParams {
    AnimParams::new(AssetId::default(), Vec::new())
}
