//! The skeletal-animation vocabulary and the CPU compute over it: skeletons and
//! the clips that animate them, pose blending and the animation state machine,
//! two-bone IK, per-joint proportions, character shapes, root motion, and morph
//! targets with the weights that drive them, and the playback system over them.
pub mod anim_graph;
#[cfg(test)]
mod bench;
pub(crate) mod character_rig;
pub mod character_shape;
pub(crate) mod clip;
pub mod flat_blend;
pub(crate) mod graph;
pub(crate) mod ground_probes;
pub mod ik;
pub mod morph_targets;
pub mod morph_weights;
pub(crate) mod params;
pub mod pose_blend;
pub mod pose_scratch;
pub mod proportions;
pub mod root_motion;
pub(crate) mod root_motion_event;
pub mod skeleton;
pub(crate) mod skeleton_pose;
pub mod skinned_mesh_map;
mod system;

pub use system::{AnimationReloadEntry, AnimationSystem, GraphStateReport};
