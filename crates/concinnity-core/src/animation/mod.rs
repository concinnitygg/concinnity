//! The skeletal-animation vocabulary and the CPU compute over it: skeletons and
//! the clips that animate them, pose blending and the animation state machine,
//! two-bone IK, per-joint proportions, root motion, and morph targets with the
//! weights that drive them.
pub mod anim_graph;
pub mod ik;
pub mod morph_targets;
pub mod morph_weights;
pub mod pose_blend;
pub mod pose_scratch;
pub mod proportions;
pub mod root_motion;
pub mod skeleton;
