// src/gfx/pose_scratch.rs

use alloc::vec::Vec;

use crate::gfx::transform::Mat4;

/// Reusable buffers for the per-frame pose sampling chain (sample, blend, IK,
/// skinning). Each animated target owns one; the buffers reach steady-state
/// capacity on the first sampled frame and are reused in place after that, so
/// steady-state sampling performs no heap allocation.
#[derive(Debug, Default)]
pub struct PoseScratch {
    /// Primary local-pose accumulator. Holds the finished blended pose.
    pub locals: Vec<Mat4>,
    /// Secondary local-pose buffer (a transition's outgoing pose, or the
    /// world-matrix scratch an IK solve composes through).
    pub aux: Vec<Mat4>,
    /// Per-clip sample buffer fed into the weighted blend.
    pub clip: Vec<Mat4>,
    /// Per-member blend weights, or a clip's sampled morph weights.
    pub weights: Vec<f32>,
    /// Morph-weight accumulator for a weighted multi-clip blend.
    pub morph: Vec<f32>,
}
