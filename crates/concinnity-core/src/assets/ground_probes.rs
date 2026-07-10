// src/assets/ground_probes.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

/// One downward ground probe: a ray and, once physics has answered it, the
/// hit it found.
#[derive(Debug, Clone, Copy, Default)]
pub struct GroundProbe {
    /// Ray origin in world space.
    pub origin: [f32; 3],
    /// Maximum hit distance along -Y from the origin.
    pub max_dist: f32,
    /// The surface point and normal the ray found, or `None` (unanswered,
    /// or no ground within range).
    pub hit: Option<([f32; 3], [f32; 3])>,
}

/// Runtime-only ground-probe exchange for one animated character.
///
/// `AnimationSystem` publishes one per graph target with IK chains and
/// refreshes each probe's ray from the animated end-joint position every
/// frame; `PhysicsSystem` (which steps earlier) answers the previous frame's
/// rays with raycast hits. The one-frame-stale ray origin is invisible at
/// animation rates.
///
/// Not authored in world files: it has no `args`.
#[derive(Debug, Clone, Default)]
pub struct GroundProbes {
    /// The `SkinnedMesh` asset whose IK chains these probes serve.
    pub target: AssetId,
    /// One probe per IK chain, in the graph's `ik_chains` order.
    pub probes: Vec<GroundProbe>,
}

/// `GroundProbes` is never authored, so its args are empty.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct GroundProbesArgs {}

impl Component for GroundProbes {
    const NAME: &'static str = "GroundProbes";
    const ORIGIN: AssetOrigin = AssetOrigin::RuntimeOnly;
    type Args = GroundProbesArgs;

    fn to_args(&self) -> GroundProbesArgs {
        GroundProbesArgs {}
    }
    fn from_args(_: GroundProbesArgs) -> Self {
        // A RuntimeOnly component never round-trips through args; real
        // instances are built by AnimationSystem.
        Self::default()
    }
}
