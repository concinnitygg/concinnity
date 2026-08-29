// What a step reports back: the ray hits a query answers, the sensor regions
// something crossed, and the contacts strong enough to be worth hearing about.
// All of it resolved to plain handles and vectors while the step still has
// both sides of the pair in hand.

use crate::physics::BodyHandle;

/// One raycast hit: the surface point, its outward normal, and the distance
/// from the ray origin.
#[derive(Debug, Clone, Copy)]
pub struct RayHit {
    /// World-space surface point the ray hit.
    pub point: [f32; 3],
    /// Unit-length normal.
    pub normal: [f32; 3],
    /// Distance from the ray origin to the hit.
    pub distance: f32,
}

/// One boundary crossing of a sensor region recorded during a step: something
/// began or stopped overlapping the sensor tagged `tag`.
#[derive(Debug, Clone, Copy)]
pub struct SensorCrossing {
    /// The caller's tag for the crossed sensor.
    pub tag: u64,
    /// The crossing body, `None` when it left the simulation this step.
    pub other: Option<BodyHandle>,
    /// `true` on the way in, `false` on the way out.
    pub entered: bool,
}

/// One contact strong enough to pass the simulation's force threshold.
#[derive(Debug, Clone, Copy)]
pub struct ContactHit {
    /// One side of the contact.
    pub a: BodyHandle,
    /// The other side of the contact.
    pub b: BodyHandle,
    /// Deepest contact point, in world space.
    pub point: [f32; 3],
    /// Unit-length normal, pointing from `a` toward `b`.
    pub normal: [f32; 3],
    /// Total contact impulse magnitude.
    pub impulse: f32,
}
