//! The world around the camera: a planet's ground and sea, the sky, open and
//! simulated water, terrain, and the wind that moves them.

pub mod planet;
pub mod sky;
pub mod terrain;
pub mod water;
pub(crate) mod wind;

/// Acceleration due to gravity in meters per second squared: a planet's
/// default pull, and the pull of a world without one. The third-person
/// controller's jump takeoff is sized from it so the rig's fall matches.
pub const GRAVITY: f32 = 20.0;
