//! The `Planet` asset.

use crate::components::TerrainLayer;
use crate::ecs::MaterialHandle;
use alloc::vec::Vec;

/// A whole world to stand on: a sphere of ground `radius` meters around
/// `center`, raised into hills, rendered with `material`, and pulling every
/// body toward its center.
///
/// The ground is generated from `seed`: hills up to `amplitude` meters above
/// the radius, the broadest about `feature_size` meters across, with
/// `octaves` layers of finer detail on them, each half the width and half the
/// height of the one before. The same seed gives the same planet on every run.
/// Nothing about the ground is stored: it is built around the camera as it
/// moves, finer near it and coarser toward the horizon, so a planet costs the
/// same to load whatever its size.
///
/// Gravity points at the center from wherever a body is, `gravity` meters per
/// second squared, so a first-person camera with a [RigidBody](#rigidbody)
/// walks all the way around. Up is always away from the center: looking
/// around turns about it, and the horizon stays level.
///
/// Far from where the world started, positions stay as precise as they were
/// near it: the world is simulated around the camera, wherever on the planet
/// it is. A world has at most one planet.
///
/// ```rust
/// # use concinnity_core::components::Planet;
/// Planet {
///     radius: 20_000.0,
///     center: [0.0, -20_000.0, 0.0],
///     amplitude: 35.0,
///     seed: 4,
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct Planet {
    /// World-space center. The default puts the top of a planet of the
    /// default radius at the origin.
    #[asset(default = [0.0, -50_000.0, 0.0])]
    pub center: [f32; 3],
    /// Radius of the lowest ground, in meters. At least 100.
    #[asset(default = 50_000.0)]
    pub radius: f32,
    /// Highest hilltop above `radius`, in meters. An `amplitude` of 0 gives a
    /// smooth sphere.
    #[asset(default = 40.0)]
    pub amplitude: f32,
    /// Width of the broadest hills, in meters.
    #[asset(default = 2_000.0)]
    pub feature_size: f32,
    /// Layers of ever finer detail on the hills. Clamped to [1, 16].
    #[asset(default = 8)]
    pub octaves: u32,
    /// Shapes the ground: each seed gives a different planet.
    pub seed: u32,
    /// The [Material](#material) the ground renders with.
    pub material: Option<MaterialHandle>,
    /// Pull toward the center, in meters per second squared.
    #[asset(default = crate::physics::GRAVITY)]
    pub gravity: f32,
    /// The grass that grows on the planet, one entry per look.
    pub layers: Vec<TerrainLayer>,
}

impl Planet {
    /// The surface this planet's args describe.
    pub fn shape(&self) -> crate::planet::PlanetShape {
        crate::planet::PlanetShape {
            center: crate::planet::dvec_from_f32(self.center),
            radius: f64::from(self.radius),
            amplitude: f64::from(self.amplitude),
            feature_size: f64::from(self.feature_size),
            octaves: self.octaves,
            seed: self.seed,
        }
    }
}
