// Spot-light schema.

use crate::math::cos;
use crate::math::vec3::{MIN_DIRECTION_LEN, normalize_or};

/// A cone-shaped local light: a point light restricted to the cone around
/// `direction`, with a soft edge between `inner_angle` and `outer_angle`.
///
/// Distance attenuation matches [`PointLight`](#pointlight); the cone adds an
/// angular falloff that is full brightness inside the inner cone and fades to
/// black at the outer cone. Spot lights share the same per-scene local-light
/// budget as point lights and are culled by the same clustered pass. Secondary
/// effects (volumetric fog, SDF raymarching, and reflection-probe capture) do
/// not consider them.
///
/// ```rust
/// # use concinnity_core::components::SpotLight;
/// SpotLight {
///     position: [0.0, 4.0, -2.0],
///     direction: [0.0, -1.0, 0.0],
///     color: [1.0, 0.9, 0.7],
///     intensity: 20.0,
///     range: 10.0,
///     inner_angle: 18.0,
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
pub struct SpotLight {
    /// World-space position of the light source.
    #[asset(default = [0.0, 4.0, 0.0])]
    pub position: [f32; 3],
    /// Direction the cone points, away from the light. Does not need to be
    /// normalized; defaults to straight down when degenerate.
    #[asset(default = [0.0, -1.0, 0.0])]
    pub direction: [f32; 3],
    /// Linear-space RGB color of the light.
    #[asset(default = [1.0, 1.0, 1.0])]
    pub color: [f32; 3],
    /// Intensity multiplier applied to the color.
    #[asset(default = 20.0)]
    pub intensity: f32,
    /// Maximum reach in world units; attenuation is zero at this distance.
    #[asset(default = 10.0)]
    pub range: f32,
    /// Half-angle in degrees of the fully lit inner cone. Clamped to
    /// `outer_angle`.
    #[asset(default = 18.0)]
    pub inner_angle: f32,
    /// Half-angle in degrees at which the cone fades to black. Clamped to
    /// (0, 89.9].
    #[asset(default = 30.0)]
    pub outer_angle: f32,
    /// Whether this light casts shadows. Shadowed spots claim one slice of the
    /// spot shadow map in declaration order; once the slices are used up the
    /// remaining spots still light the scene but cast nothing.
    #[asset(default = true)]
    pub cast_shadows: bool,
}

/// Widest half-angle a spot cone may open to. Past this the cone degenerates
/// toward a hemisphere and the clustered sphere bound stops being useful.
pub const SPOT_MAX_ANGLE_DEG: f32 = 89.9;

impl SpotLight {
    /// Unit-length cone axis, falling back to straight down when the authored
    /// `direction` is degenerate.
    pub fn unit_direction(&self) -> [f32; 3] {
        normalize_or(self.direction, MIN_DIRECTION_LEN, [0.0, -1.0, 0.0])
    }

    /// Cosine of the inner half-angle: the widest angle still at full brightness.
    pub fn cos_inner(&self) -> f32 {
        cos(self.inner_angle.clamp(0.0, self.outer_angle).to_radians())
    }

    /// Cosine of the outer half-angle: the angle at which the cone reaches black.
    pub fn cos_outer(&self) -> f32 {
        cos(self.outer_angle.clamp(0.0, SPOT_MAX_ANGLE_DEG).to_radians())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spot(direction: [f32; 3], inner: f32, outer: f32) -> SpotLight {
        SpotLight {
            direction,
            inner_angle: inner,
            outer_angle: outer,
            ..SpotLight::default()
        }
    }

    #[test]
    fn spot_direction_normalizes() {
        let d = spot([0.0, -4.0, 0.0], 10.0, 20.0).unit_direction();
        assert_eq!(d, [0.0, -1.0, 0.0]);
    }

    #[test]
    fn degenerate_spot_direction_falls_back_to_down() {
        assert_eq!(
            spot([0.0; 3], 10.0, 20.0).unit_direction(),
            [0.0, -1.0, 0.0]
        );
    }

    // The shader divides by (cos_inner - cos_outer), so the inner cone must never
    // open wider than the outer one.
    #[test]
    fn spot_inner_cosine_never_falls_below_the_outer() {
        for (inner, outer) in [(10.0, 20.0), (45.0, 20.0), (0.0, 0.0), (-5.0, 30.0)] {
            let s = spot([0.0, -1.0, 0.0], inner, outer);
            assert!(
                s.cos_inner() >= s.cos_outer() - 1e-6,
                "inner {inner} outer {outer}"
            );
        }
    }

    #[test]
    fn spot_cosines_match_the_authored_angles() {
        let s = spot([0.0, -1.0, 0.0], 15.0, 30.0);
        assert!((s.cos_inner() - 15.0f32.to_radians().cos()).abs() < 1e-6);
        assert!((s.cos_outer() - 30.0f32.to_radians().cos()).abs() < 1e-6);
    }

    // A hemisphere-wide cone would make the clustered sphere bound useless.
    #[test]
    fn spot_outer_angle_capped() {
        let s = spot([0.0, -1.0, 0.0], 0.0, 180.0);
        assert!((s.cos_outer() - SPOT_MAX_ANGLE_DEG.to_radians().cos()).abs() < 1e-6);
    }
}
