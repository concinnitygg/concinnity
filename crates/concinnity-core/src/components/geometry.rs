//! Model matrices and normals computed from asset data. These live here rather
//! than with the schema types because those stay serde-only data:
//! anything that computes over an authored struct belongs on this side of the
//! line.

use crate::components::{GlassPanel, InstancedProp, RectAreaLight, SpotLight};
use crate::math::cos;
use crate::math::vec3::normalize_or;

/// Widest half-angle a spot cone may open to. Past this the cone degenerates
/// toward a hemisphere and the clustered sphere bound stops being useful.
pub const SPOT_MAX_ANGLE_DEG: f32 = 89.9;

// The length below which an authored normal / direction field has no
// direction and takes its fallback.
const MIN_DIRECTION_LEN: f32 = 1e-6;

impl InstancedProp {
    /// Column-major model matrix for the i-th instance, or `None` when the
    /// index is past the instance list. Order matches `Prop::model_matrix`:
    /// scale, then YXZ rotation, then translation.
    pub fn instance_model_matrix(&self, idx: usize) -> Option<[[f32; 4]; 4]> {
        let xform = self.instances.get(idx)?;
        Some(crate::transform::trs_matrix(
            xform.position,
            xform.rotation_deg,
            xform.scale,
        ))
    }
}

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

impl GlassPanel {
    /// Unit-length facing direction, falling back to `+Z` when the authored
    /// `normal` is degenerate. The build-time quad generator and the runtime
    /// shader both rely on a usable normal.
    pub fn unit_normal(&self) -> [f32; 3] {
        normalize_or(self.normal, MIN_DIRECTION_LEN, [0.0, 0.0, 1.0])
    }
}

impl RectAreaLight {
    /// Unit-length emission direction, falling back to straight down when the
    /// authored `normal` is degenerate (the panel default emits downward).
    pub fn unit_normal(&self) -> [f32; 3] {
        normalize_or(self.normal, MIN_DIRECTION_LEN, [0.0, -1.0, 0.0])
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

    #[test]
    fn rect_normal_normalizes_and_falls_back_to_down() {
        let lit = RectAreaLight {
            normal: [0.0, 0.0, 3.0],
            ..RectAreaLight::default()
        };
        assert_eq!(lit.unit_normal(), [0.0, 0.0, 1.0]);
        let degenerate = RectAreaLight {
            normal: [0.0; 3],
            ..RectAreaLight::default()
        };
        assert_eq!(degenerate.unit_normal(), [0.0, -1.0, 0.0]);
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
