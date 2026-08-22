// src/assets/geometry.rs
//
// Model matrices and normals computed from asset data. These live here rather
// than in the schema crate because concinnity-asset stays serde-only data:
// anything that computes over an authored struct belongs on this side of the
// line. Exposed as extension traits so call sites keep method syntax
// (`prop.model_matrix()`).

use crate::assets::{GlassPanel, InstancedProp, RectAreaLight, SpotLight};
use crate::math::{cos, sqrt};

/// Widest half-angle a spot cone may open to. Past this the cone degenerates
/// toward a hemisphere and the clustered sphere bound stops being useful.
pub const SPOT_MAX_ANGLE_DEG: f32 = 89.9;

// `v` scaled to unit length, or `fallback` when it is too short to have a
// direction. The one degenerate-direction policy behind every authored
// normal / direction field below.
fn normalize_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if len < 1e-6 {
        fallback
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

/// Per-instance model matrices for an [InstancedProp].
pub trait InstancedPropGeometry {
    /// Column-major model matrix for the i-th instance, or `None` when the
    /// index is past the instance list.
    fn instance_model_matrix(&self, idx: usize) -> Option<[[f32; 4]; 4]>;
}

impl InstancedPropGeometry for InstancedProp {
    /// Build a column-major model matrix for the i-th instance.
    /// Order matches `Prop::model_matrix`: scale, then YXZ rotation, then translation.
    fn instance_model_matrix(&self, idx: usize) -> Option<[[f32; 4]; 4]> {
        let xform = self.instances.get(idx)?;
        Some(crate::gfx::transform::trs_matrix(
            xform.position,
            xform.rotation_deg,
            xform.scale,
        ))
    }
}

/// Cone direction and angular falloff cosines for a [SpotLight].
pub trait SpotLightGeometry {
    /// Unit-length cone axis.
    fn unit_direction(&self) -> [f32; 3];
    /// Cosine of the inner half-angle: the widest angle still at full
    /// brightness.
    fn cos_inner(&self) -> f32;
    /// Cosine of the outer half-angle: the angle at which the cone is black.
    fn cos_outer(&self) -> f32;
}

impl SpotLightGeometry for SpotLight {
    /// Unit-length cone axis, falling back to straight down when the authored
    /// `direction` is degenerate.
    fn unit_direction(&self) -> [f32; 3] {
        normalize_or(self.direction, [0.0, -1.0, 0.0])
    }

    /// Cosine of the inner half-angle: the widest angle still at full brightness.
    fn cos_inner(&self) -> f32 {
        cos(self.inner_angle.clamp(0.0, self.outer_angle).to_radians())
    }

    /// Cosine of the outer half-angle: the angle at which the cone reaches black.
    fn cos_outer(&self) -> f32 {
        cos(self.outer_angle.clamp(0.0, SPOT_MAX_ANGLE_DEG).to_radians())
    }
}

/// Unit-length facing normal for a [GlassPanel].
pub trait GlassPanelGeometry {
    /// Unit-length facing direction.
    fn unit_normal(&self) -> [f32; 3];
}

impl GlassPanelGeometry for GlassPanel {
    /// Unit-length facing direction, falling back to `+Z` when the authored
    /// `normal` is degenerate. The build-time quad generator and the runtime
    /// shader both rely on a usable normal.
    fn unit_normal(&self) -> [f32; 3] {
        normalize_or(self.normal, [0.0, 0.0, 1.0])
    }
}

/// Unit-length emission normal for a [RectAreaLight].
pub trait RectAreaLightGeometry {
    /// Unit-length emission direction.
    fn unit_normal(&self) -> [f32; 3];
}

impl RectAreaLightGeometry for RectAreaLight {
    /// Unit-length emission direction, falling back to straight down when the
    /// authored `normal` is degenerate (the panel default emits downward).
    fn unit_normal(&self) -> [f32; 3] {
        normalize_or(self.normal, [0.0, -1.0, 0.0])
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
    fn spot_direction_normalises() {
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
    fn rect_normal_normalises_and_falls_back_to_down() {
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
