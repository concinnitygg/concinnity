// src/assets/geometry.rs
//
// Model matrices and normals computed from asset data. These live in core, not
// the schema crate: they need f32 transcendentals (sin/cos/sqrt), which are std,
// not core, and concinnity-asset stays serde-only data. Exposed as extension
// traits so call sites keep method syntax (`prop.model_matrix()`).

use crate::assets::{GlassPanel, InstancedProp, Prop, RectAreaLight, SpotLight};

// Widest half-angle a spot cone may open to. Past this the cone degenerates
// toward a hemisphere and the clustered sphere bound stops being useful.
pub const SPOT_MAX_ANGLE_DEG: f32 = 89.9;

/// Column-major model matrix for a [Prop].
pub trait PropGeometry {
    fn model_matrix(&self) -> [[f32; 4]; 4];
}

impl PropGeometry for Prop {
    fn model_matrix(&self) -> [[f32; 4]; 4] {
        let [px, py, pz] = self.position;
        let [pitch_deg, yaw_deg, roll_deg] = self.rotation_deg;
        let [sx, sy, sz] = self.scale;

        let (pr, yr, rr) = (
            pitch_deg.to_radians(),
            yaw_deg.to_radians(),
            roll_deg.to_radians(),
        );
        let (sp, cp) = (pr.sin(), pr.cos());
        let (sy_, cy) = (yr.sin(), yr.cos());
        let (sr, cr) = (rr.sin(), rr.cos());

        // YXZ rotation: R = Ry * Rx * Rz
        // Combined and scaled, column-major storage: out[col][row].
        [
            [
                sx * (cy * cr + sy_ * sp * sr),
                sx * (cp * sr),
                sx * (-sy_ * cr + cy * sp * sr),
                0.0,
            ],
            [
                sy * (-cy * sr + sy_ * sp * cr),
                sy * (cp * cr),
                sy * (sy_ * sr + cy * sp * cr),
                0.0,
            ],
            [sz * (sy_ * cp), sz * (-sp), sz * (cy * cp), 0.0],
            [px, py, pz, 1.0],
        ]
    }
}

/// Per-instance model matrices for an [InstancedProp].
pub trait InstancedPropGeometry {
    fn instance_model_matrix(&self, idx: usize) -> Option<[[f32; 4]; 4]>;
}

impl InstancedPropGeometry for InstancedProp {
    /// Build a column-major model matrix for the i-th instance.
    /// Order matches `Prop::model_matrix`: scale, then YXZ rotation, then translation.
    fn instance_model_matrix(&self, idx: usize) -> Option<[[f32; 4]; 4]> {
        let xform = self.instances.get(idx)?;
        let [px, py, pz] = xform.position;
        let [pitch_deg, yaw_deg, roll_deg] = xform.rotation_deg;
        let [sx, sy, sz] = xform.scale;

        let (pr, yr, rr) = (
            pitch_deg.to_radians(),
            yaw_deg.to_radians(),
            roll_deg.to_radians(),
        );
        let (sp, cp) = (pr.sin(), pr.cos());
        let (sy_, cy) = (yr.sin(), yr.cos());
        let (sr, cr) = (rr.sin(), rr.cos());

        Some([
            [
                sx * (cy * cr + sy_ * sp * sr),
                sx * (cp * sr),
                sx * (-sy_ * cr + cy * sp * sr),
                0.0,
            ],
            [
                sy * (-cy * sr + sy_ * sp * cr),
                sy * (cp * cr),
                sy * (sy_ * sr + cy * sp * cr),
                0.0,
            ],
            [sz * (sy_ * cp), sz * (-sp), sz * (cy * cp), 0.0],
            [px, py, pz, 1.0],
        ])
    }
}

/// Cone direction and angular falloff cosines for a [SpotLight].
pub trait SpotLightGeometry {
    fn unit_direction(&self) -> [f32; 3];
    fn cos_inner(&self) -> f32;
    fn cos_outer(&self) -> f32;
}

impl SpotLightGeometry for SpotLight {
    /// Unit-length cone axis, falling back to straight down when the authored
    /// `direction` is degenerate.
    fn unit_direction(&self) -> [f32; 3] {
        let d = self.direction;
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if len < 1e-6 {
            [0.0, -1.0, 0.0]
        } else {
            [d[0] / len, d[1] / len, d[2] / len]
        }
    }

    /// Cosine of the inner half-angle: the widest angle still at full brightness.
    fn cos_inner(&self) -> f32 {
        self.inner_angle
            .clamp(0.0, self.outer_angle)
            .to_radians()
            .cos()
    }

    /// Cosine of the outer half-angle: the angle at which the cone reaches black.
    fn cos_outer(&self) -> f32 {
        self.outer_angle
            .clamp(0.0, SPOT_MAX_ANGLE_DEG)
            .to_radians()
            .cos()
    }
}

/// Unit-length facing normal for a [GlassPanel].
pub trait GlassPanelGeometry {
    fn unit_normal(&self) -> [f32; 3];
}

impl GlassPanelGeometry for GlassPanel {
    /// Unit-length facing direction, falling back to `+Z` when the authored
    /// `normal` is degenerate. The build-time quad generator and the runtime
    /// shader both rely on a usable normal.
    fn unit_normal(&self) -> [f32; 3] {
        let n = self.normal;
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len < 1e-6 {
            [0.0, 0.0, 1.0]
        } else {
            [n[0] / len, n[1] / len, n[2] / len]
        }
    }
}

/// Unit-length emission normal for a [RectAreaLight].
pub trait RectAreaLightGeometry {
    fn unit_normal(&self) -> [f32; 3];
}

impl RectAreaLightGeometry for RectAreaLight {
    /// Unit-length emission direction, falling back to straight down when the
    /// authored `normal` is degenerate (the panel default emits downward).
    fn unit_normal(&self) -> [f32; 3] {
        let n = self.normal;
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len < 1e-6 {
            [0.0, -1.0, 0.0]
        } else {
            [n[0] / len, n[1] / len, n[2] / len]
        }
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
