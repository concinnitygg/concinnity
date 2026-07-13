// src/assets/geometry.rs
//
// Model matrices and normals computed from asset data. These live in core, not
// the schema crate: they need f32 transcendentals (sin/cos/sqrt), which are std,
// not core, and concinnity-asset stays serde-only data. Exposed as extension
// traits so call sites keep method syntax (`prop.model_matrix()`).

use crate::assets::{GlassPanel, InstancedProp, Prop};

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
