// src/assets/transform.rs

/// World-space placement of an entity: translation, rotation, and scale.
///
/// Runtime-only placement state. Physics and interaction systems mutate it and
/// the renderer reads it to position draws. Not authored directly in a world
/// file; it carries the same transform fields a `Prop` declares.
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    /// World-space position [x, y, z].
    pub position: [f32; 3],
    /// Euler rotation in degrees [pitch, yaw, roll], applied in YXZ order.
    pub rotation_deg: [f32; 3],
    /// Non-uniform scale [x, y, z].
    pub scale: [f32; 3],
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

impl Transform {
    /// Build a column-major model matrix from this transform.
    /// Order: scale, then YXZ Euler rotation, then translation.
    pub fn model_matrix(&self) -> [[f32; 4]; 4] {
        crate::gfx::skinning::trs_matrix(self.position, self.rotation_deg, self.scale)
    }
}
