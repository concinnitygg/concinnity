// src/components/global_transform.rs

/// Composed world matrix for an entity, after parent transforms are applied.
///
/// Runtime-only. A transform-propagation pass writes it from an entity's
/// `Transform` and its parent chain; the renderer reads it to place draws. For
/// a root (parentless) entity it equals `Transform::model_matrix`.
#[derive(Debug, Clone, Copy)]
pub struct GlobalTransform(pub [[f32; 4]; 4]);

impl Default for GlobalTransform {
    fn default() -> Self {
        GlobalTransform([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }
}
