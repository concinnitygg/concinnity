// src/assets/model_renderer.rs

use crate::ecs::asset_id::AssetId;

/// Multi-mesh render description for an entity: a `Model` whose sub-meshes all
/// share this entity's transform, each with its own material.
///
/// Runtime-only. Mutually exclusive with `MeshRenderer` on an entity.
#[derive(Debug, Clone)]
pub struct ModelRenderer {
    /// The `Model` to render.
    pub model: AssetId,
    /// View-distance cutoff in world units; 0 keeps the draw visible at any
    /// distance.
    pub cull_distance: f32,
}

impl Default for ModelRenderer {
    fn default() -> Self {
        Self {
            model: AssetId::default(),
            cull_distance: 0.0,
        }
    }
}
