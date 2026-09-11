// src/components/mesh_renderer.rs

use crate::ecs::{MaterialHandle, MeshHandle};

/// Single-mesh render description for an entity: which mesh and material to
/// draw, plus an optional view-distance cutoff.
///
/// Runtime-only. Mutually exclusive with `ModelRenderer` on an entity (an
/// entity has one or the other), which encodes the mesh-vs-model choice a
/// `Prop` expresses with its `model` field taking precedence.
#[derive(Debug, Clone, Default)]
pub struct MeshRenderer {
    /// A `Mesh` or `ProceduralMesh` to render, addressed by its shared
    /// mesh-source handle.
    pub mesh: Option<MeshHandle>,
    /// A `Material` providing albedo plus lighting parameters, addressed by its
    /// `MaterialHandle`.
    pub material: Option<MaterialHandle>,
    /// View-distance cutoff in world units; 0 keeps the draw visible at any
    /// distance.
    pub cull_distance: f32,
}
