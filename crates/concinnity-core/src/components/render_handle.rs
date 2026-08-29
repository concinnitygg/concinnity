// src/components/render_handle.rs

use crate::memory::InlineVec;

/// The backend draw-object slot(s) an entity occupies.
///
/// Runtime-only. The renderer writes one of these per renderable entity so
/// per-frame model-matrix and visibility updates address the GPU slots by
/// entity rather than by storage row. A mesh-backed entity has one slot; a
/// model-backed entity has one per sub-mesh.
#[derive(Debug, Clone, Default)]
pub struct RenderHandle {
    /// Backend draw-object indices owned by this entity.
    ///
    /// Held inline for the single-slot case, which is most of them.
    pub draws: InlineVec<u32>,
}
