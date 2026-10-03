// Asset-streaming configuration schema.

/// Enables and tunes asset streaming.
///
/// When no `StreamingConfig` is declared, streaming is off and every texture and
/// mesh is loaded up front. When one is present, textures and static mesh
/// geometry load in gradually after startup: each frame the nearest not-yet-
/// loaded items are brought in, up to a per-frame budget, prioritized by camera
/// distance. Once more than the cap would be loaded at once, the farthest are
/// dropped to make room.
///
/// Texture streaming covers the color and normal-map textures (each capped
/// independently via `texture_budget` / `texture_cap`). Mesh streaming covers
/// static geometry; the skybox, rooms, and moving props always stay loaded.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct StreamingConfig {
    /// Maximum number of textures whose load is started per frame, applied
    /// independently to the color and normal-map pools. A low value spreads the
    /// cost over more frames.
    #[asset(default = 4)]
    pub texture_budget: u32,
    /// Maximum number of textures kept loaded at once, applied independently to
    /// the color and normal-map pools. When exceeded, the farthest-from-camera
    /// textures are dropped.
    #[asset(default = 96)]
    pub texture_cap: u32,
    /// Maximum number of mesh regions whose load is started per frame. A low
    /// value spreads the cost over more frames.
    #[asset(default = 4)]
    pub mesh_budget: u32,
    /// Maximum number of meshes kept loaded at once. When exceeded, the
    /// farthest-from-camera meshes are dropped.
    #[asset(default = 4096)]
    pub mesh_cap: u32,
    /// Resident-texture memory budget in mebibytes, spanning the color and
    /// normal-map pools together. Once resident textures exceed it the
    /// farthest-from-camera ones are dropped, so nearer textures always win the
    /// space. `0` (the default) derives the budget from the GPU's reported
    /// memory instead. `texture_cap` still applies as a hard item-count ceiling.
    pub texture_budget_mb: u32,
    /// Resident-mesh memory budget in mebibytes. Once resident meshes exceed it
    /// the farthest-from-camera ones are dropped. `0` (the default) derives the
    /// budget from the GPU's reported memory instead. `mesh_cap` still applies
    /// as a hard item-count ceiling.
    pub mesh_budget_mb: u32,
}

impl StreamingConfig {
    /// Per-frame texture load budget as a `usize`, floored at 1 so a stray 0
    /// cannot wedge streaming permanently.
    pub fn budget(&self) -> usize {
        (self.texture_budget as usize).max(1)
    }

    /// Resident-texture cap as a `usize`, floored at 1.
    pub fn cap(&self) -> usize {
        (self.texture_cap as usize).max(1)
    }

    /// Per-frame mesh load budget as a `usize`, floored at 1.
    pub fn mesh_budget(&self) -> usize {
        (self.mesh_budget as usize).max(1)
    }

    /// Resident-mesh cap as a `usize`, floored at 1.
    pub fn mesh_cap(&self) -> usize {
        (self.mesh_cap as usize).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zero_budget_or_cap_is_floored_at_one() {
        // A zero would stall streaming outright, so every accessor keeps at
        // least one slot rather than trusting the authored number.
        let c: StreamingConfig = serde_json::from_str(
            r#"{"texture_budget":0,"texture_cap":0,"mesh_budget":0,"mesh_cap":0}"#,
        )
        .unwrap();
        assert_eq!(c.budget(), 1);
        assert_eq!(c.cap(), 1);
        assert_eq!(c.mesh_budget(), 1);
        assert_eq!(c.mesh_cap(), 1);
    }
}
