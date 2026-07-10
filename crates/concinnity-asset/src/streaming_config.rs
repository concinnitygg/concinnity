// Asset-streaming configuration schema.

/// Enables and tunes asset streaming.
///
/// When no `StreamingConfig` is declared, streaming is off and every texture and
/// mesh is loaded up front. When one is present, textures and static mesh
/// geometry load in gradually after startup: each frame the nearest not-yet-
/// loaded items are brought in, up to a per-frame budget, prioritised by camera
/// distance. Once more than the cap would be loaded at once, the farthest are
/// dropped to make room.
///
/// Texture streaming covers the colour and normal-map textures (each capped
/// independently via `texture_budget` / `texture_cap`). Mesh streaming covers
/// static geometry; the skybox, rooms, and moving props always stay loaded.
///
/// ```jsonl
/// {"name":"streaming","type":"StreamingConfig","args":{}}
/// {"name":"streaming_slow","type":"StreamingConfig","args":{"texture_budget":1}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct StreamingConfig {
    /// Maximum number of textures whose load is started per frame, applied
    /// independently to the colour and normal-map pools. A low value spreads the
    /// cost over more frames.
    pub texture_budget: u32,
    /// Maximum number of textures kept loaded at once, applied independently to
    /// the colour and normal-map pools. When exceeded, the farthest-from-camera
    /// textures are dropped.
    pub texture_cap: u32,
    /// Maximum number of mesh regions whose load is started per frame. A low
    /// value spreads the cost over more frames.
    pub mesh_budget: u32,
    /// Maximum number of meshes kept loaded at once. When exceeded, the
    /// farthest-from-camera meshes are dropped.
    pub mesh_cap: u32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            texture_budget: 4,
            texture_cap: 96,
            mesh_budget: 4,
            mesh_cap: 4096,
        }
    }
}

// These accessors feed the Metal asset-streaming path for now
// (Vulkan / DirectX catch-up is a follow-up).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
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
