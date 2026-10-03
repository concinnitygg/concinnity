// Raw mesh geometry schema.

use crate::ecs::PayloadLocator;
use alloc::string::String;
use alloc::vec::Vec;

/// A single vertex as supplied in raw Mesh args.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
pub struct VertexData {
    /// Vertex position `[x, y, z]` in model space.
    pub pos: [f32; 3],
    /// Vertex color `[r, g, b]` in [0, 1]. Use `[0.75, 0.74, 0.72]` for a
    /// neutral surface that takes the material albedo.
    pub color: [f32; 3],
    /// Texture coordinates in [0, 1] space.  Defaults to [0, 0] when omitted.
    #[serde(default)]
    pub uv: [f32; 2],
}

/// Raw geometry. Supply `vertices` and `indices` directly, or import them from
/// a binary glTF file with `source` + `primitive_index`.
///
/// Use when you want full control over shape: custom furniture,
/// architectural details, signage, or any form a generator cannot
/// produce. For standard shapes use [ProceduralMesh](#proceduralmesh).
///
/// Normals and tangents are computed automatically at build time.
/// **Do not supply normals or tangents.**
///
/// **Vertex color:** use `[0.75, 0.74, 0.72]` for a neutral surface that takes
/// the material albedo, or `[1, 1, 1]` to pass through unmodified.
///
/// **Winding:** triangles must be counter-clockwise when viewed from the front.
/// Reversed winding = invisible face.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct Mesh {
    /// Optional path to a `.glb` file. When set, the build imports
    /// `vertices` / `indices` from it; inline geometry leaves this empty.
    pub source: String,
    /// Which primitive (counted across all meshes in the file) to import from
    /// `source`. Ignored when `source` is empty.
    pub primitive_index: u32,
    /// Pick a single chunk of an oversized imported primitive. `None` (the
    /// default) imports the whole primitive, which is fine whenever its vertex
    /// count fits in 16-bit indices; larger primitives are split into chunks on
    /// import, one Mesh per chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_index: Option<u32>,
    /// Vertex list.  Each vertex: `{"pos":[x,y,z], "color":[r,g,b], "uv":[u,v]}`.
    pub vertices: Vec<VertexData>,
    /// Triangle index list (16-bit values).
    pub indices: Vec<u16>,
    /// Number of level-of-detail versions to generate, including the original.
    /// `1` (the default) generates none; values are clamped to `[1, 8]`.
    #[asset(default = 1)]
    pub lod_levels: u32,
    /// Camera distances at which to switch to each lower-detail version. Length
    /// should be `lod_levels - 1`; empty lets the build derive a default
    /// sequence. The version for index `i` is used at camera distance ≥
    /// `lod_distances[i]`.
    pub lod_distances: Vec<f32>,
    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Geometry missing its position or color is a mistake, not a default.
    #[test]
    fn a_vertex_without_a_color_is_rejected() {
        assert!(serde_json::from_str::<VertexData>(r#"{"pos":[0,0,0]}"#).is_err());
    }

    #[test]
    fn an_absent_chunk_index_is_omitted_from_the_serialized_args() {
        let json = serde_json::to_string(&Mesh::default()).unwrap();
        assert!(!json.contains("chunk_index"), "{json}");
    }
}
