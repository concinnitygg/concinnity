// Raw mesh geometry schema.

use crate::ecs::PayloadLocator;
use crate::ecs::asset_id::AssetId;
use alloc::string::String;
use alloc::vec::Vec;

/// A single vertex as supplied in raw Mesh args.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VertexData {
    /// Vertex position `[x, y, z]` in model space.
    pub pos: [f32; 3],
    /// Vertex colour `[r, g, b]` in [0, 1]. Use `[0.75, 0.74, 0.72]` for a
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Mesh {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
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
    #[serde(default = "default_lod_levels")]
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

fn default_lod_levels() -> u32 {
    1
}

impl Default for Mesh {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            source: String::new(),
            primitive_index: 0,
            chunk_index: None,
            vertices: Vec::new(),
            indices: Vec::new(),
            lod_levels: 1,
            lod_distances: Vec::new(),
            locator: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_vertex_without_uvs_samples_the_texture_origin() {
        let v: VertexData = serde_json::from_str(r#"{"pos":[1,2,3],"color":[1,0,0]}"#).unwrap();
        assert_eq!(v.pos, [1.0, 2.0, 3.0]);
        assert_eq!(v.color, [1.0, 0.0, 0.0]);
        assert_eq!(v.uv, [0.0, 0.0]);
        // Position and colour are required: geometry with neither is a mistake,
        // not a default.
        assert!(serde_json::from_str::<VertexData>(r#"{"pos":[0,0,0]}"#).is_err());
    }

    #[test]
    fn a_blank_mesh_has_one_lod_and_no_geometry() {
        let m = Mesh::default();
        assert!(m.vertices.is_empty());
        assert!(m.indices.is_empty());
        assert!(m.lod_distances.is_empty());
        // One level means the mesh is drawn as authored, with no simplification.
        assert_eq!(m.lod_levels, 1);
        assert_eq!(m.primitive_index, 0);
        assert_eq!(m.chunk_index, None);
        assert!(m.locator.is_none());
    }

    #[test]
    fn an_omitted_lod_level_count_still_means_one() {
        // The field carries its own default fn, so an absent value is 1 rather
        // than the 0 an integer field would otherwise fall back to.
        let m: Mesh = serde_json::from_str(r#"{"source":"board.obj"}"#).unwrap();
        assert_eq!(m.lod_levels, 1);
        assert_eq!(m.source, "board.obj");
    }

    #[test]
    fn an_inline_mesh_round_trips_through_postcard() {
        let m: Mesh = serde_json::from_str(
            r#"{"source":"tile.obj","primitive_index":2,"chunk_index":7,"lod_levels":3,
                "lod_distances":[10,40],
                "vertices":[{"pos":[0,0,0],"color":[1,1,1],"uv":[0.5,0.5]}],
                "indices":[0,0,0]}"#,
        )
        .unwrap();
        assert_eq!(m.chunk_index, Some(7));

        let bytes = postcard::to_allocvec(&m).unwrap();
        let back: Mesh = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.primitive_index, 2);
        assert_eq!(back.chunk_index, Some(7));
        assert_eq!(back.lod_levels, 3);
        assert_eq!(back.lod_distances, [10.0, 40.0]);
        assert_eq!(back.vertices[0].uv, [0.5, 0.5]);
        assert_eq!(back.indices, [0, 0, 0]);
        assert_eq!(back.asset_id, AssetId::default());
    }

    #[test]
    fn an_absent_chunk_index_is_omitted_from_the_serialized_args() {
        let m: Mesh = serde_json::from_str(r#"{"source":"board.obj"}"#).unwrap();
        let json = serde_json::to_string(&m).unwrap();
        assert!(!json.contains("chunk_index"), "{json}");
    }
}
