// Voxel-chunk block-palette entry schema.

use crate::AssetId;

/// Describes one entry in a [VoxelChunk](#voxelchunk) palette.
///
/// Each BlockType represents either a solid block (with UVs into the chunk's atlas texture)
/// or an empty/air marker.
///
/// Per-face fields fall back to `uv_min`/`uv_max` when omitted. Set `solid=false`
/// on the air/empty palette entry; faces between solid blocks and air blocks are
/// the only faces the chunk emits.
///
/// ```rust
/// # use concinnity_asset::BlockType;
/// BlockType {
///     solid: false,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct BlockType {
    /// Asset identity; injected via `inject_name`. Not part of `args`. Lets the
    /// runtime resolve a `VoxelWorld` palette (a list of `BlockType` ids) back
    /// to the block data the chunk generator needs.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// When false the block is treated as air -- no faces are emitted for it
    /// and it does not occlude neighboring faces.
    pub solid: bool,
    /// Default atlas UV at the (0,0) corner of each face.
    pub uv_min: [f32; 2],
    /// Default atlas UV at the (1,1) corner of each face.
    pub uv_max: [f32; 2],
    /// Optional per-face override for the +Y face: `[u_min, v_min, u_max, v_max]`.
    pub uv_top: Option<[f32; 4]>,
    /// Optional per-face override for the -Y face.
    pub uv_bottom: Option<[f32; 4]>,
    /// Optional per-face override applied to all four side faces (±X, ±Z).
    pub uv_side: Option<[f32; 4]>,
}

impl Default for BlockType {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            solid: true,
            uv_min: [0.0, 0.0],
            uv_max: [1.0, 1.0],
            uv_top: None,
            uv_bottom: None,
            uv_side: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_block_is_solid_and_uses_the_whole_atlas_tile() {
        let b = BlockType::default();
        assert!(b.solid);
        assert_eq!(b.uv_min, [0.0, 0.0]);
        assert_eq!(b.uv_max, [1.0, 1.0]);
        // No per-face override means every face samples the same tile.
        assert_eq!(b.uv_top, None);
        assert_eq!(b.uv_bottom, None);
        assert_eq!(b.uv_side, None);
    }

    #[test]
    fn per_face_tiles_parse_and_round_trip_through_postcard() {
        let b: BlockType = serde_json::from_str(
            r#"{"solid":false,"uv_min":[0.25,0],"uv_max":[0.5,0.25],
                "uv_top":[0,0,0.25,0.25],"uv_side":[0.5,0,0.75,0.25]}"#,
        )
        .unwrap();
        assert!(!b.solid);
        assert_eq!(b.uv_min, [0.25, 0.0]);
        assert_eq!(b.uv_top, Some([0.0, 0.0, 0.25, 0.25]));
        assert_eq!(b.uv_side, Some([0.5, 0.0, 0.75, 0.25]));
        // An unmentioned face keeps falling back to the base tile.
        assert_eq!(b.uv_bottom, None);

        let bytes = postcard::to_allocvec(&b).unwrap();
        let back: BlockType = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.uv_top, Some([0.0, 0.0, 0.25, 0.25]));
        assert_eq!(back.uv_bottom, None);
        assert!(!back.solid);
        assert_eq!(back.asset_id, AssetId::default());
    }
}
