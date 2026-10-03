// Voxel-chunk block-palette entry schema.

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
/// # use concinnity_core::components::BlockType;
/// BlockType {
///     solid: false,
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct BlockType {
    /// When false the block is treated as air -- no faces are emitted for it
    /// and it does not occlude neighboring faces.
    #[asset(default = true)]
    pub solid: bool,
    /// Default atlas UV at the (0,0) corner of each face.
    pub uv_min: [f32; 2],
    /// Default atlas UV at the (1,1) corner of each face.
    #[asset(default = [1.0, 1.0])]
    pub uv_max: [f32; 2],
    /// Optional per-face override for the +Y face: `[u_min, v_min, u_max, v_max]`.
    pub uv_top: Option<[f32; 4]>,
    /// Optional per-face override for the -Y face.
    pub uv_bottom: Option<[f32; 4]>,
    /// Optional per-face override applied to all four side faces (±X, ±Z).
    pub uv_side: Option<[f32; 4]>,
}
