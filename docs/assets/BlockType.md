<!-- Auto-generated - do not edit. -->

# BlockType

Describes one entry in a [VoxelChunk](VoxelChunk.md) palette.

Each BlockType represents either a solid block (with UVs into the chunk's atlas texture)
or an empty/air marker.

Per-face fields fall back to `uv_min`/`uv_max` when omitted. Set `solid=false`
on the air/empty palette entry; faces between solid blocks and air blocks are
the only faces the chunk emits.

## Parameters

- `solid`: A boolean. When false the block is treated as air -- no faces are emitted for it and it does not occlude neighboring faces. Defaults to `true`.
- `uv_min`: An array of 2 floats. Default atlas UV at the (0,0) corner of each face.
- `uv_max`: An array of 2 floats. Default atlas UV at the (1,1) corner of each face.
- `uv_top`: An array of 4 floats. Optional per-face override for the +Y face: `[u_min, v_min, u_max, v_max]`.
- `uv_bottom`: An array of 4 floats. Optional per-face override for the -Y face.
- `uv_side`: An array of 4 floats. Optional per-face override applied to all four side faces (±X, ±Z).
