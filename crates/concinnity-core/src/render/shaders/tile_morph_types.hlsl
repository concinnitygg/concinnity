// The morph row a planet tile's draw carries, declared once for the object
// record and the grass kernel's tile record alike. Guarded because a pass can
// reach it through both.
#ifndef CN_TILE_MORPH_TYPES
#define CN_TILE_MORPH_TYPES

// Mirrors `GpuTileMorph` in concinnity-core/src/gfx/render_types.rs (48 B).
struct GpuTileMorph
{
    // xyz = the camera, w = how far the tile's
    // level reaches from it (0 for a draw that is not a planet tile).
    float4 eye_range;
    // xyz = the camera last frame, w = the seam sides
    // (asuint): beside a coarser tile in the low four bits, finer above.
    float4 prev_eye_seams;
    // x = the tile's first vertex in the shared vertex stream.
    uint4 base;
};

#endif
