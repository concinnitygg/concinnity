// The morph row a planet tile's draw carries and the camera it morphs by,
// declared once for the object record, the view blocks and the grass kernel's
// tile record alike. Guarded because a pass can
// reach it through both.
#ifndef CN_TILE_MORPH_TYPES
#define CN_TILE_MORPH_TYPES

// Mirrors `GpuTileMorph` in concinnity-engine/src/gfx/render_types.rs (16 B):
// what the choice of tiles sets, so a moving camera leaves the record as it is.
struct GpuTileMorph
{
    // The tile's first vertex in the shared vertex stream.
    uint vertex_base;
    // The seam sides: beside a coarser tile in the low four bits, finer above.
    uint seams;
    // The tile's quadtree level.
    uint level;
    // 1 for a planet tile's draw, 0 for every other draw.
    uint planet_tile;
};

// Mirrors `TileMorphView` in concinnity-engine/src/gfx/render_types.rs (32 B):
// the camera every tile's morph is measured from, carried by the view blocks.
struct TileMorphView
{
    // xyz = the camera, w = how far level 0 reaches from it (0 morphs nothing).
    float4 eye_range0;
    // xyz = the camera last frame, w = the coarsest drawn level (asuint).
    float4 prev_eye_coarsest;
};

#endif
