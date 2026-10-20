// A planet tile's vertices morphed toward its parent's mesh, single-sourced
// for every pass that draws or roots on a tile: mirrors planet::morph. A
// splicing pass declares `mesh_float(i)`, float `i` of the shared vertex
// stream, ahead of the marker.
//
// A child tile's even grid corners are its parent's corners. Each odd corner
// moves toward the middle of the two that bound it on the parent's grid,
// across the parent cell's diagonal where both coordinates are odd, so a
// fully morphed tile is its parent's mesh.

// Locked to TILE_CELLS in planet::tile and MORPH_START in planet::lod.
static const uint PLANET_TILE_CELLS = 16u;
static const float TILE_MORPH_START = 0.85;
// Locked to SEAM_RAMP_CELLS in planet::morph.
static const float TILE_SEAM_RAMP_CELLS = 4.0;

// Floats in one vertex of the shared stream and where each attribute the
// morph moves starts. Matches `Vertex` in gfx::mesh_payload.
static const uint TILE_VERTEX_FLOATS = 14u;
static const uint TILE_VERTEX_POS = 0u;
static const uint TILE_VERTEX_NORMAL = 3u;
static const uint TILE_VERTEX_TANGENT = 6u;

struct TileVertex
{
    float3 position;
    float3 normal;
    float3 tangent;
};

float3 tile_vertex_float3(uint vertex, uint attribute)
{
    uint f = vertex * TILE_VERTEX_FLOATS + attribute;
    return float3(mesh_float(f), mesh_float(f + 1u), mesh_float(f + 2u));
}

TileVertex tile_vertex(uint vertex)
{
    TileVertex v;
    v.position = tile_vertex_float3(vertex, TILE_VERTEX_POS);
    v.normal = tile_vertex_float3(vertex, TILE_VERTEX_NORMAL);
#ifdef TILE_MORPH_TANGENT
    v.tangent = tile_vertex_float3(vertex, TILE_VERTEX_TANGENT);
#else
    v.tangent = (float3)0.0;
#endif
    return v;
}

// The grid corner (i, j) of a tile's vertex `local`, row-major along t.
uint2 tile_grid_corner(uint local)
{
    uint side = PLANET_TILE_CELLS + 1u;
    return uint2(local % side, local / side);
}

// The seam side bit corner (i, j) lies on, 0 inside the tile and for even
// corners. Bits match planet::TileSide: -s, +s, -t, +t.
uint tile_seam_side(uint2 c)
{
    uint n = PLANET_TILE_CELLS;
    if (c.x == 0u && (c.y & 1u) != 0u) return 1u;
    if (c.x == n && (c.y & 1u) != 0u) return 2u;
    if (c.y == 0u && (c.x & 1u) != 0u) return 4u;
    if (c.y == n && (c.x & 1u) != 0u) return 8u;
    return 0u;
}

// How far corner `c` of tile `m` has morphed, `distance` meters from the
// eye: by distance over the last stretch of the range, all the way on a
// side beside a coarser tile, not at all on one beside finer tiles. Inside,
// each pin fades out over TILE_SEAM_RAMP_CELLS from its side; a corner on an
// edge takes no fade, so same-level neighbors morph it alike.
// Mirrors planet::morph_factor.
float tile_morph_factor(GpuTileMorph m, uint2 c, float distance)
{
    float range = m.eye_range.w;
    float k = saturate((distance - TILE_MORPH_START * range) / ((1.0 - TILE_MORPH_START) * range));
    uint seams = asuint(m.prev_eye_seams.w);
    uint side = tile_seam_side(c);
    if ((seams & side) != 0u) return 1.0;
    if (((seams >> 4u) & side) != 0u) return 0.0;
    uint n = PLANET_TILE_CELLS;
    if (c.x == 0u || c.y == 0u || c.x == n || c.y == n) return k;
    // Cells from the -s, +s, -t and +t sides, in ramps.
    float4 away = float4(float(c.x), float(n - c.x), float(c.y), float(n - c.y))
                  / TILE_SEAM_RAMP_CELLS;
    [unroll] for (uint s = 0u; s < 4u; s++)
    {
        if ((seams & (1u << s)) != 0u) k = max(k, 1.0 - away[s]);
    }
    [unroll] for (uint f = 0u; f < 4u; f++)
    {
        if (((seams >> 4u) & (1u << f)) != 0u) k = min(k, away[f]);
    }
    return saturate(k);
}

// The two grid corners `a` and `b` whose middle odd corner `c` morphs toward.
void tile_parent_span(uint2 c, out uint2 a, out uint2 b)
{
    uint odd = (c.x & 1u) | ((c.y & 1u) << 1u);
    int2 da = odd == 1u ? int2(-1, 0) : odd == 2u ? int2(0, -1) : int2(1, -1);
    a = uint2(int2(c) + da);
    b = uint2(int2(c) - da);
}

// Where odd grid corner `c` of the tile whose first vertex is `base` lies on
// its parent's mesh.
float3 tile_parent_position(uint base, uint2 c)
{
    uint side = PLANET_TILE_CELLS + 1u;
    uint2 a;
    uint2 b;
    tile_parent_span(c, a, b);
    return (tile_vertex_float3(base + a.y * side + a.x, TILE_VERTEX_POS)
            + tile_vertex_float3(base + b.y * side + b.x, TILE_VERTEX_POS)) * 0.5;
}

// `v`, vertex `vertex` of the shared stream on a tile whose first vertex is
// `base`, morphed `k` of the way toward its parent's mesh. Without
// TILE_MORPH_TANGENT the tangent is left as it is, for a reader that has no
// use for it.
TileVertex tile_morph_vertex(TileVertex v, uint base, uint vertex, float k)
{
    uint2 c = tile_grid_corner(vertex - base);
    uint odd = (c.x & 1u) | ((c.y & 1u) << 1u);
    if (odd == 0u || k <= 0.0)
    {
        return v;
    }
    uint side = PLANET_TILE_CELLS + 1u;
    uint2 a;
    uint2 b;
    tile_parent_span(c, a, b);
    TileVertex ta = tile_vertex(base + a.y * side + a.x);
    TileVertex tb = tile_vertex(base + b.y * side + b.x);
    TileVertex o;
    o.position = lerp(v.position, (ta.position + tb.position) * 0.5, k);
    o.normal = normalize(lerp(v.normal, normalize(ta.normal + tb.normal), k));
#ifdef TILE_MORPH_TANGENT
    o.tangent = normalize(lerp(v.tangent, normalize(ta.tangent + tb.tangent), k));
#else
    o.tangent = v.tangent;
#endif
    return o;
}

TileVertex tile_vertex_in(float3 position, float3 normal, float3 tangent)
{
    TileVertex v;
    v.position = position;
    v.normal = normal;
    v.tangent = tangent;
    return v;
}

// The vertex stage's vertex `vertex_id` (from SV_VertexID), `v`, of a draw
// with morph row `m` placed by `model`, morphed for a camera at `eye`; as it
// is for a draw that is not a planet tile. DirectX counts the id from the
// draw's first vertex, the other backends from the stream's.
TileVertex tile_morph_draw(GpuTileMorph m, float4x4 model, TileVertex v, uint vertex_id, float3 eye)
{
    if (m.eye_range.w <= 0.0)
    {
        return v;
    }
    uint base = m.base.x;
#ifdef CN_BACKEND_DIRECTX
    uint vertex = base + vertex_id;
#else
    uint vertex = vertex_id;
#endif
    float distance = length(mul(model, float4(v.position, 1.0)).xyz - eye);
    float k = tile_morph_factor(m, tile_grid_corner(vertex - base), distance);
    return tile_morph_vertex(v, base, vertex, k);
}
