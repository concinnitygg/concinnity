// Waves laid on a level shell: cascades of tiled slope maps read on the three
// cube faces around the corner nearest an anchor point, each face mapping
// the shell stereographically from the opposite face's center and blending
// with its neighbors across a band at their edges. Pure functions and the
// record a ray-traced pass reads them from; each cascade's slope texels hold
// `(dh/du, dh/dv, (dh/du)^2, (dh/dv)^2)` so a filtered tap also carries the
// slope variance its footprint averaged away.

#ifndef LEVEL_WAVES
#define LEVEL_WAVES

// The cube faces a shell's waves are read on. Mirrors `SEA_FACES`.
static const uint LEVEL_WAVE_FACES = 3u;
// The most cascades the waves carry. Mirrors the host's cascade limit
// (`MAX_OCEAN_CASCADES`).
static const uint LEVEL_WAVE_CASCADES = 4u;

// A point's place on one cube face: its face meters from the anchor's place
// there, how many face meters a meter of the shell spans, and the directions
// along the shell, in the frame the rows turn from, the face's u and v grow
// along. Mirrors `sea_point` in render::water::sea.
struct WaveFacePoint
{
    float2 s;
    float stretch;
    float3 tu;
    float3 tv;
};

// The anchor's direction from the shell's center in a face's axes: the `w`
// of its rows.
float3 wave_face_anchor(float4 rows[3])
{
    return float3(rows[0].w, rows[1].w, rows[2].w);
}

// An offset `v` in the frame the rows turn from, in the face's axes.
float3 wave_face_offset(float4 rows[3], float3 v)
{
    return float3(dot(rows[0].xyz, v), dot(rows[1].xyz, v), dot(rows[2].xyz, v));
}

// A direction in the face's axes, back in the frame the rows turn from.
float3 wave_face_from(float4 rows[3], float3 v)
{
    return v.x * rows[0].xyz + v.y * rows[1].xyz + v.z * rows[2].xyz;
}

// The face map is the stereographic projection from the opposite face's
// center, `2 radius (u, v) / (1 + normal)`, differenced against the anchor's
// in a form that stays exact for points beside it. `rest` is the point's
// offset from the anchor; `inv_radius` one over the shell's radius.
WaveFacePoint wave_face_point(float4 rows[3], float inv_radius, float3 rest)
{
    float3 a = wave_face_anchor(rows);
    float3 dl = wave_face_offset(rows, rest);
    float3 d = a + dl * inv_radius;
    float denom_a = 1.0 + a.y;
    float denom_p = 1.0 + d.y;
    float k = 2.0 / (denom_p * denom_a);
    WaveFacePoint p;
    p.s = k * (dl.xz * denom_a - a.xz * dl.y);
    p.stretch = 2.0 / denom_p;
    float3 n = normalize(d);
    float3 gu = float3(denom_p, -d.x, 0.0);
    float3 gv = float3(0.0, -d.z, denom_p);
    p.tu = wave_face_from(rows, normalize(gu - dot(gu, n) * n));
    p.tv = wave_face_from(rows, normalize(gv - dot(gv, n) * n));
    return p;
}

// A face's unnormalized share of the point `rest` from the anchor: whole well
// inside the face, falling across a band at its edge at rate `blend`.
float wave_face_share(float4 rows[3], float inv_radius, float blend, float3 rest)
{
    float3 d = wave_face_anchor(rows) + wave_face_offset(rows, rest) * inv_radius;
    float margin = d.y - max(abs(d.x), abs(d.z));
    return saturate(0.5 + margin * blend);
}

// Three faces' shares normalized so their squares sum to one, so two blended
// faces keep the roughness of one.
float3 wave_face_shares(float3 shares)
{
    return shares * rsqrt(max(dot(shares, shares), 1e-12));
}

// Cascade `c`'s tile offset among `offsets`, two `[u, v]` pairs to a lane.
float2 wave_cascade_offset(float4 offsets[2], uint c)
{
    float4 pair = offsets[c >> 1u];
    return (c & 1u) == 0u ? pair.xy : pair.zw;
}

// The texture coordinate of face meters `s` in a cascade's tile of
// `inv_length` tiles per meter whose anchor sits at `offset`: shifted half a
// texel of a `texels`-wide tile so texel `i` sits at `i` texels from the
// tile's origin on every mip.
float2 wave_cascade_uv(float2 s, float inv_length, float2 offset, float texels)
{
    return s * inv_length + offset + 0.5 / texels;
}

// The waves on a level shell as a reflection ray meets them. Layout matches
// `LevelWavesGpu` in render_types.
struct LevelWaves
{
    // xyz: the point the faces are laid out from; w: 1 when in use.
    float4 anchor;
    // x: one over the shell's radius; y: the faces' blend rate; z: how far
    // from the anchor along the shell the tiles are read, in meters; w: the
    // slope variance no tile carries (the summed waves' and the finest).
    float4 shape;
    // x: cascades; y: the first one's layer; z: texels per tile side.
    uint4 layers;
    // One over each cascade's tile side, per meter.
    float4 inv_lengths;
    // Per face, the rows turning the frame into its axes (`w`: the anchor's
    // direction from the shell's center along that axis).
    float4 face_rows[LEVEL_WAVE_FACES * 3u];
    // Per face, the anchor's place in each cascade's tile, `[u, v]` pairs.
    float4 face_offsets[LEVEL_WAVE_FACES * 2u];
};

#endif
