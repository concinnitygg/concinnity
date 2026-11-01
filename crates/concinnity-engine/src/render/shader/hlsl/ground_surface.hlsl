// A planet's ground as its layers look up close: the surface's own material
// and each cover's, their textures taken out of their repeat by hex tiling,
// met where their rules overlap by which stands higher, and a broad
// variation drifting over them all.

// Hex cells per texture repeat along each axis.
static const float GROUND_HEX_SCALE = 1.5;
// Lattice cells in one period of a planet's texture coordinates, which wrap
// every 1024 m (`UV_PERIOD_M` in planet::tile): even, so the lattice repeats
// with them along both axes.
static const float GROUND_HEX_PERIOD = 1536.0;
// How strongly a hex vertex's own sample wins near it: higher keeps the
// texture's features whole, lower blurs neighboring samples into each other.
static const float GROUND_HEX_SHARPNESS = 8.0;
// Meters of ground a pixel covers past which one sample holds: the
// texture's repeat is then a few pixels across and its mips are its mean.
static const float GROUND_HEX_FAR = 0.5;
// How far a layer's relief must stand over the one under it, in its height's
// units, to take the ground where their rules overlap.
static const float GROUND_HEIGHT_BLEND = 0.2;
// The broad variation: its patches' widths in meters, how far it moves the
// brightness, the warmth and the roughness either way.
static const float3 GROUND_MACRO_SCALES = float3(9.0, 45.0, 220.0);
static const float GROUND_MACRO_BRIGHTNESS = 0.3;
static const float GROUND_MACRO_WARMTH = 0.1;
static const float GROUND_MACRO_ROUGHNESS = 0.08;
// The damp sand above where the waves reach: how far up it runs in meters,
// and how dark and smooth it goes at the waves' reach.
static const float GROUND_DAMP_RISE = 2.0;
static const float GROUND_DAMP_DARKEN = 0.82;
static const float GROUND_DAMP_ROUGHNESS = 0.1;

// The up to three samples a texture is read with at a pixel: where each reads,
// how much of each the pixel takes, and the shared gradients of the texture
// coordinate across the pixel.
struct GroundTaps
{
    float2 uv[3];
    float weight[3];
    uint count;
    float2 dx;
    float2 dy;
};

// Lattice point `c` moved by whole periods into the first one, so points a
// texture-coordinate wrap apart draw the same offset.
float2 ground_hex_wrap(float2 c)
{
    float rows = floor(c.y / GROUND_HEX_PERIOD);
    c += rows * float2(0.5 * GROUND_HEX_PERIOD, -GROUND_HEX_PERIOD);
    c.x -= floor(c.x / GROUND_HEX_PERIOD) * GROUND_HEX_PERIOD;
    return c;
}

// The taps at texture coordinate `uv` (with screen gradients `dx` and `dy`)
// for a pixel covering `footprint` meters: three, offset by the hex vertices
// around the pixel, near the camera; one far from it.
GroundTaps ground_taps(float2 uv, float2 dx, float2 dy, float footprint)
{
    GroundTaps t = (GroundTaps)0;
    t.dx = dx;
    t.dy = dy;
    if (footprint >= GROUND_HEX_FAR)
    {
        t.uv[0] = uv;
        t.weight[0] = 1.0;
        t.count = 1u;
        return t;
    }
    // The triangle of the lattice holding the pixel and its barycentric
    // weights (Mikkelsen, "Practical Real-Time Hex-Tiling", 2022). The lattice
    // is sheared by a half cell per row rather than to equilateral triangles,
    // so it repeats where the texture coordinates wrap.
    float2 st = uv * GROUND_HEX_SCALE;
    float2 skewed = float2(st.x - 0.5 * st.y, st.y);
    float2 base = floor(skewed);
    float3 f = float3(frac(skewed), 0.0);
    f.z = 1.0 - f.x - f.y;
    float s = step(0.0, -f.z);
    float s2 = 2.0 * s - 1.0;
    float3 w = float3(-f.z * s2, s - f.y * s2, s - f.x * s2);
    float2 corners[3] = { base + float2(s, s), base + float2(s, 1.0 - s), base + float2(1.0 - s, s) };
    float total = 0.0;
    [unroll]
    for (uint i = 0u; i < 3u; i++)
    {
        uint h = grass_hash2(int2(ground_hex_wrap(corners[i])));
        t.uv[i] = uv + float2(grass_unit(h, 0x68bc21ebu), grass_unit(h, 0x02e5be93u));
        t.weight[i] = pow(max(w[i], 0.0), GROUND_HEX_SHARPNESS);
        total += t.weight[i];
    }
    [unroll]
    for (uint k = 0u; k < 3u; k++)
    {
        t.weight[k] /= max(total, 1e-6);
    }
    t.count = 3u;
    return t;
}

// One layer of ground at a pixel: its albedo, its tangent-space normal's x and
// y, and its relief's height in [0, 1].
struct GroundLayer
{
    float3 albedo;
    float2 normal_xy;
    float height;
};

// The layer whose albedo texture is pool entry `albedo_index` (alpha its
// height), tinted `tint`, and whose normal map is pool entry `normal_index`,
// read through `taps`. A tap whose relief stands higher than the others wins
// more of the pixel, so features meet rather than fade into each other.
GroundLayer ground_layer(uint albedo_index, uint normal_index, float3 tint, GroundTaps taps)
{
    GroundLayer l;
    l.albedo = tint;
    l.normal_xy = (float2)(0.0);
    l.height = 0.5;
    if (albedo_index == GROUND_COVER_UNTEXTURED)
    {
        return l;
    }
    float4 a[3];
    float w[3];
    float total = 0.0;
    [unroll]
    for (uint i = 0u; i < 3u; i++)
    {
        a[i] = (float4)(0.0);
        w[i] = 0.0;
        if (i < taps.count)
        {
            a[i] = pool_sample_grad(albedo_index, taps.uv[i], taps.dx, taps.dy);
            w[i] = taps.weight[i] * (a[i].a + 0.25);
            total += w[i];
        }
    }
    float3 albedo = (float3)(0.0);
    float height = 0.0;
    float2 normal_xy = (float2)(0.0);
    [unroll]
    for (uint k = 0u; k < 3u; k++)
    {
        if (k < taps.count)
        {
            float wk = w[k] / max(total, 1e-6);
            albedo += wk * a[k].rgb;
            height += wk * a[k].a;
            normal_xy += wk * (pool_sample_grad(normal_index, taps.uv[k], taps.dx, taps.dy).xy * 2.0 - 1.0);
        }
    }
    l.albedo = albedo * tint;
    l.normal_xy = normal_xy;
    l.height = height;
    return l;
}

// How much of the ground a layer of height `top` laid at rule weight `w` takes
// over ground of height `under`: none where its rule does not hold, all where
// it holds fully, and between, the higher relief first.
float ground_height_blend(float w, float top, float under)
{
    // The weight's ends pushed past the heights' range so they stay exact.
    float spread = w * 2.0 - 0.5;
    float a = top + spread;
    float b = under + 1.0 - spread;
    float m = max(a, b) - GROUND_HEIGHT_BLEND;
    float ta = max(a - m, 0.0);
    float tb = max(b - m, 0.0);
    return ta / max(ta + tb, 1e-6);
}

// The ground a pixel shows: its layers blended, how much of the surface's own
// material the covers leave (`keep`), and the covers' roughness and metallic
// weighted in by what they took.
struct GroundSurface
{
    GroundLayer layer;
    float keep;
    float2 roughness_metallic;
};

// The ground at `place` on a surface whose own albedo and normal maps are pool
// entries `albedo_index` and `normal_index`, tinted `tint`, each layer read
// through `taps`.
GroundSurface ground_surface_at(GroundPlace place, uint albedo_index, uint normal_index,
                                float3 tint, GroundTaps taps)
{
    GroundSurface g;
    g.layer = ground_layer(albedo_index, normal_index, tint, taps);
    g.keep = 1.0;
    g.roughness_metallic = (float2)(0.0);
    uint covers = min(CANOPY.cover_count, GROUND_MAX_COVERS);
    for (uint i = 0u; i < covers; i++)
    {
        GroundCover c = CANOPY.covers[i];
        float w = ground_rule_weight(c.rule, place.altitude, place.steepness, place.noise);
        if (w <= 0.0)
        {
            continue;
        }
        GroundLayer top = ground_layer(c.albedo_index, c.normal_index, c.tint, taps);
        float b = ground_height_blend(w, top.height, g.layer.height);
        g.layer.albedo = lerp(g.layer.albedo, top.albedo, b);
        g.layer.normal_xy = lerp(g.layer.normal_xy, top.normal_xy, b);
        g.layer.height = lerp(g.layer.height, top.height, b);
        g.keep *= 1.0 - b;
        g.roughness_metallic = lerp(g.roughness_metallic, float2(c.roughness, c.metallic), b);
    }
    return g;
}

// The broad variation at planet-fixed ground point `p` under a pixel covering
// `footprint` meters: a brightness multiplier (x), a warmth in [-1, 1] (y), and
// a roughness offset (z). Patches a pixel cannot hold fade to no change.
float3 ground_macro(float3 p, float footprint)
{
    float fine = grass_patch_noise(p, GROUND_MACRO_SCALES.x, 0x1f83d9abu, footprint);
    float mid = grass_patch_noise(p, GROUND_MACRO_SCALES.y, 0x5be0cd19u, footprint);
    float broad = grass_patch_noise(p, GROUND_MACRO_SCALES.z, 0xcbbb9d5du, footprint);
    float v = fine * 0.35 + mid * 0.35 + broad * 0.3 - 0.5;
    float warm = grass_patch_noise(p, GROUND_MACRO_SCALES.y * 1.7, 0x629a292au, footprint) - 0.5;
    return float3(1.0 + GROUND_MACRO_BRIGHTNESS * 2.0 * v, 2.0 * warm,
                  -GROUND_MACRO_ROUGHNESS * 2.0 * v);
}

// `albedo` with the broad variation `macro` laid over it.
float3 ground_macro_albedo(float3 albedo, float3 macro)
{
    float3 warmth = float3(1.0 + GROUND_MACRO_WARMTH * macro.y, 1.0,
                           1.0 - GROUND_MACRO_WARMTH * macro.y);
    return albedo * macro.x * warmth;
}

// How damp the ground is `height` meters above the planet's datum, from the
// top of the waves' reach (1) to where the ground above the beach is dry (0).
float ground_damp(float height)
{
    if (CANOPY.swash.runup <= 0.0)
    {
        return 0.0;
    }
    float above = height - CANOPY.swash.sea_level - CANOPY.swash.reach;
    return 1.0 - smoothstep(-GROUND_DAMP_RISE, GROUND_DAMP_RISE, above);
}

// The shading normal of ground whose tangent-space normal has `xy` across a
// surface of normal `normal`, `tangent` and `bitangent`.
float3 ground_shading_normal(float2 xy, float3 normal, float3 tangent, float3 bitangent)
{
    float3 n = float3(xy, sqrt(saturate(1.0 - dot(xy, xy))));
    float3x3 tbn = float3x3(normalize(tangent), normalize(bitangent), normalize(normal));
    return normalize(mul(n, tbn));
}
