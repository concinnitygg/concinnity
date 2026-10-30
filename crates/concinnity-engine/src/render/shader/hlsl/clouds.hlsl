// The clouds' noise volume and their composite: single source for every
// backend. The march between them is `cloud_march_kernel` in atmosphere.hlsl,
// which shares the haze pass's bindings.
//
//   CLOUD_NOISE      builds the tiling noise volume every layer is carved
//                    from, once, when the world loads. Mirrored channel for
//                    channel in render::environment::clouds::noise.
//   CLOUD_COMPOSITE  the fullscreen pass that blends the half-resolution
//                    cloud target over the scene, each pixel taking the
//                    target texels whose marched pixels lie at its depth.
//
// Every resource takes its Metal index from its register number.

{DEPTH_CONVENTION}
{CLOUD_TYPES}

#ifdef CLOUD_NOISE

[[vk::binding(0, 0)]] [[vk::image_format("rgba8")]] RWTexture3D<float4> cloud_noise_out : register(u0);

// Lattice cells of the weather channels, and of the base shape and the detail
// at their coarsest octave. `WEATHER_CELLS`, `SHAPE_CELLS` and `DETAIL_CELLS`
// in render::environment::clouds.
static const float CLOUD_WEATHER_CELLS = 6.0;
static const float CLOUD_SHAPE_CELLS = 4.0;
static const float CLOUD_DETAIL_CELLS = 4.0;

// The PCG hash over three lanes (Jarzynski and Olano 2020).
uint3 cloud_hash3(uint3 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

float3 cloud_hash3f(uint3 v)
{
    return float3(cloud_hash3(v) >> 8u) * (1.0 / 16777216.0);
}

uint3 cloud_wrap3(int3 c, int period)
{
    return uint3(((c % period) + period) % period);
}

float cloud_fade(float t)
{
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float cloud_corner(int3 cell, float3 f, int3 d, int period)
{
    float3 g = cloud_hash3f(cloud_wrap3(cell + d, period)) * 2.0 - 1.0;
    return dot(g, f - float3(d));
}

// Gradient noise at `p` (in cells) over a tile `period` cells across.
float cloud_perlin(float3 p, int period)
{
    int3 cell = int3(floor(p));
    float3 f = p - float3(cell);
    float3 u = float3(cloud_fade(f.x), cloud_fade(f.y), cloud_fade(f.z));
    float x00 = lerp(cloud_corner(cell, f, int3(0, 0, 0), period),
                     cloud_corner(cell, f, int3(1, 0, 0), period), u.x);
    float x10 = lerp(cloud_corner(cell, f, int3(0, 1, 0), period),
                     cloud_corner(cell, f, int3(1, 1, 0), period), u.x);
    float x01 = lerp(cloud_corner(cell, f, int3(0, 0, 1), period),
                     cloud_corner(cell, f, int3(1, 0, 1), period), u.x);
    float x11 = lerp(cloud_corner(cell, f, int3(0, 1, 1), period),
                     cloud_corner(cell, f, int3(1, 1, 1), period), u.x);
    return lerp(lerp(x00, x10, u.y), lerp(x01, x11, u.y), u.z);
}

// One minus the distance to the nearest of a point scattered per cell.
float cloud_worley(float3 p, int period)
{
    int3 cell = int3(floor(p));
    float nearest = 1.0;
    for (int dz = -1; dz <= 1; ++dz)
    {
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                int3 c = cell + int3(dx, dy, dz);
                float3 d = float3(c) + cloud_hash3f(cloud_wrap3(c, period)) - p;
                nearest = min(nearest, dot(d, d));
            }
        }
    }
    return 1.0 - min(sqrt(nearest), 1.0);
}

float cloud_perlin_fbm(float3 p, float cells, float3 weights)
{
    float sum = 0.0;
    float scale = cells;
    for (uint i = 0u; i < 3u; ++i)
    {
        sum += weights[i] * (cloud_perlin(p * scale, int(scale)) * 0.5 + 0.5);
        scale *= 2.0;
    }
    return sum / (weights.x + weights.y + weights.z);
}

float cloud_worley_fbm(float3 p, float cells, float3 weights)
{
    float sum = 0.0;
    float scale = cells;
    for (uint i = 0u; i < 3u; ++i)
    {
        sum += weights[i] * cloud_worley(p * scale, int(scale));
        scale *= 2.0;
    }
    return sum / (weights.x + weights.y + weights.z);
}

// The four channels at `p`, in tiles. `noise_channels` in
// render::environment::clouds::noise.
float4 cloud_noise_at(float3 p)
{
    float shape_perlin = cloud_perlin_fbm(p, CLOUD_SHAPE_CELLS, float3(0.5, 0.25, 0.125));
    float shape_worley = cloud_worley_fbm(p, CLOUD_SHAPE_CELLS, float3(0.625, 0.25, 0.125));
    float shape = saturate((shape_perlin - (shape_worley - 1.0)) / (1.0 - (shape_worley - 1.0)));
    float detail = cloud_worley_fbm(p, CLOUD_DETAIL_CELLS, float3(0.625, 0.25, 0.125));
    float weather = cloud_perlin_fbm(p, CLOUD_WEATHER_CELLS, float3(0.5, 0.3, 0.2));
    float cells = cloud_worley(p * CLOUD_WEATHER_CELLS, int(CLOUD_WEATHER_CELLS));
    return float4(shape, detail, saturate(weather), cells);
}

[shader("compute")]
[numthreads(4, 4, 4)]
void cloud_noise_kernel(uint3 id : SV_DispatchThreadID)
{
    if (any(id >= CLOUD_NOISE_SIZE))
    {
        return;
    }
    float3 p = (float3(id) + 0.5) / float(CLOUD_NOISE_SIZE);
    cloud_noise_out[id] = cloud_noise_at(p);
}

#endif

#ifdef CLOUD_COMPOSITE

[[vk::binding(0, 0)]] ConstantBuffer<CloudParams> cloud_cb : register(b0);
#if USE_MSAA
[[vk::binding(1, 0)]] Texture2DMS<float> scene_depth : register(t0);
#else
[[vk::binding(1, 0)]] Texture2D<float> scene_depth : register(t0);
#endif
[[vk::binding(2, 0)]] Texture2D<float4> cloud_target : register(t1);

#define CLOUD cloud_cb

float composite_depth(int2 pixel)
{
#if USE_MSAA
    return scene_depth.Load(pixel, 0);
#else
    return scene_depth.Load(int3(pixel, 0));
#endif
}

// How much two depths stand for the same surface. Reversed depth falls off as
// the reciprocal of distance, so their ratio is the ratio of the distances;
// the sky only matches the sky.
float composite_match(float a, float b)
{
    bool sky_a = depth_is_cleared(a);
    bool sky_b = depth_is_cleared(b);
    if (sky_a || sky_b)
    {
        return sky_a && sky_b ? 1.0 : 0.0;
    }
    float ratio = min(a, b) / max(a, b);
    return smoothstep(0.85, 1.0, ratio);
}

// `(cloud light, 1 - transmittance)` over the scene with the premultiplied
// `over` blend: `scene * T + light`.
[shader("pixel")]
float4 cloud_composite_fragment(
    [[vk::location(0)]] float2 vertex_uv : TEXCOORD0,
    float4 sv_pos : SV_Position) : SV_Target
{
    int2 pixel = int2(sv_pos.xy);
    int2 scene_max = int2(CLOUD.scene.xy) - 1;
    int2 target_max = int2(CLOUD.target.xy) - 1;
    if (any(pixel > scene_max))
    {
        discard;
    }
    float depth = composite_depth(pixel);
    // Target texel q marched pixel q * 2 + o; find the four whose marched
    // pixels surround this one.
    float2 at = (float2(pixel) - CLOUD.scene.zw) / float(CLOUD_TARGET_DIVISOR);
    int2 base = int2(floor(at));
    float2 f = at - float2(base);
    float4 sum = (float4)0.0;
    float total = 0.0;
    float4 best = float4(0.0, 0.0, 0.0, 1.0);
    float best_match = -1.0;
    [unroll]
    for (uint j = 0u; j < 4u; ++j)
    {
        int2 o = int2(j & 1u, j >> 1u);
        int2 q = clamp(base + o, int2(0, 0), target_max);
        int2 marched = min(q * int(CLOUD_TARGET_DIVISOR) + int2(CLOUD.scene.zw), scene_max);
        float match = composite_match(depth, composite_depth(marched));
        float2 bw = lerp(1.0 - f, f, float2(o));
        float w = bw.x * bw.y * match;
        float4 c = cloud_target.Load(int3(q, 0));
        sum += c * w;
        total += w;
        if (match > best_match)
        {
            best_match = match;
            best = c;
        }
    }
    float4 cloud = total > 1e-4 ? sum / total : best;
    return float4(cloud.rgb, 1.0 - cloud.a);
}

#endif
