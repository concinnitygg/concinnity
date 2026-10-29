// Screen-space global illumination. One fragment per compile, selected by a
// define so each variant declares exactly the resources it binds:
//
//   SSGI_DEPTH     - the closest (r) and farthest (g) linear depth of the
//                    surfaces in each trace pixel's footprint in the G-buffer:
//                    level 0 of the trace's depth pyramid.
//   SSGI_REDUCE    - one more pyramid level: the closest and farthest of the
//                    2x2 (3x3 on an odd edge) texels of the level below.
//   SSGI_AGE       - how many frames of history each trace pixel's
//                    accumulations hold (r): last frame's, reprojected by the
//                    motion vectors and rejected where last frame saw another
//                    surface, plus this one. Also the view normal (gba) of the
//                    surface the pixel stands for, which tells the next frame's
//                    reprojections one surface from another where depth alone
//                    cannot, such as where an object meets the ground.
//   SSGI_TRACE     - per trace pixel, cosine-weighted hemisphere rays traced
//                    through the depth pyramid, returning the lit scene
//                    color at each on-screen hit, blended into last frame's
//                    short (fast) accumulation reprojected the same way. The
//                    cosine-weighted sampling folds the cos(theta) / pdf factor
//                    away, so the mean hit radiance (rgb) estimates the
//                    albedo-free indirect irradiance over pi from what the rays
//                    hit, and the share of rays that hit (a) the share of the
//                    environment's irradiance those surfaces hold back.
//   SSGI_ACCUMULATE - the long accumulation: last frame's, reprojected, clamped
//                    to the fast accumulation's neighborhood, then blended
//                    with it.
//   SSGI_RESOLVE   - ahead of the main pass, last frame's long accumulation
//                    reprojected onto this frame's G-buffer with a depth-aware
//                    upsample and denoise at full resolution. Writes the
//                    environment's visibility (r: the lesser of the ambient
//                    occlusion and the rays' miss share) and the bounce (gba),
//                    which the main pass multiplies by the receiver's diffuse
//                    albedo.
//
// Pairs with `fullscreen_vertex` in fullscreen.hlsl.
//
// Each source is a `Texture2D` and a `SamplerState`: Vulkan binds a variant's n
// textures at 0..n-1 and their samplers at n..2n-1. The register numbers are
// the Metal indices (see concinnity-shader's `metal_bindings`) and the D3D root
// signature's slots alike. Every read but the hit radiance and the occlusion is
// a point load, by pixel position; each fragment still takes the fullscreen
// vertex's UV first, since D3D links a pixel shader only to a prefix of the
// vertex outputs.

{TEXTURE_SIZE}

// Layout matches `SsgiParams` in render_types.rs (48 B).
struct SsgiParams
{
    float intensity;
    float max_distance;
    float tan_half_fov_y;
    float aspect;
    // How far behind a surface, per meter of its view depth, a ray still hits it.
    float thickness_per_depth;
    // 1 once the previous frame's accumulation holds something to reproject.
    float history_valid;
    uint rays;
    // Advances every frame, so each frame draws fresh hemisphere samples.
    uint frame;
    // Levels in the depth pyramid.
    uint levels;
    // Render-resolution divisor of the trace.
    uint gi_scale;
    // What last frame's accumulation is multiplied by to read at this frame's
    // pre-exposure; the trace and its accumulation stay in the pre-exposed
    // scale of the scene they gather.
    float history_scale;
    // The least distance behind a surface a ray still hits it.
    float min_thickness;
};

[[vk::push_constant]] ConstantBuffer<SsgiParams> params : register(b0);

// Closest pyramid depth of a footprint with no surface in it. Far enough that
// every ray passes in front of it, finite so depth arithmetic stays finite. The
// farthest depth of such a footprint is 0, so sky never widens a depth range.
static const float SSGI_SKY = 1.0e20;

bool ssgi_is_sky(float depth)
{
    return depth >= SSGI_SKY;
}

// The full-resolution texel of `p`'s footprint that the trace stands for: its
// closest surface, which is what the pyramid's level 0 records. The footprint is
// the integer ratio of the two resolutions on each axis.
int2 ssgi_closest_texel(Texture2D<float4> gbuffer, int2 p, int2 full, int2 low)
{
    int2 scale = max(full / max(low, 1), 1);
    int2 base = p * scale;
    int2 best = min(base, full - 1);
    float best_depth = SSGI_SKY;
    for (int y = 0; y < scale.y; y++)
    {
        for (int x = 0; x < scale.x; x++)
        {
            int2 q = min(base + int2(x, y), full - 1);
            float d = gbuffer.Load(int3(q, 0)).a;
            if (d > 0.0 && d < best_depth)
            {
                best_depth = d;
                best = q;
            }
        }
    }
    return best;
}

#if defined(SSGI_DEPTH)

// binding 0 = the G-buffer (rgb = view normal, a = linear view depth).
[[vk::binding(0, 0)]] Texture2D<float4> gbuffer : register(t0);
[[vk::binding(1, 0)]] SamplerState gbuffer_samp : register(s0);

[shader("pixel")]
float4 ssgi_depth_fragment(
    [[vk::location(0)]] float2 vertex_uv : TEXCOORD0,
    float4 pixel : SV_Position) : SV_Target
{
    int2 full = int2(texture_size(gbuffer));
    // The pyramid's level 0 is the render resolution divided by `gi_scale`.
    int2 low = max(full / int(max(params.gi_scale, 1u)), 1);
    int2 scale = max(full / low, 1);
    int2 base = int2(pixel.xy) * scale;
    float2 range = float2(SSGI_SKY, 0.0);
    for (int y = 0; y < scale.y; y++)
    {
        for (int x = 0; x < scale.x; x++)
        {
            float d = gbuffer.Load(int3(min(base + int2(x, y), full - 1), 0)).a;
            if (d > 0.0)
            {
                range = float2(min(range.x, d), max(range.y, d));
            }
        }
    }
    return float4(range, 0.0, 0.0);
}

#elif defined(SSGI_REDUCE)

// binding 0 = the pyramid level below, as a single-level view.
[[vk::binding(0, 0)]] Texture2D<float4> finer : register(t0);
[[vk::binding(1, 0)]] SamplerState finer_samp : register(s0);

[shader("pixel")]
float4 ssgi_reduce_fragment(
    [[vk::location(0)]] float2 vertex_uv : TEXCOORD0,
    float4 pixel : SV_Position) : SV_Target
{
    int2 src = int2(texture_size(finer));
    int2 dst = max(src / 2, 1);
    int2 p = int2(pixel.xy);
    int2 base = p * 2;
    // The last row and column also fold in the odd texel a floored halving
    // leaves over, so no source texel goes unrepresented.
    int2 end = select(p == dst - 1, src, min(base + 2, src));
    float2 range = float2(SSGI_SKY, 0.0);
    for (int y = base.y; y < end.y; y++)
    {
        for (int x = base.x; x < end.x; x++)
        {
            float2 finer_range = finer.Load(int3(x, y, 0)).rg;
            range = float2(min(range.x, finer_range.x), max(range.y, finer_range.y));
        }
    }
    return float4(range, 0.0, 0.0);
}

#elif defined(SSGI_AGE) || defined(SSGI_TRACE) || defined(SSGI_ACCUMULATE) || defined(SSGI_RESOLVE)

// Longest accumulation, in frames: the blend weight of a new frame never drops
// below its reciprocal.
static const float SSGI_MAX_HISTORY = 32.0;
// Longest fast accumulation, in frames. Short enough to follow a moving light
// source within a few frames, long enough that its neighborhood is a usable
// bound on the long accumulation at one ray per pixel.
static const float SSGI_FAST_HISTORY = 3.0;
// Relative depth change past which a history texel is another surface.
static const float SSGI_DEPTH_TOLERANCE = 0.1;
// Least cosine between a history texel's normal and this pixel's below which it
// is another surface.
static const float SSGI_NORMAL_TOLERANCE = 0.9;

#if defined(SSGI_AGE)

// binding 0 = the G-buffer; 1 = the motion vectors; 2 = this frame's depth
// pyramid; 3 = last frame's pyramid; 4 = last frame's age.
[[vk::binding(0, 0)]] Texture2D<float4> gbuffer : register(t0);
[[vk::binding(5, 0)]] SamplerState gbuffer_samp : register(s0);
[[vk::binding(1, 0)]] Texture2D<float4> velocity : register(t1);
[[vk::binding(6, 0)]] SamplerState velocity_samp : register(s1);
[[vk::binding(2, 0)]] Texture2D<float4> pyramid : register(t2);
[[vk::binding(7, 0)]] SamplerState pyramid_samp : register(s2);
[[vk::binding(3, 0)]] Texture2D<float4> prev_pyramid : register(t3);
[[vk::binding(8, 0)]] SamplerState prev_pyramid_samp : register(s3);
[[vk::binding(4, 0)]] Texture2D<float4> prev_age : register(t4);
[[vk::binding(9, 0)]] SamplerState prev_age_samp : register(s4);

#elif defined(SSGI_TRACE) || defined(SSGI_ACCUMULATE)

// binding 0 = the trace's lit scene radiance (the bounce-radiance source), or
// the accumulation's fast accumulation from this frame's trace; 1 = the
// G-buffer; 2 = the motion vectors; 3 = this frame's depth pyramid, every
// level; 4 = last frame's pyramid; 5 = last frame's accumulation of the kind
// this variant writes; 6 = this frame's age; 7 = last frame's age.
#if defined(SSGI_TRACE)
[[vk::binding(0, 0)]] Texture2D<float4> scene : register(t0);
[[vk::binding(8, 0)]] SamplerState scene_samp : register(s0);
#else
[[vk::binding(0, 0)]] Texture2D<float4> fast : register(t0);
[[vk::binding(8, 0)]] SamplerState fast_samp : register(s0);
#endif
[[vk::binding(1, 0)]] Texture2D<float4> gbuffer : register(t1);
[[vk::binding(9, 0)]] SamplerState gbuffer_samp : register(s1);
[[vk::binding(2, 0)]] Texture2D<float4> velocity : register(t2);
[[vk::binding(10, 0)]] SamplerState velocity_samp : register(s2);
[[vk::binding(3, 0)]] Texture2D<float4> pyramid : register(t3);
[[vk::binding(11, 0)]] SamplerState pyramid_samp : register(s3);
[[vk::binding(4, 0)]] Texture2D<float4> prev_pyramid : register(t4);
[[vk::binding(12, 0)]] SamplerState prev_pyramid_samp : register(s4);
[[vk::binding(5, 0)]] Texture2D<float4> history : register(t5);
[[vk::binding(13, 0)]] SamplerState history_samp : register(s5);
[[vk::binding(6, 0)]] Texture2D<float4> age : register(t6);
[[vk::binding(14, 0)]] SamplerState age_samp : register(s6);
[[vk::binding(7, 0)]] Texture2D<float4> prev_age : register(t7);
[[vk::binding(15, 0)]] SamplerState prev_age_samp : register(s7);

#else

// binding 0 = last frame's long accumulation; 1 = last frame's pyramid; 2 =
// last frame's age; 3 = the G-buffer; 4 = the motion vectors; 5 = this
// frame's ambient occlusion (1x1 white without it).
[[vk::binding(0, 0)]] Texture2D<float4> history : register(t0);
[[vk::binding(6, 0)]] SamplerState history_samp : register(s0);
[[vk::binding(1, 0)]] Texture2D<float4> prev_pyramid : register(t1);
[[vk::binding(7, 0)]] SamplerState prev_pyramid_samp : register(s1);
[[vk::binding(2, 0)]] Texture2D<float4> prev_age : register(t2);
[[vk::binding(8, 0)]] SamplerState prev_age_samp : register(s2);
[[vk::binding(3, 0)]] Texture2D<float4> gbuffer : register(t3);
[[vk::binding(9, 0)]] SamplerState gbuffer_samp : register(s3);
[[vk::binding(4, 0)]] Texture2D<float4> velocity : register(t4);
[[vk::binding(10, 0)]] SamplerState velocity_samp : register(s4);
[[vk::binding(5, 0)]] Texture2D<float4> occlusion : register(t5);
[[vk::binding(11, 0)]] SamplerState occlusion_samp : register(s5);

#endif

#if !defined(SSGI_RESOLVE)

// Whether last frame's trace texel `tap` saw the surface at `depth` facing
// `normal`.
bool ssgi_same_surface(int2 tap, float depth, float3 normal)
{
    float prev_depth = prev_pyramid.Load(int3(tap, 0)).r;
    float3 prev_normal = prev_age.Load(int3(tap, 0)).gba;
    return abs(prev_depth - depth) <= depth * SSGI_DEPTH_TOLERANCE
        && dot(prev_normal, normal) >= SSGI_NORMAL_TOLERANCE;
}

// `source` at `prev_uv` in last frame's trace grid, bilinear over the four
// texels around it, each kept only where last frame saw the same surface. Zero
// when none did.
float4 ssgi_reproject(Texture2D<float4> source, float2 prev_uv, float depth, float3 normal, int2 low)
{
    if (params.history_valid < 0.5 || any(prev_uv < 0.0) || any(prev_uv > 1.0))
    {
        return (float4)(0.0);
    }
    float2 hp = prev_uv * float2(low) - 0.5;
    int2 base = int2(floor(hp));
    float2 f = hp - float2(base);
    float4 sum = (float4)(0.0);
    float wsum = 0.0;
    for (int i = 0; i < 4; i++)
    {
        int2 o = int2(i & 1, i >> 1);
        int2 tap = base + o;
        if (any(tap < 0) || any(tap >= low))
        {
            continue;
        }
        if (!ssgi_same_surface(tap, depth, normal))
        {
            continue;
        }
        float2 wb = lerp(1.0 - f, f, float2(o));
        float w = wb.x * wb.y;
        sum += source.Load(int3(tap, 0)) * w;
        wsum += w;
    }
    return wsum > 1e-3 ? sum / wsum : (float4)(0.0);
}

// Where full-resolution texel `q` reprojects from: its UV moved by its motion.
float2 ssgi_prev_uv(int2 q, int2 full)
{
    return (float2(q) + 0.5) / float2(full) + velocity.Load(int3(q, 0)).rg;
}

// The view normal of full-resolution texel `q`.
float3 ssgi_normal(int2 q)
{
    return normalize(gbuffer.Load(int3(q, 0)).xyz);
}

#if !defined(SSGI_AGE)

// Last frame's accumulation, reprojected, at this frame's pre-exposure.
float4 ssgi_history(float2 prev_uv, float depth, float3 normal, int2 low)
{
    float4 h = ssgi_reproject(history, prev_uv, depth, normal, low);
    return float4(h.rgb * params.history_scale, h.a);
}

#endif

#endif

#if defined(SSGI_AGE)

[shader("pixel")]
float4 ssgi_age_fragment(
    [[vk::location(0)]] float2 vertex_uv : TEXCOORD0,
    float4 pixel : SV_Position) : SV_Target
{
    int2 p = int2(pixel.xy);
    float depth = pyramid.Load(int3(p, 0)).r;
    if (ssgi_is_sky(depth))
    {
        return (float4)(0.0);
    }
    int2 low = int2(texture_size(pyramid));
    int2 full = int2(texture_size(gbuffer));
    int2 q = ssgi_closest_texel(gbuffer, p, full, low);
    float3 normal = ssgi_normal(q);
    float prev = ssgi_reproject(prev_age, ssgi_prev_uv(q, full), depth, normal, low).r;
    return float4(min(prev + 1.0, SSGI_MAX_HISTORY), normal);
}

#elif defined(SSGI_TRACE)

static const float SSGI_PI = 3.14159265359;
// Traversal steps a ray may take before it counts as a miss.
static const uint SSGI_MAX_ITERATIONS = 48;
// Distance, in trace pixels, a ray travels at full precision; each doubling
// past it lets the finest level the ray resolves climb by one.
static const float SSGI_EXACT_REACH = 64.0;
// Closest view depth a ray may reach before it is cut off at the camera.
static const float SSGI_NEAR = 0.05;
// Origin offset along the surface normal, as a fraction of the thickness, so a
// ray does not immediately self-intersect the surface it starts on.
static const float SSGI_NORMAL_BIAS = 0.25;
// Brightest luminance, in the scene's pre-exposed scale, one ray's hit may
// carry. A small highlight (a chrome glint) holds far more radiance than its
// few pixels can spread over a one-ray-per-pixel estimate: unclamped, the
// pixels whose rays happen to strike it light up as sparks around it.
static const float SSGI_HIT_LUMINANCE = 4.0;

// `radiance` scaled down, hue kept, to at most `SSGI_HIT_LUMINANCE`.
float3 ssgi_clamp_hit(float3 radiance)
{
    float lum = dot(radiance, float3(0.2126, 0.7152, 0.0722));
    return lum > SSGI_HIT_LUMINANCE ? radiance * (SSGI_HIT_LUMINANCE / lum) : radiance;
}

// How far behind a surface at view depth `depth` a ray still hits it.
float ssgi_thickness(float depth)
{
    return max(depth * params.thickness_per_depth, params.min_thickness);
}

// Rebuild a view-space position from a UV and its linear (view-space) depth.
// Matches ssr_view_pos / ssao_view_pos.
float3 ssgi_view_pos(float2 uv, float depth)
{
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return float3(ndc.x * params.tan_half_fov_y * params.aspect, ndc.y * params.tan_half_fov_y, -1.0)
        * depth;
}

// Project a view-space point (z < 0, in front of the camera) to a screen UV.
float2 ssgi_project(float3 q)
{
    float inv = 1.0 / max(-q.z, 1e-4);
    float2 ndc = float2(q.x * inv / (params.tan_half_fov_y * params.aspect),
                        q.y * inv / params.tan_half_fov_y);
    return float2(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
}

// Interleaved gradient noise: a cheap per-pixel hash in [0, 1).
float ssgi_ign(float2 p)
{
    return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715))));
}

// Sample `i` of the R2 low-discrepancy sequence, in fixed point so a large
// index keeps its precision.
float2 ssgi_r2(uint i)
{
    uint2 v = uint2(i * 3242174889u, i * 2447445414u) + 2147483648u;
    return float2(v) * 2.3283064365386963e-10; // / 2^32
}

// Where a ray leaving `pos` along `dir` crosses out of the `size`-wide cell
// `cell`, in the ray's own parameter.
float ssgi_cell_exit(float2 start, float2 dir, float2 cell, float size)
{
    float2 boundary = (cell + select(dir > 0.0, (float2)(1.0), (float2)(0.0))) * size;
    float2 t = select(dir != 0.0, (boundary - start) / dir, (float2)(1.0e20));
    return min(t.x, t.y);
}

// Trace one ray through the pyramid. The ray is walked in level-0 pixels with
// its inverse depth interpolated linearly, which is what perspective does to
// view depth along a screen-space line. A cell the ray's whole span inside it
// passes in front of every surface in, or behind every surface in by more than
// their thickness (`ssgi_thickness` of their depth), is skipped, and the walk
// climbs a level; one it may touch is descended into, down to the finest level
// the ray's distance allows.
//
// Near its origin a ray descends to level 0, where the pyramid's depth range
// only says the texel may be touched: the hit is judged against the G-buffer
// depth under the ray, since a surface seen at a grazing angle spans a range of
// depths across one footprint and its closest one would catch the ray's own
// origin. A surface within `thickness` in front of the ray is a hit; one
// farther in front is passed behind. Farther out the finest level coarsens with
// distance, like a cone widening, and a cell is the hit where the ray reaches
// within `thickness` behind its closest surface: diffuse bounce needs no finer
// answer, and a ray skimming a surface would otherwise spend a step per texel.
// A ray farther behind passes the cell, so it does not take the light of an
// object it passes behind.
bool ssgi_trace(float3 origin, float3 dir, float2 low, float2 full, out float2 hit_uv)
{
    hit_uv = (float2)(0.0);
    if (origin.z > -SSGI_NEAR)
    {
        return false;
    }
    float3 end = origin + dir * params.max_distance;
    if (end.z > -SSGI_NEAR)
    {
        end = lerp(origin, end, (-SSGI_NEAR - origin.z) / (end.z - origin.z));
    }
    float2 start = ssgi_project(origin) * low;
    float2 delta = ssgi_project(end) * low - start;
    float len = length(delta);
    if (len < 0.5)
    {
        return false;
    }
    float w0 = 1.0 / -origin.z;
    float w1 = 1.0 / -end.z;

    // Stop where the ray leaves the screen.
    float2 edge = select(delta > 0.0, low, (float2)(0.0));
    float2 t_edge = select(delta != 0.0, (edge - start) / delta, (float2)(1.0e20));
    float t_max = min(1.0, min(t_edge.x, t_edge.y));

    // A hundredth of a pixel, so a step lands inside the next cell.
    float eps = 0.01 / len;
    float t = ssgi_cell_exit(start, delta, floor(start), 1.0) + eps;
    int2 low_i = int2(low);
    int2 full_i = int2(full);
    float2 to_full = full / low;
    uint top = max(params.levels, 1u) - 1u;
    uint level = 0u;
    for (uint i = 0u; i < SSGI_MAX_ITERATIONS && t < t_max; i++)
    {
        float2 pos = start + delta * t;
        float size = float(1u << level);
        float2 cell = floor(pos / size);
        int2 level_size = max(low_i >> level, 1);
        float2 range = pyramid.Load(int3(min(int2(cell), level_size - 1), int(level))).rg;
        float t_exit = min(ssgi_cell_exit(start, delta, cell, size), t_max);
        float depth_in = 1.0 / lerp(w0, w1, t);
        float depth_out = 1.0 / lerp(w0, w1, t_exit);
        if (max(depth_in, depth_out) < range.x
            || min(depth_in, depth_out) > range.y + ssgi_thickness(range.y))
        {
            t = t_exit + eps;
            level = min(level + 1u, top);
        }
        else if (level > min(uint(max(log2(t * len / SSGI_EXACT_REACH) + 1.0, 0.0)), top))
        {
            level--;
        }
        else if (level > 0u)
        {
            if (min(depth_in, depth_out) <= range.x + ssgi_thickness(range.x))
            {
                hit_uv = pos / low;
                return true;
            }
            t = t_exit + eps;
            level = min(level + 1u, top);
        }
        else
        {
            int2 under = min(int2(pos * to_full), full_i - 1);
            float actual = gbuffer.Load(int3(under, 0)).a;
            if (actual > 0.0 && max(depth_in, depth_out) >= actual
                && min(depth_in, depth_out) <= actual + ssgi_thickness(actual))
            {
                hit_uv = pos / low;
                return true;
            }
            t = t_exit + eps;
        }
    }
    return false;
}

[shader("pixel")]
float4 ssgi_trace_fragment(
    [[vk::location(0)]] float2 vertex_uv : TEXCOORD0,
    float4 pixel : SV_Position) : SV_Target
{
    int2 p = int2(pixel.xy);
    float depth = pyramid.Load(int3(p, 0)).r;
    if (ssgi_is_sky(depth))
    {
        return (float4)(0.0);
    }
    float2 full = texture_size(gbuffer);
    float2 low = texture_size(pyramid);
    int2 q = ssgi_closest_texel(gbuffer, p, int2(full), int2(low));
    float4 c = gbuffer.Load(int3(q, 0));
    float3 n = normalize(c.xyz);
    float2 uv = (float2(q) + 0.5) / full;
    float3 pos = ssgi_view_pos(uv, c.a);

    // Orthonormal basis around the view-space normal.
    float3 up = abs(n.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 t = normalize(cross(up, n));
    float3 b = cross(n, t);
    float3 origin = pos + n * (ssgi_thickness(c.a) * SSGI_NORMAL_BIAS);

    // The sequence advances through time and a per-pixel rotation decorrelates
    // neighbors. The rotation moves every frame too: held still, what the
    // accumulation leaves over would keep the noise's own lattice.
    float2 cycle = (float)(params.frame % 64u) * float2(5.588238, 3.1415927);
    float2 rotation = float2(ssgi_ign(pixel.xy + cycle.x), ssgi_ign(pixel.xy + float2(47.0, 17.0) + cycle.y));
    uint rays = max(params.rays, 1u);
    float4 gathered = (float4)(0.0);
    for (uint i = 0u; i < rays; i++)
    {
        float2 u = frac(ssgi_r2(params.frame * rays + i) + rotation);
        float r = sqrt(u.x);
        float phi = 2.0 * SSGI_PI * u.y;
        float3 d = normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u.x)));
        float2 hit_uv;
        if (ssgi_trace(origin, d, low, full, hit_uv))
        {
            gathered += float4(ssgi_clamp_hit(scene.SampleLevel(scene_samp, hit_uv, 0.0).rgb), 1.0);
        }
    }
    gathered /= float(rays);

    // No neighborhood clip: at one ray per pixel a whole neighborhood misses
    // together often enough that clipping to it would keep resetting sparse
    // light to black. The fast accumulation settles a lighting change within a
    // few frames, and bounds the long one.
    float4 hist = ssgi_history(ssgi_prev_uv(q, int2(full)), depth, n, int2(low));
    float len = clamp(age.Load(int3(p, 0)).r, 1.0, SSGI_FAST_HISTORY);
    return lerp(hist, gathered, 1.0 / len);
}

#elif defined(SSGI_ACCUMULATE)

// Standard deviations of the fast accumulation's neighborhood the long
// accumulation may stray from it. Wide enough that a converged pixel's noise
// stays inside, so a still scene keeps the long accumulation's smoothness;
// a bounce that moved away leaves it, and the long accumulation follows within
// the fast one's few frames instead of its own.
static const float SSGI_HISTORY_CLAMP = 1.0;

[shader("pixel")]
float4 ssgi_accumulate_fragment(
    [[vk::location(0)]] float2 vertex_uv : TEXCOORD0,
    float4 pixel : SV_Position) : SV_Target
{
    int2 p = int2(pixel.xy);
    float depth = pyramid.Load(int3(p, 0)).r;
    if (ssgi_is_sky(depth))
    {
        return (float4)(0.0);
    }
    int2 low = int2(texture_size(pyramid));
    int2 full = int2(texture_size(gbuffer));
    float4 current = fast.Load(int3(p, 0));

    // The fast accumulation's mean and spread over the 3x3 around the pixel,
    // on its own surface.
    float4 m1 = (float4)(0.0);
    float4 m2 = (float4)(0.0);
    float n = 0.0;
    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            int2 tap = clamp(p + int2(x, y), 0, low - 1);
            float tap_depth = pyramid.Load(int3(tap, 0)).r;
            if (abs(tap_depth - depth) > depth * SSGI_DEPTH_TOLERANCE)
            {
                continue;
            }
            float4 c = fast.Load(int3(tap, 0));
            m1 += c;
            m2 += c * c;
            n += 1.0;
        }
    }
    float4 mean = m1 / n;
    float4 sigma = sqrt(max(m2 / n - mean * mean, (float4)(0.0)));
    float4 lo = min(mean - SSGI_HISTORY_CLAMP * sigma, current);
    float4 hi = max(mean + SSGI_HISTORY_CLAMP * sigma, current);

    int2 q = ssgi_closest_texel(gbuffer, p, full, low);
    float4 hist = ssgi_history(ssgi_prev_uv(q, full), depth, ssgi_normal(q), low);
    float len = clamp(age.Load(int3(p, 0)).r, 1.0, SSGI_MAX_HISTORY);
    return lerp(clamp(hist, lo, hi), current, 1.0 / len);
}

#else

// Spatial filter radius, in trace texels, for a young accumulation and for a
// converged one: a pixel with little history borrows more from its neighbors.
// The filter is a separable tent that reaches zero at its radius, and the
// window is the 4x4 texels around the pixel's position, which holds every tap
// within two texels, so no weight jumps as the window moves.
static const float SSGI_RADIUS_YOUNG = 2.0;
static const float SSGI_RADIUS_CONVERGED = 1.0;
// Frames of history over which the filter narrows from young to converged.
static const float SSGI_CONVERGE_FRAMES = 8.0;
// Relative depth difference at which a neighbor's weight falls to 1/e.
static const float SSGI_DEPTH_SIGMA = 0.1;

[shader("pixel")]
float4 ssgi_resolve_fragment(
    [[vk::location(0)]] float2 vertex_uv : TEXCOORD0,
    float4 pixel : SV_Position) : SV_Target
{
    int2 fp = int2(pixel.xy);
    float2 full = texture_size(gbuffer);
    float2 uv = (float2(fp) + 0.5) / full;
    float ao = occlusion.SampleLevel(occlusion_samp, uv, 0.0).r;
    float4 surface = gbuffer.Load(int3(fp, 0));
    float depth = surface.a;
    float2 prev_uv = uv + velocity.Load(int3(fp, 0)).rg;
    if (depth <= 0.0 || params.history_valid < 0.5 || any(prev_uv < 0.0) || any(prev_uv > 1.0))
    {
        return float4(ao, 0.0, 0.0, 0.0);
    }
    float2 low = texture_size(prev_pyramid);
    int2 low_i = int2(low);
    float2 lp = prev_uv * low - 0.5;
    int2 base = int2(floor(lp));
    int2 nearest = clamp(int2(floor(lp + 0.5)), 0, low_i - 1);
    float frames = prev_age.Load(int3(nearest, 0)).r;
    float radius = lerp(SSGI_RADIUS_YOUNG, SSGI_RADIUS_CONVERGED, saturate(frames / SSGI_CONVERGE_FRAMES));

    float3 normal = normalize(surface.xyz);
    float4 sum = (float4)(0.0);
    float wsum = 0.0;
    for (int y = -1; y <= 2; y++)
    {
        for (int x = -1; x <= 2; x++)
        {
            int2 tap = base + int2(x, y);
            float2 w2 = max(1.0 - abs(float2(tap) - lp) / radius, 0.0);
            float w = w2.x * w2.y;
            if (w <= 0.0 || any(tap < 0) || any(tap >= low_i))
            {
                continue;
            }
            float tap_depth = prev_pyramid.Load(int3(tap, 0)).r;
            float dd = abs(tap_depth - depth);
            float3 tap_normal = prev_age.Load(int3(tap, 0)).gba;
            if (ssgi_is_sky(tap_depth) || dd > depth * SSGI_DEPTH_TOLERANCE
                || dot(tap_normal, normal) < SSGI_NORMAL_TOLERANCE)
            {
                continue;
            }
            w *= exp(-dd / (depth * SSGI_DEPTH_SIGMA));
            sum += history.Load(int3(tap, 0)) * w;
            wsum += w;
        }
    }
    if (wsum <= 1e-6)
    {
        return float4(ao, 0.0, 0.0, 0.0);
    }
    float4 gi = sum / wsum;
    float visibility = min(ao, 1.0 - saturate(gi.a));
    return float4(visibility, gi.rgb * (params.history_scale * params.intensity));
}

#endif

#else
#error "ssgi.hlsl: define SSGI_DEPTH, SSGI_REDUCE, SSGI_AGE, SSGI_TRACE, SSGI_ACCUMULATE or SSGI_RESOLVE"
#endif
