// Screen-space global illumination. One fragment per compile, selected by a
// define so each variant declares exactly the resources it binds (Metal and
// DXIL indices are assigned in declaration order, so an unused declaration
// would shift the live ones):
//
//   SSGI_GATHER    - per pixel, cast cosine-weighted hemisphere rays around the
//                    surface normal, screen-march each against the SSR pre-pass
//                    G-buffer, and accumulate the lit scene colour at each
//                    on-screen hit. Misses contribute nothing (the IBL ambient
//                    already covers the off-screen / sky term). The
//                    cosine-weighted importance sampling folds the cos(theta) /
//                    pdf factor away, so the estimate of the (albedo-free)
//                    indirect irradiance is just the mean hit radiance.
//   SSGI_COMPOSITE - a depth-aware box blur of that noisy gather output, which
//                    the pipeline additively blends into the scene.
//
// Pairs with `fullscreen_vertex` in fullscreen.slang.

{POST_COMMON}

// Layout matches `SsgiParams` in render_types.rs (32 B). The composite reads
// only `intensity`, but the whole block is shared with the gather.
struct SsgiParams
{
    float intensity;
    float max_distance;
    float tan_half_fov_y;
    float aspect;
    float stride;
    float thickness;
    // Rays cast per pixel over the hemisphere, and march samples per ray. Both
    // arrive as floats and are read as int loop bounds; the Rust side derives
    // the stride from `steps`.
    float rays;
    float steps;
};

[[vk::push_constant]]
ConstantBuffer<SsgiParams> params;

#if defined(SSGI_GATHER)

// binding 0 = lit scene radiance (the bounce-radiance source); binding 1 = the
// SSR pre-pass G-buffer (rgb = view normal, a = linear view depth).
[[vk::binding(0, 0)]] Sampler2D<float4> scene;
[[vk::binding(1, 0)]] Sampler2D<float4> gbuffer;

// Origin offset along the surface normal (x stride) so a ray does not
// immediately self-intersect the surface it starts on.
static const float SSGI_NORMAL_BIAS = 0.5;
static const float SSGI_PI = 3.14159265359;

// Rebuild a view-space position from a UV and its linear (view-space) depth.
// Matches ssr_view_pos / ssao_view_pos.
float3 ssgi_view_pos(float2 uv, float depth, float tan_y, float aspect)
{
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return float3(ndc.x * tan_y * aspect, ndc.y * tan_y, -1.0) * depth;
}

// Project a view-space point (z < 0, in front of the camera) to a screen UV.
float2 ssgi_project(float3 q, float tan_y, float aspect)
{
    float inv = 1.0 / max(-q.z, 1e-4);
    float2 ndc = float2(q.x * inv / (tan_y * aspect), q.y * inv / tan_y);
    return float2(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
}

// Interleaved gradient noise: a cheap per-pixel hash in [0, 1). Decorrelates
// the hemisphere sampling spatially; the depth-aware blur + TAA clean up the
// residual high-frequency noise.
float ssgi_ign(float2 p)
{
    return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715))));
}

// Van der Corput radical inverse (base 2), for the low-discrepancy ray set.
float ssgi_vdc(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 2^32
}

[shader("fragment")]
float4 ssgi_gather_fragment(
    [[vk::location(0)]] float2 uv : TEXCOORD0,
    float4 pixel : SV_Position) : SV_Target
{
    float4 c = gbuffer.Sample(uv);
    float depth = c.a;
    if (depth <= 0.0)
    {
        return float4(0.0, 0.0, 0.0, 1.0);               // background / sky
    }

    float3 n = normalize(c.xyz);
    float3 p = ssgi_view_pos(uv, depth, params.tan_half_fov_y, params.aspect);

    // Orthonormal basis around the view-space normal.
    float3 up = abs(n.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 t = normalize(cross(up, n));
    float3 b = cross(n, t);

    float jitter = ssgi_ign(pixel.xy);
    float3 origin = p + n * (params.stride * SSGI_NORMAL_BIAS);

    int rays  = max(1, int(params.rays));
    int steps = max(1, int(params.steps));

    float3 indirect = float3(0.0);
    for (int i = 0; i < rays; i++)
    {
        // Stratified cosine-weighted hemisphere sample, jittered per pixel.
        float u1 = (float(i) + jitter) / float(rays);
        float u2 = frac(ssgi_vdc(uint(i + 1)) + jitter);
        float r   = sqrt(u1);
        float phi = 2.0 * SSGI_PI * u2;
        float3 d_t = float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
        float3 d   = normalize(t * d_t.x + b * d_t.y + n * d_t.z);

        float3 step_v = d * params.stride;
        float3 q = origin;
        for (int s = 0; s < steps; s++)
        {
            q += step_v;
            if (q.z >= 0.0) break;                       // crossed camera plane
            float2 hit_uv = ssgi_project(q, params.tan_half_fov_y, params.aspect);
            if (hit_uv.x < 0.0 || hit_uv.x > 1.0 || hit_uv.y < 0.0 || hit_uv.y > 1.0) break;
            float scene_depth = gbuffer.Sample(hit_uv).a;
            if (scene_depth <= 0.0) continue;            // sky here - keep marching
            float diff = (-q.z) - scene_depth;           // > 0: ray is behind the surface
            if (diff > 0.0 && diff < params.thickness)
            {
                indirect += scene.Sample(hit_uv).rgb;    // bounced radiance
                break;
            }
        }
    }

    indirect *= (1.0 / float(rays));
    return float4(indirect, 1.0);
}

#elif defined(SSGI_COMPOSITE)

// binding 0 = the noisy gather output; binding 1 = the SSR pre-pass G-buffer
// (depth in .a) for the depth-similarity weighting.
[[vk::binding(0, 0)]] Sampler2D<float4> gi_tex;
[[vk::binding(1, 0)]] Sampler2D<float4> gbuffer;

// Depth-aware blur footprint: a (2R+1)^2 box weighted by depth similarity, so
// the indirect term denoises without bleeding across silhouettes.
static const int SSGI_BLUR_RADIUS = 2;

[shader("fragment")]
float4 ssgi_composite_fragment([[vk::location(0)]] float2 uv : TEXCOORD0) : SV_Target
{
    float center_depth = gbuffer.Sample(uv).a;
    if (center_depth <= 0.0)
    {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    float2 texel = 1.0 / combined_size(gi_tex);
    float3 sum = float3(0.0);
    float wsum = 0.0;
    for (int dy = -SSGI_BLUR_RADIUS; dy <= SSGI_BLUR_RADIUS; dy++)
    {
        for (int dx = -SSGI_BLUR_RADIUS; dx <= SSGI_BLUR_RADIUS; dx++)
        {
            float2 tap = uv + float2(float(dx), float(dy)) * texel;
            float d = gbuffer.Sample(tap).a;
            if (d <= 0.0) continue;                      // skip background taps
            // Depth-similarity weight: taps on a different surface fall off
            // sharply so the indirect term does not bleed across edges.
            float dd = abs(d - center_depth);
            float w = exp2(-dd * 8.0);
            sum  += gi_tex.Sample(tap).rgb * w;
            wsum += w;
        }
    }
    float3 gi = wsum > 0.0 ? sum / wsum : gi_tex.Sample(uv).rgb;
    return float4(gi * params.intensity, 1.0);
}

#else
#error "ssgi.slang: define SSGI_GATHER or SSGI_COMPOSITE"
#endif
