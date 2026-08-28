// Screen-space ambient occlusion (GTAO): the horizon-search kernel and the
// depth-aware blur that cleans up its noise. One fragment per compile, selected
// by a define so each variant declares exactly the resources it binds (Metal
// and DXIL indices are assigned in declaration order, so an unused declaration
// would shift the live ones):
//
//   SSAO_KERNEL - horizon search over the G-buffer -> raw occlusion.
//   SSAO_BLUR   - depth-aware 5x5 box blur of that raw occlusion.
//
// Both read the unified G-buffer pre-pass (rgb = unit view normal, a = linear
// view depth) and write a single-channel occlusion target. Pairs with
// `fullscreen_vertex` in fullscreen.slang.

{POST_COMMON}

#if defined(SSAO_KERNEL)

// Layout matches `SsaoParams` in render_types.rs (16 B).
struct SsaoParams
{
    float radius;
    float intensity;
    float tan_half_fov_y;
    float aspect;
};

[[vk::binding(0, 0)]] Sampler2D<float4> gbuffer;

[[vk::push_constant]]
ConstantBuffer<SsaoParams> params;

static const int   SSAO_SLICES  = 3;
static const int   SSAO_STEPS   = 6;
static const float SSAO_PI      = 3.14159265359;
static const float SSAO_HALF_PI = 1.57079632679;
// Cap on the kernel's UV footprint so geometry right in front of the camera
// does not blow the search radius out to most of the screen.
static const float SSAO_MAX_UV  = 0.2;

// Rebuild a view-space position from a UV and its linear (view-space) depth.
float3 ssao_view_pos(float2 uv, float depth, float tan_y, float aspect)
{
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return float3(ndc.x * tan_y * aspect, ndc.y * tan_y, -1.0) * depth;
}

[shader("fragment")]
float ssao_kernel_fragment(
    [[vk::location(0)]] float2 uv : TEXCOORD0,
    float4 pixel : SV_Position) : SV_Target
{
    float4 c = gbuffer.Sample(uv);
    float depth = c.a;
    if (depth <= 0.0)
    {
        return 1.0;                        // background - no geometry, fully lit
    }

    float3 n_vec = normalize(c.xyz);
    float3 p = ssao_view_pos(uv, depth, params.tan_half_fov_y, params.aspect);
    float3 v = normalize(-p);              // p is in view space; camera is origin

    // UV-space radius of the world-space search radius at this depth. The
    // viewport spans 2*tan_half_fov*depth view units vertically.
    float radius_uv = params.radius / max(2.0 * params.tan_half_fov_y * depth, 1e-4);
    radius_uv = min(radius_uv, SSAO_MAX_UV);

    // Interleaved gradient noise: a per-pixel slice rotation + step jitter that
    // trades banding for high-frequency noise the blur pass then cleans up.
    float ign = frac(52.9829189 * frac(dot(pixel.xy, float2(0.06711056, 0.00583715))));

    float visibility = 0.0;
    for (int s = 0; s < SSAO_SLICES; s++)
    {
        float ang = (float(s) + ign) * (SSAO_PI / float(SSAO_SLICES));
        float2 dir = float2(cos(ang), sin(ang));

        // Slice plane: spanned by v and the screen direction lifted to view
        // space. The projected surface normal and both horizons are measured
        // inside this plane.
        float3 dir_vs  = normalize(float3(dir, 0.0));
        float3 plane_n = normalize(cross(dir_vs, v));
        float3 proj_n  = n_vec - plane_n * dot(n_vec, plane_n);
        float proj_len = length(proj_n);
        if (proj_len < 1e-4)
        {
            continue;
        }
        float3 tangent = cross(plane_n, v);
        float n = atan2(dot(proj_n, tangent), dot(proj_n, v));

        // Horizon search: march both screen directions, keeping the widest
        // horizon cosine, distance-attenuated so far occluders fade out.
        float cos_plus  = -1.0;
        float cos_minus = -1.0;
        for (int step = 1; step <= SSAO_STEPS; step++)
        {
            float t = (float(step) - 0.5 + ign) / float(SSAO_STEPS);
            float2 off = dir * radius_uv * t;

            float2 uvp = uv + off;
            float dp = gbuffer.Sample(uvp).a;
            if (dp > 0.0)
            {
                float3 sp = ssao_view_pos(uvp, dp, params.tan_half_fov_y, params.aspect) - p;
                float lp = length(sp);
                float fo = saturate(1.0 - lp / max(params.radius, 1e-4));
                cos_plus = lerp(cos_plus, max(cos_plus, dot(sp / max(lp, 1e-5), v)), fo);
            }
            float2 uvm = uv - off;
            float dm = gbuffer.Sample(uvm).a;
            if (dm > 0.0)
            {
                float3 sm = ssao_view_pos(uvm, dm, params.tan_half_fov_y, params.aspect) - p;
                float lm = length(sm);
                float fo = saturate(1.0 - lm / max(params.radius, 1e-4));
                cos_minus = lerp(cos_minus, max(cos_minus, dot(sm / max(lm, 1e-5), v)), fo);
            }
        }

        // Horizon angles, clamped into the hemisphere around the projected
        // normal, then the GTAO cosine-weighted arc integral for the slice.
        float h1 = -acos(clamp(cos_minus, -1.0, 1.0));
        float h2 =  acos(clamp(cos_plus,  -1.0, 1.0));
        h1 = n + max(h1 - n, -SSAO_HALF_PI);
        h2 = n + min(h2 - n,  SSAO_HALF_PI);
        float sin_n = sin(n);
        float cos_n = cos(n);
        float a1 = 0.25 * (-cos(2.0 * h1 - n) + cos_n + 2.0 * h1 * sin_n);
        float a2 = 0.25 * (-cos(2.0 * h2 - n) + cos_n + 2.0 * h2 * sin_n);
        visibility += proj_len * (a1 + a2);
    }

    visibility = saturate(visibility / float(SSAO_SLICES));
    // `intensity` sharpens the contact darkening; 1.0 is the integrated amount.
    return pow(visibility, max(params.intensity, 0.0));
}

#elif defined(SSAO_BLUR)

[[vk::binding(0, 0)]] Sampler2D<float4> ao_raw;
[[vk::binding(1, 0)]] Sampler2D<float4> gbuffer;

// Depth-aware 5x5 box blur. Weighting each tap by view-depth similarity keeps
// the noisy GTAO output from bleeding occlusion across silhouette edges.
[shader("fragment")]
float ssao_blur_fragment([[vk::location(0)]] float2 uv : TEXCOORD0) : SV_Target
{
    float2 texel = 1.0 / combined_size(ao_raw);
    float center_depth = gbuffer.Sample(uv).a;
    if (center_depth <= 0.0)
    {
        return 1.0;
    }
    float sum = 0.0;
    float wsum = 0.0;
    for (int y = -2; y <= 2; y++)
    {
        for (int x = -2; x <= 2; x++)
        {
            float2 tap = uv + float2(float(x), float(y)) * texel;
            float d = gbuffer.Sample(tap).a;
            // Depth-similarity weight; background taps (d <= 0) drop out.
            float w = (d > 0.0)
                ? exp(-abs(d - center_depth) * 8.0 / max(center_depth, 1e-3))
                : 0.0;
            sum  += ao_raw.Sample(tap).r * w;
            wsum += w;
        }
    }
    return (wsum > 1e-4) ? (sum / wsum) : ao_raw.Sample(uv).r;
}

#else
#error "ssao.slang: define SSAO_KERNEL or SSAO_BLUR"
#endif
