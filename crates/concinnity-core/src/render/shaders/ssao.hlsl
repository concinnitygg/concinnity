// Screen-space ambient occlusion (GTAO): a compact copy of the depth it reads,
// the horizon-search kernel, and the depth-aware blur that cleans up its noise.
// One fragment per compile, selected by a define so each variant declares
// exactly the resources it binds:
//
//   SSAO_DEPTH  - the G-buffer's linear depth alone in a half-precision
//                 channel, so each of the kernel's and blur's many depth taps
//                 fetches two bytes instead of eight.
//   SSAO_KERNEL - horizon search over that depth -> raw occlusion.
//   SSAO_BLUR   - depth-aware 5x5 box blur of that raw occlusion.
//
// The G-buffer pre-pass holds rgb = unit view normal, a = linear view depth,
// with 0 for background; the copy keeps the same convention. Pairs with
// `fullscreen_vertex` in fullscreen.hlsl; each fragment takes the vertex UV
// first, since D3D links a pixel shader only to a prefix of the vertex outputs.
//
// Each source is a `Texture2D` and a `SamplerState`: Vulkan binds a variant's n
// textures at 0..n-1 and their samplers at n..2n-1. The register numbers are
// the Metal indices (see concinnity-shader's `metal_bindings`) and the D3D root
// signature's slots alike.

{TEXTURE_SIZE}

#if defined(SSAO_DEPTH)

[[vk::binding(0, 0)]] Texture2D<float4> gbuffer : register(t0);
[[vk::binding(1, 0)]] SamplerState gbuffer_samp : register(s0);

[shader("pixel")]
float ssao_depth_fragment(
    [[vk::location(0)]] float2 uv : TEXCOORD0,
    float4 pixel : SV_Position) : SV_Target
{
    return gbuffer.Load(int3(int2(pixel.xy), 0)).a;
}

#elif defined(SSAO_KERNEL)

// Layout matches `SsaoParams` in render_types.rs (16 B).
struct SsaoParams
{
    float radius;
    float intensity;
    float tan_half_fov_y;
    float aspect;
};

[[vk::binding(0, 0)]] Texture2D<float4> gbuffer : register(t0);
[[vk::binding(2, 0)]] SamplerState gbuffer_samp : register(s0);
[[vk::binding(1, 0)]] Texture2D<float4> depth : register(t1);
[[vk::binding(3, 0)]] SamplerState depth_samp : register(s1);

[[vk::push_constant]] ConstantBuffer<SsaoParams> params : register(b0);

static const int   SSAO_SLICES  = 3;
static const int   SSAO_STEPS   = 6;
static const float SSAO_PI      = 3.14159265359;
static const float SSAO_HALF_PI = 1.57079632679;
// Cap on the kernel's UV footprint so geometry right in front of the camera
// does not blow the search radius out to most of the screen.
static const float SSAO_MAX_UV  = 0.2;
// cos and sin of the angle between consecutive slices, pi / SSAO_SLICES.
static const float2 SSAO_SLICE_STEP = float2(cos(SSAO_PI / float(SSAO_SLICES)),
                                             sin(SSAO_PI / float(SSAO_SLICES)));

// acos to within 0.0068 rad (XeGTAO's first-order fit), for a fraction of the
// cost. Only angles feeding the arc integral go through it.
float ssao_fast_acos(float x)
{
    float a = abs(x);
    float r = (-0.156583 * a + SSAO_HALF_PI) * sqrt(1.0 - a);
    return x >= 0.0 ? r : SSAO_PI - r;
}

// A view-space position from a UV and its linear depth: xy = (uv * scale +
// bias) * depth, z = -depth.
struct SsaoProjection
{
    float2 scale;
    float2 bias;
};

SsaoProjection ssao_projection()
{
    float2 half_extent = float2(params.tan_half_fov_y * params.aspect, params.tan_half_fov_y);
    SsaoProjection proj;
    proj.scale = float2(2.0, -2.0) * half_extent;
    proj.bias = float2(-1.0, 1.0) * half_extent;
    return proj;
}

float3 ssao_view_pos(SsaoProjection proj, float2 uv, float d)
{
    return float3((uv * proj.scale + proj.bias) * d, -d);
}

// Fold the tap at `uv` into a slice side's horizon cosine. The tap's pull
// toward its own cosine fades linearly to nothing at the search radius;
// background taps (depth 0) leave the horizon where it was.
float ssao_horizon(float cos_h, SsaoProjection proj, float2 uv, float3 p, float3 v,
                   float falloff)
{
    float d = depth.SampleLevel(depth_samp, uv, 0.0).r;
    float3 s = ssao_view_pos(proj, uv, d) - p;
    float len_sq = dot(s, s);
    float inv_len = rsqrt(max(len_sq, 1e-10));
    float weight = d > 0.0 ? saturate(len_sq * inv_len * falloff + 1.0) : 0.0;
    float cos_s = dot(s, v) * inv_len;
    return cos_h + weight * (max(cos_h, cos_s) - cos_h);
}

[shader("pixel")]
float ssao_kernel_fragment(
    [[vk::location(0)]] float2 uv : TEXCOORD0,
    float4 pixel : SV_Position) : SV_Target
{
    float4 c = gbuffer.SampleLevel(gbuffer_samp, uv, 0.0);
    float d = c.a;
    if (d <= 0.0)
    {
        return 1.0;                        // background - no geometry, fully lit
    }

    SsaoProjection proj = ssao_projection();
    float3 n_vec = normalize(c.xyz);
    float3 p = ssao_view_pos(proj, uv, d);
    float3 v = normalize(-p);              // p is in view space; camera is origin

    // UV-space radius of the world-space search radius at this depth. The
    // viewport spans 2*tan_half_fov*depth view units vertically.
    float radius_uv = params.radius / max(2.0 * params.tan_half_fov_y * d, 1e-4);
    radius_uv = min(radius_uv, SSAO_MAX_UV);
    float falloff = -1.0 / max(params.radius, 1e-4);

    // Interleaved gradient noise: a per-pixel slice rotation + step jitter that
    // trades banding for high-frequency noise the blur pass then cleans up.
    float ign = frac(52.9829189 * frac(dot(pixel.xy, float2(0.06711056, 0.00583715))));

    float2 dir;
    sincos(ign * (SSAO_PI / float(SSAO_SLICES)), dir.y, dir.x);
    float visibility = 0.0;
    for (int s = 0; s < SSAO_SLICES; s++)
    {
        // Slice plane: spanned by v and the screen direction lifted to view
        // space. The projected surface normal and both horizons are measured
        // inside this plane.
        float3 plane_n = normalize(cross(float3(dir, 0.0), v));
        float3 tangent = cross(plane_n, v);
        float3 proj_n  = n_vec - plane_n * dot(n_vec, plane_n);
        float proj_len_sq = dot(proj_n, proj_n);
        float2 slice_dir = dir;
        dir = float2(dir.x * SSAO_SLICE_STEP.x - dir.y * SSAO_SLICE_STEP.y,
                     dir.x * SSAO_SLICE_STEP.y + dir.y * SSAO_SLICE_STEP.x);
        if (proj_len_sq < 1e-8)
        {
            continue;
        }
        float inv_proj_len = rsqrt(proj_len_sq);
        float proj_len = proj_len_sq * inv_proj_len;
        float cos_n = clamp(dot(proj_n, v) * inv_proj_len, -1.0, 1.0);
        float sign_n = dot(proj_n, tangent) < 0.0 ? -1.0 : 1.0;
        float n = sign_n * ssao_fast_acos(cos_n);
        float sin_n = sign_n * sqrt(saturate(1.0 - cos_n * cos_n));

        // Horizon search: march both screen directions, keeping the widest
        // horizon cosine.
        float cos_plus  = -1.0;
        float cos_minus = -1.0;
        for (int step = 1; step <= SSAO_STEPS; step++)
        {
            float t = (float(step) - 0.5 + ign) * (1.0 / float(SSAO_STEPS));
            float2 off = slice_dir * (radius_uv * t);
            cos_plus  = ssao_horizon(cos_plus,  proj, uv + off, p, v, falloff);
            cos_minus = ssao_horizon(cos_minus, proj, uv - off, p, v, falloff);
        }

        // Horizon angles, clamped into the hemisphere around the projected
        // normal, then the GTAO cosine-weighted arc integral for the slice.
        float h1 = -ssao_fast_acos(clamp(cos_minus, -1.0, 1.0));
        float h2 =  ssao_fast_acos(clamp(cos_plus,  -1.0, 1.0));
        h1 = n + max(h1 - n, -SSAO_HALF_PI);
        h2 = n + min(h2 - n,  SSAO_HALF_PI);
        float a1 = -cos(2.0 * h1 - n) + cos_n + 2.0 * h1 * sin_n;
        float a2 = -cos(2.0 * h2 - n) + cos_n + 2.0 * h2 * sin_n;
        visibility += proj_len * (a1 + a2);
    }

    // The arc integral's 1/4, folded with the slice average.
    visibility = saturate(visibility * (0.25 / float(SSAO_SLICES)));
    // `intensity` sharpens the contact darkening; 1.0 is the integrated amount.
    return pow(visibility, max(params.intensity, 0.0));
}

#elif defined(SSAO_BLUR)

[[vk::binding(0, 0)]] Texture2D<float4> ao_raw : register(t0);
[[vk::binding(2, 0)]] SamplerState ao_raw_samp : register(s0);
[[vk::binding(1, 0)]] Texture2D<float4> depth : register(t1);
[[vk::binding(3, 0)]] SamplerState depth_samp : register(s1);

// Depth-aware 5x5 box blur. Weighting each tap by view-depth similarity keeps
// the noisy GTAO output from bleeding occlusion across silhouette edges.
//
// The window is read as a 3x3 grid of 2x2 gathers covering 6x6 texels, the
// last row and column masked out, so 25 taps cost 9 fetches per source. A
// gather returns its footprint as (x, y, z, w) = texels (0, 1), (1, 1), (1, 0),
// (0, 0) of that footprint.
[shader("pixel")]
float ssao_blur_fragment([[vk::location(0)]] float2 uv : TEXCOORD0) : SV_Target
{
    float center_depth = depth.SampleLevel(depth_samp, uv, 0.0).r;
    if (center_depth <= 0.0)
    {
        return 1.0;
    }
    float2 texel = 1.0 / texture_size(ao_raw);
    // exp(-|d - center| * 8 / center), as one exp2 per tap.
    float falloff = -8.0 * 1.44269504 / max(center_depth, 1e-3);
    float sum = 0.0;
    float wsum = 0.0;
    [unroll]
    for (int by = 0; by < 3; by++)
    {
        [unroll]
        for (int bx = 0; bx < 3; bx++)
        {
            // The shared corner of texels -2 + 2 * b and -1 + 2 * b on each
            // axis, which is what a gather there reads.
            float2 corner = uv + (float2(bx, by) * 2.0 - 1.5) * texel;
            float4 d = depth.GatherRed(depth_samp, corner);
            float4 ao = ao_raw.GatherRed(ao_raw_samp, corner);
            float4 keep = float4(by < 2, bx < 2 && by < 2, bx < 2, 1.0);
            // Background taps (d <= 0) drop out. The center weighs 1, so the
            // sum is never empty.
            float4 w = select(d > 0.0, exp2(abs(d - center_depth) * falloff), 0.0) * keep;
            sum  += dot(w, ao);
            wsum += dot(w, 1.0);
        }
    }
    return sum / wsum;
}

#else
#error "ssao.hlsl: define SSAO_DEPTH, SSAO_KERNEL or SSAO_BLUR"
#endif
