// Bloom chain: prefilter, downsample, upsample. One fragment per compile,
// selected by a define so each variant declares exactly the resources it binds
// (Metal and DXIL indices are assigned in declaration order, so an unused
// declaration would shift the live ones):
//
//   BLOOM_PREFILTER  - threshold the scene into bloom mip 0, Karis-weighted.
//   BLOOM_DOWNSAMPLE - 13-tap reduce of the previous mip into the next.
//   BLOOM_UPSAMPLE   - 9-tap tent expand, additively blended onto the coarser
//                      mip by the pipeline blend state.
//
// Pairs with `fullscreen_vertex` in fullscreen.slang.

{POST_COMMON}

[[vk::binding(0, 0)]] Sampler2D<float4> src;

#ifdef BLOOM_PREFILTER
// Layout must match `PostProcessParams` in render_types.rs (24 B); the
// prefilter reads only the threshold, knee and exposure.
struct PostProcessParams
{
    float bloom_intensity;
    float bloom_threshold;
    float bloom_knee;
    float exposure;
    float vignette;
    float lut_strength;
};

[[vk::push_constant]]
ConstantBuffer<PostProcessParams> post;
#endif

float bloom_luma(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

// Karis weight: attenuate bright 2x2 groups so a single firefly does not
// dominate the downsample. Applied only on the first (prefilter) downsample.
float karis_weight(float3 c)
{
    return 1.0 / (1.0 + bloom_luma(c));
}

// 13-tap downsample (Jimenez/CoD). When `karis` is set, the 13 taps are grouped
// into five overlapping 2x2 boxes and luma-weighted.
float3 downsample_13(float2 uv, float2 texel, bool karis)
{
    float3 a = src.Sample(uv + texel * float2(-2.0, -2.0)).rgb;
    float3 b = src.Sample(uv + texel * float2( 0.0, -2.0)).rgb;
    float3 c = src.Sample(uv + texel * float2( 2.0, -2.0)).rgb;
    float3 d = src.Sample(uv + texel * float2(-2.0,  0.0)).rgb;
    float3 e = src.Sample(uv + texel * float2( 0.0,  0.0)).rgb;
    float3 f = src.Sample(uv + texel * float2( 2.0,  0.0)).rgb;
    float3 g = src.Sample(uv + texel * float2(-2.0,  2.0)).rgb;
    float3 h = src.Sample(uv + texel * float2( 0.0,  2.0)).rgb;
    float3 i = src.Sample(uv + texel * float2( 2.0,  2.0)).rgb;
    float3 j = src.Sample(uv + texel * float2(-1.0, -1.0)).rgb;
    float3 k = src.Sample(uv + texel * float2( 1.0, -1.0)).rgb;
    float3 l = src.Sample(uv + texel * float2(-1.0,  1.0)).rgb;
    float3 m = src.Sample(uv + texel * float2( 1.0,  1.0)).rgb;

    if (karis)
    {
        float3 g0 = (a + b + d + e) * (0.125 / 4.0);
        float3 g1 = (b + c + e + f) * (0.125 / 4.0);
        float3 g2 = (d + e + g + h) * (0.125 / 4.0);
        float3 g3 = (e + f + h + i) * (0.125 / 4.0);
        float3 g4 = (j + k + l + m) * (0.5 / 4.0);
        g0 *= karis_weight(g0);
        g1 *= karis_weight(g1);
        g2 *= karis_weight(g2);
        g3 *= karis_weight(g3);
        g4 *= karis_weight(g4);
        return g0 + g1 + g2 + g3 + g4;
    }

    float3 result = e * 0.125;
    result += (a + c + g + i) * 0.03125;
    result += (b + d + f + h) * 0.0625;
    result += (j + k + l + m) * 0.125;
    return result;
}

#if defined(BLOOM_PREFILTER)

// Upper bound on a single bloom source sample. IBL skyboxes can be many orders
// of magnitude brighter than surface shading (e.g. an EV24 HDR); without a
// clamp one ultra-bright texel blurs out and washes the whole frame white.
static const float BLOOM_CLAMP = 16.0;

// Quadratic soft-knee threshold (CoD/Unity): pixels above `threshold`
// contribute fully, pixels within `knee` below it ramp in smoothly.
float3 prefilter_threshold(float3 c, float threshold, float knee)
{
    float br = max(c.r, max(c.g, c.b));
    float kn = max(knee, 1e-4);
    float soft = clamp(br - threshold + kn, 0.0, 2.0 * kn);
    soft = (soft * soft) / (4.0 * kn);
    float contrib = max(soft, br - threshold) / max(br, 1e-4);
    return c * max(contrib, 0.0);
}

[shader("fragment")]
float4 bloom_prefilter_fragment([[vk::location(0)]] float2 uv : TEXCOORD0) : SV_Target
{
    float2 texel = 1.0 / combined_size(src);
    float3 c = downsample_13(uv, texel, true);
    // Exposure applies before the threshold so the soft-knee cut tracks the
    // exposed scene the composite pass tonemaps, not the raw HDR radiance.
    c *= post.exposure;
    c = min(c, float3(BLOOM_CLAMP));
    c = prefilter_threshold(c, post.bloom_threshold, post.bloom_knee);
    return float4(c, 1.0);
}

#elif defined(BLOOM_DOWNSAMPLE)

[shader("fragment")]
float4 bloom_downsample_fragment([[vk::location(0)]] float2 uv : TEXCOORD0) : SV_Target
{
    float2 texel = 1.0 / combined_size(src);
    return float4(downsample_13(uv, texel, false), 1.0);
}

#elif defined(BLOOM_UPSAMPLE)

// 9-tap tent upsample filter (weights 1 2 1 / 2 4 2 / 1 2 1, /16).
float3 upsample_tent(float2 uv, float2 texel)
{
    float3 sum = float3(0.0);
    sum += src.Sample(uv + texel * float2(-1.0, -1.0)).rgb * 1.0;
    sum += src.Sample(uv + texel * float2( 0.0, -1.0)).rgb * 2.0;
    sum += src.Sample(uv + texel * float2( 1.0, -1.0)).rgb * 1.0;
    sum += src.Sample(uv + texel * float2(-1.0,  0.0)).rgb * 2.0;
    sum += src.Sample(uv + texel * float2( 0.0,  0.0)).rgb * 4.0;
    sum += src.Sample(uv + texel * float2( 1.0,  0.0)).rgb * 2.0;
    sum += src.Sample(uv + texel * float2(-1.0,  1.0)).rgb * 1.0;
    sum += src.Sample(uv + texel * float2( 0.0,  1.0)).rgb * 2.0;
    sum += src.Sample(uv + texel * float2( 1.0,  1.0)).rgb * 1.0;
    return sum * (1.0 / 16.0);
}

[shader("fragment")]
float4 bloom_upsample_fragment([[vk::location(0)]] float2 uv : TEXCOORD0) : SV_Target
{
    float2 texel = 1.0 / combined_size(src);
    return float4(upsample_tent(uv, texel), 1.0);
}

#else
#error "bloom.slang: define BLOOM_PREFILTER, BLOOM_DOWNSAMPLE, or BLOOM_UPSAMPLE"
#endif
