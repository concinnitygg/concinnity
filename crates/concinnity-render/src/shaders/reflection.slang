// Roughness-aware reflection composite, split into two passes so the wide
// glossy blur runs at reduced resolution. One fragment per compile, selected by
// a define so each variant declares exactly the resources it binds (Metal and
// DXIL indices are assigned in declaration order, so an unused declaration
// would shift the live ones):
//
//   REFLECTION_BLUR      - reduced resolution: weight-averages the SSR / RT
//                          resolve target over a roughness-scaled cone into the
//                          blur target. This is the expensive part (up to 17
//                          taps), so running it at a fraction of the pixels is
//                          the saving.
//   REFLECTION_COMPOSITE - full resolution: lerps the FULL-RES sharp reflection
//                          against the upsampled blur by roughness, then
//                          composites over the scene. A near-mirror
//                          (roughness ~0) reads the sharp full-res tap so it
//                          stays razor-sharp; a rough surface reads the cheap
//                          upsampled blur, which is low-frequency anyway.
//
// Pairs with `fullscreen_vertex` in fullscreen.slang.

// Surfaces rougher than this get no reflection (the resolve already wrote
// weight 0); below it the blur cone ramps from a sharp mirror at 0 to
// REFL_BLUR_MAX. Locked to concinnity_core::gfx::ssr::REFLECTION_ROUGHNESS_CUT
// by unit test, so the SSR, RT, and composite passes can never disagree on it.
static const float REFLECTION_ROUGHNESS_CUT = 0.6;

#if defined(REFLECTION_BLUR)

// The resolve target (rgb = reflected radiance, a = Fresnel/gloss composite
// weight) and the G-buffer roughness the cone radius keys off.
[[vk::binding(0, 0)]] Sampler2D<float4> reflection;
[[vk::binding(1, 0)]] Sampler2D<float4> rough_tex;

// Largest blur-cone radius in UV at the cut. This is the composite blur cone, a
// separate quantity from the resolve's own in-tap gather radius.
static const float REFL_BLUR_MAX = 0.02;

// Two 8-tap rings (45-degree steps) at half and full radius approximate the
// widening glossy reflection cone.
static const float2 REFL_RING[8] = {
    float2( 1.0,         0.0       ), float2( 0.70710678,  0.70710678),
    float2( 0.0,         1.0       ), float2(-0.70710678,  0.70710678),
    float2(-1.0,         0.0       ), float2(-0.70710678, -0.70710678),
    float2( 0.0,        -1.0       ), float2( 0.70710678, -0.70710678),
};

[shader("fragment")]
float4 reflection_blur_fragment([[vk::location(0)]] float2 uv : TEXCOORD0) : SV_Target
{
    float4 c = reflection.Sample(uv);
    float roughness = rough_tex.Sample(uv).r;
    float radius = saturate(roughness / REFLECTION_ROUGHNESS_CUT) * REFL_BLUR_MAX;

    // Weight each tap by its own coverage so weightless (non-reflecting) taps
    // cannot drag their colour in; a reflective edge then fades smoothly.
    float3 sum_rw = c.rgb * c.a;
    float sum_w = c.a;
    float taps = 1.0;
    if (radius > 1e-5)
    {
        for (int ring = 0; ring < 2; ring++)
        {
            float rr = radius * (ring == 0 ? 0.5 : 1.0);
            for (int i = 0; i < 8; i++)
            {
                float4 t = reflection.Sample(uv + REFL_RING[i] * rr);
                sum_rw += t.rgb * t.a;
                sum_w  += t.a;
                taps   += 1.0;
            }
        }
    }
    float3 blurred = sum_w > 1e-4 ? sum_rw / sum_w : c.rgb;
    float weight = sum_w / taps;
    return float4(blurred, weight);
}

#elif defined(REFLECTION_COMPOSITE)

// reflection: the full-res resolve target (rgb = radiance, a = weight). scene:
// the base HDR scene. gbuffer: normal+depth, with .a = linear depth. rough_tex:
// the G-buffer roughness. blur_tex: the reduced-resolution blur from pass 1.
[[vk::binding(0, 0)]] Sampler2D<float4> reflection;
[[vk::binding(1, 0)]] Sampler2D<float4> scene;
[[vk::binding(2, 0)]] Sampler2D<float4> gbuffer;
[[vk::binding(3, 0)]] Sampler2D<float4> rough_tex;
[[vk::binding(4, 0)]] Sampler2D<float4> blur_tex;

[shader("fragment")]
float4 reflection_composite_fragment([[vk::location(0)]] float2 uv : TEXCOORD0) : SV_Target
{
    float3 base = scene.Sample(uv).rgb;
    float depth = gbuffer.Sample(uv).a;
    float4 c = reflection.Sample(uv);
    // Background, or a pixel that does not reflect (weight 0): keep the scene.
    // A tiny weight is still honoured so the composite is exact at the edges.
    if (depth <= 0.0 || c.a <= 0.0)
    {
        return float4(lerp(base, c.rgb, c.a), 1.0);
    }

    // 0 at a mirror -> use the sharp full-res tap; 1 at the cut -> use the cheap
    // upsampled blur. The blur is low-frequency, so the bilinear upsample is
    // visually free; only the sharp branch needs full-res detail.
    float t = saturate(rough_tex.Sample(uv).r / REFLECTION_ROUGHNESS_CUT);
    float4 b = blur_tex.Sample(uv);

    float3 reflected = lerp(c.rgb, b.rgb, t);
    float weight = lerp(c.a, b.a, t);
    return float4(lerp(base, reflected, weight), 1.0);
}

#else
#error "reflection.slang: define REFLECTION_BLUR or REFLECTION_COMPOSITE"
#endif
