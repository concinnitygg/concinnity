// Roughness-aware reflection composite, split into two passes so the wide
// glossy blur runs at reduced resolution. One fragment per compile, selected by
// a define so each variant declares exactly the resources it binds:
//
//   REFLECTION_BLUR      - reduced resolution: weight-averages the SSR / RT
//                          resolve target over a roughness-scaled cone into the
//                          blur target. This is the expensive part (up to 17
//                          taps), so running it at a fraction of the pixels is
//                          the saving.
//   REFLECTION_COMPOSITE - full resolution: lerps the sharp reflection against
//                          the upsampled blur by roughness, then composites
//                          over the scene. A near-mirror (roughness ~0) reads
//                          the sharp tap so it stays crisp; a rough surface
//                          reads the cheap upsampled blur, which is
//                          low-frequency anyway. When the resolve traced at a
//                          reduced resolution, the sharp tap is a depth- and
//                          normal-aware upsample so it cannot bleed across an
//                          edge.
//
// Pairs with `fullscreen_vertex` in fullscreen.hlsl.
//
// Each source is a `Texture2D` and a `SamplerState`: Vulkan binds a variant's n
// textures at 0..n-1 and their samplers at n..2n-1. The register numbers are
// the Metal indices (see concinnity-shader's `metal_bindings`) and the D3D root
// signature's slots alike.

{POST_COMMON}

{TEXTURE_SIZE}

// Surfaces rougher than this get no reflection (the resolve already wrote
// weight 0); below it the blur cone ramps from a sharp mirror at 0 to
// REFL_BLUR_MAX. Locked to concinnity_core::render::post::ssr::settings::REFLECTION_ROUGHNESS_CUT
// by unit test, so the SSR, RT, and composite passes can never disagree on it.
static const float REFLECTION_ROUGHNESS_CUT = 0.6;

#if defined(REFLECTION_BLUR)

// The resolve target (rgb = reflected radiance, a = Fresnel/gloss composite
// weight) and the G-buffer roughness the cone radius keys off.
[[vk::binding(0, 0)]] Texture2D<float4> reflection : register(t0);
[[vk::binding(2, 0)]] SamplerState reflection_samp : register(s0);
[[vk::binding(1, 0)]] Texture2D<float4> rough_tex : register(t1);
[[vk::binding(3, 0)]] SamplerState rough_tex_samp : register(s1);

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

[shader("pixel")]
float4 reflection_blur_fragment([[vk::location(0)]] float2 uv : TEXCOORD0) : SV_Target
{
    float4 c = reflection.Sample(reflection_samp, uv);
    float roughness = rough_tex.Sample(rough_tex_samp, uv).r;
    float radius = saturate(roughness / REFLECTION_ROUGHNESS_CUT) * REFL_BLUR_MAX;

    // Weight each tap by its own coverage so weightless (non-reflecting) taps
    // cannot drag their color in; a reflective edge then fades smoothly.
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
                float4 t = reflection.Sample(reflection_samp, uv + REFL_RING[i] * rr);
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

// reflection: the resolve target (rgb = radiance, a = weight), at render
// resolution or at the reduced ray-traced trace resolution. scene:
// the base HDR scene. gbuffer: normal+depth, with .a = linear depth. rough_tex:
// the G-buffer roughness. blur_tex: the reduced-resolution blur from pass 1.
[[vk::binding(0, 0)]] Texture2D<float4> reflection : register(t0);
[[vk::binding(5, 0)]] SamplerState reflection_samp : register(s0);
[[vk::binding(1, 0)]] Texture2D<float4> scene : register(t1);
[[vk::binding(6, 0)]] SamplerState scene_samp : register(s1);
[[vk::binding(2, 0)]] Texture2D<float4> gbuffer : register(t2);
[[vk::binding(7, 0)]] SamplerState gbuffer_samp : register(s2);
[[vk::binding(3, 0)]] Texture2D<float4> rough_tex : register(t3);
[[vk::binding(8, 0)]] SamplerState rough_tex_samp : register(s3);
[[vk::binding(4, 0)]] Texture2D<float4> blur_tex : register(t4);
[[vk::binding(9, 0)]] SamplerState blur_tex_samp : register(s4);

// How sharply the upsample rejects a trace texel whose source pixel sits at a
// different depth (relative) or faces a different way than this pixel.
static const float SOURCE_DEPTH_FALLOFF = 64.0;
static const float SOURCE_NORMAL_POWER = 8.0;

// The resolve target at this pixel. A direct tap when the resolve ran at full
// resolution; otherwise a joint-bilateral upsample of the four nearest resolve
// texels, each weighted by its bilinear footprint and by how closely the
// G-buffer pixel it traced from matches this pixel's depth and normal. The
// small floor on each weight degrades a pixel no tap matches to plain bilinear.
float4 reflection_at(float2 uv, float depth, float3 n)
{
    float2 low = texture_size(reflection);
    float2 full = texture_size(gbuffer);
    if (all(low == full))
    {
        return reflection.Sample(reflection_samp, uv);
    }
    float2 p = uv * low - 0.5;
    float2 base = floor(p);
    float2 f = p - base;
    float3 sum_rw = (float3)(0.0);
    float sum_aw = 0.0;
    float sum_w = 0.0;
    for (int j = 0; j < 2; j++)
    {
        for (int i = 0; i < 2; i++)
        {
            float2 o = float2(float(i), float(j));
            float2 tap = (base + o + 0.5) / low;
            float2 axis = lerp(1.0 - f, f, o);
            float4 g = gbuffer.Sample(gbuffer_samp, reflection_source_uv(tap, full));
            float similarity = 0.0;
            if (g.a > 0.0)
            {
                float dz = abs(g.a - depth) / max(depth, 1e-3);
                float facing = saturate(dot(normalize(g.xyz), n));
                similarity = exp2(-dz * SOURCE_DEPTH_FALLOFF) * pow(facing, SOURCE_NORMAL_POWER);
            }
            float w = axis.x * axis.y * (similarity + 1e-4);
            float4 c = reflection.Sample(reflection_samp, tap);
            sum_rw += c.rgb * c.a * w;
            sum_aw += c.a * w;
            sum_w += w;
        }
    }
    float3 radiance = sum_aw > 1e-6 ? sum_rw / sum_aw : (float3)(0.0);
    return float4(radiance, sum_w > 0.0 ? sum_aw / sum_w : 0.0);
}

[shader("pixel")]
float4 reflection_composite_fragment([[vk::location(0)]] float2 uv : TEXCOORD0) : SV_Target
{
    float3 base = scene.Sample(scene_samp, uv).rgb;
    float4 g = gbuffer.Sample(gbuffer_samp, uv);
    float depth = g.a;
    float4 c = depth > 0.0 ? reflection_at(uv, depth, normalize(g.xyz)) : float4(base, 0.0);
    // Background, or a pixel that does not reflect (weight 0): keep the scene.
    // A tiny weight is still honored so the composite is exact at the edges.
    if (depth <= 0.0 || c.a <= 0.0)
    {
        return float4(lerp(base, c.rgb, c.a), 1.0);
    }

    // 0 at a mirror -> use the sharp tap; 1 at the cut -> use the cheap
    // upsampled blur. The blur is low-frequency, so the bilinear upsample is
    // visually free; only the sharp branch needs full-res detail.
    float t = saturate(rough_tex.Sample(rough_tex_samp, uv).r / REFLECTION_ROUGHNESS_CUT);
    float4 b = blur_tex.Sample(blur_tex_samp, uv);

    float3 reflected = lerp(c.rgb, b.rgb, t);
    float weight = lerp(c.a, b.a, t);
    return float4(lerp(base, reflected, weight), 1.0);
}

#else
#error "reflection.hlsl: define REFLECTION_BLUR or REFLECTION_COMPOSITE"
#endif
