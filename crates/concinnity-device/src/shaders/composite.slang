// Final composite: bloom onto the exposed HDR scene, then either the SDR
// chain (ACES tonemap, gamma 2.2, FXAA, colour-grading LUT) or the HDR EDR
// passthrough, plus the vignette, the scene-transition fade, and the G-buffer
// channel views. Single source for every backend; pairs with
// `fullscreen_vertex` in fullscreen.slang.
//
// Vulkan gets set 0 bindings 0-5 as combined image samplers plus the push
// constant; Metal gets texture(0..5) + sampler(0..5) and the params at
// buffer(0). Textures 3-5 are declared but sampled only while a channel view
// is active, which is also the only time the host binds them.

{POST_COMMON}

// Layout matches `CompositeParams` in render_types.rs (48 B): the
// `PostProcessParams` fields followed by the fade, the channel-view selector,
// and the camera far plane.
struct CompositeParams
{
    float bloom_intensity;
    float bloom_threshold;
    float bloom_knee;
    float exposure;
    float vignette;
    float lut_strength;
    // 0.0 = SDR path (ACES + gamma 2.2 + FXAA + ColorLut into a display-referred
    // target). 1.0 = HDR EDR path (linear extended-range values; ACES / gamma /
    // FXAA / LUT are all skipped because they assume a display-referred range).
    float hdr_output;
    // Inside the HDR branch: 0.0 = scRGB-linear passthrough, 1.0 = PQ encode
    // (SMPTE ST 2084) for HDR10 panels.
    float pq_output;
    // 1.0 = run the FXAA edge filter on the SDR path; 0.0 = skip it (the Off
    // anti-aliasing mode). Ignored on the HDR path, which never runs FXAA.
    float fxaa;
    // Scene-transition fade to black in [0, 1]. 0 leaves the frame untouched.
    float fade;
    // G-buffer channel view selector (ViewMode discriminant): 0 composites the
    // scene; 3 = normals, 4 = roughness, 5 = occlusion, 6 = depth.
    uint view_mode;
    // Camera far plane, normalizing the depth channel view.
    float far_plane;
};

[[vk::binding(0, 0)]] Sampler2D<float4> hdr_tex;
// Bloom mip 0 (half-res). Always bound - when bloom is disabled the sample is
// skipped, so an unwritten target is never read.
[[vk::binding(1, 0)]] Sampler2D<float4> bloom_tex;
// 3D colour-grading LUT. Always bound - a 2x2x2 identity LUT stands in when the
// world declares no ColorLut, so the grade is a no-op at any lut_strength.
[[vk::binding(2, 0)]] Sampler3D<float4> lut_tex;
// G-buffer channel sources for the debug view modes.
[[vk::binding(3, 0)]] Sampler2D<float4> gbuf_nd_tex;
[[vk::binding(4, 0)]] Sampler2D<float4> gbuf_rough_tex;
[[vk::binding(5, 0)]] Sampler2D<float4> ao_tex;

[[vk::push_constant]]
ConstantBuffer<CompositeParams> post;

// SDR reference white in cd/m2 (nits). BT.2408 recommends 203 nits as the HDR
// mixing reference; it keeps SDR content from looking dim alongside HDR
// highlights and matches the value mainstream HDR pipelines use as the
// linear-to-PQ mapping reference.
static const float PQ_SDR_REFERENCE_NITS = 203.0;

// Width of the colour-grading LUT along one axis.
//
// The same slangc GetDimensions mis-lowering `combined_size` works around for
// Sampler2D applies to a top-level Sampler3D, so this is the three-dimensional
// twin of that pair rather than a plain call.
void combined_dims_3d(Sampler3D<float4> s, out uint w, out uint h, out uint d)
{
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "$0.GetDimensions($2, $3, $4)";
    default:
        s.GetDimensions(w, h, d);
    }
}

float lut_axis_size(Sampler3D<float4> s)
{
    __target_switch
    {
    case metal:
        __intrinsic_asm "float($0.get_width())";
    default:
        uint w, h, d;
        combined_dims_3d(s, w, h, d);
        return float(w);
    }
}

// PQ inverse-EOTF (PQ OETF / encode). Constants from SMPTE ST 2084 / ITU-R
// BT.2100. Maps absolute luminance in cd/m2 (capped at 10000) onto the
// perceptually-quantized signal in [0, 1] the HDR10 panel expects.
float3 pq_encode(float3 l_nits)
{
    const float m1 = 0.1593017578125;     // = 2610 / 16384
    const float m2 = 78.84375;            // = 2523 / 4096 * 128
    const float c1 = 0.8359375;           // = 3424 / 4096
    const float c2 = 18.8515625;          // = 2413 / 4096 * 32
    const float c3 = 18.6875;             // = 2392 / 4096 * 32
    float3 l_n = clamp(l_nits * (1.0 / 10000.0), float3(0.0), float3(1.0));
    float3 lm1 = pow(l_n, float3(m1));
    return pow((c1 + c2 * lm1) / (1.0 + c3 * lm1), float3(m2));
}

// Narkowicz 2015 ACES fit - closed-form approximation of the ACES RRT+ODT.
float3 aces_narkowicz(float3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// FXAA luma weighting on a display-referred sRGB pixel.
float fxaa_luma(float3 rgb)
{
    return dot(rgb, float3(0.299, 0.587, 0.114));
}

// Scene colour = exposed HDR resolve + bloom. Exposure scales the HDR tap
// only - the bloom mip already carries the exposure applied in the prefilter.
// The bloom sample is skipped when bloom is disabled so an uninitialised bloom
// target is never read.
float3 scene_sample(float2 uv)
{
    float3 c = hdr_tex.Sample(uv).rgb * post.exposure;
    if (post.bloom_intensity > 0.0)
    {
        c += bloom_tex.Sample(uv).rgb * post.bloom_intensity;
    }
    return c;
}

// Scene sample -> ACES tonemap -> gamma 2.2 encode.
float3 tonemap(float2 uv)
{
    return pow(aces_narkowicz(scene_sample(uv)), float3(1.0 / 2.2));
}

// Smooth radial corner darkening. `strength` 0 disables it; 1 fully darkens the
// corners. The squared-distance falloff keeps the centre untouched.
float vignette_factor(float2 uv)
{
    float2 d = uv - 0.5;
    float dist = dot(d, d) * 2.0;
    return 1.0 - post.vignette * smoothstep(0.25, 1.0, dist);
}

// Scene-transition fade as a multiplier: 1 = un-faded, 0 = fully black.
float fade_scale()
{
    return 1.0 - saturate(post.fade);
}

// Sample the 3D colour-grading LUT with the tonemapped, display-referred sRGB
// colour. The half-texel correction maps an input of 0 / 1 to the centres of
// the first / last texels so trilinear filtering stays accurate edge to edge; a
// 2x2x2 identity LUT then reproduces the input exactly.
float3 apply_lut(float3 c)
{
    float n = lut_axis_size(lut_tex);
    float3 uvw = saturate(c) * ((n - 1.0) / n) + (0.5 / n);
    return lut_tex.Sample(uvw).rgb;
}

// Blend the LUT-graded colour over the input by `lut_strength`. Grading the
// display-referred LDR result keeps the LUT independent of exposure / tonemap.
float3 grade(float3 c)
{
    return lerp(c, apply_lut(c), post.lut_strength);
}

// One prepass channel, visualized in place of the composited scene.
float3 channel_view(float2 uv)
{
    float4 nd = gbuf_nd_tex.Sample(uv);
    if (post.view_mode == 3u)
    {
        // View-space normal; cleared alpha 0 marks "no geometry".
        return (nd.a > 0.0) ? nd.xyz * 0.5 + 0.5 : float3(0.0);
    }
    if (post.view_mode == 4u)
    {
        return float3(gbuf_rough_tex.Sample(uv).r);
    }
    if (post.view_mode == 5u)
    {
        return float3(ao_tex.Sample(uv).r);
    }
    if (post.view_mode == 6u)
    {
        // Linear view depth over far; empty pixels read as the far plane.
        return (nd.a > 0.0) ? float3(saturate(nd.a / max(post.far_plane, 1e-3)))
                            : float3(1.0);
    }
    return float3(0.0);
}

[shader("fragment")]
float4 composite_fragment([[vk::location(0)]] float2 uv : TEXCOORD0) : SV_Target
{
    // Channel views replace the composited scene with one prepass channel; the
    // text overlay still draws after in this same pass.
    if (post.view_mode != 0u)
    {
        return float4(channel_view(uv), 1.0);
    }

    float2 inv_size = 1.0 / combined_size(hdr_tex);
    // The scene fade rides the vignette multiplier: both are plain scalars on
    // the output colour, so the HDR path fades in linear light ahead of the
    // optional PQ encode and the SDR path fades the display-referred result.
    float vig = vignette_factor(uv) * fade_scale();

    // HDR EDR output: skip ACES + gamma + FXAA + LUT. Two flavours, picked by
    // `pq_output`:
    //
    //   - scRGB linear: the compositor wants linear extended-range values where
    //     1.0 is SDR reference white and values above drive the panel headroom.
    //   - HDR10 PQ: the panel decodes via the PQ EOTF, so encode in-shader with
    //     SDR reference white at 203 nits per BT.2408.
    //
    // The vignette applies in linear space ahead of the optional PQ encode,
    // because it is a multiplicative falloff defined on luminance.
    if (post.hdr_output > 0.5)
    {
        float3 hdr_vig = scene_sample(uv) * vig;
        if (post.pq_output > 0.5)
        {
            return float4(pq_encode(hdr_vig * PQ_SDR_REFERENCE_NITS), 1.0);
        }
        return float4(hdr_vig, 1.0);
    }

    // Composite bloom onto the HDR scene, then ACES tonemap + gamma encode.
    float3 c = tonemap(uv);

    // FXAA is gated by post.fxaa (off for the Off anti-aliasing mode). When
    // disabled, grade + vignette the tonemapped centre directly and skip the
    // neighbour samples the edge filter would otherwise take.
    if (post.fxaa < 0.5)
    {
        return float4(grade(c) * vig, 1.0);
    }

    // FXAA 3.11-style edge detection on the encoded image. Each neighbour is
    // composited with bloom and remapped through the same tonemap + gamma so
    // luma compares stay consistent with the centre sample.
    float3 n = tonemap(uv + float2(0.0, -inv_size.y));
    float3 s = tonemap(uv + float2(0.0,  inv_size.y));
    float3 e = tonemap(uv + float2( inv_size.x, 0.0));
    float3 w = tonemap(uv + float2(-inv_size.x, 0.0));

    float l_c = fxaa_luma(c);
    float l_n = fxaa_luma(n);
    float l_s = fxaa_luma(s);
    float l_e = fxaa_luma(e);
    float l_w = fxaa_luma(w);

    float l_min = min(l_c, min(min(l_n, l_s), min(l_e, l_w)));
    float l_max = max(l_c, max(max(l_n, l_s), max(l_e, l_w)));
    float l_range = l_max - l_min;

    // Flat regions skip the blur entirely; they are still graded and vignetted.
    if (l_range < max(0.0312, l_max * 0.125))
    {
        return float4(grade(c) * vig, 1.0);
    }

    // Pick the dominant edge direction (horizontal vs vertical) and step half a
    // texel along the perpendicular for a 2-tap blur averaged with the centre.
    float horz_diff = abs(l_n + l_s - 2.0 * l_c) * 2.0 + abs(l_e + l_w - 2.0 * l_c);
    float vert_diff = abs(l_e + l_w - 2.0 * l_c) * 2.0 + abs(l_n + l_s - 2.0 * l_c);
    bool horizontal = horz_diff >= vert_diff;

    float2 step_dir = horizontal ? float2(0.0, inv_size.y) : float2(inv_size.x, 0.0);
    float3 a = tonemap(uv + step_dir * 0.5);
    float3 b = tonemap(uv - step_dir * 0.5);
    float3 blended = (c + a + b) * (1.0 / 3.0);

    // Grade the FXAA-resolved colour, then vignette last so it darkens the
    // final composited result.
    return float4(grade(blended) * vig, 1.0);
}
