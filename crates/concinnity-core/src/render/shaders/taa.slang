// Temporal anti-aliasing resolve: blend the current HDR frame with the
// reprojected, neighbourhood-clipped history buffer. Single source for every
// backend; pairs with `fullscreen_vertex` in fullscreen.slang.
//
// Resources are declared top-level rather than in a ParameterBlock: this pass
// is engine-internal (nothing outside the engine binds against its layout), so
// the declaration order is free to be the one both binding models want. Vulkan
// gets set 0 bindings 0-2 as combined image samplers plus a 4-byte push
// constant; Metal gets texture(0..2) + sampler(0..2) and the params at
// buffer(0).

struct TaaParams
{
    // 0 on the first frame and after a resize - history is then ignored.
    float history_valid;
};

[[vk::binding(0, 0)]] Sampler2D<float4> scene_tex;
[[vk::binding(1, 0)]] Sampler2D<float4> velocity_tex;
[[vk::binding(2, 0)]] Sampler2D<float4> history_tex;

[[vk::push_constant]]
ConstantBuffer<TaaParams> params;

// History blend weight. 0.9 keeps 90% of the accumulated history each frame -
// roughly a 10-frame exponential moving average once converged.
static const float TAA_BLEND = 0.9;

// Standard deviations of the neighbourhood the history is allowed to span.
// 1.0 is the Playdead/Karis default; lower tightens (less ghosting, more
// flicker), higher loosens.
static const float TAA_VARIANCE_GAMMA = 1.0;

// Finite ceiling every sampled colour is clamped to. Largest finite half - the
// scene/history targets are RGBA16Float, so an HDR specular that overflowed the
// target reads back as +Inf, and an uninitialised history texel on the first
// frame can read back as Inf or NaN. Either one would turn the variance
// statistics into NaN (Inf-Inf) and, once a NaN lands in the history, it feeds
// back every frame as a black screen.
static const float TAA_HDR_CLAMP = 65504.0;

{POST_COMMON}

// Scrub a non-finite sample: NaN -> 0, +Inf -> the HDR ceiling, -Inf/negative
// -> 0. NaN is removed first (NaN != NaN) so the result never depends on how
// clamp tie-breaks a NaN operand.
float3 taa_sanitize(float3 c)
{
    c = select(isnan(c), float3(0.0), c);
    return clamp(c, float3(0.0), float3(TAA_HDR_CLAMP));
}

// RGB <-> YCoCg. The neighbourhood clip box is built in YCoCg because its luma
// axis aligns with perceived error: the box is tighter and better-oriented than
// an RGB AABB, so a reprojected history ghosts less. The transform is linear,
// so it is safe on the linear-light HDR values here.
float3 rgb_to_ycocg(float3 c)
{
    return float3(
         0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
         0.5  * c.r            - 0.5  * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}

float3 ycocg_to_rgb(float3 c)
{
    float t = c.x - c.z;
    return float3(t + c.y, c.x + c.z, t - c.y);
}

// Clip the history sample to the neighbourhood box along the line toward the
// box centre (Karis 2014). Unlike a per-component clamp this preserves the
// colour's direction, so a clipped history shifts hue far less.
float3 clip_to_aabb(float3 bmin, float3 bmax, float3 hist)
{
    float3 center = 0.5 * (bmax + bmin);
    float3 extent = 0.5 * (bmax - bmin) + 1e-5;
    float3 v = hist - center;
    float3 a = abs(v) / extent;
    float ma = max(a.x, max(a.y, a.z));
    return (ma > 1.0) ? (center + v / ma) : hist;
}

[shader("fragment")]
float4 taa_fragment_main([[vk::location(0)]] float2 uv : TEXCOORD0) : SV_Target
{
    float2 texel = 1.0 / combined_size(scene_tex);
    float3 cur = taa_sanitize(scene_tex.Sample(uv).rgb);

    // 3x3 neighbourhood statistics in YCoCg. The reprojected history is clipped
    // to mean +/- gamma*stddev - a variance box, tighter and better-oriented
    // than a min/max AABB, so disocclusions and sub-pixel misses ghost less.
    // Every sample is sanitised first so a non-finite HDR texel cannot make the
    // moments (and therefore the box) NaN.
    float3 m1 = float3(0.0);
    float3 m2 = float3(0.0);
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            float3 s = taa_sanitize(scene_tex.Sample(uv + float2(dx, dy) * texel).rgb);
            float3 c = rgb_to_ycocg(s);
            m1 += c;
            m2 += c * c;
        }
    }
    float3 mean  = m1 / 9.0;
    float3 sigma = sqrt(max(m2 / 9.0 - mean * mean, float3(0.0)));
    float3 bmin  = mean - TAA_VARIANCE_GAMMA * sigma;
    float3 bmax  = mean + TAA_VARIANCE_GAMMA * sigma;

    // The velocity pre-pass stored each surface's screen-space motion as the
    // offset that maps a current-frame UV onto its previous-frame UV. This
    // captures camera motion, moving props, and skinned deformation alike.
    float2 motion  = velocity_tex.Sample(uv).rg;
    float2 prev_uv = uv + motion;
    bool on_screen = all(prev_uv >= float2(0.0)) && all(prev_uv <= float2(1.0));

    // Sanitise the history too: on the first frame it is an uninitialised
    // target, and a NaN read here would survive clip_to_aabb (a NaN fails the
    // ma > 1 test, so the unclipped NaN is returned) and poison every later
    // frame through the feedback.
    float3 hist = rgb_to_ycocg(taa_sanitize(history_tex.Sample(prev_uv).rgb));
    hist = clip_to_aabb(bmin, bmax, hist);

    // Accumulate only when there is valid, on-screen history; otherwise the
    // current frame passes straight through (first frame, resize, off-screen).
    float alpha = (params.history_valid > 0.5 && on_screen) ? TAA_BLEND : 0.0;
    return float4(lerp(cur, ycocg_to_rgb(hist), alpha), 1.0);
}
