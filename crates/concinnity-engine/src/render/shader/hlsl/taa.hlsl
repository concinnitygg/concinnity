// Temporal anti-aliasing resolve: blend the current HDR frame with the
// reprojected, neighborhood-clipped history buffer. The history's alpha carries
// each pixel's thin-feature lock (see the resolve below). Single source for
// every backend; pairs with `fullscreen_vertex` in fullscreen.hlsl.
//
// Each of the six sources is a `Texture2D` and a `SamplerState`: Vulkan binds
// the textures at 0..5 and their samplers at 6..11. The register numbers are
// the Metal indices and the D3D root signature's slots alike: texture(0..5) +
// sampler(0..5) with the params at buffer(0). Only the history and the
// reactive mask are filtered; every other source is a point load.

struct TaaParams
{
    // 0 on the first frame and after a resize - history is then ignored.
    float history_valid;
    // The mask value past which the reactive mask removes no more history, or 0
    // when no mask was written this frame; the bound texture is then a stand-in
    // and is not read.
    float reactive_max;
    // Lock a thin feature loses each frame it is not seen: one over the jitter
    // cycle length, so a feature the jitter hits once per cycle stays locked.
    float lock_decay;
    // What the history is multiplied by to read at this frame's pre-exposure.
    float history_scale;
    // 1 when last frame's motion snapshot is bound, 0 when there is none; the
    // bound texture is then a stand-in, and no history is rejected by surface.
    float surface_valid;
};

[[vk::binding(0, 0)]] Texture2D<float4> scene_tex : register(t0);
[[vk::binding(6, 0)]] SamplerState scene_samp : register(s0);
[[vk::binding(1, 0)]] Texture2D<float4> velocity_tex : register(t1);
[[vk::binding(7, 0)]] SamplerState velocity_samp : register(s1);
[[vk::binding(2, 0)]] Texture2D<float4> history_tex : register(t2);
[[vk::binding(8, 0)]] SamplerState history_samp : register(s2);
// Where a particle or transparent surface tells the resolve to trust this
// frame: its history weight falls by the mask's value, up to `reactive_max`.
[[vk::binding(3, 0)]] Texture2D<float4> reactive_tex : register(t3);
[[vk::binding(9, 0)]] SamplerState reactive_samp : register(s3);
// The G-buffer: rgb = view normal, a = linear view depth (0 where there is no
// surface).
[[vk::binding(4, 0)]] Texture2D<float4> normal_depth_tex : register(t4);
[[vk::binding(10, 0)]] SamplerState normal_depth_samp : register(s4);
// Last frame's motion snapshot.
[[vk::binding(5, 0)]] Texture2D<float4> surface_tex : register(t5);
[[vk::binding(11, 0)]] SamplerState surface_samp : register(s5);

[[vk::push_constant]] ConstantBuffer<TaaParams> params : register(b0);

{TEXTURE_SIZE}

{MOTION_SURFACE}

// History blend weight. 0.9 keeps 90% of the accumulated history each frame -
// roughly a 10-frame exponential moving average once converged.
static const float TAA_BLEND = 0.9;

// Standard deviations of the neighborhood the history is allowed to span: x
// for a still pixel, y for one moving `TAA_VARIANCE_MOTION_PX` or more per
// frame, linear between. A wide box lets a slow high-contrast edge (a coastline
// or a planet's limb) settle its jitter; a fast one narrows to the
// Playdead/Karis default, which keeps a disocclusion from ghosting.
static const float2 TAA_VARIANCE_GAMMA = float2(2.5, 1.0);
static const float TAA_VARIANCE_MOTION_PX = 8.0;

// Luma ratio under which two neighborhood samples count as the same surface
// for the thin-feature test.
static const float TAA_LOCK_SIMILAR = 1.1;

// Largest screen-space motion, in pixels per frame, a thin feature may have and
// still be locked. A lock outlives the feature by up to one jitter cycle, so a
// feature this slow can leave a stale lock at most a couple of pixels behind.
static const float TAA_LOCK_MOTION_PX = 0.25;

// Largest luma standard deviation, as a fraction of the mean, the 3x3
// neighborhood may have for a lock to hold the clip off. A thin feature the
// jitter missed leaves the plain background it stood on; a textured or busy
// neighborhood (dense foliage, a blade sliding past another) is one where the
// missing feature may as well have moved, so the clip decides there.
static const float TAA_LOCK_FLAT = 0.25;

// Finite ceiling every sampled color is clamped to. Largest finite half - the
// scene/history targets are RGBA16Float, so an HDR specular that overflowed the
// target reads back as +Inf, and an uninitialized history texel on the first
// frame can read back as Inf or NaN. Either one would turn the variance
// statistics into NaN (Inf-Inf) and, once a NaN lands in the history, it feeds
// back every frame as a black screen.
static const float TAA_HDR_CLAMP = 65504.0;

// Scrub a non-finite sample: NaN -> 0, +Inf -> the HDR ceiling, -Inf/negative
// -> 0. NaN is removed first (NaN != NaN) so the result never depends on how
// clamp tie-breaks a NaN operand.
float3 taa_sanitize(float3 c)
{
    c = select(isnan(c), (float3)(0.0), c);
    return clamp(c, (float3)(0.0), (float3)(TAA_HDR_CLAMP));
}

// RGB <-> YCoCg. The neighborhood clip box is built in YCoCg because its luma
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

// Clip the history sample to the neighborhood box along the line toward the
// box center (Karis 2014). Unlike a per-component clamp this preserves the
// color's direction, so a clipped history shifts hue far less.
float3 clip_to_aabb(float3 bmin, float3 bmax, float3 hist)
{
    float3 center = 0.5 * (bmax + bmin);
    float3 extent = 0.5 * (bmax - bmin) + 1e-5;
    float3 v = hist - center;
    float3 a = abs(v) / extent;
    float ma = max(a.x, max(a.y, a.z));
    return (ma > 1.0) ? (center + v / ma) : hist;
}

// Whether two lumas are the same surface to the thin-feature test. A ratio, so
// the test reads the same at night as at noon; the tiny bias only keeps a pair
// of zeros from dividing by zero.
bool taa_similar(float a, float b)
{
    return (max(a, b) + 1e-12) < TAA_LOCK_SIMILAR * (min(a, b) + 1e-12);
}

// Whether the 3x3 lumas (row-major, center at 4) put a thin feature at the
// center: a ridge brighter or darker than every neighbor it differs from, with
// no uniform 2x2 block around it. A feature narrower than a pixel passes; an
// edge or a flat surface always has a uniform block.
bool taa_thin_feature(float luma[9])
{
    float nucleus = luma[4];
    uint similar = 1u << 4;
    float lo = 3.0e38;
    float hi = 0.0;
    for (uint i = 0; i < 9; ++i)
    {
        if (i == 4)
        {
            continue;
        }
        if (taa_similar(luma[i], nucleus))
        {
            similar |= 1u << i;
        }
        else
        {
            lo = min(lo, luma[i]);
            hi = max(hi, luma[i]);
        }
    }
    if (!(nucleus > hi || nucleus < lo))
    {
        return false;
    }
    // The four 2x2 blocks that contain the center.
    const uint blocks[4] = { 0x1Bu, 0x36u, 0xD8u, 0x1B0u };
    for (uint b = 0; b < 4; ++b)
    {
        if ((similar & blocks[b]) == blocks[b])
        {
            return false;
        }
    }
    return true;
}

// The history at `uv`, read through a 5-tap Catmull-Rom filter (the bicubic's
// four corner taps dropped, Jimenez 2016). A single bilinear tap softens the
// history a little on every frame the camera moves; the bicubic keeps it
// sharp. Its negative lobes can ring past the range of the taps, which the
// sanitize and the neighborhood clip after it absorb.
float4 taa_history_catmull_rom(float2 uv, float2 size)
{
    float2 pos = uv * size;
    float2 c1 = floor(pos - 0.5) + 0.5;
    float2 f = pos - c1;
    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);
    float2 w12 = w1 + w2;
    float2 tc0 = (c1 - 1.0) / size;
    float2 tc12 = (c1 + w2 / w12) / size;
    float2 tc3 = (c1 + 2.0) / size;
    float4 sum = history_tex.SampleLevel(history_samp, float2(tc12.x, tc0.y), 0.0) * (w12.x * w0.y)
        + history_tex.SampleLevel(history_samp, float2(tc0.x, tc12.y), 0.0) * (w0.x * w12.y)
        + history_tex.SampleLevel(history_samp, tc12, 0.0) * (w12.x * w12.y)
        + history_tex.SampleLevel(history_samp, float2(tc3.x, tc12.y), 0.0) * (w3.x * w12.y)
        + history_tex.SampleLevel(history_samp, float2(tc12.x, tc3.y), 0.0) * (w12.x * w3.y);
    float weight = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return sum / weight;
}

// How much of the history at this pixel's own previous position shows the
// surface it shows now: the four snapshot texels that position falls between,
// each weighted as a bilinear read would weight it, matched against the
// pixel's normal and own motion. A surface the motion carries out from behind
// another (what a moving object uncovers) finds the other surface there and
// keeps none of it, however thin the uncovered strip; a surface that was
// there keeps it all, and one only partly matched keeps part.
float taa_surface_history(int2 p, float3 normal, float2 own_motion, float2 size)
{
    // Where the pixel's center was, in the frame of the texel centers.
    float2 prev = float2(p) + own_motion;
    int2 base = int2(floor(prev));
    float2 f = prev - float2(base);
    int2 last = int2(size) - 1;
    float kept = 0.0;
    for (int y = 0; y <= 1; ++y)
    {
        for (int x = 0; x <= 1; ++x)
        {
            float w = (x == 0 ? 1.0 - f.x : f.x) * (y == 0 ? 1.0 - f.y : f.y);
            int2 q = clamp(base + int2(x, y), int2(0, 0), last);
            float4 snapshot = surface_tex.Load(int3(q, 0));
            kept += w * motion_surface_match(snapshot, normal, own_motion, size);
        }
    }
    return kept;
}

[shader("pixel")]
float4 taa_fragment_main(
    [[vk::location(0)]] float2 uv : TEXCOORD0,
    float4 pixel : SV_Position) : SV_Target
{
    float2 size = texture_size(scene_tex);
    int2 p = int2(pixel.xy);
    int2 last = int2(size) - 1;
    float3 cur = taa_sanitize(scene_tex.Load(int3(p, 0)).rgb);

    // 3x3 neighborhood statistics in YCoCg. The reprojected history is clipped
    // to mean +/- gamma*stddev - a variance box, tighter and better-oriented
    // than a min/max AABB, so disocclusions and sub-pixel misses ghost less.
    // Every sample is sanitized first so a non-finite HDR texel cannot make the
    // moments (and therefore the box) NaN. The same loop gathers a 1-2-1 tent
    // of the current frame, what a pixel shows where its history is dropped,
    // and the nearest surface, whose motion the history is read along.
    float3 m1 = (float3)(0.0);
    float3 m2 = (float3)(0.0);
    float3 tent = (float3)(0.0);
    float luma[9];
    float nearest = 3.0e38;
    int2 nearest_p = p;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            int2 q = clamp(p + int2(dx, dy), int2(0, 0), last);
            float3 s = taa_sanitize(scene_tex.Load(int3(q, 0)).rgb);
            float3 c = rgb_to_ycocg(s);
            m1 += c;
            m2 += c * c;
            luma[(dy + 1) * 3 + dx + 1] = c.x;
            tent += s * ((dx == 0 ? 2.0 : 1.0) * (dy == 0 ? 2.0 : 1.0));
            float depth = normal_depth_tex.Load(int3(q, 0)).a;
            if (depth > 0.0 && depth < nearest)
            {
                nearest = depth;
                nearest_p = q;
            }
        }
    }
    float3 mean  = m1 / 9.0;
    float3 sigma = sqrt(max(m2 / 9.0 - mean * mean, (float3)(0.0)));
    tent /= 16.0;

    // The velocity pre-pass stored each surface's screen-space motion as the
    // offset that maps a current-frame UV onto its previous-frame UV. This
    // captures camera motion, moving props, and skinned deformation alike.
    // The history is read along the nearest surface's motion, so an edge's
    // anti-aliased history moves with the object in front rather than staying
    // behind with the background.
    float2 motion = velocity_tex.Load(int3(nearest_p, 0)).rg;
    float2 motion_px = motion * size;
    float gamma = lerp(TAA_VARIANCE_GAMMA.x, TAA_VARIANCE_GAMMA.y,
                       saturate(length(motion_px) / TAA_VARIANCE_MOTION_PX));
    float3 bmin = mean - gamma * sigma;
    float3 bmax = mean + gamma * sigma;
    float2 prev_uv = uv + motion;
    bool on_screen = all(prev_uv >= (float2)(0.0)) && all(prev_uv <= (float2)(1.0));

    // Sanitize the history too: on the first frame it is an uninitialized
    // target, and a NaN read here would survive clip_to_aabb (a NaN fails the
    // ma > 1 test, so the unclipped NaN is returned) and poison every later
    // frame through the feedback.
    float4 hist_raw = taa_history_catmull_rom(prev_uv, size);
    float3 hist = rgb_to_ycocg(taa_sanitize(hist_raw.rgb * params.history_scale));

    // Accumulate only when there is valid, on-screen history; otherwise the
    // current frame passes straight through (first frame, resize, off-screen).
    bool has_history = params.history_valid > 0.5 && on_screen;

    // How much of that history shows this pixel's own surface. Judged where
    // the pixel's own motion puts it last frame, not the nearest surface's:
    // a strip a moving object uncovers was the object there, whatever lies
    // beside it. The color clip alone cannot tell, where the object mirrors
    // the ground it uncovers.
    float4 now = normal_depth_tex.Load(int3(p, 0));
    float surface = 1.0;
    if (has_history && params.surface_valid > 0.5 && now.a > 0.0)
    {
        float2 own_motion = velocity_tex.Load(int3(p, 0)).rg * size;
        surface = taa_surface_history(p, normalize(now.xyz), own_motion, size);
    }

    // Thin-feature lock, carried in the history's alpha. A feature narrower
    // than a pixel covers no pixel center on some jitter frames; on those the
    // 3x3 box above holds no trace of it and the clip would erase its history,
    // so it blinks. A lock, set while the feature is seen and decaying over one
    // jitter cycle, holds the clip off on those frames so the pixel settles at
    // the feature's coverage instead. On a frame that sees the feature the box
    // holds it and the clip runs as usual. A jitter miss and a thin feature
    // moving away look alike in one frame, so a lock forms only where the
    // surface is nearly still on screen (a highlight sliding across a moving
    // prop never locks), and holds the clip off only over a flat neighborhood.
    float lock = has_history ? saturate(select(isnan(hist_raw.a), 0.0, hist_raw.a)) : 0.0;
    lock = max(lock - params.lock_decay, 0.0) * surface;
    bool thin = taa_thin_feature(luma);
    if (thin && dot(motion_px, motion_px) < TAA_LOCK_MOTION_PX * TAA_LOCK_MOTION_PX)
    {
        lock = 1.0;
    }
    bool flat = sigma.x <= TAA_LOCK_FLAT * mean.x;

    float alpha = has_history ? TAA_BLEND * surface : 0.0;
    if (params.reactive_max > 0.0)
    {
        float reactive = min(saturate(reactive_tex.Sample(reactive_samp, uv).r), params.reactive_max);
        alpha *= 1.0 - reactive;
        lock *= 1.0 - reactive;
    }
    float hold = (!thin && flat) ? lock : 0.0;
    hist = lerp(clip_to_aabb(bmin, bmax, hist), hist, hold);
    // Where the history is dropped for another surface the pixel shows the
    // tent rather than its one jittered sample, which would stand out raw
    // beside its accumulated neighbors.
    float3 fresh = lerp(tent, cur, surface);
    return float4(lerp(fresh, ycocg_to_rgb(hist), alpha), lock);
}
