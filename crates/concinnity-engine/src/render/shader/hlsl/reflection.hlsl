// Roughness-aware reflection composite, split into passes so the wide glossy
// blur runs at reduced resolution, behind the ray-traced reflection's
// denoiser. One fragment per compile, selected by a define so each variant
// declares exactly the resources it binds:
//
//   REFLECTION_SURFACE   - trace resolution: the surface each reflection texel
//                          stands for (its world normal, octahedral, and its
//                          view depth) and how many frames it has been seen
//                          for, kept for the next frame's denoise.
//   REFLECTION_TEMPORAL  - trace resolution: the ray-traced reflection's
//                          accumulation. Last frame's, reprojected by the
//                          surface's motion or, for a near mirror, by where the
//                          reflected point appears behind it, is kept where
//                          last frame saw the same surface, clamped to this
//                          frame's neighborhood and blended with this frame's
//                          sampled reflection by the frames it holds. A texel
//                          whose sky was left for a later frame keeps its
//                          history alone. Writes the radiance (rgb) and this
//                          frame's composite weight (a), which the blur and
//                          composite read in place of the frame's own
//                          reflection.
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

// Surfaces rougher than REFLECTION_ROUGHNESS_CUT get no reflection (the
// resolve already wrote weight 0); below it the blur cone ramps from a sharp
// mirror at 0 to REFL_BLUR_MAX.
{REFLECTION_CUT}

#if defined(REFLECTION_TEMPORAL) || defined(REFLECTION_SURFACE)

{REFLECTION_HIT}

// Layout matches `ReflectionDenoiseParams` in render_types (128 B).
struct ReflectionDenoiseParams
{
    // This frame's view space into the previous frame's clip space.
    float4x4 reproject;
    // Rows of this frame's view-to-world rotation.
    float4 view_to_world[3];
    float tan_half_fov_y;
    float aspect;
    // 1 once the history holds an earlier frame.
    float history_valid;
    // What the history is multiplied by to read at this frame's pre-exposure.
    float history_scale;
};

[[vk::push_constant]] ConstantBuffer<ReflectionDenoiseParams> params : register(b0);

// Frames an accumulation counts at most.
static const float REFL_MAX_AGE = 32.0;
// How closely last frame's surface must face (cosine) and lie (relative
// depth) to count as this one.
static const float REFL_NORMAL_MATCH = 0.9;
static const float REFL_DEPTH_MATCH = 0.1;

float3 refl_to_world(float3 v)
{
    return float3(dot(params.view_to_world[0].xyz, v),
                  dot(params.view_to_world[1].xyz, v),
                  dot(params.view_to_world[2].xyz, v));
}

// A unit vector folded onto the octahedron, in [-1, 1]^2, and back.
float2 refl_oct_encode(float3 n)
{
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    if (n.z < 0.0)
    {
        return (1.0 - abs(n.yx)) * select(n.xy >= 0.0, (float2)(1.0), (float2)(-1.0));
    }
    return n.xy;
}

float3 refl_oct_decode(float2 e)
{
    float3 n = float3(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0)
    {
        n.xy = (1.0 - abs(n.yx)) * select(n.xy >= 0.0, (float2)(1.0), (float2)(-1.0));
    }
    return normalize(n);
}

float3 refl_view_pos(float2 uv, float depth)
{
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return float3(ndc.x * params.tan_half_fov_y * params.aspect, ndc.y * params.tan_half_fov_y, -1.0) * depth;
}

float2 refl_clip_uv(float4 clip)
{
    float2 ndc = clip.xy / clip.w;
    return float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

#endif

#if defined(REFLECTION_SURFACE)

// 0 = the G-buffer (rgb = view normal, a = linear view depth); 1 = the motion
// vectors; 2 = last frame's surface.
[[vk::binding(0, 0)]] Texture2D<float4> gbuffer : register(t0);
[[vk::binding(3, 0)]] SamplerState gbuffer_samp : register(s0);
[[vk::binding(1, 0)]] Texture2D<float4> velocity : register(t1);
[[vk::binding(4, 0)]] SamplerState velocity_samp : register(s1);
[[vk::binding(2, 0)]] Texture2D<float4> surface : register(t2);
[[vk::binding(5, 0)]] SamplerState surface_samp : register(s2);

#elif defined(REFLECTION_TEMPORAL)

// 0 = this frame's reflection (rgb = radiance, a = composite weight); 1 = its
// hits (see `REFLECTION_HIT`); 2 = the G-buffer; 3 = the motion vectors; 4 =
// last frame's accumulation (rgb = radiance); 5 = last frame's surface; 6 =
// the G-buffer roughness.
[[vk::binding(0, 0)]] Texture2D<float4> reflection : register(t0);
[[vk::binding(7, 0)]] SamplerState reflection_samp : register(s0);
[[vk::binding(1, 0)]] Texture2D<float4> hits : register(t1);
[[vk::binding(8, 0)]] SamplerState hits_samp : register(s1);
[[vk::binding(2, 0)]] Texture2D<float4> gbuffer : register(t2);
[[vk::binding(9, 0)]] SamplerState gbuffer_samp : register(s2);
[[vk::binding(3, 0)]] Texture2D<float4> velocity : register(t3);
[[vk::binding(10, 0)]] SamplerState velocity_samp : register(s3);
[[vk::binding(4, 0)]] Texture2D<float4> history : register(t4);
[[vk::binding(11, 0)]] SamplerState history_samp : register(s4);
[[vk::binding(5, 0)]] Texture2D<float4> surface : register(t5);
[[vk::binding(12, 0)]] SamplerState surface_samp : register(s5);
[[vk::binding(6, 0)]] Texture2D<float4> rough_tex : register(t6);
[[vk::binding(13, 0)]] SamplerState rough_tex_samp : register(s6);

#endif

#if defined(REFLECTION_TEMPORAL) || defined(REFLECTION_SURFACE)

// Whether last frame saw a surface at `prev_uv` facing along `n` (world)
// within `tolerance` of view depth `depth`, and how many frames it had been
// seen for.
bool refl_surface_at(float2 prev_uv, float3 n, float depth, float tolerance, out float age)
{
    age = 0.0;
    if (params.history_valid < 0.5 || any(prev_uv <= 0.0) || any(prev_uv >= 1.0))
    {
        return false;
    }
    float4 last = surface.Load(int3(int2(prev_uv * texture_size(surface)), 0));
    if (last.a <= 0.0 || dot(refl_oct_decode(last.rg), n) < REFL_NORMAL_MATCH
        || abs(last.b - depth) > tolerance * depth)
    {
        return false;
    }
    age = last.a;
    return true;
}

// Where the surface this frame shows at G-buffer UV `uv` was last frame, by
// its motion, and the view depth it was seen at there.
float2 refl_moved_uv(float2 uv, float3 pv, out float depth)
{
    depth = mul(params.reproject, float4(pv, 1.0)).w;
    return uv + velocity.Sample(velocity_samp, uv).rg;
}

#endif

#if defined(REFLECTION_SURFACE)

// rg = the octahedral world normal, b = the view depth, a = the frames the
// surface has been seen for, this one included; zero where there is none.
[shader("pixel")]
float4 reflection_surface_fragment([[vk::location(0)]] float2 uv : TEXCOORD0) : SV_Target
{
    float2 su = reflection_source_uv(uv, texture_size(gbuffer));
    float4 g = gbuffer.Sample(gbuffer_samp, su);
    if (g.a <= 0.0)
    {
        return (float4)(0.0);
    }
    float3 n = refl_to_world(normalize(g.xyz));
    float depth;
    float2 prev_uv = refl_moved_uv(su, refl_view_pos(su, g.a), depth);
    float age;
    float seen = refl_surface_at(prev_uv, n, depth, REFL_DEPTH_MATCH, age) ? min(age + 1.0, REFL_MAX_AGE) : 1.0;
    return float4(refl_oct_encode(n), g.a, seen);
}

#elif defined(REFLECTION_TEMPORAL)

// The least share this frame takes, for a mirror (x) and at the roughness cut
// (y): a mirror's reflection moves with whatever it shows, so it renews fast.
static const float2 REFL_MIN_SHARE = float2(0.2, 0.04);
// How many standard deviations of this frame's neighborhood the history may
// stray, for a mirror (x) and at the cut (y), whose single draws scatter wider.
static const float2 REFL_CLAMP_SIGMA = float2(1.0, 2.5);
// Roughness below which the history is first sought where the reflected
// point appeared, behind the mirror.
static const float REFL_VIRTUAL_ROUGHNESS = 0.15;

// Last frame's accumulation at `prev_uv` and the frames it holds: its four
// nearest texels, each by its bilinear share, over those where last frame
// saw this surface (facing along `n`, within `tolerance` of view depth
// `depth`) and reflected, so a texel past the roughness cut or of another
// surface lends nothing.
bool refl_history_at(float2 prev_uv, float3 n, float depth, float tolerance,
                     out float3 radiance, out float age)
{
    radiance = (float3)(0.0);
    age = 0.0;
    if (params.history_valid < 0.5 || any(prev_uv <= 0.0) || any(prev_uv >= 1.0))
    {
        return false;
    }
    int2 size = int2(texture_size(surface));
    float2 pos = prev_uv * float2(size) - 0.5;
    int2 base = int2(floor(pos));
    float2 f = pos - floor(pos);
    float3 sum = (float3)(0.0);
    float sum_age = 0.0;
    float sum_w = 0.0;
    for (int j = 0; j < 2; j++)
    {
        for (int i = 0; i < 2; i++)
        {
            int3 q = int3(clamp(base + int2(i, j), int2(0, 0), size - 1), 0);
            float w = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
            float4 last = surface.Load(q);
            float4 h = history.Load(q);
            if (w > 0.0 && h.a > 0.0 && last.a > 0.0
                && dot(refl_oct_decode(last.rg), n) >= REFL_NORMAL_MATCH
                && abs(last.b - depth) <= tolerance * depth)
            {
                sum += h.rgb * w;
                sum_age += last.a * w;
                sum_w += w;
            }
        }
    }
    if (sum_w < 1e-3)
    {
        return false;
    }
    radiance = sum / sum_w * params.history_scale;
    age = sum_age / sum_w;
    return all(isfinite(radiance));
}

[shader("pixel")]
float4 reflection_temporal_fragment([[vk::location(0)]] float2 vertex_uv : TEXCOORD0,
                                    float4 pixel : SV_Position) : SV_Target
{
    int2 p = int2(pixel.xy);
    int2 size = int2(texture_size(reflection));
    float4 cur = reflection.Load(int3(p, 0));
    float2 hit = hits.Load(int3(p, 0)).rg;
    float2 full = texture_size(gbuffer);
    float2 su = reflection_source_uv((float2(p) + 0.5) / float2(size), full);
    float4 g = gbuffer.Sample(gbuffer_samp, su);
    if (cur.a <= 0.0 || g.a <= 0.0)
    {
        return float4(cur.rgb, 0.0);
    }
    float roughness = rough_tex.Sample(rough_tex_samp, su).r;
    float t_rough = saturate(roughness / REFLECTION_ROUGHNESS_CUT);
    float3 pv = refl_view_pos(su, g.a);
    float3 n = refl_to_world(normalize(g.xyz));
    bool deferred = hit.g > 0.5;

    // This frame's neighborhood, over the texels that took their whole
    // reflection this frame.
    float3 m1 = (float3)(0.0);
    float3 m2 = (float3)(0.0);
    float taken = 0.0;
    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            int3 q = int3(clamp(p + int2(dx, dy), int2(0, 0), size - 1), 0);
            float4 c = reflection.Load(q);
            if (c.a > 0.0 && hits.Load(q).g < 0.5)
            {
                m1 += c.rgb;
                m2 += c.rgb * c.rgb;
                taken += 1.0;
            }
        }
    }

    float3 last = (float3)(0.0);
    float age = 0.0;
    bool found = false;
    if (roughness < REFL_VIRTUAL_ROUGHNESS)
    {
        float3 ray = normalize(pv);
        float4 clip = hit.r >= REFLECTION_HIT_ESCAPED
            ? mul(params.reproject, float4(ray, 0.0))
            : mul(params.reproject, float4(pv + ray * hit.r, 1.0));
        found = clip.w > 0.0
            && refl_history_at(refl_clip_uv(clip), n, g.a, 2.0 * REFL_DEPTH_MATCH, last, age);
    }
    if (!found)
    {
        float depth;
        float2 prev_uv = refl_moved_uv(su, pv, depth);
        found = refl_history_at(prev_uv, n, depth, REFL_DEPTH_MATCH, last, age);
    }
    if (found && taken > 0.0)
    {
        float3 mean = m1 / taken;
        float3 sigma = sqrt(max(m2 / taken - mean * mean, (float3)(0.0)));
        float3 reach = lerp(REFL_CLAMP_SIGMA.x, REFL_CLAMP_SIGMA.y, t_rough) * sigma;
        last = clamp(last, mean - reach, mean + reach);
    }

    if (deferred)
    {
        if (found)
        {
            return float4(last, cur.a);
        }
        // No history to hold the sky this texel left: take its block's texel
        // that marched it this frame, the one whose ray escaped where there is
        // one.
        int2 block = p & ~1;
        float4 marched = cur;
        bool escaped = false;
        for (int i = 0; i < 4; i++)
        {
            int3 q = int3(min(block + int2(i & 1, i >> 1), size - 1), 0);
            float4 c = reflection.Load(q);
            float2 h = hits.Load(q).rg;
            if (c.a > 0.0 && h.g < 0.5 && !escaped)
            {
                marched = c;
                escaped = h.r >= REFLECTION_HIT_ESCAPED;
            }
        }
        return float4(marched.rgb, cur.a);
    }
    if (!found)
    {
        return cur;
    }
    float share = max(1.0 / (age + 1.0), lerp(REFL_MIN_SHARE.x, REFL_MIN_SHARE.y, t_rough));
    return float4(lerp(last, cur.rgb, share), cur.a);
}

#elif defined(REFLECTION_BLUR)

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
#error "reflection.hlsl: define REFLECTION_SURFACE, REFLECTION_TEMPORAL, REFLECTION_BLUR or REFLECTION_COMPOSITE"
#endif
