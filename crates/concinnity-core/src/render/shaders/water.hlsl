// Water surface pass: single source for every backend.
//
// The second producer of the engine's transparent pass (`PassId::Transparent`,
// after the SSR resolve and before TAA), alongside `glass.hlsl`. Where a glass
// pane is a flat quad the vertex stage only projects, a water surface is a
// tessellated XZ grid (see geometry::water_grid) the vertex stage displaces by a
// sum of Gerstner waves. The fragment stage takes the world-space normal from
// the analytic derivatives of that same sum at its own rest position, so a
// coarse grid shades as finely as a dense one, then composites, in order:
//
//   * Refraction - the pre-transparent scene snapshot at a normal-perturbed
//                  screen UV, so the seabed bends under the waves.
//   * Tint       - shallow to deep by the water-column thickness the main depth
//                  gives, an exponential falloff over `depth_falloff`.
//   * Foam       - a soft band where the seabed is just below the surface.
//   * Reflection - see below, plus a GGX specular lobe for the scene's sun off
//                  the same wave normal.
//   * Fresnel    - Schlick with a water-vs-air F0 of 0.02, shaped by
//                  `fresnel_power`.
//
// One fragment entry per compile, selected by a define. Only the reflection
// differs between them, exactly as in glass.hlsl:
//
//   default                   - the sharp planar reflection when this surface
//                               has a slot (screen UV perturbed by the wave
//                               normal, so the mirror ripples), else the
//                               box-projected reflection-probe set, else the sky
//                               prefilter cube, else a hand-tuned sky gradient.
//   WATER_RT                  - that same planar reflection where the surface has
//                               a slot, else a per-pixel reflection ray against
//                               the scene acceleration structure (the shared
//                               RT_TRACE fragment) off the wave surface point, so
//                               `R` follows the waves; the miss falls back to the
//                               same probe / sky chain.
//   WATER_RT with RT_TEXTURED - the same trace, with reflected hits taking their
//                               albedo / normal / emissive maps from the
//                               bindless pool.
//
// The mirror outranks the trace for water because a water surface IS a plane:
// one mirrored scene render resolves it exactly, where a trace off the
// per-fragment wave normal is hypersensitive at grazing angles and drops to the
// probe / sky chain wherever it misses. A glass pane does the opposite (see
// glass.hlsl), which is why only this file reads `planar` on the RT path.
//
// The inputs every transparent producer shares come from TRANSPARENT_SCENE, and
// the planar reflection takes the slot glass.hlsl declares for it.
// CN_BACKEND_DIRECTX pins every register to the root signatures in
// `directx/transparent.rs`; the other targets carry one declaration with both a
// `[[vk::binding]]` and a `register()`, whose number IS the Metal index in the
// namespace the resource's kind implies. The bindless
// pool is a whole descriptor set on Metal, named by a
// `[[cn::metal_argument_buffer(n)]]`, because no per-resource register can name
// a set.

#ifndef USE_MSAA
#define USE_MSAA 0
#endif

// Waves summed per surface. Mirrors `MAX_WATER_WAVES` in concinnity-asset and
// `WATER_MAX_WAVES` in `core::render`'s uniforms; the array length is part of
// the `WaterParams` layout, so it is a constant here rather than a define.
static const uint MAX_WATER_WAVES = 4u;

{PROBE_TYPES}

#ifdef WATER_RT
{RT_TYPES}
#endif

{TRANSPARENT_TYPES}

// One Gerstner wave, packed into two float4 lanes so the CPU `[f32; 4]` pair is
// byte-identical on every target. Matches `WaterWaveGpu` (32 B).
struct WaterWave
{
    float4 dir_amp_wave;    // [direction.x, direction.y, amplitude, wavelength]
    float4 speed_steep_pad; // [speed, steepness, _, _]
};

// Per-surface tunables. Layout matches `WaterParams` / the WaterParamsBlock UBO
// (224 B); the vec3 fields are float4 so MSL's 16-byte constant-buffer float3
// cannot desynchronize them from the CPU struct.
struct WaterParams
{
    float4 center;         // world-space surface center
    float4 deep_color;    // linear RGB at full column depth
    float4 shallow_color; // linear RGB at the shore
    float depth_falloff;
    float foam_width;
    float foam_intensity;
    float fresnel_power;
    float roughness;
    float refraction_strength;
    uint wave_count;
    // Aligns `waves` to 16, which its float4 lanes require.
    float _pad;
    WaterWave waves[MAX_WATER_WAVES];
    // [strength, distortion, _, _]. `strength > 0.5` selects the sharp planar
    // reflection over the trace / probe / sky cube; `distortion` scales the
    // wave-normal perturbation of its screen-UV lookup. Set on both paths: the
    // host raises it whenever this surface holds a mirror slot and the planar
    // pass ran, and the RT entry honors it too.
    float4 planar;
};

// ---- Resource bindings ----

#define TRANSPARENT_PARAMS WaterParams
{TRANSPARENT_SCENE}

// This surface's planar reflection target, bound per surface and sampled at the
// fragment's screen UV when `planar.x > 0.5`; a surface with no planar slot
// binds a valid stand-in and never samples it. Read by both entries: the RT one
// takes the mirror over its own trace wherever a slot exists.
#ifdef CN_BACKEND_DIRECTX
Texture2D<float4> planar_reflection : register(t3);
#define planar_reflection_sampler post_samp
#else
[[vk::binding(1, 1)]] Texture2D<float4> planar_reflection : register(t3);
[[vk::binding(2, 1)]] SamplerState planar_reflection_sampler : register(s3);
#endif

float4 planar_sample(float2 uv) { return planar_reflection.Sample(planar_reflection_sampler, uv); }

{TRANSPARENT_PROBES}

#ifdef WATER_RT
{TRANSPARENT_RT}

{RT_TRACE}
#endif

// ---- Stage interface ----

struct WaterVertexIn
{
    [[vk::location(0)]] float3 pos : POSITION;
};

struct WaterVertexOut
{
    [[vk::location(0)]] float3 world_pos : TEXCOORD0;
    // The flat-rest XZ the vertex was displaced from; the fragment evaluates
    // the wave normal there itself, so the grid's density sets only the
    // silhouette of the displacement, never the shading.
    [[vk::location(1)]] float2 rest_xz : TEXCOORD1;
    float4 position : SV_Position;
};

// The displaced surface point and its normal for one flat-rest XZ position.
struct WaterPoint
{
    float3 pos;
    float3 normal;
};

// Sum up to MAX_WATER_WAVES Gerstner waves at a flat-rest XZ position. Each wave
// is a horizontal pinch plus a vertical sinusoid; the analytic partials against
// (x, z) give the world-space normal at the displaced point. Reference: NVIDIA,
// "Effective Water Simulation from Physical Models", GPU Gems 1.
WaterPoint gerstner_displace(float2 rest_xz, float time)
{
    float3 displaced = float3(rest_xz.x, 0.0, rest_xz.y);
    float3 binormal = float3(1.0, 0.0, 0.0); // dP/dx
    float3 tangent = float3(0.0, 0.0, 1.0);  // dP/dz

    uint count = min(params.wave_count, MAX_WATER_WAVES);
    for (uint i = 0u; i < count; i++)
    {
        float2 dir = normalize(params.waves[i].dir_amp_wave.xy);
        float amp = params.waves[i].dir_amp_wave.z;
        float wavelen = max(params.waves[i].dir_amp_wave.w, 1e-3);
        float speed = params.waves[i].speed_steep_pad.x;
        float steep = saturate(params.waves[i].speed_steep_pad.y);

        float k = 2.0 * 3.14159265358979 / wavelen;
        float phase = k * dot(dir, rest_xz) - speed * k * time;
        float c = cos(phase);
        float s = sin(phase);

        // Steepness normalized by wave count so summed crests cannot pinch past
        // self-intersection.
        float q = steep / (k * amp * max(float(count), 1.0));

        displaced.x += q * amp * dir.x * c;
        displaced.z += q * amp * dir.y * c;
        displaced.y += amp * s;

        float wa = k * amp;
        binormal.x += -q * dir.x * dir.x * wa * s;
        binormal.y += dir.x * wa * c;
        binormal.z += -q * dir.x * dir.y * wa * s;

        tangent.x += -q * dir.x * dir.y * wa * s;
        tangent.y += dir.y * wa * c;
        tangent.z += -q * dir.y * dir.y * wa * s;
    }

    WaterPoint p;
    p.pos = displaced;
    p.normal = normalize(cross(tangent, binormal));
    return p;
}

[shader("vertex")]
WaterVertexOut water_vertex(WaterVertexIn v)
{
    // The grid is built centered on the origin in the XZ plane, so the surface
    // center rides in through the params rather than the vertex buffer.
    float2 rest_xz = float2(v.pos.x + params.center.x, v.pos.z + params.center.z);
    WaterPoint p = gerstner_displace(rest_xz, view.time);
    p.pos.y += params.center.y;

    WaterVertexOut o;
    o.world_pos = p.pos;
    o.rest_xz = rest_xz;
    o.position = mul(view.vp, float4(p.pos, 1.0));
    return o;
}

// Depth stored at this pixel by the main pass, for the manual occlusion test and
// the water-column thickness.
float water_scene_depth(int2 pixel)
{
#if USE_MSAA
    return scene_depth.Load(pixel, 0);
#else
    return scene_depth.Load(int3(pixel, 0));
#endif
}

// A screen-space pixel coordinate clamped into the attachment.
int2 water_clamp_pixel(float2 pixel_xy)
{
    int2 last = int2(max(view.viewport, (float2)(1.0))) - int2(1, 1);
    return clamp(int2(pixel_xy), int2(0, 0), last);
}

// Linear camera distance to the scene point a screen NDC position and its stored
// non-linear depth describe. Gives the water column thickness when differenced
// against the distance to the surface itself.
float water_scene_distance(float2 ndc_xy, float depth01)
{
    float4 world = mul(view.inv_vp, float4(ndc_xy, depth01, 1.0));
    return distance(world.xyz / world.w, view.camera_pos.xyz);
}

// The surface at this fragment: the wave normal, the fragment's screen UV, and
// everything below the waterline -- the refracted scene, tinted by column depth
// and brightened to foam where the seabed is close.
struct WaterSurfacePoint
{
    float3 normal;
    float2 frag_uv;
    float3 below;
};

WaterSurfacePoint water_surface(WaterVertexOut i)
{
    WaterSurfacePoint s;
    s.normal = gerstner_displace(i.rest_xz, view.time).normal;

    float2 vp_dim = max(view.viewport, (float2)(1.0));
    s.frag_uv = i.position.xy / vp_dim;

    // Refraction: perturb the screen lookup by the wave normal's XZ so the
    // seabed bends under the waves.
    float2 refract_uv = clamp(s.frag_uv + s.normal.xz * params.refraction_strength,
                              (float2)(0.001), (float2)(0.999));
    float3 refracted = scene_sample(refract_uv).rgb;

    // Read the main depth at the REFRACTED pixel, so the thickness matches the
    // texel just sampled: a refraction that bends into a foreground edge would
    // otherwise be tinted as if that edge were underwater.
    float scene_depth01 = water_scene_depth(water_clamp_pixel(refract_uv * vp_dim));
    float2 ndc_xy = float2(s.frag_uv.x * 2.0 - 1.0, -(s.frag_uv.y * 2.0 - 1.0));
    float scene_dist = water_scene_distance(ndc_xy, scene_depth01);
    float water_dist = distance(i.world_pos, view.camera_pos.xyz);
    float water_depth = max(scene_dist - water_dist, 0.0);

    // Tint: exponential shallow to deep blend over `depth_falloff` meters.
    float depth_t = 1.0 - exp(-water_depth / max(params.depth_falloff, 1e-3));
    float3 tinted = lerp(params.shallow_color.rgb, params.deep_color.rgb, depth_t);
    float3 below = lerp(refracted * params.shallow_color.rgb, tinted, depth_t);

    // Foam: a soft band where the seabed is just below the surface, which is
    // both the shoreline and any intersection line with standing geometry.
    float foam_t = saturate(1.0 - water_depth / max(params.foam_width, 1e-3));
    s.below = lerp(below, (float3)(1.0), foam_t * foam_t * params.foam_intensity);
    return s;
}

// The mirror render for this surface, at the fragment's own screen UV perturbed
// by the wave normal. The planar render mirrors the scene across the surface's
// rest plane, so it lands exactly under the reflector; perturbing that lookup is
// what turns a flat mirror back into rippling water.
float3 water_planar_reflection(WaterSurfacePoint s)
{
    float2 uv = clamp(s.frag_uv + s.normal.xz * params.planar.y,
                      (float2)(0.001), (float2)(0.999));
    return planar_sample(uv).rgb;
}

// The reflection a surface with no sharp source falls back to: the local
// box-projected probe set where a probe actually covers this point, else the sky
// prefilter cube, else a hand-tuned vertical sky gradient (bluer overhead, paler
// at the horizon) so an environment-less world still reads as water rather than
// as flat tinted glass.
//
// The coverage test matters more here than anywhere else: a pool routinely
// stretches past every probe box in the world, and the probe set's own
// out-of-box fallback would hand the whole surface one foreign capture.
float3 water_environment(float3 world_pos, float3 r, float2 pixel)
{
    float horizon = saturate(r.y * 0.5 + 0.5);
    float3 gradient = lerp(float3(0.55, 0.62, 0.7), float3(0.25, 0.45, 0.7), horizon);
    // Blurrier water reads a coarser cube level.
    return transparent_environment(world_pos, r, saturate(params.roughness), gradient, pixel);
}

// The sun's specular lobe off this fragment's wave normal: GGX with a
// height-correlated Smith visibility term and the water-vs-air F0, driven by the
// first directional light the view block carries. A high sun draws a compact
// disc; a low one spreads the same lobe into a path running toward the camera,
// since the wave slopes that mirror it span more of the surface. Zero when the
// world declares no directional light, whose `sun_color` is zero.
float3 water_sun_glint(float3 normal, float3 view_dir)
{
    float n_dot_l = dot(normal, view.sun_dir.xyz);
    if (n_dot_l <= 0.0)
    {
        return (float3)(0.0);
    }
    float3 h = normalize(view.sun_dir.xyz + view_dir);
    float n_dot_v = max(dot(normal, view_dir), 1e-4);
    float n_dot_h = saturate(dot(normal, h));
    float v_dot_h = saturate(dot(view_dir, h));

    float alpha = max(params.roughness, 0.02);
    alpha *= alpha;
    float a2 = alpha * alpha;
    float denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    float d = a2 / max(3.14159265358979 * denom * denom, 1e-6);

    float lambda_v = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    float lambda_l = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    float vis = 0.5 / max(lambda_v + lambda_l, 1e-6);

    float f = 0.02 + 0.98 * pow(1.0 - v_dot_h, 5.0);
    return view.sun_color.rgb * (d * vis * f * n_dot_l);
}

// Schlick Fresnel mix of the reflection over everything below the waterline.
// F0 = 0.02 is the water-vs-air value; `fresnel_power` shapes the falloff, so a
// low power keeps the reflection strong head-on instead of only at grazing.
// The sun's glint is added after that mix: its lobe carries its own Fresnel
// term, which is what strengthens it toward grazing angles, so weighting it by
// the surface Fresnel as well would count that twice.
// Alpha 1: water fully covers what it drew over, and the pipeline's straight
// alpha blend leaves the composited color as-is.
float4 water_resolve(WaterSurfacePoint s, float3 view_dir, float3 reflection)
{
    float n_dot_v = saturate(dot(s.normal, view_dir));
    float fresnel = 0.02 + 0.98 * pow(1.0 - n_dot_v, max(params.fresnel_power, 1e-3));
    float3 color = lerp(s.below, reflection, fresnel) + water_sun_glint(s.normal, view_dir);
    return float4(color, 1.0);
}

// True where nearer opaque geometry occludes the surface. The transparent pass
// binds no depth attachment, so the test is manual; every backend rasterized
// this depth under the same viewport convention the main pass used, so the
// fragment position lines up with the stored texel.
bool water_occluded(WaterVertexOut i)
{
    return water_scene_depth(water_clamp_pixel(i.position.xy)) < i.position.z;
}

#ifdef WATER_RT

[shader("pixel")]
float4 water_rt_fragment(WaterVertexOut i) : SV_Target
{
    if (water_occluded(i))
    {
        discard;
    }
    float3 view_dir = normalize(view.camera_pos.xyz - i.world_pos);
    WaterSurfacePoint s = water_surface(i);

    float3 reflection;
    if (params.planar.x > 0.5)
    {
        reflection = water_planar_reflection(s);
    }
    // No mirror plane for this surface, so trace instead. The per-fragment
    // Gerstner normal makes `R` vary across the surface, so the traced
    // reflection follows the waves rather than mirroring one flat plane.
    else
    {
        float3 r = reflect(-view_dir, s.normal);
        if (!rt_trace_reflection(i.world_pos + s.normal * 0.02, r,
                                 view.prefilter_mip_count > 0.5,
                                 view.prefilter_mip_count - 1.0, reflection))
        {
            reflection = water_environment(i.world_pos, r, i.position.xy);
        }
    }
    return water_resolve(s, view_dir, reflection);
}

#else

[shader("pixel")]
float4 water_fragment(WaterVertexOut i) : SV_Target
{
    if (water_occluded(i))
    {
        discard;
    }
    float3 view_dir = normalize(view.camera_pos.xyz - i.world_pos);
    WaterSurfacePoint s = water_surface(i);

    float3 reflection;
    if (params.planar.x > 0.5)
    {
        reflection = water_planar_reflection(s);
    }
    else
    {
        reflection = water_environment(i.world_pos, reflect(-view_dir, s.normal), i.position.xy);
    }
    return water_resolve(s, view_dir, reflection);
}

#endif
