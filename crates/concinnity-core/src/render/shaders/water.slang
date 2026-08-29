// Water surface pass: single source for every backend.
//
// The second producer of the engine's transparent pass (`PassId::Transparent`,
// after the SSR resolve and before TAA), alongside `glass.slang`. Where a glass
// pane is a flat quad the vertex stage only projects, a water surface is a
// tessellated XZ grid (see geometry::water_grid) the vertex stage displaces by a
// sum of Gerstner waves, taking the world-space normal from the analytic
// derivatives of that sum. The fragment stage then composites, in order:
//
//   * Refraction - the pre-transparent scene snapshot at a normal-perturbed
//                  screen UV, so the seabed bends under the waves.
//   * Tint       - shallow to deep by the water-column thickness the main depth
//                  gives, an exponential falloff over `depth_falloff`.
//   * Foam       - a soft band where the seabed is just below the surface.
//   * Reflection - see below.
//   * Fresnel    - Schlick with a water-vs-air F0 of 0.02, shaped by
//                  `fresnel_power`.
//
// One fragment entry per compile, selected by a define. Only the reflection
// differs between them, exactly as in glass.slang:
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
// glass.slang), which is why only this file reads `planar` on the RT path.
//
// USE_MSAA is a HOST difference rather than a target one: Vulkan reads the
// multisampled main depth while Metal and DirectX read the resolved copy.
//
// Every binding here is deliberately the one glass.slang declares at the same
// slot, because the two share a transparent pass: on DirectX that lets one root
// signature serve both pipelines, on Vulkan one set of descriptor set layouts,
// and on Metal the encoder binds the shared reflection inputs once for the whole
// pass. A slot may not move on one side alone.

#ifndef POOL_SIZE
#define POOL_SIZE 1024
#endif
#ifndef MAX_PROBES
#define MAX_PROBES 8
#endif
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

// Per-frame view shared by every transparent draw. Layout matches
// `TransparentView` / the TransparentViewBlock UBO (160 B). The same
// declaration glass.slang carries -- one host block feeds both.
struct TransparentView
{
    float4x4 vp;       // world -> clip (jittered when TAA is on)
    float4x4 inv_vp;   // clip -> world
    float4 camera_pos; // xyz: world-space camera
    float2 viewport;   // attachment dimensions in pixels
    float time;        // seconds since startup, driving the wave phase
    // Mips in the sky prefilter cube; 0 = no EnvironmentMap bound, and the
    // reflection falls through to the sky gradient.
    float prefilter_mip_count;
};

// One Gerstner wave, packed into two float4 lanes so the CPU `[f32; 4]` pair is
// byte-identical on every target. Matches `WaterWaveGpu` (32 B).
struct WaterWave
{
    float4 dir_amp_wave;    // [direction.x, direction.y, amplitude, wavelength]
    float4 speed_steep_pad; // [speed, steepness, _, _]
};

// Per-surface tunables. Layout matches `WaterParams` / the WaterParamsBlock UBO
// (224 B); the vec3 fields are float4 so MSL's 16-byte constant-buffer float3
// cannot desynchronise them from the CPU struct.
struct WaterParams
{
    float4 centre;         // world-space surface centre
    float4 deep_colour;    // linear RGB at full column depth
    float4 shallow_colour; // linear RGB at the shore
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
    // pass ran, and the RT entry honours it too.
    float4 planar;
};

// ---- Resource bindings ----

#ifdef DXIL_ABI

// Pinned to the root signatures in `directx/transparent.rs`, which glass shares;
// b1 is visible to every stage there because the water vertex stage reads the
// wave table out of it.
ConstantBuffer<TransparentView> view : register(b0);
ConstantBuffer<WaterParams> params : register(b1);
Texture2D<float4> scene_color : register(t0);
#if USE_MSAA
Texture2DMS<float> scene_depth : register(t1);
#else
Texture2D<float> scene_depth : register(t1);
#endif
TextureCube<float4> prefilter_cube : register(t2);
Texture2D<float4> planar_reflection : register(t3);
SamplerState post_samp : register(s0);
SamplerState cube_sampler : register(s2);

float4 scene_sample(float2 uv) { return scene_color.Sample(post_samp, uv); }
float4 planar_sample(float2 uv) { return planar_reflection.Sample(post_samp, uv); }
float3 prefilter_level(float3 dir, float lod)
{
    return prefilter_cube.SampleLevel(cube_sampler, dir, lod).rgb;
}

#else

// Metal buffer(5) / buffer(6): the shared per-frame view and the per-surface
// params, both written with setBytes by the transparent encoder.
[[vk::binding(0, 0)]] ConstantBuffer<TransparentView> view : register(b5);
[[vk::binding(0, 1)]] ConstantBuffer<WaterParams> params : register(b6);

// Declaration order is the Metal texture index, and these are the transparent
// pass's shared slots: the scene snapshot at 0, the resolved depth at 1, the sky
// prefilter cube at 2, the probe cubes at 3..2+MAX_PROBES, and this surface's
// planar resolve at 11.
[[vk::binding(1, 0)]] Sampler2D<float4> scene_color;
#if USE_MSAA
[[vk::binding(2, 0)]] Texture2DMS<float> scene_depth;
#else
[[vk::binding(2, 0)]] Texture2D<float> scene_depth;
#endif
[[vk::binding(5, 2)]] SamplerCube<float4> prefilter_cube;

float4 scene_sample(float2 uv) { return scene_color.Sample(uv); }
float3 prefilter_level(float3 dir, float lod) { return prefilter_cube.SampleLevel(dir, lod).rgb; }

#endif

// The reflection-probe set + cube array, from the forward global set: a pond
// reflects the same local scene capture the forward IBL specular and the SSR /
// RT miss fallback use, rather than only the foreign sky cube.
#ifdef DXIL_ABI
ConstantBuffer<ProbeSet> probe_set : register(b4);
// The array spans MAX_PROBES registers from its base, so the RT variant moves it
// clear of the ray-tracing SRVs at t4..t10 rather than starting at t7.
#ifdef WATER_RT
TextureCube<float4> probe_cubes[MAX_PROBES] : register(t20);
#else
TextureCube<float4> probe_cubes[MAX_PROBES] : register(t7);
#endif

float3 probe_cube_sample_bias(uint i, float3 dir, float lod)
{
    return probe_cubes[i].SampleBias(cube_sampler, dir, lod).rgb;
}
#else
[[vk::binding(7, 2)]] ConstantBuffer<ProbeSet> probe_set : register(b7);
[[vk::binding(8, 2)]] SamplerCube<float4> probe_cubes[MAX_PROBES];

float3 probe_cube_sample_bias(uint i, float3 dir, float lod)
{
    return probe_cubes[i].SampleBias(dir, lod).rgb;
}
#endif
#define PROBE_SET probe_set

#ifndef DXIL_ABI
// This surface's planar reflection target, bound per surface and sampled at the
// fragment's screen UV when `planar.x > 0.5`. Declared after the probe array so
// it lands on Metal's texture(11); a surface with no planar slot binds a valid
// stand-in and never samples it. Read by both entries: the RT one takes the
// mirror over its own trace wherever a slot exists.
[[vk::binding(1, 1)]] Sampler2D<float4> planar_reflection;

float4 planar_sample(float2 uv) { return planar_reflection.Sample(uv); }
#endif

#ifdef WATER_RT
// The ray-tracing scene resources, at the slots glass.slang uses so the inputs
// the transparent encoder binds once are valid for both. On Metal they ride the
// pass's otherwise-free fragment buffers (0..4 and 8..10, since 5/6/7 are the
// view, the params and the probe set); on Vulkan they are a set of their own,
// past the view / params / global sets; on DirectX they follow the registers the
// pass already occupies.
#ifdef DXIL_ABI

ConstantBuffer<RtParams> rt_params : register(b5);
RaytracingAccelerationStructure scene_tlas : register(t4);
ByteAddressBuffer verts : register(t5);
ByteAddressBuffer indices : register(t6);
ByteAddressBuffer sverts : register(t8);
ByteAddressBuffer sidx : register(t9);
StructuredBuffer<RtGeomEntry> geom : register(t10);

float vert_float(uint i) { return asfloat(verts.Load(i * 4u)); }
float svert_float(uint i) { return asfloat(sverts.Load(i * 4u)); }
uint index_at(uint o) { return indices.Load(o * 4u); }
uint skinned_index_word(uint w) { return sidx.Load(w * 4u); }

#else

[[vk::binding(0, 3)]] ConstantBuffer<RtParams> rt_params : register(b0);
[[vk::binding(1, 3)]] RaytracingAccelerationStructure scene_tlas : register(t4);
[[vk::binding(2, 3)]] StructuredBuffer<RtGeomEntry> geom : register(t3);
[[vk::binding(3, 3)]] StructuredBuffer<float> verts : register(t1);
[[vk::binding(4, 3)]] StructuredBuffer<uint> indices : register(t2);
[[vk::binding(5, 3)]] StructuredBuffer<float> sverts : register(t8);
[[vk::binding(6, 3)]] StructuredBuffer<uint> sidx : register(t9);

float vert_float(uint i) { return verts[i]; }
float svert_float(uint i) { return sverts[i]; }
uint index_at(uint o) { return indices[o]; }
uint skinned_index_word(uint w) { return sidx[w]; }

#endif

#ifdef RT_TEXTURED
// The bindless pool. Metal keeps it at buffer(10): buffer(7), where the main
// pass puts it, is the probe set in the transparent pass.
uint nonuniform_index(uint i)
{
    __target_switch
    {
    case metal:
        return i;
    default:
        return NonUniformResourceIndex(i);
    }
}

#if defined(METAL_ABI)
struct TexturePool
{
    Texture2D<float4> tex_pool[POOL_SIZE];
};
ParameterBlock<TexturePool> pool;
SamplerState pool_sampler;

float3 pool_sample_level0(uint idx, float2 uv)
{
    return pool.tex_pool[nonuniform_index(idx)].SampleLevel(pool_sampler, uv, 0.0).rgb;
}
#elif defined(DXIL_ABI)
Texture2D<float4> tex_pool[] : register(t0, space1);
SamplerState pool_sampler : register(s1);

float3 pool_sample_level0(uint idx, float2 uv)
{
    return tex_pool[nonuniform_index(idx)].SampleLevel(pool_sampler, uv, 0.0).rgb;
}
#else
[[vk::binding(1, 4)]] Sampler2D<float4> tex_pool[POOL_SIZE];

float3 pool_sample_level0(uint idx, float2 uv)
{
    return tex_pool[nonuniform_index(idx)].SampleLevel(uv, 0.0).rgb;
}
#endif
#endif
#endif

{PROBE_COMMON}

#ifdef WATER_RT
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
    [[vk::location(1)]] float3 world_normal : TEXCOORD1;
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

        // Steepness normalised by wave count so summed crests cannot pinch past
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
    // The grid is built centred on the origin in the XZ plane, so the surface
    // centre rides in through the params rather than the vertex buffer.
    float2 rest_xz = float2(v.pos.x + params.centre.x, v.pos.z + params.centre.z);
    WaterPoint p = gerstner_displace(rest_xz, view.time);
    p.pos.y += params.centre.y;

    WaterVertexOut o;
    o.world_pos = p.pos;
    o.world_normal = p.normal;
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
    int2 last = int2(max(view.viewport, float2(1.0))) - int2(1, 1);
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
    s.normal = normalize(i.world_normal);

    float2 vp_dim = max(view.viewport, float2(1.0));
    s.frag_uv = i.position.xy / vp_dim;

    // Refraction: perturb the screen lookup by the wave normal's XZ so the
    // seabed bends under the waves.
    float2 refract_uv = clamp(s.frag_uv + s.normal.xz * params.refraction_strength,
                              float2(0.001), float2(0.999));
    float3 refracted = scene_sample(refract_uv).rgb;

    // Read the main depth at the REFRACTED pixel, so the thickness matches the
    // texel just sampled: a refraction that bends into a foreground edge would
    // otherwise be tinted as if that edge were underwater.
    float scene_depth01 = water_scene_depth(water_clamp_pixel(refract_uv * vp_dim));
    float2 ndc_xy = float2(s.frag_uv.x * 2.0 - 1.0, -(s.frag_uv.y * 2.0 - 1.0));
    float scene_dist = water_scene_distance(ndc_xy, scene_depth01);
    float water_dist = distance(i.world_pos, view.camera_pos.xyz);
    float water_depth = max(scene_dist - water_dist, 0.0);

    // Tint: exponential shallow to deep blend over `depth_falloff` metres.
    float depth_t = 1.0 - exp(-water_depth / max(params.depth_falloff, 1e-3));
    float3 tinted = lerp(params.shallow_colour.rgb, params.deep_colour.rgb, depth_t);
    float3 below = lerp(refracted * params.shallow_colour.rgb, tinted, depth_t);

    // Foam: a soft band where the seabed is just below the surface, which is
    // both the shoreline and any intersection line with standing geometry.
    float foam_t = saturate(1.0 - water_depth / max(params.foam_width, 1e-3));
    s.below = lerp(below, float3(1.0), foam_t * foam_t * params.foam_intensity);
    return s;
}

// The mirror render for this surface, at the fragment's own screen UV perturbed
// by the wave normal. The planar render mirrors the scene across the surface's
// rest plane, so it lands exactly under the reflector; perturbing that lookup is
// what turns a flat mirror back into rippling water.
float3 water_planar_reflection(WaterSurfacePoint s)
{
    float2 uv = clamp(s.frag_uv + s.normal.xz * params.planar.y,
                      float2(0.001), float2(0.999));
    return planar_sample(uv).rgb;
}

// The prefilter mip a surface of this roughness reflects at. Blurrier water
// reads a coarser cube level. The mip count is per-frame state, so it comes from
// the pass's view block rather than the per-surface params.
float water_reflection_mip()
{
    return saturate(params.roughness) * max(view.prefilter_mip_count - 1.0, 0.0);
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
float3 water_environment(float3 world_pos, float3 r)
{
    float mip = water_reflection_mip();
    if (probe_set.count > 0u && probe_set_covers(world_pos))
    {
        return probe_set_specular(world_pos, r, mip);
    }
    if (view.prefilter_mip_count > 0.5)
    {
        return prefilter_level(r, mip);
    }
    float horizon = saturate(r.y * 0.5 + 0.5);
    return lerp(float3(0.55, 0.62, 0.7), float3(0.25, 0.45, 0.7), horizon);
}

// Schlick Fresnel mix of the reflection over everything below the waterline.
// F0 = 0.02 is the water-vs-air value; `fresnel_power` shapes the falloff, so a
// low power keeps the reflection strong head-on instead of only at grazing.
// Alpha 1: water fully covers what it drew over, and the pipeline's straight
// alpha blend leaves the composited colour as-is.
float4 water_resolve(WaterSurfacePoint s, float3 view_dir, float3 reflection)
{
    float n_dot_v = saturate(dot(s.normal, view_dir));
    float fresnel = 0.02 + 0.98 * pow(1.0 - n_dot_v, max(params.fresnel_power, 1e-3));
    return float4(lerp(s.below, reflection, fresnel), 1.0);
}

// True where nearer opaque geometry occludes the surface. The transparent pass
// binds no depth attachment, so the test is manual; every backend rasterised
// this depth under the same viewport convention the main pass used, so the
// fragment position lines up with the stored texel.
bool water_occluded(WaterVertexOut i)
{
    return water_scene_depth(water_clamp_pixel(i.position.xy)) < i.position.z;
}

#ifdef WATER_RT

[shader("fragment")]
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
            reflection = water_environment(i.world_pos, r);
        }
    }
    return water_resolve(s, view_dir, reflection);
}

#else

[shader("fragment")]
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
        reflection = water_environment(i.world_pos, reflect(-view_dir, s.normal));
    }
    return water_resolve(s, view_dir, reflection);
}

#endif
