// Glass panel pass: single source for every backend.
//
// The simplest consumer of the engine's transparent pass -- a flat, fixed
// rectangular pane, drawn in the `PassId::Transparent` slot after the SSR
// resolve and before TAA. The quad is already in world space (see
// geometry::glass_quad), so the vertex stage only projects it. The fragment
// stage discards where nearer opaque geometry occludes the pane (a manual depth
// test: the transparent pass binds no depth attachment), refracts the
// pre-transparent scene snapshot and tints it, then mixes a reflection over the
// refraction by a Schlick Fresnel term. The pipeline straight-alpha blends the
// result.
//
// One fragment entry per compile, selected by a define; everything outside the
// reflection itself is shared, so toggling RT only changes where the reflection
// comes from:
//
//   default                   - the sharp planar reflection when this pane has
//                               one, else the box-projected reflection-probe
//                               set, else the sky prefilter cube, else a white
//                               rim.
//   GLASS_RT                  - a real per-pixel reflection ray against the
//                               scene acceleration structure (the shared
//                               RT_TRACE fragment), falling back to the same
//                               probe / sky chain when the ray escapes.
//   GLASS_RT with RT_TEXTURED - the same trace, with reflected hits taking
//                               their albedo / normal / emissive maps from the
//                               bindless pool.
//
// USE_MSAA is a HOST difference rather than a target one: Vulkan reads the
// multisampled main depth while Metal and DirectX read the resolved copy.
//
// DXIL_ABI pins every register to the root signatures in `directx/glass.rs`;
// the other targets carry one declaration with both a `[[vk::binding]]` and a
// `register()`, whose number IS the Metal buffer index. Those Metal indices are
// shared with the hand-written water and glass-mesh shaders, which the same
// transparent encoder feeds, so they are not free to move.

#ifndef POOL_SIZE
#define POOL_SIZE 1024
#endif
#ifndef MAX_PROBES
#define MAX_PROBES 8
#endif
#ifndef USE_MSAA
#define USE_MSAA 0
#endif

{PROBE_TYPES}

#ifdef GLASS_RT
{RT_TYPES}
#endif

// Per-frame view shared by every transparent draw. Layout matches
// `TransparentView` / the TransparentViewBlock UBO (240 B).
struct TransparentView
{
    float4x4 vp;       // world -> clip (jittered when TAA is on)
    float4x4 inv_vp;   // clip -> world
    float4 camera_pos; // xyz: world-space camera
    float2 viewport;   // attachment dimensions in pixels
    float time;        // seconds since startup
    // Mips in the sky prefilter cube; 0 = no EnvironmentMap bound, and the
    // reflection keeps the white rim where no probe covers.
    float prefilter_mip_count;
    // Rows of the rotation from world space into the sky cube's baked frame;
    // identity when the sky does not turn.
    float4 sky_rot[3];
    // Direction toward the scene's sun (the first directional light) and that
    // light's colour times its intensity; both zero when the world declares no
    // directional light, which the water glint reads as no sun.
    float4 sun_dir;
    float4 sun_color;
};

// A world direction in the sky cube's own frame; every sky tap goes through it.
#define SKY_DIR(d) float3(dot(view.sky_rot[0].xyz, (d)), \
                          dot(view.sky_rot[1].xyz, (d)), \
                          dot(view.sky_rot[2].xyz, (d)))

// Per-pane tunables. Layout matches `GlassParams` / the GlassParamsBlock UBO
// (64 B); the vec3 fields are float4 so MSL's 16-byte constant-buffer float3
// cannot desynchronise them from the CPU struct.
struct GlassParams
{
    float4 centre; // world-space pane centre
    float4 normal; // unit pane normal (facing direction)
    float4 tint;   // colour multiplied into the refracted scene
    float opacity;
    float refraction_strength;
    float fresnel_power;
    // 1.0 when this pane has a planar reflection slot: sample the sharp mirror
    // render projectively instead of the box-projected probe. Never set on the
    // RT path, which traces a sharper reflection than a planar render.
    float planar;
};

// ---- Resource bindings ----

#ifdef DXIL_ABI

ConstantBuffer<TransparentView> view : register(b0);
ConstantBuffer<GlassParams> params : register(b1);
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
    return prefilter_cube.SampleLevel(cube_sampler, SKY_DIR(dir), lod).rgb;
}

#else

// Metal buffer(5) / buffer(6): the shared per-frame view and the per-pane
// params, both written with setBytes by the transparent encoder.
[[vk::binding(0, 0)]] ConstantBuffer<TransparentView> view : register(b5);
[[vk::binding(0, 1)]] ConstantBuffer<GlassParams> params : register(b6);

// Declaration order is the Metal texture index, and these are the transparent
// pass's shared slots: the scene snapshot at 0, the resolved depth at 1, the sky
// prefilter cube at 2, the probe cubes at 3..2+MAX_PROBES, and this pane's
// planar resolve at 11.
[[vk::binding(1, 0)]] Sampler2D<float4> scene_color;
#if USE_MSAA
[[vk::binding(2, 0)]] Texture2DMS<float> scene_depth;
#else
[[vk::binding(2, 0)]] Texture2D<float> scene_depth;
#endif
[[vk::binding(5, 2)]] SamplerCube<float4> prefilter_cube;

float4 scene_sample(float2 uv) { return scene_color.Sample(uv); }
float3 prefilter_level(float3 dir, float lod) { return prefilter_cube.SampleLevel(SKY_DIR(dir), lod).rgb; }

#endif

// The reflection-probe set + cube array, from the forward global set: glass
// reflects the same local scene capture the forward IBL specular and the SSR /
// RT miss fallback use, rather than only the foreign sky cube.
#ifdef DXIL_ABI
ConstantBuffer<ProbeSet> probe_set : register(b4);
// The array spans MAX_PROBES registers from its base, so the RT variant moves it
// clear of the ray-tracing SRVs at t4..t10 rather than starting at t7.
#ifdef GLASS_RT
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
#ifdef METAL_ABI
// Metal reaches the cubes through an argument buffer: a resource array at
// global scope emits with no [[texture(n)]], and the compiler then places it
// at whatever slot happens to be unused.
struct ProbeCubes
{
    TextureCube<float4> probe_cubes[MAX_PROBES];
};
ParameterBlock<ProbeCubes> probe_cube_set : register(b11);
SamplerState probe_cube_sampler;

float3 probe_cube_sample_bias(uint i, float3 dir, float lod)
{
    return probe_cube_set.probe_cubes[i].SampleBias(probe_cube_sampler, dir, lod).rgb;
}
#else
[[vk::binding(8, 2)]] SamplerCube<float4> probe_cubes[MAX_PROBES];

float3 probe_cube_sample_bias(uint i, float3 dir, float lod)
{
    return probe_cubes[i].SampleBias(dir, lod).rgb;
}
#endif
#endif
#define PROBE_SET probe_set

#ifndef DXIL_ABI
// This pane's planar reflection target, bound per pane and sampled projectively
// when `planar > 0.5`. Declared after the probe array so it lands on Metal's
// texture(11); a pane with no planar slot binds a valid stand-in and never
// samples it. Unused on the RT path, but the transparent encoder binds the slot
// for every draw, so the declaration stays.
[[vk::binding(1, 1)]] Sampler2D<float4> planar_reflection;

float4 planar_sample(float2 uv) { return planar_reflection.Sample(uv); }
#endif

#ifdef GLASS_RT
// The ray-tracing scene resources. On Metal they ride the transparent pass's
// otherwise-free fragment buffers (0..4 and 8..10, since 5/6/7 are the view,
// the params and the probe set); on Vulkan they are a set of their own, past
// the view / params / global sets glass already owns; on DirectX they follow
// the registers glass already occupies.
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

#ifdef GLASS_RT
{RT_TRACE}
#endif

// ---- Stage interface ----

struct GlassVertexIn
{
    [[vk::location(0)]] float3 pos : POSITION;
};

struct GlassVertexOut
{
    [[vk::location(0)]] float3 world_pos : TEXCOORD0;
    float4 position : SV_Position;
};

[shader("vertex")]
GlassVertexOut glass_vertex(GlassVertexIn v)
{
    GlassVertexOut o;
    // Quad vertices are pre-transformed into world space at build time.
    o.world_pos = v.pos;
    o.position = mul(view.vp, float4(v.pos, 1.0));
    return o;
}

// Depth stored at this pixel by the main pass, for the manual occlusion test.
float glass_scene_depth(int2 pixel)
{
#if USE_MSAA
    return scene_depth.Load(pixel, 0);
#else
    return scene_depth.Load(int3(pixel, 0));
#endif
}

// The pane surface at this fragment: the view-facing normal (two-sided, so a
// pane lit from behind still Fresnels correctly), the fragment's screen UV and
// the refracted, tinted background behind it.
struct GlassSurface
{
    float3 normal;
    float2 frag_uv;
    float3 refracted;
};

GlassSurface glass_surface(float4 position, float3 view_dir)
{
    GlassSurface s;
    s.normal = normalize(params.normal.xyz);
    if (dot(s.normal, view_dir) < 0.0)
    {
        s.normal = -s.normal;
    }

    float2 vp_dim = max(view.viewport, float2(1.0));
    s.frag_uv = position.xy / vp_dim;

    // Refraction: perturb the screen lookup by the pane normal's screen-plane
    // component so the background bends across the pane.
    float2 refract_uv = clamp(s.frag_uv + s.normal.xy * params.refraction_strength,
                              float2(0.001), float2(0.999));
    s.refracted = scene_sample(refract_uv).rgb * params.tint.rgb;
    return s;
}

// Schlick Fresnel (F0 = 0.04 dielectric) mix of the reflection over the
// refraction: ~4% head-on, rising to a full mirror at grazing. `fresnel_power`
// stays the author's grazing-rim shaping control for the opacity ramp.
float4 glass_resolve(GlassSurface s, float3 view_dir, float3 reflection)
{
    float n_dot_v = saturate(dot(s.normal, view_dir));
    float rim = pow(1.0 - n_dot_v, max(params.fresnel_power, 1e-3));
    float refl_weight = saturate(0.04 + 0.96 * rim);
    float3 colour = lerp(s.refracted, reflection, refl_weight);
    float alpha = saturate(lerp(params.opacity, 1.0, rim));
    return float4(colour, alpha);
}

// The reflection a ray that hit nothing (or a pane with no trace at all) falls
// back to: the box-projected probe set where a probe actually covers this pane,
// else the sky prefilter cube, else a white rim so a probe-less, env-less world
// still reads as glass. A pane is smooth, so every path is sharp (mip 0).
float3 glass_environment(float3 world_pos, float3 r)
{
    if (probe_set.count > 0u && probe_set_covers(world_pos))
    {
        return probe_set_specular(world_pos, r, 0.0);
    }
    return view.prefilter_mip_count > 0.5 ? prefilter_level(r, 0.0) : float3(1.0);
}

#ifdef GLASS_RT

[shader("fragment")]
float4 glass_rt_fragment(GlassVertexOut i) : SV_Target
{
    float3 view_dir = normalize(view.camera_pos.xyz - i.world_pos);
    GlassSurface s = glass_surface(i.position, view_dir);

    int2 pixel = min(int2(i.position.xy), int2(max(view.viewport, float2(1.0))) - int2(1, 1));
    if (glass_scene_depth(pixel) < i.position.z)
    {
        discard;
    }

    // A per-pixel reflection ray off the world-space pane surface point, so a
    // window mirrors real off-screen geometry. A pane is smooth, so the trace is
    // sharp and the miss falls back to the probe / sky chain.
    float3 r = reflect(-view_dir, s.normal);
    float3 reflection;
    if (!rt_trace_reflection(i.world_pos + s.normal * 0.02, r,
                             view.prefilter_mip_count > 0.5,
                             view.prefilter_mip_count - 1.0, reflection))
    {
        reflection = glass_environment(i.world_pos, r);
    }
    return glass_resolve(s, view_dir, reflection);
}

#else

[shader("fragment")]
float4 glass_fragment(GlassVertexOut i) : SV_Target
{
    float3 view_dir = normalize(view.camera_pos.xyz - i.world_pos);
    GlassSurface s = glass_surface(i.position, view_dir);

    // Manual depth occlusion: discard where the stored scene depth at this pixel
    // is nearer than the pane. Every backend rasterises this depth under the
    // same viewport convention the main pass used, so the fragment position
    // lines up with the stored texel.
    int2 pixel = min(int2(i.position.xy), int2(max(view.viewport, float2(1.0))) - int2(1, 1));
    if (glass_scene_depth(pixel) < i.position.z)
    {
        discard;
    }

    // A flat pane is a perfect mirror, so an active planar reflection (the scene
    // re-rendered mirrored across this pane's plane) lands exactly under the
    // reflector and is sampled at the fragment's own screen UV with no
    // distortion.
    float3 r = reflect(-view_dir, s.normal);
    float3 reflection = params.planar > 0.5 ? planar_sample(s.frag_uv).rgb
                                            : glass_environment(i.world_pos, r);
    return glass_resolve(s, view_dir, reflection);
}

#endif
