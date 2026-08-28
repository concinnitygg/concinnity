// Hardware ray-traced reflections: single source for every backend.
//
// A fullscreen fragment pass that, per glossy pixel, rebuilds a world-space
// surface point + normal from the SSR pre-pass G-buffer, traces a reflection
// ray against the scene's acceleration structure, shades the hit (or falls back
// to the local reflection probe / IBL prefilter cube on a miss) and writes
// reflected radiance (.rgb) + the Fresnel/gloss composite weight (.a). The
// reflection blur + composite then blends it over the scene, exactly as they do
// for the SSR resolve. Unlike SSR the ray is a real world-space trace, so
// reflected geometry that is off-screen still appears. Pairs with
// `fullscreen_vertex` in fullscreen.slang.
//
// The traversal itself is the shared RT_TRACE fragment, so this pass and glass
// run one traversal loop between them; only the bindings and the miss fallback
// are the pass's own.
//
// One entry per compile, selected by a define, so each variant declares exactly
// the resources it binds (Metal and DXIL indices are assigned in declaration
// order, and an unused declaration would still hold its slot):
//
//   default     - flat: the per-object material tint as albedo, the fallback a
//                 non-bindless world takes.
//   RT_TEXTURED - samples the hit's albedo / normal / emissive maps from the
//                 bindless texture pool, the path standard worlds take.
//
// DXIL_ABI pins every register to the root signature in
// `directx/post/rt_reflections.rs` and keeps DirectX's raw vertex / index SRVs
// (`ByteAddressBuffer`); the other targets carry one declaration with both a
// `[[vk::binding]]` and a `register()`, whose number IS the Metal buffer index.
// The DXIL container needs shader model 6.5 for ray query, above the 6.0 floor
// the bindless main pass established.

#ifndef POOL_SIZE
#define POOL_SIZE 1024
#endif
#ifndef MAX_PROBES
#define MAX_PROBES 8
#endif

{PROBE_TYPES}

{RT_TYPES}

// ---- Resource bindings ----

#ifdef DXIL_ABI

ConstantBuffer<RtParams> rt_params : register(b0);
RaytracingAccelerationStructure scene_tlas : register(t0);
// Raw vertex / index SRVs: DirectX binds these as byte-address views.
ByteAddressBuffer verts : register(t1);
ByteAddressBuffer indices : register(t2);
StructuredBuffer<RtGeomEntry> geom : register(t3);
ByteAddressBuffer sverts : register(t8);
ByteAddressBuffer sidx : register(t9);

Texture2D<float4> scene_tex : register(t4);
Texture2D<float4> gbuffer : register(t5);
Texture2D<float4> rough_tex : register(t6);
TextureCube<float4> prefilter : register(t7);
SamplerState screen_sampler : register(s0);
SamplerState cube_sampler : register(s1);

float vert_float(uint i) { return asfloat(verts.Load(i * 4u)); }
float svert_float(uint i) { return asfloat(sverts.Load(i * 4u)); }
uint index_at(uint o) { return indices.Load(o * 4u); }
uint skinned_index_word(uint w) { return sidx.Load(w * 4u); }

float4 screen_sample(Texture2D<float4> t, float2 uv) { return t.Sample(screen_sampler, uv); }
float3 prefilter_level(float3 dir, float lod)
{
    return prefilter.SampleLevel(cube_sampler, dir, lod).rgb;
}

#else

[[vk::binding(0, 0)]] ConstantBuffer<RtParams> rt_params : register(b0);
[[vk::binding(1, 0)]] RaytracingAccelerationStructure scene_tlas : register(t4);
[[vk::binding(2, 0)]] StructuredBuffer<RtGeomEntry> geom : register(t3);
// The shared static vertex stream (14 floats / 56 B stride) + its u32 indices.
[[vk::binding(3, 0)]] StructuredBuffer<float> verts : register(t1);
[[vk::binding(4, 0)]] StructuredBuffer<uint> indices : register(t2);
// The deformed (posed) skinned vertex buffer, in the same 14-float layout the
// skin kernel writes, + the skinned index buffer (two indices per uint).
// Both bind a 1-element dummy in a scene with no skinned geometry, so the
// binding stays valid even though the skinned branch is never taken there.
[[vk::binding(9, 0)]] StructuredBuffer<float> sverts : register(t5);
[[vk::binding(10, 0)]] StructuredBuffer<uint> sidx : register(t6);

// Screen-space inputs reused from the SSR resolve. Declaration order is the
// Metal texture index: scene(0), gbuffer(1), roughness(2), prefilter(3), and
// the probe cube array below takes 4..3+MAX_PROBES.
[[vk::binding(5, 0)]] Sampler2D<float4> scene_tex;
[[vk::binding(6, 0)]] Sampler2D<float4> gbuffer;
[[vk::binding(7, 0)]] Sampler2D<float4> rough_tex;
[[vk::binding(8, 0)]] SamplerCube<float4> prefilter;

float vert_float(uint i) { return verts[i]; }
float svert_float(uint i) { return sverts[i]; }
uint index_at(uint o) { return indices[o]; }
uint skinned_index_word(uint w) { return sidx[w]; }

float4 screen_sample(Sampler2D<float4> t, float2 uv) { return t.Sample(uv); }
float3 prefilter_level(float3 dir, float lod) { return prefilter.SampleLevel(dir, lod).rgb; }

#endif

// The forward global set, bound here only for its reflection-probe count +
// per-probe parallax boxes + cube array: a ray that escapes the scene falls
// back to the local probe capture instead of the foreign sky cube.
#ifdef DXIL_ABI
ConstantBuffer<ProbeSet> probe_set : register(b4);
// D3D12 binds a shader sampler array only through a descriptor table, so the
// probe cubes ride split from the one sampler the root signature hands out
// statically -- the shape ssr.slang's SPLIT_PROBE_SAMPLER and the bindless main
// pass both use for this array.
TextureCube<float4> probe_cubes[MAX_PROBES] : register(t10);
SamplerState probe_cube_sampler : register(s3);

float3 probe_cube_sample_bias(uint i, float3 dir, float lod)
{
    return probe_cubes[i].SampleBias(probe_cube_sampler, dir, lod).rgb;
}
#else
[[vk::binding(7, 1)]] ConstantBuffer<ProbeSet> probe_set : register(b8);
[[vk::binding(8, 1)]] SamplerCube<float4> probe_cubes[MAX_PROBES];

float3 probe_cube_sample_bias(uint i, float3 dir, float lod)
{
    return probe_cubes[i].SampleBias(dir, lod).rgb;
}
#endif
#define PROBE_SET probe_set

#ifdef RT_TEXTURED
// The bindless albedo / normal / emissive pool, in whichever form its host can
// bind: a Metal argument buffer of texture handles (its sampler is bound
// alongside rather than written into the buffer), a Vulkan combined-sampler
// array on the same set the main bindless pass uses, or an unbounded DXIL array
// in space 1. `nonuniform_index` is required on the descriptor-indexing targets
// and rejected by the Metal backend, where argument-buffer indexing needs no
// annotation.
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
SamplerState pool_sampler : register(s2);

float3 pool_sample_level0(uint idx, float2 uv)
{
    return tex_pool[nonuniform_index(idx)].SampleLevel(pool_sampler, uv, 0.0).rgb;
}
#else
[[vk::binding(1, 2)]] Sampler2D<float4> tex_pool[POOL_SIZE];

float3 pool_sample_level0(uint idx, float2 uv)
{
    return tex_pool[nonuniform_index(idx)].SampleLevel(uv, 0.0).rgb;
}
#endif
#endif

// Surfaces rougher than REFLECTION_ROUGHNESS_CUT get no reflection; glossiness
// ramps in below it. Locked to concinnity_core::gfx::ssr::REFLECTION_ROUGHNESS_CUT
// by unit test so the SSR, RT, and composite gates agree.
static const float REFLECTION_ROUGHNESS_CUT = 0.6;

{PROBE_COMMON}

{RT_TRACE}

// Rebuild a view-space position from a UV and its linear (view-space) depth.
// Matches ssr_view_pos in the SSR resolve.
float3 rt_view_pos(float2 uv, float depth, float tan_y, float aspect)
{
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return float3(ndc.x * tan_y * aspect, ndc.y * tan_y, -1.0) * depth;
}

[shader("fragment")]
float4 rt_reflections_fragment([[vk::location(0)]] float2 uv : TEXCOORD0) : SV_Target
{
    float3 base = screen_sample(scene_tex, uv).rgb;
    float4 g = screen_sample(gbuffer, uv);
    float depth = g.a;
    // Background / sky, or a non-reflecting (too-rough) surface: weight 0 so the
    // reflection composite keeps the scene there. The pass writes reflected
    // radiance (.rgb) + composite weight (.a), not a blended colour.
    if (depth <= 0.0)
    {
        return float4(base, 0.0);
    }

    float roughness = screen_sample(rough_tex, uv).r;
    float gloss = saturate((REFLECTION_ROUGHNESS_CUT - roughness) / REFLECTION_ROUGHNESS_CUT);
    if (gloss <= 0.0)
    {
        return float4(base, 0.0);
    }

    float3 nv = normalize(g.xyz);
    float3 pv = rt_view_pos(uv, depth, rt_params.tan_half_fov_y, rt_params.aspect);
    float3 pw = mul(rt_params.inv_view, float4(pv, 1.0)).xyz;
    float3 nw = normalize(mul((float3x3)rt_params.inv_view, nv));
    float3 v = normalize(rt_params.cam_pos.xyz - pw);

    float3 dir = reflect(-v, nw);
    bool ibl = rt_params.prefilter_mip_count > 0.5;
    float max_mip = rt_params.prefilter_mip_count - 1.0;
    float ndv = saturate(dot(nw, v));
    float fresnel = RT_F0 + (1.0 - RT_F0) * pow(1.0 - ndv, 5.0);
    float weight = saturate(fresnel * gloss * rt_params.intensity);

    float3 reflected;
    // Origin nudged off the surface along the normal so the trace cannot
    // self-intersect the pixel's own triangle.
    if (!rt_trace_reflection(pw + nw * 0.01, dir, ibl, max_mip, reflected))
    {
        // The ray escaped the scene: the local reflection probe (box-parallax,
        // blended across covering probes) when one is baked, else the IBL
        // prefilter sky, else the base shading.
        float lod = roughness * max_mip;
        reflected = probe_set.count > 0u ? probe_set_specular(pw, dir, lod)
                  : (ibl ? prefilter_level(dir, lod) : base);
    }

    return float4(reflected, weight);
}
