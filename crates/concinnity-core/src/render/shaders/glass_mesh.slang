// See-through glass mesh pass: single source for every backend.
//
// The third producer of the engine's transparent pass (`PassId::Transparent`,
// after the SSR resolve and before TAA), alongside `glass.slang` and
// `water.slang`. Where a glass pane is a flat pre-baked world-space quad, this
// draws an IMPORTED mesh whose `Material` is flagged `see_through`: the geometry
// comes from the shared scene vertex / index buffers in LOCAL space, so the
// vertex stage applies the per-draw model matrix and the fragment shades off the
// interpolated per-vertex world normal, which is what lets a curved glass facade
// reflect correctly across its surface.
//
// The pass is ray-traced only, by design: what makes the mesh see-through rather
// than the opaque low-roughness glass of Layer 1 is a real per-pixel reflection
// ray, so there is no probe-only variant to fall back to. When RT is off the
// host leaves these meshes in the opaque pass instead of drawing them here.
// Two fragment entries, differing only in where a reflected hit's surface
// parameters come from:
//
//   default      - the reflected hit takes its flat per-object material tint.
//   RT_TEXTURED  - the same trace, with reflected hits taking their albedo /
//                  normal / emissive maps from the bindless pool.
//
// USE_MSAA is a HOST difference rather than a target one: Vulkan reads the
// multisampled main depth while Metal and DirectX read the resolved copy.
//
// Every binding here is deliberately the one glass.slang and water.slang declare
// at the same slot, because the three share a transparent pass: on DirectX the
// RT root signature serves all of them, on Vulkan one set of descriptor set
// layouts, and on Metal the encoder binds the shared reflection inputs once for
// the whole pass. A slot may not move on one side alone.

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
{RT_TYPES}

// Per-frame view shared by every transparent draw. Layout matches
// `TransparentView` (160 B); identical to glass.slang's copy.
struct TransparentView
{
    float4x4 vp;       // world -> clip (jittered when TAA is on)
    float4x4 inv_vp;   // clip -> world
    float4 camera_pos; // xyz: world-space camera
    float2 viewport;   // attachment dimensions in pixels
    float time;        // seconds since startup
    float prefilter_mip_count;
};

// Per-mesh tunables. Layout matches `GlassMeshParams` (96 B); `model` is first
// so its 16-byte alignment is satisfied at offset 0.
struct GlassMeshParams
{
    float4x4 model; // local -> world
    float4 tint;    // colour multiplied into the refracted scene
    float opacity;
    float refraction_strength;
    float fresnel_power;
    // The mesh's own copy of the sky prefilter mip count, so the ray-miss
    // fallback does not depend on which view block is bound. 0 = no
    // EnvironmentMap, and the reflection keeps the white rim.
    float prefilter_mip_count;
};

// ---- Resource bindings ----

#ifdef DXIL_ABI

// Pinned to the RT root signature in `directx/transparent.rs`, which the pane
// and water producers share; b1 is visible to every stage there, and this
// vertex stage reads the model matrix out of it.
ConstantBuffer<TransparentView> view : register(b0);
ConstantBuffer<GlassMeshParams> params : register(b1);
Texture2D<float4> scene_color : register(t0);
#if USE_MSAA
Texture2DMS<float> scene_depth : register(t1);
#else
Texture2D<float> scene_depth : register(t1);
#endif
TextureCube<float4> prefilter_cube : register(t2);
SamplerState post_samp : register(s0);
SamplerState cube_sampler : register(s2);

float4 scene_sample(float2 uv) { return scene_color.Sample(post_samp, uv); }
float3 prefilter_level(float3 dir, float lod)
{
    return prefilter_cube.SampleLevel(cube_sampler, dir, lod).rgb;
}

#else

// Metal buffer(5) / buffer(6): the shared per-frame view and the per-mesh
// params, both written with setBytes by the transparent encoder.
[[vk::binding(0, 0)]] ConstantBuffer<TransparentView> view : register(b5);
[[vk::binding(0, 1)]] ConstantBuffer<GlassMeshParams> params : register(b6);

// Declaration order is the Metal texture index, and these are the transparent
// pass's shared slots: the scene snapshot at 0, the resolved depth at 1, the sky
// prefilter cube at 2, and the probe cubes at 3..2+MAX_PROBES.
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

// The reflection-probe set + cube array, from the forward global set: a glass
// mesh falls back to the same local scene capture the forward IBL specular and
// the SSR / RT miss fallback use, rather than only the foreign sky cube.
#ifdef DXIL_ABI
ConstantBuffer<ProbeSet> probe_set : register(b4);
// The array spans MAX_PROBES registers from its base, so it starts clear of the
// ray-tracing SRVs at t4..t10. Always the RT layout here: this pass has no
// probe-only variant.
TextureCube<float4> probe_cubes[MAX_PROBES] : register(t20);

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
// A glass mesh never samples a planar reflection -- it is curved, and it traces
// a sharper reflection than a mirror render anyway. The declaration is here
// because slangc numbers Metal's texture and sampler slots by declaration order,
// and the transparent encoder binds one slot map for all three of its producers:
// dropping this would slide the bindless pool's sampler off the index the
// encoder binds it at. glass.slang declares it at the same point for the same
// reason.
[[vk::binding(1, 1)]] Sampler2D<float4> planar_reflection;
#endif

// The ray-tracing scene resources, at the slots glass.slang uses so the inputs
// the transparent encoder binds once are valid for all three producers. On Metal
// they ride the pass's otherwise-free fragment buffers (0..4 and 8..10, since
// 5/6/7 are the view, the params and the probe set); on Vulkan they are a set of
// their own, past the view / params / global sets; on DirectX they follow the
// registers the pass already occupies.
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

{PROBE_COMMON}

{RT_TRACE}

// ---- Stage interface ----

struct GlassMeshVertexIn
{
    [[vk::location(0)]] float3 pos : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
};

struct GlassMeshVertexOut
{
    [[vk::location(0)]] float3 world_pos : TEXCOORD0;
    [[vk::location(1)]] float3 world_normal : TEXCOORD1;
    float4 position : SV_Position;
};

[shader("vertex")]
GlassMeshVertexOut glass_mesh_vertex(GlassMeshVertexIn v)
{
    GlassMeshVertexOut o;
    float4 world = mul(params.model, float4(v.pos, 1.0));
    o.world_pos = world.xyz;
    // Rigid / uniform-scale model, so `M * n` needs no inverse-transpose. The
    // fragment renormalises after interpolation.
    o.world_normal = mul((float3x3)params.model, v.normal);
    o.position = mul(view.vp, world);
    return o;
}

// Depth stored at this pixel by the main pass, for the manual occlusion test.
float glass_mesh_scene_depth(int2 pixel)
{
#if USE_MSAA
    return scene_depth.Load(pixel, 0);
#else
    return scene_depth.Load(int3(pixel, 0));
#endif
}

// The mesh surface at this fragment: the view-facing interpolated normal
// (two-sided, so a pane of glass lit from behind still Fresnels correctly), the
// fragment's screen UV and the refracted, tinted background behind it.
struct GlassMeshSurface
{
    float3 normal;
    float2 frag_uv;
    float3 refracted;
};

GlassMeshSurface glass_mesh_surface(GlassMeshVertexOut i, float3 view_dir)
{
    GlassMeshSurface s;
    s.normal = normalize(i.world_normal);
    if (dot(s.normal, view_dir) < 0.0)
    {
        s.normal = -s.normal;
    }

    float2 vp_dim = max(view.viewport, float2(1.0));
    s.frag_uv = i.position.xy / vp_dim;

    float2 refract_uv = clamp(s.frag_uv + s.normal.xy * params.refraction_strength,
                              float2(0.001), float2(0.999));
    s.refracted = scene_sample(refract_uv).rgb * params.tint.rgb;
    return s;
}

// Schlick Fresnel (F0 = 0.04 dielectric) mix of the reflection over the
// refraction, identical to `glass_resolve` so a mesh and a pane read the same at
// equal inputs.
float4 glass_mesh_resolve(GlassMeshSurface s, float3 view_dir, float3 reflection)
{
    float n_dot_v = saturate(dot(s.normal, view_dir));
    float rim = pow(1.0 - n_dot_v, max(params.fresnel_power, 1e-3));
    float refl_weight = saturate(RT_F0 + 0.96 * rim);
    float3 colour = lerp(s.refracted, reflection, refl_weight);
    float alpha = saturate(lerp(params.opacity, 1.0, rim));
    return float4(colour, alpha);
}

[shader("fragment")]
float4 glass_mesh_rt_fragment(GlassMeshVertexOut i) : SV_Target
{
    float3 view_dir = normalize(view.camera_pos.xyz - i.world_pos);
    GlassMeshSurface s = glass_mesh_surface(i, view_dir);

    int2 pixel = min(int2(i.position.xy), int2(max(view.viewport, float2(1.0))) - int2(1, 1));
    if (glass_mesh_scene_depth(pixel) < i.position.z)
    {
        discard;
    }

    // A per-pixel reflection ray off the interpolated world-space surface point,
    // so a curved facade mirrors real off-screen geometry across its whole span.
    // The mesh is excluded from the BLAS (glass does not reflect glass), so the
    // trace never self-hits. Glass is smooth, so the trace is sharp and the miss
    // falls back to the probe / sky chain.
    bool ibl = params.prefilter_mip_count > 0.5;
    float3 r = reflect(-view_dir, s.normal);
    float3 reflection;
    if (!rt_trace_reflection(i.world_pos + s.normal * 0.02, r, ibl,
                             params.prefilter_mip_count - 1.0, reflection))
    {
        if (probe_set.count > 0u && probe_set_covers(i.world_pos))
        {
            reflection = probe_set_specular(i.world_pos, r, 0.0);
        }
        else
        {
            reflection = ibl ? prefilter_level(r, 0.0) : float3(1.0);
        }
    }
    return glass_mesh_resolve(s, view_dir, reflection);
}
