// Bindless forward pass: vertex + fragment, single source for every backend.
// Compile with -DPOOL_SIZE=<n> and -DMAX_PROBES=<n> (the bindless texture-pool
// and reflection-probe array lengths the device binds).
//
// The shading body is shared; only the resource declarations differ per
// binding model, selected by a define and bridged into the body through the
// UPPER_CASE resource macros and the sampling accessors below:
//
//   default    - ParameterBlock layout: scene resources = Vulkan descriptor
//                set 0 (bindings 0-16 in member order, combined image
//                samplers), objects + texture pool = set 1. DXIL lowers each
//                block to a register space.
//   DXIL_SPLIT - same blocks with split texture + sampler pairs where the
//                combined type's SampleCmp / GetDimensions lowerings are
//                broken on DXIL.
//   METAL_ABI  - the engine's Metal binding layout, which is what the host
//                encoders write: discrete buffers pinned by register() (b/t
//                numbers ARE the Metal buffer indices), the texture-only
//                argument buffer at buffer(7), and the engine sampler block at
//                buffer(10).
//   DXIL_ABI   - the engine's DirectX bindless root signature: every register
//                pinned to the layout in directx/init/pipelines.rs, and the
//                object id taken from the b0 root constant the indirect command
//                writes rather than from an instance-id builtin.
//
// A world Shader compiles from this same file with its hooks spliced at
// SURFACE_VERTEX / SURFACE_FRAGMENT, so it lands on these slots by
// construction and never names one.
//
// The records this binds are `main_types.slang` and the shading model it drives
// is `main_shading.slang`.

#ifndef POOL_SIZE
#define POOL_SIZE 1024
#endif
#ifndef MAX_PROBES
#define MAX_PROBES 8
#endif

// ---- Shared CPU-visible records ----

{MAIN_TYPES}

{PROBE_TYPES}

// ---- Resource bindings ----

#ifdef METAL_ABI

// The engine's Metal argument buffer at buffer(7): texture handles only, in
// the exact member order the host's argument encoder writes (tex_pool first,
// then the shadow / IBL / SSAO / probe / LTC set). Samplers live in the
// separate block below because indirect-command-buffer execution cannot see
// encoder-bound sampler state.
struct BindlessTextures
{
    // Bindless texture pool: [albedo textures..] ++ [normal maps..]. The
    // object record's albedo_index / normal_index address it directly.
    Texture2D<float4> tex_pool[POOL_SIZE];
    Texture2DArray<float4> shadow_map;
    TextureCube<float4> irradiance_cube;
    TextureCube<float4> prefilter_cube;
    // Blurred SSAO occlusion (1x1 white when SSAO is disabled).
    Texture2D<float4> ssao_tex;
    // Local reflection-probe prefiltered radiance, one cube per probe; unused
    // slices alias the sky prefilter.
    TextureCube<float4> probe_cubes[MAX_PROBES];
    // Spot shadow map array: one depth slice per shadow-casting spot light.
    Texture2DArray<float4> spot_shadow_map;
    // The two LTC lookup tables, sampled at (roughness, sqrt(1 - NdV)).
    Texture2D<float4> ltc_matrix;
    Texture2D<float4> ltc_magnitude;
};

// The engine's static samplers, mirrored from the host MTLSamplerDescriptors.
struct EngineSamplers
{
    SamplerState tex_sampler;
    SamplerComparisonState shadow_sampler;
    SamplerState cube_sampler;
};

// register() numbers pin the Metal buffer slots directly (b and t share the
// index space there). The three pads reserve buffers 1-3 -- the vertex stream
// rides buffer(1) and the layout keeps the legacy pass's gaps -- so the
// argument buffer lands at buffer(7) and the sampler block at buffer(10), the
// first free slots when their declarations are reached.
ConstantBuffer<ViewUniforms> view_cb : register(b0);
ConstantBuffer<float4> abi_pad1 : register(b1);
ConstantBuffer<float4> abi_pad2 : register(b2);
ConstantBuffer<float4> abi_pad3 : register(b3);
ConstantBuffer<LightUniforms> lights_cb : register(b4);
ConstantBuffer<ShadowUniforms> shadow_cb : register(b5);
ConstantBuffer<ProbeSet> probe_set_cb : register(b6);
ParameterBlock<BindlessTextures> tex;
// Per-scene local lights (point + spot + area) for the forward pass.
StructuredBuffer<GpuLight> local_lights_sb : register(t8);
StructuredBuffer<GpuObjectData> objects_sb : register(t9);
ParameterBlock<EngineSamplers> samps;
ConstantBuffer<ClusterParams> cluster_cb : register(b11);
// Per-cluster light-index lists the LightCull compute pass writes.
StructuredBuffer<uint> cluster_list_sb : register(t12);
// Spot shadow slice projections, indexed by GpuLight.shadow_index.
StructuredBuffer<SpotShadowData> spot_shadows_sb : register(t13);
StructuredBuffer<AreaLightData> area_lights_sb : register(t14);

#define VIEW view_cb
#define LIGHTS lights_cb
#define SHADOW_UNI shadow_cb
#define PROBE_SET probe_set_cb
#define CLUSTER cluster_cb
#define OBJECTS objects_sb
#define LOCAL_LIGHTS local_lights_sb
#define CLUSTER_LIST cluster_list_sb
#define SPOT_SHADOWS spot_shadows_sb
#define AREA_LIGHTS area_lights_sb

float4 pool_sample(uint idx, float2 uv)
{
    return tex.tex_pool[idx].Sample(samps.tex_sampler, uv);
}
float shadow_map_cmp(float3 uv_layer, float ref)
{
    return tex.shadow_map.SampleCmp(samps.shadow_sampler, uv_layer, ref);
}
float spot_shadow_cmp(float3 uv_layer, float ref)
{
    return tex.spot_shadow_map.SampleCmp(samps.shadow_sampler, uv_layer, ref);
}
float2 shadow_map_size()
{
    uint w, h, e;
    tex.shadow_map.GetDimensions(w, h, e);
    return float2(float(w), float(h));
}
float2 spot_shadow_map_size()
{
    uint w, h, e;
    tex.spot_shadow_map.GetDimensions(w, h, e);
    return float2(float(w), float(h));
}
float ssao_sample(float2 uv)
{
    // The cube sampler doubles as the SSAO sampler: the occlusion texture
    // wants linear + clamp-to-edge (a repeat sampler would wrap the border
    // texels at the screen edges), which is exactly its filter state.
    return tex.ssao_tex.Sample(samps.cube_sampler, uv).r;
}
float2 ssao_size()
{
    uint w, h;
    tex.ssao_tex.GetDimensions(w, h);
    return float2(float(w), float(h));
}
float3 irradiance_sample(float3 n)
{
    return tex.irradiance_cube.Sample(samps.cube_sampler, SKY_DIR(n)).rgb;
}
float3 prefilter_sample_level0(float3 dir)
{
    return tex.prefilter_cube.SampleLevel(samps.cube_sampler, SKY_DIR(dir), 0.0).rgb;
}
float3 prefilter_sample_bias(float3 dir, float lod)
{
    return tex.prefilter_cube.SampleBias(samps.cube_sampler, SKY_DIR(dir), lod).rgb;
}
float3 probe_cube_sample_bias(uint i, float3 dir, float lod)
{
    return tex.probe_cubes[i].SampleBias(samps.cube_sampler, dir, lod).rgb;
}
float4 ltc_matrix_sample(float2 uv)
{
    return tex.ltc_matrix.SampleLevel(samps.cube_sampler, uv, 0.0);
}
float2 ltc_magnitude_sample(float2 uv)
{
    return tex.ltc_magnitude.SampleLevel(samps.cube_sampler, uv, 0.0).xy;
}

#elif defined(DXIL_ABI)

// Every register below is pinned to the bindless main root signature in
// directx/init/pipelines.rs. Root constant at b0, root CBVs at b1-b5, root SRVs
// at t1/t2/t3/t15/t17, descriptor tables for the rest, and the unbounded
// texture pool in space1.
struct ObjectId { uint value; };

ConstantBuffer<ObjectId> objid_cb : register(b0);
ConstantBuffer<ViewUniforms> view_cb : register(b1);
ConstantBuffer<LightUniforms> lights_cb : register(b2);
ConstantBuffer<ShadowUniforms> shadow_cb : register(b3);
ConstantBuffer<ProbeSet> probe_set_cb : register(b4);
ConstantBuffer<ClusterParams> cluster_cb : register(b5);

Texture2DArray<float> shadow_map : register(t0);
// Per-scene local lights (point + spot + area) for the forward pass.
StructuredBuffer<GpuLight> local_lights_sb : register(t1);
// Per-cluster light-index lists the LightCull compute pass writes.
StructuredBuffer<uint> cluster_list_sb : register(t2);
StructuredBuffer<GpuObjectData> objects_sb : register(t3);
// Blurred SSAO occlusion (1x1 white when SSAO is disabled).
Texture2D<float4> ssao_tex : register(t4);
TextureCube<float4> irradiance_cube : register(t5);
TextureCube<float4> prefilter_cube : register(t6);
// Local reflection-probe prefiltered radiance; unbaked slots hold the sky
// prefilter cube, so a sample at any index is valid.
TextureCube<float4> probe_cubes[MAX_PROBES] : register(t7);
// Spot shadow slice projections, indexed by GpuLight.shadow_index.
StructuredBuffer<SpotShadowData> spot_shadows_sb : register(t15);
Texture2DArray<float> spot_shadow_map : register(t16);
StructuredBuffer<AreaLightData> area_lights_sb : register(t17);
// The two LTC lookup tables, sampled at (roughness, sqrt(1 - NdV)).
Texture2D<float4> ltc_matrix : register(t18);
Texture2D<float4> ltc_magnitude : register(t19);
// Bindless texture pool: [albedo textures..] ++ [normal maps..]. Unbounded so
// the shader never over-declares the host's per-frame descriptor region.
Texture2D<float4> tex_pool[] : register(t0, space1);

SamplerComparisonState shadow_sampler : register(s0);
SamplerState linear_sampler : register(s1);
SamplerState cube_sampler : register(s2);

#define VIEW view_cb
#define LIGHTS lights_cb
#define SHADOW_UNI shadow_cb
#define PROBE_SET probe_set_cb
#define CLUSTER cluster_cb
#define OBJECTS objects_sb
#define LOCAL_LIGHTS local_lights_sb
#define CLUSTER_LIST cluster_list_sb
#define SPOT_SHADOWS spot_shadows_sb
#define AREA_LIGHTS area_lights_sb

float4 pool_sample(uint idx, float2 uv)
{
    return tex_pool[NonUniformResourceIndex(idx)].Sample(linear_sampler, uv);
}
float shadow_map_cmp(float3 uv_layer, float ref)
{
    return shadow_map.SampleCmp(shadow_sampler, uv_layer, ref);
}
float spot_shadow_cmp(float3 uv_layer, float ref)
{
    return spot_shadow_map.SampleCmp(shadow_sampler, uv_layer, ref);
}
float2 shadow_map_size()
{
    uint w, h, e;
    shadow_map.GetDimensions(w, h, e);
    return float2(float(w), float(h));
}
float2 spot_shadow_map_size()
{
    uint w, h, e;
    spot_shadow_map.GetDimensions(w, h, e);
    return float2(float(w), float(h));
}
float ssao_sample(float2 uv)
{
    // Clamp-to-edge filtering: a repeat sampler would wrap the border texels
    // at the screen edges. The cube sampler carries exactly that state.
    return ssao_tex.Sample(cube_sampler, uv).r;
}
float2 ssao_size()
{
    uint w, h;
    ssao_tex.GetDimensions(w, h);
    return float2(float(w), float(h));
}
float3 irradiance_sample(float3 n)
{
    return irradiance_cube.Sample(cube_sampler, SKY_DIR(n)).rgb;
}
float3 prefilter_sample_level0(float3 dir)
{
    return prefilter_cube.SampleLevel(cube_sampler, SKY_DIR(dir), 0.0).rgb;
}
float3 prefilter_sample_bias(float3 dir, float lod)
{
    return prefilter_cube.SampleBias(cube_sampler, SKY_DIR(dir), lod).rgb;
}
float3 probe_cube_sample_bias(uint i, float3 dir, float lod)
{
    return probe_cubes[i].SampleBias(cube_sampler, dir, lod).rgb;
}
float4 ltc_matrix_sample(float2 uv)
{
    return ltc_matrix.SampleLevel(cube_sampler, uv, 0.0);
}
float2 ltc_magnitude_sample(float2 uv)
{
    return ltc_magnitude.SampleLevel(cube_sampler, uv, 0.0).xy;
}

#else // ParameterBlock layout (Vulkan descriptor sets / DXIL register spaces)

struct SceneResources
{
    ConstantBuffer<ViewUniforms> view;
    ConstantBuffer<LightUniforms> lights;
    ConstantBuffer<ShadowUniforms> shadow_uni;
    // The shadow arrays and the SSAO texture are combined texture-samplers on
    // the targets whose descriptor model wants them fused (the engine's
    // Vulkan layout uses COMBINED_IMAGE_SAMPLER; Metal lowers the pair
    // itself), and split texture + sampler pairs on DXIL, where the combined
    // type's SampleCmp / GetDimensions lowerings are broken.
#ifdef DXIL_SPLIT
    Texture2DArray<float4> shadow_map_t;
    SamplerComparisonState shadow_map_s;
#else
    Sampler2DArray<float4> shadow_map;
#endif
    SamplerCube<float4> irradiance_cube;
    SamplerCube<float4> prefilter_cube;
    // Blurred SSAO occlusion (1x1 white when SSAO is disabled).
#ifdef DXIL_SPLIT
    Texture2D<float4> ssao_tex_t;
    SamplerState ssao_tex_s;
#else
    Sampler2D<float4> ssao_tex;
#endif
    ConstantBuffer<ProbeSet> probe_set;
    SamplerCube<float4> probe_cubes[MAX_PROBES];
    // Per-scene local lights (point + spot + area) for the forward pass.
    StructuredBuffer<GpuLight> local_lights;
    ConstantBuffer<ClusterParams> cluster;
    // Per-cluster light-index lists the LightCull compute pass writes.
    StructuredBuffer<uint> cluster_list;
    // Spot shadow depth array: one layer per shadow-casting spot.
#ifdef DXIL_SPLIT
    Texture2DArray<float4> spot_shadow_map_t;
    SamplerComparisonState spot_shadow_map_s;
#else
    Sampler2DArray<float4> spot_shadow_map;
#endif
    StructuredBuffer<SpotShadowData> spot_shadows;
    StructuredBuffer<AreaLightData> area_lights;
    // The two LTC lookup tables, sampled at (roughness, sqrt(1 - NdV)).
    Sampler2D<float4> ltc_matrix;
    Sampler2D<float4> ltc_magnitude;
};

struct ObjectResources
{
    StructuredBuffer<GpuObjectData> objects;
    // Bindless texture pool: [albedo textures..] ++ [normal maps..]. The
    // object record's albedo_index / normal_index address it directly.
    Sampler2D<float4> tex_pool[POOL_SIZE];
};

ParameterBlock<SceneResources> scene;
ParameterBlock<ObjectResources> objs;

#define VIEW scene.view
#define LIGHTS scene.lights
#define SHADOW_UNI scene.shadow_uni
#define PROBE_SET scene.probe_set
#define CLUSTER scene.cluster
#define OBJECTS objs.objects
#define LOCAL_LIGHTS scene.local_lights
#define CLUSTER_LIST scene.cluster_list
#define SPOT_SHADOWS scene.spot_shadows
#define AREA_LIGHTS scene.area_lights

// NonUniformResourceIndex is required on the descriptor-indexing targets but
// rejected by the Metal backend, where argument-buffer indexing needs no
// annotation. The METAL_ABI block above never reaches this helper, but the
// `metal` case keeps the default block target-portable too.
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

float4 pool_sample(uint idx, float2 uv)
{
    return objs.tex_pool[nonuniform_index(idx)].Sample(uv);
}

// Sampling and size queries on the split-vs-combined resources above, so the
// shader body stays identical across the two declaration forms.
#ifdef DXIL_SPLIT

float shadow_map_cmp(float3 uv_layer, float ref)
{
    return scene.shadow_map_t.SampleCmp(scene.shadow_map_s, uv_layer, ref);
}
float spot_shadow_cmp(float3 uv_layer, float ref)
{
    return scene.spot_shadow_map_t.SampleCmp(scene.spot_shadow_map_s, uv_layer, ref);
}
float2 shadow_map_size()
{
    uint w, h, e;
    scene.shadow_map_t.GetDimensions(w, h, e);
    return float2(float(w), float(h));
}
float2 spot_shadow_map_size()
{
    uint w, h, e;
    scene.spot_shadow_map_t.GetDimensions(w, h, e);
    return float2(float(w), float(h));
}
float ssao_sample(float2 uv)
{
    return scene.ssao_tex_t.Sample(scene.ssao_tex_s, uv).r;
}
float2 ssao_size()
{
    uint w, h;
    scene.ssao_tex_t.GetDimensions(w, h);
    return float2(float(w), float(h));
}

#else

float shadow_map_cmp(float3 uv_layer, float ref)
{
    return scene.shadow_map.SampleCmp(uv_layer, ref);
}
float spot_shadow_cmp(float3 uv_layer, float ref)
{
    return scene.spot_shadow_map.SampleCmp(uv_layer, ref);
}

// The Metal target reads dimensions through inline MSL: the DXIL-style
// GetDimensions overloads on combined types do not lower there. The resource
// rides as a parameter so the inline form can reference it.
float2 combined_array_size(Sampler2DArray<float4> s)
{
    __target_switch
    {
    case metal:
        __intrinsic_asm "float2((*$0).texture_0.get_width(), (*$0).texture_0.get_height())";
    default:
        uint w, h, e;
        s.GetDimensions(w, h, e);
        return float2(float(w), float(h));
    }
}
float2 combined_size(Sampler2D<float4> s)
{
    __target_switch
    {
    case metal:
        __intrinsic_asm "float2((*$0).texture_1.get_width(), (*$0).texture_1.get_height())";
    default:
        uint w, h;
        s.GetDimensions(w, h);
        return float2(float(w), float(h));
    }
}
float2 shadow_map_size()
{
    return combined_array_size(scene.shadow_map);
}
float2 spot_shadow_map_size()
{
    return combined_array_size(scene.spot_shadow_map);
}
float ssao_sample(float2 uv)
{
    return scene.ssao_tex.Sample(uv).r;
}
float2 ssao_size()
{
    return combined_size(scene.ssao_tex);
}

#endif

float3 irradiance_sample(float3 n)
{
    return scene.irradiance_cube.Sample(SKY_DIR(n)).rgb;
}
float3 prefilter_sample_level0(float3 dir)
{
    return scene.prefilter_cube.SampleLevel(SKY_DIR(dir), 0.0).rgb;
}
float3 prefilter_sample_bias(float3 dir, float lod)
{
    return scene.prefilter_cube.SampleBias(SKY_DIR(dir), lod).rgb;
}
float3 probe_cube_sample_bias(uint i, float3 dir, float lod)
{
    return scene.probe_cubes[i].SampleBias(dir, lod).rgb;
}
float4 ltc_matrix_sample(float2 uv)
{
    return scene.ltc_matrix.SampleLevel(uv, 0.0);
}
float2 ltc_magnitude_sample(float2 uv)
{
    return scene.ltc_magnitude.SampleLevel(uv, 0.0).xy;
}

#endif // binding model

{PROBE_COMMON}

// The reflection tap `shade_surface` reads: the bound probe set where any probe
// is baked, else the imported environment prefilter cube.
float3 environment_specular(float3 world_pos, float3 R, float lod)
{
    return (PROBE_SET.count > 0u) ? probe_set_specular(world_pos, R, lod)
                                  : prefilter_sample_bias(R, lod);
}

{MAIN_SHADING}

// ---- The world's hooks ----

// A world Shader defines these; the engine's defaults delegate to
// `project_vertex` and `shade_surface`. Both stages compile from this one
// variant, so both hooks are spliced here.
VertexOut transform(float4x4 model, float3 pos, float3 normal, float3 tangent,
                    float3 color, float2 uv);
float4 shade(VertexOut in, GpuObjectData od);

{SURFACE_VERTEX}

{SURFACE_FRAGMENT}

// ---- Vertex ----

[shader("vertex")]
VertexOut vertex_main_bindless(
    VertexIn v
#ifdef DXIL_ABI
    // DirectX writes the object id into the b0 root constant ahead of each
    // indirect draw, so no instance-id builtin is read: SV_StartInstanceLocation
    // would raise the DXIL floor to shader model 6.8 for nothing.
    )
{
    uint oid = objid_cb.value;
#else
    ,
    uint instance_id : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    uint oid = object_instance_index(instance_id, first_instance);
#endif
    VertexOut o = transform(OBJECTS[oid].model, v.pos, v.normal, v.tangent, v.color, v.uv);
    o.object_id = oid;
    return o;
}

// ---- Fragment ----

// The object id joins the varyings fold: a world `shade` that ignores its
// record lets the compiler drop the only read of `in.object_id`, and the
// vertex output at its location then has no consumer.
[shader("fragment")]
float4 fragment_main_bindless(VertexOut in) : SV_Target
{
    float4 shaded = shade(in, OBJECTS[in.object_id]);
    return varyings_read(in) + float(in.object_id) > 1e30 ? float4(0.0) : shaded;
}
