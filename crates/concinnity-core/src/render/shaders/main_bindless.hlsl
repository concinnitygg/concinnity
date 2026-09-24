// Bindless forward pass: vertex + fragment, single source for every backend.
// Every array it binds is unsized or a single resource -- the texture pool, the
// probe records and the probe cube array -- so no define sizes it: the host
// decides each length at run time.
//
// The shading body is shared; only the resource declarations differ per
// binding model, selected by the backend define and bridged into the body
// through the UPPER_CASE resource macros and the sampling accessors below:
//
//   default            - the engine's Vulkan layout: scene resources and the
//                        engine samplers = descriptor set 0 (bindings 0-20),
//                        objects + texture pool = set 1.
//   CN_BACKEND_METAL   - the engine's Metal binding layout, which is what the
//                        host encoders write: discrete buffers pinned by
//                        register() (b/t numbers ARE the Metal buffer
//                        indices), the texture-only argument buffer at
//                        buffer(7), and the engine sampler block at
//                        buffer(10).
//   CN_BACKEND_DIRECTX - the engine's DirectX bindless root signature: every
//                        register pinned to the layout in
//                        directx/init/pipelines.rs, and the object id taken
//                        from the b0 root constant the indirect command writes
//                        rather than from an instance-id builtin.
//
// A world Shader compiles from this same file with its hooks spliced at
// SURFACE_VERTEX / SURFACE_FRAGMENT, so it lands on these slots by
// construction and never names one.
//
// The records this binds are `main_types.hlsl` and the shading model it drives
// is `main_shading.hlsl`.

// ---- Shared CPU-visible records ----

{MAIN_TYPES}

{PROBE_TYPES}

// ---- Resource bindings ----

#ifdef CN_BACKEND_METAL

// The Metal argument buffer at buffer(7) holds texture handles only, and
// `metal/cull.rs` writes it by argument id. A member's register number IS its
// `[[id(n)]]`: the fixed members take 0..7 and the pool comes last, unsized, so
// no id depends on how many textures the pool holds.
[[vk::binding(0, 1)]] [[cn::metal_argument_buffer(7)]]
Texture2DArray<float> shadow_map : register(t0, space1);
[[vk::binding(1, 1)]] TextureCube<float4> irradiance_cube : register(t1, space1);
[[vk::binding(2, 1)]] TextureCube<float4> prefilter_cube : register(t2, space1);
// Blurred SSAO occlusion (1x1 white when SSAO is disabled).
[[vk::binding(3, 1)]] Texture2D<float4> ssao_tex : register(t3, space1);
// Local reflection-probe prefiltered radiance, one cube-array slice per probe.
[[vk::binding(4, 1)]] TextureCubeArray<float4> probe_cubes : register(t4, space1);
// Spot shadow map array: one depth slice per shadow-casting spot light.
[[vk::binding(5, 1)]] Texture2DArray<float> spot_shadow_map : register(t5, space1);
// The two LTC lookup tables, sampled at (roughness, sqrt(1 - NdV)).
[[vk::binding(6, 1)]] Texture2D<float4> ltc_matrix : register(t6, space1);
[[vk::binding(7, 1)]] Texture2D<float4> ltc_magnitude : register(t7, space1);
// Bindless texture pool: [albedo textures..] ++ [normal maps..]. The object
// record's albedo_index / normal_index address it directly.
[[vk::binding(8, 1)]] Texture2D<float4> tex_pool[] : register(t8, space1);

// The engine's static samplers at buffer(10), mirrored from the host
// MTLSamplerDescriptors. Apart from the textures because indirect-command-buffer
// execution cannot see encoder-bound sampler state.
[[vk::binding(0, 2)]] [[cn::metal_argument_buffer(10)]]
SamplerState tex_sampler : register(s0, space2);
[[vk::binding(1, 2)]] SamplerComparisonState shadow_sampler : register(s1, space2);
[[vk::binding(2, 2)]] SamplerState cube_sampler : register(s2, space2);

// register() numbers pin the Metal buffer slots directly (b and t share the
// index space there). buffer(1) is the vertex stream.
[[vk::binding(0, 0)]] ConstantBuffer<ViewUniforms> view_cb : register(b0);
[[vk::binding(1, 0)]] ConstantBuffer<LightUniforms> lights_cb : register(b4);
[[vk::binding(2, 0)]] ConstantBuffer<ShadowUniforms> shadow_cb : register(b5);
[[vk::binding(3, 0)]] ConstantBuffer<ProbeSet> probe_set_cb : register(b6);
// Per-scene local lights (point + spot + area) for the forward pass.
[[vk::binding(4, 0)]] StructuredBuffer<GpuLight> local_lights_sb : register(t8);
[[vk::binding(5, 0)]] StructuredBuffer<GpuObjectData> objects_sb : register(t9);
[[vk::binding(6, 0)]] ConstantBuffer<ClusterParams> cluster_cb : register(b11);
// Per-cluster light lists and probe masks the LightCull compute pass writes.
[[vk::binding(7, 0)]] StructuredBuffer<uint> cluster_list_sb : register(t12);
// Spot shadow slice projections, indexed by GpuLight.shadow_index.
[[vk::binding(8, 0)]] StructuredBuffer<SpotShadowData> spot_shadows_sb : register(t13);
[[vk::binding(9, 0)]] StructuredBuffer<AreaLightData> area_lights_sb : register(t14);
// One parallax record per live reflection probe.
[[vk::binding(10, 0)]] StructuredBuffer<ProbeUniforms> probe_records_sb : register(t15);

// The pool's sampler, under the name the other hosts declare it by.
#define linear_sampler tex_sampler

#elif defined(CN_BACKEND_DIRECTX)

// Every register below is pinned to the bindless main root signature in
// directx/init/pipelines.rs. Root constant at b0, root CBVs at b1-b5, root SRVs
// at t1/t2/t3/t8/t15/t17, descriptor tables for the rest, and the unbounded
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
// Per-cluster light lists and probe masks the LightCull compute pass writes.
StructuredBuffer<uint> cluster_list_sb : register(t2);
StructuredBuffer<GpuObjectData> objects_sb : register(t3);
// Blurred SSAO occlusion (1x1 white when SSAO is disabled).
Texture2D<float4> ssao_tex : register(t4);
TextureCube<float4> irradiance_cube : register(t5);
TextureCube<float4> prefilter_cube : register(t6);
// Local reflection-probe prefiltered radiance, one cube-array slice per probe,
// and the parallax record of each.
TextureCubeArray<float4> probe_cubes : register(t7);
StructuredBuffer<ProbeUniforms> probe_records_sb : register(t8);
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


#else // the Vulkan descriptor sets

// Set 0 is the forward global set: every texture a sampled image, read through
// the three engine samplers at bindings 18-20 the way the DirectX root
// signature reads its static samplers. Set 1 holds the object records and the
// texture pool.
[[vk::binding(0, 0)]] ConstantBuffer<ViewUniforms> view_cb : register(b0);
[[vk::binding(1, 0)]] ConstantBuffer<LightUniforms> lights_cb : register(b1);
[[vk::binding(2, 0)]] ConstantBuffer<ShadowUniforms> shadow_cb : register(b2);
[[vk::binding(3, 0)]] Texture2DArray<float> shadow_map : register(t3);
[[vk::binding(4, 0)]] TextureCube<float4> irradiance_cube : register(t4);
[[vk::binding(5, 0)]] TextureCube<float4> prefilter_cube : register(t5);
// Blurred SSAO occlusion (1x1 white when SSAO is disabled).
[[vk::binding(6, 0)]] Texture2D<float4> ssao_tex : register(t6);
[[vk::binding(7, 0)]] ConstantBuffer<ProbeSet> probe_set_cb : register(b7);
[[vk::binding(8, 0)]] TextureCubeArray<float4> probe_cubes : register(t8);
// Per-scene local lights (point + spot + area) for the forward pass.
[[vk::binding(9, 0)]] StructuredBuffer<GpuLight> local_lights_sb : register(t9);
[[vk::binding(10, 0)]] ConstantBuffer<ClusterParams> cluster_cb : register(b10);
// Per-cluster light lists and probe masks the LightCull compute pass writes.
[[vk::binding(11, 0)]] StructuredBuffer<uint> cluster_list_sb : register(t11);
// Spot shadow depth array: one layer per shadow-casting spot.
[[vk::binding(12, 0)]] Texture2DArray<float> spot_shadow_map : register(t12);
// Spot shadow slice projections, indexed by GpuLight.shadow_index.
[[vk::binding(13, 0)]] StructuredBuffer<SpotShadowData> spot_shadows_sb : register(t13);
[[vk::binding(14, 0)]] StructuredBuffer<AreaLightData> area_lights_sb : register(t14);
// The two LTC lookup tables, sampled at (roughness, sqrt(1 - NdV)).
[[vk::binding(15, 0)]] Texture2D<float4> ltc_matrix : register(t15);
[[vk::binding(16, 0)]] Texture2D<float4> ltc_magnitude : register(t16);
// One parallax record per live reflection probe.
[[vk::binding(17, 0)]] StructuredBuffer<ProbeUniforms> probe_records_sb : register(t17);
[[vk::binding(18, 0)]] SamplerComparisonState shadow_sampler : register(s0);
[[vk::binding(19, 0)]] SamplerState cube_sampler : register(s1);
[[vk::binding(20, 0)]] SamplerState linear_sampler : register(s2);

[[vk::binding(0, 1)]] StructuredBuffer<GpuObjectData> objects_sb : register(t0, space1);
// Bindless texture pool: [albedo textures..] ++ [normal maps..]. The object
// record's albedo_index / normal_index address it directly.
[[vk::binding(1, 1)]] Texture2D<float4> tex_pool[] : register(t1, space1);


#endif // binding model

#define VIEW view_cb
#define LIGHTS lights_cb
#define SHADOW_UNI shadow_cb
#define PROBE_SET probe_set_cb
#define PROBE_RECORDS probe_records_sb
#define CLUSTER cluster_cb
#define OBJECTS objects_sb
#define LOCAL_LIGHTS local_lights_sb
#define CLUSTER_LIST cluster_list_sb
#define SPOT_SHADOWS spot_shadows_sb
#define AREA_LIGHTS area_lights_sb
#define probe_cube_sampler cube_sampler

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
// The cube sampler clamps to edge, so the occlusion's border texels never wrap.
float ssao_sample(float2 uv)
{
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
float4 ltc_matrix_sample(float2 uv)
{
    return ltc_matrix.SampleLevel(cube_sampler, uv, 0.0);
}
float2 ltc_magnitude_sample(float2 uv)
{
    return ltc_magnitude.SampleLevel(cube_sampler, uv, 0.0).xy;
}

{PROBE_COMMON}

// The reflection tap `shade_surface` reads for a surface of `roughness`, into
// `radiance`: the fragment's `probes` where any probe is baked, else the
// imported environment prefilter cube. False when there is neither.
bool environment_specular(ProbeMask probes, float3 world_pos, float3 R, float roughness,
                          out float3 radiance)
{
    if (PROBE_SET.count > 0u)
    {
        radiance = probe_mask_specular(probes, world_pos, R, probe_lod(roughness));
        return true;
    }
    if (VIEW.prefilter_mip_count > 0.5)
    {
        radiance = prefilter_sample_bias(R, roughness * (VIEW.prefilter_mip_count - 1.0));
        return true;
    }
    radiance = (float3)(0.0);
    return false;
}

{MAIN_SHADING}

// ---- The world's hooks ----

// A world Shader defines these; the engine's defaults delegate to
// `project_vertex` and `shade_surface`. Both stages compile from this one
// variant, so both hooks are spliced here.
VertexOut transform(float4x4 model, float3 pos, float3 normal, float3 tangent,
                    float3 color, float2 uv);
float4 shade(VertexOut v, GpuObjectData od);

{SURFACE_VERTEX}

{SURFACE_FRAGMENT}

// ---- Vertex ----

[shader("vertex")]
VertexOut vertex_main_bindless(
    VertexIn v
#ifdef CN_BACKEND_DIRECTX
    // DirectX writes the object id into the b0 root constant ahead of each
    // indirect draw, so no instance-id builtin is read: SV_StartInstanceLocation
    // would raise the DXIL floor to shader model 6.8 for nothing.
    )
{
    uint oid = objid_cb.value;
#else
    ,
    uint instance_id : SV_InstanceID)
{
    uint oid = object_instance_index(instance_id);
#endif
    VertexOut o = transform(OBJECTS[oid].model, v.pos, v.normal, v.tangent, v.color, v.uv);
    o.object_id = oid;
    return o;
}

// ---- Fragment ----

[shader("pixel")]
float4 fragment_main_bindless(VertexOut v) : SV_Target
{
    return shade(v, OBJECTS[v.object_id]);
}
