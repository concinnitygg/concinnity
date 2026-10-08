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
// Under SURFACE_PREPASS the same text compiles the G-buffer pre-pass instead:
// the world's vertex hook positions the surface, and the fragment writes the
// normal, roughness and cutout `shade_surface` would light (see
// `gbuffer_common.hlsl` for the targets). It adds the pre-pass's own view
// block, the previous frame's models and the draw args to whichever binding
// model is selected.
//
// The records this binds are `main_types.hlsl` and the shading model it drives
// is `main_shading.hlsl`.

// ---- Shared CPU-visible records ----

{MAIN_TYPES}

{PROBE_TYPES}

// Mirrors `GpuMaterialParams` in concinnity-core/src/gfx/render_types.rs (32 B):
// one material's eight Shader parameters. Row 0 is all zeros.
struct GpuMaterialParams
{
    float4 values[2];
};

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
// The material parameter table, indexed by GpuObjectData.params_index.
[[vk::binding(11, 0)]] StructuredBuffer<GpuMaterialParams> material_params_sb : register(t16);

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
// The material parameter table, indexed by GpuObjectData.params_index.
StructuredBuffer<GpuMaterialParams> material_params_sb : register(t20);
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
// The material parameter table, indexed by GpuObjectData.params_index.
[[vk::binding(2, 1)]] StructuredBuffer<GpuMaterialParams> material_params_sb : register(t2, space1);


#endif // binding model

#ifdef SURFACE_PREPASS

{GBUFFER_COMMON}

// The model-history ring slot the PREVIOUS frame's `model_history.hlsl`
// dispatch filled, and this frame's draw args, read only for
// `DRAW_NO_HISTORY`. Both indexed by object id, like the object records.
#ifdef CN_BACKEND_METAL
[[vk::binding(12, 0)]] ConstantBuffer<GbView> gb_view : register(b3);
[[vk::binding(13, 0)]] StructuredBuffer<float4x4> prev_models : register(t17);
[[vk::binding(14, 0)]] StructuredBuffer<GpuDrawArgs> draw_args : register(t18);
#elif defined(CN_BACKEND_DIRECTX)
ConstantBuffer<GbView> gb_view : register(b6);
StructuredBuffer<float4x4> prev_models : register(t21);
StructuredBuffer<GpuDrawArgs> draw_args : register(t22);
#else
[[vk::binding(0, 2)]] ConstantBuffer<GbView> gb_view : register(b0, space2);
[[vk::binding(1, 2)]] StructuredBuffer<float4x4> prev_models : register(t1, space2);
[[vk::binding(2, 2)]] StructuredBuffer<GpuDrawArgs> draw_args : register(t2, space2);
#endif

// The vertex hook reads the clock through VIEW, and the pre-pass runs it a
// second time at the previous frame's clock for motion, so VIEW is a copy the
// entry can rewind.
static ViewUniforms surface_view;
#define VIEW surface_view

#else

#define VIEW view_cb

#endif
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
#define MATERIAL_PARAMS material_params_sb
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

// The parameter table row of the draw being shaded. Each entry point sets it
// before calling a hook, so `material_param` reads the right material from
// either hook without a parameter of its own.
static uint surface_params_row;

// Parameter `index` (0-7) of the material the surface being drawn uses; 0
// for a surface drawn without a material.
float material_param(uint index)
{
    uint i = min(index, 7u);
    return MATERIAL_PARAMS[surface_params_row].values[i >> 2][i & 3u];
}

// ---- The world's hooks ----

// A world Shader defines these; the engine's defaults delegate to
// `project_vertex` and `shade_surface`. Both stages compile from this one
// variant, so both hooks are spliced here.
VertexOut transform(float4x4 model, float3 pos, float3 normal, float3 tangent,
                    float3 color, float2 uv);
float4 shade(VertexOut v, GpuObjectData od);

{SURFACE_VERTEX}

{SURFACE_FRAGMENT}

#ifndef SURFACE_PREPASS

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
    surface_params_row = OBJECTS[oid].params_index;
    VertexOut o = transform(OBJECTS[oid].model, v.pos, v.normal, v.tangent, v.color, v.uv);
    o.object_id = oid;
    return o;
}

// ---- Fragment ----

[shader("pixel")]
float4 fragment_main_bindless(VertexOut v) : SV_Target
{
    GpuObjectData od = OBJECTS[v.object_id];
    surface_params_row = od.params_index;
    return shade(v, od);
}

#else // SURFACE_PREPASS

// The main pass's vertex stream, plus the previous frame's position on a second
// stream: the static vertex buffer again (prev_pos == pos, so motion is the
// model delta plus camera), or the previous frame's deformed buffer for the
// skinned tail.
struct PrepassVertexIn
{
    [[vk::location(0)]] float3 pos      : POSITION;
    [[vk::location(1)]] float3 normal   : NORMAL;
    [[vk::location(2)]] float3 tangent  : TANGENT;
    [[vk::location(3)]] float3 color    : COLOR0;
    [[vk::location(4)]] float2 uv       : TEXCOORD0;
    [[vk::location(5)]] float3 prev_pos : PREVPOSITION;
};

struct PrepassVertexOut
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 normal     : TEXCOORD0;
    [[vk::location(1)]] float3 tangent    : TEXCOORD1;
    [[vk::location(2)]] float3 bitangent  : TEXCOORD2;
    [[vk::location(3)]] float2 uv         : TEXCOORD3;
    // Positive view-space depth (-z); the consumers rebuild view position from it.
    [[vk::location(4)]] float view_depth  : TEXCOORD4;
    [[vk::location(5)]] float4 cur_clip   : TEXCOORD5;
    [[vk::location(6)]] float4 prev_clip  : TEXCOORD6;
    [[vk::location(7)]] nointerpolation uint object_id : TEXCOORD7;
};

// The transform to reproject last frame's position through: the history entry,
// or this frame's own model where no history exists, which collapses the motion
// vector to the camera's own (and to exactly zero when `prev_vp == cur_vp`).
float4x4 prepass_prev_model(uint oid, float4x4 cur_model)
{
    if ((draw_args[oid].flags & DRAW_NO_HISTORY) != 0u)
    {
        return cur_model;
    }
    return prev_models[oid];
}

// The hook places the surface exactly as the main pass draws it, rasterized
// through the jittered VIEW.vp. When a consumer reads motion it runs again with
// the previous model, position, clock and camera position, and both world
// positions reproject through the unjittered matrices so jitter never leaks
// into the motion vector.
[shader("vertex")]
PrepassVertexOut vertex_prepass_bindless(
    PrepassVertexIn v
#ifdef CN_BACKEND_DIRECTX
    )
{
    uint oid = objid_cb.value;
#else
    ,
    uint instance_id : SV_InstanceID)
{
    uint oid = object_instance_index(instance_id);
#endif
    float4x4 model = OBJECTS[oid].model;
    surface_params_row = OBJECTS[oid].params_index;
    surface_view = view_cb;
    VertexOut cur = transform(model, v.pos, v.normal, v.tangent, v.color, v.uv);
    float3 prev_world = cur.world_pos;
    if (gb_view.motion != 0u)
    {
        surface_view.elapsed = gb_view.prev_elapsed;
        surface_view.cam_x = gb_view.prev_cam_x;
        surface_view.cam_y = gb_view.prev_cam_y;
        surface_view.cam_z = gb_view.prev_cam_z;
        prev_world = transform(prepass_prev_model(oid, model), v.prev_pos, v.normal,
                               v.tangent, v.color, v.uv).world_pos;
    }

    PrepassVertexOut o;
    o.position   = cur.position;
    o.normal     = cur.normal;
    o.tangent    = cur.tangent;
    o.bitangent  = cur.bitangent;
    o.uv         = cur.uv;
    o.view_depth = cur.view_depth;
    o.cur_clip   = mul(gb_view.cur_vp, float4(cur.world_pos, 1.0));
    o.prev_clip  = mul(gb_view.prev_vp, float4(prev_world, 1.0));
    o.object_id  = oid;
    return o;
}

// Only the inputs these targets need: the cutout, the shading normal, and the
// roughness the forward pass lights with, specular antialiasing included.
[shader("pixel")]
GbFragmentOut fragment_prepass_bindless(PrepassVertexOut p)
{
    GpuObjectData od = OBJECTS[p.object_id];
    if (od.bb_max_alpha_cutoff.w > 0.0)
    {
        surface_cutout(od, pool_sample(od.albedo_index, p.uv).a);
    }
    float3 N = surface_normal(od, p.uv, p.normal, p.tangent, p.bitangent);
    float roughness = specular_aa_roughness(N, surface_roughness_metallic(od, p.uv).x);

    GbFragmentOut o;
    o.nd    = p.view_depth > 0.0
            ? float4(normalize(mul((float3x3)gb_view.view_mat, N)), p.view_depth)
            : (float4)(0.0);
    o.rough = roughness;
    o.vel   = gb_motion(p.cur_clip, p.prev_clip);
    return o;
}

#endif // SURFACE_PREPASS
