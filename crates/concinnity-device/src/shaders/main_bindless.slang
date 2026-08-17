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
//   METAL_ABI  - the engine's Metal binding layout, frozen because
//                world-authored Shader assets hand-write MSL against it:
//                discrete buffers pinned by register() (b/t numbers ARE the
//                Metal buffer indices), the texture-only argument buffer at
//                buffer(7), and the engine sampler block at buffer(10).
//   DXIL_ABI   - the engine's DirectX bindless root signature, frozen for the
//                same reason (world Shader assets build a PSO against it, see
//                directx/world_shaders.rs): every register pinned to the layout
//                in directx/init/pipelines.rs, and the object id taken from the
//                b0 root constant the indirect command writes rather than from
//                an instance-id builtin.
//
// Layouts must match render_types.rs: GpuObjectData (176 B), GpuLight (64 B),
// ClusterParams (128 B), SpotShadowData (80 B), AreaLightData (32 B).

#ifndef POOL_SIZE
#define POOL_SIZE 1024
#endif
#ifndef MAX_PROBES
#define MAX_PROBES 8
#endif

// ---- Shared CPU-visible records ----

struct ViewUniforms
{
    float4x4 vp;
    float4x4 view_mat;
    float elapsed;
    // 1.0 when an SSR / RT reflection composite owns the sharp specular this
    // frame (fade the glossy-dielectric forward probe specular); 0.0 keeps it.
    float reflections_enabled;
    float cam_x; float cam_y; float cam_z;
    float prefilter_mip_count;
    // 1.0 while the unlit view mode is active: the surface returns its base
    // color before lighting.
    float shade_mode; float _ep1;
};

struct GpuObjectData
{
    float4x4 model;
    float3 tint;     float roughness;
    float3 emissive; float metallic;
    uint  albedo_index;
    uint  normal_index;
    float macro_variation;
    float terrain_blend;
    float3 bb_min;   float cull_distance;
    float3 bb_max;   float secondary_blend_sharpness;
    uint  albedo_secondary_index;
    uint  normal_secondary_index;
    uint  emissive_map_index;
    uint  orm_map_index;
    float alpha_cutoff;
    float _pad0;
    float _pad1;
    float _pad2;
};

struct DirLight   { float4 dir_i; float4 col; };
struct PointLight { float4 pos_r; float4 col_i; };

struct LightUniforms
{
    DirLight   dir[4];
    PointLight pt[8];
    int num_dir;
    int num_pt;
    // Indirect-ambient multiplier (PostProcessConfig.ambient_intensity); 1.0
    // is a no-op.
    float ambient_intensity;
    // Valid entry count in the local-light buffer.
    int num_local_lights;
};

struct ShadowUniforms
{
    float4x4 light_vps[4];
    float4 cascade_splits;
    // Live cascade count (1..4); slots at or beyond it are unrendered.
    uint active_cascades;
};

// GpuLight.kind discriminants (LIGHT_KIND_* in render_types.rs).
static const uint LIGHT_KIND_SPOT = 1u;
static const uint LIGHT_KIND_AREA = 2u;

struct GpuLight
{
    float3 position;
    float  range;
    float3 color;
    float  intensity;
    float3 direction;
    uint   kind;
    float  cos_inner;
    float  cos_outer;
    int    shadow_index;
    // Index into the AreaLightData table for an area light, else -1.
    int    data_index;
};

// One shadowed spot's slice projection; indexed by GpuLight.shadow_index,
// which doubles as the array layer.
struct SpotShadowData
{
    float4x4 light_vp;
    float depth_bias;
    float normal_bias;
    float2 _pad;
};

// One rectangular area light's extent, indexed by GpuLight.data_index. The
// edges are pre-scaled by the half-extents, so the corners are
// centre +/- right +/- up.
struct AreaLightData
{
    float3 right;
    uint two_sided;
    float3 up;
    float _pad;
};

struct ClusterParams
{
    float4x4 inv_view_proj;
    // xyz = camera position, w = z_near. float4 pairs rather than float3 +
    // scalar: MSL sizes a constant-buffer float3 at 16 bytes, so the packed
    // 128-byte CPU layout only survives on every target without vec3 fields.
    float4   cam_pos_znear;
    // xyz = view forward, w = z_far.
    float4   view_forward_zfar;
    uint     grid_x;
    uint     grid_y;
    uint     grid_z;
    uint     num_lights;
    float    screen_w;
    float    screen_h;
    uint     use_clusters;
    uint     _pad;
};

struct ProbeUniforms
{
    float4 box_min;
    float4 box_max;
    float4 probe_pos;
};

struct ProbeSet
{
    uint count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    ProbeUniforms probes[MAX_PROBES];
};

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
    return tex.irradiance_cube.Sample(samps.cube_sampler, n).rgb;
}
float3 prefilter_sample_level0(float3 dir)
{
    return tex.prefilter_cube.SampleLevel(samps.cube_sampler, dir, 0.0).rgb;
}
float3 prefilter_sample_bias(float3 dir, float lod)
{
    return tex.prefilter_cube.SampleBias(samps.cube_sampler, dir, lod).rgb;
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
    return irradiance_cube.Sample(cube_sampler, n).rgb;
}
float3 prefilter_sample_level0(float3 dir)
{
    return prefilter_cube.SampleLevel(cube_sampler, dir, 0.0).rgb;
}
float3 prefilter_sample_bias(float3 dir, float lod)
{
    return prefilter_cube.SampleBias(cube_sampler, dir, lod).rgb;
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
    return scene.irradiance_cube.Sample(n).rgb;
}
float3 prefilter_sample_level0(float3 dir)
{
    return scene.prefilter_cube.SampleLevel(dir, 0.0).rgb;
}
float3 prefilter_sample_bias(float3 dir, float lod)
{
    return scene.prefilter_cube.SampleBias(dir, lod).rgb;
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

// ---- Constants ----

static const float PI = 3.14159265359;

// Per-cluster light-list stride: MAX_LIGHTS_PER_CLUSTER + 1 (slot 0 is the
// count). Matches CLUSTER_LIGHT_LIST_STRIDE in render_types.rs.
static const uint CLUSTER_LIGHT_LIST_STRIDE = 64u;

// Surfaces rougher than this get no SSR / RT reflection; the forward fade
// ramps in below it. Matches the resolve gloss gate (SSR_ROUGH_CUT /
// RT_ROUGH_CUT).
static const float REFLECTION_ROUGHNESS_CUT = 0.6;

// Edge of the LTC lookup tables, and the scale / bias that map [0, 1] onto
// texel centres. Must match LTC_LUT_SIZE in concinnity-render's ltc module.
static const float LTC_LUT_SIZE  = 64.0;
static const float LTC_LUT_SCALE = (LTC_LUT_SIZE - 1.0) / LTC_LUT_SIZE;
static const float LTC_LUT_BIAS  = 0.5 / LTC_LUT_SIZE;

static const float PROBE_BLEND_MARGIN = 0.2;

static const float3 SKY_ZENITH  = float3(0.110, 0.322, 0.726);
static const float3 SKY_HORIZON = float3(0.765, 0.863, 0.941);

// ---- Stage interface ----

struct VertexIn
{
    [[vk::location(0)]] float3 pos     : POSITION;
    [[vk::location(1)]] float3 normal  : NORMAL;
    [[vk::location(2)]] float3 tangent : TANGENT;
    [[vk::location(3)]] float3 color   : COLOR0;
    [[vk::location(4)]] float2 uv      : TEXCOORD0;
};

struct VertexOut
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_pos  : TEXCOORD1;
    [[vk::location(1)]] float3 normal     : TEXCOORD2;
    [[vk::location(2)]] float3 tangent    : TEXCOORD3;
    [[vk::location(3)]] float3 bitangent  : TEXCOORD4;
    [[vk::location(4)]] float2 uv         : TEXCOORD5;
    [[vk::location(5)]] float view_depth  : TEXCOORD6;
    [[vk::location(6)]] float3 color      : TEXCOORD7;
    // The object id is needed in the fragment stage too (the instance index is
    // a vertex-only built-in), so it is forwarded as a flat varying.
    [[vk::location(7)]] nointerpolation uint object_id : TEXCOORD8;
};

// ---- Vertex ----

// Inverse-transpose of the model's upper 3x3 via the adjugate, so normals stay
// perpendicular under non-uniform scale.
float3x3 normal_matrix(float4x4 model)
{
    // Slang subscripts a matrix by row, so these are the upper 3x3's rows. The
    // cofactor matrix built from them is the inverse-transpose scaled by the
    // determinant; the normalize() at every use site absorbs that scale.
    float3 a0 = model[0].xyz;
    float3 a1 = model[1].xyz;
    float3 a2 = model[2].xyz;
    float3 r0 = cross(a1, a2);
    float3 r1 = cross(a2, a0);
    float3 r2 = cross(a0, a1);
    // mul(M, v) with these as rows applies the inverse-transpose to v.
    return float3x3(r0, r1, r2);
}

#ifndef DXIL_ABI

// The engine encodes the object id as the draw's first-instance value. The
// targets disagree on what an instance-id builtin contains: SPIR-V's
// InstanceIndex and Metal's instance_id include the base instance, while
// SV_InstanceID is defined zero-based (Slang subtracts the base to honor
// that). Reconstruct the object id explicitly from both builtins.
uint object_instance_index(uint zero_based_iid, uint first_instance)
{
    __target_switch
    {
    case metal:
        // Slang's Metal lowering passes instance_id through unsubtracted, and
        // instance_id already includes base_instance.
        return zero_based_iid;
    default:
        return zero_based_iid + first_instance;
    }
}

#endif

[shader("vertex")]
VertexOut vertex_main_bindless(
    VertexIn v
#ifdef DXIL_ABI
    // DirectX writes the object id into the b0 root constant ahead of each
    // indirect draw, so no instance-id builtin is read: SV_StartInstanceLocation
    // would raise the DXIL floor to shader model 6.8 for nothing.
    )
{
    VertexOut o;
    uint oid = objid_cb.value;
#else
    ,
    uint instance_id : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    VertexOut o;
    uint oid = object_instance_index(instance_id, first_instance);
#endif
    o.object_id = oid;
    float4x4 model = OBJECTS[oid].model;

    float4 world = mul(model, float4(v.pos, 1.0));
    o.world_pos = world.xyz;

    float3x3 nm = normal_matrix(model);
    o.normal    = normalize(mul(nm, v.normal));
    o.tangent   = normalize(mul(nm, v.tangent));
    o.bitangent = cross(o.normal, o.tangent);

    o.uv    = v.uv;
    o.color = v.color;

    o.view_depth = -mul(VIEW.view_mat, world).z;

    o.position = mul(VIEW.vp, world);

    // Skybox sentinel (blue channel 2.0): pin to the far plane.
    if (v.color.b > 1.5)
    {
        o.position.z = o.position.w * (1.0 - 1e-6);
    }
    return o;
}

// ---- Fragment helpers ----

float distribution_ggx(float3 N, float3 H, float roughness)
{
    float a  = roughness * roughness;
    float a2 = a * a;
    float NdH  = max(dot(N, H), 0.0);
    float NdH2 = NdH * NdH;
    float denom = NdH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 0.0001);
}

float geometry_schlick_ggx(float NdV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdV / (NdV * (1.0 - k) + k);
}

float geometry_smith(float3 N, float3 V, float3 L, float roughness)
{
    float NdV = max(dot(N, V), 0.0);
    float NdL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdV, roughness) * geometry_schlick_ggx(NdL, roughness);
}

float3 fresnel_schlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float2 env_brdf_approx(float NdV, float rough)
{
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4( 1.0,  0.0425,  1.040, -0.040);
    float4 r = rough * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdV)) * r.x + r.y;
    return float2(-1.04, 1.04) * a004 + r.zw;
}

// Decode a tangent-space normal map texel. Only X and Y are read; Z is
// reconstructed from them, so a two-channel source (BC5) decodes the same as
// an RGBA8 one and normal maps can ship as BC5 blocks.
float3 decode_normal_map(float2 encoded)
{
    float2 nxy = encoded * 2.0 - 1.0;
    return float3(nxy, sqrt(clamp(1.0 - dot(nxy, nxy), 0.0, 1.0)));
}

// Geometric specular antialiasing (Kaplanyan et al. 2016, as in Filament):
// widen the NDF by the screen-space variance of the shading normal so an
// undersampled high-frequency normal map at a distance does not alias into
// specular fireflies. A no-op where the normal is smooth (close up), so the
// surface detail is preserved.
float specular_aa_roughness(float3 N, float perceptual_roughness)
{
    const float VARIANCE  = 0.25;
    const float THRESHOLD = 0.18;
    float3 dndx = ddx(N);
    float3 dndy = ddy(N);
    float variance = VARIANCE * (dot(dndx, dndx) + dot(dndy, dndy));
    float alpha = perceptual_roughness * perceptual_roughness;
    float kernel = min(2.0 * variance, THRESHOLD);
    float filtered_alpha2 = clamp(alpha * alpha + kernel, 0.0, 1.0);
    return sqrt(sqrt(filtered_alpha2));
}

float hash_rotation(float2 p)
{
    float h = frac(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
    return h * 6.2831853;
}

// Per-cascade shadow-bias growth. Metal bakes this factor into the shadow
// pass's rasterizer depth bias (metal/draw/shadow.rs); every other backend
// renders the cascades unbiased and applies it at the sample instead.
float cascade_bias_scale(int cascade)
{
    __target_switch
    {
    case metal:
        return 1.0;
    default:
        return 1.0 + float(cascade) * 2.0;
    }
}

// 3x3 hash-rotated PCF of one spot shadow slice. Returns [0, 1] (1.0 fully
// lit), and 1.0 outside the cone's light frustum so an unshadowed region is
// never darkened. A smaller kernel than the cascade PCF: a spot slice covers
// far less world area per texel.
float sample_spot_shadow(int shadow_index, float3 world_pos, float3 normal, float2 screen_xy)
{
    SpotShadowData sd = SPOT_SHADOWS[shadow_index];
    // Offsetting along the normal before projecting pushes the sample off
    // surfaces near-parallel to the light, where depth slope causes acne.
    float3 biased = world_pos + normal * sd.normal_bias;
    float4 light_clip = mul(sd.light_vp, float4(biased, 1.0));
    if (light_clip.w <= 0.0)
    {
        return 1.0;
    }
    float3 ndc = light_clip.xyz / light_clip.w;
    // Flip Y to match the negative-height viewport the spot pass renders with,
    // exactly as the cascade PCF does.
    float2 uv = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0)
    {
        return 1.0;
    }

    float ref = ndc.z - sd.depth_bias;
    float angle = hash_rotation(screen_xy);
    float ca = cos(angle);
    float sa = sin(angle);
    float2 tex_size = 1.0 / spot_shadow_map_size();

    float sum = 0.0;
    const int RADIUS = 1; // 3x3
    const float SAMPLES = float((2 * RADIUS + 1) * (2 * RADIUS + 1));
    for (int dy = -RADIUS; dy <= RADIUS; dy++)
    {
        for (int dx = -RADIUS; dx <= RADIUS; dx++)
        {
            float2 off = float2(float(dx), float(dy));
            float2 rot = float2(off.x * ca - off.y * sa, off.x * sa + off.y * ca);
            sum += spot_shadow_cmp(
                float3(uv + rot * tex_size, float(shadow_index)), ref);
        }
    }
    return sum / SAMPLES;
}

// Clip a quad against the horizon plane z = 0, keeping the part above it.
// Sutherland-Hodgman rather than the usual hardcoded 16-case table: a quad cut
// by one plane yields at most 5 vertices, and the loop form cannot be got
// wrong case by case. Mirrors clip_quad_to_horizon in concinnity-render's
// ltc::polygon, which is unit-tested against brute-force integration.
int clip_quad_to_horizon(float3 quad[4], out float3 clipped[5])
{
    clipped = { float3(0.0), float3(0.0), float3(0.0), float3(0.0), float3(0.0) };
    int n = 0;
    for (int i = 0; i < 4; i++)
    {
        float3 current  = quad[i];
        float3 previous = quad[(i + 3) % 4];
        bool current_in  = current.z > 0.0;
        bool previous_in = previous.z > 0.0;
        if (current_in != previous_in)
        {
            float t = previous.z / (previous.z - current.z);
            clipped[n++] = float3(previous.xy + t * (current.xy - previous.xy), 0.0);
        }
        if (current_in)
        {
            clipped[n++] = current;
        }
    }
    return n;
}

// Twice the contribution of one edge of the spherical polygon. The cross
// product's z carries the sign, so a reversed winding flips the whole sum,
// which is what tells a front-facing polygon from a back-facing one.
float integrate_edge(float3 v1, float3 v2)
{
    float cos_theta = clamp(dot(v1, v2), -1.0, 1.0);
    float theta     = acos(cos_theta);
    float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    float ratio     = (sin_theta > 1e-4) ? (theta / sin_theta) : 1.0;
    return cross(v1, v2).z * ratio;
}

// Fraction of the clamped-cosine distribution the quad covers, in [0, 1].
// `m_inv` is the LTC inverse transform (rows follow mul(v, M) convention), or
// the identity for the diffuse term.
float ltc_evaluate(float3 N, float3 V, float3 P, float3x3 m_inv, float3 corners[4], bool two_sided)
{
    // Shading frame with the normal on +z and the first tangent in the view
    // plane, matching how the table was fitted.
    float3 t1 = normalize(V - N * dot(V, N));
    float3 t2 = cross(N, t1);

    float3 quad[4];
    for (int i = 0; i < 4; i++)
    {
        float3 d = corners[i] - P;
        float3 local = float3(dot(t1, d), dot(t2, d), dot(N, d));
        quad[i] = mul(local, m_inv);
    }

    float3 clipped[5];
    int n = clip_quad_to_horizon(quad, clipped);
    if (n < 3)
    {
        return 0.0;
    }
    for (int k = 0; k < n; k++)
    {
        clipped[k] = normalize(clipped[k]);
    }

    float sum = 0.0;
    for (int e = 0; e < n; e++)
    {
        sum += integrate_edge(clipped[e], clipped[(e + 1) % n]);
    }

    // The edge sum is twice the irradiance; dividing by pi normalises the
    // clamped cosine, so the covered fraction is sum / (2 * pi).
    float form_factor = sum / (2.0 * PI);
    return two_sided ? abs(form_factor) : max(-form_factor, 0.0);
}

// 5x5 hash-rotated PCF of a single cascade. Returns the shadow factor in
// [0, 1] (1.0 fully lit), or 1.0 when the fragment lies outside this
// cascade's light frustum.
float sample_cascade_pcf(int cascade, float3 world_pos, float2 screen_xy)
{
    float4 lc = mul(SHADOW_UNI.light_vps[cascade], float4(world_pos, 1.0));
    float3 ndc = lc.xyz / lc.w;
    float2 uv = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0)
    {
        return 1.0;
    }

    // Depth bias as a world-space offset along the light: in NDC that is the
    // world offset over the cascade depth range, i.e. world_bias * length(VP
    // row2 xyz) (the ortho z scale). The (1 + cascade * 2) factor grows that
    // offset with cascade index: a distant cascade covers more world per
    // shadow texel, so a flat bias under-biases the far cascades and leaves
    // self-shadow acne that steps at each cascade boundary. On Metal that
    // per-cascade growth is already applied by the shadow-pass rasterizer's
    // depth bias (metal/draw/shadow.rs), so the sample-side bias stays flat
    // there; applying both would double-bias the far cascades.
    // Slang indexes HLSL-style (`m[i][j]` is row i, column j), unlike the GLSL
    // and MSL counterparts where `m[i]` is column i: row 2 is spelled with the
    // indices in the opposite order there.
    float3 vp_row2 = float3(SHADOW_UNI.light_vps[cascade][2][0],
                            SHADOW_UNI.light_vps[cascade][2][1],
                            SHADOW_UNI.light_vps[cascade][2][2]);
    float bias = 0.03 * cascade_bias_scale(cascade) * length(vp_row2);
    float ref = ndc.z - bias;

    float angle = hash_rotation(screen_xy);
    float ca = cos(angle);
    float sa = sin(angle);

    float2 tex_size = 1.0 / shadow_map_size();

    float sum = 0.0;
    const int RADIUS = 2;
    const float SAMPLES = float((2 * RADIUS + 1) * (2 * RADIUS + 1));
    for (int dy = -RADIUS; dy <= RADIUS; dy++)
    {
        for (int dx = -RADIUS; dx <= RADIUS; dx++)
        {
            float2 off = float2(float(dx), float(dy));
            float2 rot = float2(off.x * ca - off.y * sa, off.x * sa + off.y * ca);
            float2 sample_uv = uv + rot * tex_size;
            sum += shadow_map_cmp(float3(sample_uv, float(cascade)), ref);
        }
    }
    return sum / SAMPLES;
}

// Cascade-aware PCF with cross-cascade blending. Selects the cascade whose far
// split exceeds the fragment's view-space depth, then blends into the next
// cascade across a band at the far edge of that cascade's depth range: each
// cascade places the shadow edge slightly differently, and a hard switch would
// sweep across the world as the camera moves.
float shadow_factor_cascaded(float3 world_pos, float view_depth, float2 screen_xy)
{
    int cascade = 4;
    if      (view_depth < SHADOW_UNI.cascade_splits[0]) cascade = 0;
    else if (view_depth < SHADOW_UNI.cascade_splits[1]) cascade = 1;
    else if (view_depth < SHADOW_UNI.cascade_splits[2]) cascade = 2;
    else if (view_depth < SHADOW_UNI.cascade_splits[3]) cascade = 3;
    if (cascade >= int(SHADOW_UNI.active_cascades))
    {
        return 1.0;
    }

    float shade = sample_cascade_pcf(cascade, world_pos, screen_xy);

    if (cascade + 1 < int(SHADOW_UNI.active_cascades))
    {
        float split_far  = SHADOW_UNI.cascade_splits[cascade];
        float split_near = (cascade == 0) ? 0.0 : SHADOW_UNI.cascade_splits[cascade - 1];
        float band = (split_far - split_near) * 0.15;
        float t = (view_depth - (split_far - band)) / max(band, 1e-4);
        if (t > 0.0)
        {
            float next = sample_cascade_pcf(cascade + 1, world_pos, screen_xy);
            shade = lerp(shade, next, clamp(t, 0.0, 1.0));
        }
    }
    return shade;
}

// Box-parallax sample of probe cube `i`: intersect the reflection ray `R` with
// the probe's influence box and re-anchor the sample direction at that hit
// relative to the capture point, so a static captured cube tracks a moving
// camera. Falls back to the raw ray when the probe has no baked box
// (box_min.w <= 0.5) or the box does not lie ahead of the ray. `lod` is passed
// as the texture bias (the reflection vector's screen-space footprint widens
// the mip at grazing / distant angles), the same SampleBias semantics the
// prefilter-cube tap uses.
float3 sample_probe_radiance(uint i, ProbeUniforms probe, float3 world_pos, float3 R, float lod)
{
    float3 sample_dir = R;
    if (probe.box_min.w > 0.5)
    {
        float3 inv_r = 1.0 / R;
        float3 t_max = (probe.box_max.xyz - world_pos) * inv_r;
        float3 t_min = (probe.box_min.xyz - world_pos) * inv_r;
        float3 t_far = max(t_max, t_min);
        float dist = min(min(t_far.x, t_far.y), t_far.z);
        if (dist > 0.0)
        {
            float3 hit = world_pos + R * dist;
            sample_dir = hit - probe.probe_pos.xyz;
        }
    }
    return probe_cube_sample_bias(i, sample_dir, lod);
}

// Probe radiance for `world_pos` along world-space ray `R`, blended across
// every probe covering the point (partition of unity). Each probe gets a
// smoothstep weight from its signed box distance (1 deep inside, 0.5 on the
// surface, 0 a margin outside); the result is the weight-normalised sum of
// each probe's box-projected sample. Where no box covers, falls back to the
// nearest probe by capture distance.
float3 probe_set_specular(float3 world_pos, float3 R, float lod)
{
    float3 acc = float3(0.0);
    float wsum = 0.0;
    float near_d = 1e30;
    uint near_i = 0u;
    for (uint i = 0u; i < PROBE_SET.count; i++)
    {
        float3 c = 0.5 * (PROBE_SET.probes[i].box_min.xyz + PROBE_SET.probes[i].box_max.xyz);
        float3 he = 0.5 * (PROBE_SET.probes[i].box_max.xyz - PROBE_SET.probes[i].box_min.xyz);
        // Signed distance to the box surface: positive inside, negative out.
        float3 q = abs(world_pos - c) - he;
        float sd = -(length(max(q, float3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0));
        float margin = max(PROBE_BLEND_MARGIN * min(he.x, min(he.y, he.z)), 1e-4);
        float w = smoothstep(-margin, margin, sd);
        if (w > 0.0)
        {
            acc += w * sample_probe_radiance(i, PROBE_SET.probes[i], world_pos, R, lod);
            wsum += w;
        }
        float d = distance(world_pos, PROBE_SET.probes[i].probe_pos.xyz);
        if (d < near_d)
        {
            near_d = d;
            near_i = i;
        }
    }
    if (wsum > 0.0)
    {
        return acc / wsum;
    }
    return sample_probe_radiance(near_i, PROBE_SET.probes[near_i], world_pos, R, lod);
}

// ---- Fragment ----

[shader("fragment")]
float4 fragment_main_bindless(VertexOut in) : SV_Target
{
    GpuObjectData od = OBJECTS[in.object_id];
    float roughness = od.roughness;
    float metallic  = od.metallic;
    float3 tint     = od.tint;
    float3 emissive = od.emissive;

    float3 cam_pos = float3(VIEW.cam_x, VIEW.cam_y, VIEW.cam_z);
    bool ibl_enabled = VIEW.prefilter_mip_count > 0.5;

    // Skybox sentinel (blue channel 2.0): sky colour from the view direction.
    if (in.color.b > 1.5)
    {
        float3 view_dir = normalize(in.world_pos - cam_pos);
        float3 sky;
        if (ibl_enabled)
        {
            sky = prefilter_sample_level0(view_dir);
        }
        else
        {
            float t = max(0.0, view_dir.y);
            sky = lerp(SKY_HORIZON, SKY_ZENITH, t);
        }
        return float4(sky, 1.0);
    }

    // `object_id` is flat from the instance index, so a fragment wave that
    // straddles two objects of one indirect draw carries two pool indices.
    // That makes every pool index non-uniform on the descriptor-indexing
    // targets (pool_sample annotates it there).
    float4 albedo_samp = pool_sample(od.albedo_index, in.uv);
    // Alpha cutout: punch the texel out entirely so foliage and decal cards
    // stay in the opaque pass. Disabled at cutoff 0.
    if (od.alpha_cutoff > 0.0 && albedo_samp.a < od.alpha_cutoff)
    {
        discard;
    }
    float3 albedo = albedo_samp.rgb * in.color * tint;

    // Unlit view mode: the surface's base color, no lighting.
    if (VIEW.shade_mode > 0.5)
    {
        return float4(albedo, 1.0);
    }

    // Per-material emissive texture carries the colour (the scalar factor is a
    // uniform strength when a map is bound). Slot 0 is the "no map" sentinel.
    if (od.emissive_map_index != 0u)
    {
        emissive *= pool_sample(od.emissive_map_index, in.uv).rgb;
    }

    // Occlusion-roughness-metallic map: green carries roughness, blue carries
    // metallic (glTF convention). Slot 0 is the "no map" sentinel.
    if (od.orm_map_index != 0u)
    {
        float3 orm = pool_sample(od.orm_map_index, in.uv).rgb;
        roughness = orm.g;
        metallic  = orm.b;
    }

    float3 norm_samp = decode_normal_map(pool_sample(od.normal_index, in.uv).rg);
    // Tangent frame as rows so mul(v, M) applies the column-basis transform.
    float3x3 TBN = float3x3(
        normalize(in.tangent),
        normalize(in.bitangent),
        normalize(in.normal));
    float3 N = normalize(mul(norm_samp, TBN));

    // Geometric specular antialiasing on the normal map. Minification aliasing
    // is handled by the texture's mip chain (trilinear + anisotropic
    // sampling); this widens the specular NDF for residual sub-pixel normal
    // variance.
    roughness = specular_aa_roughness(N, roughness);

    float3 V   = normalize(cam_pos - in.world_pos);
    float NdV  = max(dot(N, V), 0.0);

    float3 F0 = lerp(float3(0.04), albedo, metallic);

    float2 screen_xy = in.position.xy;
    float shadow = shadow_factor_cascaded(in.world_pos, in.view_depth, screen_xy);

    float2 ab        = env_brdf_approx(NdV, roughness);
    float ess        = ab.x + ab.y;
    float3 energy_ms = 1.0 + F0 * (1.0 / max(ess, 0.001) - 1.0);

    float3 Lo = float3(0.0);

    for (int i = 0; i < LIGHTS.num_dir; i++)
    {
        float3 L = normalize(LIGHTS.dir[i].dir_i.xyz);
        float intensity = LIGHTS.dir[i].dir_i.w;
        float3 radiance = LIGHTS.dir[i].col.xyz * intensity;

        float3 H = normalize(V + L);
        float NdL = max(dot(N, L), 0.0);

        float D = distribution_ggx(N, H, roughness);
        float G = geometry_smith(N, V, L, roughness);
        float3 F = fresnel_schlick(max(dot(H, V), 0.0), F0);

        float3 kd = (1.0 - F) * (1.0 - metallic);
        float3 spec = (D * G * F) / max(4.0 * NdV * NdL, 0.001) * energy_ms;
        float3 diff = kd * albedo / PI;

        float s = (i == 0) ? shadow : 1.0;
        Lo += (diff + spec) * radiance * NdL * s;
    }

    // Clustered light iteration: when clustering is active (the main camera),
    // map this fragment to its froxel cluster and shade only that cluster's
    // binned lights. Planar / probe re-renders bind use_clusters = 0 (their
    // viewpoint differs from the grid the main camera binned) and fall back
    // to iterating every local light.
    uint cluster_base = 0u;
    int  local_count;
    if (CLUSTER.use_clusters != 0u)
    {
        uint cx = min(uint(screen_xy.x / CLUSTER.screen_w * float(CLUSTER.grid_x)),
                      CLUSTER.grid_x - 1u);
        uint cy = min(uint(screen_xy.y / CLUSTER.screen_h * float(CLUSTER.grid_y)),
                      CLUSTER.grid_y - 1u);
        float zd = max(in.view_depth, CLUSTER.cam_pos_znear.w);
        uint cz = min(uint(log(zd / CLUSTER.cam_pos_znear.w) / log(CLUSTER.view_forward_zfar.w / CLUSTER.cam_pos_znear.w)
                           * float(CLUSTER.grid_z)),
                      CLUSTER.grid_z - 1u);
        uint cid = cx + cy * CLUSTER.grid_x + cz * CLUSTER.grid_x * CLUSTER.grid_y;
        cluster_base = cid * CLUSTER_LIGHT_LIST_STRIDE;
        local_count = int(CLUSTER_LIST[cluster_base]);
    }
    else
    {
        local_count = LIGHTS.num_local_lights;
    }

    for (int jj = 0; jj < local_count; jj++)
    {
        int i = (CLUSTER.use_clusters != 0u)
              ? int(CLUSTER_LIST[cluster_base + 1u + uint(jj)])
              : jj;
        float3 pos_w  = LOCAL_LIGHTS[i].position;
        float  range  = LOCAL_LIGHTS[i].range;
        float3 col    = LOCAL_LIGHTS[i].color;
        float  intens = LOCAL_LIGHTS[i].intensity;

        // Area lights integrate the whole panel rather than a single
        // direction, so they replace the point / spot BRDF evaluation.
        if (LOCAL_LIGHTS[i].kind == LIGHT_KIND_AREA)
        {
            int ai = LOCAL_LIGHTS[i].data_index;
            if (ai < 0)
            {
                continue;
            }
            float3 centre = pos_w;
            float3 right  = AREA_LIGHTS[ai].right;
            float3 up     = AREA_LIGHTS[ai].up;
            bool two_sided = AREA_LIGHTS[ai].two_sided != 0u;

            // Range is a cutoff measured from the panel centre, matching the
            // sphere the clustered cull bins this light with. The physical
            // falloff is already in the form factor: the panel subtends a
            // smaller solid angle further away.
            float centre_dist = length(centre - in.world_pos);
            float window = clamp(1.0 - centre_dist / range, 0.0, 1.0);
            window = window * window;
            if (window <= 0.0)
            {
                continue;
            }

            float3 corners[4];
            corners[0] = centre - right - up;
            corners[1] = centre + right - up;
            corners[2] = centre + right + up;
            corners[3] = centre - right + up;

            // Diffuse needs no lookup: it is the polygon integral under the
            // plain clamped cosine, i.e. an identity transform.
            float3x3 identity = float3x3(
                float3(1.0, 0.0, 0.0),
                float3(0.0, 1.0, 0.0),
                float3(0.0, 0.0, 1.0));
            float diffuse_ff = ltc_evaluate(N, V, in.world_pos, identity, corners, two_sided);

            // Specular applies the fitted transform before the same integral.
            float2 lut_uv = float2(roughness, sqrt(clamp(1.0 - NdV, 0.0, 1.0)));
            lut_uv = lut_uv * LTC_LUT_SCALE + LTC_LUT_BIAS;
            float4 t1 = ltc_matrix_sample(lut_uv);
            float2 t2 = ltc_magnitude_sample(lut_uv);
            // The table stores the inverse normalised so its middle entry is
            // 1, packed as (m00, m20, m02, m22). Rows here follow the
            // mul(v, M) convention, matching the GLSL / MSL column form.
            float3x3 m_inv = float3x3(
                float3(t1.x, 0.0, t1.y),
                float3(0.0,  1.0, 0.0),
                float3(t1.z, 0.0, t1.w));
            float specular_ff = ltc_evaluate(N, V, in.world_pos, m_inv, corners, two_sided);
            // Schlick split baked into the table: t2.x weights the base
            // reflectance, t2.y the grazing response.
            float3 area_spec = F0 * t2.x + (1.0 - F0) * t2.y;

            float3 area_radiance = col * intens * window;
            float3 area_kd = (1.0 - F0) * (1.0 - metallic);
            Lo += area_radiance * (area_kd * albedo * diffuse_ff
                                   + area_spec * specular_ff);
            continue;
        }

        float3 L    = normalize(pos_w - in.world_pos);
        float  dist = length(pos_w - in.world_pos);
        float atten = clamp(1.0 - (dist / range), 0.0, 1.0);
        atten *= atten;
        // Spot cone: full brightness inside cos_inner, squared fade to black
        // at cos_outer. Point lights leave both at zero and skip this.
        if (LOCAL_LIGHTS[i].kind == LIGHT_KIND_SPOT)
        {
            float cd = dot(LOCAL_LIGHTS[i].direction, -L);
            float ci = LOCAL_LIGHTS[i].cos_inner;
            float co = LOCAL_LIGHTS[i].cos_outer;
            float t  = clamp((cd - co) / max(ci - co, 1e-4), 0.0, 1.0);
            atten *= t * t;
            // Only spots that claimed a shadow slice sample the array; the
            // rest keep shadow_index at -1 and light without casting.
            int si = LOCAL_LIGHTS[i].shadow_index;
            if (si >= 0 && atten > 0.0)
            {
                atten *= sample_spot_shadow(si, in.world_pos, N, screen_xy);
            }
        }
        float3 radiance = col * intens * atten;

        float3 H  = normalize(V + L);
        float NdL = max(dot(N, L), 0.0);

        float D = distribution_ggx(N, H, roughness);
        float G = geometry_smith(N, V, L, roughness);
        float3 F = fresnel_schlick(max(dot(H, V), 0.0), F0);

        float3 kd   = (1.0 - F) * (1.0 - metallic);
        float3 spec = (D * G * F) / max(4.0 * NdV * NdL, 0.001) * energy_ms;
        float3 diff = kd * albedo / PI;
        Lo += (diff + spec) * radiance * NdL;
    }

    float3 ambient;
    if (ibl_enabled)
    {
        float3 F_ibl       = fresnel_schlick(NdV, F0);
        float3 kd_ibl      = (1.0 - F_ibl) * (1.0 - metallic);
        float3 irradiance  = irradiance_sample(N);
        float3 diffuse_ibl = kd_ibl * albedo * irradiance / PI;

        float3 R = reflect(-V, N);
        // SampleBias (not SampleLevel) so the reflection vector's screen-space
        // footprint widens the mip at grazing or distant angles. A forced LOD
        // defeats minification filtering and aliases the environment into
        // sparkle on near mirrors; flat close-up pixels have a near-zero
        // footprint, so they keep the plain roughness mip.
        float lod = roughness * (VIEW.prefilter_mip_count - 1.0);
        // Local reflection probes when any are baked (box-parallax partition
        // of unity), else the imported environment prefilter cube. With no
        // probe baked the probe count is 0, so this is the sky path unchanged.
        float3 prefiltered = (PROBE_SET.count > 0u)
            ? probe_set_specular(in.world_pos, R, lod)
            : prefilter_sample_bias(R, lod);
        float3 specular_ibl = prefiltered * (F0 * ab.x + ab.y);

        // When an SSR / RT reflection composite owns the sharp specular for
        // glossy surfaces this frame, fade the forward probe specular for
        // glossy dielectrics so the two do not double-count. Metals keep
        // their full albedo-tinted forward specular (the resolve adds only a
        // faint dielectric term), and surfaces rougher than the cut (which
        // the resolve skips) keep theirs too.
        if (VIEW.reflections_enabled > 0.5)
        {
            float fade = smoothstep(REFLECTION_ROUGHNESS_CUT * 0.7,
                                    REFLECTION_ROUGHNESS_CUT, roughness);
            specular_ibl *= lerp(1.0, fade, 1.0 - metallic);
        }

        ambient = diffuse_ibl + specular_ibl;
    }
    else
    {
        ambient = float3(0.03) * albedo;
    }

    // Authored indirect-fill multiplier (PostProcessConfig.ambient_intensity);
    // 1.0 is a no-op. Lifts shadow fill without touching sun-lit surfaces.
    ambient *= LIGHTS.ambient_intensity;

    // SSAO modulates the indirect (ambient / IBL) term only: direct lighting
    // is unaffected. A 1x1 white view is bound when SSAO is disabled, so this
    // samples a constant 1.0 then.
    float2 ssao_uv = screen_xy / ssao_size();
    ambient *= ssao_sample(ssao_uv);

    float3 color = ambient + Lo + emissive;

    return float4(color, albedo_samp.a);
}
