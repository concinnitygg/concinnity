// Unified geometry G-buffer pre-pass: single source for every backend.
//
// One jittered traversal of the visible set writes, in a single MRT, everything
// the screen-space + temporal passes need:
//   target(0) RGBA16F  view-space normal (xyz) + positive linear view depth (a)
//   target(1) R8       perceptual roughness
//   target(2) RG16F    screen-space motion (prev_uv - cur_uv)
// The rasterised position uses the JITTERED VP so coverage matches the main
// pass; the motion vector comes from the UN-jittered cur/prev VPs so jitter
// never leaks into it. Alpha 0 in target(0) marks "no geometry".
//
// One entry per compile, selected by a define, so each variant declares exactly
// the resources it binds and nothing reserves a slot a host never writes:
//
//   GB_STATIC            - per-draw static geometry
//   GB_INSTANCED         - GPU-instanced clusters (transforms immutable)
//   GB_SKINNED           - per-draw skinned meshes, cur + prev joint palettes
//   GB_BINDLESS          - GPU-driven: object id via first-instance, model and
//                          roughness from the per-frame GpuObjectData buffer
//   GB_FRAGMENT          - the per-draw fragment (roughness from a constant)
//   GB_FRAGMENT_BINDLESS - the GPU-driven fragment (roughness from a varying)
//
// METAL_BINDINGS selects the Metal *host's* constant shape, which is a host
// difference rather than a target one: Vulkan hands both stages one push
// constant block holding the model pair and the roughness, while the Metal
// encoder writes the model pair to vertex buffer(2) and the roughness to
// fragment buffer(0). DirectX splits them the same way Metal does, at its own
// slots. Everything else carries both a `[[vk::binding]]` and a `register()` on
// one declaration, so the Vulkan descriptor sets and the Metal buffer indices
// both reproduce what the hosts already bind.
//
// DXIL_ABI pins every register to the root signatures in
// directx/post/gbuffer.rs, which are the host's slots rather than the Metal
// buffer indices the declarations otherwise carry: the per-draw entries put the
// view CBV at b0 and the model pair at b1, the GPU-driven one gives b0 to the
// indirect command's object id and pushes the view CBV to b1, and every
// structured buffer starts from t0. The fragment's roughness constants also sit
// at b0 -- pixel-visible, so they never collide with a vertex CBV.
//
// DXIL_ABI also selects how DirectX delivers a per-draw index, matching what
// main_bindless.slang already does there: the object id rides that b0 root
// constant, and instanced clusters bind their bucket's transforms from zero, so
// neither entry declares SV_StartInstanceLocation (which would raise the
// container's floor to shader model 6.8 to reproduce an index the engine
// already has).

{OBJECT_COMMON}

// Layout matches `GBufferView` / `GbViewUniforms` (4 x float4x4, 256 B).
struct GbView
{
    float4x4 jittered_vp;
    float4x4 cur_vp;
    float4x4 prev_vp;
    float4x4 view_mat;
};

#if defined(GB_STATIC) || defined(GB_INSTANCED) || defined(GB_SKINNED) || defined(GB_BINDLESS)
#define GB_VERTEX_STAGE 1
#endif

#ifdef GB_VERTEX_STAGE
#if defined(DXIL_ABI) && defined(GB_BINDLESS)
// b0 belongs to the indirect command's object-id root constant, so the view
// CBV follows it at b1.
ConstantBuffer<GbView> gb_view : register(b1);
#else
[[vk::binding(0, 0)]] ConstantBuffer<GbView> gb_view : register(b0);
#endif
#endif

// ---- The per-draw model pair and roughness ----

#if defined(METAL_BINDINGS) || defined(DXIL_ABI)

// Layout matches `VelocityModelUniforms` / `GbModelPush` (128 B), at vertex
// buffer(2) on Metal and b1 on DirectX.
struct GbModel
{
    float4x4 cur_model;
    float4x4 prev_model;
};

// Layout matches `SsrPrepassMat` (16 B), at fragment buffer(0) on Metal and the
// pixel-visible b0 on DirectX. Padded with plain floats: a float3 would force
// 16-byte alignment and bloat the struct.
struct GbMat
{
    float roughness;
    float _pad0;
    float _pad1;
    float _pad2;
};

#if defined(GB_STATIC) || defined(GB_SKINNED)
#ifdef DXIL_ABI
ConstantBuffer<GbModel> gb_model : register(b1);
#else
ConstantBuffer<GbModel> gb_model : register(b2);
#endif
#endif
#ifdef GB_FRAGMENT
ConstantBuffer<GbMat> gb_mat : register(b0);
#endif

#define GB_CUR_MODEL  gb_model.cur_model
#define GB_PREV_MODEL gb_model.prev_model
#define GB_ROUGHNESS  gb_mat.roughness

#else

// Layout matches `GbModelPush` (144 B). Both stages see the whole block; the
// vertex reads the model pair, the fragment only `roughness` at offset 128.
struct GbModelPush
{
    float4x4 cur_model;
    float4x4 prev_model;
    float roughness;
    float _pad0;
    float _pad1;
    float _pad2;
};

#if defined(GB_STATIC) || defined(GB_SKINNED) || defined(GB_FRAGMENT)
[[vk::push_constant]] ConstantBuffer<GbModelPush> gb_push;
#endif

#define GB_CUR_MODEL  gb_push.cur_model
#define GB_PREV_MODEL gb_push.prev_model
#define GB_ROUGHNESS  gb_push.roughness

#endif

// ---- Per-variant geometry sources ----

#ifdef GB_INSTANCED
// Per-instance world matrices. Instance transforms are immutable, so cur ==
// prev and the motion vector is camera-only.
#ifdef DXIL_ABI
StructuredBuffer<float4x4> instances : register(t0);
#else
[[vk::binding(0, 1)]] StructuredBuffer<float4x4> instances : register(t6);
#endif
#endif

#ifdef GB_SKINNED
// The current and previous frame's joint palettes, so per-vertex skin
// deformation produces a motion vector. Both reuse the main-pass joint set.
#ifdef DXIL_ABI
StructuredBuffer<float4x4> cur_joints : register(t0);
StructuredBuffer<float4x4> prev_joints : register(t1);
#else
[[vk::binding(0, 1)]] StructuredBuffer<float4x4> cur_joints : register(t8);
[[vk::binding(0, 2)]] StructuredBuffer<float4x4> prev_joints : register(t9);
#endif
#endif

#ifdef GB_BINDLESS
// The cull-produced per-frame records, indexed by the object id the cull baked
// into each indirect command's first-instance value, and the parallel
// previous-frame model matrices indexed identically.
#ifdef DXIL_ABI
struct ObjectId { uint value; };
ConstantBuffer<ObjectId> objid_cb : register(b0);
StructuredBuffer<GpuObjectData> objects : register(t0);
StructuredBuffer<float4x4> prev_models : register(t1);
#else
[[vk::binding(0, 1)]] StructuredBuffer<GpuObjectData> objects : register(t9);
[[vk::binding(1, 0)]] StructuredBuffer<float4x4> prev_models : register(t10);
#endif
#endif

// ---- Stage interfaces ----

struct GbVertexIn
{
    [[vk::location(0)]] float3 pos     : POSITION;
    [[vk::location(1)]] float3 normal  : NORMAL;
    [[vk::location(2)]] float3 tangent : TANGENT;
    [[vk::location(3)]] float3 color   : COLOR0;
    [[vk::location(4)]] float2 uv      : TEXCOORD0;
};

struct GbSkinnedVertexIn
{
    [[vk::location(0)]] float3 pos     : POSITION;
    [[vk::location(1)]] float3 normal  : NORMAL;
    [[vk::location(2)]] float3 tangent : TANGENT;
    [[vk::location(3)]] float3 color   : COLOR0;
    [[vk::location(4)]] float2 uv      : TEXCOORD0;
    [[vk::location(5)]] uint4  joints  : BLENDINDICES;
    [[vk::location(6)]] float4 weights : BLENDWEIGHT;
};

// The GPU-driven variant reads a second vertex stream: the previous frame's
// position, at attribute 5. The static + instance + chunk prefix binds the same
// buffer to both streams (prev_pos == cur_pos, so motion is the model delta
// plus camera); the skinned tail binds the previous-frame deformed buffer.
//
// The previous position takes a name of its own rather than POSITION1: slangc
// appends its own index to whatever the semantic spells, so a trailing digit is
// multiplied by ten on the way out (POSITION1 lands as POSITION index 10). That
// is harmless between two stages that agree, which is why the varyings below
// keep their TEXCOORDn names, but a vertex input has to match the index a host
// input layout declares.
struct GbBindlessVertexIn
{
    [[vk::location(0)]] float3 pos      : POSITION;
    [[vk::location(1)]] float3 normal   : NORMAL;
    [[vk::location(3)]] float3 color    : COLOR0;
    [[vk::location(5)]] float3 prev_pos : PREVPOSITION;
};

struct GbVertexOut
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 view_normal : TEXCOORD0;
    // Positive view-space depth (-z); the consumers rebuild view position from it.
    [[vk::location(1)]] float  view_depth  : TEXCOORD1;
    [[vk::location(2)]] float4 cur_clip    : TEXCOORD2;
    [[vk::location(3)]] float4 prev_clip   : TEXCOORD3;
};

struct GbVertexOutBindless
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 view_normal : TEXCOORD0;
    [[vk::location(1)]] float  view_depth  : TEXCOORD1;
    [[vk::location(2)]] float4 cur_clip    : TEXCOORD2;
    [[vk::location(3)]] float4 prev_clip   : TEXCOORD3;
    // Sourced from the object record, so the fragment needs no per-draw constant.
    [[vk::location(4)]] nointerpolation float roughness : TEXCOORD4;
};

struct GbFragmentOut
{
    float4 nd    : SV_Target0;
    float  rough : SV_Target1;
    float2 vel   : SV_Target2;
};

// ---- Shared body ----

#ifdef GB_VERTEX_STAGE

// Everything but roughness, from a world-space position pair and the model
// matrix whose normal transform the surface normal rides.
GbVertexOut gb_project(float4x4 model, float4 cur_world, float4 prev_world, float3 model_normal)
{
    GbVertexOut o;
    o.position  = mul(gb_view.jittered_vp, cur_world);
    o.cur_clip  = mul(gb_view.cur_vp,  cur_world);
    o.prev_clip = mul(gb_view.prev_vp, prev_world);
    // Inverse-transpose, matching the forward pass this feeds: a non-uniform
    // scale rotates a plain model-matrix normal off the surface, which would
    // leave SSAO / SSR / SSGI shading a different normal than the lighting.
    float3 world_n = normalize(mul(normal_matrix(model), model_normal));
    o.view_normal  = mul((float3x3)gb_view.view_mat, world_n);
    o.view_depth   = -mul(gb_view.view_mat, cur_world).z;
    return o;
}

// Skybox vertices carry a blue channel of 2.0: pin them to the far plane so the
// sky never occludes scene geometry.
float4 gb_sky_pin(float4 position, float3 color)
{
    if (color.b > 1.5)
    {
        position.z = position.w * (1.0 - 1e-6);
    }
    return position;
}

// 4-influence linear blend skinning from one joint palette.
float4x4 gb_skin(StructuredBuffer<float4x4> palette, uint4 joints, float4 weights)
{
    return weights.x * palette[joints.x]
         + weights.y * palette[joints.y]
         + weights.z * palette[joints.z]
         + weights.w * palette[joints.w];
}

#endif

// Stored so the TAA pass can do `prev_uv = uv + motion`. Image-space UV with
// 0 = top, matching the upright resolve the readers sample.
float2 gb_motion(float4 cur_clip, float4 prev_clip)
{
    float2 cur_ndc  = cur_clip.xy  / cur_clip.w;
    float2 prev_ndc = prev_clip.xy / prev_clip.w;
    float2 cur_uv  = float2(cur_ndc.x  * 0.5 + 0.5, 0.5 - cur_ndc.y  * 0.5);
    float2 prev_uv = float2(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
    return prev_uv - cur_uv;
}

// ---- Entry points ----

#ifdef GB_STATIC

[shader("vertex")]
GbVertexOut gbuffer_prepass_vertex(GbVertexIn v)
{
    float4 cur_world  = mul(GB_CUR_MODEL,  float4(v.pos, 1.0));
    float4 prev_world = mul(GB_PREV_MODEL, float4(v.pos, 1.0));
    GbVertexOut o = gb_project(GB_CUR_MODEL, cur_world, prev_world, v.normal);
    o.position = gb_sky_pin(o.position, v.color);
    return o;
}

#endif

#ifdef GB_INSTANCED

// Vulkan partitions one cluster-wide transform array by LOD bucket and walks
// `firstInstance` across it, so the index has to include the base; Metal and
// DirectX bind each bucket's slice from zero.
[shader("vertex")]
GbVertexOut gbuffer_prepass_vertex_instanced(
    GbVertexIn v,
#ifdef DXIL_ABI
    uint instance_id : SV_InstanceID)
{
    uint iid = instance_id;
#else
    uint instance_id : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    uint iid = object_instance_index(instance_id, first_instance);
#endif
    float4x4 model = instances[iid];
    float4 world = mul(model, float4(v.pos, 1.0));
    // Immutable transforms: prev == cur, so the motion is camera-only.
    return gb_project(model, world, world, v.normal);
}

#endif

#ifdef GB_SKINNED

[shader("vertex")]
GbVertexOut gbuffer_prepass_vertex_skinned(GbSkinnedVertexIn v)
{
    float4x4 cur_skin  = gb_skin(cur_joints, v.joints, v.weights);
    float4x4 prev_skin = gb_skin(prev_joints, v.joints, v.weights);
    float4 cur_world  = mul(GB_CUR_MODEL,  mul(cur_skin,  float4(v.pos, 1.0)));
    float4 prev_world = mul(GB_PREV_MODEL, mul(prev_skin, float4(v.pos, 1.0)));
    float3 skinned_n  = mul((float3x3)cur_skin, v.normal);
    return gb_project(GB_CUR_MODEL, cur_world, prev_world, skinned_n);
}

#endif

#ifdef GB_BINDLESS

[shader("vertex")]
GbVertexOutBindless gbuffer_prepass_vertex_bindless(
    GbBindlessVertexIn v
#ifdef DXIL_ABI
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
    GpuObjectData obj = objects[oid];
    float4 cur_world  = mul(obj.model, float4(v.pos, 1.0));
    float4 prev_world = mul(prev_models[oid], float4(v.prev_pos, 1.0));
    GbVertexOut p = gb_project(obj.model, cur_world, prev_world, v.normal);

    GbVertexOutBindless o;
    o.position    = gb_sky_pin(p.position, v.color);
    o.view_normal = p.view_normal;
    o.view_depth  = p.view_depth;
    o.cur_clip    = p.cur_clip;
    o.prev_clip   = p.prev_clip;
    o.roughness   = obj.roughness;
    return o;
}

#endif

#ifdef GB_FRAGMENT

[shader("fragment")]
GbFragmentOut gbuffer_prepass_fragment(GbVertexOut p)
{
    GbFragmentOut o;
    o.nd    = float4(normalize(p.view_normal), p.view_depth);
    o.rough = GB_ROUGHNESS;
    o.vel   = gb_motion(p.cur_clip, p.prev_clip);
    return o;
}

#endif

#ifdef GB_FRAGMENT_BINDLESS

[shader("fragment")]
GbFragmentOut gbuffer_prepass_fragment_bindless(GbVertexOutBindless p)
{
    GbFragmentOut o;
    o.nd    = float4(normalize(p.view_normal), p.view_depth);
    o.rough = p.roughness;
    o.vel   = gb_motion(p.cur_clip, p.prev_clip);
    return o;
}

#endif
