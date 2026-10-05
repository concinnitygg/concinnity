// Shadow pass: single source for every backend.
//
// Depth-only. The scene is rendered from a light's perspective into one slice
// of a Depth32Float texture array: a cascade of the directional light, or a
// spot light's slice. The host loops the pass once per slice, pushing which of
// the bound `light_vps` to project through. There is no fragment stage on any
// backend -- depth writes are automatic -- so the entry has no varying
// interface to match.
//
// GPU-driven: the model comes from the per-frame GpuObjectData record the cull
// baked into each indirect command's first-instance value. The deformed skinned
// tail rides this same entry: its vertices are already model-space, so
// `light_vp * model * deformed_pos` matches the static case.
//
// CN_BACKEND_METAL selects the Metal host's constant shape: the view index sits
// at buffer(7). CN_BACKEND_DIRECTX pins every register to the shadow root
// signature in directx/init/pipelines.rs: the object id is the b0 root constant
// the indirect command writes, exactly as main_bindless.hlsl does, the shadow
// CBV follows at b1, the view index at b2, and the object records at t0.
// Vulkan reads the object id straight off SV_InstanceID, which includes the
// base instance the cull wrote there.

{OBJECT_COMMON}

// Layout matches `ShadowUniforms` in render_types.rs.
struct ShadowUniforms
{
    float4x4 light_vps[4];
    float4 cascade_splits;
};

#ifdef CN_BACKEND_DIRECTX
// b0 carries the object id root constant, so the shadow CBV follows at b1.
ConstantBuffer<ShadowUniforms> shadow_cb : register(b1);
#else
[[vk::binding(0, 0)]] ConstantBuffer<ShadowUniforms> shadow_cb : register(b0);
#endif

#ifdef CN_BACKEND_METAL

// Layout matches `ShadowPassPush` in render_types.rs (16 B).
struct ShadowPassPush { uint cascade_idx; uint _pad0; uint _pad1; uint _pad2; };

// The `[[vk::binding]]` is the slot dxc assigns the block from its register
// number anyway, and it is what pairs a reflected resource with the declaration
// its Metal index comes from -- so a resource without one cannot be placed,
// even on a block no Vulkan host binds.
[[vk::binding(7, 0)]] ConstantBuffer<ShadowPassPush> cascade_cb : register(b7);
#define SHADOW_CASCADE cascade_cb.cascade_idx

#else

// The view slot is the only constant: the model comes from the object record.
struct CascadePush { uint cascade_idx; };

#ifdef CN_BACKEND_DIRECTX
// One root constant per view's ExecuteIndirect; b0 is the object id.
ConstantBuffer<CascadePush> push : register(b2);
#else
[[vk::push_constant]] ConstantBuffer<CascadePush> push : register(b0);
#endif
#define SHADOW_CASCADE push.cascade_idx

#endif

#ifdef SHADOW_BINDLESS
#ifdef CN_BACKEND_DIRECTX
struct ObjectId { uint value; };
ConstantBuffer<ObjectId> objid_cb : register(b0);
StructuredBuffer<GpuObjectData> objects : register(t0);
#else
[[vk::binding(0, 1)]] StructuredBuffer<GpuObjectData> objects : register(t9);
#endif
#endif

// The full static layout is declared so the pipeline's vertex descriptor is
// consumed exactly as the other passes declare it, even though only the
// position is read.
struct ShadowVertexIn
{
    [[vk::location(0)]] float3 pos     : POSITION;
    [[vk::location(1)]] float3 normal  : NORMAL;
    [[vk::location(2)]] float3 tangent : TANGENT;
    [[vk::location(3)]] float3 color   : COLOR0;
    [[vk::location(4)]] float2 uv      : TEXCOORD0;
};

#ifdef SHADOW_BINDLESS

[shader("vertex")]
float4 shadow_vertex_bindless(
    ShadowVertexIn v
#ifdef CN_BACKEND_DIRECTX
    ) : SV_Position
{
    uint oid = objid_cb.value;
#else
    ,
    uint instance_id : SV_InstanceID) : SV_Position
{
    uint oid = object_instance_index(instance_id);
#endif
    float4x4 model = objects[oid].model;
    return mul(shadow_cb.light_vps[SHADOW_CASCADE], mul(model, float4(v.pos, 1.0)));
}

#endif
