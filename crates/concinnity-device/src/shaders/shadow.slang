// Cascaded shadow pass: single source for every backend.
//
// Depth-only. The scene is rendered from the directional light's perspective
// into one slice of a Depth32Float texture array, one slice per cascade, and
// the host loops the pass once per cascade pushing the slot to project through.
// There is no fragment stage on any backend -- depth writes are automatic --
// so these entries have no varying interface to match.
//
// One entry per compile, selected by a define, so each variant declares exactly
// the resources it binds and nothing reserves a slot a host never writes:
//
//   SHADOW_STATIC   - per-draw casters, model matrix from the host
//   SHADOW_SKINNED  - per-draw skinned casters, one joint palette
//   SHADOW_BINDLESS - GPU-driven: the model comes from the per-frame
//                     GpuObjectData record the cull baked into each indirect
//                     command's first-instance value. The deformed skinned tail
//                     rides this same entry: its vertices are already
//                     model-space, so `light_vp * model * deformed_pos` matches
//                     the static case.
//
// METAL_BINDINGS selects the Metal *host's* constant shape, a host difference
// rather than a target one: Vulkan carries the model matrix and the cascade
// index in one push constant block (a pipeline layout may declare only one),
// while the Metal encoder writes the model to buffer(2) and the cascade to
// buffer(7) -- the split the spot-shadow pass shares.
//
// DXIL_ABI pins every register to the shadow root signatures in
// directx/init/pipelines.rs and directx/resources.rs. DirectX reuses Vulkan's
// constant *shape* (the per-draw entries take one 20-DWORD block holding the
// model and the cascade; the GPU-driven one takes the cascade alone) but hands
// it over as root constants at b0 / b2, which pushes the shadow CBV to b1 on
// every variant and starts the structured buffers at t0. The object id comes
// from that b0 root constant the indirect command writes, exactly as
// main_bindless.slang does, rather than from SV_StartInstanceLocation.

{OBJECT_COMMON}

// Layout matches `ShadowUniforms` in render_types.rs.
struct ShadowUniforms
{
    float4x4 light_vps[4];
    float4 cascade_splits;
};

#ifdef DXIL_ABI
// b0 carries the per-draw root constants on every variant, so the shadow CBV
// follows at b1.
ConstantBuffer<ShadowUniforms> shadow_cb : register(b1);
#else
[[vk::binding(0, 0)]] ConstantBuffer<ShadowUniforms> shadow_cb : register(b0);
#endif

#ifdef METAL_BINDINGS

struct ModelUniforms { float4x4 model; };
// Layout matches `ShadowPassPush` in render_types.rs (16 B).
struct ShadowPassPush { uint cascade_idx; uint _pad0; uint _pad1; uint _pad2; };

#if defined(SHADOW_STATIC) || defined(SHADOW_SKINNED)
ConstantBuffer<ModelUniforms> model_cb : register(b2);
#define SHADOW_MODEL model_cb.model
#endif
ConstantBuffer<ShadowPassPush> cascade_cb : register(b7);
#define SHADOW_CASCADE cascade_cb.cascade_idx

#else

// The per-draw block: model matrix plus the cascade slot, padded to 80 B.
struct ShadowPush
{
    float4x4 model;
    uint cascade_idx;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// The GPU-driven pass pushes only the cascade slot: its model comes from the
// object record.
struct CascadePush { uint cascade_idx; };

#ifdef SHADOW_BINDLESS
#ifdef DXIL_ABI
// One root constant per cascade's ExecuteIndirect; b0 is the object id.
ConstantBuffer<CascadePush> push : register(b2);
#else
[[vk::push_constant]] ConstantBuffer<CascadePush> push;
#endif
#else
#ifdef DXIL_ABI
ConstantBuffer<ShadowPush> push : register(b0);
#else
[[vk::push_constant]] ConstantBuffer<ShadowPush> push;
#endif
#define SHADOW_MODEL push.model
#endif
#define SHADOW_CASCADE push.cascade_idx

#endif

#ifdef SHADOW_BINDLESS
#ifdef DXIL_ABI
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

#ifdef SHADOW_STATIC

[shader("vertex")]
float4 shadow_vertex_main(ShadowVertexIn v) : SV_Position
{
    return mul(shadow_cb.light_vps[SHADOW_CASCADE], mul(SHADOW_MODEL, float4(v.pos, 1.0)));
}

#endif

#ifdef SHADOW_SKINNED

// Only the attributes the depth-only skinned caster consumes are declared, so
// the pipeline is validation-clean (no "attribute not consumed" findings).
struct SkinnedShadowVertexIn
{
    [[vk::location(0)]] float3 pos     : POSITION;
    [[vk::location(5)]] uint4  joints  : BLENDINDICES;
    [[vk::location(6)]] float4 weights : BLENDWEIGHT;
};

#ifdef DXIL_ABI
StructuredBuffer<float4x4> joints : register(t0);
#else
[[vk::binding(0, 1)]] StructuredBuffer<float4x4> joints : register(t8);
#endif

[shader("vertex")]
float4 shadow_vertex_main_skinned(SkinnedShadowVertexIn v) : SV_Position
{
    float4x4 skin = v.weights.x * joints[v.joints.x]
                  + v.weights.y * joints[v.joints.y]
                  + v.weights.z * joints[v.joints.z]
                  + v.weights.w * joints[v.joints.w];
    float4 skinned_pos = mul(skin, float4(v.pos, 1.0));
    return mul(shadow_cb.light_vps[SHADOW_CASCADE], mul(SHADOW_MODEL, skinned_pos));
}

#endif

#ifdef SHADOW_BINDLESS

[shader("vertex")]
float4 shadow_vertex_bindless(
    ShadowVertexIn v
#ifdef DXIL_ABI
    ) : SV_Position
{
    uint oid = objid_cb.value;
#else
    ,
    uint instance_id : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation) : SV_Position
{
    uint oid = object_instance_index(instance_id, first_instance);
#endif
    float4x4 model = objects[oid].model;
    return mul(shadow_cb.light_vps[SHADOW_CASCADE], mul(model, float4(v.pos, 1.0)));
}

#endif
