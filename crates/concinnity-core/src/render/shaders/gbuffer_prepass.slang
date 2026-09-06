// Unified geometry G-buffer pre-pass: single source for every backend.
//
// One jittered traversal of the cull records writes, in a single MRT,
// everything the screen-space + temporal passes need:
//   target(0) RGBA16F  view-space normal (xyz) + positive linear view depth (a)
//   target(1) R8       perceptual roughness
//   target(2) RG16F    screen-space motion (prev_uv - cur_uv)
// The rasterised position uses the JITTERED VP so coverage matches the main
// pass; the motion vector comes from the UN-jittered cur/prev VPs so jitter
// never leaks into it. Alpha 0 in target(0) marks "no geometry".
//
// Every host rasterises this pre-pass off the cull records, so there are two
// entries, one per stage, each selected by a define:
//
//   GB_BINDLESS          - object id via first-instance, model and roughness
//                          from the per-frame GpuObjectData buffer
//   GB_FRAGMENT_BINDLESS - the fragment (roughness from a varying)
//
// DXIL_ABI pins every register to the root signatures in
// directx/post/gbuffer.rs, which are the host's slots rather than the Metal
// buffer indices the shared declarations otherwise carry: b0 goes to the
// indirect command's object id, the view CBV follows at b1, and every
// structured buffer starts from t0.
//
// DXIL_ABI also selects how DirectX delivers the object id, matching what
// main_bindless.slang already does there: it rides that b0 root constant, so the
// entry declares no SV_StartInstanceLocation (which would raise the container's
// floor to shader model 6.8 to reproduce an index the engine already has).

{OBJECT_COMMON}

// Layout matches `GBufferView` / `GbViewUniforms` (4 x float4x4, 256 B).
struct GbView
{
    float4x4 jittered_vp;
    float4x4 cur_vp;
    float4x4 prev_vp;
    float4x4 view_mat;
};

#ifdef GB_BINDLESS
#ifdef DXIL_ABI
// b0 belongs to the indirect command's object-id root constant, so the view
// CBV follows it at b1.
ConstantBuffer<GbView> gb_view : register(b1);
#else
[[vk::binding(0, 0)]] ConstantBuffer<GbView> gb_view : register(b0);
#endif
#endif

// ---- Per-variant geometry sources ----

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

// The vertex stage reads a second vertex stream: the previous frame's
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

#ifdef GB_BINDLESS

// Everything but roughness and the sky pin, from a world-space position pair
// and the model matrix whose normal transform the surface normal rides.
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

#ifdef GB_BINDLESS

[shader("vertex")]
GbVertexOut gbuffer_prepass_vertex_bindless(
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
    GbVertexOut o = gb_project(obj.model, cur_world, prev_world, v.normal);
    o.position  = gb_sky_pin(o.position, v.color);
    o.roughness = obj.tint_roughness.w;
    return o;
}

#endif

#ifdef GB_FRAGMENT_BINDLESS

[shader("fragment")]
GbFragmentOut gbuffer_prepass_fragment_bindless(GbVertexOut p)
{
    GbFragmentOut o;
    o.nd    = float4(normalize(p.view_normal), p.view_depth);
    o.rough = p.roughness;
    o.vel   = gb_motion(p.cur_clip, p.prev_clip);
    return o;
}

#endif
