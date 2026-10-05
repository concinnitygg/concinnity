// Unified geometry G-buffer pre-pass: single source for every backend.
//
// One jittered traversal of the cull records writes, in a single MRT,
// everything the screen-space + temporal passes need:
//   target(0) RGBA16F  view-space normal (xyz) + positive linear view depth (a)
//   target(1) R8       perceptual roughness
//   target(2) RG16F    screen-space motion (prev_uv - cur_uv)
// The rasterized position uses the JITTERED VP so coverage matches the main
// pass; the motion vector comes from the UN-jittered cur/prev VPs so jitter
// never leaks into it. Alpha 0 in target(0) marks "no geometry", the sky
// included.
//
// Every host rasterizes this pre-pass off the cull records, then draws the sky
// over whatever they left uncovered, so there are three entries, each selected
// by a define:
//
//   GB_BINDLESS          - object id via first-instance, model and roughness
//                          from the per-frame GpuObjectData buffer
//   GB_SKY               - the sky's vertex stage: a fullscreen triangle at the
//                          far plane, depth-tested without writing, whose
//                          motion is the camera's rotation alone
//   GB_FRAGMENT_BINDLESS - the fragment both draw with (roughness from a
//                          varying)
//
// CN_BACKEND_DIRECTX pins every register to the root signatures in
// directx/post/gbuffer.rs and directx/post/gbuffer_sky.rs, which are the host's
// slots rather than the Metal buffer indices the shared declarations otherwise
// carry: b0 goes to the indirect command's object id, the view CBV follows at
// b1, and every structured buffer starts from t0.
//
// CN_BACKEND_DIRECTX also selects how DirectX delivers the object id, matching
// what main_bindless.hlsl already does there: it rides that b0 root constant rather
// than an instance-id builtin, which on that leg is zero-based and says nothing
// about the record. The other two legs read it straight off SV_InstanceID,
// which includes the base instance the cull wrote there.

{OBJECT_COMMON}
{DEPTH_CONVENTION}
{SKY_RAY}

// Layout matches `GBufferView` / `GbViewUniforms` (4 x float4x4, 256 B).
struct GbView
{
    float4x4 jittered_vp;
    float4x4 cur_vp;
    float4x4 prev_vp;
    float4x4 view_mat;
};

#if defined(GB_BINDLESS) || defined(GB_SKY)
#ifdef CN_BACKEND_DIRECTX
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
// into each indirect command's first-instance value; the model-history ring
// slot the PREVIOUS frame's `model_history.hlsl` dispatch filled, indexed
// identically; and this frame's draw args, read only for `DRAW_NO_HISTORY`.
#ifdef CN_BACKEND_DIRECTX
struct ObjectId { uint value; };
ConstantBuffer<ObjectId> objid_cb : register(b0);
StructuredBuffer<GpuObjectData> objects : register(t0);
StructuredBuffer<float4x4> prev_models : register(t1);
StructuredBuffer<GpuDrawArgs> draw_args : register(t2);
#else
[[vk::binding(0, 1)]] StructuredBuffer<GpuObjectData> objects : register(t9);
[[vk::binding(1, 0)]] StructuredBuffer<float4x4> prev_models : register(t10);
[[vk::binding(2, 0)]] StructuredBuffer<GpuDrawArgs> draw_args : register(t11);
#endif

// The transform to reproject last frame's position through: the history entry,
// or this frame's own model where no history exists, which collapses the motion
// vector to the camera's own (and to exactly zero when `prev_vp == cur_vp`).
float4x4 gb_prev_model(uint oid, float4x4 cur_model)
{
    if ((draw_args[oid].flags & DRAW_NO_HISTORY) != 0u)
    {
        return cur_model;
    }
    return prev_models[oid];
}
#endif

// ---- Stage interfaces ----

// The vertex stage reads a second vertex stream: the previous frame's
// position, at attribute 5. The static + instance + chunk prefix binds the same
// buffer to both streams (prev_pos == cur_pos, so motion is the model delta
// plus camera); the skinned tail binds the previous-frame deformed buffer.
struct GbBindlessVertexIn
{
    [[vk::location(0)]] float3 pos      : POSITION;
    [[vk::location(1)]] float3 normal   : NORMAL;
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

#endif

// Any component past 1 lands every pixel off the image, which every reader
// treats as missing history.
static const float GB_MOTION_LIMIT = 2.0;
static const float GB_MIN_PREV_W = 1e-6;

// Stored so the TAA pass can do `prev_uv = uv + motion`. Image-space UV with
// 0 = top, matching the upright resolve the readers sample. A point at or
// behind the previous camera has no previous pixel, and the clamp keeps a far
// off-screen one finite in the RG16F target.
float2 gb_motion(float4 cur_clip, float4 prev_clip)
{
    if (!(prev_clip.w > GB_MIN_PREV_W))
        return (float2)(GB_MOTION_LIMIT);
    float2 cur_ndc  = cur_clip.xy  / cur_clip.w;
    float2 prev_ndc = prev_clip.xy / prev_clip.w;
    float2 cur_uv  = float2(cur_ndc.x  * 0.5 + 0.5, 0.5 - cur_ndc.y  * 0.5);
    float2 prev_uv = float2(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
    return clamp(prev_uv - cur_uv, -GB_MOTION_LIMIT, GB_MOTION_LIMIT);
}

// ---- Entry points ----

#ifdef GB_BINDLESS

[shader("vertex")]
GbVertexOut gbuffer_prepass_vertex_bindless(
    GbBindlessVertexIn v
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
    GpuObjectData obj = objects[oid];
    float4 cur_world  = mul(obj.model, float4(v.pos, 1.0));
    float4 prev_world = mul(gb_prev_model(oid, obj.model), float4(v.prev_pos, 1.0));
    GbVertexOut o = gb_project(obj.model, cur_world, prev_world, v.normal);
    o.roughness = obj.tint_roughness.w;
    return o;
}

#endif

#ifdef GB_SKY

// The sky reads as "no geometry" to every screen-space pass (view depth 0, the
// background's roughness) while still writing its motion: the view ray through
// the jittered projection, reprojected through this frame's and the previous
// frame's unjittered matrices. A direction has no position, so the camera's
// translation drops out and the motion is its rotation alone.
[shader("vertex")]
GbVertexOut gbuffer_sky_vertex(uint vid : SV_VertexID)
{
    float2 ndc = sky_corner(vid);
    float3 ray = sky_ray(gb_view.jittered_vp, ndc);
    GbVertexOut o;
    o.position    = float4(ndc, DEPTH_FAR, 1.0);
    o.view_normal = (float3)(0.0);
    o.view_depth  = 0.0;
    o.cur_clip    = mul(gb_view.cur_vp, float4(ray, 0.0));
    o.prev_clip   = mul(gb_view.prev_vp, float4(ray, 0.0));
    o.roughness   = 1.0;
    return o;
}

#endif

#ifdef GB_FRAGMENT_BINDLESS

[shader("pixel")]
GbFragmentOut gbuffer_prepass_fragment_bindless(GbVertexOut p)
{
    GbFragmentOut o;
    o.nd    = p.view_depth > 0.0 ? float4(normalize(p.view_normal), p.view_depth) : (float4)(0.0);
    o.rough = p.roughness;
    o.vel   = gb_motion(p.cur_clip, p.prev_clip);
    return o;
}

#endif
