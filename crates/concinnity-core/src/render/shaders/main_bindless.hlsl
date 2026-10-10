// Bindless forward pass: vertex + fragment, single source for every backend.
// Its records, bindings and shading model are `main_resources.hlsl`; this file
// adds the hooks and the entry points. On DirectX the object id comes from the
// b0 root constant the indirect command writes rather than from an
// instance-id builtin.
//
// A world Shader compiles from this same file with its hooks spliced at
// SURFACE_VERTEX / SURFACE_FRAGMENT, so it lands on these slots by
// construction and never names one.
//
// Under SURFACE_PREPASS the same text compiles the G-buffer pre-pass instead:
// the world's vertex hook positions the surface, and the fragment writes the
// normal, roughness and cutout `shade_surface` would light (see
// `gbuffer_common.hlsl` for the targets).
//
// The records this binds are `main_types.hlsl` and the shading model it drives
// is `main_shading.hlsl`.

{MAIN_RESOURCES}

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
