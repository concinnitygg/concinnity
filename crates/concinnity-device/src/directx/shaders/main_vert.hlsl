// D3D12 vertex stage for the main geometry pass.
//
// Two entry points sharing one binding layout:
//   vertex_main            per-draw model matrix from PushConstants
//   vertex_main_instanced  per-instance model matrix from the t3 buffer
//
// Root signature layout (must match directx/pipeline.rs):
//   b0 PushConstants : model mat4 + material (28 DWORDs)
//   b1 ViewBlock     : vp mat4, view mat4, elapsed, cam xyz
//   b3 ShadowBlock   : light_vps[4] mat4 + cascade_splits float4
//   t3 InstanceBlock : StructuredBuffer<float4x4> per-instance world matrices
//                      (vertex_main_instanced only)
//
// Input layout (56-byte Vertex, must match main_input_layout() in directx/pipeline.rs):
//   POSITION  float3  offset  0
//   NORMAL    float3  offset 12
//   TANGENT   float3  offset 24
//   COLOR     float3  offset 36
//   TEXCOORD0 float2  offset 48

#pragma pack_matrix(column_major)

cbuffer PushConstants : register(b0)
{
    // Unused by vertex_main_instanced, which reads its model from `instances`.
    float4x4 model;
    float roughness;
    float metallic;
    float _mpad0;
    float _mpad1;
    float3 tint;
    float _mpad2;
    float3 emissive;
    float _mpad3;
}

cbuffer ViewBlock : register(b1)
{
    float4x4 vp;
    float4x4 view_mat;
    float elapsed;
    float _pad0;
    float cam_x;
    float cam_y;
    float cam_z;
    float _pad1;
    // 1.0 while the unlit view mode is active; read by the fragment stage.
    float shade_mode;
    float _ep1;
}

cbuffer ShadowBlock : register(b3)
{
    float4x4 light_vps[4];
    float4   cascade_splits;
}

// FXC quirk: `pack_matrix(column_major)` reliably applies to matrices that
// are STRUCT MEMBERS inside a StructuredBuffer, but its behaviour for raw
// element-type matrices (`StructuredBuffer<float4x4>`) is ambiguous. Wrapping
// in a struct with an explicit `column_major` qualifier pins the storage
// layout so the matrix reads back as Rust uploaded it (column-major).
struct ColMat4 { column_major float4x4 m; };
StructuredBuffer<ColMat4> instances : register(t3);

struct VsIn
{
    float3 pos     : POSITION;
    float3 normal  : NORMAL;
    float3 tangent : TANGENT;
    float3 color   : COLOR;
    float2 uv      : TEXCOORD0;
};

struct VsOut
{
    float4 sv_pos      : SV_POSITION;
    float3 world_pos   : TEXCOORD0;
    float3 normal      : TEXCOORD1;
    float3 tangent     : TEXCOORD2;
    float3 bitangent   : TEXCOORD3;
    float2 uv          : TEXCOORD4;
    float  view_depth  : TEXCOORD5;
    float3 color       : TEXCOORD6;
};

VsOut transform(VsIn v, float4x4 world_mat)
{
    VsOut o;
    float4 world = mul(world_mat, float4(v.pos, 1.0));
    o.world_pos = world.xyz;

    float3x3 nm = (float3x3)world_mat;
    o.normal    = normalize(mul(nm, v.normal));
    o.tangent   = normalize(mul(nm, v.tangent));
    o.bitangent = cross(o.normal, o.tangent);

    o.uv    = v.uv;
    o.color = v.color;

    // View-space depth (positive in front of camera) for cascade selection.
    o.view_depth = -mul(view_mat, world).z;
    o.sv_pos     = mul(vp, world);
    return o;
}

VsOut vertex_main(VsIn v)
{
    VsOut o = transform(v, model);

    // Skybox sentinel: blue channel > 1.5 forces sky to far plane.
    if (v.color.b > 1.5)
        o.sv_pos.z = o.sv_pos.w * (1.0 - 1e-6);

    return o;
}

// Instanced clusters are never skyboxes, so the sentinel is not applied here.
VsOut vertex_main_instanced(VsIn v, uint iid : SV_InstanceID)
{
    return transform(v, instances[iid].m);
}
