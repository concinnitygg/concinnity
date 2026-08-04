#pragma pack_matrix(column_major)

// The vertex shader only reads `model`, but the full record is required so
// `objects[object_id]` strides correctly through the per-object buffer.
{OBJECT_DATA}

cbuffer ObjId : register(b0)
{
    uint object_id;
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
    float prefilter_mip_count;
    // 1.0 while the unlit view mode is active; read by the fragment stage.
    float shade_mode;
    float _ep1;
}

StructuredBuffer<GpuObjectData> objects : register(t3);

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
    float4 sv_pos     : SV_POSITION;
    float3 world_pos  : TEXCOORD0;
    float3 normal     : TEXCOORD1;
    float3 tangent    : TEXCOORD2;
    float3 bitangent  : TEXCOORD3;
    float2 uv         : TEXCOORD4;
    float  view_depth : TEXCOORD5;
    float3 color      : TEXCOORD6;
};

VsOut main(VsIn v)
{
    VsOut o;
    float4x4 model = objects[object_id].model;
    float4 world = mul(model, float4(v.pos, 1.0));
    o.world_pos = world.xyz;

    float3x3 nm = (float3x3)model;
    o.normal    = normalize(mul(nm, v.normal));
    o.tangent   = normalize(mul(nm, v.tangent));
    o.bitangent = cross(o.normal, o.tangent);

    o.uv    = v.uv;
    o.color = v.color;

    o.view_depth = -mul(view_mat, world).z;
    o.sv_pos     = mul(vp, world);
    // Skybox sentinel: pin to the far plane so the sky shell, whose corners sit
    // beyond a typical `far`, is never clipped and always renders behind scene.
    if (v.color.b > 1.5)
    {
        o.sv_pos.z = o.sv_pos.w * (1.0 - 1e-6);
    }
    return o;
}
