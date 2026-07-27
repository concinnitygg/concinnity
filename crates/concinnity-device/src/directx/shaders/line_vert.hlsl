#pragma pack_matrix(column_major)

// World-space line pass - vertex shader. Mirrors line_vertex in
// src/metal/shaders/line.metal. The CPU expanded each line into a
// camera-facing ribbon whose corners already sit off the line centre
// (gfx::lines::build_vertices), so this stage only applies the camera VP.

cbuffer LineView : register(b0)
{
    float4x4 vp;
    float    occluded_alpha;
    float    _pad0;
    float    _pad1;
    float    _pad2;
}

struct VsIn
{
    float3 pos   : POSITION;
    float  edge  : TEXCOORD0;
    float4 color : COLOR;
};

struct VsOut
{
    float4 sv_pos : SV_POSITION;
    float  edge   : TEXCOORD0;
    float4 color  : COLOR;
};

VsOut main(VsIn v)
{
    VsOut o;
    o.sv_pos = mul(vp, float4(v.pos, 1.0));
    o.edge   = v.edge;
    o.color  = v.color;
    return o;
}
