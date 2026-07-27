#pragma pack_matrix(column_major)

// World-space line pass - fragment shader. Mirrors line_fragment in
// src/metal/shaders/line.metal.
//
// Alpha does the two things a line needs: `edge` fades the outer pixel of the
// ribbon so the line antialiases without MSAA, and the vertex colour's own
// alpha carries the CPU-side distance fade.
//
// `USE_MSAA` is defined by the host (1 when the main pass uses MSAA, 0
// otherwise) so the depth SRV declaration and the load call match the
// underlying resource.

cbuffer LineView : register(b0)
{
    float4x4 vp;
    // Alpha multiplier for the part of a line behind scene geometry. 0 hides
    // occluded lines outright; a small value leaves a hint of the line showing
    // through, which is what makes it useful for orienting in a dense scene.
    float    occluded_alpha;
    float    _pad0;
    float    _pad1;
    float    _pad2;
}

#if USE_MSAA
Texture2DMS<float> scene_depth : register(t0);
#else
Texture2D<float>   scene_depth : register(t0);
#endif

struct VsOut
{
    float4 sv_pos : SV_POSITION;
    float  edge   : TEXCOORD0;
    float4 color  : COLOR;
};

float4 main(VsOut p) : SV_TARGET
{
    // Ribbon edge fade: `edge` runs -1..1 across the width, so one pixel of
    // its own screen-space derivative is the softest edge that still reads as
    // a solid line.
    float aa = max(fwidth(p.edge), 1e-4);
    float coverage = 1.0 - smoothstep(1.0 - aa, 1.0, abs(p.edge));

    // Manual depth test against the scene depth: this pass runs after the main
    // pass and has no depth attachment of its own. The bias keeps a line drawn
    // exactly on a surface (a ground-plane axis) from z-fighting itself into a
    // dashed mess.
    int2 pixel = int2(p.sv_pos.xy);
#if USE_MSAA
    float scene_z = scene_depth.Load(pixel, 0);
#else
    float scene_z = scene_depth.Load(int3(pixel, 0));
#endif
    float occlusion = (p.sv_pos.z > scene_z + 1e-6) ? occluded_alpha : 1.0;

    float alpha = p.color.a * coverage * occlusion;
    if (alpha <= 0.002)
    {
        discard;
    }
    return float4(p.color.rgb, alpha);
}
