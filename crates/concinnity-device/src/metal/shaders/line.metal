#include <metal_stdlib>
using namespace metal;

// World-space line geometry. The CPU expanded each line into a camera-facing
// ribbon whose corners already sit off the line centre, so the vertex stage
// only applies the camera VP.
//
// Alpha does the two things a line needs: `edge` fades the outer pixel of the
// ribbon so the line antialiases without MSAA, and the vertex colour's own
// alpha carries the CPU-side distance fade.

struct LineView {
    float4x4 vp;
    // Alpha multiplier for the part of a line behind scene geometry. 0 hides
    // occluded lines outright; a small value leaves a hint of the axis showing
    // through, which is what makes it useful for orienting in a dense scene.
    float occluded_alpha;
    float pad0;
    float pad1;
    float pad2;
};

struct LineVtxIn {
    float3 pos   [[attribute(0)]];
    float  edge  [[attribute(1)]];
    float4 color [[attribute(2)]];
};

struct LineVtxOut {
    float4 position [[position]];
    float  edge;
    float4 color;
};

vertex LineVtxOut line_vertex(
    LineVtxIn in [[stage_in]],
    constant LineView& view [[buffer(0)]]
) {
    LineVtxOut out;
    out.position = view.vp * float4(in.pos, 1.0);
    out.edge = in.edge;
    out.color = in.color;
    return out;
}

fragment float4 line_fragment(
    LineVtxOut in [[stage_in]],
    constant LineView& view [[buffer(0)]],
    depth2d<float, access::read> scene_depth [[texture(0)]]
) {
    // Ribbon edge fade: `edge` runs -1..1 across the width, so one pixel of
    // its own screen-space derivative is the softest edge that still reads as
    // a solid line.
    float aa = max(fwidth(in.edge), 1e-4);
    float coverage = 1.0 - smoothstep(1.0 - aa, 1.0, abs(in.edge));

    // Manual depth test against the resolved scene depth: this pass runs after
    // the main pass resolved, so it has no depth attachment of its own. The
    // bias keeps a line drawn exactly on a surface (a ground-plane axis) from
    // z-fighting itself into a dashed mess.
    float scene_z = scene_depth.read(uint2(in.position.xy));
    float occlusion = (in.position.z > scene_z + 1e-6) ? view.occluded_alpha : 1.0;

    float alpha = in.color.a * coverage * occlusion;
    if (alpha <= 0.002) {
        discard_fragment();
    }
    return float4(in.color.rgb, alpha);
}
