// World-space line pass: single source for every backend.
//
// The CPU expanded each line into a camera-facing ribbon whose corners already
// sit off the line centre (gfx::lines::build_vertices), so the vertex stage
// only applies the camera VP.
//
// Alpha does the two things a line needs: `edge` fades the outer pixel of the
// ribbon so the line antialiases without MSAA, and the vertex colour's own
// alpha carries the CPU-side distance fade. The pass binds no depth attachment
// -- it runs after the main pass resolved -- so the fragment tests the resolved
// scene depth itself, which is what lets an occluded line fade instead of
// vanishing.
//
// USE_MSAA is a HOST difference rather than a target one: Vulkan reads the
// multisampled main depth while Metal reads the resolved copy.

#ifndef USE_MSAA
#define USE_MSAA 0
#endif

// Per-frame view inputs, 80 B. Mirrors `LineView` in each backend's uniforms
// module.
struct LineView
{
    // The main pass's view-projection (jittered when TAA is on), so a line
    // lands on the same pixel the geometry it sits on did.
    float4x4 vp;
    // Alpha multiplier for the part of a line behind scene geometry. 0 hides
    // occluded lines outright; a small value leaves a hint of the line showing
    // through, which is what makes it useful for orienting in a dense scene.
    float occluded_alpha;
    float _pad0;
    float _pad1;
    float _pad2;
};

[[vk::binding(0, 0)]] ConstantBuffer<LineView> view : register(b0);

#if USE_MSAA
[[vk::binding(1, 0)]] Texture2DMS<float> scene_depth;
#else
[[vk::binding(1, 0)]] Texture2D<float> scene_depth;
#endif

// Matches `LineVertex` (32 B: pos at 0, edge at 12, colour at 16), which the
// host input layouts declare and `line_vertex_layout_matches_shaders` pins.
struct LineVertexIn
{
    [[vk::location(0)]] float3 pos : POSITION;
    [[vk::location(1)]] float edge : TEXCOORD0;
    [[vk::location(2)]] float4 color : COLOR;
};

// The varyings lead deliberately: D3D packs a stage signature in declaration
// order and links the two stages by matching semantic *and* register.
struct LineVertexOut
{
    [[vk::location(0)]] float edge : TEXCOORD0;
    [[vk::location(1)]] float4 color : TEXCOORD1;
    float4 position : SV_Position;
};

[shader("vertex")]
LineVertexOut line_vertex(LineVertexIn v)
{
    LineVertexOut o;
    o.position = mul(view.vp, float4(v.pos, 1.0));
    o.edge = v.edge;
    o.color = v.color;
    return o;
}

float line_scene_depth(int2 pixel)
{
#if USE_MSAA
    return scene_depth.Load(pixel, 0);
#else
    return scene_depth.Load(int3(pixel, 0));
#endif
}

[shader("fragment")]
float4 line_fragment(LineVertexOut i) : SV_Target
{
    // Ribbon edge fade: `edge` runs -1..1 across the width, so one pixel of its
    // own screen-space derivative is the softest edge that still reads as a
    // solid line.
    float aa = max(fwidth(i.edge), 1e-4);
    float coverage = 1.0 - smoothstep(1.0 - aa, 1.0, abs(i.edge));

    // Manual depth test against the scene depth. The bias keeps a line drawn
    // exactly on a surface (a ground-plane axis) from z-fighting itself into a
    // dashed mess.
    float scene_z = line_scene_depth(int2(i.position.xy));
    float occlusion = (i.position.z > scene_z + 1e-6) ? view.occluded_alpha : 1.0;

    float alpha = i.color.a * coverage * occlusion;
    if (alpha <= 0.002)
    {
        discard;
    }
    return float4(i.color.rgb, alpha);
}
