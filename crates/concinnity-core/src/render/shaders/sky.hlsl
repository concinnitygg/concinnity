// The environment drawn as the background: single source for every backend.
//
// Drawn last in each pass that renders the opaque scene from one viewpoint (the
// main camera, a reflection-probe face, a planar mirror), into that pass's own
// color and depth. The triangle sits at the far plane and is depth-tested
// inclusively without writing, so it lands exactly where no surface did, per
// sample under MSAA. Each pixel takes the environment's sharpest mip along its
// own view ray, turned by the sky rotation like every other environment tap.
//
// It reads the view block the pass it completes already binds, so a probe face
// or a mirror sees the sky from its own viewpoint. Vulkan reads that block, the
// prefilter cube and the cube sampler straight out of the main pass's global
// set (bindings 0, 5 and 19); Metal and DirectX bind the three at slot 0 of
// their own register spaces.

{DEPTH_CONVENTION}
{VIEW_UNIFORMS}
{SKY_RAY}

#ifdef CN_BACKEND_VULKAN
[[vk::binding(0, 0)]] ConstantBuffer<ViewUniforms> view_cb : register(b0);
[[vk::binding(5, 0)]] TextureCube<float4> prefilter_cube : register(t0);
[[vk::binding(19, 0)]] SamplerState cube_sampler : register(s0);
#else
[[vk::binding(0, 0)]] ConstantBuffer<ViewUniforms> view_cb : register(b0);
[[vk::binding(1, 0)]] TextureCube<float4> prefilter_cube : register(t0);
[[vk::binding(2, 0)]] SamplerState cube_sampler : register(s0);
#endif
#define VIEW view_cb

struct SkyVertexOut
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 ray : TEXCOORD0;
};

[shader("vertex")]
SkyVertexOut sky_vertex(uint vid : SV_VertexID)
{
    float2 ndc = sky_corner(vid);
    SkyVertexOut o;
    o.position = float4(ndc, DEPTH_FAR, 1.0);
    o.ray = sky_ray(VIEW.vp, ndc);
    return o;
}

[shader("pixel")]
float4 sky_fragment(SkyVertexOut p) : SV_Target
{
    float3 dir = normalize(p.ray);
    return float4(prefilter_cube.SampleLevel(cube_sampler, SKY_DIR(dir), 0.0).rgb, 1.0);
}
