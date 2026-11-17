// This frame's G-buffer view normal and motion vectors, kept for the next
// frame's temporal passes (see motion_surface.hlsl). Pairs with
// `fullscreen_vertex` in fullscreen.hlsl. Every read is a point load by pixel
// position; the register numbers are the Metal indices and the D3D root
// signature's slots alike, and Vulkan binds the two textures at 0..1 and their
// samplers at 2..3.

{MOTION_SURFACE}

// binding 0 = the G-buffer (rgb = view normal, a = linear view depth); 1 = the
// motion vectors.
[[vk::binding(0, 0)]] Texture2D<float4> gbuffer : register(t0);
[[vk::binding(2, 0)]] SamplerState gbuffer_samp : register(s0);
[[vk::binding(1, 0)]] Texture2D<float4> velocity : register(t1);
[[vk::binding(3, 0)]] SamplerState velocity_samp : register(s1);

// rg = the octahedral view normal, ba = the motion vector, or
// `MOTION_SURFACE_NONE` in b where there is no surface.
[shader("pixel")]
float4 motion_snapshot_fragment(
    [[vk::location(0)]] float2 vertex_uv : TEXCOORD0,
    float4 pixel : SV_Position) : SV_Target
{
    int3 p = int3(int2(pixel.xy), 0);
    float4 now = gbuffer.Load(p);
    if (now.a <= 0.0)
    {
        return float4(0.0, 0.0, MOTION_SURFACE_NONE, 0.0);
    }
    return float4(motion_surface_encode(normalize(now.xyz)), velocity.Load(p).rg);
}
