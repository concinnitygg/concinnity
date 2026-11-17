// Where this frame shows a surface that last frame's pixels did not: what a
// moving object uncovers, as the reactive mask's value for a temporal
// upscaler. An upscaler that judges its history by depth keeps the object's
// history wherever the two depths agree, as they do where an object meets the
// ground. The mark is full where the motion vectors carry the pixel onto none
// of last frame's surfaces near it (see motion_surface.hlsl), else none.
//
// Pairs with `fullscreen_vertex` in fullscreen.hlsl. Every read is a point
// load by pixel position; the register numbers are the Metal indices and the
// D3D root signature's slots alike, and Vulkan binds the three textures at
// 0..2 and their samplers at 3..5.

{TEXTURE_SIZE}

{MOTION_SURFACE}

// binding 0 = the G-buffer (rgb = view normal, a = linear view depth); 1 = the
// motion vectors; 2 = last frame's motion snapshot.
[[vk::binding(0, 0)]] Texture2D<float4> gbuffer : register(t0);
[[vk::binding(3, 0)]] SamplerState gbuffer_samp : register(s0);
[[vk::binding(1, 0)]] Texture2D<float4> velocity : register(t1);
[[vk::binding(4, 0)]] SamplerState velocity_samp : register(s1);
[[vk::binding(2, 0)]] Texture2D<float4> previous : register(t2);
[[vk::binding(5, 0)]] SamplerState previous_samp : register(s2);

[shader("pixel")]
float4 disocclusion_mark_fragment(
    [[vk::location(0)]] float2 vertex_uv : TEXCOORD0,
    float4 pixel : SV_Position) : SV_Target
{
    int2 p = int2(pixel.xy);
    float4 now = gbuffer.Load(int3(p, 0));
    if (now.a <= 0.0)
    {
        return (float4)(0.0);
    }
    int2 size = int2(texture_size(gbuffer));
    float2 v = velocity.Load(int3(p, 0)).rg;
    float2 prev_uv = (float2(p) + 0.5) / float2(size) + v;
    // Off screen last frame, the reader has no history here to keep.
    if (any(prev_uv < 0.0) || any(prev_uv > 1.0))
    {
        return (float4)(0.0);
    }
    // The 3x3 around where the pixel was absorbs the jitter's sub-pixel shift
    // of every edge, so only an uncovered area wider than that is marked: an
    // upscaler's own depth test covers the rest.
    float3 normal = normalize(now.xyz);
    float2 motion = v * float2(size);
    int2 c = int2(prev_uv * float2(size));
    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            int2 q = clamp(c + int2(x, y), 0, size - 1);
            float4 last = previous.Load(int3(q, 0));
            if (motion_surface_match(last, normal, motion, float2(size)) >= 1.0)
            {
                return (float4)(0.0);
            }
        }
    }
    return float4(1.0, 0.0, 0.0, 0.0);
}
