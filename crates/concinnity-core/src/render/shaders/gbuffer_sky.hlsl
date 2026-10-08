// The sky's share of the G-buffer pre-pass, drawn over whatever the surface
// entries in `main_bindless.hlsl` left uncovered: a fullscreen triangle at the
// far plane, depth-tested without writing.
//
// CN_BACKEND_DIRECTX pins the view block to b1, the root signature in
// directx/post/gbuffer_sky.rs.

{DEPTH_CONVENTION}
{SKY_RAY}
{GBUFFER_COMMON}

#ifdef CN_BACKEND_DIRECTX
ConstantBuffer<GbView> gb_view : register(b1);
#else
[[vk::binding(0, 0)]] ConstantBuffer<GbView> gb_view : register(b0);
#endif

struct SkyVertexOut
{
    float4 position : SV_Position;
    [[vk::location(0)]] float4 cur_clip  : TEXCOORD0;
    [[vk::location(1)]] float4 prev_clip : TEXCOORD1;
};

// The sky reads as "no geometry" to every screen-space pass (view depth 0, the
// background's roughness) while still writing its motion: the view ray through
// the jittered projection, reprojected through this frame's and the previous
// frame's unjittered matrices. A direction has no position, so the camera's
// translation drops out and the motion is its rotation alone.
[shader("vertex")]
SkyVertexOut gbuffer_sky_vertex(uint vid : SV_VertexID)
{
    float2 ndc = sky_corner(vid);
    float3 ray = sky_ray(gb_view.jittered_vp, ndc);
    SkyVertexOut o;
    o.position  = float4(ndc, DEPTH_FAR, 1.0);
    o.cur_clip  = mul(gb_view.cur_vp, float4(ray, 0.0));
    o.prev_clip = mul(gb_view.prev_vp, float4(ray, 0.0));
    return o;
}

[shader("pixel")]
GbFragmentOut gbuffer_sky_fragment(SkyVertexOut p)
{
    GbFragmentOut o;
    o.nd    = (float4)(0.0);
    o.rough = 1.0;
    o.vel   = gb_motion(p.cur_clip, p.prev_clip);
    return o;
}
