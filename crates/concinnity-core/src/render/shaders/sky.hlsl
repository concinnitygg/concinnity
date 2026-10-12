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
//
// Under SKY_ATMOSPHERE the sky is the world's atmosphere instead: the
// sky-view table around the camera, with the sun's disc drawn into it and
// any night sky behind it, dimmed by the air (the table's alpha). Its
// constant block, the table and its sampler take set 1 on Vulkan and slot 1,
// 0 and 0 on Metal and DirectX, beside the same view block.

{DEPTH_CONVENTION}
{VIEW_UNIFORMS}
{SKY_RAY}

#ifdef SKY_ATMOSPHERE
{ATMOSPHERE_TYPES}
[[vk::binding(0, 0)]] ConstantBuffer<ViewUniforms> view_cb : register(b0);
[[vk::binding(0, 1)]] ConstantBuffer<AtmosphereParams> atmo_cb : register(b1);
[[vk::binding(1, 1)]] Texture2D<float4> sky_view_lut : register(t0);
[[vk::binding(2, 1)]] SamplerState lut_sampler : register(s0);
#define ATMO atmo_cb
{ATMOSPHERE_COMMON}
{NIGHT_SKY}
#elif defined(CN_BACKEND_VULKAN)
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

#ifdef SKY_ATMOSPHERE
// The sky-view table along the pixel's ray over any night sky behind it, and
// the sun's disc where the ray meets it above the horizon, darkened toward
// its limb.
[shader("pixel")]
float4 sky_atmosphere_fragment(SkyVertexOut p) : SV_Target
{
    float3 dir = normalize(p.ray);
    float4 view = sky_view_lut.SampleLevel(lut_sampler, sky_view_uv_of(dir), 0.0);
    // The table holds the frame's pre-exposure; a probe or mirror face draws
    // at its own.
    float3 color = view.rgb * (VIEW.pre_exposure / ATMO.cam_pos.w);
    float3 dir_dx = ddx(dir);
    float3 dir_dy = ddy(dir);
    float3 beyond = (float3)0.0;
    if (night_sky_on())
    {
        beyond += night_sky_radiance(dir, dir_dx, dir_dy, view.a);
    }
    float cos_sun = dot(dir, ATMO.sun_direction.xyz);
    float cos_radius = ATMO.sun_direction.w;
    if (cos_sun > cos_radius && !atmo_view_hits_ground(dot(dir, atmo_camera_up())))

    {
        float x = saturate((1.0 - cos_sun) / max(1.0 - cos_radius, 1e-7));
        float limb = 1.0 - 0.6 * (1.0 - sqrt(1.0 - x));
        beyond += ATMO.sun_disc.xyz * limb;
    }
    color += beyond * VIEW.pre_exposure;
    // Kept inside the half-float range the scene is stored in: the disc's
    // radiance is not, at any exposure that shows the sky around it.
    return float4(min(color, (float3)60000.0), 1.0);
}
#else
[shader("pixel")]
float4 sky_fragment(SkyVertexOut p) : SV_Target
{
    float3 dir = normalize(p.ray);
    float3 sky = prefilter_cube.SampleLevel(cube_sampler, SKY_DIR(dir), 0.0).rgb;
    return float4(sky * VIEW.pre_exposure, 1.0);
}
#endif
