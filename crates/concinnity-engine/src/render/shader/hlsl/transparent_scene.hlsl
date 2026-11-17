// The scene inputs every transparent-pass producer (glass.hlsl,
// glass_mesh.hlsl, water.hlsl) binds, spliced into each at its
// TRANSPARENT_SCENE marker after TRANSPARENT_TYPES. Not a standalone program.
// Nothing here may spell the marker itself.
//
// The including shader defines `TRANSPARENT_PARAMS` as its per-draw params
// struct before the marker.
//
// USE_MSAA is a HOST difference rather than a target one: Vulkan and DirectX
// read the multisampled main depth, Metal the resolved copy. The two arms are
// different HLSL overloads (`Texture2DMS<float>::Load(int2, int)` against
// `Texture2D<float>::Load(int3)`), not a compiler workaround.
//
// Every slot is shared by the three producers: on DirectX one root signature
// per path serves them all, on Vulkan one set of descriptor set layouts, and on
// Metal the encoder binds the shared inputs once for the whole pass. A slot may
// not move for one producer alone.
//
// The air in front of what a producer draws (`transparent_haze`) reads the
// atmosphere's constant block, its transmittance and multiple-scattering
// tables and its aerial volume, as the opaque scene's haze pass does; they
// hold stand-ins, and `view.ambient.z` is 0, in a world without one.

{ATMOSPHERE_TYPES}

#ifdef CN_BACKEND_DIRECTX

ConstantBuffer<TransparentView> view : register(b0);
ConstantBuffer<TRANSPARENT_PARAMS> params : register(b1);
Texture2D<float4> scene_color : register(t0);
#if USE_MSAA
Texture2DMS<float> scene_depth : register(t1);
#else
Texture2D<float> scene_depth : register(t1);
#endif
TextureCube<float4> prefilter_cube : register(t2);
// Two static samplers from the root signature, not one per source, so the
// per-source names the other branch declares are aliases here and a sample site
// reads the same in both.
SamplerState post_samp : register(s0);
SamplerState cube_sampler : register(s2);
#define scene_color_sampler post_samp
#define prefilter_cube_sampler cube_sampler
#define probe_cube_sampler cube_sampler

#else

// Metal buffer(5) / buffer(6): the shared per-frame view and the per-draw
// params, both written with setBytes by the transparent encoder.
[[vk::binding(0, 0)]] ConstantBuffer<TransparentView> view : register(b5);
[[vk::binding(0, 1)]] ConstantBuffer<TRANSPARENT_PARAMS> params : register(b6);

// The shared Metal texture slots: the scene snapshot at 0, the resolved depth
// at 1, the sky prefilter cube at 2, a producer's planar resolve at 3, and the
// reduced reflection layers at 4 and 5. Vulkan's view set carries the
// snapshot's sampler at 5; the cube and its sampler are the global set's.
[[vk::binding(1, 0)]] Texture2D<float4> scene_color : register(t0);
[[vk::binding(5, 0)]] SamplerState scene_color_sampler : register(s0);
#if USE_MSAA
[[vk::binding(2, 0)]] Texture2DMS<float> scene_depth : register(t1);
#else
[[vk::binding(2, 0)]] Texture2D<float> scene_depth : register(t1);
#endif
[[vk::binding(5, 2)]] TextureCube<float4> prefilter_cube : register(t2);
#ifdef CN_BACKEND_VULKAN
// The global set's cube sampler, which TRANSPARENT_PROBES reads the probe cubes
// through as well.
[[vk::binding(19, 2)]] SamplerState prefilter_cube_sampler : register(s1);
#else
[[vk::binding(6, 0)]] SamplerState prefilter_cube_sampler : register(s1);
#endif

#endif

#ifdef CN_BACKEND_DIRECTX
ConstantBuffer<AtmosphereParams> atmo_cb : register(b2);
Texture2D<float4> transmittance_lut : register(t23);
Texture2D<float4> multiscatter_lut : register(t24);
Texture3D<float4> aerial_volume : register(t26);
SamplerState lut_sampler : register(s5);
#else
[[vk::binding(7, 0)]] ConstantBuffer<AtmosphereParams> atmo_cb : register(b15);
[[vk::binding(8, 0)]] Texture2D<float4> transmittance_lut : register(t14);
[[vk::binding(9, 0)]] Texture2D<float4> multiscatter_lut : register(t15);
[[vk::binding(10, 0)]] Texture3D<float4> aerial_volume : register(t16);
[[vk::binding(11, 0)]] SamplerState lut_sampler : register(s7);
#endif

#define AIR_HAZE
#define ATMO atmo_cb
{ATMOSPHERE_COMMON}
{AIR_HAZE}

// The HDR chain's pre-exposure: the snapshot is divided by it so producers
// shade in absolute radiance, and what they write is multiplied by it.
float pre_exposure() { return view.camera_pos.w; }

float4 scene_sample(float2 uv)
{
    float4 c = scene_color.Sample(scene_color_sampler, uv);
    return float4(c.rgb / pre_exposure(), c.a);
}

// A producer's straight-alpha color, in absolute radiance, and its reactive
// share, as the pass writes them.
TransparentOut transparent_out(float4 color, float reactive)
{
    TransparentOut o;
    o.color = float4(hdr_store_cap(color.rgb * pre_exposure()), color.a);
    o.reactive = reactive_write(reactive);
    return o;
}
float3 prefilter_level(float3 dir, float lod)
{
    return prefilter_cube.SampleLevel(prefilter_cube_sampler, SKY_DIR(dir), lod).rgb * SKY_CUBE_READ;
}

// The air between the camera and `world_pos`, seen under the fragment at
// `pixel`: `(in-scatter, 1 - T)`, the in-scatter absolute. None in a world
// without an atmosphere.
float4 transparent_air(float3 world_pos, float2 pixel)
{
    if (view.ambient.z < 0.5)
    {
        return (float4)0.0;
    }
    float3 to_point = world_pos - ATMO.cam_pos.xyz;
    float4 haze = aerial_haze(pixel * ATMO.viewport.zw, normalize(to_point),
                              length(to_point) * ATMO.radii.w);
    return float4(haze.rgb / ATMO.cam_pos.w, haze.a);
}

// `color` at a point seen through `air` (`transparent_air`).
float3 transparent_haze(float3 color, float4 air)
{
    return color * (1.0 - air.a) + air.rgb;
}

// The scene behind a producer, `seen` through `air` already, as it stands at
// the producer before that air: what the producer's own light mixes with, so
// hazing the mix hazes it once.
float3 transparent_unhaze(float3 seen, float4 air)
{
    return max(seen - air.rgb, (float3)0.0) / max(1.0 - air.a, 1e-4);
}
