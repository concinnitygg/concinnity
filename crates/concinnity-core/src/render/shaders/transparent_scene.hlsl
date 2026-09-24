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

float4 scene_sample(float2 uv) { return scene_color.Sample(scene_color_sampler, uv); }
float3 prefilter_level(float3 dir, float lod)
{
    return prefilter_cube.SampleLevel(prefilter_cube_sampler, SKY_DIR(dir), lod).rgb;
}
