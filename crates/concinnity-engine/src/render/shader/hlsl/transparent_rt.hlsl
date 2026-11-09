// The ray-tracing scene resources every transparent-pass producer (glass.hlsl,
// glass_mesh.hlsl, water.hlsl) binds when it traces, spliced into each at its
// TRANSPARENT_RT marker ahead of RT_TRACE, which reads them. Not a standalone
// program. Nothing here may spell the marker itself.
//
// One set of slots serves all three, so the inputs the transparent encoder
// binds once are valid for each. On Metal they ride the pass's otherwise-free
// fragment buffers (0..4 and 8..10, since 5/6/7 are the view, the params and
// the probe set); on Vulkan they are a set of their own, past the view /
// params / global sets; on DirectX they follow the registers the pass already
// occupies.

#ifdef CN_BACKEND_DIRECTX

ConstantBuffer<RtParams> rt_params : register(b5);
RaytracingAccelerationStructure scene_tlas : register(t4);
ByteAddressBuffer verts : register(t5);
ByteAddressBuffer indices : register(t6);
ByteAddressBuffer sverts : register(t8);
ByteAddressBuffer sidx : register(t9);
StructuredBuffer<RtGeomEntry> geom : register(t10);

#else

[[vk::binding(0, 3)]] ConstantBuffer<RtParams> rt_params : register(b0);
[[vk::binding(1, 3)]] RaytracingAccelerationStructure scene_tlas : register(t4);
[[vk::binding(2, 3)]] StructuredBuffer<RtGeomEntry> geom : register(t3);
[[vk::binding(3, 3)]] StructuredBuffer<float> verts : register(t1);
[[vk::binding(4, 3)]] StructuredBuffer<uint> indices : register(t2);
[[vk::binding(5, 3)]] StructuredBuffer<float> sverts : register(t8);
[[vk::binding(6, 3)]] StructuredBuffer<uint> sidx : register(t9);

#endif

#ifdef RT_TEXTURED
// The bindless albedo / normal / emissive pool. Metal keeps its argument buffer
// at buffer(10): buffer(7), where the main pass puts it, is the probe set in
// the transparent pass. The host binds the main pass's argument buffer there at
// the pool's offset, so the unsized array reads the pool and nothing before it.
#if defined(CN_BACKEND_METAL)
[[vk::binding(0, 6)]] [[cn::metal_argument_buffer(10)]]
Texture2D<float4> tex_pool[] : register(t0, space6);
[[vk::binding(12, 0)]] SamplerState pool_sampler : register(s4);
#elif defined(CN_BACKEND_DIRECTX)
Texture2D<float4> tex_pool[] : register(t0, space1);
SamplerState pool_sampler : register(s1);
#else
[[vk::binding(1, 4)]] Texture2D<float4> tex_pool[] : register(t0, space1);
[[vk::binding(20, 2)]] SamplerState pool_sampler : register(s2);
#endif
#endif
