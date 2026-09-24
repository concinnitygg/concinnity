// The reflection-probe set every transparent-pass producer (glass.hlsl,
// glass_mesh.hlsl, water.hlsl) reads, spliced into each at its
// TRANSPARENT_PROBES marker, with the probe sampling and the environment
// fallback the three share. Not a standalone program. Nothing here may spell
// the marker itself.
//
// One set of slots serves all three, so the transparent encoder binds it once
// for any of them, together with the main camera's cluster grid that bins the
// probes. On Vulkan it is the forward global set's own bindings, its cube
// sampler included, bound as set 2; on Metal the cube array takes texture(6)
// beside the pass's other textures, the records buffer(11) and the cluster
// params and lists buffer(12) and (13); on DirectX the registers sit clear of
// the ray-tracing SRVs at t4..t12, so the base and ray-traced variants share
// one layout.

#ifdef CN_BACKEND_DIRECTX
ConstantBuffer<ProbeSet> probe_set : register(b4);
ConstantBuffer<ClusterParams> cluster : register(b6);
TextureCubeArray<float4> probe_cubes : register(t20);
StructuredBuffer<ProbeUniforms> probe_records : register(t21);
StructuredBuffer<uint> cluster_list : register(t22);
#else
[[vk::binding(7, 2)]] ConstantBuffer<ProbeSet> probe_set : register(b7);
[[vk::binding(8, 2)]] TextureCubeArray<float4> probe_cubes : register(t6);
[[vk::binding(19, 2)]] SamplerState probe_cube_sampler : register(s2);
[[vk::binding(17, 2)]] StructuredBuffer<ProbeUniforms> probe_records : register(t11);
[[vk::binding(10, 2)]] ConstantBuffer<ClusterParams> cluster : register(b12);
[[vk::binding(11, 2)]] StructuredBuffer<uint> cluster_list : register(t13);
#endif
#define PROBE_SET probe_set
#define PROBE_RECORDS probe_records
#define CLUSTER cluster
#define CLUSTER_LIST cluster_list

// Screen UV of world-space point `p` in this view. Resolution independent, so
// the reduced reflection pre-pass places a surface where the full-resolution
// draw does.
float2 transparent_screen_uv(float3 p)
{
    float4 clip = mul(view.vp, float4(p, 1.0));
    float2 ndc = clip.xy / clip.w;
    return float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

{PROBE_COMMON}

// The reflection a transparent surface of `roughness` at `world_pos` falls back
// to along `r`, seen at render-target pixel position `pixel`: the probes
// covering the point, else the sky prefilter cube where an EnvironmentMap is
// bound, else `bare`. Each cube is read at its own roughness-keyed mip.
float3 transparent_environment(float3 world_pos, float3 r, float roughness, float3 bare,
                               float2 pixel)
{
    if (PROBE_SET.count > 0u)
    {
        float3 radiance;
        ProbeMask probes = probe_mask_at(transparent_screen_uv(world_pos), world_pos, pixel);
        if (probe_mask_blend(probes, world_pos, r, probe_lod(roughness), radiance))
        {
            return radiance;
        }
    }
    if (view.prefilter_mip_count > 0.5)
    {
        return prefilter_level(r, roughness * (view.prefilter_mip_count - 1.0));
    }
    return bare;
}
