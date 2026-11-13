// What a reflection ray sees past everything traced or on screen, shared by
// every pass that reflects: the ray-traced resolve, glass and water. Spliced
// after PROBE_COMMON, whose probe set it reads, into a shader that defines
// `prefilter_level(dir, lod)` over the sky's prefiltered cube.

// The environment a ray from `pos` along `dir` reflects: the local reflection
// probes the camera's cluster grid bins at screen `uv`, fading far outside
// every probe box to the sky's prefiltered cube when one is bound (`mips`
// levels, 0 for none), else `bare`. Each cube is read at the mip `roughness`
// keys.
float3 env_along(float2 uv, float3 pos, float3 dir, float roughness, float mips, float3 bare)
{
    float3 probe = (float3)(0.0);
    float sky_share = 1.0;
    if (PROBE_SET.count > 0u)
    {
        probe = probe_mask_specular(probe_mask_at(uv, pos), pos, dir, probe_lod(roughness), sky_share);
        if (sky_share <= 0.0)
        {
            return probe;
        }
    }
    float3 sky = mips > 0.5 ? prefilter_level(dir, roughness * (mips - 1.0)) : bare;
    return probe + sky * sky_share;
}
