// What a reflection ray sees past everything traced or on screen, shared by
// every pass that reflects: the ray-traced resolve, glass and water. Spliced
// after PROBE_COMMON, whose probe set it reads, into a shader that defines
// `prefilter_level(dir, lod)` over the sky's prefiltered cube.

// What the local reflection probes the camera's cluster grid bins at screen
// `uv` reflect from `pos` along `dir` at `roughness`, already weighted, and in
// `sky_share` the share they leave to the sky: all of it where no probe
// covers the point.
float3 env_probes(float2 uv, float3 pos, float3 dir, float roughness, out float sky_share)
{
    sky_share = 1.0;
    if (PROBE_SET.count == 0u)
    {
        return (float3)(0.0);
    }
    return probe_mask_specular(probe_mask_at(uv, pos), pos, dir, probe_lod(roughness), sky_share);
}

// The sky along `dir` at `roughness`: the prefiltered cube when one is bound
// (`mips` levels, 0 for none), else `bare`.
float3 env_sky(float3 dir, float roughness, float mips, float3 bare)
{
    return mips > 0.5 ? prefilter_level(dir, roughness * (mips - 1.0)) : bare;
}

// The environment a ray from `pos` along `dir` reflects: the local reflection
// probes the camera's cluster grid bins at screen `uv`, fading far outside
// every probe box to the sky's prefiltered cube when one is bound (`mips`
// levels, 0 for none), else `bare`. Each cube is read at the mip `roughness`
// keys.
float3 env_along(float2 uv, float3 pos, float3 dir, float roughness, float mips, float3 bare)
{
    float sky_share;
    float3 probe = env_probes(uv, pos, dir, roughness, sky_share);
    if (sky_share <= 0.0)
    {
        return probe;
    }
    return probe + env_sky(dir, roughness, mips, bare) * sky_share;
}
