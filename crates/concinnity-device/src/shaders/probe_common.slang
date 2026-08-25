// Reflection-probe sampling, spliced into a shader at its PROBE_COMMON marker
// (see slang_source.rs). The second half of the probe splice: PROBE_TYPES
// declares the records ahead of a shader's bindings, this reads the bound set.
//
// Two hooks the including shader must provide, because both differ per binding
// model: `PROBE_SET` names the bound ProbeSet, and `probe_cube_sample_bias`
// samples probe cube `i` (the cube array is combined on Metal and Vulkan,
// split into an array plus one sampler on DXIL). Nothing here may spell either
// marker -- the splice would reintroduce it.

// Box-parallax sample of probe cube `i`: intersect the world-space reflection
// ray with the probe's influence box and re-anchor the sample direction at that
// hit relative to the capture point, so a static captured cube tracks a moving
// camera. Falls back to the raw ray when the probe has no baked box
// (box_min.w <= 0.5) or the box does not lie ahead of the ray. `lod` rides as
// the texture bias, the same semantics the prefilter-cube tap uses.
float3 sample_probe_radiance(uint i, ProbeUniforms probe, float3 world_pos, float3 R, float lod)
{
    float3 sample_dir = R;
    if (probe.box_min.w > 0.5)
    {
        float3 inv_r = 1.0 / R;
        float3 t_max = (probe.box_max.xyz - world_pos) * inv_r;
        float3 t_min = (probe.box_min.xyz - world_pos) * inv_r;
        float3 t_far = max(t_max, t_min);
        float dist = min(min(t_far.x, t_far.y), t_far.z);
        if (dist > 0.0)
        {
            float3 hit = world_pos + R * dist;
            sample_dir = hit - probe.probe_pos.xyz;
        }
    }
    return probe_cube_sample_bias(i, sample_dir, lod);
}

// Blend weight of probe `i` at `world_pos`: 1 deep inside its influence box, 0.5
// on the surface, 0 a margin outside. The margin scales with the box, so a small
// probe fades over a short distance and a room-sized one over a longer one.
float probe_weight(uint i, float3 world_pos)
{
    float3 c = 0.5 * (PROBE_SET.probes[i].box_min.xyz + PROBE_SET.probes[i].box_max.xyz);
    float3 he = 0.5 * (PROBE_SET.probes[i].box_max.xyz - PROBE_SET.probes[i].box_min.xyz);
    // Signed distance to the box surface: positive inside, negative out.
    float3 q = abs(world_pos - c) - he;
    float sd = -(length(max(q, float3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0));
    float margin = max(PROBE_BLEND_MARGIN * min(he.x, min(he.y, he.z)), 1e-4);
    return smoothstep(-margin, margin, sd);
}

// Whether any probe's influence reaches `world_pos`.
//
// A caller that has a sky tier below the probe set should consult this rather
// than reading `PROBE_SET.count` alone: `probe_set_specular`'s no-coverage
// fallback is the nearest probe by capture distance, with no distance or
// box-size limit, so for a surface far outside every box it returns a capture of
// somewhere else entirely. A wide pool with the world's only probe inside a crate
// floating over it is the case that found this. The sky is the better answer
// there, and this is what lets the caller pick it.
bool probe_set_covers(float3 world_pos)
{
    for (uint i = 0u; i < PROBE_SET.count; i++)
    {
        if (probe_weight(i, world_pos) > 0.0)
        {
            return true;
        }
    }
    return false;
}

// Probe radiance for `world_pos` along world-space ray `R`, blended across
// every probe covering the point (partition of unity); the result is the
// weight-normalised sum of each probe's box-projected sample. Where no box
// covers, falls back to the nearest probe by capture distance -- see
// `probe_set_covers` for when a caller should avoid asking at all.
float3 probe_set_specular(float3 world_pos, float3 R, float lod)
{
    float3 acc = float3(0.0);
    float wsum = 0.0;
    float near_d = 1e30;
    uint near_i = 0u;
    for (uint i = 0u; i < PROBE_SET.count; i++)
    {
        float w = probe_weight(i, world_pos);
        if (w > 0.0)
        {
            acc += w * sample_probe_radiance(i, PROBE_SET.probes[i], world_pos, R, lod);
            wsum += w;
        }
        float d = distance(world_pos, PROBE_SET.probes[i].probe_pos.xyz);
        if (d < near_d)
        {
            near_d = d;
            near_i = i;
        }
    }
    if (wsum > 0.0)
    {
        return acc / wsum;
    }
    return sample_probe_radiance(near_i, PROBE_SET.probes[near_i], world_pos, R, lod);
}
