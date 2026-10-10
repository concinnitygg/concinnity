// Reflection-probe sampling, spliced into a shader at its PROBE_COMMON marker
// (see shader_source.rs). The second half of the probe splice: PROBE_TYPES
// declares the records ahead of a shader's bindings, this reads the bound set.
//
// Hooks the including shader must provide, because they differ per binding
// model: `PROBE_SET` names the bound ProbeSet, `PROBE_RECORDS` the bound
// ProbeUniforms buffer, and `probe_cubes` / `probe_cube_sampler` the bound
// probe cube array and its sampler. A shader that sees the camera's cluster
// grid also defines `CLUSTER` (the bound ClusterParams) and `CLUSTER_LIST` (the
// per-cluster lists), which lets it read only the probes binned into a
// fragment's cluster.
// Nothing here may spell either marker -- the splice would reintroduce it.

// The probe cube mip a surface of `roughness` reflects at: the cubes carry
// their own prefilter chain, whatever the environment map's is.
float probe_lod(float roughness)
{
    return roughness * (float(PROBE_SET.mip_count) - 1.0);
}

// The probe cube mip a reflection ray `R` reads at: `lod`, widened by how many
// cube texels the ray sweeps across one pixel, which grows at grazing or
// distant angles. It follows the ray itself rather than the box-projected
// direction, which kinks across a box face.
float probe_footprint_lod(float3 R, float lod)
{
    uint size, height, slices;
    probe_cubes.GetDimensions(size, height, slices);
    float sweep = max(length(ddx(R)), length(ddy(R)));
    return lod + max(log2(sweep * 0.5 * float(size)), 0.0);
}

// Slice `i` of the probe cube array at mip `lod`.
float3 probe_cube_sample(uint i, float3 dir, float lod)
{
    return probe_cubes.SampleLevel(probe_cube_sampler, float4(dir, float(i)), lod).rgb;
}

// The probes a lookup may read: a cluster's two masks at CLUSTER_LIST[`base`],
// or with `base` at PROBE_MASK_ALL every probe of the set.
struct ProbeMask
{
    uint base;
};

static const uint PROBE_MASK_ALL = 0xffffffffu;

// Every live probe.
ProbeMask probe_mask_all()
{
    ProbeMask m;
    m.base = PROBE_MASK_ALL;
    return m;
}

// Word `w` of mask `which` (0 influence, 1 nearest) in `m`'s block: bit `b`
// covers probe `32 * w + b`, and every bit is set past what a cluster mask holds.
uint probe_mask_word(ProbeMask m, uint which, uint w)
{
#ifdef CLUSTER_LIST
    if (m.base != PROBE_MASK_ALL && w < CLUSTER_PROBE_MASK_WORDS)
    {
        return CLUSTER_LIST[m.base + which * CLUSTER_PROBE_MASK_WORDS + w];
    }
#endif
    return 0xffffffffu;
}

// The bits of mask word `w` that name a live probe.
uint probe_live_bits(uint word, uint w)
{
    uint left = PROBE_SET.count - w * 32u;
    return left >= 32u ? word : word & ((1u << left) - 1u);
}

#ifdef CLUSTER_LIST

// The cluster holding a fragment at screen UV `uv` and view depth
// `view_depth`, on the grid the binning kernel built this frame.
uint cluster_at(float2 uv, float view_depth)
{
    uint cx = min(uint(uv.x * float(CLUSTER.grid_x)), CLUSTER.grid_x - 1u);
    uint cy = min(uint(uv.y * float(CLUSTER.grid_y)), CLUSTER.grid_y - 1u);
    float zd = max(view_depth, CLUSTER.cam_pos_znear.w);
    uint cz = min(uint(log(zd / CLUSTER.cam_pos_znear.w) / log(CLUSTER.view_forward_zfar.w / CLUSTER.cam_pos_znear.w)
                       * float(CLUSTER.grid_z)),
                  CLUSTER.grid_z - 1u);
    return cx + cy * CLUSTER.grid_x + cz * CLUSTER.grid_x * CLUSTER.grid_y;
}

// The probes binned into cluster `cid`.
ProbeMask probe_cluster_mask(uint cid)
{
    ProbeMask m;
    m.base = cluster_probe_mask_base(CLUSTER.grid_x * CLUSTER.grid_y * CLUSTER.grid_z, cid);
    return m;
}

// The probes that may matter at world-space point `world_pos`, seen at screen
// UV `uv`: its cluster's on the main camera's view, else the whole set.
ProbeMask probe_mask_at(float2 uv, float3 world_pos)
{
    if (CLUSTER.use_clusters == 0u)
    {
        return probe_mask_all();
    }
    float view_depth = dot(world_pos - CLUSTER.cam_pos_znear.xyz, CLUSTER.view_forward_zfar.xyz);
    return probe_cluster_mask(cluster_at(uv, view_depth));
}

#endif

// Box-parallax sample of probe cube `i`: intersect the world-space reflection
// ray with the probe's influence box and re-anchor the sample direction at that
// hit relative to the capture point, so a static captured cube tracks a moving
// camera. A point outside the box casts from the nearest point of the box, so
// the sample stays continuous across the box surface, where a neighboring
// probe blends in. Falls back to the raw ray when the probe has no baked box
// (box_min.w <= 0.5).
float3 sample_probe_radiance(uint i, ProbeUniforms probe, float3 world_pos, float3 R, float lod)
{
    float3 sample_dir = R;
    if (probe.box_min.w > 0.5)
    {
        float3 p = clamp(world_pos, probe.box_min.xyz, probe.box_max.xyz);
        float3 inv_r = 1.0 / R;
        float3 t_max = (probe.box_max.xyz - p) * inv_r;
        float3 t_min = (probe.box_min.xyz - p) * inv_r;
        float3 t_far = max(t_max, t_min);
        float dist = max(min(min(t_far.x, t_far.y), t_far.z), 0.0);
        sample_dir = p + R * dist - probe.probe_pos.xyz;
    }
    return probe_cube_sample(i, sample_dir, lod);
}

// Blend weight of probe `i` at `world_pos`: 1 deep inside its influence box, 0.5
// on the surface, 0 a margin outside. Each axis's margin scales with the box's
// extent along it, so a wide, shallow box fades across its width as gradually
// as a cube of that width would.
float probe_weight(uint i, float3 world_pos)
{
    float3 c = 0.5 * (PROBE_RECORDS[i].box_min.xyz + PROBE_RECORDS[i].box_max.xyz);
    float3 he = 0.5 * (PROBE_RECORDS[i].box_max.xyz - PROBE_RECORDS[i].box_min.xyz);
    float3 margin = max(PROBE_BLEND_MARGIN * he, (float3)(1e-4));
    // Signed distance to the box surface in margins: positive inside, negative out.
    float3 q = (abs(world_pos - c) - he) / margin;
    float sd = -(length(max(q, (float3)(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0));
    return smoothstep(-1.0, 1.0, sd);
}

// `probe_mask_blend` at mip `level`, the footprint already applied.
bool probe_mask_blend_level(ProbeMask probes, float3 world_pos, float3 R, float level,
                            out float3 radiance)
{
    float3 acc = (float3)(0.0);
    float wsum = 0.0;
    for (uint w = 0u; w * 32u < PROBE_SET.count; w++)
    {
        uint bits = probe_live_bits(probe_mask_word(probes, 0u, w), w);
        while (bits != 0u)
        {
            uint i = w * 32u + firstbitlow(bits);
            bits &= bits - 1u;
            float wt = probe_weight(i, world_pos);
            if (wt > 0.0)
            {
                acc += wt * sample_probe_radiance(i, PROBE_RECORDS[i], world_pos, R, level);
                wsum += wt;
            }
        }
    }
    radiance = wsum > 0.0 ? acc / wsum : (float3)(0.0);
    return wsum > 0.0;
}

// The influence-weighted blend of `probes` at `world_pos` along world-space ray
// `R` (partition of unity): the weight-normalized sum of each covering probe's
// box-projected sample at the roughness mip `lod`, in index order, into
// `radiance`. False, with `radiance` zero, when no probe's influence reaches the
// point.
//
// Uncovered points take one of two fallbacks. The main pass, SSR and RT
// reflections call `probe_mask_specular`, which falls back to the nearest probe
// in the cluster's nearest mask, however far away its box is. The transparent
// passes branch on this instead and fall back to the sky, since a wide surface
// outside every box would otherwise show one capture of somewhere else.
bool probe_mask_blend(ProbeMask probes, float3 world_pos, float3 R, float lod, out float3 radiance)
{
    return probe_mask_blend_level(probes, world_pos, R, probe_footprint_lod(R, lod), radiance);
}

// Probe radiance for `world_pos` along world-space ray `R`: the blend of every
// probe covering the point, else the nearest probe by capture distance.
float3 probe_mask_specular(ProbeMask probes, float3 world_pos, float3 R, float lod)
{
    float level = probe_footprint_lod(R, lod);
    float3 blended;
    if (probe_mask_blend_level(probes, world_pos, R, level, blended))
    {
        return blended;
    }
    float near_d = 1e30;
    uint near_i = 0u;
    for (uint v = 0u; v * 32u < PROBE_SET.count; v++)
    {
        uint bits = probe_live_bits(probe_mask_word(probes, 1u, v), v);
        while (bits != 0u)
        {
            uint i = v * 32u + firstbitlow(bits);
            bits &= bits - 1u;
            float d = distance(world_pos, PROBE_RECORDS[i].probe_pos.xyz);
            if (d < near_d)
            {
                near_d = d;
                near_i = i;
            }
        }
    }
    return sample_probe_radiance(near_i, PROBE_RECORDS[near_i], world_pos, R, level);
}

