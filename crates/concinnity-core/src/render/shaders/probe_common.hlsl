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

// Slice `i` of the probe cube array at an LOD bias.
float3 probe_cube_sample_bias(uint i, float3 dir, float lod)
{
    return probe_cubes.SampleBias(probe_cube_sampler, float4(dir, float(i)), lod).rgb;
}

// The probes a lookup may read: a cluster's two masks at CLUSTER_LIST[`base`],
// or with `base` at PROBE_MASK_ALL every probe of the set. `lane` is the
// fragment's pixel position within its 2x2 quad.
struct ProbeMask
{
    uint base;
    uint2 lane;
};

static const uint PROBE_MASK_ALL = 0xffffffffu;

// Every live probe.
ProbeMask probe_mask_all()
{
    ProbeMask m;
    m.base = PROBE_MASK_ALL;
    m.lane = uint2(0u, 0u);
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

// The probes binned into cluster `cid`, for the fragment at render-target pixel
// position `pixel`.
ProbeMask probe_cluster_mask(uint cid, float2 pixel)
{
    ProbeMask m;
    m.base = cluster_probe_mask_base(CLUSTER.grid_x * CLUSTER.grid_y * CLUSTER.grid_z, cid);
    m.lane = uint2(pixel) & 1u;
    return m;
}

// The probes that may matter at world-space point `world_pos`, seen at screen
// UV `uv` and render-target pixel position `pixel`: its cluster's on the main
// camera's view, else the whole set.
ProbeMask probe_mask_at(float2 uv, float3 world_pos, float2 pixel)
{
    if (CLUSTER.use_clusters == 0u)
    {
        return probe_mask_all();
    }
    float view_depth = dot(world_pos - CLUSTER.cam_pos_znear.xyz, CLUSTER.view_forward_zfar.xyz);
    return probe_cluster_mask(cluster_at(uv, view_depth), pixel);
}

#endif

// The next probe bit a blend visits: the least `own` (the fragment's next set
// bit, 32 for none) over its 2x2 quad, read back through the derivatives, but
// never below `lower` nor past `own`.
//
// Every pixel of a quad walking the same probes in the same order is what keeps
// the implicit derivatives `SampleBias` takes one probe's, however different the
// quad's masks: the quad visits the union of them, and a pixel whose own mask
// lacks a probe weighs it zero and skips its sample, as the whole set's walk
// would. The derivatives of these small integers are exact wherever the quad
// runs this in step; where it does not (a divergent caller), the clamps still
// visit every own bit and advance each step, so the walk stays complete and
// ends.
uint probe_next_bit(ProbeMask m, uint own, uint lower)
{
    uint next = own;
    if (m.base != PROBE_MASK_ALL)
    {
        float v = float(own);
        float dx = ddx_fine(v);
        float row = min(v, m.lane.x == 0u ? v + dx : v - dx);
        float dy = ddy_fine(row);
        next = uint(min(row, m.lane.y == 0u ? row + dy : row - dy));
    }
    return min(own, max(next, lower));
}

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
    float3 c = 0.5 * (PROBE_RECORDS[i].box_min.xyz + PROBE_RECORDS[i].box_max.xyz);
    float3 he = 0.5 * (PROBE_RECORDS[i].box_max.xyz - PROBE_RECORDS[i].box_min.xyz);
    // Signed distance to the box surface: positive inside, negative out.
    float3 q = abs(world_pos - c) - he;
    float sd = -(length(max(q, (float3)(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0));
    float margin = max(PROBE_BLEND_MARGIN * min(he.x, min(he.y, he.z)), 1e-4);
    return smoothstep(-margin, margin, sd);
}

// The influence-weighted blend of `probes` at `world_pos` along world-space ray
// `R` (partition of unity): the weight-normalized sum of each covering probe's
// box-projected sample, in index order, into `radiance`. False, with `radiance`
// zero, when no probe's influence reaches the point.
//
// Uncovered points take one of two fallbacks. The main pass, SSR and RT
// reflections call `probe_mask_specular`, which falls back to the nearest probe
// in the cluster's nearest mask, however far away its box is. The transparent
// passes branch on this instead and fall back to the sky, since a wide surface
// outside every box would otherwise show one capture of somewhere else.
bool probe_mask_blend(ProbeMask probes, float3 world_pos, float3 R, float lod, out float3 radiance)
{
    float3 acc = (float3)(0.0);
    float wsum = 0.0;
    for (uint w = 0u; w * 32u < PROBE_SET.count; w++)
    {
        uint bits = probe_live_bits(probe_mask_word(probes, 0u, w), w);
        uint lower = 0u;
        for (;;)
        {
            uint own = bits != 0u ? firstbitlow(bits) : 32u;
            uint b = probe_next_bit(probes, own, lower);
            if (b >= 32u)
            {
                break;
            }
            if (b == own)
            {
                uint i = w * 32u + b;
                float wt = probe_weight(i, world_pos);
                if (wt > 0.0)
                {
                    acc += wt * sample_probe_radiance(i, PROBE_RECORDS[i], world_pos, R, lod);
                    wsum += wt;
                }
            }
            bits &= ~((2u << b) - 1u);
            lower = b + 1u;
        }
    }
    radiance = wsum > 0.0 ? acc / wsum : (float3)(0.0);
    return wsum > 0.0;
}

// Probe radiance for `world_pos` along world-space ray `R`: the blend of every
// probe covering the point, else the nearest probe by capture distance.
float3 probe_mask_specular(ProbeMask probes, float3 world_pos, float3 R, float lod)
{
    float3 blended;
    if (probe_mask_blend(probes, world_pos, R, lod, blended))
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
    return sample_probe_radiance(near_i, PROBE_RECORDS[near_i], world_pos, R, lod);
}

