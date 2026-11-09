// The froxel cluster grid, spliced at a shader's CLUSTER_TYPES marker (see
// shader_source.rs): the params block the binning kernel and every reader of its
// lists share, and the list layout. Declares no resource and no entry point.
// Nothing here may spell the marker itself.
//
// One buffer holds every cluster's light list (a count followed by indices in
// ascending order), then every cluster's two reflection-probe masks: bit `i` of
// the influence mask is set when probe `i`'s influence reaches the cluster, and
// bit `i` of the nearest mask when probe `i` can be the nearest capture to a
// point in it. Probes past MAX_CLUSTERED_PROBES have no bit and are never
// masked out.
//
// Layout must match render_types.rs: ClusterParams (128 B) and the
// CLUSTER_LIGHT_LIST_STRIDE / MAX_LIGHTS_PER_CLUSTER / CLUSTER_PROBE_MASK_WORDS
// / MAX_CLUSTERED_PROBES constants.

static const uint CLUSTER_LIGHT_LIST_STRIDE = 64u;
static const uint MAX_LIGHTS_PER_CLUSTER = 63u;
static const uint CLUSTER_PROBE_MASK_WORDS = 8u;
static const uint MAX_CLUSTERED_PROBES = 256u;

struct ClusterParams
{
    float4x4 inv_view_proj;
    // xyz = camera position, w = z_near. float4 pairs rather than float3 +
    // scalar: MSL sizes a constant-buffer float3 at 16 bytes, so the packed
    // 128-byte CPU layout only survives on every target without vec3 fields.
    float4   cam_pos_znear;
    // xyz = view forward, w = z_far.
    float4   view_forward_zfar;
    uint     grid_x;
    uint     grid_y;
    uint     grid_z;
    uint     num_lights;
    float    screen_w;
    float    screen_h;
    uint     use_clusters;
    uint     num_probes;
};

// First word of cluster `cid`'s influence mask in a grid of `cluster_count`
// clusters, past every cluster's light list; its nearest mask follows it.
uint cluster_probe_mask_base(uint cluster_count, uint cid)
{
    return cluster_count * CLUSTER_LIGHT_LIST_STRIDE + cid * 2u * CLUSTER_PROBE_MASK_WORDS;
}
