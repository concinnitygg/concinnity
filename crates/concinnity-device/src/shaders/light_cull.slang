// Clustered light-binning compute kernel. One thread per froxel cluster: builds
// the cluster's world-space AABB from its screen tile + exponential depth slice,
// tests every local light's bounding sphere against it, and writes the surviving
// light indices into the per-cluster list the forward pass reads.
//
// Single source for every backend. Vulkan binds set 0 bindings 0-2; Metal and
// DXIL assign the same slots from declaration order (params 0, lights 1,
// list 2), matching the hand-assigned host bindings on all three.
//
// Layouts must match render_types.rs: ClusterParams (128 B) and GpuLight (64 B).
// CLUSTER_LIGHT_LIST_STRIDE mirrors MAX_LIGHTS_PER_CLUSTER + 1.

static const uint CLUSTER_LIGHT_LIST_STRIDE = 64u;
static const uint MAX_LIGHTS_PER_CLUSTER = 63u;

struct GpuLight
{
    float3 position;
    float  range;
    float3 color;
    float  intensity;
    float3 direction;
    uint   kind;
    float  cos_inner;
    float  cos_outer;
    int    shadow_index;
    // Index into the AreaLightData table for an area light, else -1. Unused by
    // the cull (the bounding sphere covers every light kind) but kept so this
    // declaration matches the forward pass's byte for byte.
    int    data_index;
};

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
    uint     _pad;
};

[[vk::binding(0, 0)]]
ConstantBuffer<ClusterParams> cluster;

[[vk::binding(1, 0)]]
StructuredBuffer<GpuLight> lights;

[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> cluster_list;

// Direction of the camera ray through a screen-NDC point. Unprojects the far
// plane (z = 1) to world space, then normalises from the camera: for a
// perspective projection every ray through a screen point passes through the
// eye, so the far-plane unprojection gives the direction.
float3 cluster_corner_ray(float2 ndc)
{
    float4 clip = float4(ndc, 1.0, 1.0);
    float4 world = mul(cluster.inv_view_proj, clip);
    world /= world.w;
    return normalize(world.xyz - cluster.cam_pos_znear.xyz);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void light_cull_kernel(uint3 tid : SV_DispatchThreadID)
{
    uint cid = tid.x;
    uint cluster_count = cluster.grid_x * cluster.grid_y * cluster.grid_z;
    if (cid >= cluster_count)
    {
        return;
    }

    uint cx = cid % cluster.grid_x;
    uint cy = (cid / cluster.grid_x) % cluster.grid_y;
    uint cz = cid / (cluster.grid_x * cluster.grid_y);

    // Screen-tile NDC bounds (y flipped: screen y-down to NDC y-up).
    float2 lo = float2(float(cx), float(cy)) / float2(float(cluster.grid_x), float(cluster.grid_y));
    float2 hi = float2(float(cx + 1u), float(cy + 1u)) / float2(float(cluster.grid_x), float(cluster.grid_y));
    float2 ndcs[4] = {
        float2(lo.x * 2.0 - 1.0, -(lo.y * 2.0 - 1.0)),
        float2(hi.x * 2.0 - 1.0, -(lo.y * 2.0 - 1.0)),
        float2(lo.x * 2.0 - 1.0, -(hi.y * 2.0 - 1.0)),
        float2(hi.x * 2.0 - 1.0, -(hi.y * 2.0 - 1.0)),
    };

    // Exponential depth slice: near/far view-space distances for this Z band.
    float ratio  = cluster.view_forward_zfar.w / cluster.cam_pos_znear.w;
    float near_d = cluster.cam_pos_znear.w * pow(ratio, float(cz) / float(cluster.grid_z));
    float far_d  = cluster.cam_pos_znear.w * pow(ratio, float(cz + 1u) / float(cluster.grid_z));

    // World-space AABB over the tile frustum clamped to [near_d, far_d].
    float3 aabb_min = float3(1e30);
    float3 aabb_max = float3(-1e30);
    for (uint i = 0u; i < 4u; ++i)
    {
        float3 ray = cluster_corner_ray(ndcs[i]);
        float  fdot = max(dot(ray, cluster.view_forward_zfar.xyz), 1e-4);
        float3 p_near = cluster.cam_pos_znear.xyz + ray * (near_d / fdot);
        float3 p_far  = cluster.cam_pos_znear.xyz + ray * (far_d / fdot);
        aabb_min = min(aabb_min, min(p_near, p_far));
        aabb_max = max(aabb_max, max(p_near, p_far));
    }

    uint base  = cid * CLUSTER_LIGHT_LIST_STRIDE;
    uint count = 0u;
    for (uint li = 0u; li < cluster.num_lights; ++li)
    {
        float3 lp = lights[li].position;
        float  r  = lights[li].range;
        // Sphere vs AABB: distance from the light centre to the clamped point.
        float3 d = lp - clamp(lp, aabb_min, aabb_max);
        if (dot(d, d) <= r * r)
        {
            if (count < MAX_LIGHTS_PER_CLUSTER)
            {
                cluster_list[base + 1u + count] = li;
                count += 1u;
            }
        }
    }
    cluster_list[base] = count;
}
