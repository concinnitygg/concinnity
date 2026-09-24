// Clustered binning compute kernel. One thread per froxel cluster: builds the
// cluster's world-space AABB from its screen tile + exponential depth slice,
// tests every local light's bounding sphere and every reflection probe's
// influence box and capture point against it, and writes the surviving
// lights' indices and the probes' masks into the per-cluster lists the forward,
// SSR and transparent passes read.
//
// Single source for every backend. Vulkan binds set 0 bindings 0-3; Metal puts
// the four on buffer(0..3) and DirectX on b0 / t0 / u0 / t1, which is why the
// registers take a `CN_BACKEND_DIRECTX` branch: Metal has one buffer index
// space where D3D has three, so a `t0` and a `u0` that never collide on DXIL
// would both land on buffer(0). Everywhere else the register number is the Metal index
// (see concinnity-shader's `metal_bindings`).
//
// Layouts must match render_types.rs: ClusterParams (128 B), GpuLight (64 B)
// and ProbeUniforms (48 B).

{CLUSTER_TYPES}

{PROBE_TYPES}

{LIGHT_TYPES}

#ifdef CN_BACKEND_DIRECTX
#define LIGHTS_REGISTER t0
#define CLUSTER_LIST_REGISTER u0
#define PROBE_RECORDS_REGISTER t1
#else
// Metal has one buffer namespace, so the three structured buffers have to clear
// the params at buffer(0) and each other.
#define LIGHTS_REGISTER t1
#define CLUSTER_LIST_REGISTER u2
#define PROBE_RECORDS_REGISTER t3
#endif

[[vk::binding(0, 0)]] ConstantBuffer<ClusterParams> cluster : register(b0);
[[vk::binding(1, 0)]] StructuredBuffer<GpuLight> lights : register(LIGHTS_REGISTER);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> cluster_list : register(CLUSTER_LIST_REGISTER);
[[vk::binding(3, 0)]] StructuredBuffer<ProbeUniforms> probe_records : register(PROBE_RECORDS_REGISTER);

// Direction of the camera ray through a screen-NDC point. Unprojects the far
// plane (z = 1) to world space, then normalizes from the camera: for a
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
    float3 aabb_min = (float3)(1e30);
    float3 aabb_max = (float3)(-1e30);
    float3 p_far[4];
    for (uint i = 0u; i < 4u; ++i)
    {
        float3 ray = cluster_corner_ray(ndcs[i]);
        float  fdot = max(dot(ray, cluster.view_forward_zfar.xyz), 1e-4);
        float3 p_near = cluster.cam_pos_znear.xyz + ray * (near_d / fdot);
        p_far[i] = cluster.cam_pos_znear.xyz + ray * (far_d / fdot);
        aabb_min = min(aabb_min, min(p_near, p_far[i]));
        aabb_max = max(aabb_max, max(p_near, p_far[i]));
    }

    uint base  = cid * CLUSTER_LIGHT_LIST_STRIDE;
    uint count = 0u;
    for (uint li = 0u; li < cluster.num_lights; ++li)
    {
        float3 lp = lights[li].position_range.xyz;
        float  r  = lights[li].position_range.w;
        // Sphere vs AABB: distance from the light center to the clamped point.
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

    // A reader places a fragment by where it rasterized, which the sub-pixel
    // jitter moves off the un-jittered grid, so the probe test widens the AABB
    // by one pixel at the slice's far depth.
    float pixel = max(length(p_far[1] - p_far[0]) * float(cluster.grid_x) / cluster.screen_w,
                      length(p_far[2] - p_far[0]) * float(cluster.grid_y) / cluster.screen_h);
    float3 probe_min = aabb_min - pixel;
    float3 probe_max = aabb_max + pixel;

    // Probe influence boxes, grown by the blend margin `probe_weight` fades
    // them over. The nearest-capture candidates are every probe whose closest
    // approach to the AABB is no farther than the least farthest approach of
    // any probe: no other capture can be nearest to a point inside it.
    uint probes = min(cluster.num_probes, MAX_CLUSTERED_PROBES);
    float nearest_bound = 1e30;
    for (uint pi = 0u; pi < probes; ++pi)
    {
        float3 c = probe_records[pi].probe_pos.xyz;
        float3 far_corner = max(abs(c - probe_min), abs(c - probe_max));
        nearest_bound = min(nearest_bound, length(far_corner));
    }
    uint pbase = cluster_probe_mask_base(cluster_count, cid);
    for (uint w = 0u; w < CLUSTER_PROBE_MASK_WORDS; ++w)
    {
        uint influence = 0u;
        uint nearest = 0u;
        uint first = w * 32u;
        uint last = min(first + 32u, probes);
        for (uint pj = first; pj < last; ++pj)
        {
            float3 box_min = probe_records[pj].box_min.xyz;
            float3 box_max = probe_records[pj].box_max.xyz;
            float3 he = 0.5 * (box_max - box_min);
            float margin = max(PROBE_BLEND_MARGIN * min(he.x, min(he.y, he.z)), 1e-4);
            if (all(box_min - margin <= probe_max) && all(box_max + margin >= probe_min))
            {
                influence |= 1u << (pj - first);
            }
            float3 c = probe_records[pj].probe_pos.xyz;
            if (length(c - clamp(c, probe_min, probe_max)) <= nearest_bound)
            {
                nearest |= 1u << (pj - first);
            }
        }
        cluster_list[pbase + w] = influence;
        cluster_list[pbase + CLUSTER_PROBE_MASK_WORDS + w] = nearest;
    }
}
