#include <metal_stdlib>
using namespace metal;

// Clustered light-binning compute kernel. One thread per froxel cluster: builds
// the cluster's world-space AABB from its screen tile + exponential depth slice,
// tests every local light's bounding sphere against it, and writes the surviving
// light indices into the per-cluster list the forward pass reads.
//
// Layouts must match render_types.rs: ClusterParams (128 B) and GpuLight (64 B).
// CLUSTER_LIGHT_LIST_STRIDE mirrors MAX_LIGHTS_PER_CLUSTER + 1.

constant uint CLUSTER_LIGHT_LIST_STRIDE = 64u;
constant uint MAX_LIGHTS_PER_CLUSTER = 63u;

struct GpuLight {
    packed_float3 position;
    float         range;
    packed_float3 color;
    float         intensity;
    packed_float3 direction;
    uint          kind;
    float         cos_inner;
    float         cos_outer;
    int           shadow_index;
    float         _pad;
};

struct ClusterParams {
    float4x4      inv_view_proj;
    packed_float3 cam_pos;
    float         z_near;
    packed_float3 view_forward;
    float         z_far;
    uint          grid_x;
    uint          grid_y;
    uint          grid_z;
    uint          num_lights;
    float         screen_w;
    float         screen_h;
    uint          use_clusters;
    uint          _pad;
};

// Direction of the camera ray through a screen-NDC point. Unprojects the far
// plane (z = 1) to world space, then normalises from the camera: for a
// perspective projection every ray through a screen point passes through the eye,
// so the far-plane unprojection gives the direction. Mirrors froxel_to_world in
// fog.metal.
static float3 cluster_corner_ray(float2 ndc, constant ClusterParams &c) {
    float4 clip = float4(ndc, 1.0, 1.0);
    float4 world = c.inv_view_proj * clip;
    world /= world.w;
    return normalize(world.xyz - float3(c.cam_pos));
}

kernel void light_cull_kernel(
    uint cid                       [[thread_position_in_grid]],
    constant ClusterParams &c      [[buffer(0)]],
    constant GpuLight      *lights [[buffer(1)]],
    device uint            *cluster_list [[buffer(2)]]
) {
    uint cluster_count = c.grid_x * c.grid_y * c.grid_z;
    if (cid >= cluster_count) {
        return;
    }

    uint cx = cid % c.grid_x;
    uint cy = (cid / c.grid_x) % c.grid_y;
    uint cz = cid / (c.grid_x * c.grid_y);

    // Screen-tile NDC bounds (y flipped: screen y-down to NDC y-up).
    float2 lo = float2(float(cx), float(cy)) / float2(float(c.grid_x), float(c.grid_y));
    float2 hi = float2(float(cx + 1u), float(cy + 1u)) / float2(float(c.grid_x), float(c.grid_y));
    float2 ndcs[4] = {
        float2(lo.x * 2.0 - 1.0, -(lo.y * 2.0 - 1.0)),
        float2(hi.x * 2.0 - 1.0, -(lo.y * 2.0 - 1.0)),
        float2(lo.x * 2.0 - 1.0, -(hi.y * 2.0 - 1.0)),
        float2(hi.x * 2.0 - 1.0, -(hi.y * 2.0 - 1.0)),
    };

    // Exponential depth slice: near/far view-space distances for this Z band.
    float ratio  = c.z_far / c.z_near;
    float near_d = c.z_near * pow(ratio, float(cz)        / float(c.grid_z));
    float far_d  = c.z_near * pow(ratio, float(cz + 1u)   / float(c.grid_z));

    float3 cam = float3(c.cam_pos);
    float3 fwd = float3(c.view_forward);

    // World-space AABB over the tile frustum clamped to [near_d, far_d].
    float3 aabb_min = float3(1e30);
    float3 aabb_max = float3(-1e30);
    for (uint i = 0; i < 4; ++i) {
        float3 ray = cluster_corner_ray(ndcs[i], c);
        float  fdot = max(dot(ray, fwd), 1e-4);
        float3 p_near = cam + ray * (near_d / fdot);
        float3 p_far  = cam + ray * (far_d / fdot);
        aabb_min = min(aabb_min, min(p_near, p_far));
        aabb_max = max(aabb_max, max(p_near, p_far));
    }

    uint base  = cid * CLUSTER_LIGHT_LIST_STRIDE;
    uint count = 0u;
    for (uint li = 0u; li < c.num_lights; ++li) {
        float3 lp = float3(lights[li].position);
        float  r  = lights[li].range;
        // Sphere vs AABB: distance from the light centre to the clamped point.
        float3 d = lp - clamp(lp, aabb_min, aabb_max);
        if (dot(d, d) <= r * r) {
            if (count < MAX_LIGHTS_PER_CLUSTER) {
                cluster_list[base + 1u + count] = li;
                count += 1u;
            }
        }
    }
    cluster_list[base] = count;
}
