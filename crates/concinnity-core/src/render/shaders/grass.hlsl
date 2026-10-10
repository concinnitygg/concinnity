// Procedural grass: single source for every backend.
//
// GRASS_GENERATE compiles the kernel. One group covers 64 cells of one tile:
// each thread places the cell's blade (a hash of the cell's world coordinates
// jitters it, and the nearest of a jittered clump lattice gives it the height,
// facing and lean it shares with its clump), drops it past the field's edge,
// the draw distance or the view frustum, and appends the survivors to the
// visible-blade buffer. The same dispatch fills this frame's draw-argument
// slot and resets the other one for the next frame.
//
// The other entries draw the blades with one indirect draw each: a strip of
// GRASS_BLADE_VERTICES vertices per blade, bent along a quadratic Bezier by the
// blade's lean and the wind. The pre-pass pair writes the G-buffer, the
// blade's motion included, and runs the bend a second time at the previous
// frame's clock and camera so a swaying blade reprojects to where it was drawn.
// The lit pair shades the blade into the main pass. Both splice the main
// pass's resources, so they draw inside those passes on the bindings already
// there; the grass block and the blades add one vertex-stage pair of their
// own:
//
//   CN_BACKEND_METAL   buffer(19) params, buffer(20) blades
//   CN_BACKEND_DIRECTX b7 params, t23 blades in both root signatures
//   Vulkan             set 2 (main) or set 3 (pre-pass), bindings 0 and 1

{WIND}

// Mirrors `GrassParams` in render::uniforms::grass (256 B).
struct GrassParams
{
    // min x, min z, max x, max z.
    float4 patch_rect;
    float ground_y;
    float tile_size;
    float cell_size;
    uint cells_per_side;
    int2 tile_origin;
    uint2 tile_count;
    // xyz = camera position, w = draw distance.
    float4 cam_pos_distance;
    // (normal, d) world-space planes; inside is dot(normal, p) + d >= 0.
    float4 frustum[6];
    float height;
    float height_variance;
    float width;
    float clump_size;
    float stiffness;
    float color_variation;
    uint capacity;
    uint args_slot;
    float4 root_color;
    float4 tip_color;
    float4 wind;
    float4 wind_gust;
};

// Mirrors `GpuGrassBlade` in render::uniforms::grass (32 B).
struct GrassBlade
{
    // xyz = root, w = facing angle in radians.
    float4 root_facing;
    // x = height, y = root width, z = static lean, w = random bits (an
    // integer below 2^24, stored exactly).
    float4 shape;
};

static const uint GRASS_BLADE_VERTICES = 15u;
static const float GRASS_TAU = 6.28318530718;

// One 32-bit hash step (PCG).
uint grass_hash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint grass_hash2(int2 c)
{
    return grass_hash(asuint(c.x) + grass_hash(asuint(c.y)));
}

// A unit float from `h` mixed with `salt`, so one hash yields many draws.
float grass_unit(uint h, uint salt)
{
    return float(grass_hash(h ^ salt) >> 8u) * (1.0 / 16777216.0);
}

#ifdef GRASS_GENERATE

[[vk::binding(0, 0)]] ConstantBuffer<GrassParams> grass : register(b0);
[[vk::binding(1, 0)]] RWStructuredBuffer<GrassBlade> blades_out : register(u1);
// Two slots of (vertex count, instance count, first vertex, first instance).
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> draw_args : register(u2);

// Matches GRASS_GROUP_SIZE in render::grass::tiles.
#define GRASS_GROUP_SIZE 64

groupshared uint gs_count;
groupshared uint gs_base;

// Tallest a blade of this field can stand, with its clump's boost.
float grass_max_height()
{
    return grass.height * (1.0 + grass.height_variance);
}

bool grass_outside(float3 center, float radius)
{
    [unroll]
    for (uint i = 0u; i < 6u; i++)
    {
        if (dot(grass.frustum[i].xyz, center) + grass.frustum[i].w < -radius)
        {
            return true;
        }
    }
    return false;
}

// Whether any blade rooted in `tile` can be seen: its box, grown by how far a
// blade can lean out of it, meets the frustum and the draw distance.
bool grass_tile_visible(int2 tile)
{
    float reach = grass_max_height();
    float2 lo = float2(tile) * grass.tile_size - reach;
    float2 hi = float2(tile + 1) * grass.tile_size + reach;
    float3 bmin = float3(lo.x, grass.ground_y - reach, lo.y);
    float3 bmax = float3(hi.x, grass.ground_y + reach, hi.y);
    [unroll]
    for (uint i = 0u; i < 6u; i++)
    {
        float3 n = grass.frustum[i].xyz;
        float3 p = float3(n.x >= 0.0 ? bmax.x : bmin.x,
                          n.y >= 0.0 ? bmax.y : bmin.y,
                          n.z >= 0.0 ? bmax.z : bmin.z);
        if (dot(n, p) + grass.frustum[i].w < 0.0)
        {
            return false;
        }
    }
    float3 cam = grass.cam_pos_distance.xyz;
    float3 nearest = clamp(cam, bmin, bmax);
    return distance(nearest, cam) <= grass.cam_pos_distance.w;
}

// The clump `xz` belongs to: the nearest center of a jittered lattice of
// clump-sized cells. Returns its hash and writes its center.
uint grass_clump(float2 xz, out float2 center)
{
    float2 p = xz / grass.clump_size;
    int2 base = int2(floor(p));
    float best = 1e30;
    uint best_hash = 0u;
    center = xz;
    [unroll]
    for (int dz = -1; dz <= 1; dz++)
    {
        [unroll]
        for (int dx = -1; dx <= 1; dx++)
        {
            int2 cell = base + int2(dx, dz);
            uint h = grass_hash2(cell) ^ 0x9e3779b9u;
            float2 c = float2(cell) + float2(grass_unit(h, 1u), grass_unit(h, 2u));
            float d = dot(c - p, c - p);
            if (d < best)
            {
                best = d;
                best_hash = h;
                center = c * grass.clump_size;
            }
        }
    }
    return best_hash;
}

// Place the blade of world cell `cell`. False when it falls outside the
// field, past the draw distance, or out of view.
bool grass_place(int2 cell, out GrassBlade blade)
{
    blade = (GrassBlade)0;
    uint h = grass_hash2(cell);
    float2 xz = (float2(cell) + float2(grass_unit(h, 3u), grass_unit(h, 4u))) * grass.cell_size;
    if (any(xz < grass.patch_rect.xy) || any(xz >= grass.patch_rect.zw))
    {
        return false;
    }
    float3 root = float3(xz.x, grass.ground_y, xz.y);
    float3 cam = grass.cam_pos_distance.xyz;
    float dist = distance(root, cam);
    float far = grass.cam_pos_distance.w;
    // Thin the last quarter of the draw distance out, so the field ends in a
    // fade rather than an edge.
    float keep = saturate((far - dist) / (0.25 * far));
    if (grass_unit(h, 5u) >= keep)
    {
        return false;
    }

    float2 clump_center;
    uint ch = grass_clump(xz, clump_center);
    float clump_r = grass_unit(ch, 6u);
    float height = grass.height
                 * (1.0 + grass.height_variance * ((clump_r - 0.5) * 1.2 + (grass_unit(h, 7u) - 0.5) * 0.8));
    height = max(height, grass.height * 0.1);
    if (grass_outside(root + float3(0.0, 0.5 * height, 0.0), height))
    {
        return false;
    }

    // A clump's blades lean the clump's way and splay away from its middle.
    float clump_angle = grass_unit(ch, 8u) * GRASS_TAU;
    float2 shared_dir = float2(cos(clump_angle), sin(clump_angle));
    float2 out_dir = xz - clump_center;
    float out_len = length(out_dir);
    out_dir = out_len > 1e-4 ? out_dir / out_len : shared_dir;
    float2 front = normalize(shared_dir * 0.6 + out_dir * 0.8
                             + (float2(grass_unit(h, 9u), grass_unit(h, 10u)) - 0.5) * 0.9);
    // front = (-sin, cos) of the facing angle.
    float facing = atan2(-front.x, front.y);
    float lean = (0.15 + 0.35 * grass_unit(ch, 11u)) * (0.6 + 0.8 * grass_unit(h, 12u));
    float width = grass.width * (0.75 + 0.5 * grass_unit(h, 13u));
    // 12 bits of the clump's hash beside 12 of the blade's, so the draws can
    // vary hue per clump and per blade.
    uint bits = ((ch & 0xfffu) << 12u) | (h & 0xfffu);

    blade.root_facing = float4(root, facing);
    blade.shape = float4(height, width, lean, float(bits));
    return true;
}

[shader("compute")]
[numthreads(GRASS_GROUP_SIZE, 1, 1)]
void grass_generate(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    uint slot = (grass.args_slot & 1u) * 4u;
    uint next = 4u - slot;
    if (gi == 0u && all(gid == 0u))
    {
        draw_args[next + 0u] = GRASS_BLADE_VERTICES;
        draw_args[next + 1u] = 0u;
        draw_args[next + 2u] = 0u;
        draw_args[next + 3u] = 0u;
        draw_args[slot + 0u] = GRASS_BLADE_VERTICES;
        draw_args[slot + 2u] = 0u;
        draw_args[slot + 3u] = 0u;
    }
    if (gi == 0u)
    {
        gs_count = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    GrassBlade blade = (GrassBlade)0;
    bool keep = false;
    uint n = grass.cells_per_side;
    uint index = gid.x * GRASS_GROUP_SIZE + gi;
    if (gid.y < grass.tile_count.x && gid.z < grass.tile_count.y && index < n * n)
    {
        int2 tile = grass.tile_origin + int2(gid.yz);
        if (grass_tile_visible(tile))
        {
            int2 cell = tile * int(n) + int2(index % n, index / n);
            keep = grass_place(cell, blade);
        }
    }

    uint local = 0u;
    if (keep)
    {
        InterlockedAdd(gs_count, 1u, local);
    }
    GroupMemoryBarrierWithGroupSync();

    // One reservation per group. A group that overflows the buffer keeps what
    // fits and hands the rest of its reservation back; the count never drops
    // below the capacity once it has reached it, so every later group finds
    // the buffer full and the final count is exactly what was written.
    if (gi == 0u && gs_count > 0u)
    {
        uint base;
        InterlockedAdd(draw_args[slot + 1u], gs_count, base);
        uint room = base < grass.capacity ? grass.capacity - base : 0u;
        uint over = gs_count - min(room, gs_count);
        if (over > 0u)
        {
            uint ignored;
            InterlockedAdd(draw_args[slot + 1u], 0u - over, ignored);
        }
        gs_base = base;
    }
    GroupMemoryBarrierWithGroupSync();

    if (keep)
    {
        uint i = gs_base + local;
        if (i < grass.capacity)
        {
            blades_out[i] = blade;
        }
    }
}

#else // the draws

{MAIN_RESOURCES}

#ifdef CN_BACKEND_METAL
[[vk::binding(20, 0)]] ConstantBuffer<GrassParams> grass : register(b19);
[[vk::binding(21, 0)]] StructuredBuffer<GrassBlade> grass_blades : register(t20);
#elif defined(CN_BACKEND_DIRECTX)
ConstantBuffer<GrassParams> grass : register(b7);
StructuredBuffer<GrassBlade> grass_blades : register(t23);
#elif defined(SURFACE_PREPASS)
[[vk::binding(0, 3)]] ConstantBuffer<GrassParams> grass : register(b0, space3);
[[vk::binding(1, 3)]] StructuredBuffer<GrassBlade> grass_blades : register(t1, space3);
#else
[[vk::binding(0, 2)]] ConstantBuffer<GrassParams> grass : register(b0, space2);
[[vk::binding(1, 2)]] StructuredBuffer<GrassBlade> grass_blades : register(t1, space2);
#endif

// One vertex of a bent blade.
struct GrassVertex
{
    float3 world_pos;
    // The blade's front, perpendicular to its width and its curve.
    float3 normal;
    // Across the blade, root to tip unchanged.
    float3 side;
    // 0 at the root to 1 at the tip.
    float t;
    // -1 to 1 across the blade.
    float u;
};

// How far the wind bends a blade of `stiffness` at `root_xz` at time `time`,
// as a horizontal tip offset per meter of blade. The blade's own phase makes
// neighbors flutter out of step.
float2 grass_wind_bend(float2 root_xz, float time, float phase)
{
    Wind w = wind_unpack(grass.wind, grass.wind_gust);
    float speed = wind_speed(w, root_xz, time);
    float give = lerp(3.0, 28.0, grass.stiffness);
    float push = speed / (speed + give);
    float flutter = sin(time * (4.0 + 3.0 * phase) + phase * GRASS_TAU) * 0.12 * push;
    float2 across = float2(-w.dir.y, w.dir.x);
    return w.dir * (push * 0.9 + flutter) + across * (flutter * 0.5);
}

// Vertex `vid` of `blade` at time `time`, seen from `cam`. Pairs climb the
// blade root to tip and the last vertex is the tip, so the strip is
// GRASS_BLADE_VERTICES long and narrows to a point.
GrassVertex grass_blade_vertex(GrassBlade blade, uint vid, float time, float3 cam)
{
    float3 root = blade.root_facing.xyz;
    float facing = blade.root_facing.w;
    float height = blade.shape.x;
    float width = blade.shape.y;
    float phase = grass_unit(uint(blade.shape.w), 14u);

    float2 side2 = float2(cos(facing), sin(facing));
    float2 front2 = float2(-side2.y, side2.x);
    float2 bend = front2 * blade.shape.z + grass_wind_bend(root.xz, time, phase);
    float bend_len = length(bend);
    if (bend_len > 0.95)
    {
        bend *= 0.95 / bend_len;
    }
    float rise = sqrt(1.0 - dot(bend, bend));

    // The control point sits above the root at the tip's height, so the blade
    // leaves the ground upright and arcs over. Scaling both offsets by the
    // curve's approximate length keeps every blade its own height long.
    float3 p0 = root;
    float3 d1 = float3(0.0, rise, 0.0) * height;
    float3 d2 = float3(bend.x, rise, bend.y) * height;
    float chord = length(d2);
    float approx_len = (2.0 * chord + length(d1) + length(d2 - d1)) / 3.0;
    float scale = height / max(approx_len, 1e-4);
    d1 *= scale;
    d2 *= scale;

    uint pair = vid >> 1u;
    bool tip = vid >= GRASS_BLADE_VERTICES - 1u;
    float t = tip ? 1.0 : float(pair) / 7.0;
    float u = tip ? 0.0 : ((vid & 1u) != 0u ? 1.0 : -1.0);

    float s = 1.0 - t;
    float3 pos = p0 + 2.0 * s * t * d1 + t * t * d2;
    float3 tangent = normalize(2.0 * s * d1 + 2.0 * t * (d2 - d1) + float3(0.0, 1e-5, 0.0));
    float3 side = float3(side2.x, 0.0, side2.y);
    float3 normal = normalize(cross(side, tangent));

    // A blade seen edge-on would thin to nothing and shimmer, so its width
    // swings toward the screen as it turns away.
    float3 view = normalize(cam - pos);
    float edge = 1.0 - abs(dot(view, normal));
    float3 toward = cross(view, tangent);
    float toward_len = length(toward);
    toward = toward_len > 1e-4 ? toward / toward_len : side;
    toward *= dot(toward, side) < 0.0 ? -1.0 : 1.0;
    float3 across = normalize(lerp(side, toward, edge * edge * 0.6));

    float half_width = 0.5 * width * (1.0 - pow(t, 1.4)) * (1.0 + 0.3 * edge);
    GrassVertex v;
    v.world_pos = pos + across * (u * half_width);
    v.normal = normal;
    v.side = side;
    v.t = t;
    v.u = u;
    return v;
}

// The blade's color at height `t`: the root-to-tip ramp, with hue and
// brightness shifted per clump and per blade.
float3 grass_albedo(float t, uint bits)
{
    float clump_r = grass_unit(bits >> 12u, 15u);
    float blade_r = grass_unit(bits & 0xfffu, 16u);
    float3 ramp = lerp(grass.root_color.rgb, grass.tip_color.rgb, smoothstep(0.0, 1.0, t));
    float v = grass.color_variation;
    // Drier clumps lean yellow; every blade jitters in brightness.
    float3 dry = ramp * float3(1.35, 1.1, 0.55);
    float3 color = lerp(ramp, dry, v * clump_r * clump_r);
    return color * (1.0 + v * (blade_r - 0.5) * 0.8);
}

// How much of the sky reaches a point `t` up the blade inside the field.
float grass_root_occlusion(float t)
{
    return lerp(0.25, 1.0, smoothstep(0.0, 0.6, t));
}

// The normal the light sees: the blade's front turned toward the viewer, then
// rolled across the width so the flat strip shades as a rounded blade.
float3 grass_shading_normal(float3 normal, float3 side, float u, float3 to_camera)
{
    float3 n = dot(normal, to_camera) < 0.0 ? -normal : normal;
    return normalize(n + side * (u * 0.45));
}

#ifndef SURFACE_PREPASS

struct GrassVertexOut
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_pos : TEXCOORD0;
    [[vk::location(1)]] float3 normal : TEXCOORD1;
    [[vk::location(2)]] float3 side : TEXCOORD2;
    [[vk::location(3)]] float3 to_camera : TEXCOORD3;
    [[vk::location(4)]] float3 albedo : TEXCOORD4;
    // x = t up the blade, y = u across it, z = view depth.
    [[vk::location(5)]] float3 t_u_depth : TEXCOORD5;
};

[shader("vertex")]
GrassVertexOut grass_vertex(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    GrassBlade blade = grass_blades[iid];
    float3 cam = float3(VIEW.cam_x, VIEW.cam_y, VIEW.cam_z);
    GrassVertex v = grass_blade_vertex(blade, vid, VIEW.elapsed, cam);
    GrassVertexOut o;
    o.position = mul(VIEW.vp, float4(v.world_pos, 1.0));
    o.world_pos = v.world_pos;
    o.normal = v.normal;
    o.side = v.side;
    o.to_camera = cam - v.world_pos;
    o.albedo = grass_albedo(v.t, uint(blade.shape.w));
    o.t_u_depth = float3(v.t, v.u, -mul(VIEW.view_mat, float4(v.world_pos, 1.0)).z);
    return o;
}

// Light arriving from `L` at `radiance`: wrapped diffuse on the lit face,
// light carried through the blade when the source is behind it, and a soft
// sheen.
float3 grass_light(float3 albedo, float3 N, float3 V, float3 L, float3 radiance)
{
    float ndl = dot(N, L);
    float3 diffuse = albedo * (saturate((ndl + 0.35) / 1.35) / PI);
    float behind = saturate(-ndl);
    float through = pow(saturate(dot(V, -L)), 4.0);
    float3 translucent = albedo * float3(1.1, 1.25, 0.6) * (0.35 * behind + 0.9 * through) / PI;
    float3 H = normalize(V + L);
    float nl = saturate(ndl);
    float nv = max(dot(N, V), 1e-3);
    const float roughness = 0.5;
    float spec = distribution_ggx(N, H, roughness) * geometry_smith(N, V, L, roughness)
               / max(4.0 * nv * nl, 1e-3);
    float3 sheen = fresnel_schlick(saturate(dot(H, V)), (float3)(0.04)) * spec * 0.5;
    return (diffuse + translucent) * radiance + sheen * radiance * nl;
}

[shader("pixel")]
float4 grass_fragment(GrassVertexOut p) : SV_Target
{
    float t = p.t_u_depth.x;
    float view_depth = p.t_u_depth.z;
    float3 albedo = p.albedo;
    if (VIEW.shade_mode > 0.5)
    {
        return float4(albedo, 1.0);
    }

    float3 V = normalize(p.to_camera);
    float3 N = grass_shading_normal(normalize(p.normal), normalize(p.side), p.t_u_depth.y, V);
    float occlusion = grass_root_occlusion(t);
    float2 screen_xy = p.position.xy;

    float3 Lo = (float3)(0.0);
    float shadow = shadow_factor_cascaded(p.world_pos, view_depth, screen_xy);
    for (int i = 0; i < LIGHTS.num_dir; i++)
    {
        float3 L = normalize(LIGHTS.dir[i].dir_i.xyz);
        float3 radiance = LIGHTS.dir[i].col.xyz * LIGHTS.dir[i].dir_i.w;
        float s = (i == 0) ? shadow : 1.0;
        // Blades deep in the field see less of the sky's direct light too.
        Lo += grass_light(albedo, N, V, L, radiance) * s * lerp(0.55, 1.0, t);
    }

    // Point and spot lights from the fragment's cluster; area lights are left
    // to the surfaces the panel faces.
    uint cluster_base = 0u;
    int local_count;
    if (CLUSTER.use_clusters != 0u)
    {
        uint cid = cluster_at(screen_xy / float2(CLUSTER.screen_w, CLUSTER.screen_h), view_depth);
        cluster_base = cid * CLUSTER_LIGHT_LIST_STRIDE;
        local_count = int(CLUSTER_LIST[cluster_base]);
    }
    else
    {
        local_count = LIGHTS.num_local_lights;
    }
    for (int jj = 0; jj < local_count; jj++)
    {
        int i = (CLUSTER.use_clusters != 0u)
              ? int(CLUSTER_LIST[cluster_base + 1u + uint(jj)])
              : jj;
        if (light_kind(LOCAL_LIGHTS[i]) == LIGHT_KIND_AREA)
        {
            continue;
        }
        float3 to_light = LOCAL_LIGHTS[i].position_range.xyz - p.world_pos;
        float dist = length(to_light);
        float3 L = to_light / max(dist, 1e-4);
        float atten = saturate(1.0 - dist / LOCAL_LIGHTS[i].position_range.w);
        atten *= atten;
        if (light_kind(LOCAL_LIGHTS[i]) == LIGHT_KIND_SPOT)
        {
            float cd = dot(LOCAL_LIGHTS[i].direction_kind.xyz, -L);
            float ci = LOCAL_LIGHTS[i].cos_inner;
            float co = LOCAL_LIGHTS[i].cos_outer;
            float k = saturate((cd - co) / max(ci - co, 1e-4));
            atten *= k * k;
            int si = LOCAL_LIGHTS[i].shadow_index;
            if (si >= 0 && atten > 0.0)
            {
                atten *= sample_spot_shadow(si, p.world_pos, N, screen_xy);
            }
        }
        float3 radiance = LOCAL_LIGHTS[i].color_intensity.xyz * LOCAL_LIGHTS[i].color_intensity.w * atten;
        Lo += grass_light(albedo, N, V, L, radiance);
    }

    float3 ambient = VIEW.prefilter_mip_count > 0.5
                   ? albedo * irradiance_sample(N) / PI
                   : (float3)(0.03) * albedo;
    ambient *= LIGHTS.ambient_intensity * occlusion;
    if (VIEW.ambient_occlusion > 0.5)
    {
        ambient *= ssao_sample(screen_xy / ssao_size());
    }
    return float4(ambient + Lo * occlusion, 1.0);
}

#else // SURFACE_PREPASS

struct GrassPrepassOut
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 normal : TEXCOORD0;
    [[vk::location(1)]] float3 side : TEXCOORD1;
    [[vk::location(2)]] float3 to_camera : TEXCOORD2;
    // x = u across the blade, y = view depth.
    [[vk::location(3)]] float2 u_depth : TEXCOORD3;
    [[vk::location(4)]] float4 cur_clip : TEXCOORD4;
    [[vk::location(5)]] float4 prev_clip : TEXCOORD5;
};

// The blade rasterized through the main pass's jittered view, and bent again
// at the previous frame's clock and camera for its motion. Both positions
// reproject through the unjittered matrices, so jitter never reaches the
// motion vector.
[shader("vertex")]
GrassPrepassOut grass_prepass_vertex(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    GrassBlade blade = grass_blades[iid];
    float3 cam = float3(view_cb.cam_x, view_cb.cam_y, view_cb.cam_z);
    GrassVertex v = grass_blade_vertex(blade, vid, view_cb.elapsed, cam);
    float3 prev_world = v.world_pos;
    if (gb_view.motion != 0u)
    {
        float3 prev_cam = float3(gb_view.prev_cam_x, gb_view.prev_cam_y, gb_view.prev_cam_z);
        prev_world = grass_blade_vertex(blade, vid, gb_view.prev_elapsed, prev_cam).world_pos;
    }
    GrassPrepassOut o;
    o.position = mul(view_cb.vp, float4(v.world_pos, 1.0));
    o.normal = v.normal;
    o.side = v.side;
    o.to_camera = cam - v.world_pos;
    o.u_depth = float2(v.u, -mul(view_cb.view_mat, float4(v.world_pos, 1.0)).z);
    o.cur_clip = mul(gb_view.cur_vp, float4(v.world_pos, 1.0));
    o.prev_clip = mul(gb_view.prev_vp, float4(prev_world, 1.0));
    return o;
}

// Grass is matte: the roughness the reflection resolves skip it at.
static const float GRASS_ROUGHNESS = 0.6;

[shader("pixel")]
GbFragmentOut grass_prepass_fragment(GrassPrepassOut p)
{
    float3 V = normalize(p.to_camera);
    float3 N = grass_shading_normal(normalize(p.normal), normalize(p.side), p.u_depth.x, V);
    GbFragmentOut o;
    o.nd = float4(normalize(mul((float3x3)gb_view.view_mat, N)), p.u_depth.y);
    o.rough = GRASS_ROUGHNESS;
    o.vel = gb_motion(p.cur_clip, p.prev_clip);
    return o;
}

#endif // SURFACE_PREPASS

#endif // GRASS_GENERATE
