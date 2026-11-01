// Hardware ray-traced reflections: single source for every backend.
//
// Two compute kernels over the trace target, screen space first:
//
//   RT_CLASSIFY - per glossy pixel, walks the reflection ray through SSGI's
//                 depth pyramid and blends what it finds on screen with the
//                 local reflection probe / IBL prefilter cube by how far the
//                 screen hit can be trusted. A mirror-like pixel whose ray
//                 leaves the screen, passes behind geometry or lands on a hit it
//                 cannot trust owes that share to a hardware ray: it writes the
//                 settled part of its reflection with the share still owed in
//                 .a, and appends itself to the ray list, whose indirect
//                 dispatch arguments it grows to cover. Every other pixel writes
//                 its finished reflected radiance (.rgb) + the Fresnel/gloss
//                 composite weight (.a).
//   default     - the trace, dispatched indirectly over the ray list, so sky,
//                 rough and screen-settled pixels take no thread. Rebuilds the
//                 listed pixel's world-space surface point + normal from the SSR
//                 pre-pass G-buffer, traces the ray against the scene's
//                 acceleration structure, shades the hit (or falls back to the
//                 probe / prefilter cube on a miss) and adds it in by the share
//                 the classify left, writing the composite weight over it.
//   RT_TEXTURED - the trace sampling the hit's albedo / normal / emissive maps
//                 from the bindless texture pool, the path standard worlds
//                 take; the default trace takes the per-object material tint.
//
// An on-screen hit reads the fully lit scene, which a traced hit (sun and sky
// only, no grass blades in the acceleration structure) cannot match, so the
// screen answer wins wherever it is good and the confidence blend hides the
// seam between the two. Without a pyramid this frame there is no screen trace,
// and every glossy pixel traces. The reflection blur + composite then blends
// the result over the scene, exactly as they do for the SSR resolve.
//
// The traversal itself is the shared RT_TRACE fragment, so this pass and glass
// run one traversal loop between them; only the bindings and the miss fallback
// are the pass's own.
//
// CN_BACKEND_DIRECTX pins every register to the root signature in
// `directx/post/rt_reflections.rs` and keeps DirectX's raw vertex / index SRVs
// (`ByteAddressBuffer`); the other targets carry one declaration with both a
// `[[vk::binding]]` and a `register()`, whose number is the Metal index in the
// namespace the resource's kind implies. The Metal argument buffer holding the
// bindless pool rides a descriptor set of its own, named by
// `[[cn::metal_argument_buffer(n)]]`, because no per-resource register can name
// a set. The DXIL container needs shader model 6.5 for ray query, above the 6.0
// floor the bindless main pass established.

{POST_COMMON}

{TEXTURE_SIZE}

{PROBE_TYPES}

{RT_TYPES}

{CLUSTER_TYPES}

// A world direction in the sky cube's own frame; every sky tap goes through it,
// so a missed ray lands on the same oriented sky the main pass shows.
// What a tap of the environment prefilter cube is multiplied by: the inverse
// of the scale it is stored at, carried as a base-2 log in row 0's w.
#define SKY_CUBE_READ exp2(-rt_params.sky_rot[0].w)
#define SKY_DIR(d) float3(dot(rt_params.sky_rot[0].xyz, (d)), \
                          dot(rt_params.sky_rot[1].xyz, (d)), \
                          dot(rt_params.sky_rot[2].xyz, (d)))

// ---- Resource bindings ----

#ifdef CN_BACKEND_DIRECTX

ConstantBuffer<RtParams> rt_params : register(b0);
RWTexture2D<float4> output : register(u0);
#ifdef RT_CLASSIFY
Texture2D<float4> pyramid : register(t14);
RWStructuredBuffer<uint> ray_list : register(u1);
RWStructuredBuffer<uint> ray_args : register(u2);
#else
StructuredBuffer<uint> ray_list : register(t15);
StructuredBuffer<uint> ray_args : register(t16);
RaytracingAccelerationStructure scene_tlas : register(t0);
// Raw vertex / index SRVs: DirectX binds these as byte-address views.
ByteAddressBuffer verts : register(t1);
ByteAddressBuffer indices : register(t2);
StructuredBuffer<RtGeomEntry> geom : register(t3);
ByteAddressBuffer sverts : register(t8);
ByteAddressBuffer sidx : register(t9);
#endif

Texture2D<float4> scene_tex : register(t4);
Texture2D<float4> gbuffer : register(t5);
Texture2D<float4> rough_tex : register(t6);
TextureCube<float4> prefilter : register(t7);
// Static samplers from the root signature, not one per source, so the
// per-source names the other branch declares are aliases here and a sample
// site reads the same in both.
SamplerState screen_sampler : register(s0);
SamplerState cube_sampler : register(s1);
#define scene_tex_sampler screen_sampler
#define gbuffer_sampler screen_sampler
#define rough_tex_sampler screen_sampler
#define prefilter_sampler cube_sampler

#else

[[vk::binding(0, 0)]] ConstantBuffer<RtParams> rt_params : register(b0);
// The trace target: reflected radiance + composite weight.
[[vk::binding(17, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> output : register(u6);
#ifdef RT_CLASSIFY
// SSGI's depth pyramid for this frame, every level: the closest (r) and
// farthest (g) linear depth of each texel's footprint. Read by load only.
[[vk::binding(16, 0)]] Texture2D<float4> pyramid : register(t5);
// The pixels owed a ray, packed (y << 16) | x, and the trace's indirect
// dispatch arguments: groups in x, y and z, then the listed pixel count.
[[vk::binding(18, 0)]] RWStructuredBuffer<uint> ray_list : register(u14);
[[vk::binding(19, 0)]] RWStructuredBuffer<uint> ray_args : register(u15);
#else
[[vk::binding(18, 0)]] StructuredBuffer<uint> ray_list : register(t14);
[[vk::binding(19, 0)]] StructuredBuffer<uint> ray_args : register(t15);
[[vk::binding(1, 0)]] RaytracingAccelerationStructure scene_tlas : register(t4);
[[vk::binding(2, 0)]] StructuredBuffer<RtGeomEntry> geom : register(t3);
// The shared static vertex stream (14 floats / 56 B stride) + its u32 indices.
[[vk::binding(3, 0)]] StructuredBuffer<float> verts : register(t1);
[[vk::binding(4, 0)]] StructuredBuffer<uint> indices : register(t2);
// The deformed (posed) skinned vertex buffer, in the same 14-float layout the
// skin kernel writes, + the skinned index buffer (two indices per uint).
// Both bind a 1-element dummy in a scene with no skinned geometry, so the
// binding stays valid even though the skinned branch is never taken there.
[[vk::binding(9, 0)]] StructuredBuffer<float> sverts : register(t5);
[[vk::binding(10, 0)]] StructuredBuffer<uint> sidx : register(t6);
#endif

// Screen-space inputs reused from the SSR resolve, each a texture and a sampler
// of its own. Their `t` registers are Metal texture indices -- scene(0),
// gbuffer(1), roughness(2), prefilter(3) -- a different namespace from the
// buffers above, which is why the numbers repeat without colliding.
[[vk::binding(5, 0)]] Texture2D<float4> scene_tex : register(t0);
[[vk::binding(11, 0)]] SamplerState scene_tex_sampler : register(s0);
[[vk::binding(6, 0)]] Texture2D<float4> gbuffer : register(t1);
[[vk::binding(12, 0)]] SamplerState gbuffer_sampler : register(s1);
[[vk::binding(7, 0)]] Texture2D<float4> rough_tex : register(t2);
[[vk::binding(13, 0)]] SamplerState rough_tex_sampler : register(s2);
#ifdef CN_BACKEND_VULKAN
// The global set's own prefilter cube and cube sampler, the one the probe cubes
// are read through as well.
[[vk::binding(5, 1)]] TextureCube<float4> prefilter : register(t3);
[[vk::binding(19, 1)]] SamplerState prefilter_sampler : register(s3);
#else
[[vk::binding(8, 0)]] TextureCube<float4> prefilter : register(t3);
[[vk::binding(14, 0)]] SamplerState prefilter_sampler : register(s3);
#endif

#endif

float3 prefilter_level(float3 dir, float lod)
{
    return prefilter.SampleLevel(prefilter_sampler, SKY_DIR(dir), lod).rgb * SKY_CUBE_READ;
}

// The forward global set, bound here only for its reflection probes: a ray
// that escapes the scene falls back to the local probe capture instead of the
// foreign sky cube, blending the probes the main camera's cluster grid bins at
// the pixel's surface point. On Vulkan every binding is the global set's own,
// the cube sampler included.
#ifdef CN_BACKEND_DIRECTX
ConstantBuffer<ProbeSet> probe_set : register(b4);
TextureCubeArray<float4> probe_cubes : register(t10);
SamplerState probe_cube_sampler : register(s3);
StructuredBuffer<ProbeUniforms> probe_records : register(t11);
ConstantBuffer<ClusterParams> cluster : register(b5);
StructuredBuffer<uint> cluster_list : register(t12);
#else
[[vk::binding(7, 1)]] ConstantBuffer<ProbeSet> probe_set : register(b8);
[[vk::binding(8, 1)]] TextureCubeArray<float4> probe_cubes : register(t4);
[[vk::binding(19, 1)]] SamplerState probe_cube_sampler : register(s4);
[[vk::binding(17, 1)]] StructuredBuffer<ProbeUniforms> probe_records : register(t11);
[[vk::binding(10, 1)]] ConstantBuffer<ClusterParams> cluster : register(b9);
[[vk::binding(11, 1)]] StructuredBuffer<uint> cluster_list : register(t10);
#endif
#define PROBE_SET probe_set
#define PROBE_RECORDS probe_records
#define PROBE_COMPUTE
#define CLUSTER cluster
#define CLUSTER_LIST cluster_list

#ifdef RT_TEXTURED
// The bindless albedo / normal / emissive pool, in whichever form its host can
// bind: the bindless main pass's Metal argument buffer, bound at the pool's
// offset so the unsized array starts at its first texture (its sampler is bound
// alongside rather than written into the buffer), the Vulkan image array on the
// set that pass uses, read through the global set's linear sampler as the main
// pass reads it, or an unbounded DXIL array in space 1.
#if defined(CN_BACKEND_METAL)
[[vk::binding(0, 3)]] [[cn::metal_argument_buffer(7)]]
Texture2D<float4> tex_pool[] : register(t0, space3);
[[vk::binding(15, 0)]] SamplerState pool_sampler : register(s5);
#elif defined(CN_BACKEND_DIRECTX)
Texture2D<float4> tex_pool[] : register(t0, space1);
SamplerState pool_sampler : register(s2);
#else
[[vk::binding(1, 2)]] Texture2D<float4> tex_pool[] : register(t0, space1);
[[vk::binding(20, 1)]] SamplerState pool_sampler : register(s2);
#endif

// The far field's canopy block and view tables, which a hit on ground growing
// grass is shaded from: the main pass's own on Vulkan, in the global set.
{GROUND_RULE_TYPES}

{CANOPY_TYPES}

#if defined(CN_BACKEND_DIRECTX)
ConstantBuffer<CanopyParams> canopy_cb : register(b6);
StructuredBuffer<float4> canopy_tables_sb : register(t13);
#else
[[vk::binding(21, 1)]] ConstantBuffer<CanopyParams> canopy_cb : register(b12);
[[vk::binding(22, 1)]] StructuredBuffer<float4> canopy_tables_sb : register(t13);
#endif
#define CANOPY canopy_cb
#define CANOPY_TABLES canopy_tables_sb

// The blade model's two shading helpers, which the main pass defines for it.
static const float PI = 3.14159265358979;
float3 fresnel_schlick(float cos_theta, float3 f0)
{
    return f0 + (1.0 - f0) * pow(saturate(1.0 - cos_theta), 5.0);
}

// The ground's maps, read at the base mip as every other hit texture is.
float4 pool_sample(uint idx, float2 uv)
{
    return tex_pool[NonUniformResourceIndex(idx)].SampleLevel(pool_sampler, uv, 0.0);
}
float4 pool_sample_grad(uint idx, float2 uv, float2 dx, float2 dy)
{
    return pool_sample(idx, uv);
}

{CANOPY}

{GROUND_SURFACE}

#define RT_CANOPY
// How wide a reflected ray's footprint grows per meter it travels, which the
// meadow's detail fades by.
static const float RT_CANOPY_SPREAD = 0.01;
// A distance past every hand-off band, so a hit always shows the canopy.
static const float RT_CANOPY_PAST_BLADES = 1e6;
#endif

// Surfaces rougher than REFLECTION_ROUGHNESS_CUT get no reflection; glossiness
// ramps in below it.
{REFLECTION_CUT}

{PROBE_COMMON}

#ifndef RT_CLASSIFY
{RT_TRACE}
#endif

// Rebuild a view-space position from a UV and its linear (view-space) depth.
// Matches ssr_view_pos in the SSR resolve.
float3 rt_view_pos(float2 uv, float depth, float tan_y, float aspect)
{
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return float3(ndc.x * tan_y * aspect, ndc.y * tan_y, -1.0) * depth;
}

// What a ray that finds no surface reflects, in absolute radiance: the local
// reflection probe (box-parallax, blended across covering probes) when one is
// baked, fading far outside every probe box to what a world without probes
// shows: the IBL prefilter sky, else the base shading.
float3 rt_environment(float2 uv, float3 pw, float3 dir, float roughness, float3 base)
{
    float3 probe = (float3)(0.0);
    float sky_share = 1.0;
    if (probe_set.count > 0u)
    {
        probe = probe_mask_specular(probe_mask_at(uv, pw), pw, dir, probe_lod(roughness), sky_share);
    }
    if (sky_share <= 0.0)
    {
        return probe;
    }
    float3 sky = rt_params.prefilter_mip_count > 0.5
        ? prefilter_level(dir, roughness * (rt_params.prefilter_mip_count - 1.0))
        : base / rt_params.pre_exposure;
    return lerp(probe, sky, sky_share);
}

// One glossy G-buffer pixel, as both passes rebuild it.
struct RtSurface
{
    float2 uv;
    float3 base;
    float roughness;
    float3 f0;
    float gloss;
    float ndv;
    // View space: the point, its normal and the reflected direction.
    float3 pv;
    float3 nv;
    float3 dir_v;
    // World space.
    float3 pw;
    float3 nw;
    float3 dir_w;
};

// The G-buffer pixel the trace texel at `uv` stands for, or false where it
// takes no reflection: background / sky, or a surface past the roughness cut.
bool rt_surface(float2 uv, out RtSurface s)
{
    s = (RtSurface)0;
    // At a reduced trace resolution each output texel stands for a block of
    // G-buffer pixels; trace from exactly one of them so the surface point is
    // real rather than a blend across an edge.
    s.uv = reflection_source_uv(uv, texture_size(gbuffer));
    s.base = scene_tex.SampleLevel(scene_tex_sampler, s.uv, 0.0).rgb;
    float4 g = gbuffer.SampleLevel(gbuffer_sampler, s.uv, 0.0);
    if (g.a <= 0.0)
    {
        return false;
    }
    float4 spec = rough_tex.SampleLevel(rough_tex_sampler, s.uv, 0.0);
    s.roughness = spec.r;
    s.f0 = spec.gba;
    s.gloss = saturate((REFLECTION_ROUGHNESS_CUT - s.roughness) / REFLECTION_ROUGHNESS_CUT);
    if (s.gloss <= 0.0)
    {
        return false;
    }
    s.nv = normalize(g.xyz);
    s.pv = rt_view_pos(s.uv, g.a, rt_params.tan_half_fov_y, rt_params.aspect);
    float3 v = normalize(-s.pv);
    s.dir_v = reflect(-v, s.nv);
    s.ndv = saturate(dot(s.nv, v));
    s.pw = mul(rt_params.inv_view, float4(s.pv, 1.0)).xyz;
    s.nw = normalize(mul((float3x3)rt_params.inv_view, s.nv));
    s.dir_w = normalize(mul((float3x3)rt_params.inv_view, s.dir_v));
    return true;
}

float4 rt_resolve_out(RtSurface s, float3 reflected)
{
    return reflection_resolve_out(reflected, s.f0, s.ndv, s.gloss, rt_params.intensity);
}

// Threads per side of a classify group, and per trace group. Locked to the
// dispatch sizes in concinnity_engine::render::post::rt_reflections by unit test.
static const uint RT_CLASSIFY_GROUP = 8u;
static const uint RT_TRACE_GROUP = 64u;
// Trace groups an indirect dispatch may launch along x on every API; a longer
// list is strided over.
static const uint RT_TRACE_MAX_GROUPS = 65535u;

#ifdef RT_CLASSIFY

// Levels in this frame's pyramid; 0 when SSGI built none this frame.
#define RT_SCREEN_LEVELS uint(rt_params.cam_pos.w)

// Roughness at or below which a reflection the screen cannot settle is owed
// wholly to a hardware ray, and at or above which it falls back to the probe
// alone: a rougher reflection is blurred enough that the probe stands in.
static const float RT_MIRROR_ROUGHNESS = 0.2;
static const float RT_RAY_ROUGHNESS = 0.35;
// Pyramid steps a screen ray may take before it gives up.
static const uint RT_SCREEN_ITERATIONS = 64u;
// How far behind a surface, per meter of its view depth and at least, a screen
// ray still counts as reaching it.
static const float RT_SCREEN_THICKNESS = 0.02;
static const float RT_SCREEN_MIN_THICKNESS = 0.03;
// Closest view depth a screen ray may reach before it is cut off at the camera.
static const float RT_SCREEN_NEAR = 0.05;
// UV margin over which a hit near the screen border hands over to the ray.
static const float RT_SCREEN_EDGE = 0.08;
// A share owed to the ray below which the pixel is settled without one.
static const float RT_SETTLED = 1.0 / 256.0;

float rt_thickness(float depth)
{
    return max(depth * RT_SCREEN_THICKNESS, RT_SCREEN_MIN_THICKNESS);
}

// Project a view-space point (z < 0, in front of the camera) to a screen UV.
float2 rt_project(float3 q)
{
    float inv = 1.0 / max(-q.z, 1e-4);
    float2 ndc = float2(q.x * inv / (rt_params.tan_half_fov_y * rt_params.aspect),
                        q.y * inv / rt_params.tan_half_fov_y);
    return float2(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
}

// Where a ray leaving `start` along `dir` crosses out of the `size`-wide cell
// `cell`, in the ray's own parameter.
float rt_cell_exit(float2 start, float2 dir, float2 cell, float size)
{
    float2 boundary = (cell + select(dir > 0.0, (float2)(1.0), (float2)(0.0))) * size;
    float2 t = select(dir != 0.0, (boundary - start) / dir, (float2)(1.0e20));
    return min(t.x, t.y);
}

// Walk a view-space ray through the pyramid to the first surface it reaches,
// with its inverse depth interpolated linearly in level-0 pixels. A cell whose
// depth range the ray's span passes wholly in front of, or behind by more than
// the thickness, is skipped and the walk climbs a level; one it may touch is
// descended into. Unlike SSGI's bounce rays a mirror needs the exact surface,
// so every hit is judged at level 0 against the G-buffer depth under the ray.
// `behind` is how far behind that surface the ray was, over the thickness.
bool rt_screen_trace(float3 origin, float3 dir, uint levels, out float2 hit_uv, out float behind)
{
    hit_uv = (float2)(0.0);
    behind = 0.0;
    if (origin.z > -RT_SCREEN_NEAR)
    {
        return false;
    }
    float3 end = origin + dir * rt_params.max_distance;
    if (end.z > -RT_SCREEN_NEAR)
    {
        end = lerp(origin, end, (-RT_SCREEN_NEAR - origin.z) / (end.z - origin.z));
    }
    float2 low = texture_size(pyramid);
    float2 full = texture_size(gbuffer);
    float2 start = rt_project(origin) * low;
    float2 delta = rt_project(end) * low - start;
    float len = length(delta);
    if (len < 0.5)
    {
        return false;
    }
    float w0 = 1.0 / -origin.z;
    float w1 = 1.0 / -end.z;

    // Stop where the ray leaves the screen.
    float2 edge = select(delta > 0.0, low, (float2)(0.0));
    float2 t_edge = select(delta != 0.0, (edge - start) / delta, (float2)(1.0e20));
    float t_max = min(1.0, min(t_edge.x, t_edge.y));

    // A hundredth of a pixel, so a step lands inside the next cell.
    float eps = 0.01 / len;
    float t = rt_cell_exit(start, delta, floor(start), 1.0) + eps;
    int2 low_i = int2(low);
    int2 full_i = int2(full);
    float2 to_full = full / low;
    uint top = max(levels, 1u) - 1u;
    uint level = 0u;
    for (uint i = 0u; i < RT_SCREEN_ITERATIONS && t < t_max; i++)
    {
        float2 pos = start + delta * t;
        float size = float(1u << level);
        float2 cell = floor(pos / size);
        int2 level_size = max(low_i >> level, 1);
        float2 range = pyramid.Load(int3(min(int2(cell), level_size - 1), int(level))).rg;
        float t_exit = min(rt_cell_exit(start, delta, cell, size), t_max);
        float depth_in = 1.0 / lerp(w0, w1, t);
        float depth_out = 1.0 / lerp(w0, w1, t_exit);
        float near_d = min(depth_in, depth_out);
        float far_d = max(depth_in, depth_out);
        if (far_d < range.x || near_d > range.y + rt_thickness(range.y))
        {
            t = t_exit + eps;
            level = min(level + 1u, top);
        }
        else if (level > 0u)
        {
            level--;
        }
        else
        {
            int2 under = min(int2(pos * to_full), full_i - 1);
            float actual = gbuffer.Load(int3(under, 0)).a;
            float thickness = rt_thickness(actual);
            if (actual > 0.0 && far_d >= actual && near_d <= actual + thickness)
            {
                hit_uv = (float2(under) + 0.5) / full;
                behind = saturate((near_d - actual) / thickness);
                return true;
            }
            t = t_exit + eps;
        }
    }
    return false;
}

// How far a screen hit at `hit_uv` can be trusted as what a ray along `dir`
// reflects: not near the screen border, where the true reflection may lie just
// off screen; not a surface seen from behind; not one the ray only passed
// close behind.
float rt_screen_confidence(float2 hit_uv, float3 dir, float behind)
{
    float2 e = smoothstep((float2)(0.0), (float2)(RT_SCREEN_EDGE), hit_uv)
             * smoothstep((float2)(0.0), (float2)(RT_SCREEN_EDGE), (float2)(1.0) - hit_uv);
    float3 n = normalize(gbuffer.SampleLevel(gbuffer_sampler, hit_uv, 0.0).xyz);
    float facing = saturate(-dot(n, dir) * 4.0);
    return e.x * e.y * facing * (1.0 - behind * behind);
}

groupshared uint gs_owed;
groupshared uint gs_base;

// What `p`'s reflection settles to on screen, written to the target, and
// whether it still owes a share to a hardware ray.
bool rt_classify(uint2 p, float2 size)
{
    RtSurface s;
    if (!rt_surface((float2(p) + 0.5) / size, s))
    {
        // The pass writes reflected radiance (.rgb) + composite weight (.a),
        // not a blended color: weight 0 keeps the scene here.
        output[p] = float4(s.base, 0.0);
        return false;
    }

    uint levels = RT_SCREEN_LEVELS;
    float confidence = 0.0;
    float3 screen = (float3)(0.0);
    if (levels > 0u)
    {
        float2 hit_uv;
        float behind;
        float3 origin = s.pv + s.nv * (rt_thickness(-s.pv.z) * 0.25);
        if (rt_screen_trace(origin, s.dir_v, levels, hit_uv, behind))
        {
            confidence = rt_screen_confidence(hit_uv, s.dir_v, behind);
            screen = scene_tex.SampleLevel(scene_tex_sampler, hit_uv, 0.0).rgb;
        }
    }
    // Without a screen trace there is no screen answer to fall back on, so
    // every glossy pixel is owed to the ray.
    float ray_share = levels > 0u
        ? 1.0 - smoothstep(RT_MIRROR_ROUGHNESS, RT_RAY_ROUGHNESS, s.roughness)
        : 1.0;
    float owed = (1.0 - confidence) * ray_share;
    float env_share = (1.0 - confidence) * (1.0 - ray_share);
    float3 env = env_share > 0.0
        ? rt_environment(s.uv, s.pw, s.dir_w, s.roughness, s.base) * rt_params.pre_exposure
        : (float3)(0.0);
    float3 settled = screen * confidence + env * env_share;
    if (owed <= RT_SETTLED)
    {
        output[p] = rt_resolve_out(s, settled / (1.0 - owed));
        return false;
    }
    // The trace adds its share by the owed fraction in .a and writes the
    // composite weight over it.
    output[p] = float4(rt_resolve_out(s, settled).rgb, owed);
    return true;
}

[shader("compute")]
[numthreads(RT_CLASSIFY_GROUP, RT_CLASSIFY_GROUP, 1)]
void rt_reflections_classify_kernel(uint3 id : SV_DispatchThreadID, uint index : SV_GroupIndex)
{
    if (index == 0u)
    {
        gs_owed = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    uint w, h;
    output.GetDimensions(w, h);
    bool owes = false;
    uint slot = 0u;
    if (id.x < w && id.y < h)
    {
        owes = rt_classify(id.xy, float2(w, h));
        if (owes)
        {
            InterlockedAdd(gs_owed, 1u, slot);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // One global reservation per group keeps the list compact without a
    // device-wide atomic per pixel; the reservation's end grows the trace's
    // group count to cover it.
    if (index == 0u && gs_owed > 0u)
    {
        uint base;
        InterlockedAdd(ray_args[3], gs_owed, base);
        gs_base = base;
        uint groups = min((base + gs_owed + RT_TRACE_GROUP - 1u) / RT_TRACE_GROUP,
                          RT_TRACE_MAX_GROUPS);
        uint before;
        InterlockedMax(ray_args[0], groups, before);
    }
    GroupMemoryBarrierWithGroupSync();

    if (owes)
    {
        ray_list[gs_base + slot] = (id.y << 16) | id.x;
    }
}

#else

// Trace the listed pixel `p` of a `size` target and add its owed share.
void rt_trace_listed(uint2 p, float2 size)
{
    // Only pixels the classify listed reach here, so the surface is glossy.
    RtSurface s;
    if (!rt_surface((float2(p) + 0.5) / size, s))
    {
        return;
    }

    bool ibl = rt_params.prefilter_mip_count > 0.5;
    float max_mip = rt_params.prefilter_mip_count - 1.0;
    float3 reflected;
    // Origin nudged off the surface along the normal so the trace cannot
    // self-intersect the pixel's own triangle.
    if (!rt_trace_reflection(s.pw + s.nw * 0.01, s.dir_w, ibl, max_mip, true, reflected))
    {
        reflected = rt_environment(s.uv, s.pw, s.dir_w, s.roughness, s.base);
    }
    float4 traced = rt_resolve_out(s, reflected * rt_params.pre_exposure);
    float4 settled = output[p];
    output[p] = float4(settled.rgb + traced.rgb * settled.a, traced.a);
}

[shader("compute")]
[numthreads(RT_TRACE_GROUP, 1, 1)]
void rt_reflections_kernel(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    output.GetDimensions(w, h);
    uint count = ray_args[3];
    uint stride = ray_args[0] * RT_TRACE_GROUP;
    for (uint i = id.x; i < count; i += stride)
    {
        uint packed = ray_list[i];
        rt_trace_listed(uint2(packed & 0xFFFFu, packed >> 16), float2(w, h));
    }
}

#endif
