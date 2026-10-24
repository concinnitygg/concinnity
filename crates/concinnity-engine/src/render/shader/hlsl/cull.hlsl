// GPU-driven draw cull. One thread per draw record: tests the record's AABB
// against the frustum, its cull distance and the previous frame's Hi-Z pyramid,
// then writes either a real indirect draw or an instance_count-0 no-op into the
// command buffer the indirect draw reads.
//
// Three variants out of one entry point, selected by define:
//
//   (none)       phase 1: frustum + distance + Hi-Z, and records the outcome in
//                `cull_status` so phase 2 knows what to reconsider
//   CULL_PHASE2  re-tests only phase 1's Hi-Z candidates against the rebuilt
//                pyramid, for two-pass occlusion
//   SHADOW_CULL  light-frustum only, one dispatch per shadow view
//
// Single source for all three hosts. Every variant writes the outcome into
// `cull_status` and counts each 64-record group's draws per stream into
// `group_counts`; cull_compact.hlsl then scans the counts and, on Vulkan and
// DirectX, scatters every kept record's command into its stream's list. On
// Metal the hand-written `metal/shaders/cull_encode.metal` is the scatter,
// since its ICB encoding (`render_command`, `array<command_buffer, N>`) is a
// declaration HLSL cannot express.
//
// The backend define selects the host's constant shape, and here it is a
// difference in shape rather than in slot numbers, so all three arms stay. Vulkan hands the kernel
// a push constant for the frustum and bucket routing plus a set-1 uniform
// buffer for the Hi-Z reprojection, because the Hi-Z half changes on a
// different schedule and is owned by `vulkan/hiz.rs`. DirectX fuses both into
// one b0 root-constant block (directx/cull.rs) and Metal into one inline block
// at buffer(2) (metal/cull.rs), so the same fields arrive as one struct on
// either. The accessor defines below are what let the body read them without
// caring.
//
{OBJECT_COMMON}
{DEPTH_CONVENTION}
{CULL_COMMON}

// ---- Per-host constants ----

#if defined(CN_BACKEND_METAL)

// Layout matches `metal::uniforms::CullUniforms` (208 B), the inline constant
// block at buffer(2). `cam_pos` is a whole float4 lane (w unused): MSL sizes a
// constant-buffer float3 at 16 bytes, so a scalar after it would not pack.
struct CullParams
{
    float4   planes[6];
    float4   cam_pos;
    float4x4 prev_view_proj;
    float2   hiz_size;
    uint     hiz_mip_count;
    uint     hiz_enabled;
    uint     object_count;
    uint     skinned_base;
    uint     _pad;
    uint     bucket_count;
};

[[vk::push_constant]] ConstantBuffer<CullParams> cull : register(b2);

#define CULL_PREV_VP      cull.prev_view_proj
#define CULL_HIZ_SIZE     cull.hiz_size
#define CULL_HIZ_MIPS     cull.hiz_mip_count
#define CULL_HIZ_ENABLED  cull.hiz_enabled

#elif defined(CN_BACKEND_DIRECTX)

// Layout matches `directx::uniforms::CullParams` (208 B), the b0 root
// constants. `cam_pos` shares its 16-byte row with `object_count`.
struct CullParams
{
    float4   planes[6];
    float3   cam_pos;
    uint     object_count;
    float4x4 prev_view_proj;
    float2   hiz_size;
    uint     hiz_mip_count;
    uint     hiz_enabled;
    uint     bucket_count;
    uint     skinned_base;
    uint     _pad0;
    uint     _pad1;
};

ConstantBuffer<CullParams> cull : register(b0);

#define CULL_PREV_VP      cull.prev_view_proj
#define CULL_HIZ_SIZE     cull.hiz_size
#define CULL_HIZ_MIPS     cull.hiz_mip_count
#define CULL_HIZ_ENABLED  cull.hiz_enabled

#else

// Layout matches `vulkan::uniforms::CullParams` (120 B), the push constant.
struct CullParams
{
    float4 planes[6];
    float3 cam_pos;
    uint   object_count;
    uint   bucket_count;
    uint   skinned_base;
};

// Layout matches `vulkan::uniforms::CullHizParams` (80 B), set 1 binding 1.
struct CullHizParams
{
    float4x4 prev_view_proj;
    float2   hiz_size;
    uint     hiz_mip_count;
    uint     hiz_enabled;
};

[[vk::push_constant]] ConstantBuffer<CullParams> cull : register(b0);

#ifndef SHADOW_CULL
[[vk::binding(1, 1)]] ConstantBuffer<CullHizParams> hiz : register(b1, space1);
#endif

#define CULL_PREV_VP      hiz.prev_view_proj
#define CULL_HIZ_SIZE     hiz.hiz_size
#define CULL_HIZ_MIPS     hiz.hiz_mip_count
#define CULL_HIZ_ENABLED  hiz.hiz_enabled

#endif

// ---- Buffers ----
//
// Every register is pinned rather than left to declaration order, because the
// order here is not the root signature's: `cull_status` has to be declared
// before `DrawCommand` (the shadow variant declares no status, and the struct
// differs per host), which would otherwise hand it u0 and give `commands` u1 --
// the reverse of what directx/cull.rs binds. The `[[vk::binding]]`s spell
// Vulkan's own set/binding numbers on the same declarations.

[[vk::binding(0, 0)]] StructuredBuffer<GpuObjectData> objects : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<GpuDrawArgs> draw_args : register(t1);

#ifndef SHADOW_CULL

#if defined(CN_BACKEND_METAL)
[[vk::binding(4, 0)]] ConstantBuffer<CullVariantRows> variants : register(b8);
#define CULL_VARIANTS variants.rows
#else
[[vk::binding(4, 0)]] StructuredBuffer<uint4> variants : register(t3);
#define CULL_VARIANTS variants
#endif

#endif // !SHADOW_CULL

// Per-object outcome of the cull, which the packing reads, at buffer(5) on
// Metal. The phase-1 one is read again by phase 2; under single-pass
// occlusion that is simply ignored.
//
// The GPU-driven shadow cull is frustum-only -- sun cascades have no
// light-space depth pyramid, so no Hi-Z test and no two-pass occlusion -- and
// omits the variant block along with the whole set-1 Hi-Z block, which is what
// leaves its Vulkan pipeline layout the lean four-binding set its host
// creates.
#ifdef CN_BACKEND_METAL
#define CULL_STATUS_REGISTER u5
#else
#define CULL_STATUS_REGISTER u1
#endif

[[vk::binding(3, 0)]] RWStructuredBuffer<uint> cull_status : register(CULL_STATUS_REGISTER);

// Each group's draws per stream, stream by stream: the scan's input.
#ifdef CN_BACKEND_METAL
#define CULL_GROUP_COUNTS_REGISTER u9
#else
#define CULL_GROUP_COUNTS_REGISTER u0
#endif
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> group_counts : register(CULL_GROUP_COUNTS_REGISTER);

#ifndef SHADOW_CULL

// The previous frame's depth pyramid, read by texel coordinate only and never
// sampled. DirectX binds a plain SRV through the root signature's descriptor
// table; Metal and Vulkan a plain texture, at the index the register number
// names on Metal, which is its own namespace and so does not collide with the
// buffer `objects` holds.
#ifdef CN_BACKEND_DIRECTX
Texture2D<float> hiz_tex : register(t2);
#else
[[vk::binding(0, 1)]] Texture2D<float> hiz_tex : register(t0, space1);
#endif

#endif // !SHADOW_CULL

// AABB entirely behind any plane -> outside the frustum. Negation of
// gfx::frustum::Frustum::intersects_aabb (the p-vertex test).
bool frustum_culled(float3 bb_min, float3 bb_max)
{
    [unroll] for (uint i = 0u; i < 6u; ++i)
    {
        float3 n = cull.planes[i].xyz;
        float3 farthest = float3(
            n.x >= 0.0 ? bb_max.x : bb_min.x,
            n.y >= 0.0 ? bb_max.y : bb_min.y,
            n.z >= 0.0 ? bb_max.z : bb_min.z);
        if (dot(n, farthest) + cull.planes[i].w < 0.0)
        {
            return true;
        }
    }
    return false;
}

// Squared distance from the camera to the closest point on the AABB; 0 when the
// camera is inside. Mirrors gfx::frustum::aabb_distance_sq.
float aabb_distance_sq(float3 bb_min, float3 bb_max)
{
    float3 d = max(max(bb_min - cull.cam_pos.xyz, cull.cam_pos.xyz - bb_max), float3(0.0, 0.0, 0.0));
    return dot(d, d);
}

#ifndef SHADOW_CULL

// Project the eight AABB corners through the reprojection matrix and reduce to
// a screen-space rect (NDC.xy in [-1, 1]) plus the AABB's nearest NDC depth.
// Returns false if any corner ended up behind the camera (w <= 0), in which
// case the caller conservatively keeps the object.
bool project_aabb(
    float3 bb_min,
    float3 bb_max,
    out float2 ndc_min,
    out float2 ndc_max,
    out float nearest_depth)
{
    ndc_min = float2( 1.0,  1.0);
    ndc_max = float2(-1.0, -1.0);
    nearest_depth = DEPTH_FAR;
    [unroll] for (uint i = 0u; i < 8u; ++i)
    {
        float3 corner = float3(
            (i & 1u) != 0u ? bb_max.x : bb_min.x,
            (i & 2u) != 0u ? bb_max.y : bb_min.y,
            (i & 4u) != 0u ? bb_max.z : bb_min.z);
        float4 clip = mul(CULL_PREV_VP, float4(corner, 1.0));
        if (clip.w <= 0.0)
        {
            return false;
        }
        float3 ndc = clip.xyz / clip.w;
        ndc_min = min(ndc_min, ndc.xy);
        ndc_max = max(ndc_max, ndc.xy);
        nearest_depth = depth_closer(nearest_depth, ndc.z);
    }
    return true;
}

// The pyramid half of the test reads texels through `hiz_load`.
float hiz_load(int3 texel_mip)
{
    return hiz_tex.Load(texel_mip);
}

{HIZ_TEST}

// True when the AABB is fully occluded by the Hi-Z pyramid. Conservative: any
// uncertain case returns false (keep the object). The NDC y-flip matches the
// main pass's viewport, and the pyramid's texels store the farthest occluder.
bool hiz_occluded(float3 bb_min, float3 bb_max)
{
    float2 ndc_min, ndc_max;
    float aabb_nearest_depth;
    if (!project_aabb(bb_min, bb_max, ndc_min, ndc_max, aabb_nearest_depth))
    {
        return false;
    }
    // Clip to NDC bounds so the UV math stays sane (an AABB straddling the
    // viewport on both sides of an axis is already kept by the frustum test).
    ndc_min = max(ndc_min, float2(-1.0, -1.0));
    ndc_max = min(ndc_max, float2( 1.0,  1.0));
    if (any(ndc_min > ndc_max))
    {
        return false;
    }
    return hiz_rect_occluded(ndc_min, ndc_max, aabb_nearest_depth, CULL_HIZ_SIZE, CULL_HIZ_MIPS);
}

// Whether the farthest corner of the box lies past `start` from the eye.
bool reaches_far(float3 bb_min, float3 bb_max, float start)
{
    float3 d = max(abs(cull.cam_pos.xyz - bb_min), abs(bb_max - cull.cam_pos.xyz));
    return dot(d, d) > start * start;
}

// Whether a planet's own sky may reach any point of the box: the camera is
// high enough for it to reach everywhere, or a corner lies far enough round
// the planet. The corners suffice, since the points the sky does not reach
// form a convex cone. Mirrors GpuCullVariants::daylight_reaches.
bool daylight_reaches(float3 bb_min, float3 bb_max)
{
    float4 center = asfloat(CULL_VARIANTS[1]);
    float4 camera_up = asfloat(CULL_VARIANTS[2]);
    if (camera_up.w > 0.0)
    {
        return true;
    }
    float near_cos = asfloat(CULL_VARIANTS[3].x) + DAYLIGHT_COS_MARGIN;
    [unroll] for (uint i = 0u; i < 8u; ++i)
    {
        float3 corner = float3(
            (i & 1u) != 0u ? bb_max.x : bb_min.x,
            (i & 2u) != 0u ? bb_max.y : bb_min.y,
            (i & 4u) != 0u ? bb_max.z : bb_min.z);
        float3 d = corner - center.xyz;
        float len = length(d);
        if (len <= 0.0 || dot(d, camera_up.xyz) / len < near_cos)
        {
            return true;
        }
    }
    return false;
}

// The features a record has among those `route` tests. Mirrors
// GpuCullVariants::features.
uint variant_features(uint4 route, GpuDrawArgs a, uint i)
{
    bool bounded = (a.flags & DRAW_CULLABLE) != 0u;
    float3 bb_min = (float3)(0.0);
    float3 bb_max = (float3)(0.0);
    if (bounded)
    {
        GpuObjectData obj = objects[i];
        bb_min = obj.bb_min_cull_distance.xyz;
        bb_max = obj.bb_max_alpha_cutoff.xyz;
    }
    uint features = 0u;
    if ((route.y & VARIANT_FAR) != 0u && (a.flags & DRAW_FAR_BAND) != 0u
        && (!bounded || reaches_far(bb_min, bb_max, asfloat(CULL_VARIANTS[0].x))))
    {
        features |= VARIANT_FAR;
    }
    if ((route.y & VARIANT_DAYLIGHT) != 0u && asfloat(CULL_VARIANTS[1].w) > 0.5
        && (!bounded || daylight_reaches(bb_min, bb_max)))
    {
        features |= VARIANT_DAYLIGHT;
    }
    return features;
}

// The shader bucket record `i` draws under: its route's pick among its
// Shader's variants, or its base bucket. Mirrors GpuCullVariants::bucket.
uint record_bucket(GpuDrawArgs a, uint i)
{
    uint route = (a.flags >> DRAW_ROUTE_SHIFT) & DRAW_ROUTE_MASK;
    if (cull.bucket_count <= 1u)
    {
        return 0u;
    }
    if (route == 0u || route > MAX_VARIANT_ROUTES)
    {
        return min((a.flags >> DRAW_BUCKET_SHIFT) & DRAW_BUCKET_MASK, cull.bucket_count - 1u);
    }
    uint4 r = CULL_VARIANTS[CULL_VARIANT_ROUTE_ROW + route - 1u];
    uint features = variant_features(r, a, i);
    return min((r.x >> (8u * features)) & 0xFFu, cull.bucket_count - 1u);
}

// The stream record `i`, kept with status `word`, draws in.
uint kept_stream(uint i, uint word)
{
    uint bucket = min(word >> CULL_BUCKET_SHIFT, MAX_SHADER_BUCKETS - 1u);
    return record_stream(i, bucket, cull.bucket_count, cull.skinned_base,
        CULL_REGION_SPANS(CULL_VARIANTS, bucket));
}

#endif // !SHADOW_CULL

groupshared uint group_stream_counts[DRAW_STREAMS];

// Count group `group`'s kept records per stream into `group_counts`, every
// stream the scan walks written so no stale one survives. Each thread passes
// the stream its record draws in, NO_STREAM when it draws in none.
void count_streams(uint gi, uint group, uint stream)
{
    for (uint s = gi; s < DRAW_STREAMS; s += CULL_GROUP_RECORDS)
    {
        group_stream_counts[s] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    if (stream != NO_STREAM)
    {
        InterlockedAdd(group_stream_counts[stream], 1u);
    }
    GroupMemoryBarrierWithGroupSync();
    uint groups = (cull.object_count + CULL_GROUP_RECORDS - 1u) / CULL_GROUP_RECORDS;
    // One thread a bucket, at most one bucket a thread, and the first the
    // skinned tail too.
    if (gi < cull.bucket_count)
    {
        group_counts[gi * groups + group] = group_stream_counts[gi];
    }
    if (gi == 0u)
    {
        group_counts[SKINNED_STREAM * groups + group] = group_stream_counts[SKINNED_STREAM];
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cull_kernel(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    uint i = tid.x;

    // Every thread of the group reaches the count, past the last record too.
    uint stream = NO_STREAM;
    if (i < cull.object_count)
    {
#if defined(SHADOW_CULL)
        GpuDrawArgs a = draw_args[i];
        // Light-frustum only, against this view's `planes`. Deliberately NO
        // per-object distance cull: the cascade light frustum already bounds the
        // shadow draw distance through its extents, and the per-object view
        // `cull_distance` is a view-LOD-fade concept that must not silence shadows
        // (an off-screen caster beyond cull_distance can still cast into the
        // visible scene). One depth-only stream and the skinned tail.
        bool draw = (a.flags & DRAW_ENABLED) != 0u;
        if (draw && (a.flags & DRAW_CULLABLE) != 0u)
        {
            GpuObjectData obj = objects[i];
            draw = !frustum_culled(obj.bb_min_cull_distance.xyz, obj.bb_max_alpha_cutoff.xyz);
        }
        cull_status[i] = draw ? STATUS_DRAWN : STATUS_CULLED;
        if (draw)
        {
            stream = i >= cull.skinned_base ? SKINNED_STREAM : 0u;
        }
#elif defined(CULL_PHASE2)
        // Runs after the pyramid was rebuilt from this frame's phase-1 depth.
        // Re-tests only the objects phase 1 marked STATUS_HIZ_CANDIDATE against
        // the fresh pyramid (projected through this frame's VP, carried in
        // `prev_view_proj` exactly as phase 1 used the previous frame's). A
        // candidate still occluded by this frame's actual depth stays culled;
        // one now visible is redrawn. Everything else phase 1 settled and is
        // left as it was.
        uint word = cull_status[i];
        if ((word & CULL_OUTCOME_MASK) == STATUS_HIZ_CANDIDATE)
        {
            GpuObjectData obj = objects[i];
            bool redraw = CULL_HIZ_ENABLED == 0u
                || !hiz_occluded(obj.bb_min_cull_distance.xyz, obj.bb_max_alpha_cutoff.xyz);
            cull_status[i] = (redraw ? STATUS_REDRAW : STATUS_HIZ_CULLED)
                | (word & ~CULL_OUTCOME_MASK);
            if (redraw)
            {
                stream = kept_stream(i, word);
            }
        }
#else
        // Phase 1. Records the outcome in `cull_status` for two-pass
        // occlusion: a Hi-Z cull is the only outcome phase 2 reconsiders
        // against the rebuilt pyramid; everything else is settled here.
        GpuDrawArgs a = draw_args[i];
        uint status = STATUS_DRAWN;
        if ((a.flags & DRAW_ENABLED) == 0u)
        {
            status = STATUS_CULLED;
        }
        else if ((a.flags & DRAW_CULLABLE) != 0u)
        {
            GpuObjectData obj = objects[i];
            float3 bb_min = obj.bb_min_cull_distance.xyz;
            float3 bb_max = obj.bb_max_alpha_cutoff.xyz;
            float cull_distance = obj.bb_min_cull_distance.w;
            if (frustum_culled(bb_min, bb_max))
            {
                status = STATUS_CULLED;
            }
            else if (cull_distance > 0.0
                && aabb_distance_sq(bb_min, bb_max) > cull_distance * cull_distance)
            {
                status = STATUS_CULLED;
            }
            else if (CULL_HIZ_ENABLED != 0u && hiz_occluded(bb_min, bb_max))
            {
                // Cull when the AABB is fully behind the previous frame's
                // depth pyramid. Skipped on the first frame and after a resize
                // (`hiz_enabled == 0`), where no valid pyramid exists yet.
                status = STATUS_HIZ_CANDIDATE;
            }
        }
        uint bucket = (a.flags & DRAW_ENABLED) != 0u ? record_bucket(a, i) : 0u;
        uint word = status | (bucket << CULL_BUCKET_SHIFT);
        cull_status[i] = word;
        if (status == STATUS_DRAWN)
        {
            stream = kept_stream(i, word);
        }
#endif
    }
    count_streams(gi, gid.x, stream);
}
