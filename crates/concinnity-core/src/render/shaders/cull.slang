// GPU-driven draw cull. One thread per draw record: tests the record's AABB
// against the frustum, its cull distance and the previous frame's Hi-Z pyramid,
// then writes either a real indirect draw or an instance_count-0 no-op into the
// command buffer the indirect draw reads.
//
// Three variants out of one entry point, selected by define exactly as the
// hand-written sources were:
//
//   (none)       phase 1: frustum + distance + Hi-Z, and records the outcome in
//                `cull_status` so phase 2 knows what to reconsider
//   CULL_PHASE2  re-tests only phase 1's Hi-Z candidates against the rebuilt
//                pyramid, for two-pass occlusion
//   SHADOW_CULL  light-frustum only, one dispatch per cascade
//
// Single source for all three hosts. On Vulkan and DirectX the kernel writes
// the indirect command record itself; on Metal it writes only `cull_status`,
// and the hand-written `metal/shaders/cull_encode.metal` turns that status into
// the indirect command buffer. Metal has no multi-draw-indirect and its ICB
// encoding (`render_command`, `array<command_buffer, N>`) is a declaration
// Slang cannot express, but nothing in the decision needs it, so the decision
// is what lives here and the encoding is what stays per backend.
//
// DXIL_ABI and METAL_BINDINGS select a *host's* constant shape, a host
// difference rather than a target one. Vulkan hands the kernel a push constant for the
// frustum and bucket routing plus a set-1 uniform buffer for the Hi-Z
// reprojection, because the Hi-Z half changes on a different schedule and is
// owned by `vulkan/hiz.rs`. DirectX fuses both into one b0 root-constant block
// (directx/cull.rs), so the same fields arrive as one struct there. The
// accessor defines below are what let the body read them without caring.
//
// The command record itself genuinely differs and cannot be unified: a Vulkan
// indirect draw is a bare `VkDrawIndexedIndirectCommand` whose `first_instance`
// carries the object id, while a D3D12 ExecuteIndirect command signature
// prepends the b0 root constant the shaders read the object id from, so its
// record is a word longer and starts with that id.

{OBJECT_COMMON}

// Mirrors `GpuDrawArgs` in render_types.rs (16 B).
struct GpuDrawArgs
{
    uint index_count;
    uint index_offset;
    uint base_vertex;
    uint flags;
};

static const uint DRAW_ENABLED  = 1u;
static const uint DRAW_CULLABLE = 2u;

// The record's shader bucket rides the upper flag bits; values and layout are
// locked to gfx::render_types::{DrawArgsFlags::BUCKET_SHIFT, MAX_SHADER_BUCKETS}.
static const uint DRAW_BUCKET_SHIFT = 8u;
static const uint DRAW_BUCKET_MASK  = 0xFFu;

// `cull_status` values. STATUS_HIZ_CANDIDATE is the only outcome phase 2
// re-tests against the rebuilt pyramid; the others are settled by phase 1.
static const uint STATUS_DRAWN         = 0u; // visible in phase 1 -> never re-tested
static const uint STATUS_HIZ_CANDIDATE = 1u; // Hi-Z-occluded in phase 1 -> candidate
static const uint STATUS_CULLED        = 2u; // frustum/distance/disabled -> settled
static const uint STATUS_REDRAW        = 3u; // candidate found visible in phase 2

// ---- Per-host constants ----

#if defined(METAL_BINDINGS)

// Layout matches `metal::uniforms::CullUniforms` (208 B), the inline constant
// block at buffer(2). `cam_pos` is a whole float4 lane (w unused): MSL sizes a
// constant-buffer float3 at 16 bytes, so a scalar after it would not pack.
// `skinned_base` and `cascade_base` are read by the encode kernel's host, not
// here; they ride the one block both dispatches share.
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
    uint     cascade_base;
    uint     bucket_count;
};

[[vk::push_constant]] ConstantBuffer<CullParams> cull : register(b2);

#define CULL_PREV_VP      cull.prev_view_proj
#define CULL_HIZ_SIZE     cull.hiz_size
#define CULL_HIZ_MIPS     cull.hiz_mip_count
#define CULL_HIZ_ENABLED  cull.hiz_enabled

#elif defined(DXIL_ABI)

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
    uint     bucket_stride;
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
    uint   bucket_stride;
};

// Layout matches `vulkan::uniforms::CullHizParams` (80 B), set 1 binding 1.
struct CullHizParams
{
    float4x4 prev_view_proj;
    float2   hiz_size;
    uint     hiz_mip_count;
    uint     hiz_enabled;
};

[[vk::push_constant]] ConstantBuffer<CullParams> cull;

#ifndef SHADOW_CULL
[[vk::binding(1, 1)]] ConstantBuffer<CullHizParams> hiz;
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
// Vulkan's own set/binding numbers on the same declarations, which is also what
// keeps slangc from warning about a bare `register()`.

[[vk::binding(0, 0)]] StructuredBuffer<GpuObjectData> objects : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<GpuDrawArgs> draw_args : register(t1);

#ifdef METAL_BINDINGS

// Metal's one buffer index space: the numbers are the slots `metal/cull.rs`
// binds, shared with the encode kernel that reads the status back. The status
// is the kernel's only output here, so every variant declares it, the shadow
// one at the cascade's region of the shadow status buffer. The Hi-Z pyramid
// takes its texture index from declaration order (the only texture), which
// `assert_slang_metal_abi` locks to texture(0).
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> cull_status : register(u5);
#ifndef SHADOW_CULL
Texture2D<float> hiz_tex;
#endif

#else

// The GPU-driven shadow cull is frustum-only: sun cascades have no light-space
// depth pyramid, so there is no Hi-Z test and no two-pass occlusion. It omits
// the status buffer and the whole set-1 Hi-Z block, which is what leaves its
// Vulkan pipeline layout the lean three-binding set its host creates.
#ifndef SHADOW_CULL
// Per-object outcome of phase-1 cull. Always bound and written so phase 2 sees
// valid data; under single-pass occlusion the values are simply ignored.
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> cull_status : register(u1);
#endif

#ifdef DXIL_ABI

// One ExecuteIndirect command: the b0 object-id root constant followed by
// D3D12_DRAW_INDEXED_ARGUMENTS. 24 bytes; matches the command signature in
// directx/init/pipelines.rs.
struct DrawCommand
{
    uint object_id;
    uint index_count;
    uint instance_count;
    uint start_index;
    int  base_vertex;
    uint start_instance;
};

DrawCommand make_command(uint i, GpuDrawArgs a)
{
    DrawCommand c;
    c.object_id = i;
    c.index_count = a.index_count;
    c.instance_count = 1u;
    c.start_index = a.index_offset;
    c.base_vertex = int(a.base_vertex);
    c.start_instance = 0u;
    return c;
}

#else

// Matches VkDrawIndexedIndirectCommand (20 bytes). `first_instance` delivers
// the record id to the depth-only and bindless vertex stages, which read it
// back through `object_instance_index`.
struct DrawCommand
{
    uint index_count;
    uint instance_count;
    uint first_index;
    int  vertex_offset;
    uint first_instance;
};

DrawCommand make_command(uint i, GpuDrawArgs a)
{
    DrawCommand c;
    c.index_count = a.index_count;
    c.instance_count = 1u;
    c.first_index = a.index_offset;
    c.vertex_offset = int(a.base_vertex);
    c.first_instance = i;
    return c;
}

#endif

[[vk::binding(2, 0)]] RWStructuredBuffer<DrawCommand> commands : register(u0);

#ifndef SHADOW_CULL
// Read by texel coordinate only, never sampled. Vulkan's cull-read set binds a
// combined image sampler, so this is a `Sampler2D` there; DirectX binds a plain
// SRV through the root signature's descriptor table.
#ifdef DXIL_ABI
Texture2D<float> hiz_tex : register(t2);
#else
[[vk::binding(0, 1)]] Sampler2D<float> hiz_tex;
#endif
#endif

#endif // !METAL_BINDINGS

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
// a screen-space rect (NDC.xy in [-1, 1]) plus the AABB's closest NDC depth.
// Returns false if any corner ended up behind the camera (w <= 0), in which
// case the caller conservatively keeps the object.
bool project_aabb(
    float3 bb_min,
    float3 bb_max,
    out float2 ndc_min,
    out float2 ndc_max,
    out float min_depth)
{
    ndc_min = float2( 1.0,  1.0);
    ndc_max = float2(-1.0, -1.0);
    min_depth = 1.0;
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
        min_depth = min(min_depth, ndc.z);
    }
    return true;
}

// True when the AABB is fully occluded by the Hi-Z pyramid. Conservative: any
// uncertain case returns false (keep the object). The NDC y-flip matches the
// main pass's viewport, and depth is the [0, 1] range both APIs use, so
// MAX-reduced texels store the farthest occluder.
bool hiz_occluded(float3 bb_min, float3 bb_max)
{
    float2 ndc_min, ndc_max;
    float aabb_min_depth;
    if (!project_aabb(bb_min, bb_max, ndc_min, ndc_max, aabb_min_depth))
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
    // Standard depth: nearest point of the AABB at NDC.z near 0. Behind-near or
    // behind-far means we conservatively keep the AABB.
    if (aabb_min_depth < 0.0 || aabb_min_depth > 1.0)
    {
        return false;
    }
    // Map NDC -> UV (y flips because NDC y is up, UV v is down).
    float2 uv_min = float2(ndc_min.x * 0.5 + 0.5, 0.5 - ndc_max.y * 0.5);
    float2 uv_max = float2(ndc_max.x * 0.5 + 0.5, 0.5 - ndc_min.y * 0.5);
    // Size of the rect at mip 0, in texels.
    float2 size_tex = (uv_max - uv_min) * CULL_HIZ_SIZE;
    float max_dim = max(size_tex.x, size_tex.y);
    // Pick the mip whose texels are roughly the rect size so a 2x2 footprint
    // covers the rect (the standard Hi-Z 4-tap pattern).
    int mip = int(ceil(log2(max(max_dim, 1.0))));
    mip = clamp(mip, 0, int(CULL_HIZ_MIPS) - 1);
    float2 mip_dim = max(CULL_HIZ_SIZE / float(1u << uint(mip)), float2(1.0, 1.0));
    int2 lo = int2(floor(uv_min * mip_dim));
    int2 hi = int2(floor(uv_max * mip_dim));
    int2 max_xy = int2(mip_dim) - int2(1, 1);
    lo = clamp(lo, int2(0, 0), max_xy);
    hi = clamp(hi, int2(0, 0), max_xy);
    float d0 = hiz_tex.Load(int3(lo.x, lo.y, mip));
    float d1 = hiz_tex.Load(int3(hi.x, lo.y, mip));
    float d2 = hiz_tex.Load(int3(lo.x, hi.y, mip));
    float d3 = hiz_tex.Load(int3(hi.x, hi.y, mip));
    float occluder_depth = max(max(d0, d1), max(d2, d3));
    // If the AABB's closest projected depth is strictly behind the farthest
    // previously-rasterised surface in this region, the whole AABB is hidden.
    return aabb_min_depth > occluder_depth;
}

#ifndef METAL_BINDINGS
// Write one record's command into its own shader bucket's region and a no-op
// into every other region. `c` already carries instance_count 0 when the record
// was culled, so a culled record resets every region. The reset sweep matters
// because a freed draw slot can be reused by a record of a DIFFERENT bucket,
// which would otherwise leave the old bucket's command stale and still
// executing.
void write_bucket_commands(uint i, DrawCommand c, uint flags)
{
    uint bucket = min((flags >> DRAW_BUCKET_SHIFT) & DRAW_BUCKET_MASK,
                      cull.bucket_count - 1u);
    DrawCommand noop = c;
    noop.instance_count = 0u;
    for (uint b = 0u; b < cull.bucket_count; ++b)
    {
        if (b == bucket)
        {
            commands[b * cull.bucket_stride + i] = c;
        }
        else
        {
            commands[b * cull.bucket_stride + i] = noop;
        }
    }
}
#endif // !METAL_BINDINGS

#endif // !SHADOW_CULL

[shader("compute")]
[numthreads(64, 1, 1)]
void cull_kernel(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= cull.object_count)
    {
        return;
    }

#ifdef SHADOW_CULL
    GpuDrawArgs a = draw_args[i];
    // Light-frustum only, against this cascade's `planes`. Deliberately NO
    // per-object distance cull: the cascade light frustum already bounds the
    // shadow draw distance through its extents, and the per-object view
    // `cull_distance` is a view-LOD-fade concept that must not silence shadows
    // (the legacy CPU shadow pass drew every caster; an off-screen caster
    // beyond cull_distance can still cast into the visible scene).
    bool draw = (a.flags & DRAW_ENABLED) != 0u;
    if (draw && (a.flags & DRAW_CULLABLE) != 0u)
    {
        GpuObjectData obj = objects[i];
        draw = !frustum_culled(obj.bb_min_cull_distance.xyz, obj.bb_max_alpha_cutoff.xyz);
    }
#ifdef METAL_BINDINGS
    cull_status[cull.cascade_base + i] = draw ? STATUS_DRAWN : STATUS_CULLED;
#else
    DrawCommand c = make_command(i, a);
    if (!draw)
    {
        c.instance_count = 0u;
    }
    commands[i] = c;
#endif
#elif defined(CULL_PHASE2)
    // Runs after the pyramid was rebuilt from this frame's phase-1 depth.
    // Re-tests only the objects phase 1 marked STATUS_HIZ_CANDIDATE against the
    // fresh pyramid (projected through this frame's VP, carried in
    // `prev_view_proj` exactly as phase 1 used the previous frame's). A
    // candidate still occluded by this frame's actual depth stays culled; one
    // now visible is redrawn. Everything else phase 1 settled and is left as it
    // was.
    bool candidate = cull_status[i] == STATUS_HIZ_CANDIDATE;
    bool redraw = candidate;
    if (candidate)
    {
        GpuObjectData obj = objects[i];
        if (CULL_HIZ_ENABLED != 0u
            && hiz_occluded(obj.bb_min_cull_distance.xyz, obj.bb_max_alpha_cutoff.xyz))
        {
            redraw = false;
        }
        cull_status[i] = redraw ? STATUS_REDRAW : STATUS_CULLED;
    }
#ifndef METAL_BINDINGS
    GpuDrawArgs a = draw_args[i];
    DrawCommand c = make_command(i, a);
    if (!redraw)
    {
        c.instance_count = 0u;
    }
    write_bucket_commands(i, c, a.flags);
#endif
#else
    // Phase 1. Records the outcome in `cull_status` for two-pass occlusion: a
    // Hi-Z cull is the only outcome phase 2 reconsiders against the rebuilt
    // pyramid; everything else is settled here.
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
            // Cull when the AABB is fully behind the previous frame's depth
            // pyramid. Skipped on the first frame and after a resize
            // (`hiz_enabled == 0`), where no valid pyramid exists yet.
            status = STATUS_HIZ_CANDIDATE;
        }
    }
    cull_status[i] = status;
#ifndef METAL_BINDINGS
    DrawCommand c = make_command(i, a);
    if (status != STATUS_DRAWN)
    {
        c.instance_count = 0u;
    }
    write_bucket_commands(i, c, a.flags);
#endif
#endif
}
