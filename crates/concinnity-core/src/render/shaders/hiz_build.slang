// Hi-Z (depth-mip pyramid) builder. One kernel per compile, selected by a
// define so each variant declares exactly the resources it binds (Metal and
// DXIL indices are assigned in declaration order, so an unused declaration
// would shift the live ones):
//
//   HIZ_INIT_MSAA   - hiz_init_msaa: read the MSAA main depth taking MAX over
//                     sample_count samples, write into HiZ mip 0.
//   HIZ_INIT_SINGLE - hiz_init_single: same for a single-sample main depth.
//   HIZ_DOWNSAMPLE  - hiz_downsample: MAX-reduce 2x2 texels of the previous
//                     HiZ mip (bound as a single-level view) into the next.
//   HIZ_SPD_MSAA    - hiz_spd_msaa: read the MSAA main depth and produce mips
//                     0..5 in one dispatch.
//   HIZ_SPD_SINGLE  - hiz_spd_single: same for a single-sample main depth.
//   HIZ_SPD_TAIL    - hiz_spd_tail: continue a pyramid from mip 5, producing
//                     mips 6..10 in one dispatch.
//
// The three HIZ_SPD_* kernels replace the init-plus-N-downsamples chain with two
// dispatches. Each workgroup reduces a 32x32 tile of its base level through
// six levels: two in registers (each thread owns a 2x2 patch) and four more
// through threadgroup memory. Vulkan and DirectX build the pyramid this way;
// Metal still runs the per-mip chain above.
//
// The MAX reduction is correct because the engine uses standard (not reverse)
// depth: a Hi-Z texel storing the MAX represents the farthest visible surface
// in that region, so the cull can only be conservative.
//
// Both source and destination mips are single-level R32F views, so each
// downsample reads mip M and writes mip M+1 without aliasing the same texels.

#if defined(HIZ_SPD_MSAA) || defined(HIZ_SPD_SINGLE) || defined(HIZ_SPD_TAIL)

// Levels one SPD dispatch produces, and the width of the tile of its base level
// each workgroup owns. 16x16 threads each reduce a 2x2 patch, so two levels
// come out of registers and the remaining four out of threadgroup memory.
//
// The 2x2 patch is what keeps the base-level read fast: neighboring lanes
// address neighboring texels, and the tile stays small enough that a frame
// dispatches thousands of groups rather than a couple of hundred. A wider patch
// per thread reduces in fewer groups but reads the depth buffer at a stride,
// which costs more than the dispatches it saves.
#define HIZ_SPD_LEVELS 6
#define HIZ_SPD_TILE 32
#define HIZ_SPD_GROUP 16

// Phase 1 binds the depth source at 0 and the mips after it; the tail has no
// separate source, so its mips start at 0.
#if defined(HIZ_SPD_TAIL)
#define HIZ_SPD_MIP_BINDING 0
#else
#define HIZ_SPD_MIP_BINDING 1
#endif

struct HizSpdParams
{
    // Dimensions of this dispatch's base level: mip 0 for phase 1, mip 6 for
    // the tail.
    uint base_width;
    uint base_height;
    // Levels this dispatch is responsible for, counting the base as one. A
    // store at relative level r happens only while r < level_count.
    uint level_count;
    uint sample_count;
};

[[vk::push_constant]]
ConstantBuffer<HizSpdParams> params;

#else

struct HizParams
{
    uint dst_width;
    uint dst_height;
    uint src_mip;
    uint sample_count;
};

[[vk::push_constant]]
ConstantBuffer<HizParams> params;

#endif

#if defined(HIZ_INIT_MSAA)

[[vk::binding(0, 0)]]
Texture2DMS<float> src_depth;

#elif defined(HIZ_INIT_SINGLE)

[[vk::binding(0, 0)]]
Texture2D<float> src_depth;

#elif defined(HIZ_DOWNSAMPLE)

// Previous Hi-Z mip, bound as a single-level view (read-only storage image).
[[vk::binding(0, 0)]]
[format("r32f")]
RWTexture2D<float> src_hiz;

#elif defined(HIZ_SPD_MSAA)

[[vk::binding(0, 0)]]
Texture2DMS<float> src_depth;

#elif defined(HIZ_SPD_SINGLE)

[[vk::binding(0, 0)]]
Texture2D<float> src_depth;

#elif defined(HIZ_SPD_TAIL)

// No separate source: the tail reads its base level out of `spd_mips[0]`.

#else
#error "hiz_build.slang: define HIZ_INIT_MSAA, HIZ_INIT_SINGLE, HIZ_DOWNSAMPLE, HIZ_SPD_MSAA, HIZ_SPD_SINGLE, or HIZ_SPD_TAIL"
#endif

#if defined(HIZ_INIT_MSAA) || defined(HIZ_INIT_SINGLE) || defined(HIZ_DOWNSAMPLE)

[[vk::binding(1, 0)]]
[format("r32f")]
RWTexture2D<float> dst_mip;

#else

// The mips one SPD dispatch writes, finest first. Phase 1 binds pyramid mips
// 0..5; the tail binds 5..10, so its element 0 is the level it reduces.
[[vk::binding(HIZ_SPD_MIP_BINDING, 0)]]
[format("r32f")]
RWTexture2D<float> spd_mips[HIZ_SPD_LEVELS];

#endif

#if defined(HIZ_INIT_MSAA) || defined(HIZ_SPD_MSAA)

// Slang's Metal backend emits texture2d_ms::read with an int2 coordinate,
// which MSL rejects (read takes uint2). Route the Metal target through inline
// MSL until that is fixed upstream; every other target keeps the plain Load.
float load_depth_sample(Texture2DMS<float> t, uint2 coord, uint s)
{
    __target_switch
    {
    case metal:
        __intrinsic_asm "$0.read(uint2($1), uint($2)).x";
    default:
        return t.Load(int2(coord), int(s));
    }
}

#endif

#if defined(HIZ_INIT_MSAA)

[shader("compute")]
[numthreads(8, 8, 1)]
void hiz_init_msaa(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.dst_width || tid.y >= params.dst_height)
    {
        return;
    }
    float d = 0.0;
    for (uint s = 0u; s < params.sample_count; ++s)
    {
        d = max(d, load_depth_sample(src_depth, tid.xy, s));
    }
    dst_mip[tid.xy] = d;
}

#elif defined(HIZ_INIT_SINGLE)

[shader("compute")]
[numthreads(8, 8, 1)]
void hiz_init_single(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.dst_width || tid.y >= params.dst_height)
    {
        return;
    }
    dst_mip[tid.xy] = src_depth.Load(int3(int2(tid.xy), 0));
}

#elif defined(HIZ_DOWNSAMPLE)

[shader("compute")]
[numthreads(8, 8, 1)]
void hiz_downsample(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.dst_width || tid.y >= params.dst_height)
    {
        return;
    }
    uint2 src_size;
    src_hiz.GetDimensions(src_size.x, src_size.y);
    uint sx = tid.x * 2u;
    uint sy = tid.y * 2u;
    // For odd source dimensions the right/bottom edge loses a texel, but
    // max-reduction is conservative so dropping a half-row is harmless: it can
    // only make the cull more conservative, never wrongly cull a visible
    // object. Clamp the +1 taps so an odd edge reuses the in-bounds texel.
    uint sx1 = min(sx + 1u, src_size.x - 1u);
    uint sy1 = min(sy + 1u, src_size.y - 1u);
    float d0 = src_hiz[uint2(sx, sy)];
    float d1 = src_hiz[uint2(sx1, sy)];
    float d2 = src_hiz[uint2(sx, sy1)];
    float d3 = src_hiz[uint2(sx1, sy1)];
    dst_mip[tid.xy] = max(max(d0, d1), max(d2, d3));
}

#endif

#if defined(HIZ_SPD_MSAA) || defined(HIZ_SPD_SINGLE) || defined(HIZ_SPD_TAIL)

// Level-2 results of every thread in the group: the handover from the
// per-thread registers to the cross-thread reduction.
groupshared float spd_tile[HIZ_SPD_GROUP][HIZ_SPD_GROUP];

uint2 spd_level_size(uint level)
{
    uint2 base = uint2(params.base_width, params.base_height);
    return max(base >> level, uint2(1u, 1u));
}

// Out-of-range texels reduce as 0.0, the identity for MAX over standard depth,
// so a level whose footprint runs past the edge takes the max of the real
// texels alone. That can only lower the stored occluder depth, which makes the
// cull more permissive and never wrongly rejects a visible object.
void spd_store(uint level, uint2 coord, float v)
{
    if (level < params.level_count && all(coord < spd_level_size(level)))
    {
        spd_mips[level][coord] = v;
    }
}

float spd_max4(float a, float b, float c, float d)
{
    return max(max(a, b), max(c, d));
}

// Reduce one tile of the base level, already loaded as this thread's 2x2 patch,
// down through the dispatch's levels. `write_base` is false for the tail, whose
// base level is the mip phase 1 already finished.
void spd_reduce(float src[2][2], uint2 group_id, uint2 tid, bool write_base)
{
    // Levels 0 and 1 stay in registers: a thread's contiguous 2x2 of the base
    // level is a single texel of level 1.
    if (write_base)
    {
        uint2 base_origin = group_id * uint(HIZ_SPD_TILE) + tid * 2u;
        [unroll]
        for (uint j = 0u; j < 2u; ++j)
        {
            [unroll]
            for (uint i = 0u; i < 2u; ++i)
            {
                spd_store(0u, base_origin + uint2(i, j), src[j][i]);
            }
        }
    }

    float carried = spd_max4(src[0][0], src[0][1], src[1][0], src[1][1]);
    spd_store(1u, group_id * uint(HIZ_SPD_TILE / 2) + tid, carried);

    // Levels 2..5 cross threads. Each round reads four neighbors out of
    // threadgroup memory and writes its result back for the next one; read and
    // write are separated by a barrier because the slot a thread writes is one
    // another thread reads. Every thread reaches both barriers.
    spd_tile[tid.y][tid.x] = carried;
    [unroll]
    for (uint level = 2u; level < HIZ_SPD_LEVELS; ++level)
    {
        GroupMemoryBarrierWithGroupSync();
        uint span = 1u << (level - 1u);
        bool active = all(tid < uint2(HIZ_SPD_GROUP, HIZ_SPD_GROUP) / span);
        if (active)
        {
            uint2 src_id = tid * 2u;
            carried = spd_max4(spd_tile[src_id.y][src_id.x],
                               spd_tile[src_id.y][src_id.x + 1u],
                               spd_tile[src_id.y + 1u][src_id.x],
                               spd_tile[src_id.y + 1u][src_id.x + 1u]);
            spd_store(level, group_id * max(uint(HIZ_SPD_TILE) >> level, 1u) + tid, carried);
        }
        GroupMemoryBarrierWithGroupSync();
        if (active)
        {
            spd_tile[tid.y][tid.x] = carried;
        }
    }
}

// Base-level coordinate of the (i, j) texel in this thread's 2x2 patch.
uint2 spd_src_coord(uint2 group_id, uint2 tid, uint i, uint j)
{
    return group_id * uint(HIZ_SPD_TILE) + tid * 2u + uint2(i, j);
}

#endif

#if defined(HIZ_SPD_MSAA)

[shader("compute")]
[numthreads(HIZ_SPD_GROUP, HIZ_SPD_GROUP, 1)]
void hiz_spd_msaa(uint3 group_id : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint2 base = uint2(params.base_width, params.base_height);
    float src[2][2];
    [unroll]
    for (uint j = 0u; j < 2u; ++j)
    {
        [unroll]
        for (uint i = 0u; i < 2u; ++i)
        {
            uint2 coord = spd_src_coord(group_id.xy, tid.xy, i, j);
            float d = 0.0;
            if (all(coord < base))
            {
                for (uint n = 0u; n < params.sample_count; ++n)
                {
                    d = max(d, load_depth_sample(src_depth, coord, n));
                }
            }
            src[j][i] = d;
        }
    }
    spd_reduce(src, group_id.xy, tid.xy, true);
}

#elif defined(HIZ_SPD_SINGLE)

[shader("compute")]
[numthreads(HIZ_SPD_GROUP, HIZ_SPD_GROUP, 1)]
void hiz_spd_single(uint3 group_id : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint2 base = uint2(params.base_width, params.base_height);
    float src[2][2];
    [unroll]
    for (uint j = 0u; j < 2u; ++j)
    {
        [unroll]
        for (uint i = 0u; i < 2u; ++i)
        {
            uint2 coord = spd_src_coord(group_id.xy, tid.xy, i, j);
            src[j][i] = all(coord < base) ? src_depth.Load(int3(int2(coord), 0)) : 0.0;
        }
    }
    spd_reduce(src, group_id.xy, tid.xy, true);
}

#elif defined(HIZ_SPD_TAIL)

[shader("compute")]
[numthreads(HIZ_SPD_GROUP, HIZ_SPD_GROUP, 1)]
void hiz_spd_tail(uint3 group_id : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint2 base = uint2(params.base_width, params.base_height);
    float src[2][2];
    [unroll]
    for (uint j = 0u; j < 2u; ++j)
    {
        [unroll]
        for (uint i = 0u; i < 2u; ++i)
        {
            uint2 coord = spd_src_coord(group_id.xy, tid.xy, i, j);
            src[j][i] = all(coord < base) ? spd_mips[0][coord] : 0.0;
        }
    }
    // The base level is already final -- phase 1 wrote it -- and rewriting it
    // would have this dispatch read and write the same texels.
    spd_reduce(src, group_id.xy, tid.xy, false);
}

#endif
