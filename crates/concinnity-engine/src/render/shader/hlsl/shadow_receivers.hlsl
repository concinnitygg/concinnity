// Shadow receiver bounds: the light-space bounds of the surfaces the camera
// saw in each sun cascade's depth band, reduced from the first level of the
// Hi-Z pyramid (the frame's depth, the farthest of any MSAA samples). One entry
// per compile, selected by a define, so each declares exactly what it binds:
//
//   RECEIVERS_REDUCE - receiver_bounds_reduce: one group per RECEIVER_TILE
//                      square of the depth, each thread reading a strided
//                      4x4 of it, writes the tile's bounds to `partials`.
//   RECEIVERS_MERGE  - receiver_bounds_merge: one group folds every tile's
//                      bounds into `bounds` and stamps it with the serial.
//
// A pixel's reconstructed point is taken to the light's right, up and
// toward-the-light axes and its view depth by one matrix, so each band keeps
// the least and greatest of all four. A band no pixel fell in keeps
// RECEIVER_EMPTY: its low end above its high one. Both kernels fold their
// threads' bounds a wave at a time, then the waves through group memory, so a
// group waits on one barrier. That takes a thread's wave to be
// `SV_GroupIndex / WaveGetLaneCount()`: every wave of a group full and its
// lanes consecutive group indices. No API promises that layout; desktop and
// Apple GPUs use it for a group whose size is a multiple of the wave's, as both
// of these are.
//
// Every resource takes its Metal index from the number on its `register()`, so
// the depth is texture(0), the partials buffer(0) and the bounds buffer(1).
//
// RECEIVER_BANDS, RECEIVER_TILE and RECEIVER_EMPTY mirror
// gfx::render_types::shadow; ReceiverBoundsParams, ReceiverBand,
// ReceiverBands and ReceiverBounds mirror the structs there.

{DEPTH_CONVENTION}

#define RECEIVER_BANDS 4
#define RECEIVER_GROUP 16
#define RECEIVER_TILE 64
// Texels a thread reads along each axis of its tile.
#define RECEIVER_STEPS (RECEIVER_TILE / RECEIVER_GROUP)
#define RECEIVER_MERGE_GROUP 256
// The fewest lanes a wave may have, and so room for one entry per wave of a
// group. A host runs the kernels only on a device whose waves are at least
// this wide.
#define RECEIVER_MIN_LANES 4
#define RECEIVER_MAX_WAVES (RECEIVER_MERGE_GROUP / RECEIVER_MIN_LANES)
// Finite, so the reduction stays exact under fast math.
#define RECEIVER_EMPTY 3.0e38

struct ReceiverBoundsParams
{
    // NDC x, y and device depth to the light's right, up and toward-the-light
    // coordinates and the view depth, each times `w_row`'s w.
    float4x4 to_light_space;
    // NDC x, y and device depth to the reconstructed point's homogeneous w.
    float4 w_row;
    // View depth where each band starts and ends; an unused band starts past
    // where it ends.
    float4 band_near;
    float4 band_far;
    uint width;
    uint height;
    uint groups_x;
    uint serial;
};

struct ReceiverBand
{
    float4 lo;
    float4 hi;
};

struct ReceiverBands
{
    ReceiverBand bands[RECEIVER_BANDS];
};

struct ReceiverBounds
{
    ReceiverBand bands[RECEIVER_BANDS];
    uint serial;
    uint pad0;
    uint pad1;
    uint pad2;
};

// DirectX takes the parameters as root constants at b0; Vulkan pushes them,
// and Metal binds them past each kernel's buffers.
#ifdef CN_BACKEND_DIRECTX
ConstantBuffer<ReceiverBoundsParams> params : register(b0);
#elif defined(RECEIVERS_REDUCE)
[[vk::push_constant]] ConstantBuffer<ReceiverBoundsParams> params : register(b1);
#else
[[vk::push_constant]] ConstantBuffer<ReceiverBoundsParams> params : register(b2);
#endif

groupshared float4 wave_lo[RECEIVER_MAX_WAVES][RECEIVER_BANDS];
groupshared float4 wave_hi[RECEIVER_MAX_WAVES][RECEIVER_BANDS];

// Every band empty.
void clear_bands(out float4 lo[RECEIVER_BANDS], out float4 hi[RECEIVER_BANDS])
{
    [unroll]
    for (uint b = 0u; b < RECEIVER_BANDS; ++b)
    {
        lo[b] = RECEIVER_EMPTY;
        hi[b] = -RECEIVER_EMPTY;
    }
}

// Fold every thread's per-band `lo` and `hi` in a group of `threads`: each
// wave folds its lanes, then thread `b` of the first RECEIVER_BANDS folds the
// waves' band `b` into its `lo[0]` and `hi[0]`. Every thread of the group
// calls it.
void reduce_group(
    uint tid,
    uint threads,
    inout float4 lo[RECEIVER_BANDS],
    inout float4 hi[RECEIVER_BANDS])
{
    uint lanes = WaveGetLaneCount();
    uint wave = tid / lanes;
    [unroll]
    for (uint b = 0u; b < RECEIVER_BANDS; ++b)
    {
        float4 l = WaveActiveMin(lo[b]);
        float4 h = WaveActiveMax(hi[b]);
        if (WaveIsFirstLane())
        {
            wave_lo[wave][b] = l;
            wave_hi[wave][b] = h;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid < RECEIVER_BANDS)
    {
        float4 l = RECEIVER_EMPTY;
        float4 h = -RECEIVER_EMPTY;
        uint waves = (threads + lanes - 1u) / lanes;
        for (uint w = 0u; w < waves; ++w)
        {
            l = min(l, wave_lo[w][tid]);
            h = max(h, wave_hi[w][tid]);
        }
        lo[0] = l;
        hi[0] = h;
    }
}

#ifdef RECEIVERS_REDUCE

[[vk::binding(0, 0)]] Texture2D<float> depth : register(t0);
[[vk::binding(1, 0)]] RWStructuredBuffer<ReceiverBands> partials : register(u0);

[shader("compute")]
[numthreads(RECEIVER_GROUP, RECEIVER_GROUP, 1)]
void receiver_bounds_reduce(
    uint3 group : SV_GroupID,
    uint3 local : SV_GroupThreadID,
    uint tid : SV_GroupIndex)
{
    float4 lo[RECEIVER_BANDS];
    float4 hi[RECEIVER_BANDS];
    clear_bands(lo, hi);
    uint2 size = uint2(params.width, params.height);
    float2 inv_size = 1.0 / float2(size);
    uint2 origin = group.xy * RECEIVER_TILE + local.xy;
    // Neighboring threads read neighboring texels on every step.
    for (uint step = 0u; step < RECEIVER_STEPS * RECEIVER_STEPS; ++step)
    {
        uint2 pixel = origin + uint2(step % RECEIVER_STEPS, step / RECEIVER_STEPS) * RECEIVER_GROUP;
        if (any(pixel >= size))
        {
            continue;
        }
        float d = depth.Load(int3(int2(pixel), 0));
        if (!depth_is_written(d))
        {
            continue;
        }
        float2 uv = (float2(pixel) + 0.5) * inv_size;
        float4 at = float4(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0), d, 1.0);
        float w = dot(params.w_row, at);
        if (w <= DEPTH_RECONSTRUCT_MIN_W)
        {
            continue;
        }
        float4 q = mul(params.to_light_space, at) / w;
        [unroll]
        for (uint band = 0u; band < RECEIVER_BANDS; ++band)
        {
            if (q.w >= params.band_near[band] && q.w <= params.band_far[band])
            {
                lo[band] = min(lo[band], q);
                hi[band] = max(hi[band], q);
            }
        }
    }
    reduce_group(tid, RECEIVER_GROUP * RECEIVER_GROUP, lo, hi);
    if (tid < RECEIVER_BANDS)
    {
        uint tile = group.y * params.groups_x + group.x;
        partials[tile].bands[tid].lo = lo[0];
        partials[tile].bands[tid].hi = hi[0];
    }
}

#elif defined(RECEIVERS_MERGE)

[[vk::binding(0, 0)]] RWStructuredBuffer<ReceiverBands> partials : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<ReceiverBounds> bounds : register(u1);

[shader("compute")]
[numthreads(RECEIVER_MERGE_GROUP, 1, 1)]
void receiver_bounds_merge(uint tid : SV_GroupIndex)
{
    float4 lo[RECEIVER_BANDS];
    float4 hi[RECEIVER_BANDS];
    clear_bands(lo, hi);
    uint tiles = params.groups_x * ((params.height + RECEIVER_TILE - 1u) / RECEIVER_TILE);
    for (uint tile = tid; tile < tiles; tile += RECEIVER_MERGE_GROUP)
    {
        ReceiverBands part = partials[tile];
        [unroll]
        for (uint band = 0u; band < RECEIVER_BANDS; ++band)
        {
            lo[band] = min(lo[band], part.bands[band].lo);
            hi[band] = max(hi[band], part.bands[band].hi);
        }
    }
    reduce_group(tid, RECEIVER_MERGE_GROUP, lo, hi);
    if (tid < RECEIVER_BANDS)
    {
        bounds[0].bands[tid].lo = lo[0];
        bounds[0].bands[tid].hi = hi[0];
    }
    if (tid == 0u)
    {
        bounds[0].serial = params.serial;
        bounds[0].pad0 = 0u;
        bounds[0].pad1 = 0u;
        bounds[0].pad2 = 0u;
    }
}

#endif
