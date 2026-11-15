// View samples: a coarse grid over the first level of the Hi-Z pyramid (the
// frame's depth, the farthest of any MSAA samples), one group per square cell.
// Each cell keeps the depth at its sample texel and its nearest texel with
// where that lies, for the CPU to read back and measure what the camera saw.
// The nearest is found by an atomic max over the depth's bits, which orders
// the camera's reversed depth (non-negative, nearer is greater) as it does the
// floats.
//
// Every resource takes its Metal index from the number on its `register()`, so
// the depth is texture(0) and the cells buffer(0).
//
// VIEW_SAMPLE_GROUP mirrors gfx::render_types::VIEW_SAMPLE_GROUP;
// ViewSampleParams and ViewSampleCell mirror the structs there.

{DEPTH_CONVENTION}

#define VIEW_SAMPLE_GROUP 16

struct ViewSampleParams
{
    uint width;
    uint height;
    uint cell;
    uint columns;
    uint rows;
    uint serial;
    // Where in every cell the depth is sampled, from its corner.
    uint spot_x;
    uint spot_y;
};

struct ViewSampleCell
{
    float depth;
    float nearest;
    // The nearest texel, packed (y << 16) | x.
    uint texel;
    uint serial;
};

// DirectX takes the parameters as root constants at b0; Vulkan pushes them,
// and Metal binds them past the kernel's buffers.
#ifdef CN_BACKEND_DIRECTX
ConstantBuffer<ViewSampleParams> params : register(b0);
#else
[[vk::push_constant]] ConstantBuffer<ViewSampleParams> params : register(b1);
#endif

[[vk::binding(0, 0)]] Texture2D<float> depth : register(t0);
[[vk::binding(1, 0)]] RWStructuredBuffer<ViewSampleCell> cells : register(u0);

groupshared uint group_nearest;
groupshared uint group_texel;

uint pack_texel(uint2 texel)
{
    return (texel.y << 16) | texel.x;
}

[shader("compute")]
[numthreads(VIEW_SAMPLE_GROUP, VIEW_SAMPLE_GROUP, 1)]
void view_samples(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID, uint tid : SV_GroupIndex)
{
    uint2 size = uint2(params.width, params.height);
    uint2 origin = group.xy * params.cell;
    uint2 end = min(origin + params.cell, size);
    uint2 spot = min(origin + uint2(params.spot_x, params.spot_y), size - 1u);

    if (tid == 0u)
    {
        group_nearest = asuint(DEPTH_FAR);
        group_texel = 0u;
    }
    float nearest = DEPTH_FAR;
    uint texel = 0u;
    // Neighboring threads read neighboring texels on every step.
    for (uint y = origin.y + local.y; y < end.y; y += VIEW_SAMPLE_GROUP)
    {
        for (uint x = origin.x + local.x; x < end.x; x += VIEW_SAMPLE_GROUP)
        {
            float d = depth.Load(int3(int2(x, y), 0));
            if (depth_in_front(d, nearest))
            {
                nearest = d;
                texel = pack_texel(uint2(x, y));
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    InterlockedMax(group_nearest, asuint(nearest));
    GroupMemoryBarrierWithGroupSync();
    // Of the threads holding the nearest depth, the greatest packed texel.
    if (depth_is_written(nearest) && asuint(nearest) == group_nearest)
    {
        InterlockedMax(group_texel, texel);
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0u)
    {
        ViewSampleCell cell;
        cell.depth = depth.Load(int3(int2(spot), 0));
        cell.nearest = asfloat(group_nearest);
        cell.texel = depth_is_written(cell.nearest) ? group_texel : pack_texel(spot);
        cell.serial = params.serial;
        cells[group.y * params.columns + group.x] = cell;
    }
}
