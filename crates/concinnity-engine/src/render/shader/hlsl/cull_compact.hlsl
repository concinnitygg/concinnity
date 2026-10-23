// Packs the draws a cull phase kept into one list per stream (a shader bucket,
// or the skinned tail), in record order, so each indirect draw executes only
// what its list holds. Mirrors gfx::render_types::streams.
//
// Two kernels out of one file, selected by define, after the cull counted each
// 64-record group's kept draws per stream into `group_counts`:
//
//   (none)        the scan: a group per stream turns its counts into each
//                 group's first place in the list and writes the list's length
//   CULL_SCATTER  one thread per record writes each kept record's command at
//                 its group's place plus its rank in the group (Vulkan and
//                 DirectX; Metal's is the hand-written cull_encode.metal)
//
// Neither uses atomics, so a list's order never depends on scheduling.

{OBJECT_COMMON}
{CULL_COMMON}

// Layout matches `gfx::render_types::GpuCompactParams` (16 B): the b0 root
// constants on DirectX, the push constant on Vulkan, inline at buffer(11) on
// Metal. `drawn` is the outcome the phase keeps.
struct CompactParams
{
    uint object_count;
    uint bucket_count;
    uint skinned_base;
    uint drawn;
};

#if defined(CN_BACKEND_METAL)
[[vk::push_constant]] ConstantBuffer<CompactParams> params : register(b11);
#define COMPACT_VARIANTS_REGISTER     t8
#define COMPACT_GROUP_COUNTS_REGISTER u9
#define COMPACT_OUT_REGISTER          u10
#else
[[vk::push_constant]] ConstantBuffer<CompactParams> params : register(b0);
#define COMPACT_VARIANTS_REGISTER     t0
#define COMPACT_GROUP_COUNTS_REGISTER u0
#define COMPACT_OUT_REGISTER          u2
#endif

[[vk::binding(0, 0)]] StructuredBuffer<uint4> variants : register(COMPACT_VARIANTS_REGISTER);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> group_counts : register(COMPACT_GROUP_COUNTS_REGISTER);

uint cull_group_count()
{
    return (params.object_count + CULL_GROUP_RECORDS - 1u) / CULL_GROUP_RECORDS;
}

// Where a stream's commands go: from `region` in a shared indirect buffer,
// `first` places into its bucket's region, at most `capacity` of them.
// Mirrors StreamLayout::slots.
struct StreamSlots
{
    uint region;
    uint first;
    uint capacity;
};

StreamSlots stream_slots(uint stream)
{
    uint skinned_base = min(params.skinned_base, params.object_count);
    StreamSlots s;
    s.region = 0u;
    s.first = 0u;
    s.capacity = 0u;
    if (stream == 0u)
    {
        s.capacity = skinned_base;
    }
    else if (stream == SKINNED_STREAM)
    {
        s.first = skinned_base;
        s.capacity = params.object_count - skinned_base;
    }
    else if (stream < params.bucket_count)
    {
        uint4 spans = CULL_REGION_SPANS(variants, stream);
        s.region = CULL_REGION_OFFSET(variants, stream);
        s.capacity = spans.y + spans.w;
    }
    return s;
}

#ifndef CULL_SCATTER

// The list lengths. Vulkan and DirectX read a count per stream, from the
// slots behind the commands; Metal an `MTLIndirectCommandBufferExecutionRange`
// per stream, its location and length.
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> stream_counts : register(COMPACT_OUT_REGISTER);

groupshared uint scan_row[CULL_GROUP_RECORDS];

// Group `s` scans stream `s`, the group past the buckets the skinned tail.
[shader("compute")]
[numthreads(64, 1, 1)]
void cull_scan(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    uint stream = gid.x < params.bucket_count ? gid.x : SKINNED_STREAM;
    uint groups = cull_group_count();
    uint base = stream * groups;
    uint running = 0u;
    for (uint first = 0u; first < groups; first += CULL_GROUP_RECORDS)
    {
        uint g = first + gi;
        uint count = g < groups ? group_counts[base + g] : 0u;
        scan_row[gi] = count;
        GroupMemoryBarrierWithGroupSync();
        for (uint step = 1u; step < CULL_GROUP_RECORDS; step <<= 1u)
        {
            uint add = gi >= step ? scan_row[gi - step] : 0u;
            GroupMemoryBarrierWithGroupSync();
            scan_row[gi] += add;
            GroupMemoryBarrierWithGroupSync();
        }
        if (g < groups)
        {
            group_counts[base + g] = running + scan_row[gi] - count;
        }
        running += scan_row[CULL_GROUP_RECORDS - 1u];
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi == 0u)
    {
        StreamSlots slots = stream_slots(stream);
        uint length = min(running, slots.capacity);
#if defined(CN_BACKEND_METAL)
        stream_counts[2u * stream] = slots.first;
        stream_counts[2u * stream + 1u] = length;
#else
        stream_counts[stream] = length;
#endif
    }
}

#else // CULL_SCATTER

[[vk::binding(1, 0)]] StructuredBuffer<GpuDrawArgs> draw_args : register(t1);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> cull_status : register(u1);
[[vk::binding(4, 0)]] RWStructuredBuffer<DrawCommand> commands : register(COMPACT_OUT_REGISTER);

groupshared uint group_streams[CULL_GROUP_RECORDS];

[shader("compute")]
[numthreads(64, 1, 1)]
void cull_scatter(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    uint i = tid.x;
    uint stream = NO_STREAM;
    if (i < params.object_count)
    {
        uint word = cull_status[i];
        if ((word & CULL_OUTCOME_MASK) == params.drawn)
        {
            uint bucket = min(word >> CULL_BUCKET_SHIFT, MAX_SHADER_BUCKETS - 1u);
            stream = record_stream(i, bucket, params.bucket_count, params.skinned_base,
                CULL_REGION_SPANS(variants, bucket));
        }
    }
    group_streams[gi] = stream;
    GroupMemoryBarrierWithGroupSync();
    if (stream == NO_STREAM)
    {
        return;
    }
    uint rank = 0u;
    for (uint j = 0u; j < gi; ++j)
    {
        rank += group_streams[j] == stream ? 1u : 0u;
    }
    uint place = group_counts[stream * cull_group_count() + gid.x] + rank;
    StreamSlots slots = stream_slots(stream);
    if (place < slots.capacity)
    {
        commands[slots.region + slots.first + place] = make_command(i, draw_args[i]);
    }
}

#endif // CULL_SCATTER
