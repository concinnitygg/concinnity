#include <metal_stdlib>
#include <metal_command_buffer>
using namespace metal;

// Turns the per-object `cull_status` the single-source `cull.hlsl` decision
// kernels wrote into Metal indirect command buffers. This is the only half of
// the cull Metal keeps hand-written, because the ICB encoding
// (`render_command`, `array<command_buffer, N>`) is a declaration HLSL cannot
// express.
//
//   cull_scatter  every cull: each kept record drawn at its place in its
//                 stream's list, which `cull_compact.hlsl`'s scan laid out
//                 (mirrors its scatter)

// Mirrors gfx::render_types::GpuDrawArgs (16 B).
struct GpuDrawArgs {
    uint index_count;
    uint index_offset;
    uint base_vertex;
    uint flags;
};

// The decision kernel keeps a record's outcome in its status word's low byte
// and its shader bucket above it. Locked to gfx::cull_status::CullStatus.
constant uint CULL_OUTCOME_MASK = 0xffu;
constant uint CULL_BUCKET_SHIFT = 8u;

// Locked to gfx::render_types::{MAX_SHADER_BUCKETS, MAX_VARIANT_ROUTES}.
constant uint MAX_SHADER_BUCKETS = 64u;
constant uint MAX_VARIANT_ROUTES = 32u;

// Locked to gfx::render_types::{CULL_GROUP_RECORDS, SKINNED_STREAM}.
constant uint CULL_GROUP_RECORDS = 64u;
constant uint SKINNED_STREAM = 64u;
constant uint NO_STREAM = 0xffffffffu;

// Mirrors gfx::render_types::GpuBucketRegion (32 B): a bucket's build-time
// records, then its runtime ones.
struct BucketRegion {
    uint first;
    uint count;
    uint runtime_first;
    uint runtime_count;
    uint offset;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// Mirrors gfx::render_types::GpuCullVariants; only the regions are read here.
struct CullVariants {
    float4 far;
    float4 day_center;
    float4 day_camera_up;
    float4 day_reach;
    uint4 routes[MAX_VARIANT_ROUTES];
    BucketRegion regions[MAX_SHADER_BUCKETS];
};

// Whether `record` falls in region `r`. Mirrors GpuBucketRegion::command.
static bool in_region(constant BucketRegion &r, uint record) {
    return record - r.first < r.count || record - r.runtime_first < r.runtime_count;
}

// Mirrors metal::uniforms::ScatterParams (32 B).
struct ScatterParams {
    uint object_count;
    uint skinned_base;
    uint bucket_count;
    uint draw_status;
    uint coarse_offset;
    uint coarse_count;
    uint full_count;
    uint _pad;
};

// Locked to gfx::render_types::DrawArgsFlags::PLANET_TILE.
constant uint DRAW_PLANET_TILE = 16u;

// The per-bucket indirect command buffers, reached through an argument buffer.
// Only the first `bucket_count` entries are encoded; the kernels never
// construct a command against an entry past that.
struct ICBContainer {
    array<command_buffer, MAX_SHADER_BUCKETS> icbs [[id(0)]];
};

// Encode `record`'s indexed draw at `index` of `icb`. Records at or past
// `skinned_base` draw the compute-deformed geometry through the skinned index
// buffer; the index buffer is part of the indirect command on Metal, so it is
// picked here rather than bound per draw range. `base_instance` carries the
// record id into the vertex stage.
static void encode_draw(command_buffer icb,
                        uint index,
                        uint record,
                        GpuDrawArgs a,
                        uint skinned_base,
                        const device uint *index_buf,
                        const device uint *skinned_index_buf) {
    const device uint *ib = record >= skinned_base ? skinned_index_buf : index_buf;
    render_command cmd(icb, index);
    cmd.draw_indexed_primitives(primitive_type::triangle,
                                a.index_count,
                                ib + a.index_offset,
                                1u,
                                a.base_vertex,
                                record);
}

kernel void cull_scatter(
    constant GpuDrawArgs   *draw_args         [[buffer(1)]],
    const device uint      *index_buf         [[buffer(3)]],
    device ICBContainer    *icb_c             [[buffer(4)]],
    const device uint      *cull_status       [[buffer(5)]],
    const device uint      *skinned_index_buf [[buffer(6)]],
    constant ScatterParams &p                 [[buffer(7)]],
    constant CullVariants  &variants          [[buffer(8)]],
    const device uint      *group_counts      [[buffer(9)]],
    uint                    record            [[thread_position_in_grid]],
    uint                    group             [[threadgroup_position_in_grid]],
    uint                    gi                [[thread_index_in_threadgroup]]
) {
    threadgroup uint group_streams[CULL_GROUP_RECORDS];
    uint stream = NO_STREAM;
    uint skinned_base = min(p.skinned_base, p.object_count);
    if (record < p.object_count) {
        uint word = cull_status[record];
        if ((word & CULL_OUTCOME_MASK) == p.draw_status) {
            uint bucket = min(word >> CULL_BUCKET_SHIFT, MAX_SHADER_BUCKETS - 1u);
            if (p.bucket_count <= 1u || bucket == 0u) {
                stream = record >= skinned_base ? SKINNED_STREAM : 0u;
            } else if (in_region(variants.regions[bucket], record)) {
                stream = bucket;
            }
        }
    }
    group_streams[gi] = stream;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (stream == NO_STREAM) {
        return;
    }
    uint rank = 0u;
    for (uint j = 0u; j < gi; ++j) {
        rank += group_streams[j] == stream ? 1u : 0u;
    }
    uint groups = (p.object_count + CULL_GROUP_RECORDS - 1u) / CULL_GROUP_RECORDS;
    uint place = group_counts[stream * groups + group] + rank;
    uint first = 0u;
    uint capacity = skinned_base;
    uint icb = 0u;
    if (stream == SKINNED_STREAM) {
        first = skinned_base;
        capacity = p.object_count - skinned_base;
    } else if (stream > 0u) {
        constant BucketRegion &r = variants.regions[stream];
        capacity = r.count + r.runtime_count;
        icb = stream;
    }
    if (place < capacity) {
        // A capture draws a planet tile of the full grid through the coarse
        // one, from the same corners.
        GpuDrawArgs a = draw_args[record];
        if (p.coarse_count != 0u && (a.flags & DRAW_PLANET_TILE) != 0u && a.index_count == p.full_count) {
            a.index_offset = p.coarse_offset;
            a.index_count = p.coarse_count;
        }
        encode_draw(icb_c->icbs[icb], first + place, record, a,
                    skinned_base, index_buf, skinned_index_buf);
    }
}
