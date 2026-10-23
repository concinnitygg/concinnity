#include <metal_stdlib>
#include <metal_command_buffer>
using namespace metal;

// Turns the per-object `cull_status` the single-source `cull.hlsl` decision
// kernel wrote into Metal indirect command buffers. One thread per command
// slot: a record whose status equals `draw_status` gets an indexed draw in its
// shader bucket's ICB and a reset in every other bucket; anything else resets
// every bucket. This is the only half of the cull Metal keeps hand-written,
// because the ICB encoding (`render_command`, `array<command_buffer, N>`) is a
// declaration HLSL cannot express.

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

// Mirrors gfx::render_types::GpuBucketRegion (32 B): a bucket's build-time
// records, then its runtime ones, back to back in its ICB.
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

// The command `record` takes in bucket region `r`'s ICB, or ~0 when it falls
// outside the region. Mirrors GpuBucketRegion::command less the offset, since
// each bucket has an ICB of its own.
static uint bucket_slot(constant BucketRegion &r, uint record) {
    uint k = record - r.first;
    if (k < r.count) {
        return k;
    }
    k = record - r.runtime_first;
    return k < r.runtime_count ? r.count + k : 0xffffffffu;
}

// Mirrors metal::uniforms::EncodeParams (32 B). The slot grid is
// `region_count * object_count`: one region for the main, phase-2 and mirror
// culls, one per shadow view for the shadow culls, where region `c` is the
// view's block of the shadow ICB and `region_mask` says which views were
// re-culled this frame (the others keep their prior commands). A dispatch
// covers the regions from `region_base` on, so it need not span the whole grid.
struct EncodeParams {
    uint object_count;
    uint region_count;
    uint region_mask;
    uint skinned_base;
    uint bucket_count;
    uint draw_status;
    uint region_base;
    uint _pad0;
};

// The per-bucket indirect command buffers, reached through an argument buffer.
// Only the first `bucket_count` entries are encoded; the kernel never
// constructs a command against an entry past that.
struct ICBContainer {
    array<command_buffer, MAX_SHADER_BUCKETS> icbs [[id(0)]];
};

kernel void cull_encode(
    constant GpuDrawArgs   *draw_args         [[buffer(1)]],
    const device uint      *index_buf         [[buffer(3)]],
    device ICBContainer    *icb_c             [[buffer(4)]],
    const device uint      *cull_status       [[buffer(5)]],
    const device uint      *skinned_index_buf [[buffer(6)]],
    constant EncodeParams  &p                 [[buffer(7)]],
    constant CullVariants  &variants          [[buffer(8)]],
    uint                    tid               [[thread_position_in_grid]]
) {
    uint slot = p.region_base * p.object_count + tid;
    if (slot >= p.region_count * p.object_count) {
        return;
    }
    uint region = slot / p.object_count;
    if ((p.region_mask & (1u << region)) == 0u) {
        return;
    }
    uint record = slot - region * p.object_count;
    GpuDrawArgs a = draw_args[record];
    uint word = cull_status[slot];
    bool draw = (word & CULL_OUTCOME_MASK) == p.draw_status;
    uint bucket = min(word >> CULL_BUCKET_SHIFT, p.bucket_count - 1u);
    // Records at or past `skinned_base` draw the compute-deformed geometry
    // through the skinned index buffer. The index buffer is part of the
    // indirect command on Metal, so it is picked here rather than bound per
    // draw range. `base_instance` carries the record id into the vertex stage.
    const device uint *ib = record >= p.skinned_base ? skinned_index_buf : index_buf;
    // A single stream writes the record's own slot. Otherwise every bucket
    // region holding the record is written each frame: a freed slot can be
    // reused by a record of a different bucket, which would otherwise leave
    // the old bucket's command stale and still executing.
    for (uint b = 0u; b < p.bucket_count; ++b) {
        uint index = p.bucket_count <= 1u ? slot : bucket_slot(variants.regions[b], record);
        if (index == 0xffffffffu) {
            continue;
        }
        render_command cmd(icb_c->icbs[b], index);
        if (b != bucket || !draw) {
            cmd.reset();
            continue;
        }
        cmd.draw_indexed_primitives(primitive_type::triangle,
                                    a.index_count,
                                    ib + a.index_offset,
                                    1u,
                                    a.base_vertex,
                                    record);
    }
}
