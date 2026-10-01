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

// The record's shader bucket rides the upper flag bits; values and layout are
// locked to gfx::render_types::{DrawArgsFlags::BUCKET_SHIFT, MAX_SHADER_BUCKETS}.
constant uint DRAW_BUCKET_SHIFT  = 8u;
constant uint DRAW_BUCKET_MASK   = 0xffu;
constant uint MAX_SHADER_BUCKETS = 8u;

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
    bool draw = cull_status[slot] == p.draw_status;
    // Every bucket's slot is written each frame: a freed slot can be reused by
    // a record of a different bucket, which would otherwise leave the old
    // bucket's command stale and still executing.
    uint bucket = min((a.flags >> DRAW_BUCKET_SHIFT) & DRAW_BUCKET_MASK,
                      p.bucket_count - 1u);
    // Records at or past `skinned_base` draw the compute-deformed geometry
    // through the skinned index buffer. The index buffer is part of the
    // indirect command on Metal, so it is picked here rather than bound per
    // draw range. `base_instance` carries the record id into the vertex stage.
    const device uint *ib = record >= p.skinned_base ? skinned_index_buf : index_buf;
    for (uint b = 0u; b < p.bucket_count; ++b) {
        render_command cmd(icb_c->icbs[b], slot);
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
