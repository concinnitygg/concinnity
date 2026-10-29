// What the draw cull (cull.hlsl) and the kernels that pack its draws into
// per-stream lists (cull_compact.hlsl) share: the status word, the variant
// block's rows, the streams, and the indirect command record.

// `cull_status` values. STATUS_HIZ_CANDIDATE is the only outcome phase 2
// re-tests against the rebuilt pyramid; the others are settled by phase 1.
static const uint STATUS_DRAWN         = 0u; // visible in phase 1 -> never re-tested
static const uint STATUS_HIZ_CANDIDATE = 1u; // Hi-Z-occluded in phase 1 -> candidate
static const uint STATUS_CULLED        = 2u; // frustum/distance/disabled -> settled
static const uint STATUS_REDRAW        = 3u; // candidate found visible in phase 2
static const uint STATUS_HIZ_CULLED    = 4u; // candidate still occluded in phase 2

// Phase 1 keeps the outcome in a status word's low byte and the record's
// shader bucket in the byte above, so phase 2 and the packing kernels draw
// the record where phase 1 routed it. Locked to gfx::cull_status.
static const uint CULL_OUTCOME_MASK = 0xFFu;
static const uint CULL_BUCKET_SHIFT = 8u;

// The record's variant route plus one rides flag bits 16..22, and whether it
// grows grass with a far band bit 3. Locked to gfx::render_types::DrawArgsFlags.
static const uint DRAW_FAR_BAND        = 8u;
static const uint DRAW_ROUTE_SHIFT     = 16u;
static const uint DRAW_ROUTE_MASK      = 0x3Fu;

// Locked to gfx::render_types::{MAX_SHADER_BUCKETS, MAX_VARIANT_ROUTES,
// VARIANT_FAR, VARIANT_DAYLIGHT, DAYLIGHT_COS_MARGIN}.
static const uint  MAX_SHADER_BUCKETS  = 64u;
static const uint  MAX_VARIANT_ROUTES  = 32u;
static const uint  VARIANT_FAR         = 1u;
static const uint  VARIANT_DAYLIGHT    = 2u;
static const float DAYLIGHT_COS_MARGIN = 1e-6;

// The variant block, gfx::render_types::GpuCullVariants, read as 16-byte rows
// (a structured buffer element may not reach its size): the view's four rows,
// then a row per route (the bucket per feature combination, a byte each, and
// the features tested), then two rows per bucket region (its build-time
// records' first and count, its runtime ones', then the region's offset).
static const uint CULL_VARIANT_ROUTE_ROW  = 4u;
static const uint CULL_VARIANT_REGION_ROW = 36u;
static const uint CULL_VARIANT_ROWS       = 164u;

// The whole block as one constant buffer, which is how Metal takes it: it is
// bound inline there, and inline bytes cannot back a `device` argument.
struct CullVariantRows
{
    uint4 rows[CULL_VARIANT_ROWS];
};

// The streams a cull packs its draws into: a bucket each, then the skinned
// tail, counted per 64-record group. Locked to gfx::render_types::{
// CULL_GROUP_RECORDS, DRAW_STREAMS, SKINNED_STREAM}.
static const uint CULL_GROUP_RECORDS = 64u;
static const uint DRAW_STREAMS       = 65u;
static const uint SKINNED_STREAM     = 64u;
static const uint NO_STREAM          = 0xFFFFFFFFu;

// Bucket `b`'s region in the variant block named `rows`: its spans, and the
// offset of its first command in a shared indirect buffer.
#define CULL_REGION_SPANS(rows, b)  rows[CULL_VARIANT_REGION_ROW + 2u * (b)]
#define CULL_REGION_OFFSET(rows, b) rows[CULL_VARIANT_REGION_ROW + 2u * (b) + 1u].x

// Record `i`'s place in a region with `spans`, or ~0 when it falls outside
// both. Mirrors GpuBucketRegion::command less the offset.
uint region_place(uint4 spans, uint i)
{
    uint k = i - spans.x;
    if (k < spans.y)
    {
        return k;
    }
    k = i - spans.z;
    return k < spans.w ? spans.y + k : 0xFFFFFFFFu;
}

// The stream record `i`, routed to `bucket` whose region has `spans`, draws
// in, or NO_STREAM when it falls outside the region. Mirrors
// StreamLayout::stream.
uint record_stream(uint i, uint bucket, uint bucket_count, uint skinned_base, uint4 spans)
{
    if (bucket_count <= 1u || bucket == 0u)
    {
        return i >= skinned_base ? SKINNED_STREAM : 0u;
    }
    return region_place(spans, i) != 0xFFFFFFFFu ? bucket : NO_STREAM;
}

#ifndef CN_BACKEND_METAL

#ifdef CN_BACKEND_DIRECTX

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

#endif // !CN_BACKEND_METAL
