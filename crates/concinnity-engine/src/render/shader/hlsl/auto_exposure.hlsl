// Auto-exposure histogram kernels: single source for every backend.
//
// One entry per compile, selected by a define, so each variant declares exactly
// the resources it binds (Metal and DXIL assign slots in declaration order, so
// an unused declaration would shift the live ones):
//
//   AE_BUILD   - histogram_build: one thread per HISTOGRAM_STRIDE squared
//                block of the HDR resolve, metering its first pixel. Each
//                threadgroup keeps a local 256-bin histogram in groupshared
//                memory, atomically increments its own bin per pixel, and
//                merges the local counts into the global histogram on exit.
//                The local stage absorbs most of the contention, so the
//                per-frame global atomics scale to large resolves cheaply.
//   AE_AVERAGE - histogram_average: one threadgroup of 256 threads reduces the
//                histogram to the log-luminance the brightest
//                HIGHLIGHT_FRACTION of the frame starts at and the
//                count-weighted average log-luminance of the bins no more than
//                METERED_RANGE_EV under it, then clears it for the next frame.
//
// The HDR resolve is only ever fetched by integer coordinate, on every backend,
// so it is declared as a plain texture and no backend binds a sampler for this
// pass.
//
// Every resource takes its Metal index from the number on its `register()` (see
// concinnity-shader's `metal_bindings`), which is what puts the histogram on
// buffer(0) and the output measurement on buffer(1) without a table.
//
// HISTOGRAM_BINS, HISTOGRAM_STRIDE, HIGHLIGHT_FRACTION and METERED_RANGE_EV mirror
// gfx::auto_exposure; AutoExposureParams and ExposureMeasurement mirror
// render::uniforms.

static const uint HISTOGRAM_BINS = 256u;
static const uint HISTOGRAM_STRIDE = 2u;
static const float HIGHLIGHT_FRACTION = 0.001;
static const float METERED_RANGE_EV = 10.0;

struct AutoExposureParams
{
    // Lowest log2(luminance) the bins span. Pixels darker than this fall in
    // bin 0 and are weighted out by the average pass.
    float lum_log2_min;
    // Width of the log2(luminance) range the histogram covers.
    float lum_log2_range;
    // Pre-computed HISTOGRAM_BINS / lum_log2_range, so the build kernel maps a
    // centered log-luminance to a bin index without a per-pixel divide.
    float lum_to_bin_scale;
    // The scale the measured scene was written at, divided out so the bins
    // hold absolute luminance.
    float pre_exposure;
};

// The params slot is a host difference, not a target one: DirectX hands them
// over as root constants at b0, while Vulkan pushes them and the Metal encoder
// writes them past the buffers each kernel binds -- buffer(1) after the
// histogram, buffer(2) after the histogram and the output average. So the
// DirectX leg takes a branch of its own and the register number on the shared
// one is the Metal index.
#ifdef CN_BACKEND_DIRECTX
ConstantBuffer<AutoExposureParams> params : register(b0);
#elif defined(AE_BUILD)
[[vk::push_constant]] ConstantBuffer<AutoExposureParams> params : register(b1);
#else
[[vk::push_constant]] ConstantBuffer<AutoExposureParams> params : register(b2);
#endif

#ifdef AE_BUILD

[[vk::binding(0, 0)]] Texture2D<float4> hdr_texture : register(t0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> histogram : register(u0);

groupshared uint local_hist[HISTOGRAM_BINS];

[shader("compute")]
[numthreads(16, 16, 1)]
void histogram_build(uint3 gid : SV_DispatchThreadID, uint tid : SV_GroupIndex)
{
    // 16x16 == 256 == HISTOGRAM_BINS exactly, so one thread clears one bin.
    if (tid < HISTOGRAM_BINS)
    {
        local_hist[tid] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    uint w, h;
    hdr_texture.GetDimensions(w, h);
    uint2 pixel = gid.xy * HISTOGRAM_STRIDE;
    if (pixel.x < w && pixel.y < h)
    {
        float3 c = hdr_texture.Load(int3(int2(pixel), 0)).rgb;
        // Rec. 709 luminance. The floor keeps log2 finite on a fully black
        // pixel and sits far under any histogram floor, so black lands in the
        // sub-floor bin.
        float lum = max(dot(c, float3(0.2126, 0.7152, 0.0722)) / params.pre_exposure, 1.0e-30);
        float lum_log2 = clamp(
            log2(lum),
            params.lum_log2_min,
            params.lum_log2_min + params.lum_log2_range);
        float t = (lum_log2 - params.lum_log2_min) * params.lum_to_bin_scale;
        uint bin = min(uint(t), HISTOGRAM_BINS - 1u);
        uint prev;
        InterlockedAdd(local_hist[bin], 1u, prev);
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid < HISTOGRAM_BINS)
    {
        uint count = local_hist[tid];
        if (count > 0u)
        {
            uint prev;
            InterlockedAdd(histogram[tid], count, prev);
        }
    }
}

#elif defined(AE_AVERAGE)

struct ExposureMeasurement
{
    // Count-weighted mean log2 luminance of the bins whose centers sit no
    // more than METERED_RANGE_EV under the highlight, the sub-floor bin left
    // out.
    float mean_log_lum;
    // Log2 luminance the brightest HIGHLIGHT_FRACTION of all pixels starts at,
    // the sub-floor bin counted in.
    float highlight_log_lum;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> histogram : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<ExposureMeasurement> measurement : register(u1);

groupshared uint reduce_counts[HISTOGRAM_BINS];
groupshared float reduce_weighted[HISTOGRAM_BINS];
// Inclusive suffix sums: at[i] is the count of bins i and above.
groupshared uint counts_at_or_above[HISTOGRAM_BINS];
groupshared float highlight;

[shader("compute")]
[numthreads(256, 1, 1)]
void histogram_average(uint tid : SV_GroupIndex)
{
    uint count = histogram[tid];
    // Clear for the next frame's build pass. Safe here because every thread has
    // already read its bin and the reductions below run on the groupshared
    // copies.
    histogram[tid] = 0u;

    float step = params.lum_log2_range / float(HISTOGRAM_BINS);
    float center = params.lum_log2_min + (float(tid) + 0.5) * step;
    counts_at_or_above[tid] = count;
    if (tid == 0u)
    {
        highlight = params.lum_log2_min;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint offset = 1u; offset < HISTOGRAM_BINS; offset <<= 1u)
    {
        uint above = (tid + offset < HISTOGRAM_BINS) ? counts_at_or_above[tid + offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        counts_at_or_above[tid] += above;
        GroupMemoryBarrierWithGroupSync();
    }

    // The one bin the brightest pixels' count crosses into writes the
    // highlight, placed within the bin as if its pixels spread evenly. An
    // empty frame, or a crossing in the sub-floor bin, keeps the floor.
    uint total = counts_at_or_above[0];
    float wanted = HIGHLIGHT_FRACTION * float(total);
    uint at = counts_at_or_above[tid];
    uint above = (tid + 1u < HISTOGRAM_BINS) ? counts_at_or_above[tid + 1u] : 0u;
    if (tid > 0u && total > 0u && float(at) >= wanted && float(above) < wanted)
    {
        float inside = (wanted - float(above)) / float(max(count, 1u));
        highlight = params.lum_log2_min + (float(tid) + 1.0 - inside) * step;
    }
    GroupMemoryBarrierWithGroupSync();

    // The mean meters what the highlight lights: the sub-floor bin and
    // everything more than METERED_RANGE_EV under the highlight (black space
    // around a lit planet) are left out, so a mostly-dark frame does not peg it.
    bool metered = tid > 0u && center >= highlight - METERED_RANGE_EV;
    uint effective_count = metered ? count : 0u;
    reduce_counts[tid] = effective_count;
    reduce_weighted[tid] = center * float(effective_count);
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = HISTOGRAM_BINS / 2u; stride > 0u; stride >>= 1u)
    {
        if (tid < stride)
        {
            reduce_counts[tid] += reduce_counts[tid + stride];
            reduce_weighted[tid] += reduce_weighted[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u)
    {
        measurement[0].highlight_log_lum = highlight;
        measurement[0].mean_log_lum = (reduce_counts[0] > 0u)
            ? (reduce_weighted[0] / float(reduce_counts[0]))
            : params.lum_log2_min;
    }
}

#else
#error "auto_exposure.hlsl: define AE_BUILD or AE_AVERAGE"
#endif
