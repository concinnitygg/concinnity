// Auto-exposure histogram kernels: single source for every backend.
//
// One entry per compile, selected by a define, so each variant declares exactly
// the resources it binds (Metal and DXIL assign slots in declaration order, so
// an unused declaration would shift the live ones):
//
//   AE_BUILD   - histogram_build: one thread per HDR-resolve pixel. Each
//                threadgroup keeps a local 256-bin histogram in groupshared
//                memory, atomically increments its own bin per pixel, and
//                merges the local counts into the global histogram on exit.
//                The local stage absorbs most of the contention, so the
//                per-frame global atomics scale to large resolves cheaply.
//   AE_AVERAGE - histogram_average: one threadgroup of 256 threads reduces the
//                histogram to a single count-weighted average log-luminance and
//                clears it for the next frame.
//
// The HDR resolve is only ever fetched by integer coordinate, on every backend,
// so it is declared as a plain texture rather than a combined texture-sampler:
// Metal then binds no sampler for this pass, exactly as it did before the port.
// Vulkan's descriptor at set 0 binding 0 is a COMBINED_IMAGE_SAMPLER and stays
// one -- a combined descriptor satisfies a sampled-image declaration, so the
// set layout is untouched.
//
// HISTOGRAM_BINS mirrors gfx::auto_exposure::HISTOGRAM_BINS; AutoExposureParams
// mirrors the 16-byte struct each backend's uniforms module pushes.

static const uint HISTOGRAM_BINS = 256u;

struct AutoExposureParams
{
    // Lowest log2(luminance) the bins span. Pixels darker than this fall in
    // bin 0 and are weighted out by the average pass.
    float lum_log2_min;
    // Width of the log2(luminance) range the histogram covers.
    float lum_log2_range;
    // Pre-computed HISTOGRAM_BINS / lum_log2_range, so the build kernel maps a
    // centred log-luminance to a bin index without a per-pixel divide.
    float lum_to_bin_scale;
    float _pad;
};

// The params slot is a host difference, not a target one: Vulkan pushes them
// and DirectX hands them over as root constants at b0 (which is where a bare
// push constant lands on that target), while the Metal encoder writes them to a
// different buffer index per kernel. METAL_BINDINGS carries that index.
#ifdef METAL_BINDINGS
#ifdef AE_BUILD
ConstantBuffer<AutoExposureParams> params : register(b1);
#else
ConstantBuffer<AutoExposureParams> params : register(b2);
#endif
#else
[[vk::push_constant]] ConstantBuffer<AutoExposureParams> params;
#endif

#ifdef AE_BUILD

[[vk::binding(0, 0)]] Texture2D<float4> hdr_texture;
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
    if (gid.x < w && gid.y < h)
    {
        float3 c = hdr_texture.Load(int3(int2(gid.xy), 0)).rgb;
        // Rec. 709 luminance. The 1e-6 floor keeps log2 finite on a fully black
        // pixel, which would otherwise reach bin 0 through a -inf clamp.
        float lum = max(dot(c, float3(0.2126, 0.7152, 0.0722)), 1.0e-6);
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

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> histogram : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output_avg : register(u1);

groupshared uint reduce_counts[HISTOGRAM_BINS];
groupshared float reduce_weighted[HISTOGRAM_BINS];

[shader("compute")]
[numthreads(256, 1, 1)]
void histogram_average(uint tid : SV_GroupIndex)
{
    uint count = histogram[tid];
    // Clear for the next frame's build pass. Safe here because every thread has
    // already read its bin and the reduction below runs on the groupshared
    // copies.
    histogram[tid] = 0u;

    // Drop the sub-floor bin so a mostly-black frame does not peg the average.
    uint effective_count = (tid == 0u) ? 0u : count;
    float step = params.lum_log2_range / float(HISTOGRAM_BINS);
    float centre = params.lum_log2_min + (float(tid) + 0.5) * step;
    reduce_counts[tid] = effective_count;
    reduce_weighted[tid] = centre * float(effective_count);
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
        output_avg[0] = (reduce_counts[0] > 0u)
            ? (reduce_weighted[0] / float(reduce_counts[0]))
            : params.lum_log2_min;
    }
}

#else
#error "auto_exposure.slang: define AE_BUILD or AE_AVERAGE"
#endif
