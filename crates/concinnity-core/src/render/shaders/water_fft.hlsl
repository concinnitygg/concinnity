// The simulated ocean: single source for every backend.
//
// One kernel per compile, picked by its gate, each declaring only what it
// binds. Each runs over one simulation (core::render::water::ocean), its
// cascades along the dispatch's second axis (third for the reduction), their
// layers from `params.grid.z` on:
//
//   WATER_FFT_ROWS     evolves each cascade's initial spectrum to this frame
//                      and runs the inverse FFT along every row, into `rows`
//   WATER_FFT_COLUMNS  runs the inverse FFT along every column of `rows` and
//                      writes mip 0 of both arrays
//   WATER_MIP          reduces both arrays' mip `L - 1` into mip `L` by 2x2
//                      box averages, which keeps the slope moments exact
//
// Each FFT group transforms one row or column of one cascade in threadgroup
// memory: a radix-2 Stockham transform, so each stage reads one buffer and
// writes the other in natural order. A cascade's spectrum is stored in FFT
// order (texel `i` holds frequency `i`, or `i - size` past the middle), so the
// transform needs no shift. Eight real fields come out of four complex
// transforms: each pair of real fields `A`, `B` rides one transform as
// `A + iB`.
//
// The outputs per texel, in the arrays' layer for the cascade:
//
//   displacement  (dx, dy, dz, jacobian): the horizontal displacement scaled
//                 by the choppiness, the height, and the Jacobian of the
//                 horizontal map, which falls below zero where crests fold
//   slopes        (sx, sz, sx^2, sz^2): the slope of the displaced surface and
//                 its square, whose mip averages give the slope variance
//
// Every resource takes its Metal index from its register number: the two
// buffers sit past the parameter block in the buffer namespace, and the four
// array views (two read-only, two writable) fold into the texture namespace in
// register order.

struct WaterFftParams
{
    uint4 grid;       // [texels per side, log2 of it, first layer, cascades]
    float4 lengths;   // each cascade's side, in meters
    float time_fraction;
    float choppiness;
    uint mip_size;
    uint _pad;
};

// The largest cascade side, and threads per FFT group: one butterfly per pair
// of points. Mirror `OceanQuality::MAX_SIZE` and `OCEAN_FFT_THREADS`.
static const uint WATER_FFT_MAX_SIZE = 256u;
static const uint WATER_FFT_THREADS = 128u;
// The reduction's square group. Mirrors `OCEAN_MIP_GROUP`.
static const uint WATER_MIP_GROUP = 8u;

static const float WATER_FFT_TAU = 6.28318530718;

[[vk::push_constant]] ConstantBuffer<WaterFftParams> params : register(b0);

#if defined(WATER_FFT_ROWS)
[[vk::binding(0, 0)]] StructuredBuffer<float4> spectrum : register(t5);
#endif
#if defined(WATER_FFT_ROWS) || defined(WATER_FFT_COLUMNS)
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> rows : register(u6);
#endif
#if defined(WATER_MIP)
// Read-only views of mip `L - 1`, so the reduction needs no typed loads from
// writable views.
[[vk::binding(2, 0)]] Texture2DArray<float4> src_displacement : register(t0);
[[vk::binding(3, 0)]] Texture2DArray<float4> src_slopes : register(t1);
#endif
#if defined(WATER_FFT_COLUMNS) || defined(WATER_MIP)
[[vk::binding(4, 0)]] [[vk::image_format("rgba16f")]]
RWTexture2DArray<float4> dst_displacement : register(u2);
[[vk::binding(5, 0)]] [[vk::image_format("rgba16f")]]
RWTexture2DArray<float4> dst_slopes : register(u3);
#endif

#if defined(WATER_FFT_ROWS) || defined(WATER_FFT_COLUMNS)

// Two complex points per float4, two float4 signals per point, ping-ponged.
groupshared float4 fft_data[2][2][WATER_FFT_MAX_SIZE];

float2 water_cmul(float2 a, float2 b)
{
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Both complex points of `v` times `w`.
float4 water_cmul2(float4 v, float2 w)
{
    return float4(water_cmul(v.xy, w), water_cmul(v.zw, w));
}

// The inverse FFT of `fft_data[0]` over `size` points; returns the buffer the
// result landed in. Thread `j` runs butterfly `j` of every stage.
uint water_ifft(uint j, uint size, uint log2_size)
{
    uint half_size = size >> 1u;
    uint src = 0u;
    for (uint s = 0u; s < log2_size; s++)
    {
        uint span = 1u << s;
        if (j < half_size)
        {
            uint k = j & (span - 1u);
            float2 w;
            sincos(WATER_FFT_TAU * float(k) / float(2u * span), w.y, w.x);
            uint out0 = ((j >> s) << (s + 1u)) + k;
            [unroll]
            for (uint p = 0u; p < 2u; p++)
            {
                float4 a = fft_data[src][p][j];
                float4 b = water_cmul2(fft_data[src][p][j + half_size], w);
                fft_data[1u - src][p][out0] = a + b;
                fft_data[1u - src][p][out0 + span] = a - b;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        src = 1u - src;
    }
    return src;
}

// Where texel (x, y) of `layer` lives in a `size`-square layer stack.
uint water_texel(uint layer, uint x, uint y, uint size)
{
    return (layer * size + y) * size + x;
}

#endif

#if defined(WATER_FFT_ROWS)

// The signed frequency of FFT index `i`. Mirrors `initial::frequency_index`.
float water_frequency(uint i, uint size)
{
    return i < size / 2u ? float(i) : float(i) - float(size);
}

// Two real fields' spectra `a`, `b` as one complex transform's: `a + i b`.
float2 water_pack(float2 a, float2 b)
{
    return a + float2(-b.y, b.x);
}

// `i z`.
float2 water_i(float2 z)
{
    return float2(-z.y, z.x);
}

// The four packed transforms texel (x, y) of `cascade` contributes this
// frame: the spectrum `h0(k) e^(-i w t) + conj(h0(-k)) e^(i w t)`, carrying
// each wave along k, and the spectra of the fields derived from it. The
// horizontal displacement `i k^ h` moves each point toward the nearest crest,
// so crests sharpen and troughs round out; its derivatives `i k (i k^ h)`
// are the Jacobian's terms. Mirrored by `ocean::fields` in core.
void water_evolve(uint x, uint y, uint cascade, out float4 pair0, out float4 pair1)
{
    uint size = params.grid.x;
    uint layer = params.grid.z + cascade;
    uint mask = size - 1u;
    float4 t = spectrum[water_texel(layer, x, y, size)];
    float4 m = spectrum[water_texel(layer, (size - x) & mask, (size - y) & mask, size)];
    // Whole turns per repeat times the fraction of it elapsed: the host
    // wrapped the time, so the phase never sees a large number.
    float2 e;
    sincos(WATER_FFT_TAU * frac(t.z * params.time_fraction), e.y, e.x);
    float2 h = water_cmul(t.xy, float2(e.x, -e.y)) + water_cmul(float2(m.x, -m.y), e);

    float2 k = float2(water_frequency(x, size), water_frequency(y, size))
               * (WATER_FFT_TAU / params.lengths[cascade]);
    float kl = length(k);
    float2 kn = kl > 0.0 ? k / kl : (float2)0.0;
    float2 dx = water_i(h) * kn.x;
    float2 dz = water_i(h) * kn.y;
    float2 jxz = h * (-kn.x * k.y);
    float2 sx = water_i(h) * k.x;
    float2 sz = water_i(h) * k.y;
    float2 jxx = h * (-kn.x * k.x);
    float2 jzz = h * (-kn.y * k.y);
    pair0 = float4(water_pack(dx, dz), water_pack(h, jxz));
    pair1 = float4(water_pack(sx, sz), water_pack(jxx, jzz));
}

[shader("compute")]
[numthreads(WATER_FFT_THREADS, 1, 1)]
void water_fft_rows(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint size = params.grid.x;
    uint half_size = size >> 1u;
    uint j = thread.x;
    uint y = group.x;
    uint cascade = group.y;
    if (j < half_size)
    {
        [unroll]
        for (uint h = 0u; h < 2u; h++)
        {
            uint x = j + h * half_size;
            float4 a, b;
            water_evolve(x, y, cascade, a, b);
            fft_data[0][0][x] = a;
            fft_data[0][1][x] = b;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint src = water_ifft(j, size, params.grid.y);
    if (j < half_size)
    {
        uint layer = params.grid.z + cascade;
        [unroll]
        for (uint h = 0u; h < 2u; h++)
        {
            uint x = j + h * half_size;
            uint at = 2u * water_texel(layer, x, y, size);
            rows[at] = fft_data[src][0][x];
            rows[at + 1u] = fft_data[src][1][x];
        }
    }
}

#elif defined(WATER_FFT_COLUMNS)

// The least the Jacobian of the horizontal map is taken to be when turning
// rest slopes into the displaced surface's: a folding crest keeps a steep
// but finite slope.
static const float WATER_MIN_JACOBIAN = 0.1;

[shader("compute")]
[numthreads(WATER_FFT_THREADS, 1, 1)]
void water_fft_columns(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint size = params.grid.x;
    uint half_size = size >> 1u;
    uint j = thread.x;
    uint x = group.x;
    uint cascade = group.y;
    uint layer = params.grid.z + cascade;
    if (j < half_size)
    {
        [unroll]
        for (uint h = 0u; h < 2u; h++)
        {
            uint y = j + h * half_size;
            uint at = 2u * water_texel(layer, x, y, size);
            fft_data[0][0][y] = rows[at];
            fft_data[0][1][y] = rows[at + 1u];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint src = water_ifft(j, size, params.grid.y);
    if (j < half_size)
    {
        float chop = params.choppiness;
        [unroll]
        for (uint h = 0u; h < 2u; h++)
        {
            uint y = j + h * half_size;
            // (dx, dz, height, d dx / dz) and (sx, sz, d dx / dx, d dz / dz).
            float4 a = fft_data[src][0][y];
            float4 b = fft_data[src][1][y];
            float jxx = 1.0 + chop * b.z;
            float jzz = 1.0 + chop * b.w;
            float jxz = chop * a.w;
            float jacobian = jxx * jzz - jxz * jxz;
            // The rest slopes through the inverse Jacobian: the slope of the
            // surface where the displacement carried the point.
            float2 slope = float2(jzz * b.x - jxz * b.y, jxx * b.y - jxz * b.x)
                           / max(jacobian, WATER_MIN_JACOBIAN);
            uint3 texel = uint3(x, y, layer);
            dst_displacement[texel] = float4(chop * a.x, a.z, chop * a.y, jacobian);
            dst_slopes[texel] = float4(slope, slope * slope);
        }
    }
}

#elif defined(WATER_MIP)

[shader("compute")]
[numthreads(WATER_MIP_GROUP, WATER_MIP_GROUP, 1)]
void water_mip(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= params.mip_size || id.y >= params.mip_size)
    {
        return;
    }
    uint layer = params.grid.z + id.z;
    int2 s = int2(id.xy * 2u);
    int4 t00 = int4(s, layer, 0);
    int4 t10 = int4(s + int2(1, 0), layer, 0);
    int4 t01 = int4(s + int2(0, 1), layer, 0);
    int4 t11 = int4(s + int2(1, 1), layer, 0);
    uint3 dst = uint3(id.xy, layer);
    dst_displacement[dst] = 0.25 * (src_displacement.Load(t00) + src_displacement.Load(t10)
                                    + src_displacement.Load(t01) + src_displacement.Load(t11));
    dst_slopes[dst] = 0.25 * (src_slopes.Load(t00) + src_slopes.Load(t10) + src_slopes.Load(t01)
                              + src_slopes.Load(t11));
}

#endif
