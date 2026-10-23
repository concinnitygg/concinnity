// The water's wake: single source for every backend.
//
// One kernel per compile, picked by its gate, each declaring only what it
// binds. Together they run the window of water around the camera
// (core::render::environment::water::wake) one frame forward as linear
// deep-water waves:
//
//   WATER_WAKE_ROWS     reads last frame's field through the window's move
//                       (and the simulated frame's, when it moved), pushes in
//                       this frame's stirs, damps the band at the window's
//                       edge, steps the foam, and transforms two rows of
//                       `height + i rise` per group into `work`
//   WATER_WAKE_COLUMNS  transforms a column and its mirror per group, turns
//                       every wave by its own frequency, and transforms them
//                       back
//   WATER_WAKE_INVERSE  transforms two rows per group back into this frame's
//                       half of `state`
//   WATER_WAKE_SHADE    takes each cell's slope, foams the steep ones, and
//                       writes mip 0 of the field the water draw samples:
//                       (height, slope x, slope z, foam)
//   WATER_WAKE_MIP      reduces the field's mip `L - 1` into mip `L` by 2x2
//                       box averages
//
// The window is addressed toroidally: a cell's slot is its coordinates
// wrapped by the window's size, so the transforms run over slots, a torus on
// which the waves are the same wherever the window lies. `state` keeps two
// halves of (height, rise, foam, _), this frame's and last frame's, so the
// rows read last frame's field wherever the frame's move puts it.
//
// Each FFT group transforms one pair of lines in threadgroup memory, a
// radix-2 Stockham transform that reads one buffer and writes the other in
// natural order, two complex signals per float4. The forward transform is
// unscaled and the inverse divides by the cell count.
//
// Every resource takes its Metal index from its register number: the two
// buffers sit past the parameter block in the buffer namespace, and the
// field's views in the texture namespace.

struct WaterWakeStir
{
    float4 now;  // [x, z, radius, volume] this frame
    float4 prev; // the same last frame, in this frame's coordinates
};

// Matches `WaterWakeParams` (2160 B).
struct WaterWakeParams
{
    int4 window;     // this frame's corner cell, last frame's
    float4 carry[2]; // rows over (x, z, 1) into last frame's x, z
    uint4 grid;      // [cells per side, log2 of it, stirs, flags]
    float4 cell;     // [cell size, time step, gravity, sponge width]
    float4 damping;  // [per second, per second per k^2, sponge per second, _]
    float4 foam;     // [kept, per meter stirred, steepness onset, per second past it]
    WaterWakeStir stirs[64];
};

// Mirror `WAKE_RESOLUTION`, `MAX_WAKE_STIRS`, `STIR_SPREAD` and the flags in
// core.
static const uint WATER_WAKE_SIZE = 256u;
static const uint WATER_WAKE_THREADS = 128u;
static const uint WATER_WAKE_MAX_STIRS = 64u;
static const float WATER_WAKE_SPREAD = 2.0;
static const uint WATER_WAKE_GROUP = 8u;
static const uint WATER_WAKE_LAST_VALID = 1u;
static const uint WATER_WAKE_CARRIED = 2u;
static const uint WATER_WAKE_WRITE_SECOND = 4u;

static const float WATER_WAKE_TAU = 6.28318530718;
static const float WATER_WAKE_PI = 3.14159265359;

#if !defined(WATER_WAKE_MIP)
[[vk::binding(0, 0)]] ConstantBuffer<WaterWakeParams> params : register(b0);
#endif
#if defined(WATER_WAKE_ROWS) || defined(WATER_WAKE_INVERSE) || defined(WATER_WAKE_SHADE)
// Two halves of (height, rise, foam, _), slot after slot.
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> state : register(u1);
#endif
#if defined(WATER_WAKE_ROWS) || defined(WATER_WAKE_COLUMNS) || defined(WATER_WAKE_INVERSE)
// The transform between the passes, one complex value per slot.
[[vk::binding(2, 0)]] RWStructuredBuffer<float2> work : register(u2);
#endif
#if defined(WATER_WAKE_SHADE) || defined(WATER_WAKE_MIP)
[[vk::binding(3, 0)]] [[vk::image_format("rgba16f")]]
RWTexture2D<float4> dst_field : register(u3);
#endif
#if defined(WATER_WAKE_MIP)
[[vk::binding(4, 0)]] Texture2D<float4> src_field : register(t4);
#endif

#if !defined(WATER_WAKE_MIP)

uint wake_size()
{
    return params.grid.x;
}

// The slot cell `cell` is kept in. Mirrors `window::slot_of`.
uint wake_slot(int2 cell)
{
    uint mask = wake_size() - 1u;
    return (uint(cell.x) & mask) + (uint(cell.y) & mask) * wake_size();
}

// Where slot `s` of half `half` sits in `state`.
uint wake_at(uint half_index, uint s)
{
    return half_index * wake_size() * wake_size() + s;
}

uint wake_write_half()
{
    return (params.grid.w & WATER_WAKE_WRITE_SECOND) != 0u ? 1u : 0u;
}

// The cell this frame's window keeps in slot (x, y).
int2 wake_cell(uint2 s)
{
    int2 o = params.window.xy;
    int mask = int(wake_size()) - 1;
    return o + ((int2(s) - o) & mask);
}

#endif

#if defined(WATER_WAKE_ROWS) || defined(WATER_WAKE_COLUMNS) || defined(WATER_WAKE_INVERSE)

// Two complex signals per float4, ping-ponged.
groupshared float4 wake_fft_data[2][WATER_WAKE_SIZE];

float2 wake_cmul(float2 a, float2 b)
{
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// The transform of `wake_fft_data[0]`, forward for a `sign` of -1 and back
// for +1; returns the buffer the result landed in. Thread `j` runs
// butterfly `j` of every stage.
uint wake_fft(uint j, float sign)
{
    uint size = wake_size();
    uint half_size = size >> 1u;
    uint src = 0u;
    for (uint s = 0u; s < params.grid.y; s++)
    {
        uint span = 1u << s;
        if (j < half_size)
        {
            uint k = j & (span - 1u);
            float2 w;
            sincos(sign * WATER_WAKE_TAU * float(k) / float(2u * span), w.y, w.x);
            uint out0 = ((j >> s) << (s + 1u)) + k;
            float4 a = wake_fft_data[src][j];
            float4 t = wake_fft_data[src][j + half_size];
            float4 b = float4(wake_cmul(t.xy, w), wake_cmul(t.zw, w));
            wake_fft_data[1u - src][out0] = a + b;
            wake_fft_data[1u - src][out0 + span] = a - b;
        }
        GroupMemoryBarrierWithGroupSync();
        src = 1u - src;
    }
    return src;
}

#endif

#if defined(WATER_WAKE_ROWS)

// Whether last frame's window held `cell`, in last frame's coordinates.
bool wake_held(int2 cell)
{
    int2 o = params.window.zw;
    int n = int(wake_size());
    return all(cell >= o) && all(cell < o + n);
}

// Last frame's (height, rise, foam, _) in `cell` of its own window, or still
// water where it held none.
float4 wake_last(int2 cell)
{
    if ((params.grid.w & WATER_WAKE_LAST_VALID) == 0u || !wake_held(cell))
    {
        return (float4)0.0;
    }
    return state[wake_at(1u - wake_write_half(), wake_slot(cell))];
}

// Last frame's field under point `p` of this frame: the cell itself unless
// the simulated frame moved, else bilinear at where the move came from.
float4 wake_carried(int2 cell, float2 p)
{
    if ((params.grid.w & WATER_WAKE_CARRIED) == 0u)
    {
        return wake_last(cell);
    }
    float2 q = float2(dot(params.carry[0].xyz, float3(p, 1.0)), dot(params.carry[1].xyz, float3(p, 1.0)));
    float2 f = q / params.cell.x - 0.5;
    int2 c = int2(floor(f));
    float2 t = f - float2(c);
    float4 a = lerp(wake_last(c), wake_last(c + int2(1, 0)), t.x);
    float4 b = lerp(wake_last(c + int2(0, 1)), wake_last(c + int2(1, 1)), t.x);
    return lerp(a, b, t.y);
}

// The share of a stir's volume per square meter at squared distance `d2`
// from a piece of `radius`. Mirrors `model::disc`.
float wake_disc(float d2, float radius)
{
    float r = max(WATER_WAKE_SPREAD * radius, 2.0 * params.cell.x);
    float d = sqrt(d2);
    if (d >= r)
    {
        return 0.0;
    }
    return (0.5 + 0.5 * cos(WATER_WAKE_PI * d / r)) / (r * r * (0.5 * WATER_WAKE_PI - 2.0 / WATER_WAKE_PI));
}

// The height stir `s` pushes in at `p`. Mirrors `model::stir`.
float wake_stir(WaterWakeStir s, float2 p)
{
    float2 a = p - s.now.xy;
    float2 b = p - s.prev.xy;
    return s.now.w * wake_disc(dot(a, a), s.now.z) - s.prev.w * wake_disc(dot(b, b), s.prev.z);
}

// The share of the field a cell keeps `inside` meters in from the window's
// edge. Mirrors `model::sponge`.
float wake_sponge(float inside)
{
    float t = saturate(1.0 - inside / params.cell.w);
    return exp(-params.damping.z * t * t * params.cell.y);
}

// Meters from the center of `cell` to the window's nearest edge.
float wake_inside(int2 cell)
{
    float n = float(wake_size());
    float2 lo = float2(cell - params.window.xy) + 0.5;
    float2 hi = n - lo;
    return min(min(lo.x, lo.y), min(hi.x, hi.y)) * params.cell.x;
}

[shader("compute")]
[numthreads(WATER_WAKE_THREADS, 1, 1)]
void water_wake_rows(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint size = wake_size();
    uint half_size = size >> 1u;
    uint j = thread.x;
    uint write = wake_write_half();
    if (j < half_size)
    {
        [unroll]
        for (uint h = 0u; h < 2u; h++)
        {
            uint x = j + h * half_size;
            float4 lines = (float4)0.0;
            [unroll]
            for (uint r = 0u; r < 2u; r++)
            {
                uint2 s = uint2(x, 2u * group.x + r);
                int2 cell = wake_cell(s);
                float2 p = (float2(cell) + 0.5) * params.cell.x;
                float4 last = wake_carried(cell, p);
                float push = 0.0;
                for (uint i = 0u; i < params.grid.z; i++)
                {
                    push += wake_stir(params.stirs[i], p);
                }
                float keep = wake_sponge(wake_inside(cell));
                float height = (last.x + push) * keep;
                float rise = last.y * keep;
                float foam = min(last.z * params.foam.x + params.foam.y * abs(push), 1.0) * keep;
                state[wake_at(write, s.x + s.y * size)] = float4(height, rise, foam, 0.0);
                float2 z = float2(height, rise);
                lines = r == 0u ? float4(z, lines.zw) : float4(lines.xy, z);
            }
            wake_fft_data[0][x] = lines;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint src = wake_fft(j, -1.0);
    if (j < half_size)
    {
        [unroll]
        for (uint h = 0u; h < 2u; h++)
        {
            uint x = j + h * half_size;
            float4 t = wake_fft_data[src][x];
            work[x + 2u * group.x * size] = t.xy;
            work[x + (2u * group.x + 1u) * size] = t.zw;
        }
    }
}

#elif defined(WATER_WAKE_COLUMNS)

// The signed frequency of FFT index `i`.
float wake_frequency(uint i)
{
    uint size = wake_size();
    return i < size / 2u ? float(i) : float(i) - float(size);
}

// The packed spectrum `z` at wavenumber `k`, whose mirror at `-k` is `m`,
// after one step. Mirrors `model::propagate`.
float2 wake_propagate(float2 z, float2 m, float k)
{
    float2 h = float2(0.5 * (z.x + m.x), 0.5 * (z.y - m.y));
    float2 v = float2(0.5 * (z.y + m.y), -0.5 * (z.x - m.x));
    float dt = params.cell.y;
    float w = sqrt(params.cell.z * k);
    float s;
    float c;
    sincos(w * dt, s, c);
    float sw = w > 1e-4 ? s / w : dt;
    float keep = exp(-(params.damping.x + params.damping.y * k * k) * dt);
    float2 h2 = (h * c + v * sw) * keep;
    float2 v2 = (v * c - h * (w * s)) * keep;
    return float2(h2.x - v2.y, h2.y + v2.x);
}

[shader("compute")]
[numthreads(WATER_WAKE_THREADS, 1, 1)]
void water_wake_columns(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint size = wake_size();
    uint mask = size - 1u;
    uint half_size = size >> 1u;
    uint j = thread.x;
    uint kx = group.x;
    uint px = (size - kx) & mask;
    if (j < half_size)
    {
        [unroll]
        for (uint h = 0u; h < 2u; h++)
        {
            uint y = j + h * half_size;
            wake_fft_data[0][y] = float4(work[kx + y * size], work[px + y * size]);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint src = wake_fft(j, -1.0);
    float4 turned[2];
    float dk = WATER_WAKE_TAU / (float(size) * params.cell.x);
    [unroll]
    for (uint h = 0u; h < 2u; h++)
    {
        uint y = j + h * half_size;
        float4 a = wake_fft_data[src][y];
        float4 m = wake_fft_data[src][(size - y) & mask];
        float fy = wake_frequency(y);
        float k0 = dk * length(float2(wake_frequency(kx), fy));
        float k1 = dk * length(float2(wake_frequency(px), fy));
        turned[h] = float4(wake_propagate(a.xy, m.zw, k0), wake_propagate(a.zw, m.xy, k1));
    }
    GroupMemoryBarrierWithGroupSync();
    if (j < half_size)
    {
        [unroll]
        for (uint h = 0u; h < 2u; h++)
        {
            wake_fft_data[0][j + h * half_size] = turned[h];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    src = wake_fft(j, 1.0);
    if (j < half_size)
    {
        [unroll]
        for (uint h = 0u; h < 2u; h++)
        {
            uint y = j + h * half_size;
            float4 t = wake_fft_data[src][y];
            work[kx + y * size] = t.xy;
            if (px != kx)
            {
                work[px + y * size] = t.zw;
            }
        }
    }
}

#elif defined(WATER_WAKE_INVERSE)

[shader("compute")]
[numthreads(WATER_WAKE_THREADS, 1, 1)]
void water_wake_inverse(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint size = wake_size();
    uint half_size = size >> 1u;
    uint j = thread.x;
    uint y0 = 2u * group.x;
    if (j < half_size)
    {
        [unroll]
        for (uint h = 0u; h < 2u; h++)
        {
            uint x = j + h * half_size;
            wake_fft_data[0][x] = float4(work[x + y0 * size], work[x + (y0 + 1u) * size]);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint src = wake_fft(j, 1.0);
    if (j < half_size)
    {
        float scale = 1.0 / float(size * size);
        uint write = wake_write_half();
        [unroll]
        for (uint h = 0u; h < 2u; h++)
        {
            uint x = j + h * half_size;
            float4 t = wake_fft_data[src][x] * scale;
            uint a = wake_at(write, x + y0 * size);
            uint b = wake_at(write, x + (y0 + 1u) * size);
            state[a] = float4(t.xy, state[a].z, 0.0);
            state[b] = float4(t.zw, state[b].z, 0.0);
        }
    }
}

#elif defined(WATER_WAKE_SHADE)

[shader("compute")]
[numthreads(WATER_WAKE_GROUP, WATER_WAKE_GROUP, 1)]
void water_wake_shade(uint3 id : SV_DispatchThreadID)
{
    uint size = wake_size();
    if (id.x >= size || id.y >= size)
    {
        return;
    }
    uint mask = size - 1u;
    uint write = wake_write_half();
    float height_at[4];
    uint2 n[4] = {
        uint2((id.x + 1u) & mask, id.y), uint2((id.x - 1u) & mask, id.y),
        uint2(id.x, (id.y + 1u) & mask), uint2(id.x, (id.y - 1u) & mask)
    };
    [unroll]
    for (uint i = 0u; i < 4u; i++)
    {
        height_at[i] = state[wake_at(write, n[i].x + n[i].y * size)].x;
    }
    float2 slope = float2(height_at[0] - height_at[1], height_at[2] - height_at[3]) / (2.0 * params.cell.x);
    uint at = wake_at(write, id.x + id.y * size);
    float4 here = state[at];
    float steep = max(length(slope) - params.foam.z, 0.0);
    float foam = min(here.z + steep * params.foam.w * params.cell.y, 1.0);
    state[at] = float4(here.xy, foam, 0.0);
    dst_field[id.xy] = float4(here.x, slope, foam);
}

#elif defined(WATER_WAKE_MIP)

[shader("compute")]
[numthreads(WATER_WAKE_GROUP, WATER_WAKE_GROUP, 1)]
void water_wake_mip(uint3 id : SV_DispatchThreadID)
{
    uint w;
    uint h;
    dst_field.GetDimensions(w, h);
    if (id.x >= w || id.y >= h)
    {
        return;
    }
    int2 s = int2(id.xy * 2u);
    dst_field[id.xy] = 0.25 * (src_field.Load(int3(s, 0)) + src_field.Load(int3(s + int2(1, 0), 0))
                               + src_field.Load(int3(s + int2(0, 1), 0))
                               + src_field.Load(int3(s + int2(1, 1), 0)));
}

#endif
