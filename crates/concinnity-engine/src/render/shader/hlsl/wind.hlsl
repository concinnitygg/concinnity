// The world's wind, spliced at a shader's WIND marker: what anything swaying
// in it evaluates. The two float4 rows are `WindField::gpu_rows` in
// render::wind; how a shader binds them is its own business.
//
// The air moves along `dir` at `strength` meters per second. Gusts are a
// smooth noise field about `1 / inv_gust_scale` meters across, carried
// downwind at the wind's own speed, that swing the speed by up to
// `gustiness` of it either way.

struct Wind
{
    float2 dir;
    float strength;
    float gustiness;
    float inv_gust_scale;
};

Wind wind_unpack(float4 dir_strength_gustiness, float4 gust)
{
    Wind w;
    w.dir = dir_strength_gustiness.xy;
    w.strength = dir_strength_gustiness.z;
    w.gustiness = dir_strength_gustiness.w;
    w.inv_gust_scale = gust.x;
    return w;
}

// A unit float from a lattice point, for the gust noise.
float wind_lattice(float2 cell)
{
    uint2 c = uint2(int2(cell));
    uint h = c.x * 1597334677u ^ c.y * 3812015801u;
    h = (h ^ (h >> 16u)) * 2246822519u;
    h ^= h >> 13u;
    return float(h >> 8u) * (1.0 / 16777216.0);
}

// Smooth value noise in [0, 1].
float wind_noise(float2 p)
{
    float2 cell = floor(p);
    float2 f = p - cell;
    float2 s = f * f * (3.0 - 2.0 * f);
    float a = wind_lattice(cell);
    float b = wind_lattice(cell + float2(1.0, 0.0));
    float c = wind_lattice(cell + float2(0.0, 1.0));
    float d = wind_lattice(cell + float2(1.0, 1.0));
    return lerp(lerp(a, b, s.x), lerp(c, d, s.x), s.y);
}

// The wind speed at ground position `xz` at time `t`, in meters per second
// along `w.dir`. Never negative: a calm gust stops the air rather than
// reversing it.
float wind_speed(Wind w, float2 xz, float t)
{
    float2 p = (xz - w.dir * (w.strength * t)) * w.inv_gust_scale;
    float gust = wind_noise(p) * 0.65 + wind_noise(p * 2.13 + 17.0) * 0.35;
    return w.strength * max(1.0 + w.gustiness * (2.0 * gust - 1.0), 0.0);
}
