// The lattice the clouds' noise is drawn on: a hash and gradient noise over
// a tile of whole cells. Spliced by the noise volume's kernel and the weather
// map's. `render::environment::clouds::noise` mirrors it.

// The PCG hash over three lanes (Jarzynski and Olano 2020).
uint3 cloud_hash3(uint3 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

float3 cloud_hash3f(uint3 v)
{
    return float3(cloud_hash3(v) >> 8u) * (1.0 / 16777216.0);
}

uint3 cloud_wrap3(int3 c, int period)
{
    return uint3(((c % period) + period) % period);
}

float cloud_fade(float t)
{
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float cloud_corner(int3 cell, float3 f, int3 d, int period)
{
    float3 g = cloud_hash3f(cloud_wrap3(cell + d, period)) * 2.0 - 1.0;
    return dot(g, f - float3(d));
}

// Gradient noise at `p` (in cells) over a tile `period` cells across.
float cloud_perlin(float3 p, int period)
{
    int3 cell = int3(floor(p));
    float3 f = p - float3(cell);
    float3 u = float3(cloud_fade(f.x), cloud_fade(f.y), cloud_fade(f.z));
    float x00 = lerp(cloud_corner(cell, f, int3(0, 0, 0), period),
                     cloud_corner(cell, f, int3(1, 0, 0), period), u.x);
    float x10 = lerp(cloud_corner(cell, f, int3(0, 1, 0), period),
                     cloud_corner(cell, f, int3(1, 1, 0), period), u.x);
    float x01 = lerp(cloud_corner(cell, f, int3(0, 0, 1), period),
                     cloud_corner(cell, f, int3(1, 0, 1), period), u.x);
    float x11 = lerp(cloud_corner(cell, f, int3(0, 1, 1), period),
                     cloud_corner(cell, f, int3(1, 1, 1), period), u.x);
    return lerp(lerp(x00, x10, u.y), lerp(x01, x11, u.y), u.z);
}
