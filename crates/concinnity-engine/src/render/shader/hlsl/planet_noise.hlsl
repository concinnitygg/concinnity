// The planet's trilinear value noise, shared by the ground's normals and the
// sea's shore, which read the same continents.

// The planet height noise's hash step: `lcg_hash` in concinnity-core's
// math::noise.
uint ground_lcg(uint v)
{
    v = v * 1664525u + 1013904223u;
    return v ^ (v >> 16u);
}

// The height noise's value at lattice point `c`, in [0, 1].
float ground_lattice(int3 c, uint salt)
{
    uint h = ground_lcg(ground_lcg(ground_lcg(asuint(c.x) ^ salt) + asuint(c.y) * 0x85ebca6bu)
                        + asuint(c.z) * 0xc2b2ae35u);
    return float(h & 0xffffu) / 65535.0;
}

// One octave of the ground's trilinear value noise at place `t` within
// lattice cell `c`: xyz = its gradient along the lattice axes, w = its value.
float4 ground_noise_cell(int3 c, float3 t, uint salt)
{
    float3 s = t * t * (3.0 - 2.0 * t);
    float3 ds = 6.0 * t * (1.0 - t);
    float a = ground_lattice(c, salt);
    float b = ground_lattice(c + int3(1, 0, 0), salt);
    float d0 = ground_lattice(c + int3(0, 1, 0), salt);
    float d1 = ground_lattice(c + int3(1, 1, 0), salt);
    float e = ground_lattice(c + int3(0, 0, 1), salt);
    float f = ground_lattice(c + int3(1, 0, 1), salt);
    float g = ground_lattice(c + int3(0, 1, 1), salt);
    float h = ground_lattice(c + int3(1, 1, 1), salt);
    float k1 = b - a;
    float k2 = d0 - a;
    float k3 = e - a;
    float k4 = a - b - d0 + d1;
    float k5 = a - d0 - e + g;
    float k6 = a - b - e + f;
    float k7 = -a + b + d0 - d1 + e - f - g + h;
    float value = a + k1 * s.x + k2 * s.y + k3 * s.z + k4 * s.x * s.y + k5 * s.y * s.z
                + k6 * s.z * s.x + k7 * s.x * s.y * s.z;
    return float4(ds * float3(k1 + k4 * s.y + k6 * s.z + k7 * s.y * s.z,
                              k2 + k5 * s.z + k4 * s.x + k7 * s.z * s.x,
                              k3 + k6 * s.x + k5 * s.y + k7 * s.x * s.y),
                  value);
}

// One octave of the ground's trilinear value noise at lattice position `p`.
float4 ground_noise(float3 p, uint salt)
{
    float3 cell = floor(p);
    return ground_noise_cell(int3(cell), p - cell, salt);
}
