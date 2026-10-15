// What a grass field is wherever it is evaluated: the hashes its blades are
// placed and varied with, the noise its meadow and its slow color drift are
// laid out by, its clump lattice, and the mapping between a planet's
// directions and its cube faces. The grass kernel places blades with these,
// and the main pass shades the far field past the blades with the same
// functions, so the two agree wherever they meet.

static const float GRASS_TAU = 6.28318530718;

// One 32-bit hash step (PCG).
uint grass_hash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint grass_hash2(int2 c)
{
    return grass_hash(asuint(c.x) + grass_hash(asuint(c.y)));
}

// A unit float from `h` mixed with `salt`, so one hash yields many draws.
float grass_unit(uint h, uint salt)
{
    return float(grass_hash(h ^ salt) >> 8u) * (1.0 / 16777216.0);
}

// The clump `p` (meters on the ground's plane) belongs to, for clumps
// `clump_size` across salted by `seed` and `salt`: the nearest center of a
// jittered lattice of clump-sized cells. Returns its hash and writes its
// center.
uint grass_clump_at(float2 p_m, float clump_size, uint seed, uint salt, out float2 center)
{
    float2 p = p_m / clump_size;
    int2 base = int2(floor(p));
    float best = 1e30;
    uint best_hash = 0u;
    center = p_m;
    [unroll]
    for (int dz = -1; dz <= 1; dz++)
    {
        [unroll]
        for (int dx = -1; dx <= 1; dx++)
        {
            int2 cell = base + int2(dx, dz);
            uint h = grass_hash(grass_hash2(cell) ^ seed ^ salt) ^ 0x9e3779b9u;
            float2 c = float2(cell) + float2(grass_unit(h, 1u), grass_unit(h, 2u));
            float d = dot(c - p, c - p);
            if (d < best)
            {
                best = d;
                best_hash = h;
                center = c * clump_size;
            }
        }
    }
    return best_hash;
}

// How far a clump's hue leans toward the dry tint, from its hash. The draws
// read the hash's low 12 bits, which is all a blade carries of it.
float grass_clump_dryness(uint clump_hash)
{
    float r = grass_unit(clump_hash & 0xfffu, 15u);
    return r * r;
}

// How much taller or shorter than its layer's mean a clump grows, as a
// multiplier, from its hash.
float grass_clump_height(uint clump_hash, float height_variance)
{
    return 1.0 + height_variance * (grass_unit(clump_hash, 6u) - 0.5) * 1.2;
}

// Matches DRY_TINT in render::grass::variation.
static const float3 GRASS_DRY_TINT = float3(1.35, 1.1, 0.55);

// A unit value at a 3D lattice point, salted. Mirrors
// render::grass::planet::density.
float grass_lattice3(int3 c, uint salt)
{
    uint h = grass_hash(asuint(c.x) + grass_hash(asuint(c.y) + grass_hash(asuint(c.z) ^ salt)));
    return float(h >> 8u) * (1.0 / 16777216.0);
}

float grass_value_noise3(float3 p, uint salt)
{
    float3 cell = floor(p);
    float3 f = p - cell;
    f = f * f * (3.0 - 2.0 * f);
    int3 c = int3(cell);
    float x00 = lerp(grass_lattice3(c, salt), grass_lattice3(c + int3(1, 0, 0), salt), f.x);
    float x10 = lerp(grass_lattice3(c + int3(0, 1, 0), salt), grass_lattice3(c + int3(1, 1, 0), salt), f.x);
    float x01 = lerp(grass_lattice3(c + int3(0, 0, 1), salt), grass_lattice3(c + int3(1, 0, 1), salt), f.x);
    float x11 = lerp(grass_lattice3(c + int3(0, 1, 1), salt), grass_lattice3(c + int3(1, 1, 1), salt), f.x);
    return lerp(lerp(x00, x10, f.y), lerp(x01, x11, f.y), f.z);
}

// The share of a layer's density the meadow grows in unit direction `dir`
// from the planet's center. Mirrors
// render::grass::planet::density::meadow_density.
float grass_meadow_density(float3 dir, float scale, uint seed)
{
    float3 p = dir * scale;
    float3 q = p * 2.07 + 13.1;
    float n = grass_value_noise3(p, seed) * 0.65 + grass_value_noise3(q, seed ^ 0x68e31da4u) * 0.35;
    float t = saturate((n - 0.30) / (0.42 - 0.30));
    return t * t * (3.0 - 2.0 * t);
}

// Matches MACRO_NEAR_SIZE and MACRO_FAR_SIZE in render::grass::variation.
static const float GRASS_MACRO_NEAR = 150.0;
static const float GRASS_MACRO_FAR = 1200.0;

// The field's slow drift at ground point `p` (meters) for a layer salted by
// `seed`: how dry (x, [0, 1]) and how lush (y, [-1, 1]) the grass grows there,
// scaled by `amount`; `slope` is 1 - cos of the ground's angle from the blades'
// up, `near_detail` fades the smaller patches toward their mean. Mirrors
// render::grass::variation::macro_variation.
float2 grass_macro(float3 p, uint seed, float slope, float amount, float near_detail)
{
    float far = grass_value_noise3(p * (1.0 / GRASS_MACRO_FAR), seed ^ 0x2c1b3c6du);
    float near = 0.5;
    if (near_detail > 0.0)
    {
        near = lerp(near, grass_value_noise3(p * (1.0 / GRASS_MACRO_NEAR), seed ^ 0x297a2d39u),
                    near_detail);
    }
    float dry = saturate((far - 0.55) * 2.5 + (near - 0.5) * 1.2 + slope * 4.0);
    float lush = clamp((0.5 - far) * 1.4 + (near - 0.5) * 2.0, -1.0, 1.0);
    return float2(dry, lush) * amount;
}

// The multiplier a drift `m` puts on a blade's color. Mirrors
// render::grass::variation::macro_tint.
float3 grass_macro_tint(float2 m)
{
    return lerp((float3)(1.0), GRASS_DRY_TINT, m.x) * (1.0 + 0.3 * m.y);
}

// The unit direction through face coordinates (s, t) of cube face `face`.
// Mirrors render::grass::planet::cells::face_direction and
// planet::CubeFace::direction.
float3 grass_face_direction(uint face, float s, float t)
{
    float3 n;
    float3 u;
    float3 v;
    switch (face)
    {
    case 0u: n = float3(1, 0, 0); u = float3(0, 0, -1); v = float3(0, 1, 0); break;
    case 1u: n = float3(-1, 0, 0); u = float3(0, 0, 1); v = float3(0, 1, 0); break;
    case 2u: n = float3(0, 1, 0); u = float3(1, 0, 0); v = float3(0, 0, -1); break;
    case 3u: n = float3(0, -1, 0); u = float3(1, 0, 0); v = float3(0, 0, 1); break;
    case 4u: n = float3(0, 0, 1); u = float3(1, 0, 0); v = float3(0, 1, 0); break;
    default: n = float3(0, 0, -1); u = float3(-1, 0, 0); v = float3(0, 1, 0); break;
    }
    float3 c = n + u * s + v * t;
    float3 c2 = c * c;
    float3 p = c * sqrt(max(float3(1.0 - c2.y * 0.5 - c2.z * 0.5 + c2.y * c2.z / 3.0,
                                   1.0 - c2.z * 0.5 - c2.x * 0.5 + c2.z * c2.x / 3.0,
                                   1.0 - c2.x * 0.5 - c2.y * 0.5 + c2.x * c2.y / 3.0), 0.0));
    float len = length(p);
    return len > 1e-12 ? p / len : float3(0.0, 1.0, 0.0);
}

// The cube face unit direction `dir` passes through (the axis it leans along
// most) and its face coordinates there: grass_face_direction inverted.
// Mirrors render::grass::planet::cells::face_coords.
float2 grass_face_coords(float3 dir, out uint face)
{
    float3 a = abs(dir);
    float x;
    float y;
    if (a.x >= a.y && a.x >= a.z)
    {
        face = dir.x >= 0.0 ? 0u : 1u;
        x = dir.x >= 0.0 ? -dir.z : dir.z;
        y = dir.y;
    }
    else if (a.y >= a.z)
    {
        face = dir.y >= 0.0 ? 2u : 3u;
        x = dir.x;
        y = dir.y >= 0.0 ? -dir.z : dir.z;
    }
    else
    {
        face = dir.z >= 0.0 ? 4u : 5u;
        x = dir.z >= 0.0 ? dir.x : -dir.x;
        y = dir.y;
    }
    float x2 = 2.0 * x * x;
    float y2 = 2.0 * y * y;
    float root = sqrt(max((3.0 + x2 - y2) * (3.0 + x2 - y2) - 12.0 * x2, 0.0));
    float s = x * sqrt(12.0 / (3.0 + x2 - y2 + root));
    float t = y * sqrt(12.0 / (3.0 - x2 + y2 + root));
    return clamp(float2(s, t), -1.0, 1.0);
}

// The salt a planet face's clump lattice carries, so lattices on different
// faces differ.
uint grass_face_salt(uint face)
{
    return grass_hash(face + 0x51ed27u);
}
