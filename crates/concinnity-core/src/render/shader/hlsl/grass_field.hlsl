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

// Where the blades hand the field to the canopy, whether pixel `pixel` shows
// the blades this frame when a share `blades` of the pixels at its distance
// should: interleaved gradient noise (Jimenez 2014), stepped by the frame's
// `phase` so the temporal antialiasing resolves the mix. The blades' draws and
// the canopy's ground ask the same question of the same pixel, so each pixel
// shows exactly one of them.
bool grass_crossfade_shows_blades(float2 pixel, float phase, float blades)
{
    float2 p = floor(pixel) + 5.588238 * phase;
    float n = frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715))));
    return n < blades;
}

// The clump `p` (meters on the ground's plane) belongs to, for clumps
// `clump_size` across salted by `seed` and `salt`: the nearest center of a
// jittered lattice of clump-sized cells. Returns its hash and writes its
// center, and the point's spot in the clump's cell: its distance to the
// nearest center over its distance to the second nearest, 0 at the center and
// 1 on the cell's edge. Mirrors render::grass::clumping::clump_at.
uint grass_clump_at(float2 p_m, float clump_size, uint seed, uint salt, out float2 center,
                    out float spot)
{
    float2 p = p_m / clump_size;
    int2 base = int2(floor(p));
    float best = 1e30;
    float second = 1e30;
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
                second = best;
                best = d;
                best_hash = h;
                center = c * clump_size;
            }
            else if (d < second)
            {
                second = d;
            }
        }
    }
    spot = min(sqrt(best / max(second, 1e-12)), 1.0);
    return best_hash;
}

// Matches GATHER_STRENGTH in render::grass::clumping.
static const float GRASS_GATHER = 0.65;

// The share of its distance to its clump's center a blade at `spot` keeps
// under `clumping`. Mirrors render::grass::clumping::gather_scale.
float grass_gather_scale(float spot, float clumping)
{
    float center = 1.0 - GRASS_GATHER * clumping;
    return center + (1.0 - center) * spot;
}

// How densely blades grow under `clumping` once a blade placed at `spot` is
// gathered toward its clump's center, as a multiple of the layer's mean: the
// inverse of the area the gathering shrinks the ground around it to.
float grass_crowding(float spot, float clumping)
{
    float center = 1.0 - GRASS_GATHER * clumping;
    float pull = 1.0 - center;
    return 1.0 / ((center + pull * spot) * (center + 2.0 * pull * spot));
}

// How densely blades grow at `spot` under `clumping`, as a multiple of the
// layer's mean. Mirrors render::grass::clumping::gathered_density.
float grass_gathered_density(float spot, float clumping)
{
    float center = 1.0 - GRASS_GATHER * clumping;
    float pull = 1.0 - center;
    float from = 2.0 * spot / (center + sqrt(center * center + 4.0 * pull * spot));
    return grass_crowding(from, clumping);
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

// Where position `p` lies in the noise lattice salted by `salt`: turned off
// the axes, then shifted by a salted fraction of a cell. Matches lattice_point
// in render::grass::planet::density.
float3 grass_lattice_point(float3 p, uint salt)
{
    float3 shift = float3(uint3(salt, salt >> 10u, salt >> 20u) & 0x3ffu) * (1.0 / 1024.0);
    return float3(dot(float3(0.8118384, 0.5341444, -0.2358139), p),
                  dot(float3(-0.45943317, 0.8336292, 0.30656743), p),
                  dot(float3(0.36033263, -0.14054248, 0.9221758), p)) + shift;
}

// Trilinear value noise in [0, 1] at position `p` (in lattice cells), salted,
// on a lattice turned off the axes so it has no creases along them.
float grass_value_noise3(float3 p, uint salt)
{
    p = grass_lattice_point(p, salt);
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

// How much lighter and paler litter lies where it is driest.
static const float3 GRASS_LITTER_DRY = float3(1.5, 1.3, 0.95);

// Matches LITTER_SCALES in render::grass::litter.
static const float3 GRASS_LITTER_SCALES = float3(0.45, 1.8, 7.0);

// A unit value at a 3D lattice point, salted, from a single hash step: cheap
// enough for noise evaluated on every pixel of the ground. Mirrors
// render::grass::litter::patch_lattice.
float grass_patch_lattice(int3 c, uint salt)
{
    uint h = grass_hash((asuint(c.x) * 0x8da6b343u) ^ (asuint(c.y) * 0xd8163841u)
                        ^ (asuint(c.z) * 0xcb1ab31fu) ^ salt);
    return float(h >> 8u) * (1.0 / 16777216.0);
}

float grass_patch_value_noise3(float3 p, uint salt)
{
    float3 cell = floor(p);
    float3 f = p - cell;
    f = f * f * (3.0 - 2.0 * f);
    int3 c = int3(cell);
    float x00 = lerp(grass_patch_lattice(c, salt), grass_patch_lattice(c + int3(1, 0, 0), salt), f.x);
    float x10 = lerp(grass_patch_lattice(c + int3(0, 1, 0), salt),
                     grass_patch_lattice(c + int3(1, 1, 0), salt), f.x);
    float x01 = lerp(grass_patch_lattice(c + int3(0, 0, 1), salt),
                     grass_patch_lattice(c + int3(1, 0, 1), salt), f.x);
    float x11 = lerp(grass_patch_lattice(c + int3(0, 1, 1), salt),
                     grass_patch_lattice(c + int3(1, 1, 1), salt), f.x);
    return lerp(lerp(x00, x10, f.y), lerp(x01, x11, f.y), f.z);
}

// A patch noise at fixed ground point `p` for patches `scale` meters across,
// faded toward its mean where a pixel covering `footprint` meters cannot
// hold them.
float grass_patch_noise(float3 p, float scale, uint salt, float footprint)
{
    float detail = 1.0 - smoothstep(scale * 0.5, scale * 2.0, footprint);
    if (detail <= 0.0)
    {
        return 0.5;
    }
    return 0.5 + (grass_patch_value_noise3(p / scale, salt) - 0.5) * detail;
}

// What lies on the ground under grass at fixed point `p` (meters) under a
// pixel covering `footprint` meters: how much of it is litter rather than
// bare earth (x, in [0.35, 1]), how dark the earth is (y, a multiplier in
// [0.65, 1]), and how dry and light the litter is (z, in [0, 1]). Mirrors
// render::grass::litter::ground_variation.
float3 grass_ground_variation(float3 p, float footprint)
{
    float fine = grass_patch_noise(p, GRASS_LITTER_SCALES.x, 0x6a09e667u, footprint);
    float mid = grass_patch_noise(p, GRASS_LITTER_SCALES.y, 0xbb67ae85u, footprint);
    float broad = grass_patch_noise(p, GRASS_LITTER_SCALES.z, 0x3c6ef372u, footprint);
    float patch_ = fine * 0.25 + mid * 0.35 + broad * 0.4;
    // Weighted so the dry litter falls about as often on earth as on litter.
    float dry = fine * 0.6 + (1.0 - mid) * 0.4;
    return float3(0.35 + 0.65 * smoothstep(0.36, 0.52, patch_),
                  0.65 + 0.35 * smoothstep(0.3, 0.46, patch_),
                  smoothstep(0.55, 0.7, dry));
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

// Matches WIDTH_TAPER in render::grass::canopy: a blade's width tapers as
// 1 - t^GRASS_WIDTH_TAPER from root to tip.
static const float GRASS_WIDTH_TAPER = 1.4;

// Matches the outer CANOPY_HEIGHT_CLASSES in render::grass::canopy.
static const float GRASS_HEIGHT_CLASS = 0.4665;

// The share of one blade's area above height `t` along it.
float grass_blade_area_above(float t)
{
    t = saturate(t);
    float k = 1.0 + GRASS_WIDTH_TAPER;
    return ((1.0 - t) - (1.0 - pow(t, k)) / k) / (1.0 - 1.0 / k);
}

// The share of a field's leaf area standing above height `z`, in units of its
// mean blade height, for blades varying in height by `variance`: what a direct
// light crosses on its way down to there. Mirrors
// render::grass::canopy::leaf_above.
float grass_leaf_above(float z, float variance)
{
    float3 h = max(1.0 + variance * float3(-GRASS_HEIGHT_CLASS, 0.0, GRASS_HEIGHT_CLASS), 0.1);
    float3 a = h * float3(grass_blade_area_above(z / h.x), grass_blade_area_above(z / h.y),
                          grass_blade_area_above(z / h.z));
    return (a.x + a.y + a.z) / (h.x + h.y + h.z);
}
