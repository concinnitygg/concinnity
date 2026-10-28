// The night sky behind the air: stars, the Milky Way, distant galaxies and the
// moon, spliced after `atmosphere_common.hlsl` and read from the night lanes
// of the atmosphere's block through `ATMO`. Directions on the celestial sphere
// are world directions taken through the sky's rotation rows, so everything
// here turns with the sky. `render::night_sky` mirrors the hash, the star
// grid, the galaxy profiles, the Milky Way's envelope and the moon's face,
// and its tests lock the constants below to the Rust ones.
//
// A star is a Gaussian integrated over the pixel's square, found through the
// screen derivatives of the view ray, so a sub-pixel jitter moves its light
// between pixels without changing its total. Views whose pixels are large
// against the star grid (probe faces, the sky cube) leave the stars out.

static const uint NIGHT_STAR_GRID = 128u;
static const float NIGHT_STAR_SIGMA_PX = 0.6;
static const float NIGHT_FACE_MARGIN = 0.25;
static const float NIGHT_STAR_SATURATION = 0.7;
static const float NIGHT_COUNT_SLOPE = 0.5;
static const float NIGHT_BRIGHTEST_MAGNITUDE = -1.5;

bool night_sky_on()
{
    return ATMO.moon_up.w > 0.5;
}

// `night_hash` / `night_unit` in render::night_sky.
uint night_hash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float night_unit(uint h)
{
    return float(h >> 8u) * (1.0 / 16777216.0);
}

float night_smoothstep(float a, float b, float x)
{
    float t = saturate((x - a) / (b - a));
    return t * t * (3.0 - 2.0 * t);
}

// A world direction on the celestial sphere.
float3 night_celestial(float3 d)
{
    return float3(dot(ATMO.sky_rot[0].xyz, d), dot(ATMO.sky_rot[1].xyz, d),
                  dot(ATMO.sky_rot[2].xyz, d));
}

// Value noise on the integer lattice, smoothly interpolated, in [0, 1].
float night_lattice(int3 i)
{
    return night_unit(night_hash(asuint(i.x) + night_hash(asuint(i.y) + night_hash(asuint(i.z)))));
}

float night_noise(float3 p)
{
    float3 cell = floor(p);
    int3 i = int3(cell);
    float3 f = p - cell;
    f = f * f * (3.0 - 2.0 * f);
    float x00 = lerp(night_lattice(i), night_lattice(i + int3(1, 0, 0)), f.x);
    float x10 = lerp(night_lattice(i + int3(0, 1, 0)), night_lattice(i + int3(1, 1, 0)), f.x);
    float x01 = lerp(night_lattice(i + int3(0, 0, 1)), night_lattice(i + int3(1, 0, 1)), f.x);
    float x11 = lerp(night_lattice(i + int3(0, 1, 1)), night_lattice(i + int3(1, 1, 1)), f.x);
    return lerp(lerp(x00, x10, f.y), lerp(x01, x11, f.y), f.z);
}

float night_fbm(float3 p, uint octaves)
{
    float sum = 0.0;
    float amplitude = 0.5;
    float total = 0.0;
    for (uint k = 0u; k < octaves; ++k)
    {
        sum += amplitude * night_noise(p);
        total += amplitude;
        p = p * 2.03 + 11.7;
        amplitude *= 0.5;
    }
    return sum / max(total, 1e-6);
}

// Stars, mirrored in `render::night_sky::stars`.

// The cube face `c` points through and its coordinates on it.
void night_cube_face(float3 c, out uint face, out float u, out float v)
{
    float3 a = abs(c);
    uint axis = (a.x >= a.y && a.x >= a.z) ? 0u : (a.y >= a.z ? 1u : 2u);
    float major = max(a[axis], 1e-20);
    face = axis * 2u + (c[axis] < 0.0 ? 1u : 0u);
    u = c[(axis + 1u) % 3u] / major;
    v = c[(axis + 2u) % 3u] / major;
}

float3 night_face_direction(uint face, float u, float v)
{
    uint axis = face / 2u;
    float s = (face % 2u) == 0u ? 1.0 : -1.0;
    if (axis == 0u)
    {
        return float3(s, u, v);
    }
    if (axis == 1u)
    {
        return float3(v, s, u);
    }
    return float3(u, v, s);
}

float night_cell_solid_angle(uint i, uint j)
{
    float w = 2.0 / float(NIGHT_STAR_GRID);
    float u = -1.0 + (float(i) + 0.5) * w;
    float v = -1.0 + (float(j) + 0.5) * w;
    float k = 1.0 + u * u + v * v;
    return w * w / (k * sqrt(k));
}

float3 night_blackbody(float temperature)
{
    const float c2 = 14388.0;
    const float3 wavelengths = float3(0.610, 0.550, 0.465);
    float3 l5 = wavelengths * wavelengths * wavelengths * wavelengths * wavelengths;
    float3 planck = 1.0 / (l5 * (exp(c2 / (wavelengths * temperature)) - 1.0));
    float lum = max(dot(planck, float3(0.2126, 0.7152, 0.0722)), 1e-30);
    return 1.0 + NIGHT_STAR_SATURATION * (planck / lum - 1.0);
}

float night_erf(float x)
{
    x = clamp(x * (1.1283792 + 0.10091 * x * x), -15.0, 15.0);
    return 1.0 - 2.0 / (exp(2.0 * x) + 1.0);
}

float night_footprint(float offset_px)
{
    float k = 1.0 / (NIGHT_STAR_SIGMA_PX * 1.41421356);
    return 0.5 * (night_erf((offset_px + 0.5) * k) - night_erf((offset_px - 0.5) * k));
}

// The star in cell (i, j) of `face`, if it holds one.
bool night_star(uint face, uint i, uint j, out float3 dir, out float magnitude,
                out float temperature)
{
    uint n = NIGHT_STAR_GRID;
    uint seed = uint(ATMO.galactic_pole.w);
    uint h0 = night_hash((face * n + j) * n + i + night_hash(seed));
    dir = (float3)0.0;
    magnitude = 0.0;
    temperature = 0.0;
    if (night_unit(h0) >= ATMO.galactic_center.w * night_cell_solid_angle(i, j))
    {
        return false;
    }
    uint h1 = night_hash(h0);
    uint h2 = night_hash(h1);
    uint h3 = night_hash(h2);
    uint h4 = night_hash(h3);
    bool border = i == 0u || j == 0u || i == n - 1u || j == n - 1u;
    float margin = border ? NIGHT_FACE_MARGIN : 0.0;
    float w = 2.0 / float(n);
    float u = -1.0 + (float(i) + margin + (1.0 - 2.0 * margin) * night_unit(h1)) * w;
    float v = -1.0 + (float(j) + margin + (1.0 - 2.0 * margin) * night_unit(h2)) * w;
    dir = normalize(night_face_direction(face, u, v));
    float limit = ATMO.night.y;
    float decades = log(max(night_unit(h3), 1e-7)) / 2.30258509;
    magnitude = max(limit + decades / NIGHT_COUNT_SLOPE, NIGHT_BRIGHTEST_MAGNITUDE);
    temperature = 3000.0 * pow(10.0, pow(night_unit(h4), 1.6));
    return true;
}

// The stars' light at celestial direction `c`, whose screen derivatives are
// `cx` and `cy`: every star in the 2 x 2 cells nearest it, spread over the
// pixel by its footprint.
float3 night_stars(float3 c, float3 cx, float3 cy)
{
    float pixel = sqrt(length(cx) * length(cy));
    float fade = 1.0 - night_smoothstep(0.15, 0.3, pixel * float(NIGHT_STAR_GRID) * 0.5);
    if (fade <= 0.0 || ATMO.night.x <= 0.0)
    {
        return (float3)0.0;
    }
    float g11 = dot(cx, cx);
    float g12 = dot(cx, cy);
    float g22 = dot(cy, cy);
    float det = g11 * g22 - g12 * g12;
    float omega = length(cross(cx, cy));
    if (det <= 0.0 || omega <= 0.0)
    {
        return (float3)0.0;
    }
    uint face;
    float u;
    float v;
    night_cube_face(c, face, u, v);
    float n = float(NIGHT_STAR_GRID);
    int i0 = int(floor((u + 1.0) * 0.5 * n - 0.5));
    int j0 = int(floor((v + 1.0) * 0.5 * n - 0.5));
    float3 sum = (float3)0.0;
    for (int dj = 0; dj < 2; ++dj)
    {
        for (int di = 0; di < 2; ++di)
        {
            int i = i0 + di;
            int j = j0 + dj;
            if (i < 0 || j < 0 || i >= int(NIGHT_STAR_GRID) || j >= int(NIGHT_STAR_GRID))
            {
                continue;
            }
            float3 dir;
            float magnitude;
            float temperature;
            if (!night_star(face, uint(i), uint(j), dir, magnitude, temperature))
            {
                continue;
            }
            float3 delta = dir - c;
            delta -= c * dot(c, delta);
            float b1 = dot(cx, delta);
            float b2 = dot(cy, delta);
            float a = (g22 * b1 - g12 * b2) / det;
            float b = (g11 * b2 - g12 * b1) / det;
            if (abs(a) > 3.0 || abs(b) > 3.0)
            {
                continue;
            }
            float light = ATMO.night.x * pow(10.0, -0.4 * magnitude);
            sum += night_blackbody(temperature) * (light * night_footprint(a) * night_footprint(b));
        }
    }
    return sum * (fade / omega);
}

// The Milky Way, mirrored in `render::night_sky::milky_way`.

// `milky_way_envelope` in render::night_sky.
float milky_way_envelope(float latitude, float longitude)
{
    float l2 = longitude * longitude;
    float b2 = latitude * latitude;
    float core = exp(-l2 / (0.5 * 0.5));
    float width = 0.1 + 0.12 * core;
    float band = exp(-0.5 * b2 / (width * width));
    float along = 0.3 + 0.7 * exp(-l2 / (1.6 * 1.6));
    float bulge = exp(-l2 / (0.35 * 0.35) - b2 / (0.16 * 0.16));
    return (band * along + 0.6 * bulge) / 1.6 + 0.03;
}

// The Milky Way's radiance at celestial direction `c`. `detailed` breaks the
// band into clumps and cuts it with dark dust lanes; without it the band is
// its smooth envelope at the same mean, for the sky's ambient light.
float3 milky_way(float3 c, bool detailed)
{
    float3 g = ATMO.galactic_pole.xyz;
    float3 x = ATMO.galactic_center.xyz;
    float3 y = cross(g, x);
    float sin_lat = clamp(dot(c, g), -1.0, 1.0);
    float latitude = asin(sin_lat);
    float longitude = atan2(dot(c, y), dot(c, x));
    float envelope = milky_way_envelope(latitude, longitude);
    float l2 = longitude * longitude;
    float warm = saturate(1.5 * exp(-l2 / 0.25 - latitude * latitude / 0.04)
                          + 0.3 * exp(-l2 / 2.56));
    float3 tint = lerp(float3(0.86, 0.91, 1.0), float3(1.0, 0.9, 0.76), warm);
    float band = envelope - 0.03;
    if (!detailed)
    {
        return ATMO.night.z * (band * 0.6 * tint + 0.03);
    }
    float detail = 0.6;
    if (band > 0.003)
    {
        float3 q = float3(dot(c, x), dot(c, y), sin_lat);
        float clumps = night_fbm(q * float3(16.0, 16.0, 36.0), 5u);
        float structure = 0.35 + 1.3 * clumps * clumps;
        float dust_noise = night_fbm(q * float3(9.0, 9.0, 40.0) + 17.0, 5u);
        float wander = latitude - 0.01 - 0.04 * (night_noise(q * 3.0 + 5.0) - 0.5);
        float lane = exp(-0.5 * wander * wander / (0.05 * 0.05))
                   * (0.4 + 0.6 * exp(-l2 / 1.0));
        float dust = exp(-2.5 * night_smoothstep(0.45, 0.7, dust_noise) * lane);
        detail = structure * dust;
    }
    return ATMO.night.z * (band * detail * tint + 0.03);
}

// Galaxies, placed by `render::night_sky::galaxies`.

float3 night_galaxies(float3 c)
{
    float3 sum = (float3)0.0;
    uint count = uint(ATMO.night.w);
    for (uint k = 0u; k < count; ++k)
    {
        float4 center = ATMO.galaxies[2u * k];
        float4 axis = ATMO.galaxies[2u * k + 1u];
        if (dot(c, center.xyz) <= 0.0)
        {
            continue;
        }
        float ratio = abs(axis.w);
        float x = dot(c, axis.xyz);
        float y = dot(c, cross(center.xyz, axis.xyz)) / ratio;
        float r = sqrt(x * x + y * y);
        if (r >= 1.0)
        {
            continue;
        }
        float edge = 1.0 - night_smoothstep(0.8, 1.0, r);
        float3 light;
        if (axis.w > 0.0)
        {
            float arms = 0.5 + 0.5 * cos(2.0 * atan2(y, x) - 6.0 * log(r + 0.05));
            float disc = exp(-r / 0.22) * (0.3 + 0.7 * arms * arms);
            float bulge = 3.0 * exp(-r / 0.045);
            light = disc * float3(0.75, 0.85, 1.0) + bulge * float3(1.0, 0.85, 0.65);
        }
        else
        {
            light = exp(-r / 0.18) * float3(1.0, 0.88, 0.7);
        }
        sum += light * (center.w * edge);
    }
    return sum;
}

// The moon, mirrored in `render::night_sky::moon`.

// The moon's albedo at point `p` of its unit sphere, about 1 on average:
// bright highlands, dark maria, and craters at two scales.
float moon_albedo(float3 p)
{
    float maria = night_fbm(p * 2.2 + 3.1, 3u);
    float albedo = lerp(1.18, 0.62, night_smoothstep(0.48, 0.58, maria));
    float scale = 7.0;
    for (uint k = 0u; k < 2u; ++k)
    {
        float3 q = p * scale;
        float3 cell = floor(q);
        int3 i = int3(cell);
        uint h = night_hash(asuint(i.x) + night_hash(asuint(i.y) + night_hash(asuint(i.z) + k)));
        float3 center = cell + 0.25 + 0.5 * float3(night_unit(h), night_unit(night_hash(h)),
                                                 night_unit(night_hash(h + 1u)));
        float radius = 0.15 + 0.25 * night_unit(night_hash(h + 2u));
        float d = length(q - center) / radius;
        float rim = d - 1.0;
        albedo *= d < 1.0 ? lerp(0.8, 1.0, d * d) : 1.0;
        albedo *= 1.0 + 0.25 * exp(-rim * rim / (0.15 * 0.15));
        scale *= 2.2;
    }
    return albedo;
}

// The share of a `pixel`-wide view along `d` the moon covers, and its
// radiance there: its face lit by the sun, the dark side faintly.
float night_moon(float3 d, float pixel, out float3 radiance)
{
    radiance = (float3)0.0;
    float cos_radius = ATMO.moon_direction.w;
    if (cos_radius > 1.0)
    {
        return 0.0;
    }
    float3 m = ATMO.moon_direction.xyz;
    float radius = acos(cos_radius);
    float theta = acos(clamp(dot(d, m), -1.0, 1.0));
    if (theta > radius + 2.0 * pixel)
    {
        return 0.0;
    }
    float coverage = saturate((radius - theta) / max(pixel, 1e-6) + 0.5);
    float3 up = ATMO.moon_up.xyz;
    float3 right = cross(m, up);
    float s = sin(radius);
    float2 xy = float2(dot(d, right), dot(d, up)) / s;
    float r2 = dot(xy, xy);
    if (r2 > 1.0)
    {
        xy /= sqrt(r2);
        r2 = 1.0;
    }
    float z = sqrt(max(1.0 - r2, 0.0));
    float3 n = right * xy.x + up * xy.y - m * z;
    float lit = max(dot(n, ATMO.sun_direction.xyz), 0.0);
    radiance = ATMO.moon_face.xyz * moon_albedo(float3(xy, z)) * (lit + ATMO.moon_face.w);
    return coverage;
}

// The night sky along view direction `d` whose screen derivatives are `dx`
// and `dy`, behind air that lets `transmittance` of it through: the stars,
// the Milky Way and the galaxies dimmed by the air, the moon in front of them
// (its light is already dimmed by the air toward it).
float3 night_sky_radiance(float3 d, float3 dx, float3 dy, float transmittance)
{
    float pixel = sqrt(length(dx) * length(dy));
    float3 moon;
    float cover = night_moon(d, pixel, moon);
    float3 c = night_celestial(d);
    float3 behind = milky_way(c, true) + night_galaxies(c)
                  + night_stars(c, night_celestial(dx), night_celestial(dy));
    return transmittance * behind * (1.0 - cover) + moon * cover;
}

// The night sky as a reflection sees it: everything but the stars, which
// would alias at a cube's resolution, and the moon, whose light's specular
// term draws it.
float3 night_sky_reflected(float3 d, float transmittance)
{
    float3 c = night_celestial(d);
    return transmittance * (milky_way(c, true) + night_galaxies(c));
}

// The night sky's share of the ambient light: the Milky Way's smooth band.
// The moon lights the scene as a directional light instead.
float3 night_sky_ambient(float3 d, float transmittance)
{
    return transmittance * milky_way(night_celestial(d), false);
}
