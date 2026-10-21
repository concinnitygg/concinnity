// The blade shading model, shared by the grass draws and the far field past
// them, and the far field's canopy: the blade model averaged over a field of
// blades seen from afar (render::grass::canopy).
//
// A view ray enters the field from above and is stopped by blade area at a
// rate set by the leaf area and how much of a blade faces along the ray, so
// along the ground it meets the upper blades and looking down it often reaches
// the shaded ground between them (canopy_add_view, read from each layer's
// table). Where it stops, the blades it
// sees lean every way around the view: CANOPY_FACINGS of them, each turned
// toward the viewer and lit by the blade model at that height. The clumps,
// density and slow color drift are the blades' own (grass_field.hlsl), on the
// same ground-fixed coordinates, so the far field lines up with the blades it
// takes over from.

{GRASS_FIELD}

// The parts of grass_light_terms set by the view and the light alone, which
// every normal lit from them shares.
struct GrassLightPair
{
    float3 H;
    float fresnel;
    float through;
};

GrassLightPair grass_light_pair(float3 V, float3 L)
{
    GrassLightPair p;
    p.H = normalize(V + L);
    p.fresnel = fresnel_schlick(saturate(dot(p.H, V)), (float3)(0.04)).x;
    p.through = pow(saturate(dot(V, -L)), 4.0);
    return p;
}

// The two parts of the light a blade of unit albedo returns from `L`: what
// multiplies the albedo (xyz: wrapped diffuse on the lit face, and light
// carried through the blade when the source is behind it), and the soft
// sheen (w), which does not. Taken from the normal's cosines with the light,
// the view and the half vector; the GGX lobe and its shadowing share one
// division.
float4 grass_light_terms_dots(float ndl, float ndv, float ndh, GrassLightPair p)
{
    float diffuse = saturate((ndl + 0.35) / 1.35) / PI;
    float behind = saturate(-ndl);
    float carried = (0.35 * behind + 0.9 * p.through) / PI;
    float nl = saturate(ndl);
    float nv = max(ndv, 0.0);
    const float roughness = 0.5;
    const float a2 = roughness * roughness * roughness * roughness;
    const float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float nh = max(ndh, 0.0);
    float d = nh * nh * (a2 - 1.0) + 1.0;
    float spec = a2 * nv * nl
               / ((PI * d * d + 0.0001) * (nv * (1.0 - k) + k) * (nl * (1.0 - k) + k)
                  * max(4.0 * max(ndv, 1e-3) * nl, 1e-3));
    float sheen = p.fresnel * spec * 0.5 * nl;
    return float4(diffuse + carried * float3(1.1, 1.25, 0.6), sheen);
}

float4 grass_light_terms(float3 N, float3 V, float3 L)
{
    GrassLightPair p = grass_light_pair(V, L);
    return grass_light_terms_dots(dot(N, L), dot(N, V), dot(N, p.H), p);
}

// Light arriving at a blade from `L` at `radiance`.
float3 grass_light(float3 albedo, float3 N, float3 V, float3 L, float3 radiance)
{
    float4 k = grass_light_terms(N, V, L);
    return (albedo * k.xyz + k.w) * radiance;
}

// How much of the sky reaches a point `t` up the blade inside the field.
float grass_root_occlusion(float t)
{
    return lerp(0.25, 1.0, smoothstep(0.0, 0.6, t));
}

// How much of a direct light from `L` reaches a point inside the field, given
// `depth`: the blade's up times the leaf area above the point times the top-down
// axis of its projection ellipse (xyz), and the leaf above times the side-on
// axis (w). Mirrors render::grass::canopy::ellipse_projection.
float grass_light_through(float4 depth, float3 L)
{
    float top = length(depth.xyz);
    if (top <= 1e-6)
    {
        return 1.0;
    }
    float mu = dot(L, depth.xyz) / top;
    float side = depth.w * depth.w * (1.0 - saturate(mu * mu));
    return exp(-sqrt(top * top * mu * mu + side) / max(mu, 0.02));
}

// ---- The far field's canopy ----

// Matches CANOPY_FACINGS in render::grass::canopy.
#define CANOPY_FACINGS 8

// The cosine and sine of how far a blade leaning by `bend` tilts from upright
// at height `t`. Mirrors render::grass::canopy::blade_tilt.
float2 canopy_tilt(float bend, float t)
{
    float2 v = float2((1.0 - t) * sqrt(max(1.0 - bend * bend, 0.0)), t * bend);
    float len = length(v);
    return len > 1e-6 ? v / len : float2(1.0, 0.0);
}

// What one band of blade height shows a view, given a ray stopped in it.
// Mirrors render::grass::canopy::CanopyBand.
struct CanopyBand
{
    float share;
    float height;
    float ramp;
    float occlusion;
    float ramp_occlusion;
    float above;
};

// Where a view stops in the canopy: the ground's share, and the two bands
// with their terms summed by share.
struct CanopyView
{
    float ground;
    CanopyBand upper;
    CanopyBand lower;
};

// What the canopy is at one ground point, and how it is seen and lit there.
// The bands' shares and blade colors are folded into the colors each kind of
// light is multiplied by, so little stays live through the light loops.
struct CanopyShade
{
    // The ground's albedo under the grass: the surface's own, strewn with the
    // layers' litter.
    float3 ground_albedo;
    // Blades' up, and the view's direction along the ground.
    float3 up;
    float3 along;
    float leaf_area;
    // 1 where this pixel shows the canopy rather than the blades (see
    // canopy_handed), 0 where it shows the ground between them.
    float handed;
    // The layer whose table a light's way through is read from: the one
    // growing the most leaf.
    uint table;
    // The ground's share of the view, and how much of the sky reaches it.
    float ground;
    float ground_sky;
    // The albedo the bands show, for the unlit view.
    float3 seen;
    // Each band's blade tilt, and what its share of the view multiplies a
    // direct light's albedo term (rgb) and sheen (w) by. The lower band takes
    // direct light only where a view can reach it.
    float2 upper_tilt;
    float2 lower_tilt;
    bool lower_lit;
    float4 upper_direct;
    float4 lower_direct;
    // The leaf area standing above each band, which a direct light crosses on
    // its way down to it.
    float2 above;
    // What both bands multiply the sky and local lights' albedo term (rgb)
    // and sheen (w) by.
    float4 open;
    // The light gathered so far.
    float3 light;
};

// Where the blades hand the field to the canopy, whether pixel `pixel`
// `dist` meters from the camera shows the canopy rather than the blades: the
// same choice the blades' draws make there (grass_crossfade_keeps), so each
// pixel shows exactly one of them. 0 before the hand-off, 1 past the blades.
float canopy_handed(float2 pixel, float dist)
{
    float blades = saturate((CANOPY.band.y - dist) / max(CANOPY.band.y - CANOPY.band.x, 1e-3));
    return grass_crossfade_shows_blades(pixel, CANOPY.band.z, blades) ? 0.0 : 1.0;
}

// Matches render::grass::canopy_table: each layer's view, tabulated over the
// view's angle and the optical depth it crosses.
#define CANOPY_TABLE_ANGLES 16u
#define CANOPY_TABLE_DEPTHS 16u
#define CANOPY_TABLE_ROWS 3u
static const float CANOPY_MIN_MU = 0.02;
static const float CANOPY_DEPTH_KNEE = 0.25;
static const float CANOPY_MAX_DEPTH = 400.0;

// A fractional cell of a layer's table, its four corners and their weights.
struct CanopyTableAt
{
    uint base;
    uint2 i;
    uint2 j;
    float2 f;
};

// The cell at fractional angle row `x` and depth row `y` of `layer`'s table.
CanopyTableAt canopy_table_at(uint layer, float x, float y)
{
    CanopyTableAt at;
    at.base = layer * CANOPY_TABLE_ANGLES * CANOPY_TABLE_DEPTHS;
    float2 lo = floor(float2(x, y));
    at.f = float2(x, y) - lo;
    at.i = uint2(lo.x, min(lo.x + 1.0, float(CANOPY_TABLE_ANGLES - 1u)));
    at.j = uint2(lo.y, min(lo.y + 1.0, float(CANOPY_TABLE_DEPTHS - 1u)));
    return at;
}

float4 canopy_table_row(CanopyTableAt at, uint r)
{
    uint4 cells = at.base + uint4(at.i.x, at.i.y, at.i.x, at.i.y) * CANOPY_TABLE_DEPTHS
                + uint4(at.j.x, at.j.x, at.j.y, at.j.y);
    cells = cells * CANOPY_TABLE_ROWS + r;
    return lerp(lerp(CANOPY_TABLES[cells.x], CANOPY_TABLES[cells.y], at.f.x),
                lerp(CANOPY_TABLES[cells.z], CANOPY_TABLES[cells.w], at.f.x), at.f.y);
}

// The fractional angle row of view cosine `mu`.
float canopy_angle_coord(float mu)
{
    return saturate(acos(clamp(mu, CANOPY_MIN_MU, 1.0)) / acos(CANOPY_MIN_MU))
         * float(CANOPY_TABLE_ANGLES - 1u);
}

// Blade area of `layer` seen per unit of leaf area at fractional angle row
// `x`. It depends on the angle alone, so only the first depth row is read.
float canopy_table_projection(uint layer, float x)
{
    CanopyTableAt at = canopy_table_at(layer, x, 0.0);
    uint2 cells = (at.base + at.i * CANOPY_TABLE_DEPTHS) * CANOPY_TABLE_ROWS;
    return lerp(CANOPY_TABLES[cells.x].y, CANOPY_TABLES[cells.y].y, at.f.x);
}

// Add where rays seen along `mu` (at fractional angle row `x`) stop in
// `layer`'s canopy of `leaf_area`, weighted by `w`, into `view`. Mirrors
// render::grass::canopy_table::table_view.
void canopy_add_view(inout CanopyView view, uint layer, float x, float mu, float leaf_area, float w)
{
    float depth = canopy_table_projection(layer, x) * leaf_area / mu;
    float stopped = 1.0 - exp(-depth);
    float y = saturate(log(1.0 + max(depth, 0.0) / CANOPY_DEPTH_KNEE)
                       / log(1.0 + CANOPY_MAX_DEPTH / CANOPY_DEPTH_KNEE))
            * float(CANOPY_TABLE_DEPTHS - 1u);
    CanopyTableAt at = canopy_table_at(layer, x, y);
    float4 r0 = canopy_table_row(at, 0u);
    float4 r1 = canopy_table_row(at, 1u);
    float4 r2 = canopy_table_row(at, 2u);
    float upper = w * saturate(r0.x) * stopped;
    float lower = w * stopped - upper;
    view.ground += w * (1.0 - stopped);
    view.upper.share += upper;
    view.upper.height += upper * r0.z;
    view.upper.ramp += upper * r0.w;
    view.upper.occlusion += upper * r1.x;
    view.upper.ramp_occlusion += upper * r1.y;
    view.upper.above += upper * r1.z;
    view.lower.share += lower;
    view.lower.height += lower * r2.x;
    view.lower.ramp += lower * r2.y;
    view.lower.occlusion += lower * r2.z;
    view.lower.ramp_occlusion += lower * r2.w;
    view.lower.above += lower * r1.w;
}

// A band's terms summed by share, as means.
CanopyBand canopy_band_mean(CanopyBand band)
{
    float n = 1.0 / max(band.share, 1e-9);
    band.height *= n;
    band.ramp *= n;
    band.occlusion *= n;
    band.ramp_occlusion *= n;
    band.above *= n;
    return band;
}

// How much of a light along `L` gets through the canopy to the ground.
float canopy_transmit(CanopyShade c, float3 L)
{
    float mu = max(dot(L, c.up), CANOPY_MIN_MU);
    return exp(-canopy_table_projection(c.table, canopy_angle_coord(mu)) * c.leaf_area / mu);
}

// What a band of blade colors `root` to `root + span` shows a light whose
// albedo term `ramp` and `plain` weight the tip and the root by, with its
// sheen weighted by `plain`, scaled by the band's share of the view.
float4 canopy_band_weights(float share, float3 root, float3 span, float plain, float ramp)
{
    return share * float4(root * plain + span * ramp, plain);
}

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

// One octave of the ground's trilinear value noise at lattice position `p`:
// xyz = its gradient along the lattice axes, w = its value.
float4 ground_noise(float3 p, uint salt)
{
    float3 cell = floor(p);
    float3 t = p - cell;
    float3 s = t * t * (3.0 - 2.0 * t);
    float3 ds = 6.0 * t * (1.0 - t);
    int3 c = int3(cell);
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

// A sum of `octaves` noise octaves at unit direction `dir`, the broadest
// `frequency` cells per unit of direction and `cell` meters wide, octave `o`
// salted by `salts[o / 4][o % 4]`, each octave faded to its mean as its cells
// shrink below two pixels of `footprint` meters: xyz = the sum's gradient
// over directions, w = the sum.
float4 ground_octaves(float3 dir, float frequency, float cell, uint octaves, uint4 salts[4],
                      float footprint)
{
    float3x3 turn = float3x3(CANOPY.ground_turn[0].xyz, CANOPY.ground_turn[1].xyz,
                             CANOPY.ground_turn[2].xyz);
    float3x3 to_lattice = turn * frequency;
    float weight = 1.0;
    float4 sum = (float4)(0.0);
    for (uint o = 0u; o < octaves; o++)
    {
        float fade = saturate(log2(cell / max(2.0 * footprint, 1e-4)));
        if (fade <= 0.0)
        {
            // This octave and every finer one sit at their mean, and the
            // weights left sum to twice this one's, less the last's.
            sum.w += weight * (2.0 - exp2(1.0 - float(octaves - o))) * 0.5;
            break;
        }
        uint salt = salts[o >> 2u][o & 3u];
        float3 offset = float3(uint3(salt, salt >> 10u, salt >> 20u) & 0x3ffu) / 1024.0;
        float4 n = ground_noise(mul(to_lattice, dir) + offset, salt);
        sum.xyz += mul(n.xyz, to_lattice) * (weight * fade);
        sum.w += (n.w * fade + 0.5 * (1.0 - fade)) * weight;
        to_lattice = mul(turn, to_lattice) * 2.0;
        weight *= 0.5;
        cell *= 0.5;
    }
    return sum;
}

// PlanetShape's coast (planet/coast.rs `shore`): how the ground's height
// changes per meter `inland` of the coast (x) and per meter of the hills
// standing `hill` high there (y).
float2 ground_shore(float inland, float hill)
{
    float rise = max(CANOPY.hill_rise, 1e-3);
    if (inland >= 0.0)
    {
        float w = CANOPY.beach_width;
        float t = min(inland / w, 1.0);
        float x = saturate((inland - w) / rise);
        float fade = x * x * (3.0 - 2.0 * x);
        float fade_d = 6.0 * x * (1.0 - x);
        return float2(CANOPY.beach_rise * 2.0 * (1.0 - t) / w + hill * fade_d / rise, fade);
    }
    if (CANOPY.ocean_depth <= 0.0)
    {
        return (float2)(0.0);
    }
    float out_m = -inland;
    float e = exp(-out_m / CANOPY.shelf_length);
    float fall = 1.0 - e;
    float fall_d = e / CANOPY.shelf_length;
    float x = saturate(out_m / rise);
    float fade = x * x * (3.0 - 2.0 * x);
    float fade_d = 6.0 * x * (1.0 - x);
    float lift = CANOPY.seabed_lift;
    float d_out = -CANOPY.ocean_depth * fall_d
                + lift * hill * (fall_d * fade + fall * fade_d / rise);
    return float2(-d_out, lift * fall * fade);
}

// The planet ground's normal at unit direction `dir`, both in the planet's
// axes: PlanetShape::normal, with each octave faded out as its cells shrink
// below two pixels of `footprint` meters, so the detail a pixel cannot hold
// averages away rather than aliasing, whatever tile the pixel lies on.
float3 ground_normal_at(float3 dir, float footprint)
{
    float4 hills = ground_octaves(dir, CANOPY.ground_frequency, CANOPY.ground_cell,
                                  min(CANOPY.ground_octaves, 16u), CANOPY.ground_salts, footprint)
                 * CANOPY.ground_height_scale;
    float3 gradient = hills.xyz;
    if (CANOPY.continent_octaves > 0u)
    {
        uint4 salts[4] = { CANOPY.continent_salts[0], CANOPY.continent_salts[1], (uint4)(0u),
                           (uint4)(0u) };
        float4 field = ground_octaves(dir, CANOPY.continent_frequency,
                                      CANOPY.planet_radius / CANOPY.continent_frequency,
                                      min(CANOPY.continent_octaves, 8u), salts, footprint)
                     * CANOPY.continent_scale;
        float2 shore = ground_shore(field.w - CANOPY.continent_offset, hills.w);
        gradient = field.xyz * shore.x + hills.xyz * shore.y;
    }
    float3 along = gradient - dir * dot(dir, gradient);
    return normalize(dir - along / CANOPY.planet_radius);
}

// The normal surface `od` shades with at `pos`, seen from a pixel covering
// `footprint` meters: on a planet, its ground's own, independent of the tile
// it lies on; elsewhere the interpolated `normal`.
float3 canopy_ground_normal(GpuObjectData od, float3 pos, float3 normal, float footprint)
{
    bool planet = ((od.far_canopy >> 16u) & 1u) != 0u;
    if (!planet || CANOPY.planet_radius <= 0.0 || CANOPY.ground_octaves == 0u)
    {
        return normal;
    }
    float4 p = float4(pos, 1.0);
    float3 rel = float3(dot(CANOPY.planet_rows[0], p), dot(CANOPY.planet_rows[1], p),
                        dot(CANOPY.planet_rows[2], p));
    float3 n = ground_normal_at(normalize(rel), footprint);
    return normalize(CANOPY.planet_rows[0].xyz * n.x + CANOPY.planet_rows[1].xyz * n.y
                     + CANOPY.planet_rows[2].xyz * n.z);
}

// The ground and the canopy at `pos` on surface `od` of albedo `albedo`, the
// vertex color `color` carrying a terrain's layer densities and `normal` the
// ground's, seen along `V` from a pixel covering `footprint` meters of ground.
// Every pixel of a surface growing grass takes its ground from here: litter
// strewn by each layer's density share, broken up by patches of earth and
// dry litter, and how much of the sky the leaf above lets reach it. Only a
// pixel past where the blades start handing the field over (`far`), `dist`
// meters from the camera at screen pixel `pixel`, goes on to the canopy, and
// only where that pixel shows the canopy rather than the blades.
CanopyShade canopy_at(GpuObjectData od, float3 pos, float3 normal, float3 color, float3 albedo,
                      float3 V, float footprint, float2 pixel, float dist, bool far)
{
    // A block not filled this frame (no grass resolved, or no planet tile
    // drawn) holds no layers and no planet, and the canopy is bare ground.
    uint first = min(od.far_canopy & 0xffu, CANOPY.layer_count);
    uint count = min((od.far_canopy >> 8u) & 0xffu, CANOPY.layer_count - first);
    bool planet = ((od.far_canopy >> 16u) & 1u) != 0u;
    if (planet && CANOPY.planet_radius <= 0.0)
    {
        count = 0u;
        planet = false;
    }

    CanopyShade c = (CanopyShade)0;
    float3 dir = float3(0.0, 1.0, 0.0);
    float2 plane = pos.xz;
    float3 ground_point = float3(pos.x, 0.0, pos.z);
    uint salt = 0u;
    c.up = float3(0.0, 1.0, 0.0);
    c.ground_albedo = albedo;
    if (planet)
    {
        float4 p = float4(pos, 1.0);
        float3 rel = float3(dot(CANOPY.planet_rows[0], p), dot(CANOPY.planet_rows[1], p),
                            dot(CANOPY.planet_rows[2], p));
        dir = normalize(rel);
        c.up = normalize(CANOPY.planet_rows[0].xyz * dir.x + CANOPY.planet_rows[1].xyz * dir.y
                         + CANOPY.planet_rows[2].xyz * dir.z);
        uint face;
        float2 st = grass_face_coords(dir, face);
        plane = (st + 1.0) * (0.25 * PI * CANOPY.planet_radius);
        ground_point = dir * CANOPY.planet_radius;
        salt = grass_face_salt(face);
    }

    // Detail a pixel cannot hold fades to its mean rather than aliasing.
    float meadow_detail = 1.0 - smoothstep(4.0, 12.0, footprint);
    float macro_detail = 1.0 - smoothstep(30.0, 90.0, footprint);
    float slope = 1.0 - saturate(dot(normalize(normal), c.up));

    // The first layers' weights, which their views are blended by, and the
    // leaf-area weighted blade colors at the root and the tip; the litter by
    // density share, and how much of the ground no layer covers.
    float4 weights = (float4)(0.0);
    float weight = 0.0;
    float most = -1.0;
    float3 root = (float3)(0.0);
    float3 tip = (float3)(0.0);
    float bend = 0.0;
    float sky_projection = 0.0;
    float3 litter = (float3)(0.0);
    float litter_weight = 0.0;
    float open_ground = 1.0;
    for (uint i = 0u; i < count; i++)
    {
        CanopyLayer L = CANOPY.layers[first + i];
        float share = i == 0u ? color.r : (i == 1u ? color.g : color.b);
        if (planet)
        {
            share = CANOPY.meadow_mean;
            if (meadow_detail > 0.0)
            {
                share = lerp(share, grass_meadow_density(dir, CANOPY.meadow_scale, L.seed),
                             meadow_detail);
            }
        }
        litter += share * L.litter_clumping.rgb;
        litter_weight += share;
        open_ground *= 1.0 - saturate(share);

        // A clump's own hue, height and crowding, where a pixel past the
        // blades can hold a clump.
        float clump_detail = far ? 1.0 - smoothstep(0.35, 1.0, footprint / L.shape.x) : 0.0;
        float dryness = 1.0 / 3.0;
        float height = 1.0;
        float crowd = 1.0;
        if (clump_detail > 0.0)
        {
            float2 center;
            float spot;
            uint ch = grass_clump_at(plane, L.shape.x, L.seed, salt, center, spot);
            dryness = lerp(dryness, grass_clump_dryness(ch), clump_detail);
            height = lerp(height, grass_clump_height(ch, L.shape.y), clump_detail);
            crowd = lerp(crowd, grass_gathered_density(spot, L.litter_clumping.w), clump_detail);
        }
        float area = L.shape.z * share * height * crowd;
        c.leaf_area += area;
        // Every layer weighs in a little even where it is bare, so the colors
        // stay defined where nothing grows.
        float w = area + 1e-4;
        sky_projection += w * L.sky_projection;
        weight += w;
        if (far)
        {
            float dry = L.root_color_variation.w * dryness;
            float2 drift = grass_macro(ground_point, L.seed, slope, L.tip_color_macro.w,
                                       macro_detail);
            float3 tint = lerp((float3)(1.0), GRASS_DRY_TINT, dry) * grass_macro_tint(drift);
            root += w * L.root_color_variation.rgb * tint;
            tip += w * L.tip_color_macro.rgb * tint;
            bend += w * L.shape.w;
            if (i < 4u)
            {
                weights[i] = w;
            }
            if (w > most)
            {
                most = w;
                c.table = first + i;
            }
        }
    }
    weight = max(weight, 1e-6);
    sky_projection /= weight;

    // Matches SKY_MU in render::grass::canopy.
    const float sky_mu = 2.0 / 3.0;
    c.ground_sky = exp(-sky_projection * c.leaf_area / sky_mu);
    c.handed = far ? canopy_handed(pixel, dist) : 0.0;
    float handed_area = c.handed * c.leaf_area;

    // The ground: the surface's own albedo as bare earth, darker in patches,
    // under litter where the layers grow, lighter and yellower where it lies
    // dry.
    if (count > 0u)
    {
        float3 v = grass_ground_variation(ground_point, footprint);
        float3 shed = litter / max(litter_weight, 1e-6);
        shed *= lerp((float3)(1.0), GRASS_LITTER_DRY, v.z);
        c.ground_albedo = lerp(albedo * v.y, shed, (1.0 - open_ground) * v.x);
    }
    if (!far)
    {
        return c;
    }

    root /= weight;
    tip /= weight;
    bend /= weight;

    // Where the view stops, each layer's table read at the field's whole leaf
    // area where the pixel shows the canopy (none where it shows the ground
    // between kept blades) and blended by the layer's share of it.
    float mu = max(dot(V, c.up), CANOPY_MIN_MU);
    float x = canopy_angle_coord(mu);
    float viewed = max(weights.x + weights.y + weights.z + weights.w, 1e-6);
    CanopyView view = (CanopyView)0;
    view.ground = count == 0u ? 1.0 : 0.0;
    for (uint k = 0u; k < min(count, 4u); k++)
    {
        canopy_add_view(view, first + k, x, mu, handed_area, weights[k] / viewed);
    }
    CanopyBand upper = canopy_band_mean(view.upper);
    CanopyBand lower = canopy_band_mean(view.lower);

    float3 span = tip - root;
    c.ground = view.ground;
    c.seen = upper.share * (root + span * upper.ramp) + lower.share * (root + span * lower.ramp);
    c.upper_tilt = canopy_tilt(bend, upper.height);
    c.lower_tilt = canopy_tilt(bend, lower.height);
    c.lower_lit = lower.share > 1e-3;
    c.upper_direct = canopy_band_weights(upper.share, root, span, upper.occlusion,
                                         upper.ramp_occlusion);
    c.lower_direct = canopy_band_weights(lower.share, root, span, lower.occlusion,
                                         lower.ramp_occlusion);
    c.open = c.upper_direct + c.lower_direct;
    c.above = float2(upper.above, lower.above) * c.leaf_area;

    float3 flat_view = V - c.up * dot(V, c.up);
    float flat_len = length(flat_view);
    float3 any_side = abs(c.up.x) < 0.9 ? float3(1.0, 0.0, 0.0) : float3(0.0, 0.0, 1.0);
    c.along = flat_len > 1e-4 ? flat_view / flat_len : normalize(cross(c.up, any_side));
    return c;
}

// A direction in the canopy's frame: along the view's way over the ground,
// across it, and up.
float3 canopy_frame(CanopyShade c, float3 d)
{
    return float3(dot(d, c.along), dot(d, cross(c.up, c.along)), dot(d, c.up));
}

// The light a facing returns, weighted by how much of it faces the viewer,
// given its normal's cosines with the view, the light and the half vector;
// the normal is turned toward the viewer first.
float4 canopy_facing_light(float nv, float nl, float nh, GrassLightPair pair, inout float total)
{
    float s = nv < 0.0 ? -1.0 : 1.0;
    total += s * nv;
    return s * nv * grass_light_terms_dots(s * nl, s * nv, s * nh, pair);
}

// The light the facings around the view return at a height whose blades tilt
// by `tilt` (cosine, sine), averaged by how much of each faces the viewer.
// `v`, `l` and `h` are the view, light and half vector in the canopy's frame.
// Facing j and facing CANOPY_FACINGS - 1 - j mirror each other across the
// view, so they differ only in the across part of their normals.
float4 canopy_facings_light(float2 tilt, float3 v, float3 l, float3 h, GrassLightPair pair)
{
    float4 k = (float4)(0.0);
    float total = 0.0;
    [unroll]
    for (uint j = 0u; j < CANOPY_FACINGS / 2u; j++)
    {
        float a = GRASS_TAU * (float(j) + 0.5) / float(CANOPY_FACINGS);
        float3 n = float3(cos(a) * tilt.x, sin(a) * tilt.x, -tilt.y);
        float3 rest = float3(n.x * v.x + n.z * v.z, n.x * l.x + n.z * l.z, n.x * h.x + n.z * h.z);
        float3 side = n.y * float3(v.y, l.y, h.y);
        float3 one = rest + side;
        float3 other = rest - side;
        k += canopy_facing_light(one.x, one.y, one.z, pair, total);
        k += canopy_facing_light(other.x, other.y, other.z, pair, total);
    }
    return k / max(total, 1e-4);
}

// Gather a direct light from `L` at `radiance` (shadowing applied) into the
// canopy: the facings around the view at each band's height, through the leaf
// above it, the lower band only where a view can reach it.
void canopy_add_light(inout CanopyShade c, float3 V, float3 L, float3 radiance)
{
    // A light below the ground's horizon reaches no blade of the field: near
    // the camera the shadow cascades see the ground take it, past them this
    // does.
    // A light too faint to show once exposed (the moon by day) is skipped.
    radiance *= smoothstep(-0.03, 0.03, dot(L, c.up));
    if (max(radiance.r, max(radiance.g, radiance.b)) * VIEW.pre_exposure < 1e-3)
    {
        return;
    }
    GrassLightPair pair = grass_light_pair(V, L);
    float3 v = canopy_frame(c, V);
    float3 l = canopy_frame(c, L);
    float3 h = canopy_frame(c, pair.H);
    float mu = max(l.z, CANOPY_MIN_MU);
    float2 through = exp(-canopy_table_projection(c.table, canopy_angle_coord(mu)) / mu * c.above);
    float4 hi = canopy_facings_light(c.upper_tilt, v, l, h, pair) * through.x;
    float3 lit = c.upper_direct.rgb * hi.xyz + c.upper_direct.w * hi.w;
    if (c.lower_lit)
    {
        float4 lo = canopy_facings_light(c.lower_tilt, v, l, h, pair) * through.y;
        lit += c.lower_direct.rgb * lo.xyz + c.lower_direct.w * lo.w;
    }
    c.light += lit * radiance;
}

// Gather a local light from `L` at `radiance` into the canopy, lit at the
// facing toward the viewer.
void canopy_add_local_light(inout CanopyShade c, float3 V, float3 L, float3 radiance)
{
    float4 k = grass_light_terms(c.along, V, L);
    c.light += (c.open.rgb * k.xyz + c.open.w * k.w) * radiance;
}

// The canopy's color: the blades lit by the gathered light plus `sky` (the
// irradiance their facing toward the viewer sees, ambient scale and
// screen-space occlusion applied), over `ground`, the surface's own radiance
// under the canopy.
float3 canopy_radiance(CanopyShade c, float3 sky, float3 ground)
{
    return c.light + c.open.rgb * sky + c.ground * ground;
}

// The canopy's albedo seen along the view, for the unlit view.
float3 canopy_albedo_seen(CanopyShade c, float3 ground)
{
    return c.seen + c.ground * ground;
}

// How far into this frame's band a surface `dist` from the camera is: 0
// before the blades start thinning out (and for any surface growing no
// grass), 1 past where they end.
float canopy_band(GpuObjectData od, float dist)
{
    if (od.far_canopy == 0u)
    {
        return 0.0;
    }
    return saturate((dist - CANOPY.band.x) / max(CANOPY.band.y - CANOPY.band.x, 1e-3));
}
