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

// The two parts of the light a blade of unit albedo returns from `L`: what
// multiplies the albedo (xyz: wrapped diffuse on the lit face, and light
// carried through the blade when the source is behind it), and the soft
// sheen (w), which does not.
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

float4 grass_light_terms_from(float3 N, float3 V, float3 L, GrassLightPair p)
{
    float ndl = dot(N, L);
    float diffuse = saturate((ndl + 0.35) / 1.35) / PI;
    float behind = saturate(-ndl);
    float carried = (0.35 * behind + 0.9 * p.through) / PI;
    float nl = saturate(ndl);
    float nv = max(dot(N, V), 1e-3);
    const float roughness = 0.5;
    float spec = distribution_ggx(N, p.H, roughness) * geometry_smith(N, V, L, roughness)
               / max(4.0 * nv * nl, 1e-3);
    float sheen = p.fresnel * spec * 0.5 * nl;
    return float4(diffuse + carried * float3(1.1, 1.25, 0.6), sheen);
}

float4 grass_light_terms(float3 N, float3 V, float3 L)
{
    return grass_light_terms_from(N, V, L, grass_light_pair(V, L));
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

// How much of a direct light reaches a point `t` up the blade inside the
// field.
float grass_direct_share(float t)
{
    return lerp(0.55, 1.0, t);
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
    float direct_occlusion;
    float ramp_direct_occlusion;
};

// What the canopy is at one ground point, and how it is seen and lit there.
struct CanopyShade
{
    // Blades' up, and the view's direction along the ground with its side.
    float3 up;
    float3 along;
    float3 across;
    float leaf_area;
    float bend;
    // The blade area seen per unit of leaf area along the sky's mean
    // direction, and the layer whose table a light's way through is read
    // from: the one growing the most leaf.
    float sky_projection;
    uint table;
    // The leaf-area weighted blade colors at the root and the tip.
    float3 root;
    float3 tip;
    // Where the view stops: the ground's share, and the two bands.
    float ground;
    CanopyBand upper;
    CanopyBand lower;
    // Direct light gathered for each band, per unit of radiance at the band's
    // height: what the albedo multiplies, and the sheen it does not. Local
    // lights are gathered apart, since the blades take them whole.
    float3 upper_light;
    float3 upper_sheen;
    float3 lower_light;
    float3 lower_sheen;
    float3 local_light;
    float3 local_sheen;
};

// Matches render::grass::canopy_table: each layer's view, tabulated over the
// view's angle and the optical depth it crosses.
#define CANOPY_TABLE_ANGLES 16u
#define CANOPY_TABLE_DEPTHS 16u
#define CANOPY_TABLE_ROWS 4u
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

// Blade area of `layer` seen per unit of leaf area along view cosine `mu`.
float canopy_table_projection(uint layer, float mu)
{
    return canopy_table_row(canopy_table_at(layer, canopy_angle_coord(mu), 0.0), 0u).y;
}

// Add where rays seen along `mu` stop in `layer`'s canopy of `leaf_area`,
// weighted by `w`, into `c`: the ground's share, and each band's share with
// its terms summed by share. Mirrors render::grass::canopy_table::table_view.
void canopy_add_view(inout CanopyShade c, uint layer, float mu, float leaf_area, float w)
{
    mu = max(mu, CANOPY_MIN_MU);
    float x = canopy_angle_coord(mu);
    float depth = canopy_table_row(canopy_table_at(layer, x, 0.0), 0u).y * leaf_area / mu;
    float stopped = 1.0 - exp(-depth);
    float y = saturate(log(1.0 + max(depth, 0.0) / CANOPY_DEPTH_KNEE)
                       / log(1.0 + CANOPY_MAX_DEPTH / CANOPY_DEPTH_KNEE))
            * float(CANOPY_TABLE_DEPTHS - 1u);
    CanopyTableAt at = canopy_table_at(layer, x, y);
    float4 r0 = canopy_table_row(at, 0u);
    float4 r1 = canopy_table_row(at, 1u);
    float4 r2 = canopy_table_row(at, 2u);
    float2 r3 = canopy_table_row(at, 3u).xy;
    float upper = w * saturate(r0.x) * stopped;
    float lower = w * stopped - upper;
    c.ground += w * (1.0 - stopped);
    c.upper.share += upper;
    c.upper.height += upper * r0.z;
    c.upper.ramp += upper * r0.w;
    c.upper.occlusion += upper * r1.x;
    c.upper.ramp_occlusion += upper * r1.y;
    c.upper.direct_occlusion += upper * r1.z;
    c.upper.ramp_direct_occlusion += upper * r1.w;
    c.lower.share += lower;
    c.lower.height += lower * r2.x;
    c.lower.ramp += lower * r2.y;
    c.lower.occlusion += lower * r2.z;
    c.lower.ramp_occlusion += lower * r2.w;
    c.lower.direct_occlusion += lower * r3.x;
    c.lower.ramp_direct_occlusion += lower * r3.y;
}

// A band's terms summed by share, as means.
CanopyBand canopy_band_mean(CanopyBand band)
{
    float n = 1.0 / max(band.share, 1e-9);
    band.height *= n;
    band.ramp *= n;
    band.occlusion *= n;
    band.ramp_occlusion *= n;
    band.direct_occlusion *= n;
    band.ramp_direct_occlusion *= n;
    return band;
}

// How much of a light along `L` gets through the canopy to the ground.
float canopy_transmit(CanopyShade c, float3 L)
{
    float mu = max(dot(L, c.up), CANOPY_MIN_MU);
    return exp(-canopy_table_projection(c.table, mu) * c.leaf_area / mu);
}

// The canopy at `pos` on surface `od`, the vertex color `color` carrying a
// terrain's layer densities and `normal` the ground's, seen along `V` from a
// pixel covering `footprint` meters of ground.
CanopyShade canopy_at(GpuObjectData od, float3 pos, float3 normal, float3 color, float3 V,
                      float footprint)
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

    // The first layers' weights, which their views are blended by.
    float4 weights = (float4)(0.0);
    float weight = 0.0;
    float most = -1.0;
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
        // A clump's own hue and height, where a pixel can hold a clump.
        float clump_detail = 1.0 - smoothstep(0.35, 1.0, footprint / L.shape.x);
        float dryness = 1.0 / 3.0;
        float height = 1.0;
        if (clump_detail > 0.0)
        {
            float2 center;
            uint ch = grass_clump_at(plane, L.shape.x, L.seed, salt, center);
            dryness = lerp(dryness, grass_clump_dryness(ch), clump_detail);
            height = lerp(height, grass_clump_height(ch, L.shape.y), clump_detail);
        }
        float dry = L.root_color_variation.w * dryness;
        float2 drift = grass_macro(ground_point, L.seed, slope, L.tip_color_macro.w, macro_detail);
        float3 tint = lerp((float3)(1.0), GRASS_DRY_TINT, dry) * grass_macro_tint(drift);
        float area = L.shape.z * share * height;
        c.leaf_area += area;
        // Every layer weighs in a little even where it is bare, so the colors
        // stay defined where nothing grows.
        float w = area + 1e-4;
        c.root += w * L.root_color_variation.rgb * tint;
        c.tip += w * L.tip_color_macro.rgb * tint;
        c.bend += w * L.shape.w;
        c.sky_projection += w * L.sky_projection;
        weight += w;
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
    weight = max(weight, 1e-6);
    c.root /= weight;
    c.tip /= weight;
    c.bend /= weight;
    c.sky_projection /= weight;

    // Where the view stops, each layer's table read at the field's whole leaf
    // area and blended by the layer's share of it.
    float mu = dot(V, c.up);
    float viewed = max(weights.x + weights.y + weights.z + weights.w, 1e-6);
    c.ground = count == 0u ? 1.0 : 0.0;
    for (uint k = 0u; k < min(count, 4u); k++)
    {
        canopy_add_view(c, first + k, mu, c.leaf_area, weights[k] / viewed);
    }
    c.upper = canopy_band_mean(c.upper);
    c.lower = canopy_band_mean(c.lower);
    float3 flat_view = V - c.up * dot(V, c.up);
    float flat_len = length(flat_view);
    float3 any_side = abs(c.up.x) < 0.9 ? float3(1.0, 0.0, 0.0) : float3(0.0, 0.0, 1.0);
    c.along = flat_len > 1e-4 ? flat_view / flat_len : normalize(cross(c.up, any_side));
    c.across = cross(c.up, c.along);
    return c;
}

// The normal of the blade facing `front`, tilted by its curve (cosine and
// sine of the tilt in `tilt`), turned toward the viewer, and how much of it
// faces the viewer.
float4 canopy_facing(CanopyShade c, float3 front, float2 tilt, float3 V)
{
    float3 n = front * tilt.x - c.up * tilt.y;
    n = dot(n, V) < 0.0 ? -n : n;
    return float4(n, dot(n, V));
}

// Gather a direct light from `L` at `radiance` (shadowing applied) into the
// canopy's bands: the facings around the view at each band's height, each
// weighted by how much of it faces the viewer; the lower band only where a
// view can reach it.
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
    bool lower = c.lower.share > 1e-3;
    float2 cs_hi = canopy_tilt(c.bend, c.upper.height);
    float2 cs_lo = canopy_tilt(c.bend, c.lower.height);
    float4 hi = (float4)(0.0);
    float4 lo = (float4)(0.0);
    float2 total = (float2)(0.0);
    [unroll]
    for (uint j = 0u; j < CANOPY_FACINGS; j++)
    {
        float a = GRASS_TAU * (float(j) + 0.5) / float(CANOPY_FACINGS);
        float3 front = c.along * cos(a) + c.across * sin(a);
        float4 n = canopy_facing(c, front, cs_hi, V);
        hi += n.w * grass_light_terms_from(n.xyz, V, L, pair);
        total.x += n.w;
        if (lower)
        {
            n = canopy_facing(c, front, cs_lo, V);
            lo += n.w * grass_light_terms_from(n.xyz, V, L, pair);
            total.y += n.w;
        }
    }
    hi /= max(total.x, 1e-4);
    c.upper_light += hi.xyz * radiance;
    c.upper_sheen += hi.w * radiance;
    if (lower)
    {
        lo /= max(total.y, 1e-4);
        c.lower_light += lo.xyz * radiance;
        c.lower_sheen += lo.w * radiance;
    }
}

// Gather a local light from `L` at `radiance` into the canopy, lit at the
// facing toward the viewer.
void canopy_add_local_light(inout CanopyShade c, float3 V, float3 L, float3 radiance)
{
    float4 k = grass_light_terms(c.along, V, L);
    c.local_light += k.xyz * radiance;
    c.local_sheen += k.w * radiance;
}

// One band's radiance: its blades' mean color times what their heights let
// through of the gathered direct light and of the sky and local lights, plus
// the sheen.
float3 canopy_band_radiance(CanopyShade c, CanopyBand b, float3 direct, float3 sheen, float3 sky)
{
    float3 span = c.tip - c.root;
    float3 lit = c.root * b.direct_occlusion + span * b.ramp_direct_occlusion;
    float3 open = c.root * b.occlusion + span * b.ramp_occlusion;
    return lit * direct + open * (sky + c.local_light) + sheen * b.direct_occlusion
         + c.local_sheen * b.occlusion;
}

// The canopy's color seen along `V`: the upper and lower blades lit by what
// was gathered plus `sky` (the irradiance their facing sees, ambient scale
// and screen-space occlusion applied), over `ground`, the surface's own
// radiance under the canopy.
float3 canopy_radiance(CanopyShade c, float3 sky, float3 ground)
{
    return c.upper.share * canopy_band_radiance(c, c.upper, c.upper_light, c.upper_sheen, sky)
         + c.lower.share * canopy_band_radiance(c, c.lower, c.lower_light, c.lower_sheen, sky)
         + c.ground * ground;
}

// The canopy's albedo seen along `V`, for the unlit view.
float3 canopy_albedo_seen(CanopyShade c, float3 ground)
{
    float3 span = c.tip - c.root;
    return c.upper.share * (c.root + span * c.upper.ramp)
         + c.lower.share * (c.root + span * c.lower.ramp) + c.ground * ground;
}

// How much of the sky reaches the ground under the canopy.
float canopy_ground_sky(CanopyShade c)
{
    // Matches SKY_MU in render::grass::canopy.
    const float mu = 2.0 / 3.0;
    return exp(-c.sky_projection * c.leaf_area / mu);
}

// How far into its far-field band a surface `dist` from the camera is: 0
// before it (and for any surface without one), 1 past it.
float canopy_band(GpuObjectData od, float dist)
{
    if (od.far_end <= 0.0)
    {
        return 0.0;
    }
    return saturate((dist - od.far_start) / max(od.far_end - od.far_start, 1e-3));
}
