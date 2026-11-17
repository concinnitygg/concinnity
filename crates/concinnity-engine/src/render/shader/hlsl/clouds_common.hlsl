// The clouds' density and lighting, for the march kernel. The includer
// defines CLOUD as its `CloudParams` block and DAYLIGHT as that block's
// daylight, splices `daylight.hlsl`, binds the noise volume as `cloud_noise`
// with the repeating `cloud_noise_sampler`, and the weather map as
// `cloud_weather` with the clamping `lut_sampler`. Every position is
// kilometers from the planet's center in its fixed axes. The shaping
// functions are mirrored in render::environment::clouds::shape.

static const float CLOUD_PI = 3.14159265358979;
// How far the base shape erodes into the detail at full strength.
static const float CLOUD_DETAIL_EROSION = 0.45;
// How the weather map's channels scale a layer: its towering at the least and
// the most convection, and its density from the thinnest veil to the densest
// core, as authored at the channel's middle. `render::environment::clouds::weather` mirrors them.
static const float CLOUD_TOWERING_LOW = 0.35;
static const float CLOUD_TOWERING_HIGH = 1.65;
static const float CLOUD_DENSITY_LOW = 0.05;
static const float CLOUD_DENSITY_HIGH = 3.0;
// How far a sheet flattens the base shape's puffs toward an even deck.
static const float CLOUD_SHEET_FLATTEN = 0.7;
// How far the cover's edge widens once a pixel spans a whole weather cell,
// and the pixel's share of a cell at which it starts and stops widening.
static const float CLOUD_SOFT_COVER = 0.5;
static const float CLOUD_SOFT_COVER_START = 0.25;
static const float CLOUD_SOFT_COVER_END = 1.0;
// How much the powder effect darkens a cloud's sunward skin.
static const float CLOUD_POWDER = 0.6;
// How strongly a cloud's first scattering throws light forward: the silver
// lining round a cloud before the sun.
static const float CLOUD_FORWARD_G = 0.6;
// The asymmetry the two-stream approximation takes for cloud droplets, and
// its factor on the share not scattered forward, which set how fast a
// cloud's optical depth turns light back. `render::environment::clouds::light`
// mirrors them.
static const float CLOUD_TWO_STREAM_G = 0.85;
static const float CLOUD_TWO_STREAM_SCALE = 0.75;
// The diffuse light's profile through a level column (see
// `cloud_diffuse_share`), tabulated by the column's optical depth at
// CLOUD_DIFFUSE_STEPS a doubling from 2^CLOUD_DIFFUSE_LOW, each entry the
// profile's (constant, slope, bottom) terms. `diffuse_table` in
// render::environment::clouds::light solves and locks them.
static const uint CLOUD_DIFFUSE_COUNT = 41u;
static const float CLOUD_DIFFUSE_LOW = -4.0;
static const float CLOUD_DIFFUSE_STEPS = 4.0;
static const float3 CLOUD_DIFFUSE_TABLE[CLOUD_DIFFUSE_COUNT] = {
    float3(0.0, 0.0, 0.0152806),
    float3(0.0, 0.0, 0.0162661),
    float3(0.0, 0.0, 0.017434),
    float3(0.0, 0.0, 0.0188226),
    float3(0.0, 0.0, 0.0204799),
    float3(0.0, 0.0, 0.022466),
    float3(0.0, 0.0, 0.0248569),
    float3(0.0, 0.0, 0.0277488),
    float3(0.0, 0.0, 0.0312648),
    float3(0.0, 0.0, 0.035562),
    float3(0.0, 0.0, 0.0408424),
    float3(0.0, 0.0, 0.0473654),
    float3(0.0, 0.0, 0.0554639),
    float3(0.0, 0.0, 0.065563),
    float3(0.0, 0.0, 0.0781999),
    float3(0.0, 0.0, 0.0940408),
    float3(0.0, 0.0, 0.113885),
    float3(0.0, 0.0, 0.138633),
    float3(0.0, 0.0, 0.169191),
    float3(0.0, 0.0, 0.206238),
    float3(0.0, 0.0, 0.249821),
    float3(0.00727555, 0.0, 0.281236),
    float3(0.036526, 0.0, 0.254191),
    float3(0.0766121, -0.00497657, 0.206097),
    float3(0.155243, -0.0203818, 0.0974473),
    float3(0.211652, -0.026393, 0.00655241),
    float3(0.201834, -0.0171337, 0.0),
    float3(0.185843, -0.00929125, 0.0),
    float3(0.169452, -0.00358666, 0.0),
    float3(0.153371, 0.000401232, 0.0),
    float3(0.137989, 0.00306466, 0.0),
    float3(0.123475, 0.00473596, 0.0),
    float3(0.109896, 0.00568004, 0.0),
    float3(0.0972876, 0.00610173, 0.0),
    float3(0.085673, 0.00615759, 0.0),
    float3(0.0750637, 0.00596652, 0.0),
    float3(0.0654549, 0.00561788, 0.0),
    float3(0.0568238, 0.00517779, 0.0),
    float3(0.0491315, 0.00469407, 0.0),
    float3(0.0423255, 0.00420031, 0.0),
    float3(0.0363434, 0.00371914, 0.0),
};
// The share of the light above a deep cloud its first scattering sends back
// up out of it, which the diffuse light past the table leaves out.
static const float CLOUD_FIRST_BACK = 0.023994;

float cloud_remap(float v, float lo, float hi)
{
    return saturate((v - lo) / max(hi - lo, 1e-4));
}

// `height_profile` in render::environment::clouds.
float cloud_height_profile(float h, float towering, float weather)
{
    if (h < 0.0 || h > 1.0)
    {
        return 0.0;
    }
    float top = clamp(lerp(0.25, 1.0, towering) * (0.55 + 0.45 * weather), 0.1, 1.0);
    float base = saturate(h / 0.06);
    return base * (1.0 - smoothstep(top * 0.45, top, h));
}

// `weather_cover` in render::environment::clouds.
float cloud_weather_cover(float weather, float coverage, float soft)
{
    float threshold = 1.0 - coverage;
    float band = 0.2 + soft;
    float cover = smoothstep(threshold - band, threshold + band, weather);
    return cover * saturate(coverage * 8.0);
}

// How far the cover's edge widens for a pixel `footprint_km` across over
// cells `cell_km` apart. `soft_cover` in render::environment::clouds.
float cloud_soft_cover(float footprint_km, float cell_km)
{
    return CLOUD_SOFT_COVER
         * smoothstep(CLOUD_SOFT_COVER_START, CLOUD_SOFT_COVER_END, footprint_km / cell_km);
}

// `lean_coverage` in render::environment::clouds.
float cloud_lean_coverage(float coverage, float lean)
{
    float room = lean > 0.0 ? 1.0 - coverage : coverage;
    return saturate(coverage + lean * room);
}

// `coverage_density` in render::environment::clouds.
float cloud_coverage_density(float shape, float profile, float cover)
{
    if (cover <= 0.0)
    {
        return 0.0;
    }
    float lo = 1.0 - cover;
    return saturate((shape * profile - lo) / max(1.0 - lo, 1e-4)) * cover;
}

// Distances along a unit ray from `origin` at which it crosses the sphere of
// `radius`, nearest first; false when it misses.
bool cloud_sphere_hits(float3 origin, float3 dir, float radius, out float2 hits)
{
    float b = dot(origin, dir);
    float c = dot(origin, origin) - radius * radius;
    float disc = b * b - c;
    hits = (float2)0.0;
    if (disc < 0.0)
    {
        return false;
    }
    float s = sqrt(disc);
    hits = float2(-b - s, -b + s);
    return true;
}

// `shell_spans` in render::environment::clouds: the stretches of the ray
// inside the shell between `lower` and `upper` before it reaches `ground`,
// nearer first in `spans.xy` and then `spans.zw`, and how many there are. A
// ray that dips under the shell's bottom and misses the ground rises back
// into the shell beyond, its second stretch.
uint cloud_shell_spans(float3 origin, float3 dir, float ground, float lower, float upper,
                       out float4 spans)
{
    spans = (float4)0.0;
    float r = length(origin);
    float2 up_hits;
    if (!cloud_sphere_hits(origin, dir, upper, up_hits))
    {
        return 0u;
    }
    float2 low_hits;
    bool low = cloud_sphere_hits(origin, dir, lower, low_hits);
    float2 ground_hits;
    bool hits_ground = cloud_sphere_hits(origin, dir, ground, ground_hits) && ground_hits.x > 0.0;
    float2 first;
    float2 second = (float2)0.0;
    if (r < lower)
    {
        if (hits_ground)
        {
            return 0u;
        }
        first = float2(low ? low_hits.y : 0.0, up_hits.y);
    }
    else
    {
        float start = 0.0;
        if (r > upper)
        {
            if (up_hits.x < 0.0)
            {
                return 0u;
            }
            start = up_hits.x;
        }
        bool down = low && low_hits.x > 0.0;
        first = float2(start, down ? low_hits.x : up_hits.y);
        if (down && !hits_ground)
        {
            second = float2(low_hits.y, up_hits.y);
        }
    }
    first.x = max(first.x, 0.0);
    if (first.y <= first.x)
    {
        return 0u;
    }
    spans = float4(first, second);
    return second.y > second.x ? 2u : 1u;
}

// `light_span` in render::environment::clouds: the stretch of the line from
// `q` along `light` inside the shell between `lower` and `upper`, toward the
// light: the one holding `q` or the first past it, short of the bottom
// sphere.
bool cloud_light_span(float3 q, float3 light, float lower, float upper, out float2 span)
{
    span = (float2)0.0;
    float2 up_hits;
    if (!cloud_sphere_hits(q, light, upper, up_hits) || up_hits.y <= 0.0)
    {
        return false;
    }
    float2 low_hits;
    bool low = cloud_sphere_hits(q, light, lower, low_hits);
    if (low && low_hits.x > 0.0)
    {
        span = float2(up_hits.x, low_hits.x);
    }
    else
    {
        span = float2(low ? max(up_hits.x, low_hits.y) : up_hits.x, up_hits.y);
    }
    span.x = max(span.x, 0.0);
    return span.y > span.x;
}

float4 cloud_noise_at(float3 coord)
{
    return cloud_noise.SampleLevel(cloud_noise_sampler, coord, 0.0);
}

// The radius a point drawn at radius `r` stands at as authored.
// `CloudHeights::authored` in render::environment::clouds.
float cloud_authored_radius(float r)
{
    return CLOUD.horizon.w + (r - CLOUD.horizon.w) / CLOUD.horizon.z;
}

// One view ray's frame: the camera and its up, in the fixed axes.
struct CloudRay
{
    float3 camera;
    float3 camera_up;
    float3 dir;
};

// `p` with its part along layer `L`'s stretch axis shrunk by its stretch, so
// the noise spans that many times as far along the wind. `stretched` in
// render::environment::clouds.
float3 cloud_stretched(CloudLayerBlock L, float3 p)
{
    float3 axis = float3(L.stretch.x, 0.0, L.stretch.z);
    return p + axis * (dot(p, axis) * L.stretch.w);
}

// A layer's weather over one point of its bottom: how much of the cell is
// cloud, the weather field there, how tall its clouds grow, how dense they
// are against the layer's own density, and how far they merge into a sheet.
// Zero cover is clear sky.
struct CloudWeather
{
    float cover;
    float field;
    float towering;
    float density;
    float sheet;
};

CloudWeather cloud_clear_weather()
{
    CloudWeather w;
    w.cover = 0.0;
    w.field = 0.0;
    w.towering = 0.0;
    w.density = 0.0;
    w.sheet = 0.0;
    return w;
}

// The weather map's texel over the unit direction `d` (fixed axes), read
// within its face's block, whose gutter carries a bilinear read across the
// face's edge. `map_point` in render::environment::clouds::weather.
float4 cloud_weather_at(float3 d)
{
    float2 uv;
    uint face = cloud_weather_face(d, uv);
    float size = float(CLOUD_WEATHER_FACE);
    float2 texel = clamp((uv * 0.5 + 0.5) * size + 1.0, 1.0, size + 1.0);
    float2 corner = float2(face % 3u, face / 3u) * float(CLOUD_WEATHER_STRIDE);
    float2 map = float2(3.0, 2.0) * float(CLOUD_WEATHER_STRIDE);
    return cloud_weather.SampleLevel(lut_sampler, (corner + texel) / map, 0.0);
}

// The weather of layer `i` over the point `up` (unit, from the planet's
// center) for a pixel `footprint_km` across there; the weather reads the
// layer's bottom there, so a cloud's cell does not lean with height.
CloudWeather cloud_layer_weather(uint i, CloudRay ray, float3 up, float footprint_km)
{
    CloudLayerBlock L = CLOUD.layers[i];
    float3 below = cloud_stretched(L, (up - ray.camera_up) * L.weather_second.w);
    float4 w1 = cloud_noise_at(L.weather.xyz + below * L.scales.y);
    float w2 = cloud_noise_at(L.weather_second.xyz + below * L.weather.w).b;
    CloudWeather w;
    w.field = saturate(lerp(w1.b, w2, 0.35) * 0.75 + w1.a * 0.4 - 0.1);
    float coverage = L.shell.z;
    w.towering = L.shell.w;
    w.density = 1.0;
    w.sheet = 0.0;
    CloudWeatherBlock W = CLOUD.weather;
    if (W.pole.w > 0.5)
    {
        // The map holds the systems where they started; read it turned back
        // by their drift.
        float4 m = cloud_weather_at(cloud_weather_turn(up, W.pole.xyz, W.drift.x, -W.drift.y));
        coverage = cloud_lean_coverage(coverage, m.r * 2.0 - 1.0);
        w.towering = saturate(w.towering * lerp(CLOUD_TOWERING_LOW, CLOUD_TOWERING_HIGH, m.g));
        w.density = exp(m.b < 0.5 ? log(CLOUD_DENSITY_LOW) * (1.0 - 2.0 * m.b)
                                  : log(CLOUD_DENSITY_HIGH) * (2.0 * m.b - 1.0));
        w.sheet = m.a;
    }
    // A pixel wider than the cells sees their share, not each one.
    float cell_km = 1.0 / (L.scales.y * CLOUD_WEATHER_CELLS);
    float soft = cloud_soft_cover(footprint_km, cell_km);
    w.cover = lerp(cloud_weather_cover(w.field, coverage, soft), coverage, w.sheet);
    return w;
}

// How much of layer `L`'s detail a pixel `footprint_km` across resolves: all
// of it while the detail's coarsest cell spans two pixels, none under one. A
// footprint of 0 resolves it all.
float cloud_detail_share(CloudLayerBlock L, float footprint_km)
{
    float cell_km = 1.0 / (L.scales.w * CLOUD_DETAIL_CELLS);
    return smoothstep(1.0, 2.0, cell_km / max(footprint_km, 1e-9));
}

// Extinction per kilometer of layer `i` at radius `r`, `rel` from the camera,
// under weather `w`; `detail` wears the edges with the detail noise, of which
// a pixel resolves the share `share` (see `cloud_detail_share`).
float cloud_layer_density(uint i, CloudWeather w, float r, float3 rel, bool detail, float share)
{
    CloudLayerBlock L = CLOUD.layers[i];
    float height = (r - L.shell.x) / (L.shell.y - L.shell.x);
    if (w.cover <= 0.0 || height < 0.0 || height > 1.0)
    {
        return 0.0;
    }
    float profile = cloud_height_profile(height, w.towering, w.field);
    if (profile <= 0.0)
    {
        return 0.0;
    }
    // The noise is read where the point stands as authored, so a layer drawn
    // shrunk keeps its clouds' shapes.
    rel += (CLOUD.camera.xyz + rel) * (cloud_authored_radius(r) / r - 1.0);
    rel = cloud_stretched(L, rel);
    float shape = cloud_noise_at(L.shape.xyz + rel * L.scales.z).r;
    shape = lerp(shape, 1.0, w.sheet * CLOUD_SHEET_FLATTEN);
    float d = cloud_coverage_density(shape, profile, w.cover);
    if (detail && d > 0.0)
    {
        // Wispy toward the base, billowed toward the top. Where a pixel
        // resolves none of the detail, the edges wear by its mean.
        float n = share > 0.0
            ? lerp(CLOUD_DETAIL_MEAN, cloud_noise_at(L.detail.xyz + rel * L.scales.w).g, share)
            : CLOUD_DETAIL_MEAN;
        float wear = lerp(1.0 - n, n, saturate(height * 3.0)) * CLOUD_DETAIL_EROSION;
        d = cloud_remap(d, wear, 1.0);
    }
    return d * L.scales.x * w.density;
}

// The layers' weather over the point `up`, for a pixel `footprint_km` across.
void cloud_weathers(CloudRay ray, float3 up, float footprint_km, out CloudWeather w[CLOUD_MAX_LAYERS])
{
    uint count = uint(CLOUD.shell.z);
    [unroll]
    for (uint i = 0u; i < CLOUD_MAX_LAYERS; ++i)
    {
        w[i] = cloud_clear_weather();
        if (i < count)
        {
            w[i] = cloud_layer_weather(i, ray, up, footprint_km);
        }
    }
}

// The layers' weather over the point `up`, for a pixel `footprint_km` across,
// where layer `known`'s is already `weather`.
void cloud_weathers_with(CloudRay ray, float3 up, float footprint_km, uint known,
                         CloudWeather weather, out CloudWeather w[CLOUD_MAX_LAYERS])
{
    uint count = uint(CLOUD.shell.z);
    [unroll]
    for (uint i = 0u; i < CLOUD_MAX_LAYERS; ++i)
    {
        w[i] = cloud_clear_weather();
        if (i == known)
        {
            w[i] = weather;
        }
        else if (i < count)
        {
            w[i] = cloud_layer_weather(i, ray, up, footprint_km);
        }
    }
}

// Extinction per kilometer of every layer at radius `r`, `rel` from the
// camera, under the layers' weather `w`, and the height fraction through the
// layer holding the most of it.
float cloud_density(CloudWeather w[CLOUD_MAX_LAYERS], float r, float3 rel, bool detail,
                    float share, out float height)
{
    float total = 0.0;
    height = 0.0;
    [unroll]
    for (uint i = 0u; i < CLOUD_MAX_LAYERS; ++i)
    {
        float d = cloud_layer_density(i, w[i], r, rel, detail, share);
        if (d > total)
        {
            CloudLayerBlock L = CLOUD.layers[i];
            height = (r - L.shell.x) / (L.shell.y - L.shell.x);
        }
        total += d;
    }
    return total;
}

float cloud_hg(float nu, float g)
{
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * CLOUD_PI * pow(max(1.0 + g2 - 2.0 * g * nu, 1e-4), 1.5));
}

// How one view ray's first scattering of each body's light turns toward the
// viewer, constant along the ray.
struct CloudPhase
{
    float first[DAYLIGHT_BODIES];
    // How far round toward the body the ray looks, 0 away to 1 into it.
    float toward[DAYLIGHT_BODIES];
};

CloudPhase cloud_phase_along(float3 dir)
{
    CloudPhase phase;
    [unroll]
    for (uint i = 0u; i < DAYLIGHT_BODIES; i++)
    {
        float nu = dot(dir, DAYLIGHT.body_dir[i].xyz);
        phase.first[i] = cloud_hg(nu, CLOUD_FORWARD_G);
        phase.toward[i] = saturate(nu * 0.5 + 0.5);
    }
    return phase;
}

// How fast a cloud's optical depth turns light back in the two-stream
// approximation: three quarters of the share of scattering that is not
// forward. `two_stream_rate` in render::environment::clouds::light.
float cloud_two_stream_rate()
{
    return CLOUD_TWO_STREAM_SCALE * (1.0 - CLOUD_TWO_STREAM_G);
}

// The share of a light that comes through `optical_depth` of cloud, straight
// or scattered. `sunlit_share` in render::environment::clouds::light.
float cloud_sunlit_share(float optical_depth)
{
    return 1.0 / (1.0 + cloud_two_stream_rate() * optical_depth);
}

// The share of a light that keeps on straight in its beam through
// `optical_depth` of cloud: what lights the ground under a cloud as
// sunlight. The light the cloud scatters reaches the ground as the sky's
// light, through the clouds the sky's light is drawn with.
// `beam_share` in render::environment::clouds::light.
float cloud_beam_share(float optical_depth)
{
    return exp(-optical_depth);
}

// The light a point `depth` deep toward a light, in a level column of cloud
// `column` deep, scatters toward the viewer per unit extinction after its
// first scattering, per unit of the light above the cloud: `u + v (T - d) +
// w e^-(T - d)` for a column T deep, solved so that with the first
// scattering the column sends up and down what the two-stream approximation
// reflects and lets through scattered. A thin column whose first scattering
// alone sends back too much keeps the light it is given by sending the rest
// down from its bottom. `diffuse_share` in render::environment::clouds::light.
float cloud_diffuse_share(float column, float depth)
{
    float T = max(column, 0.0);
    float below = T - clamp(depth, 0.0, T);
    float top = exp2(float(CLOUD_DIFFUSE_COUNT - 1u) / CLOUD_DIFFUSE_STEPS + CLOUD_DIFFUSE_LOW);
    if (T > top)
    {
        // Past the table the profile is the deep column's straight line.
        float k = cloud_two_stream_rate();
        float through = 1.0 / (1.0 + k * T);
        float back = 1.0 - through - CLOUD_FIRST_BACK;
        float v = (back - through) / (CLOUD_PI * (T - 4.0 / 3.0));
        return max(through / CLOUD_PI - 2.0 / 3.0 * v + v * below, 0.0);
    }
    float x = clamp((log2(max(T, 1e-6)) - CLOUD_DIFFUSE_LOW) * CLOUD_DIFFUSE_STEPS, 0.0,
                    float(CLOUD_DIFFUSE_COUNT - 1u));
    uint i = min(uint(x), CLOUD_DIFFUSE_COUNT - 2u);
    float3 c = lerp(CLOUD_DIFFUSE_TABLE[i], CLOUD_DIFFUSE_TABLE[i + 1u], x - float(i));
    return max(c.x + c.y * below + c.z * exp(-below), 0.0);
}

// Where the shadow maps lie in the clouds' frame: the point on their sphere
// the line from `p` (at radius `r`) along the unit `light` meets, nearest
// `p`, as a point on the maps in [0, 1], and how far along the line from that
// point `p` lies, toward the light positive. The maps lay the sphere out
// round their center by the straight distance to it (a Lambert azimuthal
// equal-area projection). False off the maps; `trust` fades from 1 to 0
// toward their edges.
bool cloud_shadow_point(float3 p, float r, float3 light, out float2 uv, out float along,
                        out float trust)
{
    uv = (float2)0.0;
    along = 0.0;
    trust = 0.0;
    float radius = CLOUD.shadow_east.w;
    // The root nearest `p`, in the stable form so a planet's radius loses no
    // precision.
    float b = dot(p, light);
    float c = (r - radius) * (r + radius);
    float disc = b * b - c;
    if (disc < 0.0)
    {
        return false;
    }
    float s = sqrt(disc);
    float d = b + (b >= 0.0 ? s : -s);
    float t = d != 0.0 ? -c / d : 0.0;
    float3 q = p + light * t;
    if (dot(q, CLOUD.shadow_center.xyz) < CLOUD.shadow_north.w * radius * radius)
    {
        return false;
    }
    float2 side = float2(dot(q, CLOUD.shadow_east.xyz), dot(q, CLOUD.shadow_north.xyz));
    float3 chord = q - CLOUD.shadow_center.xyz;
    uv = side * (sqrt(dot(chord, chord)) * rsqrt(max(dot(side, side), 1e-12))
                 / CLOUD.shadow_center.w);
    trust = 1.0 - smoothstep(0.35, 0.5, max(abs(uv.x), abs(uv.y)));
    uv += 0.5;
    along = -t;
    return trust > 0.0;
}

// Layer `layer`'s tile of the shadow map read at `uv` (see
// `cloud_shadow_point`), kept off the other tile.
float4 cloud_shadow_layer(uint layer, float2 uv)
{
    float edge = 0.5 / float(CLOUD_SHADOW_SIZE);
    float2 at = float2(uv.x, (float(layer) + clamp(uv.y, edge, 1.0 - edge)) * 0.5);
    return cloud_shadow_map.SampleLevel(lut_sampler, at, 0.0);
}

// The light march's last steps the shadow map stands in for where it is
// trusted.
static const uint CLOUD_MAPPED_STEPS = 2u;

// The least sine of the light's height over the horizon the cloud behind a
// sample is reckoned down to its layer's bottom at, so a low light's long
// path through the layer does not take in clouds far behind.
static const float CLOUD_BEHIND_MIN_SINE = 0.2;

// One layer's Beer shadow map texel: where along the line its cloud starts
// toward the light (x) and ends away from it (y), and its optical depth along
// the line (z). The optical depth from the cloud's start toward the light
// down to `along`, spread evenly over the cloud between.
float cloud_beer_depth(float4 layer, float along)
{
    float span = max(layer.x - layer.y, 1e-4);
    return layer.z / span * clamp(layer.x - along, 0.0, span);
}

// The optical depth of the cloud behind the point at `along`, away from the
// light, down to its layer's bottom `height_km` below it with the light at
// sine `sine` over the horizon. `beer_behind` in
// render::environment::clouds::light.
float cloud_beer_behind(float4 layer, float along, float height_km, float sine)
{
    float reach_km = max(height_km, 0.0) / max(sine, CLOUD_BEHIND_MIN_SINE);
    return cloud_beer_depth(layer, along - reach_km) - cloud_beer_depth(layer, along);
}

// The daylight curves' cosine for a light at cosine `mu` over the horizon of
// a point `height` km over a ground `ground` km in radius: the point sees the
// light down to its own horizon, which sinks as it rises, and that horizon
// reads as the ground's. The ground is the one the sky is drawn over, so a
// cloud stays lit after sunset as long as the sky would let it.
float cloud_light_cosine(float mu, float height, float ground)
{
    float r = ground + max(height, 0.0);
    float sink = -sqrt(saturate(1.0 - (ground * ground) / (r * r)));
    return (mu - sink) / (1.0 - sink);
}

// The up of a point whose true up is `up`, set where the ground the sky is
// drawn over would put it: as far from the camera, so its angle from the
// camera's up shrinks by the true ground's share of that ground's radius.
float3 cloud_sky_up(float3 up)
{
    float3 camera_up = normalize(CLOUD.camera.xyz);
    float angle = acos(clamp(dot(camera_up, up), -1.0, 1.0));
    float s = sin(angle);
    if (s < 1e-6)
    {
        return up;
    }
    float t = CLOUD.camera.w / CLOUD.horizon.x;
    return (sin((1.0 - t) * angle) * camera_up + sin(t * angle) * up) / s;
}

// The thickness of the layer holding radius `r`, or of the thinnest layer
// when none does.
float cloud_layer_thickness(float r)
{
    float thinnest = 1e9;
    uint count = uint(CLOUD.shell.z);
    [unroll]
    for (uint i = 0u; i < CLOUD_MAX_LAYERS; ++i)
    {
        if (i < count)
        {
            float2 shell = CLOUD.layers[i].shell.xy;
            float thickness = shell.y - shell.x;
            if (r >= shell.x && r <= shell.y)
            {
                return thickness;
            }
            thinnest = min(thinnest, thickness);
        }
    }
    return thinnest;
}

// A coarse march from a sample toward a light, in steps growing from a small
// share of its layer's thickness: how far it has come and its next step.
struct CloudLightMarch
{
    float t;
    float step_km;
};

CloudLightMarch cloud_light_march(float3 p)
{
    CloudLightMarch m;
    m.t = 0.0;
    m.step_km = cloud_layer_thickness(length(p)) * 0.06;
    return m;
}

// The optical depth from `p` along unit `light` over march `m`'s next
// `steps` steps, its first two (`first` the march's count so far) resolving
// the share `share` of the detail. The weather barely changes over the
// march, so it keeps the sample's `w`.
float cloud_light_depth(CloudWeather w[CLOUD_MAX_LAYERS], float3 p, float3 rel, float3 light,
                        float share, uint first, uint steps, inout CloudLightMarch m)
{
    float depth = 0.0;
    for (uint i = 0u; i < 8u; ++i)
    {
        if (i >= steps)
        {
            break;
        }
        float dt = m.step_km;
        m.t += dt * 0.5;
        float h;
        float3 q = p + light * m.t;
        depth += cloud_density(w, length(q), rel + light * m.t, first + i < 2u, share, h) * dt;
        m.t += dt * 0.5;
        m.step_km *= 1.7;
    }
    return depth;
}

// The light a sample `p` in layer `layer` scatters toward the viewer, per
// unit extinction: the bodies' light through the cloud toward it, scattered
// once and on, and the sky's and ground's light from all round. `height` is
// its fraction through its layer. The cloud is marched toward the brighter
// body only, and the other's light takes the same path's depth: the two are
// seldom both bright. Toward the shadowed body the shadow map adds what lies
// past the march, its own layer's and the layers above, and the cloud behind
// the sample, which sets how its column holds the scattered light. A pixel
// resolves the share `share` of the detail there.
float3 cloud_sample_light(CloudPhase phase, CloudWeather w[CLOUD_MAX_LAYERS], uint layer, float3 p,
                          float3 rel, float height, float3 ground_albedo, float share)
{
    float r = length(p);
    float3 up = cloud_sky_up(p / r);
    float above = (cloud_authored_radius(r) - CLOUD.camera.w) * CLOUD.horizon.y;
    float3 sky = (float3)0.0;
    float3 lit_ground = (float3)0.0;
    float3 reach[DAYLIGHT_BODIES];
    float brightest = -1.0;
    uint shadowing = 0u;
    [unroll]
    for (uint i = 0u; i < DAYLIGHT_BODIES; i++)
    {
        reach[i] = (float3)0.0;
        if (DAYLIGHT.body_dir[i].w <= 0.5)
        {
            continue;
        }
        float3 l = DAYLIGHT.body_dir[i].xyz;
        float3 light = DAYLIGHT.body_light[i].rgb;
        Daylight d = daylight_at(cloud_light_cosine(dot(up, l), above, CLOUD.horizon.x));
        sky += light * d.irradiance;
        Daylight below = daylight_at(dot(up, l));
        lit_ground += light * (below.transmittance * saturate(dot(up, l)) + below.irradiance);
        reach[i] = light * d.transmittance;
        float bright = dot(reach[i], (float3)(1.0 / 3.0));
        if (bright > brightest)
        {
            brightest = bright;
            shadowing = i;
        }
    }
    float3 direct = (float3)0.0;
    if (brightest > 0.0)
    {
        float3 l = DAYLIGHT.body_dir[shadowing].xyz;
        float2 uv;
        float along;
        float trust = 0.0;
        bool mapped = shadowing == 0u && cloud_shadow_point(p, r, l, uv, along, trust);
        // Where the map is trusted it stands in for the march's last, longest
        // steps, blended in by its trust.
        uint steps = uint(CLOUD.march.y);
        uint near = steps > CLOUD_MAPPED_STEPS ? steps - CLOUD_MAPPED_STEPS : steps;
        CloudLightMarch m = cloud_light_march(p);
        float depth = cloud_light_depth(w, p, rel, l, share, 0u, near, m);
        float reach_km = m.t;
        float last = trust < 1.0
            ? cloud_light_depth(w, p, rel, l, share, near, steps - near, m)
            : 0.0;
        float behind = 0.0;
        float upper = 0.0;
        if (mapped)
        {
            // The map carries on toward the light from where the march
            // stopped, and holds the cloud behind the sample down to its
            // layer's bottom.
            float4 own = cloud_shadow_layer(layer, uv);
            float height_km = r - CLOUD.layers[layer].shell.x;
            depth += lerp(last, cloud_beer_depth(own, along + reach_km), trust);
            behind = trust * cloud_beer_behind(own, along, height_km, dot(p, l) / r);
            upper = layer == 0u ? trust * own.w : 0.0;
        }
        else
        {
            depth += last;
        }
        float lit = cloud_sunlit_share(upper);
        float first = exp(-depth);
        float diffuse = cloud_diffuse_share(depth + behind, depth);
        // Seen with a light behind the viewer, a cloud's thin lit skin
        // scatters less back than its deeper folds (the powder effect), which
        // draws its billows.
        float skin = 1.0 - CLOUD_POWDER * exp(-2.0 * depth);
        [unroll]
        for (uint i = 0u; i < DAYLIGHT_BODIES; i++)
        {
            float powder = lerp(skin, 1.0, phase.toward[i]);
            direct += reach[i] * lit * (phase.first[i] * first * powder + diffuse);
        }
    }
    // The sky lights the tops, the ground the bases; both reach a cloud's
    // inside only dimly.
    float3 ambient = sky / CLOUD_PI * lerp(0.35, 1.0, height)
                   + lit_ground * ground_albedo / CLOUD_PI * lerp(0.6, 0.15, height);
    return direct + ambient;
}
