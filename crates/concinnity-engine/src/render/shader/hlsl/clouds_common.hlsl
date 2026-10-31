// The clouds' density and lighting, for the march kernel. The includer
// defines CLOUD as its `CloudParams` block and DAYLIGHT as that block's
// daylight, splices `daylight.hlsl`, and binds the noise volume as
// `cloud_noise` with the repeating `cloud_noise_sampler`. Every position is
// kilometers from the planet's center in its fixed axes. The shaping
// functions are mirrored in render::environment::clouds::shape.

static const float CLOUD_PI = 3.14159265358979;
// How far the base shape erodes into the detail at full strength.
static const float CLOUD_DETAIL_EROSION = 0.45;
// How much the powder effect darkens a cloud's sunward skin.
static const float CLOUD_POWDER = 0.6;
// The phase lobes: forward (the silver lining), backward (the glow away from
// the sun), and the forward lobe's share.
static const float CLOUD_FORWARD_G = 0.6;
static const float CLOUD_BACKWARD_G = -0.3;
static const float CLOUD_FORWARD_SHARE = 0.5;

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
float cloud_weather_cover(float weather, float coverage)
{
    float threshold = 1.0 - coverage;
    float cover = smoothstep(threshold - 0.2, threshold + 0.2, weather);
    return cover * saturate(coverage * 8.0);
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

// `shell_span` in render::environment::clouds: the stretch of the ray inside
// the shell between `lower` and `upper` before it reaches `ground` or the
// shell's bottom. False when the ray never enters the shell.
bool cloud_shell_span(float3 origin, float3 dir, float ground, float lower, float upper,
                      out float2 span)
{
    span = (float2)0.0;
    float r = length(origin);
    float2 up_hits;
    if (!cloud_sphere_hits(origin, dir, upper, up_hits))
    {
        return false;
    }
    float2 low_hits;
    bool low = cloud_sphere_hits(origin, dir, lower, low_hits);
    float2 ground_hits;
    bool hits_ground = cloud_sphere_hits(origin, dir, ground, ground_hits) && ground_hits.x > 0.0;
    bool down = low && low_hits.x > 0.0;
    float start;
    float end;
    if (r < lower)
    {
        if (hits_ground)
        {
            return false;
        }
        start = low ? low_hits.y : 0.0;
        end = up_hits.y;
    }
    else if (r <= upper)
    {
        start = 0.0;
        end = down ? low_hits.x : up_hits.y;
    }
    else
    {
        if (up_hits.x < 0.0)
        {
            return false;
        }
        start = up_hits.x;
        end = down ? low_hits.x : up_hits.y;
    }
    span = float2(max(start, 0.0), end);
    return end > start;
}

float4 cloud_noise_at(float3 coord)
{
    return cloud_noise.SampleLevel(cloud_noise_sampler, coord, 0.0);
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
    return p + L.stretch.xyz * (dot(p, L.stretch.xyz) * L.stretch.w);
}

// A layer's weather over one point of its bottom: how much of the cell is
// cloud, and the weather field there. Zero cover is clear sky.
struct CloudWeather
{
    float cover;
    float field;
};

// The weather of layer `i` over the point `up` (unit, from the planet's
// center); the weather reads the layer's bottom there, so a cloud's cell does
// not lean with height.
CloudWeather cloud_layer_weather(uint i, CloudRay ray, float3 up)
{
    CloudLayerBlock L = CLOUD.layers[i];
    float3 below = cloud_stretched(L, (up - ray.camera_up) * L.shell.x);
    float4 w1 = cloud_noise_at(L.weather.xyz + below * L.scales.y);
    float w2 = cloud_noise_at(L.weather_second.xyz + below * L.weather.w).b;
    CloudWeather w;
    w.field = saturate(lerp(w1.b, w2, 0.35) * 0.75 + w1.a * 0.4 - 0.1);
    w.cover = cloud_weather_cover(w.field, L.shell.z);
    return w;
}

// Extinction per kilometer of layer `i` at radius `r`, `rel` from the camera,
// under weather `w`; `detail` wears the edges with the detail noise.
float cloud_layer_density(uint i, CloudWeather w, float r, float3 rel, bool detail)
{
    CloudLayerBlock L = CLOUD.layers[i];
    float height = (r - L.shell.x) / (L.shell.y - L.shell.x);
    if (w.cover <= 0.0 || height < 0.0 || height > 1.0)
    {
        return 0.0;
    }
    float profile = cloud_height_profile(height, L.shell.w, w.field);
    if (profile <= 0.0)
    {
        return 0.0;
    }
    rel = cloud_stretched(L, rel);
    float shape = cloud_noise_at(L.shape.xyz + rel * L.scales.z).r;
    float d = cloud_coverage_density(shape, profile, w.cover);
    if (detail && d > 0.0)
    {
        // Wispy toward the base, billowed toward the top.
        float n = cloud_noise_at(L.detail.xyz + rel * L.scales.w).g;
        float wear = lerp(1.0 - n, n, saturate(height * 3.0)) * CLOUD_DETAIL_EROSION;
        d = cloud_remap(d, wear, 1.0);
    }
    return d * L.scales.x;
}

// The layers' weather over the point `up`.
void cloud_weathers(CloudRay ray, float3 up, out CloudWeather w[CLOUD_MAX_LAYERS])
{
    uint count = uint(CLOUD.shell.z);
    [unroll]
    for (uint i = 0u; i < CLOUD_MAX_LAYERS; ++i)
    {
        w[i].cover = 0.0;
        w[i].field = 0.0;
        if (i < count)
        {
            w[i] = cloud_layer_weather(i, ray, up);
        }
    }
}

// The layers' weather over the point `up`, where layer `known`'s is already
// `weather`.
void cloud_weathers_with(CloudRay ray, float3 up, uint known, CloudWeather weather,
                         out CloudWeather w[CLOUD_MAX_LAYERS])
{
    uint count = uint(CLOUD.shell.z);
    [unroll]
    for (uint i = 0u; i < CLOUD_MAX_LAYERS; ++i)
    {
        w[i].cover = 0.0;
        w[i].field = 0.0;
        if (i == known)
        {
            w[i] = weather;
        }
        else if (i < count)
        {
            w[i] = cloud_layer_weather(i, ray, up);
        }
    }
}

// Extinction per kilometer of every layer at radius `r`, `rel` from the
// camera, under the layers' weather `w`, and the height fraction through the
// layer holding the most of it.
float cloud_density(CloudWeather w[CLOUD_MAX_LAYERS], float r, float3 rel, bool detail,
                    out float height)
{
    float total = 0.0;
    height = 0.0;
    [unroll]
    for (uint i = 0u; i < CLOUD_MAX_LAYERS; ++i)
    {
        float d = cloud_layer_density(i, w[i], r, rel, detail);
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

// The two-lobed phase, its lobes narrowed by `k`.
float cloud_phase(float nu, float k)
{
    return lerp(cloud_hg(nu, CLOUD_BACKWARD_G * k), cloud_hg(nu, CLOUD_FORWARD_G * k),
                CLOUD_FORWARD_SHARE);
}

// How one view ray scatters each body's light toward the viewer, constant
// along the ray. Light scatters on through a cloud in four octaves, each
// fainter, reaching farther and spreading wider (Wrenninge's approximation);
// `octaves` holds each octave's phase weighted by its share.
struct CloudPhase
{
    float4 octaves[DAYLIGHT_BODIES];
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
        phase.octaves[i] = float4(cloud_phase(nu, 1.0), 0.5 * cloud_phase(nu, 0.5),
                                  0.25 * cloud_phase(nu, 0.25), 0.125 * cloud_phase(nu, 0.125));
        phase.toward[i] = saturate(nu * 0.5 + 0.5);
    }
    return phase;
}

// The share of a light reaching a point `optical_depth` deep toward it that
// scatters toward the viewer under `octaves`, every octave of scattering on
// through the cloud included. Octave `o` reaches `2^o` times as deep, so its
// extinction is the last octave's raised to the `2^(3 - o)`.
float cloud_scattered(float optical_depth, float4 octaves)
{
    float e3 = exp(-optical_depth * 0.125);
    float e2 = e3 * e3;
    float e1 = e2 * e2;
    return dot(octaves, float4(e1 * e1, e1, e2, e3));
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

// The optical depth from `p` toward unit `light`, marched coarsely. The
// weather barely changes over the march, so it keeps the sample's `w`.
float cloud_light_depth(CloudWeather w[CLOUD_MAX_LAYERS], float3 p, float3 rel, float3 light)
{
    uint steps = uint(CLOUD.march.y);
    float step_km = cloud_layer_thickness(length(p)) * 0.06;
    float t = 0.0;
    float depth = 0.0;
    for (uint i = 0u; i < 8u; ++i)
    {
        if (i >= steps)
        {
            break;
        }
        float dt = step_km;
        t += dt * 0.5;
        float h;
        float3 q = p + light * t;
        depth += cloud_density(w, length(q), rel + light * t, i < 2u, h) * dt;
        t += dt * 0.5;
        step_km *= 1.7;
    }
    return depth;
}

// The light a sample `p` scatters toward the viewer, per unit extinction: the
// bodies' light through the cloud toward it, and the sky's and ground's
// light from all round. `height` is its fraction through its layer. The
// cloud is marched toward the brighter body only, and the other's light
// takes the same path's depth: the two are seldom both bright.
float3 cloud_sample_light(CloudPhase phase, CloudWeather w[CLOUD_MAX_LAYERS], float3 p, float3 rel,
                          float height, float3 ground_albedo)
{
    float r = length(p);
    float3 up = p / r;
    float above = r - CLOUD.camera.w;
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
        float depth = cloud_light_depth(w, p, rel, DAYLIGHT.body_dir[shadowing].xyz);
        // Seen with a light behind the viewer, a cloud's thin lit skin
        // scatters less back than its deeper folds (the powder effect), which
        // draws its billows.
        float skin = 1.0 - CLOUD_POWDER * exp(-2.0 * depth);
        [unroll]
        for (uint i = 0u; i < DAYLIGHT_BODIES; i++)
        {
            float powder = lerp(skin, 1.0, phase.toward[i]);
            direct += reach[i] * cloud_scattered(depth, phase.octaves[i]) * powder;
        }
    }
    // The sky lights the tops, the ground the bases; both reach a cloud's
    // inside only dimly.
    float3 ambient = sky / CLOUD_PI * lerp(0.35, 1.0, height)
                   + lit_ground * ground_albedo / CLOUD_PI * lerp(0.6, 0.15, height);
    return direct + ambient;
}
