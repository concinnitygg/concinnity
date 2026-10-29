// Lighting a point on a planet's ground from its own sky rather than the
// camera's. The includer defines DAYLIGHT as the `DaylightBlock` its view
// carries.

// The light a point receives from a light at cosine `mu` to its up, per unit
// of that light above the air.
struct Daylight
{
    float3 transmittance;
    float3 irradiance;
    float3 zenith;
    float3 horizon;
};

bool daylight_on()
{
    return DAYLIGHT.center.w > 0.5;
}

// Where `mu` falls among the knots. `knot_coord` in render::atmosphere.
float daylight_coord(float mu)
{
    mu = clamp(mu, -1.0, 1.0);
    float s = mu < 0.0 ? -sqrt(-mu) : sqrt(mu);
    return (s + 1.0) * 0.5 * float(DAYLIGHT_KNOTS - 1u);
}

// The curves at `mu`, linear between knots. `DaylightCurves::sample` in
// render::atmosphere.
Daylight daylight_at(float mu)
{
    float x = daylight_coord(mu);
    uint i = min(uint(x), DAYLIGHT_KNOTS - 2u);
    float t = x - float(i);
    float4 a = lerp(DAYLIGHT.knots[i * 3u], DAYLIGHT.knots[i * 3u + 3u], t);
    float4 b = lerp(DAYLIGHT.knots[i * 3u + 1u], DAYLIGHT.knots[i * 3u + 4u], t);
    float4 c = lerp(DAYLIGHT.knots[i * 3u + 2u], DAYLIGHT.knots[i * 3u + 5u], t);
    Daylight d;
    d.transmittance = a.xyz;
    d.irradiance = b.xyz;
    d.horizon = c.xyz;
    d.zenith = float3(a.w, b.w, c.w);
    return d;
}

// Up at world position `pos`.
float3 daylight_up(float3 pos)
{
    return normalize(pos - DAYLIGHT.center.xyz);
}

// How much the sky at a point whose up is `up` stands in for the camera's:
// none near the camera on the ground, all of it far round the planet or
// seen from high above it.
float daylight_reach(float3 up)
{
    float c = dot(up, DAYLIGHT.camera_up.xyz);
    float far = 1.0 - smoothstep(DAYLIGHT.reach.y, DAYLIGHT.reach.x, c);
    return max(far, DAYLIGHT.camera_up.w);
}

// How much the point at `pos` is lit as the ground under its own sky: all of
// it near the ground, none high above it, where the light the camera gets
// holds.
float daylight_ground(float3 pos)
{
    float h = length(pos - DAYLIGHT.center.xyz) - DAYLIGHT.reach.z;
    return 1.0 - smoothstep(0.5 * DAYLIGHT.reach.w, DAYLIGHT.reach.w, h);
}

// Whether directional light `i` is a body the curves light.
bool daylight_body(uint i)
{
    return i < DAYLIGHT_BODIES && DAYLIGHT.body_dir[i].w > 0.5;
}

// The light of body `i` reaching a point whose up is `up`.
float3 daylight_body_light(uint i, float3 up)
{
    return DAYLIGHT.body_light[i].rgb * daylight_at(dot(up, DAYLIGHT.body_dir[i].xyz)).transmittance;
}

// The sky's light on a level surface at a point whose up is `up`, from every
// body.
float3 daylight_level_irradiance(float3 up)
{
    float3 e = (float3)(0.0);
    [unroll]
    for (uint i = 0u; i < DAYLIGHT_BODIES; i++)
    {
        if (DAYLIGHT.body_dir[i].w > 0.5)
        {
            e += DAYLIGHT.body_light[i].rgb * daylight_at(dot(up, DAYLIGHT.body_dir[i].xyz)).irradiance;
        }
    }
    return e;
}

// The sky's light on a surface facing `n` at a point whose up is `up`, from
// every body: the level surface's, less as the surface turns away from the
// sky.
float3 daylight_irradiance(float3 up, float3 n)
{
    return daylight_level_irradiance(up) * saturate(0.5 + 0.5 * dot(n, up));
}

// The sky's radiance along `dir` at a point whose up is `up`, from every
// body: the horizon's toward the zenith's as `dir` rises, the horizon's below
// it.
float3 daylight_sky(float3 up, float3 dir)
{
    float rise = sqrt(saturate(dot(dir, up)));
    float3 l = (float3)0.0;
    [unroll]
    for (uint i = 0u; i < DAYLIGHT_BODIES; i++)
    {
        if (DAYLIGHT.body_dir[i].w > 0.5)
        {
            Daylight d = daylight_at(dot(up, DAYLIGHT.body_dir[i].xyz));
            l += DAYLIGHT.body_light[i].rgb * lerp(d.horizon, d.zenith, rise);
        }
    }
    return l;
}

// A point's daylight looked up once for every body, for a shader that lights
// the same point several ways. Past `reach` 0 the rest is unset.
struct DaylightPoint
{
    float3 up;
    float reach;
    // Each body's light reaching the point; zero for one the curves skip.
    float3 light[DAYLIGHT_BODIES];
    // The sky's light on a level surface, from every body.
    float3 irradiance;
    // The sky's radiance at the horizon and the zenith, from every body.
    float3 horizon;
    float3 zenith;
};

DaylightPoint daylight_point(float3 pos)
{
    DaylightPoint p = (DaylightPoint)0;
    if (!daylight_on())
    {
        return p;
    }
    p.up = daylight_up(pos);
    p.reach = daylight_reach(p.up) * daylight_ground(pos);
    if (p.reach <= 0.0)
    {
        return p;
    }
    [unroll]
    for (uint i = 0u; i < DAYLIGHT_BODIES; i++)
    {
        if (DAYLIGHT.body_dir[i].w > 0.5)
        {
            Daylight d = daylight_at(dot(p.up, DAYLIGHT.body_dir[i].xyz));
            float3 light = DAYLIGHT.body_light[i].rgb;
            p.light[i] = light * d.transmittance;
            p.irradiance += light * d.irradiance;
            p.horizon += light * d.horizon;
            p.zenith += light * d.zenith;
        }
    }
    return p;
}

// The light of body `i` at the point, `i` any directional light's index.
float3 daylight_point_light(DaylightPoint p, uint i)
{
    float3 light = (float3)0.0;
    [unroll]
    for (uint b = 0u; b < DAYLIGHT_BODIES; b++)
    {
        if (b == i)
        {
            light = p.light[b];
        }
    }
    return light;
}

// `daylight_irradiance` at the point.
float3 daylight_point_irradiance(DaylightPoint p, float3 n)
{
    return p.irradiance * saturate(0.5 + 0.5 * dot(n, p.up));
}

// `daylight_sky` at the point.
float3 daylight_point_sky(DaylightPoint p, float3 dir)
{
    return lerp(p.horizon, p.zenith, sqrt(saturate(dot(dir, p.up))));
}
