// The cascaded shadow of the first directional light, spliced at a shader's
// SHADOW_CASCADE marker. The forward main pass and the water body both splice
// it, so a surface and the water beside it take the same shadow at one point.
// Not a standalone program. Nothing here may spell the marker itself.
//
// Needs `SHADOW_UNI`, `shadow_map_cmp` and `shadow_map_size()` from the
// including shader, and its DEPTH_CONVENTION. An includer that binds the
// clouds' shadow map defines `CLOUD_SHADOW_TAP(uv)` to read it, and the
// clouds then shade what lies under them.

float hash_rotation(float2 p)
{
    float h = frac(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
    return h * 6.2831853;
}

{SHADOW_BIAS}

// 5x5 hash-rotated PCF of a single cascade. Returns the shadow factor in
// [0, 1] (1.0 fully lit), or 1.0 when the fragment lies outside this
// cascade's light frustum.
float sample_cascade_pcf(int cascade, float3 world_pos, float2 screen_xy)
{
    float4 lc = mul(SHADOW_UNI.light_vps[cascade], float4(world_pos, 1.0));
    float3 ndc = lc.xyz / lc.w;
    float2 uv = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0)
    {
        return 1.0;
    }

    float ref = depth_offset_near(ndc.z, cascade_depth_bias(cascade));

    float angle = hash_rotation(screen_xy);
    float ca = cos(angle);
    float sa = sin(angle);

    float2 tex_size = 1.0 / shadow_map_size();

    float sum = 0.0;
    const int RADIUS = 2;
    const float SAMPLES = float((2 * RADIUS + 1) * (2 * RADIUS + 1));
    for (int dy = -RADIUS; dy <= RADIUS; dy++)
    {
        for (int dx = -RADIUS; dx <= RADIUS; dx++)
        {
            float2 off = float2(float(dx), float(dy));
            float2 rot = float2(off.x * ca - off.y * sa, off.x * sa + off.y * ca);
            float2 sample_uv = uv + rot * tex_size;
            sum += shadow_map_cmp(float3(sample_uv, float(cascade)), ref);
        }
    }
    return sum / SAMPLES;
}

// Cascade-aware PCF with cross-cascade blending. Selects the cascade whose far
// split exceeds the fragment's view-space depth, then blends into the next
// cascade across a band at the far edge of that cascade's depth range: each
// cascade places the shadow edge slightly differently, and a hard switch would
// sweep across the world as the camera moves.
float cascade_shadow(float3 world_pos, float view_depth, float2 screen_xy)
{
    int cascade = 4;
    if      (view_depth < SHADOW_UNI.cascade_splits[0]) cascade = 0;
    else if (view_depth < SHADOW_UNI.cascade_splits[1]) cascade = 1;
    else if (view_depth < SHADOW_UNI.cascade_splits[2]) cascade = 2;
    else if (view_depth < SHADOW_UNI.cascade_splits[3]) cascade = 3;
    if (cascade >= int(SHADOW_UNI.active_cascades))
    {
        return 1.0;
    }

    float shade = sample_cascade_pcf(cascade, world_pos, screen_xy);

    if (cascade + 1 < int(SHADOW_UNI.active_cascades))
    {
        float split_far  = SHADOW_UNI.cascade_splits[cascade];
        float split_near = (cascade == 0) ? 0.0 : SHADOW_UNI.cascade_splits[cascade - 1];
        float band = (split_far - split_near) * 0.15;
        float t = (view_depth - (split_far - band)) / max(band, 1e-4);
        if (t > 0.0)
        {
            float next = sample_cascade_pcf(cascade + 1, world_pos, screen_xy);
            shade = lerp(shade, next, clamp(t, 0.0, 1.0));
        }
    }
    return shade;
}

// The share of the shadowed light the clouds let through to `world_pos`: the
// clouds' shadow map read where the light's path from the point crosses the
// sphere the map lies on. All of it outside the map, which fades out toward
// its edges, and with no clouds.
float cloud_shadow_at(float3 world_pos)
{
#ifdef CLOUD_SHADOW_TAP
    CloudShadowLanes c = SHADOW_UNI.clouds;
    if (c.north.w < 0.5)
    {
        return 1.0;
    }
    // From the square's center: the sphere through it is |p + L t + D| = R,
    // with D the center's offset from the planet's (its normal times R),
    // solved in the stable form so a planet's radius loses no precision.
    float3 n = cross(c.east.xyz, c.north.xyz);
    float3 p = world_pos - c.center.xyz;
    float3 l = c.light.xyz;
    float b = dot(l, p) + c.center.w * dot(l, n);
    float k = dot(p, p) + 2.0 * c.center.w * dot(p, n);
    float disc = b * b - k;
    if (b <= 0.0 || disc < 0.0)
    {
        return 1.0;
    }
    // A point above the sphere has the clouds below it, out of the light's way.
    float t = -k / (b + sqrt(disc));
    if (t < 0.0)
    {
        return 1.0;
    }
    // The map's texels lie on the square's plane and are pushed out along the
    // radius onto the sphere, so the hit is read back through the center.
    float3 q = p + l * t;
    float reach = c.center.w + dot(q, n);
    if (reach <= 0.0)
    {
        return 1.0;
    }
    float2 uv = float2(dot(q, c.east.xyz), dot(q, c.north.xyz)) * (c.east.w * c.center.w / reach);
    float edge = 1.0 - smoothstep(0.35, 0.5, max(abs(uv.x), abs(uv.y)));
    return lerp(1.0, CLOUD_SHADOW_TAP(uv + 0.5), edge);
#else
    return 1.0;
#endif
}

// The cascades' shadow at a surface point, under the clouds'.
float shadow_factor_cascaded(float3 world_pos, float view_depth, float2 screen_xy)
{
    return cascade_shadow(world_pos, view_depth, screen_xy) * cloud_shadow_at(world_pos);
}

