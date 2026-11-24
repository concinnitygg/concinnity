// The air in front of a surface: the transmittance and multiple-scattering
// table reads, the in-scatter of one segment of air, and the haze between the
// camera and a point, from the aerial volume or, from high above the air,
// marched along the pixel's own ray. Spliced into the atmosphere's kernels at
// its AIR_HAZE marker and into the transparent producers, which define
// AIR_HAZE; not a standalone program. Nothing here may spell the marker
// itself.
//
// Needs `ATMO` (the AtmosphereParams block), ATMOSPHERE_COMMON, and as each
// gate's code reads them `transmittance_lut`, `multiscatter_lut`,
// `aerial_volume` and the clamping `lut_sampler`.


#if defined(ATMOSPHERE_MULTISCATTER) || defined(ATMOSPHERE_SKY_VIEW) || defined(ATMOSPHERE_AERIAL) \
    || defined(ATMOSPHERE_IRRADIANCE) || defined(ATMOSPHERE_SKY_CUBE) || defined(AERIAL_PERSPECTIVE) \
    || defined(CLOUD_KERNEL) || defined(AIR_HAZE)
// Light surviving the path from radius `r` with up-cosine `mu` to the top of
// the air; none when the path runs into the ground.
float3 atmo_transmittance(float r, float mu)
{
    if (ray_hits_ground(r, mu))
    {
        return (float3)0.0;
    }
    return transmittance_lut.SampleLevel(lut_sampler, transmittance_uv(r, mu), 0.0).rgb;
}
#endif

#if defined(ATMOSPHERE_SKY_VIEW) || defined(ATMOSPHERE_AERIAL) || defined(AERIAL_PERSPECTIVE) \
    || defined(CLOUD_KERNEL) || defined(AIR_HAZE)
// Light scattered more than once, per unit sunlight and scattering, at radius
// `r` with the sun at up-cosine `mu_s`.
float3 atmo_multiscatter(float r, float mu_s)
{
    float2 uv = float2(
        unit_to_texels(saturate(mu_s * 0.5 + 0.5), float(MULTISCATTER_SIZE)),
        unit_to_texels(saturate((r - atmo_bottom()) / (atmo_top() - atmo_bottom())),
                       float(MULTISCATTER_SIZE)));
    return multiscatter_lut.SampleLevel(lut_sampler, uv, 0.0).rgb;
}

// The light one sunlit sample at `pos` (km from the center) scatters toward
// the viewer along `dir`, per unit sunlight: single scattering with the
// planet's shadow, plus every higher order through the multiple-scattering
// table.
float3 atmo_inscatter(float3 pos, float3 dir, float3 sun, Medium m)
{
    float r = length(pos);
    float mu_s = dot(pos / r, sun);
    float3 to_sun = atmo_transmittance(r, mu_s);
    float nu = dot(dir, sun);
    float3 phased = m.rayleigh * rayleigh_phase(nu) + m.mie * mie_phase(nu, ATMO.mie_extinction.w);
    return phased * to_sun + (m.rayleigh + m.mie) * atmo_multiscatter(r, mu_s);
}

// Add one segment of `dt` km at `pos` to a march: the in-scatter integrated
// exactly over the segment's own extinction, then that extinction applied.
void atmo_step(float3 pos, float3 dir, float3 sun, float dt,
               inout float3 radiance, inout float3 throughput)
{
    Medium m = atmo_medium(length(pos) - atmo_bottom());
    float3 s = atmo_inscatter(pos, dir, sun, m);
    float3 step_t = exp(-m.extinction * dt);
    radiance += throughput * (s - s * step_t) / max(m.extinction, (float3)1e-7);
    throughput *= step_t;
}
#endif

#if defined(AERIAL_PERSPECTIVE) || defined(CLOUD_KERNEL) || defined(AIR_HAZE)
// Camera heights, in the air's thicknesses above the ground, over which the
// haze hands over from the volume to each pixel's own ray.
static const float2 AERIAL_OWN_RAY_HEIGHTS = float2(0.5, 2.0);
static const uint AERIAL_MARCH_STEPS = 32u;

// The share of the haze each pixel's own ray carries at this camera height.
float aerial_own_ray_share()
{
    float thickness = atmo_top() - atmo_bottom();
    float height = ATMO.camera.w - atmo_bottom();
    return smoothstep(AERIAL_OWN_RAY_HEIGHTS.x * thickness, AERIAL_OWN_RAY_HEIGHTS.y * thickness,
                      height);
}

// `(in-scatter, 1 - T)` along `dir` from where it enters the air to
// `distance` km past that, marched as the volume's columns are.
float4 aerial_march(float3 dir, float enter, float distance)
{
    float3 origin = atmo_camera_up() * atmo_camera_radius() + dir * enter;
    float3 sun = atmo_table_sun();
    float3 radiance = (float3)0.0;
    float3 throughput = (float3)1.0;
    float dt = distance / float(AERIAL_MARCH_STEPS);
    for (uint i = 0u; i < AERIAL_MARCH_STEPS; ++i)
    {
        atmo_step(origin + dir * ((float(i) + 0.5) * dt), dir, sun, dt, radiance, throughput);
    }
    float t = dot(throughput, (float3)(1.0 / 3.0));
    return float4(radiance * ATMO.sun_illuminance.xyz * ATMO.cam_pos.w, 1.0 - t);
}

// `(in-scatter, 1 - T)` of the air between the camera and the point
// `distance_km` along `dir`, seen at screen `uv`; nothing for a ray that
// never enters the air.
float4 aerial_haze(float2 uv, float3 dir, float distance_km)
{
    float2 span = atmo_aerial_span(dir);
    if (span.x < 0.0)
    {
        return (float4)0.0;
    }
    float distance = max(distance_km - span.x, 0.0);
    float own = aerial_own_ray_share();
    float4 haze = (float4)0.0;
    if (own < 1.0)
    {
        // Slices are centered, so a surface nearer than the first slice's
        // center fades in from no haze at the camera.
        float slice = aerial_slice_of(distance, span.y) * float(AERIAL_SIZE);
        float fade = saturate(slice * 2.0);
        float w = max(slice, 0.5) / float(AERIAL_SIZE);
        // The volume already holds its in-scattered light pre-exposed. Across
        // the screen it is read through the B-spline: a froxel column is wide,
        // and a linear read draws its grid into the haze.
        haze = atmo_bspline_slice(aerial_volume, lut_sampler, uv, w, float(AERIAL_SIZE)) * fade
             * (1.0 - own);
    }
    if (own > 0.0)
    {
        haze += aerial_march(dir, span.x, distance) * own;
    }
    return haze;
}
#endif
