// The atmosphere's tables and its haze: single source for every backend.
//
// One entry per compile, picked by its gate, so each declares only what it
// binds. The five kernels share one binding layout: the constant block, the
// three tables they may read (transmittance, multiple scattering, sky view)
// with one linear clamp sampler, and the one table each writes. In order:
//
//   ATMOSPHERE_TRANSMITTANCE  light surviving a path to the top of the air
//   ATMOSPHERE_MULTISCATTER   all orders of scattering past the first, folded
//                             into one factor per height and sun angle
//   ATMOSPHERE_SKY_VIEW       the sky's radiance around the camera
//   ATMOSPHERE_AERIAL         the air's in-scatter and transmittance along each
//                             view ray, slice by slice in depth
//   ATMOSPHERE_IRRADIANCE     the sky projected onto spherical harmonics and
//                             written out as the irradiance cube
//   ATMOSPHERE_SKY_CUBE       the sky drawn into the faces of a cube, without
//                             the sun's disc, for the reflection prefilter
//
// Over a planet, the last two see the ground around the camera below the
// horizon (`surface_seen`) instead of the sky-view table's gray albedo.
//
// The first two depend on the air and the tables' ground alone and run when
// either changes, the next three every frame, and the sky cube when its
// reflections are refreshed. AERIAL_PERSPECTIVE is the fullscreen haze pass:
// it reads the scene depth and the aerial volume and blends
// `(in-scatter, 1 - T)` over the scene with the premultiplied `over` blend
// fog uses.

//
// Every resource takes its Metal index from its register number. The written
// table is the exception: D3D gives a UAV its own `u` space where Metal folds
// it into the texture namespace beside the three tables, so it takes a
// `CN_BACKEND_DIRECTX` branch, as in fog.hlsl.

{DEPTH_CONVENTION}
{ATMOSPHERE_TYPES}

#ifdef AERIAL_PERSPECTIVE

[[vk::binding(0, 0)]] ConstantBuffer<AtmosphereParams> atmo_cb : register(b0);
#if USE_MSAA
[[vk::binding(1, 0)]] Texture2DMS<float> scene_depth : register(t0);
#else
[[vk::binding(1, 0)]] Texture2D<float> scene_depth : register(t0);
#endif
[[vk::binding(2, 0)]] Texture3D<float4> aerial_volume : register(t1);
[[vk::binding(3, 0)]] SamplerState lut_sampler : register(s0);

#else

[[vk::binding(0, 0)]] ConstantBuffer<AtmosphereParams> atmo_cb : register(b0);
#if defined(ATMOSPHERE_MULTISCATTER) || defined(ATMOSPHERE_SKY_VIEW) || defined(ATMOSPHERE_AERIAL) \
    || defined(ATMOSPHERE_IRRADIANCE) || defined(ATMOSPHERE_SKY_CUBE)
[[vk::binding(1, 0)]] Texture2D<float4> transmittance_lut : register(t0);
#endif
#if defined(ATMOSPHERE_SKY_VIEW) || defined(ATMOSPHERE_AERIAL)
[[vk::binding(2, 0)]] Texture2D<float4> multiscatter_lut : register(t1);
#endif
#if defined(ATMOSPHERE_IRRADIANCE) || defined(ATMOSPHERE_SKY_CUBE)
[[vk::binding(3, 0)]] Texture2D<float4> sky_view_lut : register(t2);
#endif
[[vk::binding(4, 0)]] SamplerState lut_sampler : register(s0);

#if defined(ATMOSPHERE_AERIAL)
#define ATMO_OUTPUT RWTexture3D<float4>
#elif defined(ATMOSPHERE_IRRADIANCE) || defined(ATMOSPHERE_SKY_CUBE)
#define ATMO_OUTPUT RWTexture2DArray<float4>
#else
#define ATMO_OUTPUT RWTexture2D<float4>
#endif
// The irradiance cube is full float, every table half float.
#ifdef ATMOSPHERE_IRRADIANCE
#define ATMO_OUTPUT_FORMAT "rgba32f"
#else
#define ATMO_OUTPUT_FORMAT "rgba16f"
#endif
#ifdef CN_BACKEND_DIRECTX
[[vk::binding(5, 0)]] [[vk::image_format(ATMO_OUTPUT_FORMAT)]] ATMO_OUTPUT atmo_out : register(u0);
#else
[[vk::binding(5, 0)]] [[vk::image_format(ATMO_OUTPUT_FORMAT)]] ATMO_OUTPUT atmo_out : register(u3);
#endif

#endif

#define ATMO atmo_cb
{ATMOSPHERE_COMMON}
{NIGHT_SKY}

#if defined(ATMOSPHERE_MULTISCATTER) || defined(ATMOSPHERE_SKY_VIEW) || defined(ATMOSPHERE_AERIAL) \
    || defined(ATMOSPHERE_IRRADIANCE) || defined(ATMOSPHERE_SKY_CUBE)
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

#if defined(ATMOSPHERE_SKY_VIEW) || defined(ATMOSPHERE_AERIAL)
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

#ifdef ATMOSPHERE_TRANSMITTANCE
[shader("compute")]
[numthreads(8, 8, 1)]
void atmosphere_transmittance_kernel(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= TRANSMITTANCE_WIDTH || id.y >= TRANSMITTANCE_HEIGHT)
    {
        return;
    }
    float2 uv = (float2(id.xy) + 0.5) / float2(TRANSMITTANCE_WIDTH, TRANSMITTANCE_HEIGHT);
    float r;
    float mu;
    transmittance_r_mu(uv, r, mu);
    // `optical_depth` in render::atmosphere.
    float dt = ray_exit(r, mu, atmo_top()) / float(TRANSMITTANCE_STEPS);
    float3 depth = (float3)0.0;
    for (uint i = 0u; i < TRANSMITTANCE_STEPS; ++i)
    {
        float t = (float(i) + 0.5) * dt;
        float rt = sqrt(t * t + 2.0 * r * mu * t + r * r);
        depth += medium_extinction(rt - atmo_bottom()) * dt;
    }
    atmo_out[id.xy] = float4(exp(-depth), 1.0);
}
#endif

#ifdef ATMOSPHERE_MULTISCATTER
// One group per texel, one thread per direction, the directions summed in
// shared memory.
static const uint MULTISCATTER_DIRECTIONS = 64u;
groupshared float3 ms_radiance[MULTISCATTER_DIRECTIONS];
groupshared float3 ms_transfer[MULTISCATTER_DIRECTIONS];

[shader("compute")]
[numthreads(64, 1, 1)]
void atmosphere_multiscatter_kernel(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    float2 uv = (float2(group.xy) + 0.5) / float(MULTISCATTER_SIZE);
    float mu_s = texels_to_unit(uv.x, float(MULTISCATTER_SIZE)) * 2.0 - 1.0;
    float r = lerp(atmo_bottom() + 1e-3, atmo_top() - 1e-3,
                   texels_to_unit(uv.y, float(MULTISCATTER_SIZE)));
    float3 sun = float3(sqrt(saturate(1.0 - mu_s * mu_s)), mu_s, 0.0);
    float3 origin = float3(0.0, r, 0.0);
    float3 fib = fibonacci_direction(lane, MULTISCATTER_DIRECTIONS);
    float3 dir = float3(fib.x, fib.z, fib.y);

    float ground = ray_ground(r, dir.y);
    float length_km = ground >= 0.0 ? ground : ray_exit(r, dir.y, atmo_top());
    float dt = length_km / float(MULTISCATTER_STEPS);
    float3 radiance = (float3)0.0;
    float3 transfer = (float3)0.0;
    float3 throughput = (float3)1.0;
    for (uint i = 0u; i < MULTISCATTER_STEPS; ++i)
    {
        float3 pos = origin + dir * ((float(i) + 0.5) * dt);
        float rp = length(pos);
        Medium m = atmo_medium(rp - atmo_bottom());
        float3 scattering = m.rayleigh + m.mie;
        float3 to_sun = atmo_transmittance(rp, dot(pos / rp, sun));
        float3 s = scattering * to_sun / (4.0 * ATMO_PI);
        float3 step_t = exp(-m.extinction * dt);
        float3 inv_ext = 1.0 / max(m.extinction, (float3)1e-7);
        radiance += throughput * (s - s * step_t) * inv_ext;
        transfer += throughput * (scattering - scattering * step_t) * inv_ext;
        throughput *= step_t;
    }
    if (ground >= 0.0)
    {
        float3 hit = origin + dir * ground;
        float3 n = normalize(hit);
        float mu_g = dot(n, sun);
        radiance += throughput * atmo_transmittance(atmo_bottom(), mu_g) * saturate(mu_g)
                  * ATMO.ground_albedo.xyz / ATMO_PI;
    }
    ms_radiance[lane] = radiance;
    ms_transfer[lane] = transfer;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = MULTISCATTER_DIRECTIONS / 2u; stride > 0u; stride >>= 1u)
    {
        if (lane < stride)
        {
            ms_radiance[lane] += ms_radiance[lane + stride];
            ms_transfer[lane] += ms_transfer[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u)
    {
        // Averages over the sphere: the light scattered once more, and the
        // fraction of it scattered again, whose geometric series sums every
        // further order.
        float3 second = ms_radiance[0] / float(MULTISCATTER_DIRECTIONS);
        float3 again = ms_transfer[0] / float(MULTISCATTER_DIRECTIONS);
        atmo_out[group.xy] = float4(second / max(1.0 - again, (float3)1e-3), 1.0);
    }
}
#endif

#ifdef ATMOSPHERE_SKY_VIEW
[shader("compute")]
[numthreads(8, 8, 1)]
void atmosphere_sky_view_kernel(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= SKY_VIEW_WIDTH || id.y >= SKY_VIEW_HEIGHT)
    {
        return;
    }
    float r = atmo_camera_radius();
    float2 uv = (float2(id.xy) + 0.5) / float2(SKY_VIEW_WIDTH, SKY_VIEW_HEIGHT);
    float2 view = sky_view_direction(r, uv);
    float sin_view = sin(view.x);
    float sin_az = sqrt(saturate(1.0 - view.y * view.y));
    // The table's own frame: the camera on +Y, the sun in the XY plane.
    float3 dir = float3(sin_view * view.y, cos(view.x), sin_view * sin_az);
    float mu_s = dot(atmo_camera_up(), atmo_table_sun());
    float3 sun = float3(sqrt(saturate(1.0 - mu_s * mu_s)), mu_s, 0.0);
    // From above the air the march starts where the ray enters it; a ray
    // that misses it sees only what lies beyond.
    float enter = ray_enter(r, dir.y, atmo_top());
    if (enter < 0.0)
    {
        atmo_out[id.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    float3 origin = float3(0.0, r, 0.0) + dir * enter;
    float r0 = length(origin);
    float mu0 = dot(origin, dir) / r0;

    float ground = ray_ground(r0, mu0);
    float length_km = ground >= 0.0 ? ground : ray_exit(r0, mu0, atmo_top());
    float dt = length_km / float(SKY_VIEW_STEPS);
    float3 radiance = (float3)0.0;
    float3 throughput = (float3)1.0;
    for (uint i = 0u; i < SKY_VIEW_STEPS; ++i)
    {
        atmo_step(origin + dir * ((float(i) + 0.5) * dt), dir, sun, dt, radiance, throughput);
    }
    // A ray into the ground ends on it: the sunlit ground seen through the
    // air, which the sky shows below the horizon where nothing was drawn.
    if (ground >= 0.0)
    {
        float3 n = normalize(origin + dir * ground);
        float mu_g = dot(n, sun);
        radiance += throughput * atmo_transmittance(atmo_bottom(), mu_g) * saturate(mu_g)
                  * ATMO.ground_albedo.xyz / ATMO_PI;
    }
    // Alpha: the light from beyond the air that survives the ray, for the
    // night sky drawn behind it; none past the ground. The radiance is stored
    // at the frame's pre-exposure, which keeps a twilight sky in half float's
    // normal range.
    float beyond = ground >= 0.0 ? 0.0 : dot(throughput, (float3)(1.0 / 3.0));
    atmo_out[id.xy] = float4(radiance * ATMO.sun_illuminance.xyz * ATMO.cam_pos.w, beyond);
}
#endif

#ifdef ATMOSPHERE_AERIAL
// One thread per froxel column; it marches the column's view ray slice by
// slice, each slice storing what lies between the ray's entry into the air
// (the camera, inside it) and its depth, out to the ray's reach.
[shader("compute")]
[numthreads(8, 8, 1)]
void atmosphere_aerial_kernel(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= AERIAL_SIZE || id.y >= AERIAL_SIZE)
    {
        return;
    }
    float2 uv = (float2(id.xy) + 0.5) / float(AERIAL_SIZE);
    float2 ndc = float2(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));
    float3 dir = normalize(camera_view_ray(ATMO.inv_vp, ndc, ATMO.cam_pos.xyz));
    float2 span = atmo_aerial_span(dir);
    if (span.x < 0.0)
    {
        for (uint z = 0u; z < AERIAL_SIZE; ++z)
        {
            atmo_out[uint3(id.xy, z)] = (float4)0.0;
        }
        return;
    }
    float3 origin = atmo_camera_up() * atmo_camera_radius() + dir * span.x;
    float3 sun = atmo_table_sun();
    float3 radiance = (float3)0.0;
    float3 throughput = (float3)1.0;
    float near = 0.0;
    for (uint z = 0u; z < AERIAL_SIZE; ++z)
    {
        float far = aerial_slice_distance(float(z), span.y);
        float dt = far - near;
        atmo_step(origin + dir * (near + 0.5 * dt), dir, sun, dt, radiance, throughput);
        near = far;
        float t = dot(throughput, (float3)(1.0 / 3.0));
        atmo_out[uint3(id.xy, z)] =
            float4(radiance * ATMO.sun_illuminance.xyz * ATMO.cam_pos.w, 1.0 - t);
    }
}
#endif

#if defined(ATMOSPHERE_IRRADIANCE) || defined(ATMOSPHERE_SKY_CUBE)
// The direction a cube texel at (u, v) in [-1, 1] faces. `probe_face_dir` in
// probe_prefilter.hlsl, the hardware cube convention.
float3 cube_face_dir(uint face, float u, float v)
{
    switch (face)
    {
    case 0u:
        return float3(1.0, -v, -u);
    case 1u:
        return float3(-1.0, -v, u);
    case 2u:
        return float3(u, 1.0, v);
    case 3u:
        return float3(u, -1.0, -v);
    case 4u:
        return float3(u, -v, 1.0);
    default:
        return float3(-u, -v, -1.0);
    }
}

// The world direction cube texel `xy` of a `size` square `face` stands for:
// the cube is sampled through the sky rotation's rows, so its texel faces
// the direction those rows take it to.
float3 cube_texel_world_dir(uint face, uint2 xy, uint size)
{
    float2 st = (float2(xy) + 0.5) / float(size) * 2.0 - 1.0;
    float3 c = normalize(cube_face_dir(face, st.x, st.y));
    return ATMO.sky_rot[0].xyz * c.x + ATMO.sky_rot[1].xyz * c.y + ATMO.sky_rot[2].xyz * c.z;
}
#endif

#if defined(ATMOSPHERE_IRRADIANCE) || defined(ATMOSPHERE_SKY_CUBE)
// The sky's light on the ground around the camera, the same for every
// direction a dispatch draws: one lane of each group reads it for the rest.
groupshared float3 surface_sky_shared;

// Whether the block carries the ground around the camera.
bool surface_on()
{
    return ATMO.surface_land.w > 0.5;
}

// The table's band coordinate, in [0, SURFACE_BANDS - 1], of a ray
// `depression` radians below the level: `band_depression_deg` in
// render::atmosphere inverted.
float surface_band_of(float depression)
{
    float deg = max(degrees(depression), 1e-4);
    float t = log(deg / SURFACE_STEEPEST_DEG) / log(SURFACE_SHALLOWEST_DEG / SURFACE_STEEPEST_DEG);
    return saturate(t) * float(SURFACE_BANDS - 1u);
}

float surface_share_at(uint band, uint azimuth)
{
    uint i = band * SURFACE_AZIMUTHS + azimuth % SURFACE_AZIMUTHS;
    return ATMO.surface_share[i >> 2u][i & 3u];
}

// How much of the ground seen along `dir`, `depression` radians below the
// level about `up`, is water: the table read between its directions and
// bands.
float surface_water_share(float3 dir, float3 up, float depression)
{
    float3 east = ATMO.surface_east.xyz;
    float3 north = cross(up, east);
    float turn = atan2(dot(dir, north), dot(dir, east)) / (2.0 * ATMO_PI);
    float a = frac(turn) * float(SURFACE_AZIMUTHS);
    uint a0 = uint(a);
    float fa = a - float(a0);
    float b = surface_band_of(depression);
    uint b0 = min(uint(b), SURFACE_BANDS - 2u);
    float fb = b - float(b0);
    float near = lerp(surface_share_at(b0, a0), surface_share_at(b0, a0 + 1u), fa);
    float far = lerp(surface_share_at(b0 + 1u, a0), surface_share_at(b0 + 1u, a0 + 1u), fa);
    return lerp(near, far, fb);
}

// The sky's light on level ground under `up`, absolute: the sky-view table
// straight up and halfway down toward the four sides, cosine weighted.
float3 surface_sky_irradiance(float3 up)
{
    float3 east = ATMO.surface_east.xyz;
    float3 north = cross(up, east);
    float3 sum = atmo_sky_view(sky_view_lut, lut_sampler, up).rgb;
    const float side = 0.70710678;
    sum += side * atmo_sky_view(sky_view_lut, lut_sampler, normalize(up + east)).rgb;
    sum += side * atmo_sky_view(sky_view_lut, lut_sampler, normalize(up - east)).rgb;
    sum += side * atmo_sky_view(sky_view_lut, lut_sampler, normalize(up + north)).rgb;
    sum += side * atmo_sky_view(sky_view_lut, lut_sampler, normalize(up - north)).rgb;
    return ATMO_PI * sum / ((1.0 + 4.0 * side) * ATMO.cam_pos.w);
}

// `surface_sky_irradiance` around the camera, read by lane 0 of the group
// and shared; zero without the ground. Every lane of the group calls it.
float3 surface_sky_light(uint lane)
{
    if (lane == 0u)
    {
        surface_sky_shared = surface_on() ? surface_sky_irradiance(atmo_camera_up()) : (float3)0.0;
    }
    GroupMemoryBarrierWithGroupSync();
    return surface_sky_shared;
}

// The light reaching the camera along `dir` from the ground around it,
// absolute. The sky-view table already holds that ray's air and, past it,
// the atmosphere's own ground; that ground, seen through the air, is taken
// back out and the ground around the camera put in its place, sea or land by
// the table. The ray is taken in the tables' space, its angle below the
// horizon stretched onto theirs as the table's is. The sea mirrors the sky
// about level ground by its Fresnel term over its lit body; the land is lit
// by the sun through the air and by `sky_light` (`surface_sky_light`).
float3 surface_seen(float3 dir, float3 sky_light)
{
    float3 up = atmo_camera_up();
    float mu = dot(dir, up);
    float table_mu = cos(atmo_table_view_angle(acos(clamp(mu, -1.0, 1.0))));
    float3 flat = dir - up * mu;
    float flat_len = length(flat);
    flat = flat_len > 1e-6 ? flat / flat_len : ATMO.surface_east.xyz;
    float3 table_dir = up * table_mu + flat * sqrt(saturate(1.0 - table_mu * table_mu));

    // Where the ray meets the tables' ground; a ray the true ground meets can
    // pass over theirs near their horizon, and then ends where it is nearest
    // their ground, with no ground of theirs in the table to take out.
    float r = atmo_camera_radius();
    float ground = ray_ground(r, table_mu);
    bool table_ground = ground >= 0.0;
    if (!table_ground)
    {
        ground = max(-r * table_mu, 0.0);
    }
    float3 hit = up * r + table_dir * ground;
    float hit_r = max(length(hit), atmo_bottom());
    float3 n = hit / hit_r;
    // The air between the camera and the ground: the light surviving from the
    // ground to the top of the air, over that from the camera, the path read
    // backward so both run upward. From above the air only the first holds.
    float3 to_top = atmo_transmittance(hit_r, dot(n, -table_dir));
    float3 from_camera = r < atmo_top() ? atmo_transmittance(r, -table_mu) : (float3)1.0;
    float3 air = saturate(to_top / max(from_camera, (float3)1e-4));

    float3 sun = atmo_table_sun();
    float mu_g = dot(n, sun);
    float3 sun_light = ATMO.sun_illuminance.xyz;
    float3 sunlit = atmo_transmittance(atmo_bottom(), mu_g) * saturate(mu_g) * sun_light;
    float3 table_view = atmo_sky_view(sky_view_lut, lut_sampler, dir).rgb / ATMO.cam_pos.w;
    float3 authored_ground = table_ground ? sunlit * ATMO.ground_albedo.xyz / ATMO_PI : (float3)0.0;
    float3 in_air = max(table_view - air * authored_ground, (float3)0.0);

    float3 lit = sunlit + sky_light;
    float3 land = ATMO.surface_land.rgb * lit / ATMO_PI;
    float fresnel = 0.02 + 0.98 * pow(saturate(1.0 + mu), max(ATMO.surface_water.w, 1e-3));
    float3 mirrored = dir - 2.0 * mu * up;
    float4 sky = atmo_sky_view(sky_view_lut, lut_sampler, mirrored);
    float3 reflected = sky.rgb / ATMO.cam_pos.w;
    if (night_sky_on())
    {
        reflected += night_sky_ambient(mirrored, sky.a);
    }
    float3 sea = fresnel * reflected + (1.0 - fresnel) * ATMO.surface_water.rgb * lit / ATMO_PI;

    float water = surface_water_share(dir, up, asin(saturate(-mu)));
    return in_air + air * lerp(land, sea, water);
}

// Whether `dir` sees the ground around the camera rather than the sky-view
// table's.
bool surface_seen_along(float3 dir)
{
    return surface_on() && atmo_view_hits_ground(dot(dir, atmo_camera_up()));
}
#endif

#ifdef ATMOSPHERE_SKY_CUBE
// The scale the cube is stored at, so a night sky's faint glow keeps its
// half-float steps. Mirrors `render::radiance_storage::SKY_CUBE_STORAGE_SCALE`.
static const float SKY_CUBE_STORAGE_SCALE = 1024.0;

// One thread per texel of the cube's top mip, six faces deep: the sky-view
// table along the texel's direction, the ground below the horizon included,
// and any night sky behind it but its stars. The sun's disc is left out: the
// sun light's own specular term draws it.
[shader("compute")]
[numthreads(8, 8, 1)]
void atmosphere_sky_cube_kernel(uint3 id : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    float3 sky_light = surface_sky_light(lane);
    if (id.x >= SKY_CUBE_FACE || id.y >= SKY_CUBE_FACE)
    {
        return;
    }
    float3 n = cube_texel_world_dir(id.z, id.xy, SKY_CUBE_FACE);
    float3 l;
    if (surface_seen_along(n))
    {
        l = surface_seen(n, sky_light);
    }
    else
    {
        float4 view = atmo_sky_view(sky_view_lut, lut_sampler, n);
        // The cube holds radiance at its storage scale; the table holds it
        // pre-exposed.
        l = view.rgb / ATMO.cam_pos.w;
        if (night_sky_on())
        {
            l += night_sky_reflected(n, 0.5 * ATMO_PI / float(SKY_CUBE_FACE), view.a);
        }
    }
    atmo_out[id] = float4(hdr_store_cap(l * SKY_CUBE_STORAGE_SCALE), 1.0);
}
#endif

#ifdef ATMOSPHERE_IRRADIANCE
// One group: each lane projects its share of the sky's samples, the lanes are
// summed coefficient by coefficient, and then every lane writes its share of
// the cube's texels.
static const uint IRRADIANCE_LANES = 64u;
groupshared float3 sh_lanes[IRRADIANCE_LANES * 9u];
groupshared float3 sh_total[9];


[shader("compute")]
[numthreads(64, 1, 1)]
void atmosphere_irradiance_kernel(uint lane : SV_GroupIndex)
{
    float basis[9];
    float3 acc[9];
    for (uint j = 0u; j < 9u; ++j)
    {
        acc[j] = (float3)0.0;
    }
    uint per_lane = IRRADIANCE_SAMPLES / IRRADIANCE_LANES;
    float weight = 4.0 * ATMO_PI / float(IRRADIANCE_SAMPLES);
    float3 sky_light = surface_sky_light(lane);
    for (uint k = 0u; k < per_lane; ++k)
    {
        float3 d = fibonacci_direction(lane * per_lane + k, IRRADIANCE_SAMPLES);
        float3 l;
        if (surface_seen_along(d))
        {
            l = surface_seen(d, sky_light);
        }
        else
        {
            float4 view = atmo_sky_view(sky_view_lut, lut_sampler, d);
            l = view.rgb / ATMO.cam_pos.w;
            if (night_sky_on())
            {
                l += night_sky_ambient(d, view.a);
            }
        }
        sh9_basis(d, basis);
        for (uint j = 0u; j < 9u; ++j)
        {
            acc[j] += l * (basis[j] * weight);
        }
    }
    for (uint j = 0u; j < 9u; ++j)
    {
        sh_lanes[lane * 9u + j] = acc[j];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane < 9u)
    {
        float3 sum = (float3)0.0;
        for (uint i = 0u; i < IRRADIANCE_LANES; ++i)
        {
            sum += sh_lanes[i * 9u + lane];
        }
        // The cosine lobe, band by band: radiance to irradiance.
        float lobe = lane == 0u ? ATMO_PI : (lane < 4u ? 2.0 * ATMO_PI / 3.0 : ATMO_PI / 4.0);
        sh_total[lane] = sum * lobe;
    }
    GroupMemoryBarrierWithGroupSync();

    uint texels = 6u * IRRADIANCE_FACE * IRRADIANCE_FACE;
    for (uint p = lane; p < texels; p += IRRADIANCE_LANES)
    {
        uint face = p / (IRRADIANCE_FACE * IRRADIANCE_FACE);
        uint rest = p % (IRRADIANCE_FACE * IRRADIANCE_FACE);
        uint2 xy = uint2(rest % IRRADIANCE_FACE, rest / IRRADIANCE_FACE);
        sh9_basis(cube_texel_world_dir(face, xy, IRRADIANCE_FACE), basis);

        float3 e = (float3)0.0;
        for (uint j = 0u; j < 9u; ++j)
        {
            e += sh_total[j] * basis[j];
        }
        atmo_out[uint3(xy, face)] = float4(max(e, (float3)0.0), 1.0);
    }
}
#endif

#ifdef AERIAL_PERSPECTIVE
// The fragment takes the shared fullscreen varying but derives its own `uv`
// from SV_Position, for the reason fog_fragment gives: the volume was filled
// in screen tile order.
[shader("pixel")]
float4 aerial_perspective_fragment(
    [[vk::location(0)]] float2 vertex_uv : TEXCOORD0,
    float4 sv_pos : SV_Position) : SV_Target
{
    int2 pixel = int2(sv_pos.xy);
    if (pixel.x < 0 || pixel.y < 0 ||
        pixel.x >= int(ATMO.viewport.x) || pixel.y >= int(ATMO.viewport.y))
    {
        discard;
    }
#if USE_MSAA
    float depth = scene_depth.Load(pixel, 0);
#else
    float depth = scene_depth.Load(int3(pixel, 0));
#endif
    float2 uv = sv_pos.xy * ATMO.viewport.zw;
    float2 ndc_xy = float2(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));
    float3 world;
    // The sky already holds the air along its rays.
    if (!depth_reconstruct(ATMO.depth_inv_vp, ndc_xy, depth, world))
    {
        discard;
    }
    float3 to_world = world - ATMO.cam_pos.xyz;
    float2 span = atmo_aerial_span(normalize(to_world));
    if (span.x < 0.0)
    {
        discard;
    }
    float distance = max(length(to_world) * ATMO.radii.w - span.x, 0.0);
    // Slices are centered, so a surface nearer than the first slice's center
    // fades in from no haze at the camera.
    float slice = aerial_slice_of(distance, span.y) * float(AERIAL_SIZE);
    float fade = saturate(slice * 2.0);
    float w = max(slice, 0.5) / float(AERIAL_SIZE);
    // The volume already holds its in-scattered light pre-exposed. Across
    // the screen it is read through the B-spline: a froxel column is wide,
    // and a linear read draws its grid into the haze.
    return atmo_bspline_slice(aerial_volume, lut_sampler, uv, w, float(AERIAL_SIZE)) * fade;
}
#endif
