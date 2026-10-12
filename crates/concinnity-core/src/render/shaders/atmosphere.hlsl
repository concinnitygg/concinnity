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
#if defined(ATMOSPHERE_MULTISCATTER) || defined(ATMOSPHERE_SKY_VIEW) || defined(ATMOSPHERE_AERIAL)
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

#if defined(ATMOSPHERE_MULTISCATTER) || defined(ATMOSPHERE_SKY_VIEW) || defined(ATMOSPHERE_AERIAL)
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
    float sin_view = sqrt(saturate(1.0 - view.x * view.x));
    float sin_az = sqrt(saturate(1.0 - view.y * view.y));
    // The table's own frame: the camera on +Y, the sun in the XY plane.
    float3 dir = float3(sin_view * view.y, view.x, sin_view * sin_az);
    float mu_s = dot(atmo_camera_up(), atmo_table_sun());
    float3 sun = float3(sqrt(saturate(1.0 - mu_s * mu_s)), mu_s, 0.0);
    float3 origin = float3(0.0, r, 0.0);

    float ground = ray_ground(r, dir.y);
    float length_km = ground >= 0.0 ? ground : ray_exit(r, dir.y, atmo_top());
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
// slice, each slice storing what lies between the camera and its depth.
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
    float3 origin = atmo_camera_up() * atmo_camera_radius();
    float3 sun = atmo_table_sun();
    float3 radiance = (float3)0.0;
    float3 throughput = (float3)1.0;
    float near = 0.0;
    for (uint z = 0u; z < AERIAL_SIZE; ++z)
    {
        float far = aerial_slice_distance(float(z));
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
void atmosphere_sky_cube_kernel(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= SKY_CUBE_FACE || id.y >= SKY_CUBE_FACE)
    {
        return;
    }
    float3 n = cube_texel_world_dir(id.z, id.xy, SKY_CUBE_FACE);
    float4 view = sky_view_lut.SampleLevel(lut_sampler, sky_view_uv_of(n), 0.0);
    // The cube holds radiance at its storage scale; the table holds it
    // pre-exposed.
    float3 l = view.rgb / ATMO.cam_pos.w;
    if (night_sky_on())
    {
        l += night_sky_reflected(n, 0.5 * ATMO_PI / float(SKY_CUBE_FACE), view.a);
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
    for (uint k = 0u; k < per_lane; ++k)
    {
        float3 d = fibonacci_direction(lane * per_lane + k, IRRADIANCE_SAMPLES);
        float4 view = sky_view_lut.SampleLevel(lut_sampler, sky_view_uv_of(d), 0.0);
        float3 l = view.rgb / ATMO.cam_pos.w;
        if (night_sky_on())
        {
            l += night_sky_ambient(d, view.a);
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
    if (!depth_reconstruct(ATMO.inv_vp, ndc_xy, depth, world))
    {
        discard;
    }
    float distance = length(world - ATMO.cam_pos.xyz) * ATMO.radii.w;
    // Slices are centered, so a surface nearer than the first slice's center
    // fades in from no haze at the camera.
    float slice = aerial_slice_of(distance) * float(AERIAL_SIZE);
    float fade = saturate(slice * 2.0);
    float w = max(slice, 0.5) / float(AERIAL_SIZE);
    // The volume already holds its in-scattered light pre-exposed.
    return aerial_volume.SampleLevel(lut_sampler, float3(uv, w), 0.0) * fade;
}
#endif
