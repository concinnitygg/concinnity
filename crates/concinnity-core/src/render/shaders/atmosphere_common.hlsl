// The atmosphere's shared math: the medium, the table parameterizations, the
// phase functions and the spherical harmonics, spliced into the table kernels,
// the haze pass and the atmosphere's sky after `atmosphere_types.hlsl` and the
// program's bindings. Reads the constant block through `ATMO`. The
// parameterizations, the transmittance integral and the spherical harmonics
// are mirrored line for line in `render::atmosphere`.

float atmo_bottom() { return ATMO.radii.x; }
float atmo_top() { return ATMO.radii.y; }

// The camera's distance from the center, kept inside the air: from above the
// top the tables are read as if from just under it.
float atmo_camera_radius()
{
    return clamp(ATMO.camera.w, atmo_bottom() + 1e-3, atmo_top() - 1e-3);
}

float3 atmo_camera_up()
{
    return ATMO.camera.xyz / max(ATMO.camera.w, 1e-3);
}

// Scattering, extinction and the Mie share of scattering at a height above
// the ground. `medium_extinction` mirrors `AtmosphereSettings::extinction_at`.
struct Medium
{
    float3 rayleigh;
    float3 mie;
    float3 extinction;
};

Medium atmo_medium(float height)
{
    float h = max(height, 0.0);
    float rayleigh_density = exp(h * ATMO.rayleigh.w);
    float mie_density = exp(h * ATMO.mie_scattering.w);
    float ozone_density = max(1.0 - abs(height - ATMO.ozone.w) / ATMO.ground_albedo.w, 0.0);
    Medium m;
    m.rayleigh = ATMO.rayleigh.xyz * rayleigh_density;
    m.mie = ATMO.mie_scattering.xyz * mie_density;
    m.extinction = m.rayleigh + ATMO.mie_extinction.xyz * mie_density
                 + ATMO.ozone.xyz * ozone_density;
    return m;
}

float3 medium_extinction(float height)
{
    return atmo_medium(height).extinction;
}

// Distance along a ray from radius `r` with up-cosine `mu` to where it leaves
// the sphere of `radius`. `ray_exit` in render::atmosphere.
float ray_exit(float r, float mu, float radius)
{
    float disc = r * r * (mu * mu - 1.0) + radius * radius;
    return max(-r * mu + sqrt(max(disc, 0.0)), 0.0);
}

// Distance to the ground along the ray, or -1 when it misses.
float ray_ground(float r, float mu)
{
    float b = atmo_bottom();
    float disc = r * r * (mu * mu - 1.0) + b * b;
    if (mu >= 0.0 || disc < 0.0)
    {
        return -1.0;
    }
    return max(-r * mu - sqrt(disc), 0.0);
}

bool ray_hits_ground(float r, float mu)
{
    float b = atmo_bottom();
    return mu < 0.0 && r * r * (mu * mu - 1.0) + b * b >= 0.0;
}

// `unit_to_texels` / `texels_to_unit` in render::atmosphere.
float unit_to_texels(float unit, float size)
{
    return unit * (size - 1.0) / size + 0.5 / size;
}

float texels_to_unit(float uv, float size)
{
    return (uv - 0.5 / size) * size / (size - 1.0);
}

float2 transmittance_uv(float r, float mu)
{
    float b = atmo_bottom();
    float t = atmo_top();
    float h = sqrt(max(t * t - b * b, 0.0));
    float rho = sqrt(max(r * r - b * b, 0.0));
    float disc = r * r * (mu * mu - 1.0) + t * t;
    float d = max(-r * mu + sqrt(max(disc, 0.0)), 0.0);
    float d_min = t - r;
    float d_max = rho + h;
    float x_mu = (d - d_min) / max(d_max - d_min, 1e-6);
    float x_r = rho / max(h, 1e-6);
    return float2(unit_to_texels(x_mu, float(TRANSMITTANCE_WIDTH)),
                  unit_to_texels(x_r, float(TRANSMITTANCE_HEIGHT)));
}

void transmittance_r_mu(float2 uv, out float r, out float mu)
{
    float b = atmo_bottom();
    float t = atmo_top();
    float x_mu = texels_to_unit(uv.x, float(TRANSMITTANCE_WIDTH));
    float x_r = texels_to_unit(uv.y, float(TRANSMITTANCE_HEIGHT));
    float h = sqrt(max(t * t - b * b, 0.0));
    float rho = h * x_r;
    r = sqrt(rho * rho + b * b);
    float d_min = t - r;
    float d_max = rho + h;
    float d = d_min + x_mu * (d_max - d_min);
    mu = d == 0.0 ? 1.0 : clamp((h * h - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
}

// `horizon_angles` in render::atmosphere: the zenith-to-horizon angle and the
// horizon's depth below the level.
float2 atmo_horizon_angles(float r)
{
    float b = atmo_bottom();
    float v_horizon = sqrt(max(r * r - b * b, 0.0));
    float beta = acos(clamp(v_horizon / r, -1.0, 1.0));
    return float2(ATMO_PI - beta, beta);
}

float2 sky_view_uv(float r, float view_cos, float azimuth_cos)
{
    float2 angles = atmo_horizon_angles(r);
    float view_angle = acos(clamp(view_cos, -1.0, 1.0));
    float v;
    if (view_angle < angles.x)
    {
        v = 0.5 * (1.0 - sqrt(max(1.0 - view_angle / angles.x, 0.0)));
    }
    else
    {
        v = 0.5 + 0.5 * sqrt(max((view_angle - angles.x) / max(angles.y, 1e-6), 0.0));
    }
    float azimuth = acos(clamp(azimuth_cos, -1.0, 1.0)) / ATMO_PI;
    return float2(unit_to_texels(sqrt(max(azimuth, 0.0)), float(SKY_VIEW_WIDTH)),
                  unit_to_texels(v, float(SKY_VIEW_HEIGHT)));
}

// The view and azimuth cosines a sky-view coordinate holds.
float2 sky_view_direction(float r, float2 uv)
{
    float2 angles = atmo_horizon_angles(r);
    float u = texels_to_unit(uv.x, float(SKY_VIEW_WIDTH));
    float v = texels_to_unit(uv.y, float(SKY_VIEW_HEIGHT));
    float view_angle;
    if (v < 0.5)
    {
        float c = 1.0 - 2.0 * v;
        view_angle = angles.x * (1.0 - c * c);
    }
    else
    {
        float c = 2.0 * v - 1.0;
        view_angle = angles.x + angles.y * c * c;
    }
    return float2(cos(view_angle), cos(u * u * ATMO_PI));
}

// The sky-view coordinate of world direction `dir` seen from the camera:
// its cosine to the camera's up, and the cosine of its azimuth from the sun's
// about that up.
float2 sky_view_uv_of(float3 dir)
{
    float3 up = atmo_camera_up();
    float view_cos = dot(dir, up);
    float3 sun = ATMO.sun_direction.xyz;
    float3 view_flat = dir - up * view_cos;
    float3 sun_flat = sun - up * dot(sun, up);
    float lv = dot(view_flat, view_flat);
    float ls = dot(sun_flat, sun_flat);
    float azimuth_cos = (lv > 1e-10 && ls > 1e-10)
        ? dot(view_flat, sun_flat) * rsqrt(lv * ls)
        : 1.0;
    return sky_view_uv(atmo_camera_radius(), view_cos, azimuth_cos);
}

// `aerial_slice_of` in render::atmosphere: the volume's depth coordinate of a
// point `distance` km away.
float aerial_slice_of(float distance)
{
    return min(sqrt(max(distance / AERIAL_MAX_DISTANCE_KM, 0.0)), 1.0);
}

float aerial_slice_distance(float slice)
{
    float w = (slice + 0.5) / float(AERIAL_SIZE);
    return AERIAL_MAX_DISTANCE_KM * w * w;
}

float rayleigh_phase(float nu)
{
    return 3.0 / (16.0 * ATMO_PI) * (1.0 + nu * nu);
}

// Cornette-Shanks.
float mie_phase(float nu, float g)
{
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * nu;
    return 3.0 / (8.0 * ATMO_PI) * ((1.0 - g2) * (1.0 + nu * nu))
         / ((2.0 + g2) * pow(max(denom, 1e-4), 1.5));
}

// `sh9_basis` in render::atmosphere.
void sh9_basis(float3 d, out float basis[9])
{
    basis[0] = 0.282095;
    basis[1] = 0.488603 * d.y;
    basis[2] = 0.488603 * d.z;
    basis[3] = 0.488603 * d.x;
    basis[4] = 1.092548 * d.x * d.y;
    basis[5] = 1.092548 * d.y * d.z;
    basis[6] = 0.315392 * (3.0 * d.z * d.z - 1.0);
    basis[7] = 1.092548 * d.x * d.z;
    basis[8] = 0.546274 * (d.x * d.x - d.y * d.y);
}

// `fibonacci_direction` in render::atmosphere.
float3 fibonacci_direction(uint i, uint n)
{
    float z = 1.0 - (2.0 * float(i) + 1.0) / float(n);
    float ring = sqrt(max(1.0 - z * z, 0.0));
    float phi = 2.399963 * float(i);
    return float3(ring * cos(phi), ring * sin(phi), z);
}
