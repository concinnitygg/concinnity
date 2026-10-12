// The atmosphere's constant block and table sizes, spliced ahead of the
// program's own bindings; `atmosphere_common.hlsl` follows them with the math,
// reading the block through the `ATMO` name each program defines. Lengths are
// kilometers. `render::atmosphere` mirrors the sizes, and its tests lock the
// constants below to the Rust ones.
//
// AtmosphereParams (928 B) mirrors `render::atmosphere::AtmosphereParams`;
// every member is a float4 lane so the three targets lay it out alike.

struct AtmosphereParams
{
    float4x4 inv_vp;
    // xyz = Rayleigh scattering per km, w = -1 / scale height.
    float4 rayleigh;
    // xyz = Mie scattering per km, w = -1 / scale height.
    float4 mie_scattering;
    // xyz = Mie extinction per km, w = phase anisotropy.
    float4 mie_extinction;
    // xyz = ozone absorption per km, w = layer center km.
    float4 ozone;
    // xyz = ground albedo, w = half the ozone layer's width km.
    float4 ground_albedo;
    // The tables' planet: x = ground radius, y = top radius, z = aerial
    // reach (km), w = km per world unit.
    float4 radii;
    // xyz = camera relative to the tables' planet center (km), w = its length.
    float4 camera;
    // The planet the sky is seen over: x = ground radius, y = top radius,
    // z = the camera's distance from its center. The tables' own unless the
    // atmosphere has an apparent radius.
    float4 view_ground;
    // xyz = toward the sun, w = cosine of the disc's radius.
    float4 sun_direction;
    float4 sun_illuminance;
    // xyz = the disc's radiance seen from the camera.
    float4 sun_disc;
    // xyz = the camera's world position, w = the frame's pre-exposure, which the
    // sky-view table and the aerial volume store their radiance at.
    float4 cam_pos;
    // xy = target size in pixels, zw = its reciprocal.
    float4 viewport;
    float4 sky_rot[3];
    // The night sky behind the air (render::night_sky::NightLanes): x = a
    // magnitude-0 star's light, y = the faintest magnitude, z = the Milky
    // Way's radiance at its heart, w = galaxies drawn.
    float4 night;
    // xyz = the Milky Way's pole on the celestial sphere, w = the seed.
    float4 galactic_pole;
    // xyz = the direction of its core, w = stars per steradian.
    float4 galactic_center;
    // xyz = toward the moon, w = cosine of its radius (2 without a moon).
    float4 moon_direction;
    // xyz = the moon's face lit head-on, w = its dark side's share.
    float4 moon_face;
    // xyz = up across the moon's face, w = 1 when the night sky draws.
    float4 moon_up;
    // Two lanes per galaxy, NIGHT_MAX_GALAXIES of them.
    float4 galaxies[32];
};

static const uint NIGHT_MAX_GALAXIES = 16u;

static const uint TRANSMITTANCE_STEPS = 40u;
static const uint TRANSMITTANCE_WIDTH = 256u;
static const uint TRANSMITTANCE_HEIGHT = 64u;
static const uint MULTISCATTER_SIZE = 32u;
static const uint SKY_VIEW_WIDTH = 192u;
static const uint SKY_VIEW_HEIGHT = 108u;
static const uint AERIAL_SIZE = 32u;
static const float AERIAL_MAX_DISTANCE_KM = 32.0;
static const uint IRRADIANCE_FACE = 16u;
static const uint IRRADIANCE_SAMPLES = 512u;
static const uint SKY_CUBE_FACE = 512u;

// Steps along each sky-view and multiple-scattering ray.
static const uint SKY_VIEW_STEPS = 30u;
static const uint MULTISCATTER_STEPS = 20u;
static const float ATMO_PI = 3.14159265358979;

