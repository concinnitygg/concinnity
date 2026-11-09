// The clouds' constant block, sizes and depth match, spliced ahead of a
// program's own bindings behind DEPTH_CONVENTION; `clouds_common.hlsl` reads
// the block through the `CLOUD` name each program defines. Lengths are
// kilometers. `render::environment::clouds` mirrors the sizes, and its tests
// lock the constants below to the Rust ones.
//
// CloudParams (2208 B) mirrors `render::environment::clouds::CloudParams`;
// every member is a float4 lane so the three targets lay it out alike.

#ifndef CLOUD_TYPES
#define CLOUD_TYPES

{DAYLIGHT_TYPES}

static const uint CLOUD_NOISE_SIZE = 128u;
static const uint CLOUD_MAX_LAYERS = 2u;
static const uint CLOUD_TARGET_DIVISOR = 2u;
static const uint CLOUD_PANORAMA_WIDTH = 256u;
static const uint CLOUD_PANORAMA_HEIGHT = 128u;
static const uint CLOUD_SHADOW_SIZE = 256u;

struct CloudLayerBlock
{
    // x, y: radii of the bottom and the top; z: coverage; w: towering.
    float4 shell;
    // x: extinction per km at full density; y, z, w: noise tiles per km for
    // the weather, the base shape and the detail.
    float4 scales;
    // xyz: the weather's origin in tiles; w: its second field's tiles per km.
    float4 weather;
    // xyz: the weather's second field's origin in tiles.
    float4 weather_second;
    // xyz: the base shape's origin in tiles.
    float4 shape;
    // xyz: the detail's origin in tiles.
    float4 detail;
    // x, z: the axis the clouds are drawn out along, fixed axes; y: their
    // clustering; w: one over the stretch, less one.
    float4 stretch;
    // xyz: the regions' origin in tiles; w: their tiles per km.
    float4 region;
};

struct CloudParams
{
    CloudLayerBlock layers[CLOUD_MAX_LAYERS];
    // Rows of the rotation into the planet's fixed axes.
    float4 to_fixed[3];
    // xyz: the camera from the planet's center, fixed axes; w: ground radius.
    float4 camera;
    // x, y: the radii every layer lies between; z: layer count; w: frame index.
    float4 shell;
    // x: most view steps; y: light steps; z: farthest march (km); w: 1 when
    // the detail wears the edges.
    float4 march;
    // xy: the cloud target's size in texels; zw: its reciprocal.
    float4 target;
    // xy: the scene's size in pixels; zw: the target texel of each 2x2 block
    // of them this frame marches.
    float4 scene;
    // The shadow map's square on the layers' middle, fixed axes: xyz its
    // center from the planet's center (km), w its width (km).
    float4 shadow_center;
    // xyz: the square's east; w: the middle's radius (km).
    float4 shadow_east;
    // xyz: the square's north.
    float4 shadow_north;
    // x: the radius of the ground the clouds' horizon is measured from (km);
    // y: how far over it a kilometer over the true ground stands.
    float4 horizon;
    // A vector from the previous frame's camera in fixed axes (w = 1) to its
    // clip space, which the texels this frame does not march reproject
    // through.
    float4x4 prev_view;
    // xyz: the previous frame's camera, as `camera`; w: 1 when the previous
    // frame's target holds its clouds.
    float4 prev_camera;
    // x: the share of a marched panorama texel this frame's ray replaces; 1
    // marches every texel afresh.
    float4 panorama;
    // The light the clouds are lit by, directions in the fixed axes.
    DaylightBlock daylight;
};

static const float CLOUD_TYPES_PI = 3.14159265358979;

// How much two depths stand for the same surface. Reversed depth falls off as
// the reciprocal of distance, so their ratio is the ratio of the distances;
// the sky only matches the sky.
float cloud_depth_match(float a, float b)
{
    bool sky_a = depth_is_cleared(a);
    bool sky_b = depth_is_cleared(b);
    if (sky_a || sky_b)
    {
        return sky_a && sky_b ? 1.0 : 0.0;
    }
    float ratio = min(a, b) / max(a, b);
    return smoothstep(0.85, 1.0, ratio);
}

// `v` turned by the shortest rotation taking +y to the unit `up`, or back
// from `up` to +y when `back`. The panorama stands on the camera's up so it
// stays on the clouds when the frame's axes turn under the camera.
float3 cloud_turn_up(float3 v, float3 up, bool back)
{
    float c = up.y;
    if (c < -0.9999)
    {
        return v;
    }
    float3 k = back ? float3(-up.z, 0.0, up.x) : float3(up.z, 0.0, -up.x);
    return v * c + cross(k, v) + k * (dot(k, v) / (1.0 + c));
}

// Where direction `dir` falls in the cloud panorama around the camera's unit
// `up`: longitude about it across, from straight up at the top to straight
// down at the bottom.
float2 cloud_panorama_uv(float3 dir, float3 up)
{
    float3 d = cloud_turn_up(dir, up, true);
    float u = atan2(d.z, d.x) / (2.0 * CLOUD_TYPES_PI) + 0.5;
    float v = acos(clamp(d.y, -1.0, 1.0)) / CLOUD_TYPES_PI;
    return float2(u, v);
}

// The direction panorama coordinate `uv` stands for around the camera's unit
// `up`.
float3 cloud_panorama_dir(float2 uv, float3 up)
{
    float phi = (uv.x - 0.5) * 2.0 * CLOUD_TYPES_PI;
    float theta = uv.y * CLOUD_TYPES_PI;
    float s = sin(theta);
    return cloud_turn_up(float3(s * cos(phi), cos(theta), s * sin(phi)), up, false);
}

#endif
