// The clouds' constant block and sizes, spliced ahead of a program's own
// bindings; `clouds_common.hlsl` reads the block through the `CLOUD` name each
// program defines. Lengths are kilometers. `render::environment::clouds`
// mirrors the sizes, and its tests lock the constants below to the Rust ones.
//
// CloudParams (2080 B) mirrors `render::environment::clouds::CloudParams`;
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
    // xyz: the axis the clouds are drawn out along, fixed axes; w: one over
    // the stretch, less one.
    float4 stretch;
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
    // xy: the scene's size in pixels; zw: the pixel of each 2x2 block this
    // frame marches.
    float4 scene;
    // The shadow map's square on the layers' middle, fixed axes: xyz its
    // center from the planet's center (km), w its width (km).
    float4 shadow_center;
    // xyz: the square's east; w: the middle's radius (km).
    float4 shadow_east;
    // xyz: the square's north.
    float4 shadow_north;
    // x: the radius of the ground the clouds' horizon is measured from (km).
    float4 horizon;
    // The light the clouds are lit by, directions in the fixed axes.
    DaylightBlock daylight;
};

static const float CLOUD_TYPES_PI = 3.14159265358979;

// Where direction `dir` falls in the cloud panorama: longitude about +y
// across, from straight up at the top to straight down at the bottom.
// `panorama_uv` in render::environment::clouds.
float2 cloud_panorama_uv(float3 dir)
{
    float u = atan2(dir.z, dir.x) / (2.0 * CLOUD_TYPES_PI) + 0.5;
    float v = acos(clamp(dir.y, -1.0, 1.0)) / CLOUD_TYPES_PI;
    return float2(u, v);
}

// The direction panorama coordinate `uv` stands for.
float3 cloud_panorama_dir(float2 uv)
{
    float phi = (uv.x - 0.5) * 2.0 * CLOUD_TYPES_PI;
    float theta = uv.y * CLOUD_TYPES_PI;
    float s = sin(theta);
    return float3(s * cos(phi), cos(theta), s * sin(phi));
}

#endif
