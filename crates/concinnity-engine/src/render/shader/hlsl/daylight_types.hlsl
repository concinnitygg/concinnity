// The light a point on a planet's ground receives, by how high each light
// stands above the point's own horizon: the curves `render::atmosphere`
// bakes from the air, and the lights that ride them. Spliced into the view
// blocks of the passes that shade the ground and the sea; `daylight.hlsl`
// reads it. Layout matches `DaylightBlock` in render::uniforms.

#ifndef DAYLIGHT_TYPES
#define DAYLIGHT_TYPES

static const uint DAYLIGHT_KNOTS = 32u;
// The directional lights a planet's sky lights by: the first is the shadowed
// one, the sun by day and the moon by night, the second the other.
static const uint DAYLIGHT_BODIES = 2u;

struct DaylightBlock
{
    // xyz: the planet's center; w: 1 when the curves light this view.
    float4 center;
    // xyz: up where the camera is; w: how much each point's own sky stands
    // in for the camera's wherever it is, from the camera's height.
    float4 camera_up;
    // x, y: the cosines between a point's up and the camera's at which the
    // point's own sky starts to take over from the camera's, and has. z: the
    // ground's radius; w: the height over it below which a point is lit as
    // the ground, fading out over its upper half.
    float4 reach;
    // xyz: toward the body lighting as directional light i; w: 1 when it is
    // lit by the curves.
    float4 body_dir[DAYLIGHT_BODIES];
    // rgb: that body's light above the air.
    float4 body_light[DAYLIGHT_BODIES];
    // xyz: the planet sea's deep-water albedo; w: its height above the
    // ground's datum. Read only by a view whose ground stands in for the sea.
    float4 sea;
    // Three per knot: the transmittance and the zenith's red, the irradiance
    // and the zenith's green, the horizon and the zenith's blue.
    float4 knots[DAYLIGHT_KNOTS * 3u];
};

#endif
