// The view ray under a pixel, for the passes that draw the sky: one triangle
// covering the viewport at the far plane, each corner carrying the
// camera-relative direction of the view ray through it. That direction is
// linear in NDC, so the rasterizer's interpolation hands every pixel its own
// ray exactly. The CPU mirror is `render::sky`.

// Corner `vid` (0-2) of a triangle covering the whole viewport, in NDC.
float2 sky_corner(uint vid)
{
    return float2(vid == 1u ? 3.0 : -1.0, vid == 2u ? 3.0 : -1.0);
}

// The camera-relative direction of the view ray through NDC point `ndc` under
// the view-projection `vp`, scaled so the ray's clip w is 1. Only the x, y and
// w rows take part, so the projection's depth mapping (reversed, infinite, or
// an oblique near plane) never moves it, and neither does the camera position.
float3 sky_ray(float4x4 vp, float2 ndc)
{
    float3 rx = vp[0].xyz;
    float3 ry = vp[1].xyz;
    float3 rw = vp[3].xyz;
    float3 ray = cross(ry, rw) * ndc.x + cross(rw, rx) * ndc.y + cross(rx, ry);
    return ray / dot(rx, cross(ry, rw));
}
