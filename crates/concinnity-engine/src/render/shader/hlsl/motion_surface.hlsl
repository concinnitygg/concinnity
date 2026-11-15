// The surface a G-buffer pixel shows, as the motion snapshot keeps it for the
// next frame: its view normal folded onto the octahedron and its motion
// vector. A temporal pass compares a pixel with last frame's snapshot to tell
// whether its history shows the same surface. Depth cannot tell them apart
// where a moving object meets the ground; the uncovered ground faces and moves
// unlike the object that hid it, while a surface that only turns, or a camera
// that only starts moving, changes one and keeps the other. Spliced into a
// shader at its MOTION_SURFACE marker; not a standalone program.

// A snapshot texel with no surface: its motion lies past every motion vector
// the G-buffer stores (they clamp to 2), so nothing matches it.
static const float MOTION_SURFACE_NONE = 4.0;

// Cosine between last frame's normal and this pixel's below which they are
// another surface, and past which they are surely the same.
static const float2 MOTION_SURFACE_NORMAL = float2(0.8, 0.9);

// Motion difference, in pixels, within which two pixels are surely one
// surface: a pixel's worth, plus a share of the motion itself for a camera
// that accelerates. Past twice that they are another surface.
static const float MOTION_SURFACE_SLACK = 0.5;
static const float MOTION_SURFACE_SHARE = 0.25;

// A unit vector folded onto the octahedron, in [-1, 1]^2.
float2 motion_surface_encode(float3 n)
{
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    if (n.z < 0.0)
    {
        float2 fold = (1.0 - abs(n.yx)) * select(n.xy >= 0.0, (float2)(1.0), (float2)(-1.0));
        return fold;
    }
    return n.xy;
}

float3 motion_surface_decode(float2 e)
{
    float3 n = float3(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0)
    {
        n.xy = (1.0 - abs(n.yx)) * select(n.xy >= 0.0, (float2)(1.0), (float2)(-1.0));
    }
    return normalize(n);
}

// How surely last frame's snapshot texel `last` showed the surface facing
// `normal` and moving `motion` pixels: 1 for the same surface, 0 for another,
// and between the two near the tolerances. `size` is the target's size in
// pixels, which scales the stored UV motion.
float motion_surface_match(float4 last, float3 normal, float2 motion, float2 size)
{
    if (last.z >= MOTION_SURFACE_NONE)
    {
        return 0.0;
    }
    float2 last_motion = last.zw * size;
    float slack = MOTION_SURFACE_SLACK
        + MOTION_SURFACE_SHARE * max(length(motion), length(last_motion));
    float by_motion = 1.0 - smoothstep(slack, 2.0 * slack, length(motion - last_motion));
    float by_normal = smoothstep(
        MOTION_SURFACE_NORMAL.x,
        MOTION_SURFACE_NORMAL.y,
        dot(motion_surface_decode(last.xy), normal));
    return max(by_motion, by_normal);
}
