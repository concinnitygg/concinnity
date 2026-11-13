// A reflection ray that left the scene toward a sky the trace leaves to a
// pass of its own: the ray-traced reflections append one per such ray, and
// the sky's pass shades the list, adding each ray's sky into the reflection
// output. Layout matches `ReflectionMissGpu` in render_types (32 B).

#ifndef REFLECTION_MISS
#define REFLECTION_MISS

struct ReflectionMiss
{
    // The output texel, packed (y << 16) | x.
    uint pixel;
    // The ray's unit direction, in the frame the scene is drawn in.
    float3 dir;
    // What the sky along the ray lands in the texel's color by: the ray's
    // owed share, its Fresnel tint and the frame's pre-exposure together.
    float3 weight;
    // How fast the ray's footprint widens, in radians.
    float spread;
};

// Threads per group of the pass that shades the list, which the list's
// indirect arguments count groups of.
static const uint REFLECTION_MISS_GROUP = 64u;
// Groups an indirect dispatch may launch along x on every API; a longer list
// is strided over.
static const uint REFLECTION_MISS_MAX_GROUPS = 65535u;

#endif
