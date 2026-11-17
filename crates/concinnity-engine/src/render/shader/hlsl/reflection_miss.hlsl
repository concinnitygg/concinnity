// A reflection ray the trace leaves to a pass of its own: one that left the
// scene toward the sky, or one that met far ground or a level surface past
// the authored reach, behind air and clouds. The ray-traced reflections append
// one per such ray, and the sky's pass shades the list, adding what each ray
// brings into the reflection output. Layout matches `ReflectionMissGpu` in
// render_types (64 B).

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
    // A far hit's light without the sun, absolute.
    float3 rest;
    // How far along the ray a far hit lies.
    float reach;
    // The sun's share of a far hit's light, which the clouds above it dim.
    float3 sunlit;
    // What the ray brings: `REFLECTION_MISS_SKY` or `REFLECTION_MISS_FAR`.
    uint kind;
};

// The ray left the scene toward the sky.
static const uint REFLECTION_MISS_SKY = 0u;
// The ray met far ground or a level surface `reach` along it.
static const uint REFLECTION_MISS_FAR = 1u;

// Threads per group of the pass that shades the list, which the list's
// indirect arguments count groups of.
static const uint REFLECTION_MISS_GROUP = 64u;
// Groups an indirect dispatch may launch along x on every API; a longer list
// is strided over.
static const uint REFLECTION_MISS_MAX_GROUPS = 65535u;

#endif
