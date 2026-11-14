// What a ray-traced reflection texel records of its ray for the denoiser: how
// far along it the reflected surface lies (r), and whether the texel's sky was
// left for a later frame (g, 1 when it was).

#ifndef REFLECTION_HIT
#define REFLECTION_HIT

// The distance a texel records for a ray that left the scene: past any reach,
// so the denoiser follows its reflection as a direction.
static const float REFLECTION_HIT_ESCAPED = 60000.0;

#endif
