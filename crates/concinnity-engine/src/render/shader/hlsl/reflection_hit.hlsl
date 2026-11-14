// What a ray-traced reflection texel records of its ray for the denoiser: how
// far along it the reflected surface lies (r), and how its look past the
// authored reach went (g): 0 for a ray the reach answered,
// `REFLECTION_HIT_PAST` for one that looked past it this frame, and
// `REFLECTION_HIT_DEFERRED` for one whose look was left for a later frame.

#ifndef REFLECTION_HIT
#define REFLECTION_HIT

// The distance a texel records for a ray that left the scene: past any reach,
// so the denoiser follows its reflection as a direction.
static const float REFLECTION_HIT_ESCAPED = 60000.0;
// The look past the reach was taken this frame.
static const float REFLECTION_HIT_PAST = 0.25;
// The look past the reach was left for a later frame.
static const float REFLECTION_HIT_DEFERRED = 1.0;

#endif
