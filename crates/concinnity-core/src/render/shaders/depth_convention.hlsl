// DEPTH_CONVENTION marker: how device depth orders surfaces, for every shader
// that reads, reduces, compares or writes a hardware depth value. The CPU half
// is `render::depth`; the two must agree.
//
// Camera depth is the main camera's buffer and every target tested against it.
// Shadow depth is the shadow maps. Near is 0 and far is 1 in both.
//
// The helpers are macros, so each expands to the exact expression it names and
// compiles to the same code as writing that expression in place. Guarded,
// because a shader and a fragment it splices may both carry the marker.

#ifndef CN_DEPTH_CONVENTION
#define CN_DEPTH_CONVENTION

// Camera device depth at the near and far planes. `DEPTH_FAR` is also the
// cleared value: a pixel no surface reached.
#define DEPTH_NEAR 0.0
#define DEPTH_FAR 1.0

// The nearer / farther of two camera depths. The identity of a `depth_closer`
// reduction is `DEPTH_FAR`, of a `depth_farther` one `DEPTH_NEAR`.
#define depth_closer(a, b) min((a), (b))
#define depth_farther(a, b) max((a), (b))

// Whether camera depth `a` is strictly in front of / behind `b`.
#define depth_in_front(a, b) ((a) < (b))
#define depth_behind(a, b) ((a) > (b))

// Whether a stored camera depth still holds the clear value, or holds a surface.
#define depth_is_cleared(d) ((d) >= DEPTH_FAR)
#define depth_is_written(d) ((d) < DEPTH_FAR)

// Camera depth `d` moved `offset` toward the far plane.
#define depth_offset_far(d, offset) ((d) + (offset))

// The clip-space z that pins a vertex just inside the far plane, for a clip w
// of `w`. The sky shell uses it so the camera far plane never clips it.
#define depth_pin_far(w) ((w) * (1.0 - 1e-6))

// The homogeneous world position (divide by w) of the surface stored at camera
// depth `d` under NDC `ndc_xy`, through the inverse view-projection `inv_vp`.
// Where `d` is cleared it is the far-plane point on the pixel's ray; a caller
// that needs a no-hit result tests `depth_is_cleared` / `depth_is_written`
// first.
#define depth_unproject(inv_vp, ndc_xy, d) mul((inv_vp), float4((ndc_xy), (d), 1.0))

// Conservative depth output for a pass writing camera depth no farther than
// the rasterized fragment's, which keeps early depth testing.
#define CAMERA_DEPTH_CONSERVATIVE SV_DepthLessEqual

// Shadow depth `d` moved `offset` toward the light: the side a sample's compare
// reference is biased to, so a surface does not shadow itself.
#define shadow_depth_offset_near(d, offset) ((d) - (offset))

// Conservative depth output for a shadow caster writing a depth no farther from
// the light than the rasterized fragment's.
#define SHADOW_DEPTH_CONSERVATIVE SV_DepthLessEqual

#endif
