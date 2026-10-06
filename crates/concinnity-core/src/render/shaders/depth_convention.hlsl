// DEPTH_CONVENTION marker: how device depth orders surfaces, for every shader
// that reads, reduces, compares or writes a hardware depth value. The CPU half
// is `render::depth`; the two must agree.
//
// Camera depth is the main camera's buffer and every target tested against it:
// reversed and infinite, near is 1 and depth reaches 0 only at infinity. Shadow
// depth is the shadow maps: near is 0 and far is 1.
//
// The comparison helpers are macros, so each expands to the exact expression it
// names and compiles to the same code as writing that expression in place.
// Guarded, because a shader and a fragment it splices may both carry the
// marker.

#ifndef CN_DEPTH_CONVENTION
#define CN_DEPTH_CONVENTION

// Camera device depth at the near plane and at infinity. `DEPTH_FAR` is also
// the cleared value: a pixel no surface reached.
#define DEPTH_NEAR 1.0
#define DEPTH_FAR 0.0

// The nearer / farther of two camera depths. The identity of a `depth_closer`
// reduction is `DEPTH_FAR`, of a `depth_farther` one `DEPTH_NEAR`.
#define depth_closer(a, b) max((a), (b))
#define depth_farther(a, b) min((a), (b))

// Whether camera depth `a` is strictly in front of / behind `b`.
#define depth_in_front(a, b) ((a) > (b))
#define depth_behind(a, b) ((a) < (b))

// Whether a stored camera depth still holds the clear value, or holds a surface.
#define depth_is_cleared(d) ((d) <= DEPTH_FAR)
#define depth_is_written(d) ((d) > DEPTH_FAR)

// Camera depth `d` moved toward the far plane by about `fraction` of its view
// distance, at any distance: reversed depth falls off as the reciprocal of
// distance, so a relative step in depth is a relative step in distance.
#define depth_offset_far_relative(d, fraction) ((d) * (1.0 - (fraction)))

// Smallest homogeneous w a reconstructed camera-depth point may have. Under the
// infinite projection w is the reciprocal of the view-axis distance, so a
// surface past 1e7 m reconstructs as no hit rather than a point at infinity.
#define DEPTH_RECONSTRUCT_MIN_W 1e-7

// The world position of the surface stored at camera depth `d` under NDC
// `ndc_xy`, through the inverse view-projection `inv_vp`, into `world`. False
// where there is no surface to place: a cleared pixel, or a depth so close to
// the far plane that its point lies at infinity. `world` is finite either way.
bool depth_reconstruct(float4x4 inv_vp, float2 ndc_xy, float d, out float3 world)
{
    float4 h = mul(inv_vp, float4(ndc_xy, d, 1.0));
    world = h.xyz / max(h.w, DEPTH_RECONSTRUCT_MIN_W);
    return depth_is_written(d) && h.w > DEPTH_RECONSTRUCT_MIN_W;
}

// World-space unit direction of the camera ray through NDC `ndc_xy`, for the
// camera at `cam_pos` whose inverse view-projection is `inv_vp`. It reads the
// ray's point at infinity, which the infinite projection puts at `DEPTH_FAR`
// with w = 0: nothing is divided by w, and the camera's position never cancels
// against a nearby point, which would cost the direction its precision far from
// the world origin.
float3 camera_view_ray(float4x4 inv_vp, float2 ndc_xy, float3 cam_pos)
{
    float4 h = mul(inv_vp, float4(ndc_xy, DEPTH_FAR, 1.0));
    return normalize(h.xyz - h.w * cam_pos);
}

// Conservative depth output for a pass writing camera depth no farther than
// the rasterized fragment's, which keeps early depth testing.
#define CAMERA_DEPTH_CONSERVATIVE SV_DepthGreaterEqual

// Shadow depth `d` moved `offset` toward the light: the side a sample's compare
// reference is biased to, so a surface does not shadow itself.
#define shadow_depth_offset_near(d, offset) ((d) - (offset))

// Conservative depth output for a shadow caster writing a depth no farther from
// the light than the rasterized fragment's.
#define SHADOW_DEPTH_CONSERVATIVE SV_DepthLessEqual

#endif
