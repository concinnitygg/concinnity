// The G-buffer pre-pass's shared vocabulary: its view block, its MRT, and the
// motion encoding every reprojecting reader decodes. Spliced into the surface
// entries in `main_bindless.hlsl` and the sky's in `gbuffer_sky.hlsl`.
//
//   target(0) RGBA16F  view-space normal (xyz) + positive linear view depth (a)
//   target(1) R8       perceptual roughness
//   target(2) RG16F    screen-space motion (prev_uv - cur_uv)
//
// Alpha 0 in target(0) marks "no geometry", the sky included.

// Layout matches `GBufferView` (4 x float4x4 + two float4 rows, 288 B).
struct GbView
{
    float4x4 jittered_vp;
    float4x4 cur_vp;
    float4x4 prev_vp;
    float4x4 view_mat;
    // `VIEW.elapsed` and the camera position as the previous frame saw them,
    // so a vertex hook animated by the clock or facing the camera reprojects
    // to where it was drawn.
    float prev_elapsed;
    float prev_cam_x;
    float prev_cam_y;
    float prev_cam_z;
    // Nonzero when a consumer reads motion this frame; otherwise the previous
    // frame is this one and the surfaces skip reprojecting.
    uint motion;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct GbFragmentOut
{
    float4 nd    : SV_Target0;
    float  rough : SV_Target1;
    float2 vel   : SV_Target2;
};

// Any component past 1 lands every pixel off the image, which every reader
// treats as missing history.
static const float GB_MOTION_LIMIT = 2.0;
static const float GB_MIN_PREV_W = 1e-6;

// Stored so the TAA pass can do `prev_uv = uv + motion`. Image-space UV with
// 0 = top, matching the upright resolve the readers sample. A point at or
// behind the previous camera has no previous pixel, and the clamp keeps a far
// off-screen one finite in the RG16F target.
float2 gb_motion(float4 cur_clip, float4 prev_clip)
{
    if (!(prev_clip.w > GB_MIN_PREV_W))
        return (float2)(GB_MOTION_LIMIT);
    float2 cur_ndc  = cur_clip.xy  / cur_clip.w;
    float2 prev_ndc = prev_clip.xy / prev_clip.w;
    float2 cur_uv  = float2(cur_ndc.x  * 0.5 + 0.5, 0.5 - cur_ndc.y  * 0.5);
    float2 prev_uv = float2(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
    return clamp(prev_uv - cur_uv, -GB_MOTION_LIMIT, GB_MOTION_LIMIT);
}
