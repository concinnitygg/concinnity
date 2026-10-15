// The camera block every pass drawing the scene from one viewpoint binds: the
// forward main pass, the sky, and the probe and mirror re-renders of both.
// Layout must match `ViewUniforms` in render::uniforms.

{HDR_STORE}

struct ViewUniforms
{
    float4x4 vp;
    float4x4 view_mat;
    float elapsed;
    // 1.0 when an SSR / RT reflection composite owns the sharp specular this
    // frame (fade the glossy-dielectric forward probe specular); 0.0 keeps it.
    float reflections_enabled;
    float cam_x; float cam_y; float cam_z;
    float prefilter_mip_count;
    // 1.0 while the unlit view mode is active: the surface returns its base
    // color before lighting.
    float shade_mode;
    // 1.0 when the screen-space occlusion describes this view; 0.0 for a probe
    // or mirror face, which renders another viewpoint.
    float ambient_occlusion;
    // Rows of the rotation from world space into the environment cubemaps'
    // baked frame; identity when the sky does not turn.
    float4 sky_rot[3];
    // 1.0 when the irradiance cube holds the environment's diffuse light (an
    // environment map's or an atmosphere's); 0.0 falls back to a flat ambient.
    float irradiance;
    // The scale the pass's color target is written at: the frame's
    // pre-exposure for the main pass, PROBE_STORAGE_SCALE for a probe face,
    // 1.0 for a mirror face.
    float pre_exposure;
    // 1.0 when the screen-space indirect target carries the bounce of nearby
    // surfaces beside the occlusion; 0.0 when it holds occlusion alone.
    float screen_bounce;
    float _pad_view2;
};

// A world direction in the environment cubemaps' own frame. Every sky tap goes
// through this, so the sky, the ambient fill and the reflections turn together.
// Defined as a macro because VIEW is a per-backend binding named below.
// What a tap of the environment prefilter cube is multiplied by: the inverse
// of the scale it is stored at, carried as a base-2 log in row 0's w.
#define SKY_CUBE_READ exp2(-VIEW.sky_rot[0].w)
#define SKY_DIR(d) float3(dot(VIEW.sky_rot[0].xyz, (d)), \
                          dot(VIEW.sky_rot[1].xyz, (d)), \
                          dot(VIEW.sky_rot[2].xyz, (d)))
