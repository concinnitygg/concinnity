// The camera block every pass drawing the scene from one viewpoint binds: the
// forward main pass, the sky, and the probe and mirror re-renders of both.
// Layout must match `ViewUniforms` in render::uniforms.

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
};

// A world direction in the environment cubemaps' own frame. Every sky tap goes
// through this, so the sky, the ambient fill and the reflections turn together.
// Defined as a macro because VIEW is a per-backend binding named below.
#define SKY_DIR(d) float3(dot(VIEW.sky_rot[0].xyz, (d)), \
                          dot(VIEW.sky_rot[1].xyz, (d)), \
                          dot(VIEW.sky_rot[2].xyz, (d)))
