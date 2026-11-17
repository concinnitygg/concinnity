// The camera block every pass drawing the scene from one viewpoint binds: the
// forward main pass, the sky, and the probe and mirror re-renders of both.
// Layout must match `ViewUniforms` in render::uniforms.

{HDR_STORE}

{DAYLIGHT_TYPES}

{TILE_MORPH_TYPES}

struct ViewUniforms
{
    float4x4 vp;
    float4x4 view_mat;
    float elapsed;
    // 1.0 for a probe or mirror face: the sky leaves out the sun's and the
    // moon's discs, which those lights' specular terms draw.
    float reflection;
    float cam_x; float cam_y; float cam_z;
    float prefilter_mip_count;
    // 1.0 while the unlit view mode is active: the surface returns its base
    // color before lighting.
    float shade_mode;
    // How much of the screen-space occlusion the indirect term takes, 0..1:
    // 1.0 in full, between while it fades, 0.0 without it and for a probe or
    // mirror face, which renders another viewpoint.
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
    // How much of the bounce of nearby surfaces the screen-space indirect
    // target carries the indirect term adds, 0..1: 1.0 in full, between while
    // it fades, 0.0 when the target holds occlusion alone.
    float screen_bounce;
    // What the view's ground is shaded as in place of what the view does not
    // draw: 0 nothing; 1 the grass blades, by the canopy; 2 those and the
    // planet's sea, by its surface over the ground under it.
    float stand_ins;
    // The planet's per-point sky, off for a world without one.
    DaylightBlock daylight;
    // The camera a planet's tiles morph by, the main camera's in every view.
    TileMorphView tile_morph;
    // The share of each pixel's sharp specular an SSR / RT reflection
    // composite owns this frame (`reflection_share`): none keeps the
    // glossy-dielectric forward probe specular, and a probe or mirror face
    // owns none.
    float4 reflection_rect;
    float reflection_whole;
    float reflection_prop;
    float2 reflection_ramp;
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
