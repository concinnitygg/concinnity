// The per-frame view every transparent-pass producer (glass.hlsl,
// glass_mesh.hlsl, water.hlsl) reads and the targets each writes, spliced into
// each at its TRANSPARENT_TYPES marker (see shader_source.rs). Not a standalone
// program.
// Nothing here may spell the marker itself.

// Layout matches `TransparentView` / the TransparentViewBlock UBO (240 B). One
// host block feeds every producer.
struct TransparentView
{
    float4x4 vp;       // world -> clip (jittered when TAA is on)
    float4x4 inv_vp;   // clip -> world
    float4 camera_pos; // xyz: world-space camera
    float2 viewport;   // attachment dimensions in pixels
    float time;        // seconds since startup
    // Mips in the sky prefilter cube; 0 = no EnvironmentMap bound, and each
    // producer's reflection takes its own no-sky fallback.
    float prefilter_mip_count;
    // Rows of the rotation from world space into the sky cube's baked frame;
    // identity when the sky does not turn.
    float4 sky_rot[3];
    // Direction toward the scene's sun (the first directional light) and that
    // light's color times its intensity; both zero when the world declares no
    // directional light, which the water glint reads as no sun.
    float4 sun_dir;
    float4 sun_color;
};

{CLUSTER_TYPES}

// A world direction in the sky cube's own frame; every sky tap goes through it.
#define SKY_DIR(d) float3(dot(view.sky_rot[0].xyz, (d)), \
                          dot(view.sky_rot[1].xyz, (d)), \
                          dot(view.sky_rot[2].xyz, (d)))

{REACTIVE_MASK}

// What a producer's shading fragment writes: its straight-alpha color into the
// scene, and its share of the pixel that does not follow the motion vector under
// it into the reactive mask, which the pipeline max-blends so the most reactive
// layer wins.
struct TransparentOut
{
    float4 color    : SV_Target0;
    float  reactive : SV_Target1;
};

TransparentOut transparent_out(float4 color, float reactive)
{
    TransparentOut o;
    o.color = color;
    o.reactive = reactive_write(reactive);
    return o;
}
