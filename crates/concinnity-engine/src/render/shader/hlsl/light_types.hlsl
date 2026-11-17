// The scene-light and cascade-shadow records, spliced at a shader's LIGHT_TYPES
// marker. Every pass that lights a surface from the same two uniform buffers
// binds these: the forward main pass through `main_types.hlsl`, the raymarched
// SDF volumes through `raymarch_types.hlsl`. The light-cull kernel splices it
// for the local-light record.
//
// Layouts must match `LightUniforms` and `ShadowUniforms` in
// `render/uniforms/`, and `GpuLight` in render_types.rs. The array lengths are
// locked to gfx::render_types by unit test.

static const uint MAX_DIRECTIONAL_LIGHTS = 4u;
static const uint MAX_POINT_LIGHTS = 8u;
static const uint MAX_SKY_OCCLUDERS = 8u;
static const uint NUM_SHADOW_CASCADES = 4u;

// col.w: the sine of the angular radius of the light's disc, 0 for a point.
struct DirLight   { float4 dir_i; float4 col; };
struct PointLight { float4 pos_r; float4 col_i; };

struct LightUniforms
{
    DirLight   dir[MAX_DIRECTIONAL_LIGHTS];
    PointLight pt[MAX_POINT_LIGHTS];
    int num_dir;
    int num_pt;
    // Indirect-ambient multiplier (IndirectLighting.ambient_intensity); 1.0
    // is a no-op.
    float ambient_intensity;
    // Valid entry count in the local-light buffer.
    int num_local_lights;
    // Spheres that keep part of the sky's light off the surfaces around them:
    // center (xyz) and radius (w), the first num_sky_occluders live.
    float4 sky_occluders[MAX_SKY_OCCLUDERS];
    uint num_sky_occluders;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// The clouds' shadow map over the ground around the camera, in the frame's
// axes and meters (`CloudShadowLanes` in gfx::render_types).
struct CloudShadowLanes
{
    // xyz: the square's center; w: the radius of the sphere it lies on.
    float4 center;
    // xyz: the square's east; w: one over its width.
    float4 east;
    // xyz: the square's north; w: 1 when the clouds cast shadows.
    float4 north;
    // xyz: toward the shadowed light.
    float4 light;
};

// The far shadow past the cascades (`FarShadowLanes` in gfx::render_types).
struct FarShadowLanes
{
    float4x4 light_vp;
    // xyz: the center its coverage is measured from; w: the distance from it
    // where the far shadow is gone, 0 when there is none.
    float4 center;
    // The distance from `center` where it starts fading out.
    float fade_start;
    // The depth compare offset, in the map's depth units.
    float compare_offset;
    // One texel, in map coordinates.
    float texel;
    float _pad;
};

struct ShadowUniforms
{
    float4x4 light_vps[NUM_SHADOW_CASCADES];
    float4 cascade_splits;
    // Live cascade count (1..4); slots at or beyond it are unrendered.
    uint active_cascades;
    // View depth past which the cascades fade out, to none at the last split.
    float fade_start;
    uint _pad0;
    uint _pad1;
    CloudShadowLanes clouds;
    FarShadowLanes far;
};

// Each (vec3, scalar) pair is spelled as one float4: MSL sizes a float3 at 16
// bytes in a structured buffer as well as in a constant buffer, so a literal
// transcription pushes every following field four bytes late on Metal alone.
struct GpuLight
{
    // xyz = world-space position, w = range.
    float4 position_range;
    // xyz = linear RGB, w = intensity.
    float4 color_intensity;
    // xyz = direction, w = the LIGHT_KIND_* discriminant's bits.
    float4 direction_kind;
    float  cos_inner;
    float  cos_outer;
    int    shadow_index;
    // Index into the AreaLightData table for an area light, else -1.
    int    data_index;
};
