// The scene-light and cascade-shadow records, spliced at a shader's LIGHT_TYPES
// marker. Every pass that lights a surface from the same two uniform buffers
// binds these: the forward main pass through `main_types.hlsl`, the raymarched
// SDF volumes through `raymarch_types.hlsl`. The light-cull kernel splices it
// for the local-light record.
//
// Layouts must match `LightUniforms` and `ShadowUniforms` in
// `render/uniforms/`, and `GpuLight` in render_types.rs. The two trailing scalars of `LightUniforms` are live
// fields, not padding: a pass that binds the buffer and calls them padding is
// reading a stale layout.

struct DirLight   { float4 dir_i; float4 col; };
struct PointLight { float4 pos_r; float4 col_i; };

struct LightUniforms
{
    DirLight   dir[4];
    PointLight pt[8];
    int num_dir;
    int num_pt;
    // Indirect-ambient multiplier (PostProcessConfig.ambient_intensity); 1.0
    // is a no-op.
    float ambient_intensity;
    // Valid entry count in the local-light buffer.
    int num_local_lights;
};

struct ShadowUniforms
{
    float4x4 light_vps[4];
    float4 cascade_splits;
    // Live cascade count (1..4); slots at or beyond it are unrendered.
    uint active_cascades;
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
