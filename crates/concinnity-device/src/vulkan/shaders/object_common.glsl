// object_common.glsl
//
// The bindless per-object record. The main pass, the G-buffer prepass, the
// shadow pass and the cull kernel all stride the same per-frame SSBO, so this
// declaration is shared rather than repeated: one drifting copy would misindex
// every other consumer at once. Mirrors `GpuObjectData` in
// concinnity-core/src/gfx/render_types.rs (176 bytes) alongside
// directx/shaders/object_common.hlsl and metal/shaders/object_common.msl.
//
// shaderc has no #include resolver, so this file is substituted into the
// consuming shader at its OBJECT_DATA marker (builtins::GlslProgram::source).
//
// std430 aligns vec3 and mat4 to 16 bytes. Every vec3 below sits on a 16-byte
// boundary and is followed by a scalar that fills the trailing gap, so the
// packing matches the Rust record with no implicit padding.

struct GpuObjectData {
    mat4  model;
    vec3  tint;      float roughness;
    vec3  emissive;  float metallic;
    uint  albedo_index;
    uint  normal_index;
    float macro_variation;
    float terrain_blend;
    vec3  bb_min;    float cull_distance;
    vec3  bb_max;    float secondary_blend_sharpness;
    uint  albedo_secondary_index;
    uint  normal_secondary_index;
    uint  emissive_map_index;
    uint  orm_map_index;
    float alpha_cutoff;
    float _pad0;
    float _pad1;
    float _pad2;
};
