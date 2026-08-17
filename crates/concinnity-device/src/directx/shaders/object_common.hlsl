// object_common.hlsl
//
// The bindless per-object record, for the HLSL shaders that still stride the
// per-frame StructuredBuffer: the cull kernel is the only one left, the rest
// having moved to `shaders/object_common.slang`. Both copies mirror
// `GpuObjectData` in concinnity-core/src/gfx/render_types.rs (176 bytes) and
// must agree, since the kernel writes the records the single-source passes read.
//
// The DX HLSL path has no include handler, so this file is substituted into the
// consuming shader at its OBJECT_DATA marker (builtins::HlslProgram::source).
//
// `model` carries an explicit `column_major` qualifier rather than relying on
// each consumer's `#pragma pack_matrix`: the qualifier pins the storage layout
// to the column-major matrix Rust uploads no matter where this text lands.

struct GpuObjectData
{
    column_major float4x4 model;
    float3 tint;
    float roughness;
    float3 emissive;
    float metallic;
    uint albedo_index;
    uint normal_index;
    float macro_variation;
    float terrain_blend;
    float3 bb_min;
    float cull_distance;
    float3 bb_max;
    float secondary_blend_sharpness;
    uint albedo_secondary_index;
    uint normal_secondary_index;
    uint emissive_map_index;
    uint orm_map_index;
    float alpha_cutoff;
    float _pad0;
    float _pad1;
    float _pad2;
};
