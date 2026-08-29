// The per-object vertex vocabulary every GPU-driven pass shares: the bindless
// object record, the object-id reconstruction the engine's first-instance
// encoding needs, and the normal matrix.
//
// Spliced at the OBJECT_COMMON marker rather than imported as a Slang module,
// for the same reason the variant defines ride the source text: the
// replacement lands in the assembled source, so the content-addressed shader
// cache keys it. `slang_source::assemble` is the runtime half of the splice and
// `SLANG_SHADER_FRAGMENTS` in the device build script the build-time half.
//
// This is the single-source counterpart of the per-language `object_common.msl`
// / `.glsl` / `.hlsl` fragments, which stay until their remaining consumers
// (the cull kernel, the legacy per-draw main pass) port.

// Mirrors `GpuObjectData` in concinnity-core/src/gfx/render_types.rs (176 B).
// Each (vec3, scalar) pair is spelled as one float4: MSL sizes a float3 at 16
// bytes in a structured buffer as well as in a constant buffer, so a literal
// transcription pushes every following field four bytes late on Metal alone.
struct GpuObjectData
{
    float4x4 model;
    // xyz = tint, w = roughness.
    float4 tint_roughness;
    // xyz = emissive, w = metallic.
    float4 emissive_metallic;
    uint  albedo_index;
    uint  normal_index;
    float macro_variation;
    float terrain_blend;
    // xyz = bounding-box minimum, w = cull distance.
    float4 bb_min_cull_distance;
    // xyz = bounding-box maximum, w = secondary blend sharpness.
    float4 bb_max_blend_sharpness;
    uint  albedo_secondary_index;
    uint  normal_secondary_index;
    uint  emissive_map_index;
    uint  orm_map_index;
    float alpha_cutoff;
    float _pad0;
    float _pad1;
    float _pad2;
};

// The engine encodes the object id as the draw's first-instance value. The
// targets disagree on what an instance-id builtin contains: SPIR-V's
// InstanceIndex and Metal's instance_id include the base instance, while
// SV_InstanceID is defined zero-based (Slang subtracts the base to honor
// that). Reconstruct the object id explicitly from both builtins.
uint object_instance_index(uint zero_based_iid, uint first_instance)
{
    __target_switch
    {
    case metal:
        // Slang's Metal lowering passes instance_id through unsubtracted, and
        // instance_id already includes base_instance.
        return zero_based_iid;
    default:
        return zero_based_iid + first_instance;
    }
}

float3x3 normal_matrix(float4x4 model)
{
    // Slang subscripts a matrix by row, so these are the upper 3x3's rows. The
    // cofactor matrix built from them is the inverse-transpose scaled by the
    // determinant; the normalize() at every use site absorbs that scale.
    float3 a0 = model[0].xyz;
    float3 a1 = model[1].xyz;
    float3 a2 = model[2].xyz;
    float3 r0 = cross(a1, a2);
    float3 r1 = cross(a2, a0);
    float3 r2 = cross(a0, a1);
    // mul(M, v) with these as rows applies the inverse-transpose to v.
    return float3x3(r0, r1, r2);
}
