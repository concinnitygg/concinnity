// The per-object vertex vocabulary every GPU-driven pass shares: the bindless
// object record, the object-id reconstruction the engine's first-instance
// encoding needs, and the normal matrix.
//
// Spliced at the OBJECT_COMMON marker rather than imported as a module, for the
// same reason the variant defines ride the source text: the replacement lands
// in the assembled source, so the content-addressed shader cache keys it.
// `shader_source::assemble` performs the splice, for the runtime compile and
// the build script's precompile alike.
//
// This is the record's only declaration: `shader_layout` in concinnity-device
// reflects it against `GpuObjectData` on every target, and its
// `no_shader_redeclares_the_object_record` scan keeps it that way.

// Mirrors `GpuObjectData` in concinnity-core/src/gfx/render_types.rs (144 B).
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
    uint  emissive_map_index;
    uint  orm_map_index;
    // xyz = bounding-box minimum, w = cull distance.
    float4 bb_min_cull_distance;
    // xyz = bounding-box maximum, w = alpha cutoff.
    float4 bb_max_alpha_cutoff;
};

// Mirrors `GpuDrawArgs` in concinnity-core/src/gfx/render_types.rs (16 B): the
// per-frame cull decision + indexed-draw slice for the record at the same index
// as the `GpuObjectData` above. Declared here rather than in the cull, because
// the G-buffer pre-pass reads `flags` too.
struct GpuDrawArgs
{
    uint index_count;
    uint index_offset;
    uint base_vertex;
    uint flags;
};

// `GpuDrawArgs::flags` bits, locked to gfx::render_types::DrawArgsFlags.
static const uint DRAW_ENABLED    = 1u;
static const uint DRAW_CULLABLE   = 2u;
// The record's history entry in the model-history ring belongs to a different
// occupant, so a motion vector must fall back to the current model.
static const uint DRAW_NO_HISTORY = 4u;

// The record's shader bucket rides the upper flag bits; values and layout are
// locked to gfx::render_types::{DrawArgsFlags::BUCKET_SHIFT, MAX_SHADER_BUCKETS}.
static const uint DRAW_BUCKET_SHIFT = 8u;
static const uint DRAW_BUCKET_MASK  = 0xFFu;

// The engine encodes the object id as the draw's first-instance value. dxc
// lowers SV_InstanceID to SPIR-V's InstanceIndex, which includes the base
// instance, and spirv-cross carries that to Metal's instance_id, which does
// too: the builtin already is the object id on both legs, so no entry declares
// a base-instance builtin at all.
//
// The DirectX leg gets no definition on purpose. There SV_InstanceID keeps
// D3D's zero-based meaning, and the id arrives as the indirect command's b0
// root constant instead, so an entry reaching for this function under
// CN_BACKEND_DIRECTX is a mistake the compile should name rather than a value it
// should guess.
#ifndef CN_BACKEND_DIRECTX
uint object_instance_index(uint instance_index)
{
    return instance_index;
}
#endif

float3x3 normal_matrix(float4x4 model)
{
    // A matrix subscript is a row, so these are the upper 3x3's rows. The
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
