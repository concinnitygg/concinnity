// The CPU-visible records the ray-traced passes bind, spliced into a shader at
// its RT_TYPES marker (see slang_source.rs). Declares no resource and no entry
// point: the records have to exist before a shader's bindings name them, while
// the traversal that reads them (RT_TRACE) comes after. Nothing here may spell
// either marker -- the splice would reintroduce it.

// Layout matches `RtParams` in render_types.rs (144 B).
struct RtParams
{
    float intensity;
    float max_distance;
    float tan_half_fov_y;
    float aspect;
    // IBL prefilter cubemap mip count; 0 means no EnvironmentMap is bound.
    float prefilter_mip_count;
    float _pad0;
    float _pad1;
    float _pad2;
    // float4 rather than float3 + scalar: MSL sizes a constant-buffer float3 at
    // 16 bytes, so only the padded form matches the CPU layout on every target.
    float4 cam_pos;    // xyz: world camera position
    float4 sun_dir;    // xyz: world unit direction toward the sun (= L)
    float4 sun_color;  // xyz: sun radiance
    float4x4 inv_view; // camera-to-world
};

// One traced instance's geometry range + material, indexed by the hit's
// instance index. Scalar `tint` / `emissive` components rather than float3
// fields, so the offsets match the 128-byte `#[repr(C)]` `RtGeomEntry` on every
// target: a float3 would be 16-byte aligned and shift `model`, which faults the
// trace rather than merely shading it wrong.
struct RtGeomEntry
{
    uint index_offset;  // element offset of this object's first index
    uint base_vertex;   // added to each fetched index
    uint albedo_index;  // bindless albedo pool index
    uint normal_index;  // bindless normal-map index, RT_SKINNED_FLAG in bit 31
    float tint_r;
    float tint_g;
    float tint_b;
    float roughness;
    float metallic;
    float emissive_r;
    float emissive_g;
    float emissive_b;
    float4x4 model;          // object-to-world
    uint emissive_map_index; // bindless emissive-map index (0 = none)
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
