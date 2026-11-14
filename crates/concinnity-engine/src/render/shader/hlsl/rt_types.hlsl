// The CPU-visible records the ray-traced passes bind, spliced into a shader at
// its RT_TYPES marker (see shader_source.rs). Declares no resource and no entry
// point: the records have to exist before a shader's bindings name them, while
// the traversal that reads them (RT_TRACE) comes after. Nothing here may spell
// either marker -- the splice would reintroduce it.

{LEVEL_LAYER}

{LEVEL_WAVES}

// The visibility bit a planet ground tile's TLAS instance adds to the one
// every instance carries: a ray tracing it alone meets only the ground.
// Mirrors `RT_MASK_GROUND` in render::rt_accel::geom.
static const uint RT_MASK_GROUND = 2u;

// The level surfaces rays meet analytically, the unused ones off, and the
// waves on the shell among them. Layout matches `LevelSurfacesGpu`.
struct LevelSurfaces
{
    LevelLayer layers[LEVEL_LAYERS];
    LevelWaves waves;
};

// The sphere a planet's ground stays under. Layout matches `GroundBoundGpu`.
struct GroundBound
{
    // xyz: the center; w: the radius, 0 in a world without a planet.
    float4 sphere;
    // x: the camera's height over the sphere, formed in double precision.
    float4 camera;
};

// Layout matches `RtParams` in render_types.rs (816 B).
struct RtParams
{
    float intensity;
    float max_distance;
    float tan_half_fov_y;
    float aspect;
    // IBL prefilter cubemap mip count; 0 means no EnvironmentMap is bound.
    float prefilter_mip_count;
    // 1.0 when a reflected hit casts a sun-shadow ray, 0.0 when it is lit
    // unshadowed.
    float sun_shadows;
    // Per-axis divisor of the reduced glass reflection pre-pass; 1.0 when glass
    // traces its reflection in place.
    float trace_divisor;
    // The scale the HDR chain is written at: the reflection pass shades in
    // absolute radiance and multiplies what it writes by it.
    float pre_exposure;
    // float4 rather than float3 + scalar: MSL sizes a constant-buffer float3 at
    // 16 bytes, so only the padded form matches the CPU layout on every target.
    float4 cam_pos;    // xyz: world camera position; w: screen-trace pyramid levels
    float4 sun_dir;    // xyz: world unit direction toward the sun (= L)
    float4 sun_color;  // xyz: sun radiance
    float4x4 inv_view; // camera-to-world
    // Rows of the rotation from world space into the sky cube's baked frame;
    // identity when the sky does not turn.
    float4 sky_rot[3];
    // The level surfaces rays meet analytically, and the waves on them.
    LevelSurfaces level;
    // x: 1 when a ray that leaves the scene with a sharp lobe is listed for
    // the sky's own pass, 0 when it takes the prefiltered sky.
    float4 sky;
    // x: the frame's index, which the sampled directions and the sky's turns
    // advance by.
    uint4 frame;
    // The sphere a planet's ground stays under.
    GroundBound ground;
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
    uint far_canopy;         // the far field's packed canopy layers (0 = none)
    uint _pad1;
    uint _pad2;
};
