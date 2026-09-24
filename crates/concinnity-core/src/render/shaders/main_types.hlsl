// The CPU-visible records the forward main pass binds, spliced at a shader's
// MAIN_TYPES marker. The first half of the main-pass splice: this declares what
// a binding names, `main_shading.hlsl` the body that reads it.
//
// Layouts must match render_types.rs: GpuLight (64 B), SpotShadowData (80 B),
// AreaLightData (32 B).
//
// `SKY_DIR` is a macro rather than a function because `VIEW` is a per-backend
// binding the including shader names after this splice.

struct ViewUniforms
{
    float4x4 vp;
    float4x4 view_mat;
    float elapsed;
    // 1.0 when an SSR / RT reflection composite owns the sharp specular this
    // frame (fade the glossy-dielectric forward probe specular); 0.0 keeps it.
    float reflections_enabled;
    float cam_x; float cam_y; float cam_z;
    float prefilter_mip_count;
    // 1.0 while the unlit view mode is active: the surface returns its base
    // color before lighting.
    float shade_mode; float _ep1;
    // Rows of the rotation from world space into the environment cubemaps'
    // baked frame; identity when the sky does not turn.
    float4 sky_rot[3];
};

// A world direction in the environment cubemaps' own frame. Every sky tap goes
// through this, so the sky, the ambient fill and the reflections turn together.
// Defined as a macro because VIEW is a per-backend binding named below.
#define SKY_DIR(d) float3(dot(VIEW.sky_rot[0].xyz, (d)), \
                          dot(VIEW.sky_rot[1].xyz, (d)), \
                          dot(VIEW.sky_rot[2].xyz, (d)))

{OBJECT_COMMON}

{LIGHT_TYPES}

// GpuLight.kind discriminants (LIGHT_KIND_* in render_types.rs).
static const uint LIGHT_KIND_SPOT = 1u;
static const uint LIGHT_KIND_AREA = 2u;

uint light_kind(GpuLight l) { return asuint(l.direction_kind.w); }

// One shadowed spot's slice projection; indexed by GpuLight.shadow_index,
// which doubles as the array layer.
struct SpotShadowData
{
    float4x4 light_vp;
    float depth_bias;
    float normal_bias;
    float2 _pad;
};

// One rectangular area light's extent, indexed by GpuLight.data_index. The
// edges are pre-scaled by the half-extents, so the corners are
// center +/- right +/- up.
struct AreaLightData
{
    // xyz = right edge, w = the two-sided flag's bits.
    float4 right_two_sided;
    // xyz = up edge, w unused.
    float4 up_pad;
};

{CLUSTER_TYPES}
