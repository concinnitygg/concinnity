// The CPU-visible records the forward main pass binds, spliced at a shader's
// MAIN_TYPES marker. The first half of the main-pass splice: this declares what
// a binding names, `main_shading.hlsl` the body that reads it.
//
// Layouts must match render_types.rs: GpuLight (64 B), SpotShadowData (80 B),
// AreaLightData (32 B).

{VIEW_UNIFORMS}

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
