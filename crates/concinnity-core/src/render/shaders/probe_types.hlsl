// Reflection-probe records, spliced into a shader at its PROBE_TYPES marker
// (see shader_source.rs). Declares no resource and no entry point, so it is not
// a standalone program; it exists because the probe set is bound at a different
// slot by every pass that reads it, while the record layout is one CPU struct.
//
// Splices in two halves on purpose: the records have to be declared before a
// shader's resource bindings (a binding names ProbeSet) and the sampling
// helpers after them (they read the bound set through PROBE_SET). Nothing here
// may spell either marker -- the splice would reintroduce it.
//
// The set is three bindings: a ProbeSet header with the live count, a
// StructuredBuffer of ProbeUniforms records and a TextureCubeArray with one cube
// per record. Their lengths are whatever the host allocated; the shader reads
// only the first `count`.

// Fraction of a probe box's smallest half-extent over which its blend weight
// ramps from 0 to 1 across the box surface.
static const float PROBE_BLEND_MARGIN = 0.2;

struct ProbeUniforms
{
    float4 box_min;   // xyz = influence-box min, w = enabled
    float4 box_max;   // xyz = influence-box max
    float4 probe_pos; // xyz = capture position
};

// Mirror of `uniforms::ProbeSet`: the live record count and each cube's mip
// count, padded to one register.
struct ProbeSet
{
    uint count;
    uint mip_count;
    uint _pad0;
    uint _pad1;
};
