// Reflection-probe records, spliced into a shader at its PROBE_TYPES marker
// (see slang_source.rs). Declares no resource and no entry point, so it is not
// a standalone program; it exists because the probe set is bound at a different
// slot by every pass that reads it, while the record layout is one CPU struct.
//
// Splices in two halves on purpose: the records have to be declared before a
// shader's resource bindings (a binding names ProbeSet) and the sampling
// helpers after them (they read the bound set through PROBE_SET). Nothing here
// may spell either marker -- the splice would reintroduce it.
//
// MAX_PROBES is the probe-array length the host binds, injected as a define.

// Fraction of a probe box's smallest half-extent over which its blend weight
// ramps from 0 to 1 across the box surface.
static const float PROBE_BLEND_MARGIN = 0.2;

struct ProbeUniforms
{
    float4 box_min;   // xyz = influence-box min, w = enabled
    float4 box_max;   // xyz = influence-box max
    float4 probe_pos; // xyz = capture position
};

// Mirror of `metal::uniforms::ProbeSet` / the forward global set's ProbeBlock:
// `count` is padded with THREE scalar uints rather than a uint3, which would be
// 16-byte aligned and push `probes` to offset 32.
struct ProbeSet
{
    uint count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    ProbeUniforms probes[MAX_PROBES];
};
