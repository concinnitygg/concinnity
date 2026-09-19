// The reduced-resolution glass reflection, spliced into the ray-traced glass
// shaders at their GLASS_REFLECTION marker. Not a standalone program. Nothing
// here may spell the marker itself.
//
// When `rt_params.trace_divisor` is above 1, the host first draws every
// see-through glass surface into two targets that many times smaller per axis,
// through the shader's reflection entry, depth-tested so each keeps the nearest
// surface it accepts: rgb is the traced reflection and a is the glass surface's
// distance from the camera (0 where no glass drew). The first layer is the
// nearest glass; the second is the next surface behind it, which is what a
// closed mesh (a pane modeled as a slab, a bottle) shows through its front. The
// full-resolution glass draw then reads its reflection from the two layers
// instead of tracing, and traces only where no nearby texel belongs to the same
// surface -- a third layer, or a sliver thinner than a texel.
//
// Hooks the including shader must provide: `rt_params`, `glass_reflection` (the
// first layer, and while drawing the second, the layer to peel behind) and
// `glass_reflection_back` (the second layer), each a plain texture read with
// Load, so neither takes a sampler slot.

// How sharply a texel is rejected for sitting at a different distance
// (relative) than the fragment reading it.
static const float GLASS_REFLECTION_DEPTH_FALLOFF = 32.0;
// The least matching bilinear footprint the lookup accepts before tracing.
static const float GLASS_REFLECTION_MIN_COVERAGE = 0.25;
// How far (relative) behind the first layer a surface must lie to be the
// second layer rather than the first one drawn again.
static const float GLASS_REFLECTION_LAYER_GAP = 0.01;

// The full-resolution pixel position a fragment of the reduced pre-pass
// stands for.
float2 glass_reflection_full_position(float2 position)
{
    return position * rt_params.trace_divisor;
}

// Whether a pre-pass fragment `dist` from the camera lies on or in front of
// the layer `glass_reflection` holds at its texel, and so belongs to that
// layer rather than the one behind it. Never true against an empty layer.
bool glass_reflection_in_front_layer(float2 position, float dist)
{
    uint w, h;
    glass_reflection.GetDimensions(w, h);
    int2 texel = min(int2(position), int2(int(w), int(h)) - int2(1, 1));
    float front = glass_reflection.Load(int3(texel, 0)).a;
    return front > 0.0 && dist <= front * (1.0 + GLASS_REFLECTION_LAYER_GAP);
}

// Accumulate one layer's four nearest texels around `p` (texel space, the
// bilinear origin), each weighted by its bilinear footprint and by how closely
// its stored distance matches `dist`.
void glass_reflection_taps(Texture2D<float4> layer, float2 p, float dist, inout float3 sum,
                           inout float wsum)
{
    uint w, h;
    layer.GetDimensions(w, h);
    float2 base = floor(p);
    float2 f = p - base;
    int2 last = int2(int(w), int(h)) - int2(1, 1);
    for (int j = 0; j < 2; j++)
    {
        for (int i = 0; i < 2; i++)
        {
            float2 o = float2(float(i), float(j));
            int2 texel = clamp(int2(base + o), int2(0, 0), last);
            float4 c = layer.Load(int3(texel, 0));
            if (c.a <= 0.0)
            {
                continue;
            }
            float2 axis = lerp(1.0 - f, f, o);
            float dz = abs(c.a - dist) / max(dist, 1e-3);
            float weight = axis.x * axis.y * exp2(-dz * GLASS_REFLECTION_DEPTH_FALLOFF);
            sum += c.rgb * weight;
            wsum += weight;
        }
    }
}

// The reflection the pre-pass traced for the glass surface at `frag_uv`,
// `dist` from the camera, from whichever layer holds that surface. False when
// no pre-pass ran or too little of the footprint is this surface.
bool glass_reflection_lookup(float2 frag_uv, float dist, out float3 reflection)
{
    reflection = float3(0.0);
    if (rt_params.trace_divisor < 1.5)
    {
        return false;
    }
    uint w, h;
    glass_reflection.GetDimensions(w, h);
    float2 p = frag_uv * float2(float(w), float(h)) - 0.5;
    float3 sum = float3(0.0);
    float wsum = 0.0;
    glass_reflection_taps(glass_reflection, p, dist, sum, wsum);
    glass_reflection_taps(glass_reflection_back, p, dist, sum, wsum);
    if (wsum < GLASS_REFLECTION_MIN_COVERAGE)
    {
        return false;
    }
    reflection = sum / wsum;
    return true;
}
