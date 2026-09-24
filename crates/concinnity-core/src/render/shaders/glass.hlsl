// Glass panel pass: single source for every backend.
//
// The simplest consumer of the engine's transparent pass -- a flat, fixed
// rectangular pane, drawn in the `PassId::Transparent` slot after the SSR
// resolve and before TAA. The quad is already in world space (see
// geometry::glass_quad), so the vertex stage only projects it. The fragment
// stage discards where nearer opaque geometry occludes the pane (a manual depth
// test: the transparent pass binds no depth attachment), refracts the
// pre-transparent scene snapshot and tints it, then mixes a reflection over the
// refraction by a Schlick Fresnel term. The pipeline straight-alpha blends the
// result.
//
// One fragment entry per compile, selected by a define; everything outside the
// reflection itself is shared, so toggling RT only changes where the reflection
// comes from:
//
//   default                   - the sharp planar reflection when this pane has
//                               one, else the box-projected reflection-probe
//                               set, else the sky prefilter cube, else a white
//                               rim.
//   GLASS_RT                  - a real per-pixel reflection ray against the
//                               scene acceleration structure (the shared
//                               RT_TRACE fragment), falling back to the same
//                               probe / sky chain when the ray escapes.
//                               Also carries `glass_rt_reflection_fragment`,
//                               the reduced reflection pre-pass the shading
//                               reads back when `rt_params.trace_divisor` is
//                               above 1 (see glass_reflection.hlsl).
//   GLASS_RT with RT_TEXTURED - the same trace, with reflected hits taking
//                               their albedo / normal / emissive maps from the
//                               bindless pool.
//
// CN_BACKEND_DIRECTX pins every register to the root signatures in
// `directx/glass.rs`; the other targets carry one declaration with both a
// `[[vk::binding]]` and a `register()`, whose number IS the Metal index in the
// namespace the resource's kind implies. The inputs every transparent producer
// shares come from TRANSPARENT_SCENE, whose slots are not free to move. The
// bindless pool is a whole descriptor set on Metal, named by a
// `[[cn::metal_argument_buffer(n)]]`, because no per-resource register can name
// a set.

#ifndef USE_MSAA
#define USE_MSAA 0
#endif

{PROBE_TYPES}

#ifdef GLASS_RT
{RT_TYPES}
#endif

{TRANSPARENT_TYPES}

// Per-pane tunables. Layout matches `GlassParams` / the GlassParamsBlock UBO
// (64 B); the vec3 fields are float4 so MSL's 16-byte constant-buffer float3
// cannot desynchronize them from the CPU struct.
struct GlassParams
{
    float4 center; // world-space pane center
    float4 normal; // unit pane normal (facing direction)
    float4 tint;   // color multiplied into the refracted scene
    float opacity;
    float refraction_strength;
    float fresnel_power;
    // 1.0 when this pane has a planar reflection slot: sample the sharp mirror
    // render projectively instead of the box-projected probe. Never set on the
    // RT path, which traces a sharper reflection than a planar render.
    float planar;
};

// ---- Resource bindings ----

#define TRANSPARENT_PARAMS GlassParams
{TRANSPARENT_SCENE}

#ifdef CN_BACKEND_DIRECTX
Texture2D<float4> planar_reflection : register(t3);
#define planar_reflection_sampler post_samp
#endif

{TRANSPARENT_PROBES}

#ifndef CN_BACKEND_DIRECTX
// This pane's planar reflection target, bound per pane and sampled projectively
// when `planar > 0.5`. A pane with no planar slot binds a valid stand-in and
// never samples it. Only the non-RT entry reads it; the RT variants trace a
// sharper reflection than a mirror render, and drop it from their reflection.
[[vk::binding(1, 1)]] Texture2D<float4> planar_reflection : register(t3);
[[vk::binding(2, 1)]] SamplerState planar_reflection_sampler : register(s3);
#endif

float4 planar_sample(float2 uv)
{
    return planar_reflection.Sample(planar_reflection_sampler, uv);
}

#ifdef GLASS_RT
{TRANSPARENT_RT}

{RT_TRACE}

{GLASS_REFLECTION}
#endif

// ---- Stage interface ----

struct GlassVertexIn
{
    [[vk::location(0)]] float3 pos : POSITION;
};

struct GlassVertexOut
{
    [[vk::location(0)]] float3 world_pos : TEXCOORD0;
    float4 position : SV_Position;
};

[shader("vertex")]
GlassVertexOut glass_vertex(GlassVertexIn v)
{
    GlassVertexOut o;
    // Quad vertices are pre-transformed into world space at build time.
    o.world_pos = v.pos;
    o.position = mul(view.vp, float4(v.pos, 1.0));
    return o;
}

// Depth stored at this pixel by the main pass, for the manual occlusion test.
float glass_scene_depth(int2 pixel)
{
#if USE_MSAA
    return scene_depth.Load(pixel, 0);
#else
    return scene_depth.Load(int3(pixel, 0));
#endif
}

// The pane surface at this fragment: the view-facing normal (two-sided, so a
// pane lit from behind still Fresnels correctly), the fragment's screen UV and
// the refracted, tinted background behind it.
struct GlassSurface
{
    float3 normal;
    float2 frag_uv;
    float3 refracted;
};

GlassSurface glass_surface(float4 position, float3 view_dir)
{
    GlassSurface s;
    s.normal = normalize(params.normal.xyz);
    if (dot(s.normal, view_dir) < 0.0)
    {
        s.normal = -s.normal;
    }

    float2 vp_dim = max(view.viewport, (float2)(1.0));
    s.frag_uv = position.xy / vp_dim;

    // Refraction: perturb the screen lookup by the pane normal's screen-plane
    // component so the background bends across the pane.
    float2 refract_uv = clamp(s.frag_uv + s.normal.xy * params.refraction_strength,
                              (float2)(0.001), (float2)(0.999));
    s.refracted = scene_sample(refract_uv).rgb * params.tint.rgb;
    return s;
}

// Schlick Fresnel (F0 = 0.04 dielectric) mix of the reflection over the
// refraction: ~4% head-on, rising to a full mirror at grazing. `fresnel_power`
// stays the author's grazing-rim shaping control for the opacity ramp.
float4 glass_resolve(GlassSurface s, float3 view_dir, float3 reflection)
{
    float n_dot_v = saturate(dot(s.normal, view_dir));
    float rim = pow(1.0 - n_dot_v, max(params.fresnel_power, 1e-3));
    float refl_weight = saturate(0.04 + 0.96 * rim);
    float3 color = lerp(s.refracted, reflection, refl_weight);
    float alpha = saturate(lerp(params.opacity, 1.0, rim));
    return float4(color, alpha);
}

// The reflection a ray that hit nothing (or a pane with no trace at all) falls
// back to: the box-projected probe set where a probe actually covers this pane,
// else the sky prefilter cube, else a white rim so a probe-less, env-less world
// still reads as glass. A pane is smooth, so every path is sharp (mip 0).
float3 glass_environment(float3 world_pos, float3 r, float2 pixel)
{
    return transparent_environment(world_pos, r, 0.0, (float3)(1.0), pixel);
}

#ifdef GLASS_RT

// The pane's reflection at `world_pos`: a ray off the world-space pane surface
// point, so a window mirrors real off-screen geometry. A pane is smooth, so the
// trace is sharp and the miss falls back to the probe / sky chain.
float3 glass_rt_reflection(float3 world_pos, float3 view_dir, float3 normal, float2 pixel)
{
    float3 r = reflect(-view_dir, normal);
    float3 reflection;
    if (!rt_trace_reflection(world_pos + normal * 0.02, r,
                             view.prefilter_mip_count > 0.5,
                             view.prefilter_mip_count - 1.0, reflection))
    {
        reflection = glass_environment(world_pos, r, pixel);
    }
    return reflection;
}

// Whether nearer opaque geometry covers the pane at full-resolution pixel
// position `full_position`.
bool glass_occluded(float2 full_position, float depth)
{
    int2 pixel = min(int2(full_position), int2(max(view.viewport, (float2)(1.0))) - int2(1, 1));
    return glass_scene_depth(pixel) < depth;
}

[shader("pixel")]
float4 glass_rt_fragment(GlassVertexOut i) : SV_Target
{
    float3 view_dir = normalize(view.camera_pos.xyz - i.world_pos);
    GlassSurface s = glass_surface(i.position, view_dir);
    if (glass_occluded(i.position.xy, i.position.z))
    {
        discard;
    }

    float3 reflection;
    if (!glass_reflection_lookup(s.frag_uv, distance(view.camera_pos.xyz, i.world_pos),
                                 reflection))
    {
        reflection = glass_rt_reflection(i.world_pos, view_dir, s.normal, i.position.xy);
    }
    return glass_resolve(s, view_dir, reflection);
}

// The reduced reflection pre-pass: the traced reflection and this pane's
// distance from the camera, for `glass_reflection_lookup` to upsample. Drawn
// once per layer; a fragment on or in front of the layer `glass_reflection`
// holds is left to that layer.
[shader("pixel")]
float4 glass_rt_reflection_fragment(GlassVertexOut i) : SV_Target
{
    float dist = distance(view.camera_pos.xyz, i.world_pos);
    if (glass_occluded(glass_reflection_full_position(i.position.xy), i.position.z)
        || glass_reflection_in_front_layer(i.position.xy, dist))
    {
        discard;
    }
    float3 view_dir = normalize(view.camera_pos.xyz - i.world_pos);
    float3 normal = normalize(params.normal.xyz);
    if (dot(normal, view_dir) < 0.0)
    {
        normal = -normal;
    }
    return float4(glass_rt_reflection(i.world_pos, view_dir, normal, i.position.xy), dist);
}

#else

[shader("pixel")]
float4 glass_fragment(GlassVertexOut i) : SV_Target
{
    float3 view_dir = normalize(view.camera_pos.xyz - i.world_pos);
    GlassSurface s = glass_surface(i.position, view_dir);

    // Manual depth occlusion: discard where the stored scene depth at this pixel
    // is nearer than the pane. Every backend rasterizes this depth under the
    // same viewport convention the main pass used, so the fragment position
    // lines up with the stored texel.
    int2 pixel = min(int2(i.position.xy), int2(max(view.viewport, (float2)(1.0))) - int2(1, 1));
    if (glass_scene_depth(pixel) < i.position.z)
    {
        discard;
    }

    // A flat pane is a perfect mirror, so an active planar reflection (the scene
    // re-rendered mirrored across this pane's plane) lands exactly under the
    // reflector and is sampled at the fragment's own screen UV with no
    // distortion.
    float3 r = reflect(-view_dir, s.normal);
    float3 reflection = params.planar > 0.5 ? planar_sample(s.frag_uv).rgb
                                            : glass_environment(i.world_pos, r, i.position.xy);
    return glass_resolve(s, view_dir, reflection);
}

#endif
