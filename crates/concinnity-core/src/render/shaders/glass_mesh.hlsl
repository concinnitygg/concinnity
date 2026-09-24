// See-through glass mesh pass: single source for every backend.
//
// The third producer of the engine's transparent pass (`PassId::Transparent`,
// after the SSR resolve and before TAA), alongside `glass.hlsl` and
// `water.hlsl`. Where a glass pane is a flat pre-baked world-space quad, this
// draws an IMPORTED mesh whose `Material` is flagged `see_through`: the geometry
// comes from the shared scene vertex / index buffers in LOCAL space, so the
// vertex stage applies the per-draw model matrix and the fragment shades off the
// interpolated per-vertex world normal, which is what lets a curved glass facade
// reflect correctly across its surface.
//
// The pass is ray-traced only, by design: what makes the mesh see-through rather
// than the opaque low-roughness glass of Layer 1 is a real per-pixel reflection
// ray, so there is no probe-only variant to fall back to. When RT is off the
// host leaves these meshes in the opaque pass instead of drawing them here.
// Two compiles, differing only in where a reflected hit's surface parameters
// come from:
//
//   default      - the reflected hit takes its flat per-object material tint.
//   RT_TEXTURED  - the same trace, with reflected hits taking their albedo /
//                  normal / emissive maps from the bindless pool.
//
// Each carries two fragment entries: `glass_mesh_rt_fragment` shades the mesh,
// and `glass_mesh_reflection_fragment` is the reduced reflection pre-pass that
// shading reads back when `rt_params.trace_divisor` is above 1 (see
// glass_reflection.hlsl).
//
// The inputs every transparent producer shares come from TRANSPARENT_SCENE.

#ifndef USE_MSAA
#define USE_MSAA 0
#endif

{PROBE_TYPES}
{RT_TYPES}

{TRANSPARENT_TYPES}

// Per-mesh tunables. Layout matches `GlassMeshParams` (96 B); `model` is first
// so its 16-byte alignment is satisfied at offset 0.
struct GlassMeshParams
{
    float4x4 model; // local -> world
    float4 tint;    // color multiplied into the refracted scene
    float opacity;
    float refraction_strength;
    float fresnel_power;
    // The mesh's own copy of the sky prefilter mip count, so the ray-miss
    // fallback does not depend on which view block is bound. 0 = no
    // EnvironmentMap, and the reflection keeps the white rim.
    float prefilter_mip_count;
};

// ---- Resource bindings ----

#define TRANSPARENT_PARAMS GlassMeshParams
{TRANSPARENT_SCENE}

{TRANSPARENT_PROBES}

{TRANSPARENT_RT}

{RT_TRACE}

{GLASS_REFLECTION}

// ---- Stage interface ----

struct GlassMeshVertexIn
{
    [[vk::location(0)]] float3 pos : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
};

struct GlassMeshVertexOut
{
    [[vk::location(0)]] float3 world_pos : TEXCOORD0;
    [[vk::location(1)]] float3 world_normal : TEXCOORD1;
    float4 position : SV_Position;
};

[shader("vertex")]
GlassMeshVertexOut glass_mesh_vertex(GlassMeshVertexIn v)
{
    GlassMeshVertexOut o;
    float4 world = mul(params.model, float4(v.pos, 1.0));
    o.world_pos = world.xyz;
    // Rigid / uniform-scale model, so `M * n` needs no inverse-transpose. The
    // fragment renormalizes after interpolation.
    o.world_normal = mul((float3x3)params.model, v.normal);
    o.position = mul(view.vp, world);
    return o;
}

// Depth stored at this pixel by the main pass, for the manual occlusion test.
float glass_mesh_scene_depth(int2 pixel)
{
#if USE_MSAA
    return scene_depth.Load(pixel, 0);
#else
    return scene_depth.Load(int3(pixel, 0));
#endif
}

// The mesh surface at this fragment: the view-facing interpolated normal
// (two-sided, so a pane of glass lit from behind still Fresnels correctly), the
// fragment's screen UV and the refracted, tinted background behind it.
struct GlassMeshSurface
{
    float3 normal;
    float2 frag_uv;
    float3 refracted;
};

GlassMeshSurface glass_mesh_surface(GlassMeshVertexOut i, float3 view_dir)
{
    GlassMeshSurface s;
    s.normal = normalize(i.world_normal);
    if (dot(s.normal, view_dir) < 0.0)
    {
        s.normal = -s.normal;
    }

    float2 vp_dim = max(view.viewport, (float2)(1.0));
    s.frag_uv = i.position.xy / vp_dim;

    float2 refract_uv = clamp(s.frag_uv + s.normal.xy * params.refraction_strength,
                              (float2)(0.001), (float2)(0.999));
    s.refracted = scene_sample(refract_uv).rgb * params.tint.rgb;
    return s;
}

// Schlick Fresnel (F0 = 0.04 dielectric) mix of the reflection over the
// refraction, identical to `glass_resolve` so a mesh and a pane read the same at
// equal inputs.
float4 glass_mesh_resolve(GlassMeshSurface s, float3 view_dir, float3 reflection)
{
    float n_dot_v = saturate(dot(s.normal, view_dir));
    float rim = pow(1.0 - n_dot_v, max(params.fresnel_power, 1e-3));
    float refl_weight = saturate(RT_F0 + 0.96 * rim);
    float3 color = lerp(s.refracted, reflection, refl_weight);
    float alpha = saturate(lerp(params.opacity, 1.0, rim));
    return float4(color, alpha);
}

// The mesh's reflection at `world_pos`: a ray off the interpolated world-space
// surface point, so a curved facade mirrors real off-screen geometry across its
// whole span. The mesh is excluded from the BLAS (glass does not reflect glass),
// so the trace never self-hits. Glass is smooth, so the trace is sharp and the
// miss falls back to the probe / sky chain.
float3 glass_mesh_reflection(float3 world_pos, float3 view_dir, float3 normal, float2 pixel)
{
    bool ibl = params.prefilter_mip_count > 0.5;
    float3 r = reflect(-view_dir, normal);
    float3 reflection;
    if (!rt_trace_reflection(world_pos + normal * 0.02, r, ibl,
                             params.prefilter_mip_count - 1.0, reflection))
    {
        // Glass is smooth, so the fallback is sharp (mip 0), with a white rim
        // where there is neither a probe nor a sky.
        reflection = transparent_environment(world_pos, r, 0.0, (float3)(1.0), pixel);
    }
    return reflection;
}

// Whether nearer opaque geometry covers the mesh at full-resolution pixel
// position `full_position`.
bool glass_mesh_occluded(float2 full_position, float depth)
{
    int2 pixel = min(int2(full_position), int2(max(view.viewport, (float2)(1.0))) - int2(1, 1));
    return glass_mesh_scene_depth(pixel) < depth;
}

[shader("pixel")]
float4 glass_mesh_rt_fragment(GlassMeshVertexOut i) : SV_Target
{
    float3 view_dir = normalize(view.camera_pos.xyz - i.world_pos);
    GlassMeshSurface s = glass_mesh_surface(i, view_dir);
    if (glass_mesh_occluded(i.position.xy, i.position.z))
    {
        discard;
    }

    float3 reflection;
    if (!glass_reflection_lookup(s.frag_uv, distance(view.camera_pos.xyz, i.world_pos),
                                 reflection))
    {
        reflection = glass_mesh_reflection(i.world_pos, view_dir, s.normal, i.position.xy);
    }
    return glass_mesh_resolve(s, view_dir, reflection);
}

// The reduced reflection pre-pass: the traced reflection and this surface's
// distance from the camera, for `glass_reflection_lookup` to upsample. Drawn
// once per layer; a fragment on or in front of the layer `glass_reflection`
// holds is left to that layer.
[shader("pixel")]
float4 glass_mesh_reflection_fragment(GlassMeshVertexOut i) : SV_Target
{
    float dist = distance(view.camera_pos.xyz, i.world_pos);
    if (glass_mesh_occluded(glass_reflection_full_position(i.position.xy), i.position.z)
        || glass_reflection_in_front_layer(i.position.xy, dist))
    {
        discard;
    }
    float3 view_dir = normalize(view.camera_pos.xyz - i.world_pos);
    float3 normal = normalize(i.world_normal);
    if (dot(normal, view_dir) < 0.0)
    {
        normal = -normal;
    }
    return float4(glass_mesh_reflection(i.world_pos, view_dir, normal, i.position.xy), dist);
}
