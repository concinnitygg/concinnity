// Hardware ray-traced reflections: single source for every backend.
//
// A fullscreen fragment pass that, per glossy pixel, rebuilds a world-space
// surface point + normal from the SSR pre-pass G-buffer, traces a reflection
// ray against the scene's acceleration structure, shades the hit (or falls back
// to the local reflection probe / IBL prefilter cube on a miss) and writes
// reflected radiance (.rgb) + the Fresnel/gloss composite weight (.a). The
// reflection blur + composite then blends it over the scene, exactly as they do
// for the SSR resolve. Unlike SSR the ray is a real world-space trace, so
// reflected geometry that is off-screen still appears. Pairs with
// `fullscreen_vertex` in fullscreen.hlsl.
//
// The traversal itself is the shared RT_TRACE fragment, so this pass and glass
// run one traversal loop between them; only the bindings and the miss fallback
// are the pass's own.
//
// One entry per compile, selected by a define, so each variant declares exactly
// the resources it binds:
//
//   default     - flat: the per-object material tint as albedo, the fallback a
//                 non-bindless world takes.
//   RT_TEXTURED - samples the hit's albedo / normal / emissive maps from the
//                 bindless texture pool, the path standard worlds take.
//
// CN_BACKEND_DIRECTX pins every register to the root signature in
// `directx/post/rt_reflections.rs` and keeps DirectX's raw vertex / index SRVs
// (`ByteAddressBuffer`); the other targets carry one declaration with both a
// `[[vk::binding]]` and a `register()`, whose number is the Metal index in the
// namespace the resource's kind implies. The Metal argument buffer holding the
// bindless pool rides a descriptor set of its own, named by
// `[[cn::metal_argument_buffer(n)]]`, because no per-resource register can name
// a set. The DXIL container needs shader model 6.5 for ray query, above the 6.0
// floor the bindless main pass established.

{POST_COMMON}

{TEXTURE_SIZE}

{PROBE_TYPES}

{RT_TYPES}

{CLUSTER_TYPES}

// A world direction in the sky cube's own frame; every sky tap goes through it,
// so a missed ray lands on the same oriented sky the main pass shows.
#define SKY_DIR(d) float3(dot(rt_params.sky_rot[0].xyz, (d)), \
                          dot(rt_params.sky_rot[1].xyz, (d)), \
                          dot(rt_params.sky_rot[2].xyz, (d)))

// ---- Resource bindings ----

#ifdef CN_BACKEND_DIRECTX

ConstantBuffer<RtParams> rt_params : register(b0);
RaytracingAccelerationStructure scene_tlas : register(t0);
// Raw vertex / index SRVs: DirectX binds these as byte-address views.
ByteAddressBuffer verts : register(t1);
ByteAddressBuffer indices : register(t2);
StructuredBuffer<RtGeomEntry> geom : register(t3);
ByteAddressBuffer sverts : register(t8);
ByteAddressBuffer sidx : register(t9);

Texture2D<float4> scene_tex : register(t4);
Texture2D<float4> gbuffer : register(t5);
Texture2D<float4> rough_tex : register(t6);
TextureCube<float4> prefilter : register(t7);
// Static samplers from the root signature, not one per source, so the
// per-source names the other branch declares are aliases here and a sample
// site reads the same in both.
SamplerState screen_sampler : register(s0);
SamplerState cube_sampler : register(s1);
#define scene_tex_sampler screen_sampler
#define gbuffer_sampler screen_sampler
#define rough_tex_sampler screen_sampler
#define prefilter_sampler cube_sampler

#else

[[vk::binding(0, 0)]] ConstantBuffer<RtParams> rt_params : register(b0);
[[vk::binding(1, 0)]] RaytracingAccelerationStructure scene_tlas : register(t4);
[[vk::binding(2, 0)]] StructuredBuffer<RtGeomEntry> geom : register(t3);
// The shared static vertex stream (14 floats / 56 B stride) + its u32 indices.
[[vk::binding(3, 0)]] StructuredBuffer<float> verts : register(t1);
[[vk::binding(4, 0)]] StructuredBuffer<uint> indices : register(t2);
// The deformed (posed) skinned vertex buffer, in the same 14-float layout the
// skin kernel writes, + the skinned index buffer (two indices per uint).
// Both bind a 1-element dummy in a scene with no skinned geometry, so the
// binding stays valid even though the skinned branch is never taken there.
[[vk::binding(9, 0)]] StructuredBuffer<float> sverts : register(t5);
[[vk::binding(10, 0)]] StructuredBuffer<uint> sidx : register(t6);

// Screen-space inputs reused from the SSR resolve, each a texture and a sampler
// of its own. Their `t` registers are Metal texture indices -- scene(0),
// gbuffer(1), roughness(2), prefilter(3) -- a different namespace from the
// buffers above, which is why the numbers repeat without colliding.
[[vk::binding(5, 0)]] Texture2D<float4> scene_tex : register(t0);
[[vk::binding(11, 0)]] SamplerState scene_tex_sampler : register(s0);
[[vk::binding(6, 0)]] Texture2D<float4> gbuffer : register(t1);
[[vk::binding(12, 0)]] SamplerState gbuffer_sampler : register(s1);
[[vk::binding(7, 0)]] Texture2D<float4> rough_tex : register(t2);
[[vk::binding(13, 0)]] SamplerState rough_tex_sampler : register(s2);
#ifdef CN_BACKEND_VULKAN
// The global set's own prefilter cube and cube sampler, the one the probe cubes
// are read through as well.
[[vk::binding(5, 1)]] TextureCube<float4> prefilter : register(t3);
[[vk::binding(19, 1)]] SamplerState prefilter_sampler : register(s3);
#else
[[vk::binding(8, 0)]] TextureCube<float4> prefilter : register(t3);
[[vk::binding(14, 0)]] SamplerState prefilter_sampler : register(s3);
#endif

#endif

float3 prefilter_level(float3 dir, float lod)
{
    return prefilter.SampleLevel(prefilter_sampler, SKY_DIR(dir), lod).rgb;
}

// The forward global set, bound here only for its reflection probes: a ray
// that escapes the scene falls back to the local probe capture instead of the
// foreign sky cube, blending the probes the main camera's cluster grid bins at
// the pixel's surface point. On Vulkan every binding is the global set's own,
// the cube sampler included.
#ifdef CN_BACKEND_DIRECTX
ConstantBuffer<ProbeSet> probe_set : register(b4);
TextureCubeArray<float4> probe_cubes : register(t10);
SamplerState probe_cube_sampler : register(s3);
StructuredBuffer<ProbeUniforms> probe_records : register(t11);
ConstantBuffer<ClusterParams> cluster : register(b5);
StructuredBuffer<uint> cluster_list : register(t12);
#else
[[vk::binding(7, 1)]] ConstantBuffer<ProbeSet> probe_set : register(b8);
[[vk::binding(8, 1)]] TextureCubeArray<float4> probe_cubes : register(t4);
[[vk::binding(19, 1)]] SamplerState probe_cube_sampler : register(s4);
[[vk::binding(17, 1)]] StructuredBuffer<ProbeUniforms> probe_records : register(t11);
[[vk::binding(10, 1)]] ConstantBuffer<ClusterParams> cluster : register(b9);
[[vk::binding(11, 1)]] StructuredBuffer<uint> cluster_list : register(t10);
#endif
#define PROBE_SET probe_set
#define PROBE_RECORDS probe_records
#define CLUSTER cluster
#define CLUSTER_LIST cluster_list

#ifdef RT_TEXTURED
// The bindless albedo / normal / emissive pool, in whichever form its host can
// bind: the bindless main pass's Metal argument buffer, bound at the pool's
// offset so the unsized array starts at its first texture (its sampler is bound
// alongside rather than written into the buffer), the Vulkan image array on the
// set that pass uses, read through the global set's linear sampler as the main
// pass reads it, or an unbounded DXIL array in space 1.
#if defined(CN_BACKEND_METAL)
[[vk::binding(0, 3)]] [[cn::metal_argument_buffer(7)]]
Texture2D<float4> tex_pool[] : register(t0, space3);
[[vk::binding(15, 0)]] SamplerState pool_sampler : register(s5);
#elif defined(CN_BACKEND_DIRECTX)
Texture2D<float4> tex_pool[] : register(t0, space1);
SamplerState pool_sampler : register(s2);
#else
[[vk::binding(1, 2)]] Texture2D<float4> tex_pool[] : register(t0, space1);
[[vk::binding(20, 1)]] SamplerState pool_sampler : register(s2);
#endif
#endif

// Surfaces rougher than REFLECTION_ROUGHNESS_CUT get no reflection; glossiness
// ramps in below it. Locked to concinnity_core::render::post::ssr::settings::REFLECTION_ROUGHNESS_CUT
// by unit test so the SSR, RT, and composite gates agree.
static const float REFLECTION_ROUGHNESS_CUT = 0.6;

{PROBE_COMMON}

{RT_TRACE}

// Rebuild a view-space position from a UV and its linear (view-space) depth.
// Matches ssr_view_pos in the SSR resolve.
float3 rt_view_pos(float2 uv, float depth, float tan_y, float aspect)
{
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return float3(ndc.x * tan_y * aspect, ndc.y * tan_y, -1.0) * depth;
}

[shader("pixel")]
float4 rt_reflections_fragment([[vk::location(0)]] float2 uv : TEXCOORD0,
                               float4 frag_pos : SV_Position) : SV_Target
{
    // At a reduced trace resolution each output texel stands for a block of
    // G-buffer pixels; trace from exactly one of them so the surface point is
    // real rather than a blend across an edge.
    uv = reflection_source_uv(uv, texture_size(gbuffer));
    float3 base = scene_tex.Sample(scene_tex_sampler, uv).rgb;
    float4 g = gbuffer.Sample(gbuffer_sampler, uv);
    float depth = g.a;
    // Background / sky, or a non-reflecting (too-rough) surface: weight 0 so the
    // reflection composite keeps the scene there. The pass writes reflected
    // radiance (.rgb) + composite weight (.a), not a blended color.
    if (depth <= 0.0)
    {
        return float4(base, 0.0);
    }

    float roughness = rough_tex.Sample(rough_tex_sampler, uv).r;
    float gloss = saturate((REFLECTION_ROUGHNESS_CUT - roughness) / REFLECTION_ROUGHNESS_CUT);
    if (gloss <= 0.0)
    {
        return float4(base, 0.0);
    }

    float3 nv = normalize(g.xyz);
    float3 pv = rt_view_pos(uv, depth, rt_params.tan_half_fov_y, rt_params.aspect);
    float3 pw = mul(rt_params.inv_view, float4(pv, 1.0)).xyz;
    float3 nw = normalize(mul((float3x3)rt_params.inv_view, nv));
    float3 v = normalize(rt_params.cam_pos.xyz - pw);

    float3 dir = reflect(-v, nw);
    bool ibl = rt_params.prefilter_mip_count > 0.5;
    float max_mip = rt_params.prefilter_mip_count - 1.0;
    float ndv = saturate(dot(nw, v));
    float fresnel = RT_F0 + (1.0 - RT_F0) * pow(1.0 - ndv, 5.0);
    float weight = saturate(fresnel * gloss * rt_params.intensity);

    float3 reflected;
    // Origin nudged off the surface along the normal so the trace cannot
    // self-intersect the pixel's own triangle.
    if (!rt_trace_reflection(pw + nw * 0.01, dir, ibl, max_mip, reflected))
    {
        // The ray escaped the scene: the local reflection probe (box-parallax,
        // blended across covering probes) when one is baked, else the IBL
        // prefilter sky, else the base shading.
        if (probe_set.count > 0u)
        {
            reflected = probe_mask_specular(probe_mask_at(uv, pw, frag_pos.xy), pw, dir,
                                            probe_lod(roughness));
        }
        else
        {
            reflected = ibl ? prefilter_level(dir, roughness * max_mip) : base;
        }
    }

    return float4(reflected, weight);
}
