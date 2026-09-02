// Screen-space reflection resolve: a fullscreen ray-march over the pre-pass
// G-buffer that writes reflected radiance (.rgb) and the composite weight (.a)
// the reflection blur + composite then blends over the scene. Single source for
// every backend; pairs with `fullscreen_vertex` in fullscreen.slang.
//
// Compile with -DMAX_PROBES=<n> (the reflection-probe array length the device
// binds). Vulkan gets set 0 bindings 0-3 plus the forward global set's probe
// bindings (7 + 8) at set 1, which is how the hand-written GLSL bound them;
// Metal gets texture(0..3) for the screen sources and the prefilter cube,
// texture(4..4+MAX_PROBES) for the probe cubes, the push constant at buffer(0)
// and the probe set at buffer(1). DXIL gets t0..t3 / s0..s3 for the screen
// sources, b0 for the params and b1 for the probe set, and (under
// SPLIT_PROBE_SAMPLER) the probe cube array at t4.. with its one sampler at s4.

#ifndef MAX_PROBES
#define MAX_PROBES 8
#endif

{POST_COMMON}

// Layout matches `SsrParams` in render_types.rs (144 B).
struct SsrParams
{
    float intensity;
    float max_distance;
    float tan_half_fov_y;
    float aspect;
    float stride;
    float thickness;
    // IBL prefilter cubemap mip count; 0 means no EnvironmentMap is bound and
    // the cube fallback is skipped (missed rays keep the base shading).
    float prefilter_mip_count;
    float _pad;
    // Camera-to-world transform (the rigid inverse of the view matrix): its 3x3
    // turns the view-space reflection ray into the world-space direction the
    // cubemap is sampled with, and its translation column lets the resolve
    // rebuild the world-space surface position a probe box-projects against.
    float4x4 inv_view;
    // Rows of the rotation from world space into the sky cube's baked frame;
    // identity when the sky does not turn.
    float4 sky_rot[3];
};

// A world direction in the sky cube's own frame; the sky fallback goes through
// it so a resolved reflection matches the sky the main pass shows.
#define SKY_DIR(d) float3(dot(params.sky_rot[0].xyz, (d)), \
                          dot(params.sky_rot[1].xyz, (d)), \
                          dot(params.sky_rot[2].xyz, (d)))

{PROBE_TYPES}

// The push constant leads so it lands on Metal's buffer(0), where every post
// pass puts its params and where the host writes it.
[[vk::push_constant]]
ConstantBuffer<SsrParams> params;

[[vk::binding(0, 0)]] Sampler2D<float4> scene;
[[vk::binding(1, 0)]] Sampler2D<float4> gbuffer;
[[vk::binding(2, 0)]] Sampler2D<float4> rough_tex;
[[vk::binding(3, 0)]] SamplerCube<float4> prefilter;

// The forward global set, bound here only for its reflection-probe count +
// per-probe parallax boxes + cube array: a screen-space ray that escapes the
// frame falls back to the local probe capture instead of the foreign sky cube.
[[vk::binding(7, 1)]] ConstantBuffer<ProbeSet> probe_set;
#define PROBE_SET probe_set

// The probe cube array, in whichever form its host can bind.
//
// A top-level combined `SamplerCube` array lowers to a texture array plus a
// sampler array, and D3D12 binds a shader sampler array only through a
// descriptor table -- a static sampler covers a single register, so twelve of
// them still leave s4..s11 "not fully bound". SPLIT_PROBE_SAMPLER therefore
// declares the array separated, with one sampler the root signature hands out
// statically, the same shape the bindless main pass uses for the same array.
// Metal and Vulkan keep the combined form: it lands on texture(4..11) /
// sampler(4..11) and on the forward global set's binding 8 respectively, which
// is what both hosts already bind.
#if defined(SPLIT_PROBE_SAMPLER)
TextureCube<float4> probe_cubes[MAX_PROBES];
SamplerState probe_cube_sampler;

float3 probe_cube_sample_bias(uint i, float3 dir, float lod)
{
    return probe_cubes[i].SampleBias(probe_cube_sampler, dir, lod).rgb;
}
#else
[[vk::binding(8, 1)]] SamplerCube<float4> probe_cubes[MAX_PROBES];

float3 probe_cube_sample_bias(uint i, float3 dir, float lod)
{
    return probe_cubes[i].SampleBias(dir, lod).rgb;
}
#endif

static const int   SSR_MAX_STEPS = 48;
static const int   SSR_REFINE    = 5;
// Surfaces rougher than REFLECTION_ROUGHNESS_CUT get no SSR; glossiness ramps
// in below it. Locked to concinnity_core::gfx::ssr::REFLECTION_ROUGHNESS_CUT by
// unit test so the SSR, RT, and composite passes can never disagree on it.
static const float REFLECTION_ROUGHNESS_CUT = 0.6;
// Dielectric base reflectance (water, glass, polished stone) for the Fresnel.
static const float SSR_F0        = 0.04;
// UV margin over which a hit near the screen border fades out.
static const float SSR_EDGE_FADE = 0.12;

// Rebuild a view-space position from a UV and its linear (view-space) depth.
// The inverse of ssr_project; matches ssao_view_pos in the SSAO kernel.
float3 ssr_view_pos(float2 uv, float depth, float tan_y, float aspect)
{
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return float3(ndc.x * tan_y * aspect, ndc.y * tan_y, -1.0) * depth;
}

// Project a view-space point (z < 0, in front of the camera) to a screen UV.
float2 ssr_project(float3 q, float tan_y, float aspect)
{
    float inv = 1.0 / max(-q.z, 1e-4);
    float2 ndc = float2(q.x * inv / (tan_y * aspect), q.y * inv / tan_y);
    return float2(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
}

{PROBE_COMMON}

[shader("fragment")]
float4 ssr_resolve_fragment([[vk::location(0)]] float2 uv : TEXCOORD0) : SV_Target
{
    float3 base = scene.Sample(uv).rgb;
    float4 c = gbuffer.Sample(uv);
    float depth = c.a;
    // Background / sky, or a non-reflecting (too-rough) surface: weight 0 so the
    // reflection composite keeps the scene there. The resolve does not blend
    // inline; it writes reflected radiance (.rgb) + composite weight (.a).
    if (depth <= 0.0)
    {
        return float4(base, 0.0);
    }

    float roughness = rough_tex.Sample(uv).r;
    // Glossy surfaces reflect sharply; rough ones get nothing.
    float gloss = saturate((REFLECTION_ROUGHNESS_CUT - roughness) / REFLECTION_ROUGHNESS_CUT);
    if (gloss <= 0.0)
    {
        return float4(base, 0.0);
    }

    float3 n = normalize(c.xyz);
    float3 p = ssr_view_pos(uv, depth, params.tan_half_fov_y, params.aspect);
    float3 v = normalize(-p);                       // p in view space, camera at origin
    float3 r_dir = reflect(-v, n);                  // reflected ray direction

    // Environment fallback for a missed (or screen-edge) ray, in the reflected
    // direction at a roughness-keyed mip so a rougher surface reflects a
    // blurrier environment. With a baked reflection probe this is the local
    // scene capture (box-projected + blended across covering probes), the same
    // source the forward IBL specular term uses, rather than the foreign sky
    // HDR; otherwise it is the IBL prefilter cube. With no EnvironmentMap bound
    // there is nothing to fall back to, so missed rays keep the base shading.
    bool ibl = params.prefilter_mip_count > 0.5;
    float3 env = base;
    if (ibl)
    {
        float3 r_world = mul((float3x3)params.inv_view, r_dir);
        float lod = roughness * (params.prefilter_mip_count - 1.0);
        if (probe_set.count > 0u)
        {
            // The full inv_view (its translation column carries the camera
            // position) lifts the view-space surface point p to world space,
            // which the probe box-projection needs.
            float3 world_pos = mul(params.inv_view, float4(p, 1.0)).xyz;
            env = probe_set_specular(world_pos, r_world, lod);
        }
        else
        {
            env = prefilter.SampleLevel(SKY_DIR(r_world), lod).rgb;
        }
    }

    float3 step_v = r_dir * params.stride;
    float3 q = p;
    bool hit = false;
    float2 hit_uv = uv;
    int steps_taken = SSR_MAX_STEPS;
    for (int i = 0; i < SSR_MAX_STEPS; i++)
    {
        q += step_v;
        if (q.z >= 0.0) { steps_taken = i; break; }  // crossed the camera plane
        float2 march_uv = ssr_project(q, params.tan_half_fov_y, params.aspect);
        if (march_uv.x < 0.0 || march_uv.x > 1.0 || march_uv.y < 0.0 || march_uv.y > 1.0)
        {
            steps_taken = i;
            break;
        }
        float scene_depth = gbuffer.Sample(march_uv).a;
        if (scene_depth <= 0.0) continue;            // sky here - keep marching
        float diff = (-q.z) - scene_depth;           // > 0: ray is behind the surface
        if (diff > 0.0 && diff < params.thickness)
        {
            // Binary-search refine between the last two samples.
            float3 lo = q - step_v;
            float3 hi = q;
            for (int r = 0; r < SSR_REFINE; r++)
            {
                float3 mid = (lo + hi) * 0.5;
                float2 muv = ssr_project(mid, params.tan_half_fov_y, params.aspect);
                float sd = gbuffer.Sample(muv).a;
                if (sd > 0.0 && (-mid.z) - sd > 0.0) hi = mid; else lo = mid;
            }
            hit_uv = ssr_project(hi, params.tan_half_fov_y, params.aspect);
            hit = true;
            steps_taken = i;
            break;
        }
    }

    // The reflected colour: the screen-space hit (a single sharp tap - the
    // reflection composite blurs it by roughness), or the environment cube when
    // the ray missed. A hit near the screen border or at the end of its march
    // fades toward the environment rather than snapping flat to the base
    // shading.
    float3 reflected;
    if (hit)
    {
        float3 hit_color = scene.Sample(hit_uv).rgb;
        float2 e = smoothstep(float2(0.0), float2(SSR_EDGE_FADE), hit_uv)
                 * smoothstep(float2(0.0), float2(SSR_EDGE_FADE), float2(1.0) - hit_uv);
        float edge = e.x * e.y;
        float march = float(steps_taken) / float(SSR_MAX_STEPS);
        float dist_fade = 1.0 - smoothstep(0.7, 1.0, march);
        reflected = lerp(env, hit_color, edge * dist_fade);
    }
    else
    {
        reflected = env;
    }

    float ndv = saturate(dot(n, v));
    float fresnel = SSR_F0 + (1.0 - SSR_F0) * pow(1.0 - ndv, 5.0);
    float w = saturate(fresnel * gloss * params.intensity);
    // Reflected radiance (.rgb) + composite weight (.a). The reflection
    // composite blurs this by surface roughness and blends it over the scene.
    return float4(reflected, w);
}
