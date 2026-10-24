// GPU particle system: the render half, single source for every backend.
//
// The renderer keeps one persistent particle pool per emitter, simulated by
// `particle_simulate.hlsl` each frame and rasterized by this vertex / fragment
// pair as camera-facing billboards. `particle_vertex` is invoked with 4
// vertices per particle and the pool size as the instance count: each instance
// reads its `Particle`, derives a camera-facing axis pair from the bound view,
// and emits one corner of the quad. Dead particles emit a degenerate quad
// behind the near plane so they cost nothing past the vertex stage.
//
// The composited color is alpha-blended into the resolved HDR target by the
// pipeline's blend state (Src.A * Src + (1 - Src.A) * Dst), the same envelope
// the projected-decal pass uses. A sprite carries no motion of its own, so it
// also marks the reactive mask (target 1, max-blended), telling the temporal
// passes to trust this frame where it draws.
//
// `Particle` and `ParticleParams` arrive from the shared PARTICLE_TYPES
// fragment the simulation kernel splices too, so the pool written there and the
// pool read here have one declaration.
//
// CN_BACKEND_DIRECTX is the one constant shape that has to move registers: the
// render root signature in `directx/particle.rs` puts the view at b0, the
// per-emitter params at b1 and the albedo at t1, where the Metal buffer indices
// are 1 and 2 and the albedo is texture(0). Everywhere else one register number serves
// both, since the two agree.
//
// The pass binds no depth attachment, so the fragment tests the scene depth
// itself. USE_MSAA is a host difference: Vulkan and DirectX read the
// multisampled main depth while Metal reads the resolved copy.
//
// PARTICLE_LIT compiles the pair for a lit emitter: each particle is lit by
// the sun and the sky as a small cloud of droplets or dust, brightest seen
// toward the sun, and fades out where it nears the surface behind it, over
// its own size, rather than cutting into it.

#ifndef USE_MSAA
#define USE_MSAA 0
#endif

{DEPTH_CONVENTION}

{PARTICLE_TYPES}

{REACTIVE_MASK}

// Per-frame view inputs to the render pass, 144 B. Mirrors `ParticleView` in
// each backend's uniforms module. The vectors are float4 for the reason the
// params' pairs are: MSL sizes a constant-buffer float3 at 16 bytes.
struct ParticleView
{
    float4x4 vp;
    // xyz = world-space camera right, the first billboard axis; w = the scale
    // the HDR chain is written at.
    float4 cam_right;
    // xyz = world-space camera up, the second billboard axis.
    float4 cam_up;
    // xyz = unit direction toward the sun.
    float4 sun_dir;
    // rgb = the light a white surface facing the sun reflects, and the light
    // one under the open sky reflects from it, where the camera is.
    float4 sun_light;
    float4 sky_light;
};

// How much more light a lit particle sends on seen straight toward the sun
// than across it, and how tightly that glow gathers around the sun.
static const float PARTICLE_FORWARD_GLOW = 1.5;
static const float PARTICLE_FORWARD_POWER = 4.0;
static const float PARTICLE_SIDE_LIGHT = 0.4;

// The per-emitter pool, written by the simulation kernel and read here. Vulkan
// binds it as a read-only SSBO in the per-emitter set; Metal binds the same
// buffer at vertex buffer(0), which is what `register(t0)` names on a buffer
// resource.
[[vk::binding(0, 1)]] StructuredBuffer<Particle> pool : register(t0);

#ifdef CN_BACKEND_DIRECTX
ConstantBuffer<ParticleView> view : register(b0);
ConstantBuffer<ParticleParams> params : register(b1);
#define ALBEDO_TEXTURE_REGISTER t1
#else
[[vk::binding(0, 0)]] ConstantBuffer<ParticleView> view : register(b1);
[[vk::push_constant]] ConstantBuffer<ParticleParams> params : register(b2);
#define ALBEDO_TEXTURE_REGISTER t0
#endif

// The emitter's albedo, which Vulkan binds beside its sampler in the
// per-emitter set.
[[vk::binding(1, 1)]] Texture2D<float4> albedo : register(ALBEDO_TEXTURE_REGISTER);
[[vk::binding(2, 1)]] SamplerState albedo_sampler : register(s0);

// Scene depth under the pixel. Vulkan puts it beside the view in the per-frame
// set; DirectX and Metal both bind it at t2 / texture(2).
#if USE_MSAA
[[vk::binding(1, 0)]] Texture2DMS<float> scene_depth : register(t2);
#else
[[vk::binding(1, 0)]] Texture2D<float> scene_depth : register(t2);
#endif

// The varyings lead deliberately: D3D packs a stage signature in declaration
// order and links the two stages by matching semantic *and* register, so a
// fragment reading TEXCOORD0 must find it where the vertex handed it out.
// Metal and Vulkan are order-blind here (every varying carries an attribute or
// an explicit location).
struct ParticleVertexOut
{
    [[vk::location(0)]] float2 uv : TEXCOORD0;
    [[vk::location(1)]] float4 color : TEXCOORD1;
    // Constant across the quad, so interpolating it would only cost bandwidth.
    [[vk::location(2)]] nointerpolation float discard_flag : TEXCOORD2;
    // The view block's pre-exposure, handed on because only the vertex stage
    // binds the view.
    [[vk::location(3)]] nointerpolation float pre_exposure : TEXCOORD3;
#ifdef PARTICLE_LIT
    // x = the particle's distance along the view, y = how far behind it the
    // scene must lie for it to show whole.
    [[vk::location(4)]] float2 soft : TEXCOORD4;
#endif
    float4 position : SV_Position;
};

[shader("vertex")]
ParticleVertexOut particle_vertex(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    ParticleVertexOut o;
    Particle pt = pool[iid];

    // Dead slot -> a degenerate quad clipped behind the near plane. The
    // fragment also discards on `discard_flag`, so any pixel that still
    // rasterizes (the numerical edge case at exactly w = 0) draws nothing.
    if (pt.velocity_lifetime.w <= 0.0)
    {
        o.position = float4(0.0, 0.0, -2.0, 1.0);
        o.uv = float2(0.0, 0.0);
        o.color = float4(0.0, 0.0, 0.0, 0.0);
        o.discard_flag = 1.0;
        o.pre_exposure = view.cam_right.w;
#ifdef PARTICLE_LIT
        o.soft = float2(1.0, 1.0);
#endif
        return o;
    }

    float t = clamp(pt.position_age.w / pt.velocity_lifetime.w, 0.0, 1.0);
    float size = lerp(params.size_start, params.size_end, t);
    float4 color = lerp(params.color_start, params.color_end, t);

    // 0..3 -> (-1,-1), (+1,-1), (-1,+1), (+1,+1) for a triangle strip.
    float2 corner = float2(
        (vid & 1u) == 0u ? -1.0 : 1.0,
        (vid & 2u) == 0u ? -1.0 : 1.0);

    float3 right = view.cam_right.xyz * (corner.x * 0.5 * size);
    float3 up = view.cam_up.xyz * (corner.y * 0.5 * size);
    float3 world = pt.position_age.xyz + right + up;
    o.position = mul(view.vp, float4(world, 1.0));

    // 0..1 in each axis; V is flipped at sample time to match the rest of the
    // engine's textures (V = 0 at the top of the image).
    o.uv = corner * 0.5 + 0.5;
#ifdef PARTICLE_LIT
    float3 forward = cross(view.cam_up.xyz, view.cam_right.xyz);
    float toward = pow(saturate(dot(forward, view.sun_dir.xyz)), PARTICLE_FORWARD_POWER);
    float3 light = view.sun_light.rgb * (PARTICLE_SIDE_LIGHT + PARTICLE_FORWARD_GLOW * toward) + view.sky_light.rgb;
    color.rgb *= light;
    o.soft = float2(o.position.w, 0.5 * size);
#endif
    o.color = color;
    o.discard_flag = 0.0;
    o.pre_exposure = view.cam_right.w;
    return o;
}

float particle_scene_depth(int2 pixel)
{
#if USE_MSAA
    return scene_depth.Load(pixel, 0);
#else
    return scene_depth.Load(int3(pixel, 0));
#endif
}

struct ParticleFragmentOut
{
    float4 color    : SV_Target0;
    float  reactive : SV_Target1;
};

[shader("pixel")]
ParticleFragmentOut particle_fragment(ParticleVertexOut i)
{
    if (i.discard_flag > 0.5)
    {
        discard;
    }
    // Manual depth test: a sprite behind opaque scene geometry draws nothing.
    float behind = particle_scene_depth(int2(i.position.xy));
    if (depth_behind(i.position.z, behind))
    {
        discard;
    }
    float2 uv = float2(i.uv.x, 1.0 - i.uv.y);
    float4 sampled = albedo.Sample(albedo_sampler, uv);
    ParticleFragmentOut o;
    float4 color = float4(sampled.rgb * i.color.rgb, sampled.a * i.color.a);
#ifdef PARTICLE_LIT
    // Device depth falls as the reciprocal of the distance along the view,
    // so the scene's distance is the particle's scaled by their depths' ratio.
    if (depth_is_written(behind))
    {
        float gap = i.soft.x * (i.position.z / behind - 1.0);
        color.a *= saturate(gap / max(i.soft.y, 1e-4));
    }
#endif
    // The light a sprite adds counts as much as its coverage: a bright spark
    // fading out is nearly transparent yet still the brightest thing there.
    o.reactive = reactive_write(max(color.a, color.a * reactive_luminance(color.rgb)));
    o.color = float4(color.rgb * i.pre_exposure, color.a);
    return o;
}
