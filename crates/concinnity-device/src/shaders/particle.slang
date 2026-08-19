// GPU particle system: the render half, single source for every backend.
//
// The renderer keeps one persistent particle pool per emitter, simulated by
// `particle_simulate.slang` each frame and rasterised by this vertex / fragment
// pair as camera-facing billboards. `particle_vertex` is invoked with 4
// vertices per particle and the pool size as the instance count: each instance
// reads its `Particle`, derives a camera-facing axis pair from the bound view,
// and emits one corner of the quad. Dead particles emit a degenerate quad
// behind the near plane so they cost nothing past the vertex stage.
//
// The composited colour is alpha-blended into the resolved HDR target by the
// pipeline's blend state (Src.A * Src + (1 - Src.A) * Dst), the same envelope
// the projected-decal pass uses.
//
// `Particle` and `ParticleParams` arrive from the shared PARTICLE_TYPES
// fragment the simulation kernel splices too, so the pool written there and the
// pool read here have one declaration.
//
// METAL_BINDINGS selects the Metal host's constant shape, which is a host
// difference rather than a target one: Vulkan pushes the per-emitter params and
// binds the view as a UBO, while the Metal encoder writes both with setBytes,
// the view at buffer(1) and the params at buffer(2). Everything else carries a
// `[[vk::binding]]` and a `register()` on one declaration, so the Vulkan
// descriptor sets and the Metal buffer indices both reproduce what the hosts
// already bind.
//
// DXIL_ABI is the third constant shape, and the only one that has to move a
// register: the render root signature in `directx/particle.rs` puts the view at
// b0 and the params at b1, where the Metal buffer indices are 1 and 2. The pool
// and the albedo need no such block -- t0 and t1 / s0 are what that root
// signature binds and what declaration order already yields.

{PARTICLE_TYPES}

// Per-frame view inputs to the render pass, 96 B. Mirrors `ParticleView` in
// each backend's uniforms module. The two axis vectors are float4 for the
// reason the params' pairs are: MSL sizes a constant-buffer float3 at 16 bytes.
struct ParticleView
{
    float4x4 vp;
    // xyz = world-space camera right, the first billboard axis.
    float4 cam_right;
    // xyz = world-space camera up, the second billboard axis.
    float4 cam_up;
};

// The per-emitter pool, written by the simulation kernel and read here. Vulkan
// binds it as a read-only SSBO in the per-emitter set; Metal binds the same
// buffer at vertex buffer(0).
[[vk::binding(0, 1)]] StructuredBuffer<Particle> pool : register(t0);

#ifdef DXIL_ABI
ConstantBuffer<ParticleView> view : register(b0);
ConstantBuffer<ParticleParams> params : register(b1);
#else
[[vk::binding(0, 0)]] ConstantBuffer<ParticleView> view : register(b1);

#ifdef METAL_BINDINGS
ConstantBuffer<ParticleParams> params : register(b2);
#else
[[vk::push_constant]] ConstantBuffer<ParticleParams> params;
#endif
#endif

// The emitter's albedo. A combined sampler lowers to a texture(0) + sampler(0)
// pair on Metal, which is what that host binds.
[[vk::binding(1, 1)]] Sampler2D<float4> albedo;

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
    float4 position : SV_Position;
};

[shader("vertex")]
ParticleVertexOut particle_vertex(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    ParticleVertexOut o;
    Particle pt = pool[iid];

    // Dead slot -> a degenerate quad clipped behind the near plane. The
    // fragment also discards on `discard_flag`, so any pixel that still
    // rasterises (the numerical edge case at exactly w = 0) draws nothing.
    if (pt.lifetime <= 0.0)
    {
        o.position = float4(0.0, 0.0, -2.0, 1.0);
        o.uv = float2(0.0, 0.0);
        o.color = float4(0.0, 0.0, 0.0, 0.0);
        o.discard_flag = 1.0;
        return o;
    }

    float t = clamp(pt.age / pt.lifetime, 0.0, 1.0);
    float size = lerp(params.size_start, params.size_end, t);
    float4 color = lerp(params.color_start, params.color_end, t);

    // 0..3 -> (-1,-1), (+1,-1), (-1,+1), (+1,+1) for a triangle strip.
    float2 corner = float2(
        (vid & 1u) == 0u ? -1.0 : 1.0,
        (vid & 2u) == 0u ? -1.0 : 1.0);

    float3 right = view.cam_right.xyz * (corner.x * 0.5 * size);
    float3 up = view.cam_up.xyz * (corner.y * 0.5 * size);
    float3 world = pt.position + right + up;
    o.position = mul(view.vp, float4(world, 1.0));

    // 0..1 in each axis; V is flipped at sample time to match the rest of the
    // engine's textures (V = 0 at the top of the image).
    o.uv = corner * 0.5 + 0.5;
    o.color = color;
    o.discard_flag = 0.0;
    return o;
}

[shader("fragment")]
float4 particle_fragment(ParticleVertexOut i) : SV_Target
{
    if (i.discard_flag > 0.5)
    {
        discard;
    }
    float2 uv = float2(i.uv.x, 1.0 - i.uv.y);
    float4 sampled = albedo.Sample(uv);
    return float4(sampled.rgb * i.color.rgb, sampled.a * i.color.a);
}
