// GPU particle simulation kernel: single source for every backend.
//
// One thread per slot in the per-emitter pool. Each thread ages and integrates
// whatever particle currently occupies its slot; if the age reaches the
// lifetime the slot is marked dead. A dead slot then tries to consume one unit
// of the frame's spawn budget (atomically) and respawns with a fresh velocity
// sampled inside a cone of half-angle acos(spread_cos) around `direction`.
//
// The render half of the particle system (the billboard vertex + fragment
// pair) is still per-backend, so `ParticleParams` is declared there too; the
// two declarations are locked to the same Rust struct by
// `particle_params_layout_matches_msl` in render_types.rs.

// One particle slot, 32 B. Matches `GpuParticle` in each backend's particle
// module. `float3` packs to 12 bytes inside a structured buffer on every
// target (unlike a constant buffer, where MSL pads it to 16), so the
// (vec3, float) pairs need no rewriting here.
struct Particle
{
    float3 position;
    float age;
    float3 velocity;
    float lifetime;
};

// Per-frame emitter uniform, 112 B. Mirrors `ParticleParams` in
// gfx/render_types.rs. The three (vec3, scalar) pairs are spelled as float4:
// MSL sizes a constant-buffer float3 at 16 bytes, so a literal transcription
// would push every following field four bytes late on Metal alone.
struct ParticleParams
{
    // xyz = world-space spawn origin, w = cosine of the emission cone's
    // half-angle (1 = straight jet, -1 = full sphere).
    float4 position_spread;
    // xyz = unit-length mean emission direction, w = minimum spawn speed.
    float4 direction_speed_min;
    // xyz = constant per-frame acceleration, w = maximum spawn speed.
    float4 gravity_speed_max;
    float4 color_start;
    float4 color_end;
    float lifetime_min;
    float lifetime_max;
    float size_start;
    float size_end;
    float dt;
    // Particles the kernel may emit this frame. Threads race for slots through
    // the counter below, so only this many succeed.
    uint spawn_budget;
    uint random_seed;
    uint max_particles;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<Particle> pool : register(u0);
// Remaining spawn budget for this dispatch, as a single counter at element 0.
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> spawn_counter : register(u1);

// A host difference, not a target one: Vulkan pushes the params and DirectX
// takes them as root constants at b0 (where a bare push constant lands there),
// while the Metal encoder writes them to buffer(2).
#ifdef METAL_BINDINGS
ConstantBuffer<ParticleParams> params : register(b2);
#else
[[vk::push_constant]] ConstantBuffer<ParticleParams> params;
#endif

// Cheap fixed-point hash to a unit float in [0, 1). Mutates `state` so a thread
// needing several uncorrelated samples advances it between calls.
float prng(inout uint state)
{
    state = state * 1664525u + 1013904223u;
    // Top 24 bits of the hash become the mantissa.
    return float(state >> 8) * (1.0 / 16777216.0);
}

// Sample a unit vector inside a cone of half-angle acos(cone_cos) centred on
// `axis`. The cap is uniformly sampled in solid angle, so the spawn cloud has
// no axial bunching.
float3 sample_cone(inout uint rng, float3 axis, float cone_cos)
{
    // `u` picks a polar angle whose cosine is uniform in [cone_cos, 1].
    float u = lerp(cone_cos, 1.0, prng(rng));
    float r = sqrt(max(1.0 - u * u, 0.0));
    float phi = prng(rng) * 6.2831853;
    float3 local = float3(r * cos(phi), r * sin(phi), u);

    // Any orthonormal basis around `axis`. Picking the world axis least
    // parallel to it keeps the cross product well conditioned.
    float3 up = abs(axis.y) < 0.9 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 t = normalize(cross(up, axis));
    float3 b = cross(axis, t);
    return normalize(local.x * t + local.y * b + local.z * axis);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void particle_simulate(uint3 gid : SV_DispatchThreadID)
{
    uint id = gid.x;
    if (id >= params.max_particles)
    {
        return;
    }
    Particle pt = pool[id];

    // Age the particle in this slot, if any. A lifetime of 0 flags a dead slot.
    if (pt.lifetime > 0.0)
    {
        pt.age += params.dt;
        if (pt.age >= pt.lifetime)
        {
            pt.lifetime = 0.0;
        }
        else
        {
            pt.velocity = pt.velocity + params.gravity_speed_max.xyz * params.dt;
            pt.position = pt.position + pt.velocity * params.dt;
        }
    }

    // A dead slot claims one unit of the remaining budget. Adding -1 in two's
    // complement returns the pre-subtract value, so only threads that observed
    // a positive remaining count spawn; threads racing past zero see 0 or a
    // wrapped very large unsigned number.
    if (pt.lifetime == 0.0 && params.spawn_budget > 0u)
    {
        uint claimed;
        InterlockedAdd(spawn_counter[0], 0xFFFFFFFFu, claimed);
        if (claimed > 0u && claimed <= params.spawn_budget)
        {
            uint rng = (id * 747796405u) ^ (params.random_seed * 2891336453u);
            // Warm the RNG so adjacent threads decorrelate; the value is dropped.
            float warm = prng(rng);
            float3 dir = sample_cone(
                rng,
                normalize(params.direction_speed_min.xyz),
                params.position_spread.w);
            float speed = lerp(
                params.direction_speed_min.w, params.gravity_speed_max.w, prng(rng));
            float life = lerp(params.lifetime_min, params.lifetime_max, prng(rng));
            pt.position = params.position_spread.xyz;
            pt.velocity = dir * speed;
            pt.age = 0.0;
            pt.lifetime = max(life, 0.001);
        }
    }

    pool[id] = pt;
}
