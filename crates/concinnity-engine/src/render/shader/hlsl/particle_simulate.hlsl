// GPU particle simulation kernel: single source for every backend.
//
// One thread per slot in the per-emitter pool. Each thread ages and integrates
// whatever particle currently occupies its slot; if the age reaches the
// lifetime the slot is marked dead. The frame's spawns then take the run of
// dead slots the CPU's ring cursor names, each with a fresh velocity sampled inside
// a cone of half-angle acos(spread_cos) around `direction`. A spawn's slot and
// random stream depend only on its place in the run and the emitter's frame
// seed, so a frame replays the same on every launch.
//
// `Particle` and `ParticleParams` arrive from the shared PARTICLE_TYPES
// fragment, which the render half (`particle.hlsl`) splices too, so the pool
// this kernel writes and the pool that pass reads have one declaration. Both
// are locked to the Rust structs by the layout mirrors in concinnity-device's
// shader_layout/mirrors/geometry.rs.
//
// Every emitter also reads the frame's rows (core::render::particles::burst):
// the frame's move, which carries every live particle into the frame the
// world moved to, then the bursts systems asked for this frame. An emitter's
// bursts spawn the last `burst_spawns` of its run, each from its own point
// along its own axis.
//
// The pool takes its Metal index from the number on its `register()` (see
// concinnity-shader's `metal_bindings`), which is buffer(0), the slot
// `metal/particle.rs` writes; the rows take buffer(1).

{PARTICLE_TYPES}

[[vk::binding(0, 0)]] RWStructuredBuffer<Particle> pool : register(u0);

// Rows of `[R | t]` for the frame's move, then `[moved, _, _, _]`, then per
// burst `[position, count]` and `[direction, speed scale]`.
#ifdef CN_BACKEND_DIRECTX
StructuredBuffer<float4> frame_rows : register(t1);
#else
[[vk::binding(0, 1)]] StructuredBuffer<float4> frame_rows : register(t1);
#endif

// Mirror `PARTICLE_FRAME_HEADER` and `MAX_EMITTER_BURSTS` in core.
static const uint PARTICLE_FRAME_HEADER = 4u;
static const uint PARTICLE_MAX_EMITTER_BURSTS = 16u;

// A host difference, not a target one: DirectX takes the params as root
// constants at b0, while Vulkan pushes them and the Metal encoder writes them to
// buffer(2), past the pool. So the DirectX leg branches and the
// register on the shared one is the Metal index.
#ifdef CN_BACKEND_DIRECTX
ConstantBuffer<ParticleParams> params : register(b0);
#else
[[vk::push_constant]] ConstantBuffer<ParticleParams> params : register(b2);
#endif

// Cheap fixed-point hash to a unit float in [0, 1). Mutates `state` so a thread
// needing several uncorrelated samples advances it between calls.
float prng(inout uint state)
{
    state = state * 1664525u + 1013904223u;
    // Top 24 bits of the hash become the mantissa.
    return float(state >> 8) * (1.0 / 16777216.0);
}

// Sample a unit vector inside a cone of half-angle acos(cone_cos) centered on
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

    // Carry a live particle into the frame the world moved to.
    if (pt.velocity_lifetime.w > 0.0 && frame_rows[3].x > 0.0)
    {
        float4 p = float4(pt.position_age.xyz, 1.0);
        float4 v = float4(pt.velocity_lifetime.xyz, 0.0);
        pt.position_age.xyz = float3(dot(frame_rows[0], p), dot(frame_rows[1], p), dot(frame_rows[2], p));
        pt.velocity_lifetime.xyz = float3(dot(frame_rows[0], v), dot(frame_rows[1], v), dot(frame_rows[2], v));
    }

    // Age the particle in this slot, if any. A lifetime of 0 flags a dead slot.
    if (pt.velocity_lifetime.w > 0.0)
    {
        pt.position_age.w += params.dt;
        if (pt.position_age.w >= pt.velocity_lifetime.w)
        {
            pt.velocity_lifetime.w = 0.0;
        }
        else
        {
            pt.velocity_lifetime.xyz =
                pt.velocity_lifetime.xyz + params.gravity_speed_max.xyz * params.dt;
            pt.position_age.xyz = pt.position_age.xyz + pt.velocity_lifetime.xyz * params.dt;
        }
    }

    // This slot's place in the frame's spawn run. The CPU hands out only slots
    // whose last particle has died by now.
    uint spawn = (id + params.max_particles - params.spawn_first) % params.max_particles;
    if (spawn < params.spawn_count)
    {
        uint rng = (spawn * 747796405u) ^ (params.random_seed * 2891336453u);
        // Warm the RNG so adjacent spawns decorrelate; the value is dropped.
        float warm = prng(rng);
        float3 origin = params.position_spread.xyz;
        float3 axis = params.direction_speed_min.xyz;
        float scale = 1.0;
        // The run's last spawns are the bursts', each burst's in turn.
        uint rate_spawns = params.spawn_count - params.burst_spawns;
        if (spawn >= rate_spawns)
        {
            uint left = spawn - rate_spawns;
            for (uint b = 0u; b < min(params.burst_count, PARTICLE_MAX_EMITTER_BURSTS); b++)
            {
                uint row = PARTICLE_FRAME_HEADER + 2u * (params.burst_first + b);
                float4 at = frame_rows[row];
                uint count = uint(at.w);
                if (left < count)
                {
                    float4 along = frame_rows[row + 1u];
                    origin = at.xyz;
                    axis = along.xyz;
                    scale = along.w;
                    break;
                }
                left -= count;
            }
        }
        float3 dir = sample_cone(rng, normalize(axis), params.position_spread.w);
        float speed = scale * lerp(
            params.direction_speed_min.w, params.gravity_speed_max.w, prng(rng));
        float life = lerp(params.lifetime_min, params.lifetime_max, prng(rng));
        pt.position_age = float4(origin, 0.0);
        pt.velocity_lifetime = float4(dir * speed, max(life, 0.001));
    }

    pool[id] = pt;
}
