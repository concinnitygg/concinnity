// The particle pool record and the per-emitter uniform, spliced into both
// halves of the particle system at their PARTICLE_TYPES marker (see
// slang_source.rs). Not a standalone program: it declares no entry point and no
// resources. The simulation kernel writes the pool the render pair reads and
// both stride the same uniform, so a per-shader copy is a silent layout-drift
// hazard. Nothing here may spell the marker itself -- the splice would
// reintroduce it into the assembled source.

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
    // the simulation's spawn counter, so only this many succeed.
    uint spawn_budget;
    uint random_seed;
    uint max_particles;
};
