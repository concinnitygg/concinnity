// The particle pool record and the per-emitter uniform, spliced into both
// halves of the particle system at their PARTICLE_TYPES marker (see
// shader_source.rs). Not a standalone program: it declares no entry point and no
// resources. The simulation kernel writes the pool the render pair reads and
// both stride the same uniform, so a per-shader copy is a silent layout-drift
// hazard. Nothing here may spell the marker itself -- the splice would
// reintroduce it into the assembled source.

// One particle slot, 48 B. Matches `GpuParticle` in each backend's particle
// module. Each (vec3, float) pair is spelled as float4: MSL sizes a float3
// at 16 bytes inside a structured buffer as well as inside a constant buffer,
// so a literal transcription would push the later pairs late on Metal alone.
struct Particle
{
    // xyz = world-space position, w = seconds lived.
    float4 position_age;
    // xyz = world-space velocity, w = total lifetime in seconds.
    float4 velocity_lifetime;
    // xyz = up of the plane the particle dies falling back through (zero for
    // none), w = the plane's height along it.
    float4 floor_plane;
};

// Per-frame emitter uniform, 128 B. Mirrors `ParticleParams` in
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
    // This frame's spawns fill `spawn_count` pool slots from `spawn_first` on,
    // wrapping past the end of the pool.
    uint spawn_count;
    uint spawn_first;
    uint random_seed;
    uint max_particles;
    // The emitter's bursts this frame: its first burst in the frame's rows in
    // the low 16 bits, how many it has in the high 16. Their spawns are the
    // last of the run.
    uint bursts;
    // Seconds of its motion each particle is drawn stretched along.
    float stretch;
    // How far each particle's size strays from the emitter's, as a share of it
    // either way.
    float size_variance;
};
