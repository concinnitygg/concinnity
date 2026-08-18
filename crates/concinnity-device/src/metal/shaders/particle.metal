#include <metal_stdlib>
using namespace metal;

// --- GPU-compute particle system: the render half ---
//
// The renderer keeps one persistent particle pool per emitter, simulated by a
// compute kernel each frame and rasterised by this vertex / fragment pipeline
// as camera-facing billboards.
//
// The simulation half ships from single source (`shaders/particle_simulate.slang`).
// `Particle` and `ParticleParams` are declared in both places until this render
// pair ports too; both are locked to the Rust structs by
// `particle_params_layout_matches_msl` in render_types.rs.
//
// `particle_vertex` is invoked with 4 vertices per particle and the pool size
// as the instance count. Each instance reads its `Particle` from the pool,
// derives a camera-facing axis pair from the view-projection matrix, and emits
// one corner of the billboard quad.
//
// The composited colour is alpha-blended into the resolved HDR target via
// the pipeline's blend state (Src.A · Src + (1 - Src.A) · Dst), same as the
// projected-decal pass.

// Persistent per-particle slot. Live slots have `lifetime > 0`; a slot whose
// age reaches its lifetime is killed by setting `lifetime = 0`, and is then a
// candidate for respawn the same frame. 32 bytes, matching the buffer the
// `ParticleEmitterGpuState` allocates Rust-side.
struct Particle {
    packed_float3 position;
    float age;
    packed_float3 velocity;
    float lifetime;
};

// Per-frame compute + render uniform. Mirrors `ParticleParams` in
// `gfx/render_types.rs` exactly (112 bytes). `position` / `direction` /
// `gravity` are packed_float3 so the trailing scalar slot of each float4
// holds the matching scalar (spread_cos / speed_min / speed_max).
struct ParticleParams {
    packed_float3 position;
    float         spread_cos;
    packed_float3 direction;
    float         speed_min;
    packed_float3 gravity;
    float         speed_max;
    float4        color_start;
    float4        color_end;
    float         lifetime_min;
    float         lifetime_max;
    float         size_start;
    float         size_end;
    float         dt;
    uint          spawn_budget;
    uint          random_seed;
    uint          max_particles;
};

// Per-frame view inputs to the render pass. The vertex shader projects each
// particle by `vp` and reads `cam_right` / `cam_up` to derive a camera-facing
// quad without recomputing the basis per-vertex.
struct ParticleView {
    float4x4 vp;
    packed_float3 cam_right;
    float _pad0;
    packed_float3 cam_up;
    float _pad1;
};

// Render pass - one quad (4 vertices, drawn as a triangle strip) per live
// particle. Dead particles emit a degenerate (zero-area) quad off-screen so
// they cost nothing past the vertex stage.

struct ParticleVtxOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
    float discard_flag;
};

vertex ParticleVtxOut particle_vertex(
    device   const Particle       *pool   [[buffer(0)]],
    constant ParticleView         &v      [[buffer(1)]],
    constant ParticleParams       &p      [[buffer(2)]],
    uint     vid                          [[vertex_id]],
    uint     iid                          [[instance_id]]
) {
    ParticleVtxOut out;
    Particle pt = pool[iid];

    // Dead slot → emit a degenerate quad clipped behind the near plane. The
    // fragment shader also discards on the `discard_flag` so any stray
    // rasterised pixels (numerical edge case at exactly w=0) draw nothing.
    if (pt.lifetime <= 0.0) {
        out.position = float4(0.0, 0.0, -2.0, 1.0);
        out.uv = float2(0.0);
        out.color = float4(0.0);
        out.discard_flag = 1.0;
        return out;
    }

    float t = clamp(pt.age / pt.lifetime, 0.0, 1.0);
    float size = mix(p.size_start, p.size_end, t);
    float4 color = mix(p.color_start, p.color_end, t);

    // 0..3 → ( -1, -1 ), ( +1, -1 ), ( -1, +1 ), ( +1, +1 ) for a strip.
    float2 corner = float2(
        (vid & 1u) == 0u ? -1.0 : 1.0,
        (vid & 2u) == 0u ? -1.0 : 1.0
    );

    float3 right = float3(v.cam_right) * (corner.x * 0.5 * size);
    float3 up    = float3(v.cam_up)    * (corner.y * 0.5 * size);
    float3 world = float3(pt.position) + right + up;
    out.position = v.vp * float4(world, 1.0);

    // 0..1 in each axis; V is flipped at sample time below to match the rest
    // of the engine's textures (V=0 at the top of the image).
    out.uv = corner * 0.5 + 0.5;
    out.color = color;
    out.discard_flag = 0.0;
    return out;
}

fragment float4 particle_fragment(
    ParticleVtxOut       in   [[stage_in]],
    texture2d<float>     tex  [[texture(0)]],
    sampler              samp [[sampler(0)]]
) {
    if (in.discard_flag > 0.5) {
        discard_fragment();
    }
    float2 uv = float2(in.uv.x, 1.0 - in.uv.y);
    float4 sampled = tex.sample(samp, uv);
    float4 c;
    c.rgb = sampled.rgb * in.color.rgb;
    c.a = sampled.a * in.color.a;
    return c;
}
