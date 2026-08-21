// src/uniforms/geometry.rs
//
// The passes that draw their own geometry into the main target: projected
// decals, world-space lines, and the GPU particle system.

// Per-frame view inputs to the projected-decal pass. Matches `DecalView` in
// `shaders/decal.slang`. 144 bytes.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct DecalView {
    // View-projection matrix used by the main pass (jittered when TAA is on).
    pub vp: [[f32; 4]; 4],
    // Inverse of `vp`. The fragment shader uses it to reconstruct world space
    // from the depth attachment at each pixel.
    pub inv_vp: [[f32; 4]; 4],
    // HDR target dimensions in pixels: drives the screen->NDC conversion.
    pub viewport: [f32; 2],
    pub _pad: [f32; 2],
}

// Per-decal uniforms uploaded before each draw. Matches `DecalParams` in
// `shaders/decal.slang`. 160 bytes (two float4x4s, a float4 tint, four scalars).
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct DecalParams {
    pub model: [[f32; 4]; 4],
    pub inv_model: [[f32; 4]; 4],
    pub tint: [f32; 4],
    pub fade_pow: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

// Per-frame view inputs to the line pass. Matches `LineView` in
// `shaders/line.slang`. 80 bytes.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct LineView {
    // View-projection matrix used by the main pass (jittered when TAA is on),
    // so a line lands on the same pixel the geometry it sits on did.
    pub vp: [[f32; 4]; 4],
    // Alpha multiplier applied where a line falls behind scene geometry.
    pub occluded_alpha: f32,
    pub _pad: [f32; 3],
}

// Per-frame view inputs to the particle render pass. Matches `ParticleView` in
// `shaders/particle.slang`. 96 bytes.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct ParticleView {
    // View-projection matrix used by the main pass.
    pub vp: [[f32; 4]; 4],
    // World-space camera right vector: drives the first billboard axis. The
    // shader reads a float4 whose trailing component is unused padding.
    pub cam_right: [f32; 3],
    pub _pad0: f32,
    // World-space camera up vector: drives the second billboard axis.
    pub cam_up: [f32; 3],
    pub _pad1: f32,
}

// One particle slot in the pool the simulation kernel writes and the render
// pair reads. Matches `Particle` in `shaders/particle_types.slang`, which
// spells the same 32 bytes as two float4 lanes.
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct GpuParticle {
    pub position: [f32; 3],
    pub age: f32,
    pub velocity: [f32; 3],
    pub lifetime: f32,
}
