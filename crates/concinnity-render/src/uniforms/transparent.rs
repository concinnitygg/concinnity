// src/uniforms/transparent.rs
//
// What the transparent pass binds: the shared per-frame view block, and the
// per-panel glass tunables.

// Per-frame view inputs shared by every draw in the transparent pass (water,
// glass, ...), bound once for the whole pass. Matches `TransparentView` in
// `shaders/glass.slang`. 160 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct TransparentView {
    pub vp: [[f32; 4]; 4],
    pub inv_vp: [[f32; 4]; 4],
    // World-space camera position (xyz). `.w` is ignored by the shader.
    pub camera_pos: [f32; 4],
    // Render-target width / height in pixels: the shader uses this to
    // turn its fragment position into a normalised screen UV.
    pub viewport: [f32; 2],
    // Wall-clock seconds since startup, fed to the Gerstner sum.
    pub time: f32,
    // Mip count of the bound IBL prefilter cube; 0 signals "no environment map",
    // where the glass reflection falls back to a white rim. Per-frame state, so
    // it rides the shared view rather than a per-draw params block.
    pub prefilter_mip_count: f32,
}

// Per-panel tunables for a `GlassPanel`, uploaded once per panel per frame.
// The vec3-ish fields are `[f32; 4]` so the layout is byte-identical to the
// shader's `float4`. Matches `GlassParams` in `shaders/glass.slang`. 64 bytes.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct GlassParams {
    // `[x, y, z, _]`: world-space panel centre.
    pub centre: [f32; 4],
    // `[nx, ny, nz, _]`: unit panel normal (facing direction).
    pub normal: [f32; 4],
    // `[r, g, b, _]`: colour multiplied into the refracted scene.
    pub tint: [f32; 4],
    // Base alpha at normal incidence.
    pub opacity: f32,
    // Screen-space refraction offset strength.
    pub refraction_strength: f32,
    // Schlick-Fresnel exponent for the grazing-angle rim.
    pub fresnel_power: f32,
    // Planar reflection strength: `> 0.5` selects the sharp planar reflection
    // (the scene re-rendered mirrored across this pane's plane, sampled
    // projectively at screen UV) over the probe / sky cube. 0 when planar is off
    // (RT on, no planar slot, or the plane overflowed the budget), keeping the
    // probe / sky path. Patched per-frame in `collect_glass_transparent_draws`.
    pub planar: f32,
}
