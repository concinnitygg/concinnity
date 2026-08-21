// src/uniforms/view.rs
//
// The camera blocks: what the forward pass and the G-buffer pre-pass need to
// place a fragment, and the per-draw model pair the pre-pass differentiates for
// motion vectors.

// Per-frame view-projection uniforms, uploaded once per frame and shared across
// every draw in it. `view` is the standalone view matrix the vertex shader uses
// to compute view-space depth for cascade selection in the fragment shader.
//
// **The field names are a published contract.** A world-authored Metal shader
// declares its own `ViewUniforms` and binds it at Metal buffer(0); the runtime
// validator (`metal::shader_layout`) compares that declaration against this
// struct field by field, by name. Renaming one here rejects every world shader
// that spells it the old way. The `.slang` source declares the same bytes with
// its own spelling (`view_mat`, and `cam_x`/`cam_y`/`cam_z` in place of
// `cam_pos`), which is why the two are checked as byte ranges rather than by
// name.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct ViewUniforms {
    // Combined view-projection matrix (column-major).
    pub vp: [[f32; 4]; 4],
    // Camera view matrix (column-major). Used to compute view-space depth
    // in the vertex shader for shadow cascade selection.
    pub view: [[f32; 4]; 4],
    // Elapsed seconds, available to shaders for animation.
    pub elapsed: f32,
    // 1.0 when a screen-space / ray-traced reflection resolve composites this
    // frame, else 0.0. The forward fragment shader uses it to yield the sharp
    // specular for glossy surfaces to that resolve (whose miss-fallback samples
    // the same probe set), so a glossy surface does not show both the
    // parallax-approximate forward probe reflection and the exact resolved one.
    pub reflections_enabled: f32,
    // World-space camera position (packed_float3 in the MSL contract, alignment 4).
    pub cam_pos: [f32; 3],
    // Number of mip levels in the bound IBL prefilter cubemap. 0 means
    // "no EnvironmentMap bound": the fragment shader uses this as the IBL
    // enable flag and falls back to a flat ambient placeholder.
    pub prefilter_mip_count: f32,
    // 1.0 while the unlit view mode is active: shade_surface returns the base
    // color before lighting. Occupies what was pad space, so the offsets in
    // the user-shader binding contract are unchanged.
    pub shade_mode: f32,
    // End-padding: the shader rounds the block up to a multiple of float4x4's
    // 16-byte alignment, so the upload rounds explicitly to match.
    pub _end_pad: f32,
}

// Per-frame view inputs to the unified G-buffer pre-pass. The jittered current
// VP drives the rasterised position (matching the main pass); `view` takes the
// normal + position into view space (where SSR / SSAO / SSGI / RT work); the
// un-jittered cur/prev VPs derive a jitter-free motion vector. Matches `GbView`
// in `shaders/gbuffer_prepass.slang`. 256 bytes (four float4x4, all naturally
// 16-aligned, no padding).
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct GBufferView {
    pub jittered_vp: [[f32; 4]; 4],
    pub cur_vp: [[f32; 4]; 4],
    pub prev_vp: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
}

// Per-draw model matrices for the G-buffer pre-pass. Matches `GbModel` in
// `shaders/gbuffer_prepass.slang`, which Metal and DirectX bind as a constant
// buffer; Vulkan carries the same pair plus the roughness in one push-constant
// block (`vulkan::uniforms::GbModelPush`), because a pipeline layout may declare
// only one. For a static or skinned object with no motion the caller sets
// `prev == cur`.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct GBufferModel {
    pub cur_model: [[f32; 4]; 4],
    pub prev_model: [[f32; 4]; 4],
}
