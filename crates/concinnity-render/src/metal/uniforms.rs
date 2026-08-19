// src/metal/uniforms.rs
//
// repr(C) uniform structs only the Metal frame encoder and its passes bind.
// Each layout must match the corresponding struct in an `.metal` shader under
// `metal/shaders/`.
//
// Blocks whose shader counterpart is a single-source `.slang` declaration are
// declared once for every backend in `crate::uniforms`; what is left here is
// what only this backend binds. Their layouts are checked by `shader_layout` in
// concinnity-device, which reads the expected offsets out of slangc's
// reflection per target. The hand-written asserts below are for the families
// whose shaders are still per backend -- the cull kernel, the skinning and
// morph kernels, the raymarch SDF templates, the legacy per-draw main and
// velocity passes, and Metal's water / glass_mesh_rt.

// Per-draw-call model matrix pushed at buffer(2) before each draw.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ModelUniforms {
    // Model-to-world matrix (column-major).
    pub model: [[f32; 4]; 4],
}

// Per-draw material roughness pushed to the SSR pre-pass fragment at
// buffer(0). Layout matches the `PpMat` struct in the SSR pre-pass MSL.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct SsrPrepassMat {
    // Perceptual roughness `[0, 1]` of this draw's material.
    pub roughness: f32,
    pub _pad: [f32; 3],
}

// Per-frame inputs to the GPU-driven cull kernel, pushed inline at
// the compute encoder's buffer(2). Layout (208 bytes, a multiple of 16) must
// match the `CullUniforms` struct in the cull kernel MSL (`build_cull_pipeline`).
#[derive(Copy, Clone)]
#[repr(C)]
pub struct CullUniforms {
    // The six frustum planes (left/right/bottom/top/near/far), each
    // `[normal.x, normal.y, normal.z, d]`, extracted CPU-side and already
    // normalised so the kernel's plane test matches `gfx::frustum` exactly.
    pub planes: [[f32; 4]; 6],
    // World-space camera position (packed_float3 in MSL, alignment 4).
    pub cam_pos: [f32; 3],
    // Number of valid `DrawObject` records; kernel threads past it return.
    pub object_count: u32,
    // Previous frame's un-jittered view-projection. The kernel projects each
    // AABB through this so the NDC depths line up with the Hi-Z values the
    // previous frame's main pass produced. `float4x4` lands at offset 112,
    // already 16-aligned, so the layout matches MSL with no padding.
    pub prev_view_proj: [[f32; 4]; 4],
    // Hi-Z mip-0 dimensions in texels. `[1.0, 1.0]` when no Hi-Z is bound.
    pub hiz_size: [f32; 2],
    // Mip levels in the bound Hi-Z texture.
    pub hiz_mip_count: u32,
    // `0` skips the Hi-Z occlusion test (first frame / after a resize, before
    // a valid pyramid exists); `1` runs it.
    pub hiz_enabled: u32,
    // Unified-cull index where the folded skinned records begin (= static +
    // instances). The kernel draws records at or past this through the u16
    // skinned index buffer instead of the static u32 one. Equals `object_count`
    // when no skinned mesh is folded.
    pub skinned_base: u32,
    // Command-slot base offset for the GPU-driven shadow cull: the
    // shadow ICB holds NUM_SHADOW_CASCADES * object_count slots and cascade `c`
    // writes its survivors at `cascade_base + tid` (= c * object_count). The
    // main cull leaves it 0 (writes at `tid`).
    pub cascade_base: u32,
    // How many shader-bucket ICBs this dispatch's argument buffer carries.
    // The main cull passes the world's bucket count; single-stream dispatches
    // (shadow, mirror) pass 1. Trailing `_pad_skin` rounds the struct to 208
    // bytes so it matches the 16-aligned MSL `CullUniforms`.
    pub bucket_count: u32,
    pub _pad_skin: u32,
}

// Per-frame uniforms for the TAA velocity pre-pass at buffer(0). Layout must
// match `VelUniforms` in `pipeline.rs`'s velocity MSL.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct VelocityUniforms {
    // Jittered current view-projection: drives the rasterised position so
    // the pre-pass covers exactly the same pixels as the main pass.
    pub jittered_vp: [[f32; 4]; 4],
    // Un-jittered current view-projection: keeps the stored motion vector
    // free of the sub-pixel projection jitter.
    pub cur_vp: [[f32; 4]; 4],
    // Un-jittered previous-frame view-projection.
    pub prev_vp: [[f32; 4]; 4],
}

// One Gerstner wave coefficient set, packed for MSL float4 alignment.
// Matches `WaterWave` in `shaders/water.metal`. 32 bytes.
#[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct WaterWaveGpu {
    // `[direction.x, direction.y, amplitude, wavelength]`.
    pub dir_amp_wave: [f32; 4],
    // `[speed, steepness, pad, pad]`.
    pub speed_steep_pad: [f32; 4],
}

// Maximum waves per `WaterParams`. Mirrors `MAX_WATER_WAVES` in the MSL.
pub const WATER_MAX_WAVES: usize = 4;

// Per-surface tunables uploaded once per WaterSurface per frame. Layout
// matches `WaterParams` in `shaders/water.metal`. Vec3-ish fields are
// stored as `[f32; 4]` (with the trailing element unused) so the layout
// is byte-identical to MSL's `float4` regardless of how the MSL
// compiler packs `float3` and adjacent scalars: that packing rule has
// already bitten this struct once.
// 48 + 32 + 32 × WATER_MAX_WAVES = 208 bytes.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct WaterParams {
    // `[x, y, z, _]`: world-space surface centre.
    pub centre: [f32; 4],
    // `[r, g, b, _]`: water tint at full depth.
    pub deep_colour: [f32; 4],
    // `[r, g, b, _]`: water tint just above the seabed.
    pub shallow_colour: [f32; 4],
    pub depth_falloff: f32,
    pub foam_width: f32,
    pub foam_intensity: f32,
    pub fresnel_power: f32,
    pub roughness: f32,
    pub refraction_strength: f32,
    pub wave_count: u32,
    // Mip count of the bound IBL prefilter cube; 0 disables the
    // cube-sample path and the shader falls back to a hand-tuned sky tint.
    pub prefilter_mip_count: f32,
    pub waves: [WaterWaveGpu; WATER_MAX_WAVES],
    // Planar reflection control: `[strength, distortion, _, _]`. `strength > 0.5`
    // selects the sharp planar reflection (the scene re-rendered mirrored across
    // the water plane, sampled projectively at screen UV) over the probe / sky
    // cube; `distortion` scales the wave-normal screen-space ripple offset. A
    // float4 so the trailing struct stays 16-byte aligned. 0 when planar is off
    // (RT on, no water plane, or unsupported), keeping the probe/sky path.
    pub planar: [f32; 4],
}

// Per-draw tunables for a transparent glass MESH (an imported `Material` with
// `transparent: true` on an RT-capable device), uploaded at vertex + fragment
// buffer(6). Unlike `GlassParams` (a pre-baked world-space pane), a mesh is
// LOCAL-space, so this carries the model matrix the vertex shader applies; the
// fragment uses the interpolated per-vertex world normal. Matches
// `GlassMeshParams` in `shaders/glass_mesh_rt.metal`. 96 bytes (model is the
// first field, so its 16-byte GPU alignment is satisfied at offset 0).
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct GlassMeshParams {
    // Column-major local-to-world model matrix.
    pub model: [[f32; 4]; 4],
    // `[r, g, b, _]`: colour multiplied into the refracted scene (material tint).
    pub tint: [f32; 4],
    // Base alpha at normal incidence (from `Material.opacity`).
    pub opacity: f32,
    // Screen-space refraction offset strength.
    pub refraction_strength: f32,
    // Schlick-Fresnel exponent for the grazing-angle rim.
    pub fresnel_power: f32,
    // Mip count of the bound IBL prefilter cube (ray-miss fallback); 0 = none.
    pub prefilter_mip_count: f32,
}

// Per-frame view inputs the raymarch pass binds at buffer(0). Layout matches
// `RaymarchView` in `shaders/raymarch_helpers.metal`. 160 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct RaymarchView {
    pub vp: [[f32; 4]; 4],
    pub inv_vp: [[f32; 4]; 4],
    // World-space camera position (xyz). `.w` is ignored.
    pub cam_pos: [f32; 4],
    // HDR target width / height in pixels: the shader divides `position.xy` by
    // this to read the depth attachment with integer pixel coordinates.
    pub viewport: [f32; 2],
    // Wall-clock seconds since startup, available to the user SDF.
    pub time: f32,
    // Mip count of the bound IBL prefilter cube; 0 disables the cube-sample IBL
    // path and the helper falls back to the hand-tuned hemispheric ambient.
    // Mirrors `ViewUniforms.prefilter_mip_count` from the Main pass: same
    // semantics, same gate.
    pub prefilter_mip_count: f32,
}

// Per-volume uniforms uploaded at buffer(1). Layout matches `SdfVolumeUniforms`
// in `shaders/raymarch_helpers.metal`. 176 bytes (two packed_float3 + pad = 32,
// four scalars = 16, 32 float params = 128).
#[derive(Copy, Clone)]
#[repr(C)]
pub struct RaymarchVolumeUniforms {
    // World-space centre (`packed_float3` + pad).
    pub centre: [f32; 3],
    pub _pad0: f32,
    // XYZ half-widths of the bounding box (`packed_float3` + pad).
    pub extent: [f32; 3],
    pub _pad1: f32,
    // `1 / max_gradient`; the cone-step scale factor in `coneRaymarch`.
    pub cone_ratio: f32,
    // Per-volume march far-clip in metres.
    pub max_distance: f32,
    // Per-volume step cap (clamped 8..256 at asset load).
    pub max_steps: i32,
    // Currently unused; reserved in the layout so user shaders that probe it
    // find a stable slot.
    pub receive_shadows: i32,
    // Generic parameter block; the user shader casts it to whatever struct it
    // interprets.
    pub params: [f32; crate::assets::sdf_volume::SDF_PARAMS_LEN],
}

// Cascade selector pushed at buffer(4) for the raymarch shadow-caster pipeline.
// Picks `shadow.light_vps[cascade_idx]` in both stages. Matches
// `RaymarchShadowCascade` in `shaders/raymarch_shadow.metal`. 16 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct RaymarchShadowCascade {
    pub cascade_idx: u32,
    pub _pad: [u32; 3],
}

// Morph-target cap per skinned mesh: the fixed weight-array length in the
// skinned VS params (ARKit-style faces use ~52 targets).
pub const MAX_MORPH_TARGETS: usize = 64;

// Per-draw morph parameters for the legacy skinned vertex shader. Matches the
// MSL `VsMorphParams` in main.metal: four uints then the weight array.
// 272 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VsMorphParams {
    pub vertex_base: u32,
    pub vertex_count: u32,
    pub target_count: u32,
    pub _pad: u32,
    pub weights: [f32; MAX_MORPH_TARGETS],
}

// Per-dispatch parameters for the `rt_skin` compute kernel. Matches the MSL
// `SkinParams` in `shaders/rt_skin.metal`. 16 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SkinParams {
    pub vertex_base: u32,
    pub vertex_count: u32,
    pub joint_count: u32,
    // Morph targets bound at buffer(4)/(5); zero when the object has none.
    pub target_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn cull_uniforms_layout_matches_msl() {
        // MSL `CullUniforms` in cull.metal: float4 planes[6], packed_float3
        // cam_pos + object_count, then a float4x4 at the 16-aligned offset 112,
        // a float2 + two uints, then skinned_base + cascade_base +
        // bucket_count + 4B pad rounding to 208.
        assert_eq!(size_of::<CullUniforms>(), 208);
        assert_eq!(offset_of!(CullUniforms, planes), 0);
        assert_eq!(offset_of!(CullUniforms, cam_pos), 96);
        assert_eq!(offset_of!(CullUniforms, object_count), 108);
        assert_eq!(offset_of!(CullUniforms, prev_view_proj), 112);
        assert_eq!(offset_of!(CullUniforms, hiz_size), 176);
        assert_eq!(offset_of!(CullUniforms, hiz_mip_count), 184);
        assert_eq!(offset_of!(CullUniforms, hiz_enabled), 188);
        assert_eq!(offset_of!(CullUniforms, skinned_base), 192);
        assert_eq!(offset_of!(CullUniforms, cascade_base), 196);
        assert_eq!(offset_of!(CullUniforms, bucket_count), 200);
        assert_eq!(size_of::<CullUniforms>() % 16, 0);
    }

    #[test]
    fn velocity_uniforms_layout_matches_msl() {
        // MSL `VelUniforms` in velocity.metal: three float4x4.
        assert_eq!(size_of::<VelocityUniforms>(), 192);
        assert_eq!(offset_of!(VelocityUniforms, jittered_vp), 0);
        assert_eq!(offset_of!(VelocityUniforms, cur_vp), 64);
        assert_eq!(offset_of!(VelocityUniforms, prev_vp), 128);
    }

    #[test]
    fn water_wave_gpu_layout_matches_msl() {
        // MSL `WaterWave` in water.metal: two float4.
        assert_eq!(size_of::<WaterWaveGpu>(), 32);
        assert_eq!(offset_of!(WaterWaveGpu, dir_amp_wave), 0);
        assert_eq!(offset_of!(WaterWaveGpu, speed_steep_pad), 16);
    }

    #[test]
    fn water_params_layout_matches_msl() {
        // MSL `WaterParams` in water.metal: three float4, eight scalars, the
        // WaterWave array at the 16-aligned offset 80, then a trailing float4
        // `planar` at 208.
        assert_eq!(size_of::<WaterParams>(), 224);
        assert_eq!(offset_of!(WaterParams, centre), 0);
        assert_eq!(offset_of!(WaterParams, deep_colour), 16);
        assert_eq!(offset_of!(WaterParams, shallow_colour), 32);
        assert_eq!(offset_of!(WaterParams, depth_falloff), 48);
        assert_eq!(offset_of!(WaterParams, foam_width), 52);
        assert_eq!(offset_of!(WaterParams, foam_intensity), 56);
        assert_eq!(offset_of!(WaterParams, fresnel_power), 60);
        assert_eq!(offset_of!(WaterParams, roughness), 64);
        assert_eq!(offset_of!(WaterParams, refraction_strength), 68);
        assert_eq!(offset_of!(WaterParams, wave_count), 72);
        assert_eq!(offset_of!(WaterParams, prefilter_mip_count), 76);
        assert_eq!(offset_of!(WaterParams, waves), 80);
        assert_eq!(offset_of!(WaterParams, planar), 208);
        assert_eq!(size_of::<WaterParams>() % 16, 0);
    }

    #[test]
    fn glass_mesh_params_layout_matches_msl() {
        // MSL `GlassMeshParams` in glass_mesh_rt.metal: a float4x4 model, a float4
        // tint, then four scalars. model is first, so its 16-byte GPU alignment is
        // satisfied at offset 0 and the Rust [[f32; 4]; 4] matches byte-for-byte.
        assert_eq!(size_of::<GlassMeshParams>(), 96);
        assert_eq!(offset_of!(GlassMeshParams, model), 0);
        assert_eq!(offset_of!(GlassMeshParams, tint), 64);
        assert_eq!(offset_of!(GlassMeshParams, opacity), 80);
        assert_eq!(offset_of!(GlassMeshParams, refraction_strength), 84);
        assert_eq!(offset_of!(GlassMeshParams, fresnel_power), 88);
        assert_eq!(offset_of!(GlassMeshParams, prefilter_mip_count), 92);
        assert_eq!(size_of::<GlassMeshParams>() % 16, 0);
    }

    #[test]
    fn raymarch_view_layout_matches_msl() {
        // MSL `RaymarchView` in raymarch_helpers.metal: two float4x4, a
        // packed_float3 cam_pos (+ pad), float2 viewport, then two scalars.
        // The Rust `cam_pos: [f32; 4]` covers the same 16 bytes (xyz + pad)
        // as the MSL `packed_float3 cam_pos; float _pad0;`.
        assert_eq!(size_of::<RaymarchView>(), 160);
        assert_eq!(offset_of!(RaymarchView, vp), 0);
        assert_eq!(offset_of!(RaymarchView, inv_vp), 64);
        assert_eq!(offset_of!(RaymarchView, cam_pos), 128);
        assert_eq!(offset_of!(RaymarchView, viewport), 144);
        assert_eq!(offset_of!(RaymarchView, time), 152);
        assert_eq!(offset_of!(RaymarchView, prefilter_mip_count), 156);
        assert_eq!(size_of::<RaymarchView>() % 16, 0);
    }

    #[test]
    fn raymarch_volume_uniforms_layout_matches_msl() {
        // MSL `SdfVolumeUniforms` in raymarch_helpers.metal: two packed_float3
        // (+ pad), four scalars, then `SdfParams { float vals[32]; }` at offset
        // 48. The 176-byte size pins SDF_PARAMS_LEN == 32 (48 + 32*4).
        assert_eq!(size_of::<RaymarchVolumeUniforms>(), 176);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, centre), 0);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, _pad0), 12);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, extent), 16);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, _pad1), 28);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, cone_ratio), 32);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, max_distance), 36);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, max_steps), 40);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, receive_shadows), 44);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, params), 48);
    }

    #[test]
    fn raymarch_shadow_cascade_layout_matches_msl() {
        // MSL `RaymarchShadowCascade` in raymarch_shadow.metal: a uint + pad.
        assert_eq!(size_of::<RaymarchShadowCascade>(), 16);
        assert_eq!(offset_of!(RaymarchShadowCascade, cascade_idx), 0);
        assert_eq!(offset_of!(RaymarchShadowCascade, _pad), 4);
    }

    #[test]
    fn vs_morph_params_layout_matches_msl() {
        // MSL `VsMorphParams` in main.metal: four uints then float[64].
        assert_eq!(size_of::<VsMorphParams>(), 272);
        assert_eq!(offset_of!(VsMorphParams, vertex_base), 0);
        assert_eq!(offset_of!(VsMorphParams, vertex_count), 4);
        assert_eq!(offset_of!(VsMorphParams, target_count), 8);
        assert_eq!(offset_of!(VsMorphParams, _pad), 12);
        assert_eq!(offset_of!(VsMorphParams, weights), 16);
    }

    #[test]
    fn skin_params_layout_matches_msl() {
        // MSL `SkinParams` in rt_skin.metal: four tightly packed uints.
        assert_eq!(size_of::<SkinParams>(), 16);
        assert_eq!(offset_of!(SkinParams, vertex_base), 0);
        assert_eq!(offset_of!(SkinParams, vertex_count), 4);
        assert_eq!(offset_of!(SkinParams, joint_count), 8);
        assert_eq!(offset_of!(SkinParams, target_count), 12);
    }
}
