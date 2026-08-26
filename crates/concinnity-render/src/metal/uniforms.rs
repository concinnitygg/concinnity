//! repr(C) uniform structs only the Metal frame encoder and its passes bind.
//! Each layout must match the corresponding struct in an `.metal` shader under
//! `metal/shaders/`.
//!
//! Blocks whose shader counterpart is a single-source `.slang` declaration are
//! declared once for every backend in `crate::uniforms`; what is left here is
//! what only this backend binds. Their layouts are checked by `shader_layout` in
//! concinnity-device, which reads the expected offsets out of slangc's
//! reflection per target. The hand-written asserts below are for the families
//! whose shaders are still per backend -- the cull kernel, the skinning and
//! morph kernels, the raymarch SDF templates, the legacy per-draw main and
//! velocity passes.

/// Per-draw-call model matrix pushed at buffer(2) before each draw.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct ModelUniforms {
    /// Model-to-world matrix (column-major).
    pub model: [[f32; 4]; 4],
}

/// Per-draw material roughness pushed to the SSR pre-pass fragment at
/// buffer(0). Layout matches the `PpMat` struct in the SSR pre-pass MSL.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct SsrPrepassMat {
    /// Perceptual roughness `[0, 1]` of this draw's material.
    pub roughness: f32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad: [f32; 3],
}

/// Per-frame inputs to the GPU-driven cull kernel, pushed inline at
/// the compute encoder's buffer(2). Layout (208 bytes, a multiple of 16) must
/// match the `CullUniforms` struct in the cull kernel MSL (`build_cull_pipeline`).
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct CullUniforms {
    /// The six frustum planes (left/right/bottom/top/near/far), each
    /// `[normal.x, normal.y, normal.z, d]`, extracted CPU-side and already
    /// normalised so the kernel's plane test matches `gfx::frustum` exactly.
    pub planes: [[f32; 4]; 6],
    /// World-space camera position (packed_float3 in MSL, alignment 4).
    pub cam_pos: [f32; 3],
    /// Number of valid `DrawObject` records; kernel threads past it return.
    pub object_count: u32,
    /// Previous frame's un-jittered view-projection. The kernel projects each
    /// AABB through this so the NDC depths line up with the Hi-Z values the
    /// previous frame's main pass produced. `float4x4` lands at offset 112,
    /// already 16-aligned, so the layout matches MSL with no padding.
    pub prev_view_proj: [[f32; 4]; 4],
    /// Hi-Z mip-0 dimensions in texels. `[1.0, 1.0]` when no Hi-Z is bound.
    pub hiz_size: [f32; 2],
    /// Mip levels in the bound Hi-Z texture.
    pub hiz_mip_count: u32,
    /// `0` skips the Hi-Z occlusion test (first frame / after a resize, before
    /// a valid pyramid exists); `1` runs it.
    pub hiz_enabled: u32,
    /// Unified-cull index where the folded skinned records begin (= static +
    /// instances). The kernel draws records at or past this through the u16
    /// skinned index buffer instead of the static u32 one. Equals `object_count`
    /// when no skinned mesh is folded.
    pub skinned_base: u32,
    /// Command-slot base offset for the GPU-driven shadow cull: the
    /// shadow ICB holds NUM_SHADOW_CASCADES * object_count slots and cascade `c`
    /// writes its survivors at `cascade_base + tid` (= c * object_count). The
    /// main cull leaves it 0 (writes at `tid`).
    pub cascade_base: u32,
    /// How many shader-bucket ICBs this dispatch's argument buffer carries.
    /// The main cull passes the world's bucket count; single-stream dispatches
    /// (shadow, mirror) pass 1. Trailing `_pad_skin` rounds the struct to 208
    /// bytes so it matches the 16-aligned MSL `CullUniforms`.
    pub bucket_count: u32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad_skin: u32,
}

/// Per-frame uniforms for the TAA velocity pre-pass at buffer(0). Layout must
/// match `VelUniforms` in `pipeline.rs`'s velocity MSL.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct VelocityUniforms {
    /// Jittered current view-projection: drives the rasterised position so
    /// the pre-pass covers exactly the same pixels as the main pass.
    pub jittered_vp: [[f32; 4]; 4],
    /// Un-jittered current view-projection: keeps the stored motion vector
    /// free of the sub-pixel projection jitter.
    pub cur_vp: [[f32; 4]; 4],
    /// Un-jittered previous-frame view-projection.
    pub prev_vp: [[f32; 4]; 4],
}

/// Per-frame view inputs the raymarch pass binds at buffer(0). Layout matches
/// `RaymarchView` in `shaders/raymarch_helpers.metal`. 160 bytes.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct RaymarchView {
    /// View-projection matrix, column-major.
    pub vp: [[f32; 4]; 4],
    /// Inverse view-projection matrix, column-major.
    pub inv_vp: [[f32; 4]; 4],
    /// World-space camera position (xyz). `.w` is ignored.
    pub cam_pos: [f32; 4],
    /// HDR target width / height in pixels: the shader divides `position.xy` by
    /// this to read the depth attachment with integer pixel coordinates.
    pub viewport: [f32; 2],
    /// Wall-clock seconds since startup, available to the user SDF.
    pub time: f32,
    /// Mip count of the bound IBL prefilter cube; 0 disables the cube-sample IBL
    /// path and the helper falls back to the hand-tuned hemispheric ambient.
    /// Mirrors `ViewUniforms.prefilter_mip_count` from the Main pass: same
    /// semantics, same gate.
    pub prefilter_mip_count: f32,
}

/// Per-volume uniforms uploaded at buffer(1). Layout matches `SdfVolumeUniforms`
/// in `shaders/raymarch_helpers.metal`. 176 bytes (two packed_float3 + pad = 32,
/// four scalars = 16, 32 float params = 128).
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct RaymarchVolumeUniforms {
    /// World-space centre (`packed_float3` + pad).
    pub centre: [f32; 3],
    /// Padding so the field layout matches the shader-side struct.
    pub _pad0: f32,
    /// XYZ half-widths of the bounding box (`packed_float3` + pad).
    pub extent: [f32; 3],
    /// Padding so the field layout matches the shader-side struct.
    pub _pad1: f32,
    /// `1 / max_gradient`; the cone-step scale factor in `coneRaymarch`.
    pub cone_ratio: f32,
    /// Per-volume march far-clip in metres.
    pub max_distance: f32,
    /// Per-volume step cap (clamped 8..256 at asset load).
    pub max_steps: i32,
    /// Currently unused; reserved in the layout so user shaders that probe it
    /// find a stable slot.
    pub receive_shadows: i32,
    /// Generic parameter block; the user shader casts it to whatever struct it
    /// interprets.
    pub params: [f32; crate::components::sdf_volume::SDF_PARAMS_LEN],
}

/// Cascade selector pushed at buffer(4) for the raymarch shadow-caster pipeline.
/// Picks `shadow.light_vps[cascade_idx]` in both stages. Matches
/// `RaymarchShadowCascade` in `shaders/raymarch_shadow.metal`. 16 bytes.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct RaymarchShadowCascade {
    /// Which shadow cascade is being rendered.
    pub cascade_idx: u32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad: [u32; 3],
}

/// Morph-target cap per skinned mesh: the fixed weight-array length in the
/// skinned VS params (ARKit-style faces use ~52 targets).
pub const MAX_MORPH_TARGETS: usize = 64;

/// Per-draw morph parameters for the legacy skinned vertex shader. Matches the
/// MSL `VsMorphParams` in main.metal: four uints then the weight array.
/// 272 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
pub struct VsMorphParams {
    /// First vertex of this slot's region in the shared vertex buffer.
    pub vertex_base: u32,
    /// Vertices in this slot's region.
    pub vertex_count: u32,
    /// Morph targets on this slot's mesh.
    pub target_count: u32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad: u32,
    /// One weight per morph target, in target order.
    pub weights: [f32; MAX_MORPH_TARGETS],
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
}
