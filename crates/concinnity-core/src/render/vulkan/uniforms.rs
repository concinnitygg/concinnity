//! repr(C) uniform / push-constant structs only the Vulkan frame encoders bind
//! (std140 / std430 / push-constant layouts). Each is mirrored field-for-field
//! in a `.glsl`/`.vert`/`.frag`/`.comp` shader under `vulkan/shaders/`, or in
//! the one `.slang` block Vulkan alone declares.
//!
//! Blocks whose shader counterpart is a single-source `.slang` declaration are
//! declared once for every backend in `crate::render::uniforms`; what is left here is
//! what only this backend binds. Their layouts are checked by `shader_layout` in
//! concinnity-device, which reads the expected offsets out of slangc's
//! reflection per target. The hand-written asserts below are for the families
//! whose shaders are still per backend -- the cull kernel, the skinning and
//! morph kernels, the raymarch SDF templates, the velocity pass, and Metal's
//! water.

/// Byte size of the auto-exposure push-constant range. Pins the struct size to
/// what the pipeline layout declares.
pub const AUTO_EXPOSURE_PUSH_BYTES: u32 = 16;

/// The GPU-cull push constant (cull.slang): six already-normalised frustum
/// planes (xyz = normal, w = d), the camera position sharing its 16-byte slot with
/// the build-time object count, then the shader-bucket routing (120 B total).
#[derive(Copy, Clone)]
#[repr(C)]
pub struct CullParams {
    /// Frustum planes, each `(normal.xyz, d)`.
    pub planes: [[f32; 4]; 6],
    /// World-space camera position.
    pub cam_pos: [f32; 3],
    /// Draw records the kernel iterates.
    pub object_count: u32,
    /// Shader-bucket command regions in the indirect buffer. The kernel writes
    /// every record's slot in all `bucket_count` regions (a draw in the record's
    /// own bucket, a no-op everywhere else); region `b` starts at command
    /// `b * bucket_stride`. `bucket_count = 1` degenerates to a single region.
    pub bucket_count: u32,
    /// `u32` slots one bucket occupies in the output list.
    pub bucket_stride: u32,
}

/// Cull-side Hi-Z uniforms (cull.slang, 80 bytes): the previous frame's
/// un-jittered view-projection, the Hi-Z mip-0 dimensions, the mip count, and an
/// enable flag. Mirrors the Metal / DirectX CullUniforms tail.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct CullHizParams {
    /// Previous frame's un-jittered view-projection. Projects each AABB into the
    /// depth space the Hi-Z pyramid was reduced from (`M * v`).
    pub prev_view_proj: [[f32; 4]; 4],
    /// Hi-Z mip-0 dimensions (in texels).
    pub hiz_size: [f32; 2],
    /// How many mip levels live in the bound texture.
    pub hiz_mip_count: u32,
    /// 0 skips the Hi-Z test entirely (first frame / after a resize, before a
    /// valid pyramid exists).
    pub hiz_enabled: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    // CullParams must match the `CullParams` push-constant block in cull.slang:
    // six frustum planes, then cam_pos sharing its 16-byte slot with
    // object_count. `shader_layout` in concinnity-device reflects the same
    // source; this is the copy that runs without slangc.
    #[test]
    fn cull_params_layout_matches_the_shader() {
        assert_eq!(size_of::<CullParams>(), 120);
        assert_eq!(offset_of!(CullParams, planes), 0);
        assert_eq!(offset_of!(CullParams, cam_pos), 96);
        assert_eq!(offset_of!(CullParams, object_count), 108);
        assert_eq!(offset_of!(CullParams, bucket_count), 112);
        assert_eq!(offset_of!(CullParams, bucket_stride), 116);
    }

    // CullHizParams in cull.slang: mat4 (64) + float2 (8) + two uints. Total 80
    // bytes, tightly packed after the mat4.
    #[test]
    fn cull_hiz_params_layout_matches_the_shader() {
        assert_eq!(size_of::<CullHizParams>(), 80);
        assert_eq!(offset_of!(CullHizParams, prev_view_proj), 0);
        assert_eq!(offset_of!(CullHizParams, hiz_size), 64);
        assert_eq!(offset_of!(CullHizParams, hiz_mip_count), 72);
        assert_eq!(offset_of!(CullHizParams, hiz_enabled), 76);
    }
}
