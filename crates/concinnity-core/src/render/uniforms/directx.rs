//! The DirectX GPU-cull `CullParams` cbuffer, whose layout the test below
//! asserts by hand against the `DXIL_ABI` block in `cull.slang`.

/// The GPU-cull `CullParams` cbuffer (b0, 208 bytes): six already-normalized
/// frustum planes, the camera position sharing its row with the object count, the
/// previous frame's view-projection, the Hi-Z metadata (dims, mip count, enable
/// flag), then the shader-bucket routing. DirectX fuses the cull + Hi-Z uniforms
/// into one cbuffer.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct CullParams {
    /// Frustum planes, each `(normal.xyz, d)`.
    pub planes: [[f32; 4]; 6],
    /// World-space camera position.
    pub cam_pos: [f32; 3],
    /// Draw records the kernel iterates.
    pub object_count: u32,
    /// The previous frame's view-projection, for velocity and Hi-Z reprojection.
    pub prev_view_proj: [[f32; 4]; 4],
    /// Hi-Z pyramid base size in pixels.
    pub hiz_size: [f32; 2],
    /// Mip levels in the Hi-Z pyramid.
    pub hiz_mip_count: u32,
    /// Non-zero when the Hi-Z occlusion test runs.
    pub hiz_enabled: u32,
    /// Shader-bucket command regions in the indirect buffer. The kernel writes
    /// every record's slot in all `bucket_count` regions (a draw in the record's
    /// own bucket, a no-op everywhere else); region `b` starts at command
    /// `b * bucket_stride`. `bucket_count = 1` degenerates to a single region.
    pub bucket_count: u32,
    /// `u32` slots one bucket occupies in the output list.
    pub bucket_stride: u32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad: [u32; 2],
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    // CullParams must match the `CullParams` root-constant block (b0) in
    // cull.slang under DXIL_ABI: six frustum planes, cam_pos sharing its row
    // with object_count, the previous view-projection, the Hi-Z metadata, then
    // the bucket routing pair opening a fresh 16-byte row (208 B total).
    #[test]
    fn cull_params_layout_matches_the_shader() {
        assert_eq!(size_of::<CullParams>(), 208);
        assert_eq!(offset_of!(CullParams, planes), 0);
        assert_eq!(offset_of!(CullParams, cam_pos), 96);
        assert_eq!(offset_of!(CullParams, object_count), 108);
        assert_eq!(offset_of!(CullParams, prev_view_proj), 112);
        assert_eq!(offset_of!(CullParams, hiz_size), 176);
        assert_eq!(offset_of!(CullParams, hiz_mip_count), 184);
        assert_eq!(offset_of!(CullParams, hiz_enabled), 188);
        assert_eq!(offset_of!(CullParams, bucket_count), 192);
        assert_eq!(offset_of!(CullParams, bucket_stride), 196);
    }
}
