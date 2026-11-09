//! The GPU-driven draw cull's inputs (`cull.hlsl`), declared once for every
//! backend: the view half every dispatch reads and the Hi-Z half only the
//! occlusion-tested dispatches read.
//!
//! The halves arrive differently per backend. Vulkan pushes [`CullView`] as a
//! push constant and binds [`CullHiz`] as a uniform buffer, since the Hi-Z half
//! changes on its own schedule; DirectX and Metal bind both as one
//! [`CullParams`] block.

/// What every cull dispatch reads: the frustum, the camera, and the records it
/// covers. 128 bytes, the push-constant size every Vulkan device guarantees.
///
/// The camera position is three scalars in the shader, at the end: a `float3`
/// takes 16 bytes on Metal, and a 16-byte lane would not fit in 128 bytes
/// beside the five counts.
#[derive(Copy, Clone, Default, bytemuck::NoUninit)]
#[repr(C)]
pub struct CullView {
    /// Frustum planes (left/right/bottom/top/near/far), each
    /// `[normal.x, normal.y, normal.z, d]`, already normalized so the kernel's
    /// plane test matches `gfx::frustum` exactly.
    pub planes: [[f32; 4]; 6],
    /// Draw records the kernel iterates; threads past it return.
    pub object_count: u32,
    /// Shader buckets the kernel routes between; 1 sends every record to the
    /// default's stream.
    pub bucket_count: u32,
    /// The first skinned record, whose draws take the skinned tail's stream.
    /// Equals `object_count` when no skinned mesh is folded in.
    pub skinned_base: u32,
    /// First of the records `[exclude_first, exclude_end)` the view leaves out:
    /// a prop's own capture leaves out the prop. Empty for every other view.
    pub exclude_first: u32,
    /// End of the records the view leaves out.
    pub exclude_end: u32,
    /// World-space camera position.
    pub cam_pos: [f32; 3],
}

/// What the Hi-Z occlusion test reads. 80 bytes.
#[derive(Copy, Clone, Default, bytemuck::NoUninit)]
#[repr(C)]
pub struct CullHiz {
    /// Previous frame's un-jittered view-projection. Projects each AABB into
    /// the depth space the Hi-Z pyramid was reduced from.
    pub prev_view_proj: [[f32; 4]; 4],
    /// Hi-Z mip-0 dimensions in texels.
    pub hiz_size: [f32; 2],
    /// Mip levels in the bound Hi-Z texture.
    pub hiz_mip_count: u32,
    /// 0 skips the Hi-Z test (before a valid pyramid exists); 1 runs it.
    pub hiz_enabled: u32,
}

/// Both halves as one block, for a backend that binds the cull's inputs
/// together. 208 bytes.
#[derive(Copy, Clone, Default, bytemuck::NoUninit)]
#[repr(C)]
pub struct CullParams {
    /// The view half.
    pub view: CullView,
    /// The Hi-Z half.
    pub hiz: CullHiz,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    // `shader_layout` in concinnity-device reflects these against cull.hlsl
    // per target; this is the copy that runs without dxc.
    #[test]
    fn cull_view_layout_matches_the_shader() {
        assert_eq!(size_of::<CullView>(), 128);
        assert_eq!(offset_of!(CullView, planes), 0);
        assert_eq!(offset_of!(CullView, object_count), 96);
        assert_eq!(offset_of!(CullView, bucket_count), 100);
        assert_eq!(offset_of!(CullView, skinned_base), 104);
        assert_eq!(offset_of!(CullView, exclude_first), 108);
        assert_eq!(offset_of!(CullView, exclude_end), 112);
        assert_eq!(offset_of!(CullView, cam_pos), 116);
    }

    #[test]
    fn cull_hiz_layout_matches_the_shader() {
        assert_eq!(size_of::<CullHiz>(), 80);
        assert_eq!(offset_of!(CullHiz, prev_view_proj), 0);
        assert_eq!(offset_of!(CullHiz, hiz_size), 64);
        assert_eq!(offset_of!(CullHiz, hiz_mip_count), 72);
        assert_eq!(offset_of!(CullHiz, hiz_enabled), 76);
    }

    // The Hi-Z half opens on a 16-byte row, where every target's rules put a
    // block holding a matrix.
    #[test]
    fn cull_params_layout_matches_the_shader() {
        assert_eq!(size_of::<CullParams>(), 208);
        assert_eq!(offset_of!(CullParams, view), 0);
        assert_eq!(offset_of!(CullParams, hiz), 128);
    }
}
