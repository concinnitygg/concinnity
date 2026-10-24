//! repr(C) uniform structs only the Metal frame encoder and its passes bind.
//! Each layout must match the corresponding struct in an `.metal` shader under
//! `metal/shaders/`.
//!
//! Blocks whose shader counterpart is a single-source declaration are
//! declared once for every backend in the parent module; what is left here is
//! what only this backend binds. Their layouts are checked by `shader_layout` in
//! concinnity-device, which reads the expected offsets out of the compiled
//! module per target. The hand-written asserts below stay alongside that
//! check for the blocks this backend alone binds.

/// Per-frame inputs to the GPU-driven cull, pushed inline at buffer(2) of the
/// encoder both cull dispatches share. Layout (208 bytes) must match the
/// `CN_BACKEND_METAL` `CullParams` in `cull.hlsl`, which `shader_layout` reflects;
/// the scatter reads none of it and takes [`ScatterParams`] instead.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct CullUniforms {
    /// The six frustum planes (left/right/bottom/top/near/far), each
    /// `[normal.x, normal.y, normal.z, d]`, extracted CPU-side and already
    /// normalized so the kernel's plane test matches `gfx::frustum` exactly.
    pub planes: [[f32; 4]; 6],
    /// World-space camera position in `xyz`; `w` is unused. A whole lane
    /// because the shader-side `float3` is 16 bytes on Metal.
    pub cam_pos: [f32; 4],
    /// Previous frame's un-jittered view-projection. The kernel projects each
    /// AABB through this so the NDC depths line up with the Hi-Z values the
    /// previous frame's main pass produced.
    pub prev_view_proj: [[f32; 4]; 4],
    /// Hi-Z mip-0 dimensions in texels. `[1.0, 1.0]` when no Hi-Z is bound.
    pub hiz_size: [f32; 2],
    /// Mip levels in the bound Hi-Z texture.
    pub hiz_mip_count: u32,
    /// `0` skips the Hi-Z occlusion test (first frame / after a resize, before
    /// a valid pyramid exists); `1` runs it.
    pub hiz_enabled: u32,
    /// Number of valid `DrawObject` records; kernel threads past it return.
    pub object_count: u32,
    /// Unified-cull index where the folded skinned records begin (= static +
    /// instances). Equals `object_count` when no skinned mesh is folded. Read
    /// by the host when it fills [`ScatterParams`], not by the decision kernel.
    pub skinned_base: u32,
    /// Padding ahead of `bucket_count`.
    pub _pad: u32,
    /// How many shader-bucket ICBs this dispatch's argument buffer carries.
    /// The main cull passes the world's bucket count; single-stream dispatches
    /// (shadow, mirror) pass 1.
    pub bucket_count: u32,
}

/// Parameters of the Metal ICB scatter, pushed inline at buffer(7) after the
/// scan. Layout (16 bytes) must match `ScatterParams` in `cull_encode.metal`.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct ScatterParams {
    /// Records the cull covered.
    pub object_count: u32,
    /// Record index where the folded skinned tail begins; those records draw
    /// through the skinned index buffer.
    pub skinned_base: u32,
    /// How many shader-bucket ICBs the argument buffer carries.
    pub bucket_count: u32,
    /// The `cull_status` value that encodes a draw (`CullStatus::DRAWN` for
    /// every dispatch but phase 2, which encodes `CullStatus::REDRAW`).
    pub draw_status: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    #[test]
    fn cull_uniforms_layout_matches_the_shader() {
        // `CullParams` under CN_BACKEND_METAL in cull.hlsl: float4 planes[6], a
        // float4 camera lane, a float4x4 at 112, a float2 and six uints.
        assert_eq!(size_of::<CullUniforms>(), 208);
        assert_eq!(offset_of!(CullUniforms, planes), 0);
        assert_eq!(offset_of!(CullUniforms, cam_pos), 96);
        assert_eq!(offset_of!(CullUniforms, prev_view_proj), 112);
        assert_eq!(offset_of!(CullUniforms, hiz_size), 176);
        assert_eq!(offset_of!(CullUniforms, hiz_mip_count), 184);
        assert_eq!(offset_of!(CullUniforms, hiz_enabled), 188);
        assert_eq!(offset_of!(CullUniforms, object_count), 192);
        assert_eq!(offset_of!(CullUniforms, skinned_base), 196);
        assert_eq!(offset_of!(CullUniforms, _pad), 200);
        assert_eq!(offset_of!(CullUniforms, bucket_count), 204);
        assert_eq!(size_of::<CullUniforms>() % 16, 0);
    }

    #[test]
    fn scatter_params_layout_matches_msl() {
        // `ScatterParams` in cull_encode.metal: four tightly packed uints.
        assert_eq!(size_of::<ScatterParams>(), 16);
        assert_eq!(offset_of!(ScatterParams, object_count), 0);
        assert_eq!(offset_of!(ScatterParams, skinned_base), 4);
        assert_eq!(offset_of!(ScatterParams, bucket_count), 8);
        assert_eq!(offset_of!(ScatterParams, draw_status), 12);
    }
}
