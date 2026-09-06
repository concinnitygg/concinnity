//! repr(C) uniform structs only the Metal frame encoder and its passes bind.
//! Each layout must match the corresponding struct in an `.metal` shader under
//! `metal/shaders/`.
//!
//! Blocks whose shader counterpart is a single-source `.slang` declaration are
//! declared once for every backend in `crate::render::uniforms`; what is left here is
//! what only this backend binds. Their layouts are checked by `shader_layout` in
//! concinnity-device, which reads the expected offsets out of slangc's
//! reflection per target. The hand-written asserts below stay alongside that
//! check for the blocks this backend alone binds.

/// Per-draw-call model matrix pushed at buffer(2) before each draw.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct ModelUniforms {
    /// Model-to-world matrix (column-major).
    pub model: [[f32; 4]; 4],
}

/// Per-frame inputs to the GPU-driven cull, pushed inline at buffer(2) of the
/// encoder both cull dispatches share. Layout (208 bytes) must match the
/// `METAL_BINDINGS` `CullParams` in `cull.slang`, which `shader_layout` reflects;
/// the encode kernel reads none of it and takes [`EncodeParams`] instead.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct CullUniforms {
    /// The six frustum planes (left/right/bottom/top/near/far), each
    /// `[normal.x, normal.y, normal.z, d]`, extracted CPU-side and already
    /// normalised so the kernel's plane test matches `gfx::frustum` exactly.
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
    /// by the host when it fills [`EncodeParams`], not by the decision kernel.
    pub skinned_base: u32,
    /// Status-slot base for the GPU-driven shadow cull: cascade `c` writes its
    /// outcomes at `cascade_base + tid` (= `c * object_count`). The main cull
    /// leaves it 0.
    pub cascade_base: u32,
    /// How many shader-bucket ICBs this dispatch's argument buffer carries.
    /// The main cull passes the world's bucket count; single-stream dispatches
    /// (shadow, mirror) pass 1.
    pub bucket_count: u32,
}

/// Parameters of the Metal ICB encode kernel, pushed inline at buffer(7) after
/// the decision dispatch. Layout (32 bytes) must match `EncodeParams` in
/// `cull_encode.metal`.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct EncodeParams {
    /// Records per region: the slot grid is `region_count * object_count`.
    pub object_count: u32,
    /// Regions in the target ICB: 1 for the main, phase-2 and mirror culls,
    /// one per cascade for the shadow cull.
    pub region_count: u32,
    /// Bit `r` set means region `r` is encoded this dispatch; a clear bit
    /// leaves that region's commands untouched.
    pub region_mask: u32,
    /// Record index where the folded skinned tail begins; those records draw
    /// through the skinned index buffer.
    pub skinned_base: u32,
    /// How many shader-bucket ICBs the argument buffer carries.
    pub bucket_count: u32,
    /// The `cull_status` value that encodes a draw (`CullStatus::DRAWN` for
    /// every dispatch but phase 2, which encodes `CullStatus::REDRAW`).
    pub draw_status: u32,
    /// Padding to a 16-byte multiple.
    pub _pad: [u32; 2],
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

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    #[test]
    fn cull_uniforms_layout_matches_the_shader() {
        // `CullParams` under METAL_BINDINGS in cull.slang: float4 planes[6], a
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
        assert_eq!(offset_of!(CullUniforms, cascade_base), 200);
        assert_eq!(offset_of!(CullUniforms, bucket_count), 204);
        assert_eq!(size_of::<CullUniforms>() % 16, 0);
    }

    #[test]
    fn encode_params_layout_matches_msl() {
        // `EncodeParams` in cull_encode.metal: eight tightly packed uints.
        assert_eq!(size_of::<EncodeParams>(), 32);
        assert_eq!(offset_of!(EncodeParams, object_count), 0);
        assert_eq!(offset_of!(EncodeParams, region_count), 4);
        assert_eq!(offset_of!(EncodeParams, region_mask), 8);
        assert_eq!(offset_of!(EncodeParams, skinned_base), 12);
        assert_eq!(offset_of!(EncodeParams, bucket_count), 16);
        assert_eq!(offset_of!(EncodeParams, draw_status), 20);
        assert_eq!(offset_of!(EncodeParams, _pad), 24);
    }

    #[test]
    fn velocity_uniforms_layout_matches_msl() {
        // MSL `VelUniforms` in velocity.metal: three float4x4.
        assert_eq!(size_of::<VelocityUniforms>(), 192);
        assert_eq!(offset_of!(VelocityUniforms, jittered_vp), 0);
        assert_eq!(offset_of!(VelocityUniforms, cur_vp), 64);
        assert_eq!(offset_of!(VelocityUniforms, prev_vp), 128);
    }
}
