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
/// the encode kernel reads none of it and takes [`EncodeParams`] instead.
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
    /// The first region this dispatch covers: its threads start at that
    /// region's block of the slot grid. See [`EncodeParams::encoded_span`].
    pub region_base: u32,
    /// Padding to a 16-byte multiple.
    pub _pad: u32,
}

impl EncodeParams {
    /// The regions a dispatch has to cover to encode every set bit of
    /// `region_mask` among `region_count`: the first set region and how many
    /// follow it up to the last one. A mask with no region set covers none.
    pub fn encoded_span(region_mask: u32, region_count: u32) -> (u32, u32) {
        let live = if region_count >= u32::BITS {
            region_mask
        } else {
            region_mask & ((1u32 << region_count) - 1)
        };
        if live == 0 {
            return (0, 0);
        }
        let first = live.trailing_zeros();
        let last = u32::BITS - 1 - live.leading_zeros();
        (first, last - first + 1)
    }
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
        assert_eq!(offset_of!(EncodeParams, region_base), 24);
        assert_eq!(offset_of!(EncodeParams, _pad), 28);
    }

    #[test]
    fn the_encoded_span_runs_from_the_first_set_region_to_the_last() {
        assert_eq!(EncodeParams::encoded_span(0b0100, 8), (2, 1));
        assert_eq!(EncodeParams::encoded_span(0b1010, 8), (1, 3));
        assert_eq!(EncodeParams::encoded_span(u32::MAX, 4), (0, 4));
        // Bits past the region count name regions that do not exist.
        assert_eq!(EncodeParams::encoded_span(0b1_0000, 4), (0, 0));
        assert_eq!(EncodeParams::encoded_span(0, 8), (0, 0));
        assert_eq!(EncodeParams::encoded_span(1 << 31, 32), (31, 1));
    }
}
