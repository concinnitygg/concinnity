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
/// encoder both cull dispatches share. Layout (224 bytes) must match the
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
    /// First of the records `[exclude_first, exclude_end)` the view leaves
    /// out: a prop's own capture leaves out the prop. Empty for every other
    /// view.
    pub exclude_first: u32,
    /// How many shader-bucket ICBs this dispatch's argument buffer carries.
    /// The main cull passes the world's bucket count; single-stream dispatches
    /// (shadow, mirror) pass 1.
    pub bucket_count: u32,
    /// End of the records the view leaves out.
    pub exclude_end: u32,
    /// Padding to a whole 16-byte row.
    pub _pad: [u32; 3],
}

/// Parameters of the Metal ICB scatter, pushed inline at buffer(7) after the
/// scan. Layout (32 bytes) must match `ScatterParams` in `cull_encode.metal`.
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
    /// First index of the coarse planet-tile grid a capture draws its tiles
    /// through.
    pub coarse_offset: u32,
    /// Indices in that grid; 0 draws every tile through its own grid, as
    /// every view but a capture does.
    pub coarse_count: u32,
    /// Indices in the full tile grid a planet tile must draw to be drawn
    /// coarse instead.
    pub full_count: u32,
    /// Padding to a whole 16-byte row.
    pub _pad: u32,
}

impl ScatterParams {
    /// This scatter drawing planet tiles through `tiles`, a capture's
    /// coarse grid, or through their own when `None`.
    #[must_use]
    pub fn with_capture_tiles(
        self,
        tiles: Option<crate::render::scene_state::CaptureTiles>,
    ) -> Self {
        let Some(t) = tiles else {
            return self;
        };
        Self {
            coarse_offset: t.offset,
            coarse_count: t.count,
            full_count: t.full_count,
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    #[test]
    fn cull_uniforms_layout_matches_the_shader() {
        // `CullParams` under CN_BACKEND_METAL in cull.hlsl: float4 planes[6], a
        // float4 camera lane, a float4x4 at 112, a float2 and ten uints.
        assert_eq!(size_of::<CullUniforms>(), 224);
        assert_eq!(offset_of!(CullUniforms, planes), 0);
        assert_eq!(offset_of!(CullUniforms, cam_pos), 96);
        assert_eq!(offset_of!(CullUniforms, prev_view_proj), 112);
        assert_eq!(offset_of!(CullUniforms, hiz_size), 176);
        assert_eq!(offset_of!(CullUniforms, hiz_mip_count), 184);
        assert_eq!(offset_of!(CullUniforms, hiz_enabled), 188);
        assert_eq!(offset_of!(CullUniforms, object_count), 192);
        assert_eq!(offset_of!(CullUniforms, skinned_base), 196);
        assert_eq!(offset_of!(CullUniforms, exclude_first), 200);
        assert_eq!(offset_of!(CullUniforms, bucket_count), 204);
        assert_eq!(offset_of!(CullUniforms, exclude_end), 208);
        assert_eq!(offset_of!(CullUniforms, _pad), 212);
        assert_eq!(size_of::<CullUniforms>() % 16, 0);
    }

    #[test]
    fn scatter_params_layout_matches_msl() {
        // `ScatterParams` in cull_encode.metal: eight tightly packed uints.
        assert_eq!(size_of::<ScatterParams>(), 32);
        assert_eq!(offset_of!(ScatterParams, object_count), 0);
        assert_eq!(offset_of!(ScatterParams, skinned_base), 4);
        assert_eq!(offset_of!(ScatterParams, bucket_count), 8);
        assert_eq!(offset_of!(ScatterParams, draw_status), 12);
        assert_eq!(offset_of!(ScatterParams, coarse_offset), 16);
        assert_eq!(offset_of!(ScatterParams, coarse_count), 20);
        assert_eq!(offset_of!(ScatterParams, full_count), 24);
        assert_eq!(offset_of!(ScatterParams, _pad), 28);
    }

    #[test]
    fn a_scatter_draws_planet_tiles_coarse_only_when_given_a_grid() {
        let plain = ScatterParams {
            object_count: 4,
            skinned_base: 4,
            bucket_count: 1,
            draw_status: 1,
            coarse_offset: 0,
            coarse_count: 0,
            full_count: 0,
            _pad: 0,
        };
        assert_eq!(plain.with_capture_tiles(None).coarse_count, 0);
        let tiles = crate::render::scene_state::CaptureTiles {
            offset: 1536,
            count: 96,
            full_count: 1536,
        };
        let coarse = plain.with_capture_tiles(Some(tiles));
        assert_eq!(
            (coarse.coarse_offset, coarse.coarse_count, coarse.full_count),
            (1536, 96, 1536)
        );
        assert_eq!(coarse.object_count, 4);
    }
}
