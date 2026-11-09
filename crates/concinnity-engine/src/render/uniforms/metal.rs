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
    /// every dispatch but the late pass, which encodes `CullStatus::REDRAW`).
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
    pub fn with_capture_tiles(self, tiles: Option<crate::gfx::render_types::CaptureTiles>) -> Self {
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
        let tiles = crate::gfx::render_types::CaptureTiles {
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
