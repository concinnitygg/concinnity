//! The shapes of the targets the SSGI pass owns.

use crate::render::render_graph::{
    ClearValue, PixelFormat, TextureDesc, TextureSize, TextureUsage,
};

use super::super::device::{PostExtent, resolve_extent};

/// Most levels the trace's closest-depth pyramid holds. The coarsest covers
/// 16x16 trace pixels, which is as far as a near-field ray skips in one step.
pub const DEPTH_LEVELS: u32 = 5;

/// A trace-resolution target: the render resolution divided by `gi_scale` on
/// each axis, rendered to and sampled.
///
/// `gi_scale` is a small power of two, so its reciprocal is exact and the
/// graph's fractional resolve floors to the same size integer division does.
fn gi_shape(gi_scale: u32, format: PixelFormat, mip_levels: u32) -> TextureDesc {
    let scale = TextureSize::DrawableScaled(1.0 / gi_scale.max(1) as f32);
    TextureDesc {
        width: scale,
        height: scale,
        depth: 1,
        format,
        sample_count: 1,
        array_layers: 1,
        mip_levels,
        usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
        clear: ClearValue::Color([0.0, 0.0, 0.0, 0.0]),
    }
}

/// The shape of the trace output and of each accumulation slot: HDR color at
/// the trace resolution.
pub fn gi_desc(gi_scale: u32) -> TextureDesc {
    gi_shape(gi_scale, PixelFormat::Rgba16Float, 1)
}

/// The shape of one depth pyramid for a render resolution of `extent`: the
/// closest (`r`) and farthest (`g`) linear view depth at the trace resolution,
/// with as many of [`DEPTH_LEVELS`] levels as that size has.
pub fn depth_desc(gi_scale: u32, extent: PostExtent) -> TextureDesc {
    let levels = pyramid_levels(resolve_extent(&gi_desc(gi_scale), extent));
    gi_shape(gi_scale, PixelFormat::Rg32Float, levels)
}

/// Levels a pyramid over a `base`-sized top level holds: [`DEPTH_LEVELS`], or
/// fewer where the full chain down to one texel is shorter.
pub fn pyramid_levels(base: PostExtent) -> u32 {
    let full_chain = u32::BITS - base.width.max(base.height).max(1).leading_zeros();
    full_chain.min(DEPTH_LEVELS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extent(width: u32, height: u32) -> PostExtent {
        PostExtent { width, height }
    }

    #[test]
    fn a_pyramid_is_capped_at_its_level_count() {
        assert_eq!(pyramid_levels(extent(1280, 720)), DEPTH_LEVELS);
    }

    #[test]
    fn a_small_pyramid_stops_at_one_texel() {
        assert_eq!(pyramid_levels(extent(1, 1)), 1);
        assert_eq!(pyramid_levels(extent(2, 1)), 2);
        assert_eq!(pyramid_levels(extent(5, 3)), 3);
        assert_eq!(pyramid_levels(extent(0, 0)), 1);
    }

    #[test]
    fn the_pyramid_follows_the_trace_resolution() {
        let d = depth_desc(2, extent(2560, 1440));
        assert_eq!(d.format, PixelFormat::Rg32Float);
        assert_eq!(d.mip_levels, DEPTH_LEVELS);
        assert_eq!(
            resolve_extent(&d, extent(2560, 1440)),
            resolve_extent(&gi_desc(2), extent(2560, 1440))
        );
        // A render target small enough that the trace resolution cannot hold
        // every level asks for only the levels it has.
        assert_eq!(depth_desc(4, extent(8, 8)).mip_levels, 2);
    }

    #[test]
    fn the_accumulation_is_sampled_hdr_color() {
        let d = gi_desc(2);
        assert_eq!(d.format, PixelFormat::Rgba16Float);
        assert_eq!(d.mip_levels, 1);
        assert!(d.usage.contains(TextureUsage::RENDER_TARGET));
        assert!(d.usage.contains(TextureUsage::SHADER_READ));
    }
}
