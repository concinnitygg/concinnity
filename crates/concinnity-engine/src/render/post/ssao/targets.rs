//! The shapes of the targets the SSAO pass owns.

use crate::render::render_graph::{
    ClearValue, PixelFormat, TextureDesc, TextureSize, TextureUsage,
};

/// The format of both occlusion targets: single-channel visibility, 1.0
/// unoccluded.
pub const OCCLUSION_FORMAT: PixelFormat = PixelFormat::R8Unorm;

/// The depth copy's format: the linear view depth as a fraction of the reach,
/// in even steps a few percent of the search radius apart. A floating-point
/// copy's steps double at each power of two and past a few hundred meters
/// reach the radius itself, which the kernel reads as occlusion that jumps
/// along rows of constant depth.
pub const DEPTH_FORMAT: PixelFormat = PixelFormat::R16Unorm;

// A single-level target at the render resolution, rendered to and sampled.
fn full_resolution(format: PixelFormat) -> TextureDesc {
    TextureDesc {
        width: TextureSize::Drawable,
        height: TextureSize::Drawable,
        depth: 1,
        format,
        sample_count: 1,
        array_layers: 1,
        mip_levels: 1,
        usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
        clear: ClearValue::Color([0.0, 0.0, 0.0, 0.0]),
    }
}

/// The raw occlusion's shape: single-channel at the render resolution.
pub fn raw_desc() -> TextureDesc {
    full_resolution(OCCLUSION_FORMAT)
}

/// The depth copy's shape: single-channel linear depth at the render
/// resolution.
pub fn depth_desc() -> TextureDesc {
    full_resolution(DEPTH_FORMAT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::post::device::{PostExtent, resolve_extent};

    const EXTENT: PostExtent = PostExtent {
        width: 2560,
        height: 1440,
    };

    #[test]
    fn the_raw_occlusion_is_one_sampled_level_at_the_render_resolution() {
        let d = raw_desc();
        assert_eq!(d.format, PixelFormat::R8Unorm);
        assert_eq!(d.mip_levels, 1);
        assert_eq!(resolve_extent(&d, EXTENT), EXTENT);
        assert!(d.usage.contains(TextureUsage::RENDER_TARGET));
        assert!(d.usage.contains(TextureUsage::SHADER_READ));
    }

    #[test]
    fn the_depth_copy_is_evenly_spaced_depth_at_the_render_resolution() {
        let d = depth_desc();
        assert_eq!(d.format, PixelFormat::R16Unorm);
        assert_eq!(d.mip_levels, 1);
        assert_eq!(resolve_extent(&d, EXTENT), EXTENT);
        assert!(d.usage.contains(TextureUsage::RENDER_TARGET));
        assert!(d.usage.contains(TextureUsage::SHADER_READ));
    }
}
