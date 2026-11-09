//! The depth-bias raster state every shadow pass binds, on every host.
//!
//! The sample side owns per-cascade growth (`shadow_bias.hlsl`); the raster
//! side contributes slope alone, and does not vary by cascade or between the
//! cascade and spot passes.
//!
//! Slope scale and clamp are the two bias terms Metal, Vulkan and D3D12 define
//! identically: each multiplies the primitive's maximum depth slope per pixel
//! by the scale and adds it in NDC depth units, then clamps the sum in the
//! same units. The constant term is not portable -- Vulkan and D3D12 scale it
//! by an implementation-defined resolution derived from the primitive's
//! exponent for a float depth format, Metal's documentation gives no such
//! scaling -- so the same literal can mean offsets seven orders of magnitude
//! apart on the D32 float shadow maps all three hosts use. It is zero here
//! rather than converted per host.
//!
//! The bias pushes a caster away from the light, toward the far plane, so its
//! sign follows the depth convention. All three hosts add the signed sum to the
//! fragment's depth and clamp a negative sum from below by a negative clamp,
//! so the slope and the clamp both carry the direction.

use crate::render::depth::toward_far;

/// Constant depth bias. Zero: see the module note.
pub const RASTER_CONSTANT: f32 = 0.0;

/// Slope-scale factor, in NDC depth per unit depth slope, signed toward the
/// far plane.
pub const RASTER_SLOPE: f32 = toward_far(2.0);

/// Bound on the summed bias's magnitude, in NDC depth units, signed like
/// [`RASTER_SLOPE`]. Keeps a triangle nearly edge-on to the light, where the
/// slope term diverges, from pushing its casters far enough to detach their
/// contact shadows.
pub const RASTER_CLAMP: f32 = toward_far(0.01);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::depth::{DEPTH_FAR, DEPTH_NEAR};

    // How every host applies the raster bias (D3D12's rule, which Vulkan and
    // Metal share): the slope term scales the primitive's non-negative maximum
    // depth slope, and a non-zero clamp bounds the sum on the clamp's own side.
    fn applied_bias(max_depth_slope: f32) -> f32 {
        let bias = RASTER_CONSTANT + RASTER_SLOPE * max_depth_slope;
        if RASTER_CLAMP > 0.0 {
            bias.min(RASTER_CLAMP)
        } else if RASTER_CLAMP < 0.0 {
            bias.max(RASTER_CLAMP)
        } else {
            bias
        }
    }

    // A biased caster sits farther from the light than where it was
    // rasterized, by twice its depth slope, up to 0.01 of the depth range.
    #[test]
    fn the_bias_pushes_casters_toward_the_far_plane() {
        let toward_far = DEPTH_FAR - DEPTH_NEAR;
        for slope in [0.0, 1e-5, 1e-3, 4e-3, 0.1, 10.0] {
            let bias = applied_bias(slope);
            assert!(bias * toward_far >= 0.0, "{slope}: {bias}");
            let expected = (2.0 * slope).min(0.01);
            assert!((bias.abs() - expected).abs() < 1e-9, "{slope}: {bias}");
        }
    }
}
