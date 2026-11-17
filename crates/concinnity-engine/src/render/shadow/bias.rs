//! The depth-bias raster state the shadow passes bind, on every host.
//!
//! The sample side owns per-cascade growth (`shadow_bias.hlsl`); the raster
//! side contributes slope alone. The cascade and spot passes share one slope
//! and clamp; the far shadow pass shares the slope with a tighter clamp, as
//! its depth range spans kilometers.
//!
//! Slope scale and clamp are the two bias terms Metal, Vulkan and D3D12 define
//! identically: each multiplies the primitive's maximum depth slope per pixel
//! by the scale and adds it in NDC depth units, then clamps the sum in the
//! same units. The constant term is not portable: on the D32 float cascade and
//! spot maps, Vulkan and D3D12 scale it by an implementation-defined
//! resolution derived from the primitive's exponent, and Metal's
//! documentation gives no such scaling, so the same literal can mean offsets
//! seven orders of magnitude apart. On the D16 far map it would be one unit of
//! the format, but the slope term already covers it there. It is zero
//! everywhere rather than converted per host and format.
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

/// [`RASTER_CLAMP`] for the far shadow map, whose depth range spans
/// kilometers: 0.001 of the 12 km a 4 km far reach covers is 12 m, three of its
/// texels, where the cascades' 0.01 would let a grazing caster slide 120 m off
/// its own shadow.
pub const FAR_RASTER_CLAMP: f32 = toward_far(0.001);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::depth::{DEPTH_FAR, DEPTH_NEAR};

    // How every host applies the raster bias (D3D12's rule, which Vulkan and
    // Metal share): the slope term scales the primitive's non-negative maximum
    // depth slope, and a non-zero clamp bounds the sum on the clamp's own side.
    fn applied_bias(max_depth_slope: f32, clamp: f32) -> f32 {
        let bias = RASTER_CONSTANT + RASTER_SLOPE * max_depth_slope;
        if clamp > 0.0 {
            bias.min(clamp)
        } else if clamp < 0.0 {
            bias.max(clamp)
        } else {
            bias
        }
    }

    // A biased caster sits farther from the light than where it was
    // rasterized, by twice its depth slope, up to 0.01 of the depth range in
    // the cascade and spot passes and 0.001 in the far pass.
    #[test]
    fn the_bias_pushes_casters_toward_the_far_plane() {
        let toward_far = DEPTH_FAR - DEPTH_NEAR;
        for (clamp, bound) in [(RASTER_CLAMP, 0.01), (FAR_RASTER_CLAMP, 0.001)] {
            for slope in [0.0, 1e-5, 1e-4, 1e-3, 4e-3, 0.1, 10.0] {
                let bias = applied_bias(slope, clamp);
                assert!(bias * toward_far >= 0.0, "{slope}: {bias}");
                let expected = (2.0 * slope).min(bound);
                assert!((bias.abs() - expected).abs() < 1e-9, "{slope}: {bias}");
            }
        }
    }
}
