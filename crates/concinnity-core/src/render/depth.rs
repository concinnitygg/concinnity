//! The engine's depth convention: which end of the device-depth range is near,
//! what a depth target clears to, which comparison keeps the nearer fragment,
//! and the projections that produce the depth in the first place.
//!
//! Every depth target uses the same convention: the main camera and every
//! target tested against its depth, the reflection-probe faces, the planar
//! reflections, the directional shadow cascades and the spot shadow slices.
//! Depth is reversed (near is 1, far is 0), which on a 32-bit float buffer
//! spreads precision evenly over distance. Every depth clear, depth-test state,
//! compare sampler and projection reads it from here instead of spelling a
//! literal. The shader-side counterpart is `shaders/depth_convention.hlsl`.
//!
//! The views differ only in their projections: the camera's has no far plane
//! (depth reaches the far value only at infinity), while a shadow view covers a
//! finite range from its near plane to its far plane.

use crate::gfx::projection::{
    reversed_infinite_oblique_rh, reversed_infinite_perspective_rh, reversed_ortho_rh,
    reversed_perspective_rh,
};
use crate::transform::Mat4;

/// A depth test: how an incoming fragment's depth compares against the stored
/// one for the fragment to pass.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DepthCompare {
    /// Passes when the incoming depth is less than the stored one.
    Less,
    /// Passes when the incoming depth is less than or equal to the stored one.
    LessEqual,
    /// Passes when the incoming depth is greater than the stored one.
    Greater,
    /// Passes when the incoming depth is greater than or equal to the stored one.
    GreaterEqual,
}

/// Device depth at the near plane.
pub const DEPTH_NEAR: f32 = 1.0;

/// Device depth at the far plane, which for the camera is infinity.
pub const DEPTH_FAR: f32 = 0.0;

/// The value every depth target clears to: the far plane, so any surface drawn
/// wins over the cleared background.
pub const DEPTH_CLEAR: f32 = DEPTH_FAR;

/// The test a depth-writing pass draws with. Only a strictly nearer fragment
/// passes, so the first of two coplanar draws keeps the pixel.
pub const DEPTH_WRITE_COMPARE: DepthCompare = DepthCompare::Greater;

/// The test that also passes at equal depth: a read-only test against depth
/// another pass already wrote, a pass whose shader replaces its fragment depth
/// with a value no farther than the rasterized one, and a shadow-map sample,
/// which is lit where its reference depth is no farther from the light than the
/// stored caster.
pub const DEPTH_INCLUSIVE_COMPARE: DepthCompare = DepthCompare::GreaterEqual;

/// `offset` as a signed step in device depth toward the far plane: the
/// direction a shadow caster's raster bias pushes it, away from the light.
pub const fn toward_far(offset: f32) -> f32 {
    offset * (DEPTH_FAR - DEPTH_NEAR)
}

/// How a projection maps distance onto device depth, in the terms an
/// upscaler's depth flags describe it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DepthMapping {
    /// The near plane sits at device depth 1 and the far plane at 0.
    pub reversed: bool,
    /// The projection has no far plane: depth reaches the far value only at
    /// infinity.
    pub infinite: bool,
}

/// The mapping [`camera_projection`] produces, which is what every upscaler
/// reads from the camera's depth buffer.
pub const CAMERA_DEPTH: DepthMapping = DepthMapping {
    reversed: DEPTH_NEAR > DEPTH_FAR,
    infinite: true,
};

/// The camera's perspective projection: the main camera and the reflection
/// probe faces. `fov_y_radians` is the full vertical field of view; `aspect` is
/// width over height. `near` maps to [`DEPTH_NEAR`], and depth falls toward
/// [`DEPTH_FAR`] with distance, reaching it only at infinity.
pub fn camera_projection(fov_y_radians: f32, aspect: f32, near: f32) -> Mat4 {
    reversed_infinite_perspective_rh(fov_y_radians, aspect, near)
}

/// A camera projection whose near clip plane is replaced by `clip_plane`, given
/// in the projection's view space, so everything on its negative side clips.
/// Depth still reaches the far plane only at infinity. A planar reflection uses
/// it to clip the geometry behind its mirror.
pub(crate) fn camera_oblique_projection(proj: Mat4, clip_plane: [f32; 4]) -> Mat4 {
    reversed_infinite_oblique_rh(proj, clip_plane)
}

/// A spot light's perspective shadow projection over `near..far` from the
/// bulb: `near` maps to [`DEPTH_NEAR`] and `far` to [`DEPTH_FAR`].
pub(crate) fn shadow_perspective(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    reversed_perspective_rh(fov_y_radians, aspect, near, far)
}

/// A directional cascade's orthographic shadow projection over the given
/// light-space box: its near face maps to [`DEPTH_NEAR`] and its far face to
/// [`DEPTH_FAR`].
pub(crate) fn shadow_ortho(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> Mat4 {
    reversed_ortho_rh(left, right, bottom, top, near, far)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::projection::{ortho_rh, perspective_rh};

    fn device_depth(m: Mat4, view_z: f32) -> f32 {
        let z = m[2][2] * view_z + m[3][2];
        let w = m[2][3] * view_z + m[3][3];
        z / w
    }

    // Whether device depth `a` is strictly nearer than `b`, by the write test.
    fn passes_write_test(a: f32, b: f32) -> bool {
        match DEPTH_WRITE_COMPARE {
            DepthCompare::Less => a < b,
            DepthCompare::LessEqual => a <= b,
            DepthCompare::Greater => a > b,
            DepthCompare::GreaterEqual => a >= b,
        }
    }

    #[test]
    fn depth_is_reversed() {
        assert_eq!((DEPTH_NEAR, DEPTH_FAR, DEPTH_CLEAR), (1.0, 0.0, 0.0));
        assert_eq!(DEPTH_WRITE_COMPARE, DepthCompare::Greater);
        assert_eq!(DEPTH_INCLUSIVE_COMPARE, DepthCompare::GreaterEqual);
        assert_eq!(
            CAMERA_DEPTH,
            DepthMapping {
                reversed: true,
                infinite: true
            }
        );
    }

    // A cleared pixel must lose to every surface a write test can produce, and
    // the write test must be the strict form of the inclusive one.
    #[test]
    fn the_clear_is_the_far_plane() {
        assert_eq!(DEPTH_CLEAR, DEPTH_FAR);
        assert!(passes_write_test(DEPTH_NEAR, DEPTH_CLEAR));
        assert!(passes_write_test(0.5, DEPTH_CLEAR));
        assert!(!passes_write_test(DEPTH_CLEAR, DEPTH_CLEAR));
        assert!(!passes_write_test(0.5, 0.5));
        let strict = match DEPTH_INCLUSIVE_COMPARE {
            DepthCompare::LessEqual => DepthCompare::Less,
            DepthCompare::GreaterEqual => DepthCompare::Greater,
            other => other,
        };
        assert_eq!(strict, DEPTH_WRITE_COMPARE);
    }

    // A caster's raster bias moves it toward the far plane, the side a sample
    // reference biased toward the near plane then wins against.
    #[test]
    fn toward_far_points_at_the_far_plane() {
        let d = 0.5;
        assert!(passes_write_test(d, d + toward_far(0.01)));
        assert_eq!(toward_far(0.0), 0.0);
        assert_eq!(toward_far(2.0).abs(), 2.0);
    }

    // The projections must map their near and far planes onto the convention's
    // device depths.
    #[test]
    fn projections_land_on_the_convention_planes() {
        let camera = camera_projection(1.2, 1.6, 0.1);
        assert_eq!(device_depth(camera, -0.1), DEPTH_NEAR);

        let spot = shadow_perspective(1.0, 1.0, 0.05, 40.0);
        assert!((device_depth(spot, -0.05) - DEPTH_NEAR).abs() < 1e-6);
        assert!((device_depth(spot, -40.0) - DEPTH_FAR).abs() < 1e-6);

        let cascade = shadow_ortho(-4.0, 4.0, -4.0, 4.0, -2.0, 8.0);
        assert!((device_depth(cascade, 2.0) - DEPTH_NEAR).abs() < 1e-6);
        assert!((device_depth(cascade, -8.0) - DEPTH_FAR).abs() < 1e-6);
    }

    // The shadow entry points mirror the standard-depth builders: the same
    // x, y and w rows, and depth exactly 1 - standard over the whole range.
    #[test]
    fn the_shadow_entry_points_mirror_standard_depth() {
        for (fov, aspect, near, far) in [(1.2, 1.6, 0.1, 500.0), (0.4, 0.5, 0.01, 10.0)] {
            let standard = perspective_rh(fov, aspect, near, far);
            let reversed = shadow_perspective(fov, aspect, near, far);
            for t in [0.0, 0.2, 0.5, 1.0] {
                let z = -(near + t * (far - near));
                let s = device_depth(standard, z);
                assert!((device_depth(reversed, z) - (1.0 - s)).abs() < 1e-5);
            }
        }
        let standard = ortho_rh(-3.0, 5.0, -2.0, 6.0, -1.0, 9.0);
        let reversed = shadow_ortho(-3.0, 5.0, -2.0, 6.0, -1.0, 9.0);
        for z in [1.0, 0.0, -4.0, -9.0] {
            let s = device_depth(standard, z);
            assert!((device_depth(reversed, z) - (1.0 - s)).abs() < 1e-6);
        }
    }

    // Nearer the light passes the write test at every depth a shadow view
    // covers, so the caster nearest the light keeps each texel.
    #[test]
    fn the_shadow_write_test_keeps_the_caster_nearest_the_light() {
        let spot = shadow_perspective(0.9, 1.0, 0.05, 40.0);
        let cascade = shadow_ortho(-16.0, 16.0, -16.0, 16.0, -80.0, 32.0);
        for (m, zs) in [
            (spot, [-0.06, -0.5, -3.0, -10.0, -25.0, -39.9]),
            (cascade, [79.0, 40.0, 0.0, -10.0, -20.0, -31.9]),
        ] {
            for pair in zs.windows(2) {
                let (nearer, farther) = (device_depth(m, pair[0]), device_depth(m, pair[1]));
                assert!(passes_write_test(nearer, farther), "{pair:?}");
            }
        }
    }

    // The camera has no far plane: depth keeps falling with distance and stays
    // in front of the far plane however far a surface is.
    #[test]
    fn camera_depth_approaches_the_far_plane_only_at_infinity() {
        for (fov, aspect, near) in [(1.2, 1.6, 0.1), (0.5, 2.35, 0.01), (1.6, 1.0, 2.0)] {
            let camera = camera_projection(fov, aspect, near);
            let mut prev = device_depth(camera, -near);
            assert_eq!(prev, DEPTH_NEAR);
            for distance in [0.5, 1.0, 10.0, 1.0e3, 1.0e5, 1.0e6, 1.0e7, 1.0e9, 1.0e12] {
                if distance <= near {
                    continue;
                }
                let d = device_depth(camera, -distance);
                assert!(passes_write_test(prev, d), "{prev} vs {d} at {distance}");
                assert!(passes_write_test(d, DEPTH_FAR), "{d} at {distance}");
                prev = d;
            }
            assert!(device_depth(camera, -1.0e9) < 1.0e-8);
        }
    }

    // A camera matrix keeps nearer surfaces passing its write test at every
    // distance, oblique clip included.
    #[test]
    fn the_camera_write_test_keeps_the_nearer_surface() {
        let proj = camera_projection(1.1, 1.7, 0.2);
        let oblique = camera_oblique_projection(proj, [0.0, 0.1, -1.0, -0.25]);
        for m in [proj, oblique] {
            let mut prev = device_depth(m, -0.3);
            for z in [-0.5, -1.0, -10.0, -40.0, -79.0, -5.0e3, -1.0e6] {
                let d = device_depth(m, z);
                assert!(passes_write_test(prev, d), "{prev} vs {d} at {z}");
                prev = d;
            }
        }
    }

    // The shader half of the convention has to name the same planes, the same
    // reductions, the same near-ward offset and the conservative-depth
    // direction the write test implies.
    #[test]
    fn the_shader_convention_matches() {
        let src = crate::render::shaders::DEPTH_CONVENTION;
        assert!(src.contains(&alloc::format!("#define DEPTH_NEAR {DEPTH_NEAR:?}\n")));
        assert!(src.contains(&alloc::format!("#define DEPTH_FAR {DEPTH_FAR:?}\n")));
        let reversed = DEPTH_NEAR > DEPTH_FAR;
        let (closer, farther) = if reversed {
            ("max", "min")
        } else {
            ("min", "max")
        };
        assert!(src.contains(&alloc::format!(
            "#define depth_closer(a, b) {closer}((a), (b))\n"
        )));
        assert!(src.contains(&alloc::format!(
            "#define depth_farther(a, b) {farther}((a), (b))\n"
        )));
        let toward_near = if reversed { '+' } else { '-' };
        assert!(src.contains(&alloc::format!(
            "#define depth_offset_near(d, offset) ((d) {toward_near} (offset))\n"
        )));
        let conservative = match DEPTH_WRITE_COMPARE {
            DepthCompare::Less | DepthCompare::LessEqual => "SV_DepthLessEqual",
            DepthCompare::Greater | DepthCompare::GreaterEqual => "SV_DepthGreaterEqual",
        };
        assert!(src.contains(&alloc::format!(
            "#define DEPTH_CONSERVATIVE {conservative}\n"
        )));
    }
}
