//! Depth-buffer conventions: which end of the device-depth range is near, what a
//! depth target clears to, which comparison keeps the nearer fragment, and the
//! projections that produce the depth in the first place.
//!
//! Every depth clear, depth-test state and projection in the engine reads its
//! convention from here instead of spelling a literal, so the mapping from
//! distance to device depth has one source per target family. The shader-side
//! counterpart is `shaders/depth_convention.hlsl`.
//!
//! The camera and the shadow maps are separate conventions: no pass tests one
//! against the other's depth. Camera depth is reversed and infinite (near 1,
//! infinity 0), which on a 32-bit float buffer spreads precision evenly over
//! distance and leaves the camera no far clip; shadow depth is standard (near 0,
//! far 1) over a finite range.

use crate::gfx::projection::{
    ortho_rh, perspective_rh, reversed_infinite_oblique_rh, reversed_infinite_perspective_rh,
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

/// How a family of depth targets maps distance to device depth.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DepthConvention {
    /// The main camera and every target tested against its depth: the G-buffer
    /// pre-pass, reflection-probe faces, planar reflections, and the
    /// transparent, glass, water and raymarched passes.
    Camera,
    /// The directional shadow cascades, the spot-light shadow slices and the
    /// raymarched shadow casters drawn into them.
    Shadow,
}

impl DepthConvention {
    /// Device depth at the near plane.
    pub const fn near(self) -> f32 {
        match self {
            Self::Camera => 1.0,
            Self::Shadow => 0.0,
        }
    }

    /// Device depth at the far plane, which for an infinite convention is
    /// infinity.
    pub const fn far(self) -> f32 {
        match self {
            Self::Camera => 0.0,
            Self::Shadow => 1.0,
        }
    }

    /// Whether the near plane sits at device depth 1 and the far plane at 0:
    /// what an upscaler's inverted-depth flag asks.
    pub const fn is_reversed(self) -> bool {
        match self {
            Self::Camera => true,
            Self::Shadow => false,
        }
    }

    /// Whether the projection has no far plane: depth reaches [`Self::far`]
    /// only at infinity, which is what an upscaler's infinite-depth flag asks.
    pub const fn is_infinite(self) -> bool {
        match self {
            Self::Camera => true,
            Self::Shadow => false,
        }
    }

    /// The value a depth target clears to: the far plane, so any surface drawn
    /// wins over the cleared background.
    pub const fn clear(self) -> f32 {
        self.far()
    }

    /// The test a depth-writing pass draws with. Only a strictly nearer
    /// fragment passes, so the first of two coplanar draws keeps the pixel.
    pub const fn write_compare(self) -> DepthCompare {
        match self {
            Self::Camera => DepthCompare::Greater,
            Self::Shadow => DepthCompare::Less,
        }
    }

    /// The test that also passes at equal depth: a read-only test against depth
    /// another pass already wrote, and a pass whose shader replaces its
    /// fragment depth with a value no farther than the rasterized one.
    pub const fn inclusive_compare(self) -> DepthCompare {
        match self {
            Self::Camera => DepthCompare::GreaterEqual,
            Self::Shadow => DepthCompare::LessEqual,
        }
    }

    /// Whether device depth `a` is strictly nearer than `b`.
    pub fn is_closer(self, a: f32, b: f32) -> bool {
        if self.is_reversed() { a > b } else { a < b }
    }

    /// The nearer of two device depths.
    pub fn closer(self, a: f32, b: f32) -> f32 {
        if self.is_closer(b, a) { b } else { a }
    }
}

/// The camera's perspective projection: the main camera and the reflection
/// probe faces. `fov_y_radians` is the full vertical field of view; `aspect` is
/// width over height. Depth is reversed and infinite: `near` maps to 1, and
/// depth falls toward 0 with distance, reaching it only at infinity.
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

/// A spot light's perspective shadow projection.
pub(crate) fn shadow_perspective(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    perspective_rh(fov_y_radians, aspect, near, far)
}

/// A directional cascade's orthographic shadow projection over the given
/// light-space box.
pub(crate) fn shadow_ortho(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> Mat4 {
    ortho_rh(left, right, bottom, top, near, far)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOTH: [DepthConvention; 2] = [DepthConvention::Camera, DepthConvention::Shadow];

    fn bits(m: Mat4) -> [[u32; 4]; 4] {
        m.map(|col| col.map(f32::to_bits))
    }

    fn device_depth(m: Mat4, view_z: f32) -> f32 {
        let z = m[2][2] * view_z + m[3][2];
        let w = m[2][3] * view_z + m[3][3];
        z / w
    }

    #[test]
    fn the_camera_is_reversed_and_shadows_are_standard() {
        let camera = DepthConvention::Camera;
        assert!(camera.is_reversed());
        assert!(camera.is_infinite());
        assert_eq!(
            (camera.near(), camera.far(), camera.clear()),
            (1.0, 0.0, 0.0)
        );
        assert_eq!(camera.write_compare(), DepthCompare::Greater);
        assert_eq!(camera.inclusive_compare(), DepthCompare::GreaterEqual);

        let shadow = DepthConvention::Shadow;
        assert!(!shadow.is_reversed());
        assert!(!shadow.is_infinite());
        assert_eq!(
            (shadow.near(), shadow.far(), shadow.clear()),
            (0.0, 1.0, 1.0)
        );
        assert_eq!(shadow.write_compare(), DepthCompare::Less);
        assert_eq!(shadow.inclusive_compare(), DepthCompare::LessEqual);
    }

    // A cleared pixel must lose to every surface a write test can produce.
    #[test]
    fn the_clear_is_the_far_plane() {
        for c in BOTH {
            assert_eq!(c.clear(), c.far());
            assert!(c.is_closer(c.near(), c.clear()));
            assert!(c.is_closer(0.5, c.clear()));
        }
    }

    #[test]
    fn closer_picks_the_nearer_depth() {
        for c in BOTH {
            assert_eq!(c.closer(c.near(), c.far()), c.near());
            assert_eq!(c.closer(c.far(), c.near()), c.near());
            let toward_near = if c.is_reversed() { 0.75 } else { 0.25 };
            assert_eq!(c.closer(0.25, 0.75), toward_near);
            assert_eq!(c.closer(0.5, 0.5), 0.5);
            assert!(!c.is_closer(0.5, 0.5));
            assert!(!c.is_closer(c.far(), c.near()));
        }
    }

    // The projections must map their near and far planes onto the device depth
    // their convention names.
    #[test]
    fn projections_land_on_the_convention_planes() {
        let camera = camera_projection(1.2, 1.6, 0.1);
        assert_eq!(device_depth(camera, -0.1), DepthConvention::Camera.near());

        let spot = shadow_perspective(1.0, 1.0, 0.05, 40.0);
        assert!((device_depth(spot, -0.05) - DepthConvention::Shadow.near()).abs() < 1e-4);
        assert!((device_depth(spot, -40.0) - DepthConvention::Shadow.far()).abs() < 1e-4);

        let cascade = shadow_ortho(-4.0, 4.0, -4.0, 4.0, -2.0, 8.0);
        assert!((device_depth(cascade, 2.0) - DepthConvention::Shadow.near()).abs() < 1e-5);
        assert!((device_depth(cascade, -8.0) - DepthConvention::Shadow.far()).abs() < 1e-5);
    }

    // The camera has no far plane: depth keeps falling with distance and stays
    // in front of the far plane however far a surface is.
    #[test]
    fn camera_depth_approaches_the_far_plane_only_at_infinity() {
        for (fov, aspect, near) in [(1.2, 1.6, 0.1), (0.5, 2.35, 0.01), (1.6, 1.0, 2.0)] {
            let camera = camera_projection(fov, aspect, near);
            let mut prev = device_depth(camera, -near);
            assert_eq!(prev, 1.0);
            for distance in [0.5, 1.0, 10.0, 1.0e3, 1.0e5, 1.0e6, 1.0e7, 1.0e9, 1.0e12] {
                if distance <= near {
                    continue;
                }
                let d = device_depth(camera, -distance);
                assert!(
                    DepthConvention::Camera.is_closer(prev, d),
                    "{prev} vs {d} at {distance}"
                );
                assert!(
                    DepthConvention::Camera.is_closer(d, DepthConvention::Camera.far()),
                    "{d} at {distance}"
                );
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
                assert!(
                    DepthConvention::Camera.is_closer(prev, d),
                    "{prev} vs {d} at {z}"
                );
                prev = d;
            }
        }
    }

    // The shader half of the camera convention has to name the same planes, the
    // same reductions and the conservative-depth direction its write test
    // implies.
    #[test]
    fn the_shader_convention_matches() {
        let src = crate::render::shaders::DEPTH_CONVENTION;
        let camera = DepthConvention::Camera;
        assert!(src.contains(&alloc::format!("#define DEPTH_NEAR {:?}\n", camera.near())));
        assert!(src.contains(&alloc::format!("#define DEPTH_FAR {:?}\n", camera.far())));
        let (closer, farther) = if camera.is_reversed() {
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
        let conservative = |compare| match compare {
            DepthCompare::Less | DepthCompare::LessEqual => "SV_DepthLessEqual",
            DepthCompare::Greater | DepthCompare::GreaterEqual => "SV_DepthGreaterEqual",
        };
        assert!(src.contains(&alloc::format!(
            "#define CAMERA_DEPTH_CONSERVATIVE {}\n",
            conservative(camera.write_compare())
        )));
        assert!(src.contains(&alloc::format!(
            "#define SHADOW_DEPTH_CONSERVATIVE {}\n",
            conservative(DepthConvention::Shadow.write_compare())
        )));
    }

    // The shadow entry points still carry the standard-depth builders'
    // matrices, bit for bit.
    #[test]
    fn the_shadow_entry_points_stay_standard() {
        for (fov, aspect, near, far) in [(1.2, 1.6, 0.1, 500.0), (0.4, 0.5, 0.01, 10.0)] {
            assert_eq!(
                bits(shadow_perspective(fov, aspect, near, far)),
                bits(perspective_rh(fov, aspect, near, far))
            );
        }
        assert_eq!(
            bits(shadow_ortho(-3.0, 5.0, -2.0, 6.0, -1.0, 9.0)),
            bits(ortho_rh(-3.0, 5.0, -2.0, 6.0, -1.0, 9.0))
        );
    }
}
