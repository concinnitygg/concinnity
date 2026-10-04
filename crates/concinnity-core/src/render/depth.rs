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
//! against the other's depth.

use crate::gfx::projection::{oblique_rh, ortho_rh, perspective_rh};
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
            Self::Camera | Self::Shadow => 0.0,
        }
    }

    /// Device depth at the far plane.
    pub const fn far(self) -> f32 {
        match self {
            Self::Camera | Self::Shadow => 1.0,
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
            Self::Camera | Self::Shadow => DepthCompare::Less,
        }
    }

    /// The test that also passes at equal depth: a read-only test against depth
    /// another pass already wrote, and a pass whose shader replaces its
    /// fragment depth with a value no farther than the rasterized one.
    pub const fn inclusive_compare(self) -> DepthCompare {
        match self {
            Self::Camera | Self::Shadow => DepthCompare::LessEqual,
        }
    }

    /// Whether device depth `a` is strictly nearer than `b`.
    pub fn is_closer(self, a: f32, b: f32) -> bool {
        match self {
            Self::Camera | Self::Shadow => a < b,
        }
    }

    /// The nearer of two device depths.
    pub fn closer(self, a: f32, b: f32) -> f32 {
        if self.is_closer(b, a) { b } else { a }
    }
}

/// The camera's perspective projection: the main camera and the reflection
/// probe faces. `fov_y_radians` is the full vertical field of view; `aspect` is
/// width over height.
pub fn camera_projection(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    perspective_rh(fov_y_radians, aspect, near, far)
}

/// A camera projection whose near clip plane is replaced by `clip_plane`, given
/// in the projection's view space, so everything on its negative side clips.
/// The far plane is kept. A planar reflection uses it to clip the geometry
/// behind its mirror.
pub(crate) fn camera_oblique_projection(proj: Mat4, clip_plane: [f32; 4]) -> Mat4 {
    oblique_rh(proj, clip_plane)
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
    fn both_conventions_are_standard_depth() {
        for c in BOTH {
            assert_eq!(c.near(), 0.0);
            assert_eq!(c.far(), 1.0);
            assert_eq!(c.clear(), 1.0);
            assert_eq!(c.write_compare(), DepthCompare::Less);
            assert_eq!(c.inclusive_compare(), DepthCompare::LessEqual);
        }
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
            assert_eq!(c.closer(0.25, 0.75), 0.25);
            assert_eq!(c.closer(0.5, 0.5), 0.5);
            assert!(!c.is_closer(0.5, 0.5));
            assert!(!c.is_closer(c.far(), c.near()));
        }
    }

    // The projections must map their near and far planes onto the device depth
    // their convention names.
    #[test]
    fn projections_land_on_the_convention_planes() {
        let camera = camera_projection(1.2, 1.6, 0.1, 500.0);
        let near = device_depth(camera, -0.1);
        let far = device_depth(camera, -500.0);
        assert!((near - DepthConvention::Camera.near()).abs() < 1e-4);
        assert!((far - DepthConvention::Camera.far()).abs() < 1e-4);

        let spot = shadow_perspective(1.0, 1.0, 0.05, 40.0);
        assert!((device_depth(spot, -0.05) - DepthConvention::Shadow.near()).abs() < 1e-4);
        assert!((device_depth(spot, -40.0) - DepthConvention::Shadow.far()).abs() < 1e-4);

        let cascade = shadow_ortho(-4.0, 4.0, -4.0, 4.0, -2.0, 8.0);
        assert!((device_depth(cascade, 2.0) - DepthConvention::Shadow.near()).abs() < 1e-5);
        assert!((device_depth(cascade, -8.0) - DepthConvention::Shadow.far()).abs() < 1e-5);
    }

    // The shader half of the camera convention has to name the same planes and
    // the conservative-depth direction its write test implies.
    #[test]
    fn the_shader_convention_matches() {
        let src = crate::render::shaders::DEPTH_CONVENTION;
        let camera = DepthConvention::Camera;
        assert!(src.contains(&alloc::format!("#define DEPTH_NEAR {:?}\n", camera.near())));
        assert!(src.contains(&alloc::format!("#define DEPTH_FAR {:?}\n", camera.far())));
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

    // The entry points carry the same matrices as the generic builders they
    // wrap, bit for bit.
    #[test]
    fn the_entry_points_match_the_generic_builders() {
        for (fov, aspect, near, far) in [(1.2, 1.6, 0.1, 500.0), (0.4, 0.5, 0.01, 10.0)] {
            assert_eq!(
                bits(camera_projection(fov, aspect, near, far)),
                bits(perspective_rh(fov, aspect, near, far))
            );
            assert_eq!(
                bits(shadow_perspective(fov, aspect, near, far)),
                bits(perspective_rh(fov, aspect, near, far))
            );
        }
        assert_eq!(
            bits(shadow_ortho(-3.0, 5.0, -2.0, 6.0, -1.0, 9.0)),
            bits(ortho_rh(-3.0, 5.0, -2.0, 6.0, -1.0, 9.0))
        );
        let proj = perspective_rh(1.1, 1.7, 0.2, 80.0);
        let plane = [0.1, 0.9, -0.3, 2.0];
        assert_eq!(
            bits(camera_oblique_projection(proj, plane)),
            bits(oblique_rh(proj, plane))
        );
    }
}
