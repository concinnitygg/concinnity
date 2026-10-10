//! Which blades cast into the nearest shadow cascade. The kernel places them a
//! second time for the light: the same blades the view keeps, thinned by the
//! same distance from the camera, but culled against the cascade's light
//! frustum instead of the view's, so a blade off screen whose shadow falls on
//! screen still casts, and only out to [`GRASS_SHADOW_DISTANCE`] from the
//! camera. Every blade it keeps draws with the coarsest strip.

use crate::gfx::frustum::Frustum;

/// Distance from the camera, in meters, past which no blade casts.
pub const GRASS_SHADOW_DISTANCE: f32 = 24.0;

/// The nearest cascade of the directional light, as the grass casts into it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassShadowView {
    /// The cascade's light view-projection, column-major.
    pub vp: [[f32; 4]; 4],
    /// Unit vector toward the light.
    pub to_light: [f32; 3],
}

impl GrassShadowView {
    /// The light frustum the cascade's blades are culled against.
    pub fn frustum(&self) -> Frustum {
        Frustum::from_shadow(self.vp)
    }
}

/// Whether the blade rooted at `root`, `height` tall and `width` wide, casts
/// into the cascade whose frustum is `frustum` with the camera at `camera`:
/// it stands within the cast distance and its bounding sphere meets the light
/// frustum. Mirrors the kernel's per-blade test under the shadow block.
pub fn casts(frustum: &Frustum, camera: [f32; 3], root: [f32; 3], height: f32, width: f32) -> bool {
    let d = [
        root[0] - camera[0],
        root[1] - camera[1],
        root[2] - camera[2],
    ];
    if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > GRASS_SHADOW_DISTANCE * GRASS_SHADOW_DISTANCE {
        return false;
    }
    let center = [root[0], root[1] + 0.5 * height, root[2]];
    let radius = height + width;
    frustum.planes.iter().all(|p| {
        p.normal[0] * center[0] + p.normal[1] * center[1] + p.normal[2] * center[2] + p.d >= -radius
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::projection::look_at;
    use crate::gfx::render_types::NUM_SHADOW_CASCADES;
    use crate::render::csm::{ShadowUniformInputs, compute_shadow_uniforms};

    // The nearest cascade of a camera at eye height looking down -Z, lit by a
    // low sun from `to_light`.
    fn cascade(to_light: [f32; 3]) -> GrassShadowView {
        let eye = [0.0, 1.6, 0.0];
        let u = compute_shadow_uniforms(ShadowUniformInputs {
            view: look_at(eye, [0.0, 1.4, -1.0], [0.0, 1.0, 0.0]),
            cam_pos: eye,
            fov_y_rad: 60f32.to_radians(),
            aspect: 16.0 / 9.0,
            near: 0.1,
            shadow_distance: 80.0,
            light_dir_to_source: to_light,
            shadow_map_size: 2048,
            active_cascades: NUM_SHADOW_CASCADES as u32,
        });
        GrassShadowView {
            vp: u.light_vps[0],
            to_light,
        }
    }

    const CAMERA: [f32; 3] = [0.0, 1.6, 0.0];

    #[test]
    fn blades_in_view_near_the_camera_cast() {
        let f = cascade([0.3, 0.6, 0.2]).frustum();
        assert!(casts(&f, CAMERA, [0.0, 0.0, -4.0], 0.5, 0.03));
        assert!(casts(&f, CAMERA, [1.5, 0.0, -1.0], 0.5, 0.03));
    }

    // A blade just behind the camera is off screen, yet the cascade's box,
    // which bounds a sphere around the near slice, holds it: it casts, so its
    // shadow does not pop when the camera turns.
    #[test]
    fn blades_off_screen_inside_the_cascade_cast() {
        let f = cascade([0.3, 0.6, 0.2]).frustum();
        assert!(casts(&f, CAMERA, [0.0, 0.0, 1.0], 0.5, 0.03));
        assert!(casts(&f, CAMERA, [-3.0, 0.0, 0.5], 0.5, 0.03));
    }

    #[test]
    fn blades_past_the_cast_distance_do_not_cast() {
        let f = cascade([0.0, 1.0, 0.0]).frustum();
        let far = GRASS_SHADOW_DISTANCE + 1.0;
        assert!(!casts(&f, CAMERA, [0.0, 0.0, -far], 0.5, 0.03));
    }

    // Sideways of the near slice, beyond its bounding sphere, a blade falls
    // outside the cascade's box even within the cast distance.
    #[test]
    fn blades_outside_the_cascade_box_do_not_cast() {
        let f = cascade([0.0, 1.0, 0.0]).frustum();
        assert!(!casts(&f, CAMERA, [20.0, 0.0, 5.0], 0.5, 0.03));
    }

    // Lit from low over the camera's shoulder, the light frustum stretches
    // toward the sun: blades up-sun cast onto the slice the camera sees.
    #[test]
    fn a_low_sun_reaches_blades_up_sun() {
        let f = cascade([0.0, 0.2, 1.0]).frustum();
        assert!(casts(&f, CAMERA, [0.0, 0.0, 6.0], 0.5, 0.03));
    }
}
