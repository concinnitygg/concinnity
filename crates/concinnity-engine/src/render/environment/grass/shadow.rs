//! Which blades cast into the shadow cascades. The kernel places them a second
//! time for the light: the same blades the view keeps, thinned by the same
//! distance from the camera, but culled against a light-space box around every
//! cascade they cast into instead of the view's frustum, so a blade off screen
//! whose shadow falls on screen still casts, and only out to
//! [`GRASS_SHADOW_DISTANCE`] from the camera. Every cascade that starts within
//! that distance draws the placed blades through its own projection, so the
//! blades shadow the ground wherever they stand, whichever cascade a receiver
//! samples. Every blade it keeps draws with the coarsest strip.

use crate::gfx::frustum::Frustum;
use crate::gfx::render_types::{NUM_SHADOW_CASCADES, ShadowUniforms};
use concinnity_core::math::sqrt;
use concinnity_core::math::vec3::{dot, length_sq, scale};

/// Distance from the camera, in meters, past which no blade casts.
pub const GRASS_SHADOW_DISTANCE: f32 = 24.0;

/// The live cascades of `uniforms` that start within
/// [`GRASS_SHADOW_DISTANCE`] of the camera, one bit each: the cascades a
/// receiver near enough to be shadowed by a blade can sample.
pub fn grass_cascades(uniforms: &ShadowUniforms) -> u32 {
    let live = (uniforms.active_cascades as usize).min(NUM_SHADOW_CASCADES);
    let mut mask = 0u32;
    let mut near = 0.0_f32;
    for (i, split) in uniforms.cascade_splits.iter().take(live).enumerate() {
        if near < GRASS_SHADOW_DISTANCE {
            mask |= 1 << i;
        }
        near = *split;
    }
    mask
}

/// The cascades the grass casts into this frame, as the kernel places their
/// blades.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassShadowView {
    /// An orthographic light view-projection whose box holds every casting
    /// cascade's, column-major: the blades are culled against it.
    pub vp: [[f32; 4]; 4],
    /// Unit vector toward the light.
    pub to_light: [f32; 3],
}

impl GrassShadowView {
    /// The view for the cascades of `uniforms` in `cascades` (one bit each) lit
    /// from `to_light`, or `None` when the set is empty.
    pub fn new(uniforms: &ShadowUniforms, cascades: u32, to_light: [f32; 3]) -> Option<Self> {
        let vps = uniforms
            .light_vps
            .iter()
            .enumerate()
            .filter(|(i, _)| cascades & (1 << i) != 0)
            .map(|(_, vp)| *vp);
        cast_box(vps).map(|vp| Self { vp, to_light })
    }

    /// The light frustum the casting blades are culled against.
    pub fn frustum(&self) -> Frustum {
        Frustum::from_shadow(self.vp)
    }
}

// The range of `dot(axis, p)` one orthographic row maps onto `ndc`, with the
// row's unit axis.
fn row_range(vp: &[[f32; 4]; 4], row: usize, ndc: [f32; 2]) -> ([f32; 3], [f32; 2]) {
    let xyz = [vp[0][row], vp[1][row], vp[2][row]];
    let len = sqrt(length_sq(xyz)).max(1e-12);
    let w = vp[3][row];
    let a = (ndc[0] - w) / len;
    let b = (ndc[1] - w) / len;
    (scale(xyz, 1.0 / len), [a.min(b), a.max(b)])
}

/// The orthographic view-projection whose box holds the boxes of every one of
/// `vps`, cascades of one light sharing its axes, or `None` for none. It maps
/// the union onto the same clip ranges each of them uses: x and y over
/// [-1, 1], and reversed depth over [0, 1] rising toward the light.
pub fn cast_box(vps: impl IntoIterator<Item = [[f32; 4]; 4]>) -> Option<[[f32; 4]; 4]> {
    const RANGES: [[f32; 2]; 3] = [[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0]];
    let mut axes = [[0.0_f32; 3]; 3];
    let mut bounds = [[f32::INFINITY, f32::NEG_INFINITY]; 3];
    let mut any = false;
    for vp in vps {
        for (row, ndc) in RANGES.iter().enumerate() {
            let (axis, [lo, hi]) = row_range(&vp, row, *ndc);
            if !any {
                axes[row] = axis;
            }
            // Cascades of one light share their axes, so the range along the
            // first one's axis is the cascade's own.
            let along = dot(axis, axes[row]);
            let (lo, hi) = if along < 0.0 { (-hi, -lo) } else { (lo, hi) };
            bounds[row] = [bounds[row][0].min(lo), bounds[row][1].max(hi)];
        }
        any = true;
    }
    if !any {
        return None;
    }
    // Row r maps `dot(axis, p)` over its bounds linearly onto RANGES[r].
    let mut out = [[0.0_f32; 4]; 4];
    for (row, ndc) in RANGES.iter().enumerate() {
        let [lo, hi] = bounds[row];
        let k = (ndc[1] - ndc[0]) / (hi - lo).max(1e-6);
        for (col, c) in axes[row].iter().enumerate() {
            out[col][row] = c * k;
        }
        out[3][row] = ndc[0] - lo * k;
    }
    out[3][3] = 1.0;
    Some(out)
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
    use crate::render::shadow::csm::{ShadowUniformInputs, compute_shadow_uniforms};
    use concinnity_core::math::SaturatingFrom;

    const CAMERA: [f32; 3] = [0.0, 1.6, 0.0];

    // The cascades of a camera at eye height looking down -Z, lit from
    // `to_light`, reaching `reach`.
    fn uniforms(to_light: [f32; 3], reach: f32, cascades: u32) -> ShadowUniforms {
        compute_shadow_uniforms(ShadowUniformInputs {
            view: look_at(CAMERA, [0.0, 1.4, -1.0], [0.0, 1.0, 0.0]),
            cam_pos: CAMERA,
            fov_y_rad: 60f32.to_radians(),
            aspect: 16.0 / 9.0,
            near: 0.1,
            shadow_distance: reach,
            light_dir_to_source: to_light,
            shadow_map_size: 2048,
            active_cascades: cascades,
        })
    }

    fn view(to_light: [f32; 3], reach: f32) -> GrassShadowView {
        let u = uniforms(to_light, reach, u32::saturating_from(NUM_SHADOW_CASCADES));
        GrassShadowView::new(&u, grass_cascades(&u), to_light).expect("the grass casts")
    }

    fn clip(vp: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
        core::array::from_fn(|r| vp[0][r] * p[0] + vp[1][r] * p[1] + vp[2][r] * p[2] + vp[3][r])
    }

    fn inside(vp: &[[f32; 4]; 4], p: [f32; 3], slack: f32) -> bool {
        let c = clip(vp, p);
        c[0].abs() <= 1.0 + slack
            && c[1].abs() <= 1.0 + slack
            && (-slack..=1.0 + slack).contains(&c[2])
    }

    #[test]
    fn every_cascade_starting_within_the_cast_distance_casts() {
        let light = [0.3, 0.6, 0.2];
        let n = u32::saturating_from(NUM_SHADOW_CASCADES);
        // At 80 m the last cascade starts at about 22 m, inside the reach.
        assert_eq!(grass_cascades(&uniforms(light, 80.0, n)), 0b1111);
        // At 320 m the third starts at about 29 m.
        assert_eq!(grass_cascades(&uniforms(light, 320.0, n)), 0b0011);
        // Only live cascades cast.
        assert_eq!(grass_cascades(&uniforms(light, 80.0, 2)), 0b0011);
        // Without cascades, the fallback's one slot.
        let empty = crate::render::shadow::csm::empty_shadow_uniforms();
        assert_eq!(grass_cascades(&empty), 0b0001);
    }

    #[test]
    fn no_cascade_casts_no_view() {
        let u = uniforms([0.3, 0.6, 0.2], 80.0, 4);
        assert_eq!(GrassShadowView::new(&u, 0, [0.3, 0.6, 0.2]), None);
        assert_eq!(cast_box(core::iter::empty()), None);
    }

    // Whatever a casting cascade's box holds, the cast box holds too, so no
    // blade a cascade draws is culled before it is placed.
    #[test]
    fn the_cast_box_holds_every_casting_cascade() {
        for light in [[0.3, 0.6, 0.2], [0.0, 0.2, 1.0], [-0.7, 0.3, -0.4]] {
            for reach in [20.0, 80.0, 320.0] {
                let u = uniforms(light, reach, 4);
                let mask = grass_cascades(&u);
                let cast = GrassShadowView::new(&u, mask, light).unwrap().vp;
                let mut held = 0;
                for x in -12..=12 {
                    for y in -4..=8 {
                        for z in -12..=12 {
                            let p = [x as f32 * 4.0, y as f32 * 4.0, z as f32 * 4.0];
                            let in_any = (0..NUM_SHADOW_CASCADES)
                                .filter(|i| mask & (1 << i) != 0)
                                .any(|i| inside(&u.light_vps[i], p, 0.0));
                            if in_any {
                                held += 1;
                                assert!(inside(&cast, p, 1e-3), "{light:?} {reach} {p:?}");
                            }
                        }
                    }
                }
                assert!(held > 0);
            }
        }
    }

    #[test]
    fn blades_in_view_near_the_camera_cast() {
        let f = view([0.3, 0.6, 0.2], 80.0).frustum();
        assert!(casts(&f, CAMERA, [0.0, 0.0, -4.0], 0.5, 0.03));
        assert!(casts(&f, CAMERA, [1.5, 0.0, -1.0], 0.5, 0.03));
    }

    // Past the first cascade, a blade still casts into the cascade its
    // shadow falls in.
    #[test]
    fn blades_beyond_the_first_cascade_cast() {
        let light = [0.3, 0.6, 0.2];
        let u = uniforms(light, 80.0, 4);
        let blade = [0.0, 0.0, -15.0];
        assert!(!inside(&u.light_vps[0], blade, 0.0));
        let f = view(light, 80.0).frustum();
        assert!(casts(&f, CAMERA, blade, 0.5, 0.03));
    }

    // A blade just behind the camera is off screen, yet the cascades' boxes,
    // which bound spheres around their slices, hold it: it casts, so its
    // shadow does not pop when the camera turns.
    #[test]
    fn blades_off_screen_inside_the_cascades_cast() {
        let f = view([0.3, 0.6, 0.2], 80.0).frustum();
        assert!(casts(&f, CAMERA, [0.0, 0.0, 1.0], 0.5, 0.03));
        assert!(casts(&f, CAMERA, [-3.0, 0.0, 0.5], 0.5, 0.03));
    }

    #[test]
    fn blades_past_the_cast_distance_do_not_cast() {
        let f = view([0.0, 1.0, 0.0], 80.0).frustum();
        let far = GRASS_SHADOW_DISTANCE + 1.0;
        assert!(!casts(&f, CAMERA, [0.0, 0.0, -far], 0.5, 0.03));
    }

    // Behind the camera, beyond every casting cascade's sphere, a blade falls
    // outside the cast box even within the cast distance.
    #[test]
    fn blades_outside_every_casting_cascade_do_not_cast() {
        let light = [0.0, 1.0, 0.0];
        let u = uniforms(light, 20.0, 4);
        let blade = [0.0, 0.0, 22.0];
        assert!((0..NUM_SHADOW_CASCADES).all(|i| !inside(&u.light_vps[i], blade, 0.0)));
        let f = view(light, 20.0).frustum();
        assert!(!casts(&f, CAMERA, blade, 0.5, 0.03));
    }

    // Lit from low over the camera's shoulder, the light frustum stretches
    // toward the sun: blades up-sun cast onto the slice the camera sees.
    #[test]
    fn a_low_sun_reaches_blades_up_sun() {
        let f = view([0.0, 0.2, 1.0], 80.0).frustum();
        assert!(casts(&f, CAMERA, [0.0, 0.0, 6.0], 0.5, 0.03));
    }
}
