//! Which blades cast into the shadow cascades. The kernel places them a second
//! time for the light: the same blades the view keeps, thinned by the same
//! distance from the camera, but culled against a light-space box instead of
//! the view's frustum, so a blade off screen whose shadow falls on screen
//! still casts, and only out to [`GRASS_SHADOW_DISTANCE`] from the camera. The
//! box holds every cascade the grass casts into. When all of them are drawn
//! for the frame alone, it is cut down to what the frame's views see out to
//! as far as a blade's shadow reaches: a shadow lands where its caster lies
//! across the light, and only on what lies farther from the light, so a blade
//! outside that span shadows nothing in view. Every cascade sampled within the
//! distance draws the placed blades through its own projection, so the blades
//! shadow the ground wherever they stand, whichever cascade a receiver
//! samples. Every blade it keeps draws with the coarsest strip.

use crate::gfx::frustum::Frustum;
use crate::gfx::render_types::ShadowUniforms;
use crate::render::shadow::csm::cascades_sampled_within;
use crate::render::shadow::receivers::{ReceiverCut, cast_box};
use concinnity_core::math::sqrt;

/// Distance from the camera, in meters, past which no blade casts.
pub const GRASS_SHADOW_DISTANCE: f32 = 24.0;

/// The live cascades of `uniforms` sampled within [`GRASS_SHADOW_DISTANCE`]
/// of the camera, one bit each: the cascades a receiver near enough to be
/// shadowed by a blade can sample.
pub fn grass_cascades(uniforms: &ShadowUniforms) -> u32 {
    cascades_sampled_within(uniforms, GRASS_SHADOW_DISTANCE)
}

/// How far from the camera, as view depth, the shadow of a blade up to
/// `tallest` meters high can land with the light toward `to_light`: past the
/// cast distance by the blade's shadow on level ground, and no farther than
/// the last cascade of `uniforms` the grass casts into.
pub fn grass_receiver_depth(uniforms: &ShadowUniforms, to_light: [f32; 3], tallest: f32) -> f32 {
    let cascades = grass_cascades(uniforms);
    let last = (u32::BITS - 1).saturating_sub(cascades.leading_zeros()) as usize;
    let limit = uniforms.cascade_splits[last.min(uniforms.cascade_splits.len() - 1)];
    // The frame's +Y is the ground's up near the camera.
    let [x, y, z] = to_light;
    let across = sqrt(x * x + z * z);
    let shadow = if y > 0.0 {
        tallest * across / y
    } else {
        f32::INFINITY
    };
    (GRASS_SHADOW_DISTANCE + shadow)
        .min(limit)
        .max(GRASS_SHADOW_DISTANCE.min(limit))
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
    /// from `to_light`, holding only the casters of `cut`'s receivers when
    /// given, or `None` when no blade can cast.
    pub(crate) fn new(
        uniforms: &ShadowUniforms,
        cascades: u32,
        to_light: [f32; 3],
        cut: Option<ReceiverCut<'_>>,
    ) -> Option<Self> {
        let vps = uniforms
            .light_vps
            .iter()
            .enumerate()
            .filter(|(i, _)| cascades & (1 << i) != 0)
            .map(|(_, vp)| *vp);
        cast_box(vps, cut).map(|vp| Self { vp, to_light })
    }

    /// The light frustum the casting blades are culled against.
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
    use crate::render::shadow::csm::{ShadowUniformInputs, compute_shadow_uniforms};
    use crate::render::shadow::schedule::CascadeCamera;
    use concinnity_core::math::SaturatingFrom;
    use concinnity_core::math::vec3::normalize_clamped;

    const CAMERA: [f32; 3] = [0.0, 1.6, 0.0];

    const TALLEST: f32 = 1.0;

    fn camera() -> CascadeCamera {
        CascadeCamera {
            view: look_at(CAMERA, [0.0, 1.4, -1.0], [0.0, 1.0, 0.0]),
            position: CAMERA,
            fov_y_rad: 60f32.to_radians(),
            aspect: 16.0 / 9.0,
            near: 0.1,
            view_distance: None,
            focus_depth: None,
        }
    }

    // The cascades of a camera at eye height looking down -Z, lit from
    // `to_light`, reaching `reach`.
    fn uniforms(to_light: [f32; 3], reach: f32, cascades: u32) -> ShadowUniforms {
        let camera = camera();
        compute_shadow_uniforms(ShadowUniformInputs {
            view: camera.view,
            cam_pos: camera.position,
            fov_y_rad: camera.fov_y_rad,
            aspect: camera.aspect,
            near: camera.near,
            shadow_distance: reach,
            light_dir_to_source: to_light,
            shadow_map_size: 2048,
            active_cascades: cascades,
        })
    }

    // What the camera sees as far as a blade's shadow reaches.
    fn receivers(u: &ShadowUniforms, to_light: [f32; 3]) -> [[f32; 3]; 8] {
        let toward = normalize_clamped(to_light, 1e-6);
        camera()
            .reader()
            .corners(0.0, grass_receiver_depth(u, toward, TALLEST))
    }

    fn cut(receivers: &[[f32; 3]]) -> ReceiverCut<'_> {
        ReceiverCut {
            receivers,
            map_size: 2048,
        }
    }

    // The cast box for cascades the schedule may keep: every one whole.
    fn view(to_light: [f32; 3], reach: f32) -> GrassShadowView {
        let u = uniforms(to_light, reach, u32::saturating_from(NUM_SHADOW_CASCADES));
        GrassShadowView::new(&u, grass_cascades(&u), to_light, None).expect("the grass casts")
    }

    // The cast box for cascades drawn for this frame alone.
    fn cut_view(to_light: [f32; 3], reach: f32) -> GrassShadowView {
        let u = uniforms(to_light, reach, u32::saturating_from(NUM_SHADOW_CASCADES));
        let r = receivers(&u, to_light);
        GrassShadowView::new(&u, grass_cascades(&u), to_light, Some(cut(&r)))
            .expect("the grass casts")
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

    // A point of a view's slice, at (u, v, t) across, up and along it.
    fn in_slice(corners: &[[f32; 3]; 8], [u, v, t]: [f32; 3]) -> [f32; 3] {
        let lerp = |a: [f32; 3], b: [f32; 3], f: f32| -> [f32; 3] {
            core::array::from_fn(|i| a[i] + (b[i] - a[i]) * f)
        };
        let quad = |c: &[[f32; 3]]| lerp(lerp(c[0], c[1], u), lerp(c[3], c[2], u), v);
        lerp(quad(&corners[..4]), quad(&corners[4..]), t)
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
        let light = [0.3, 0.6, 0.2];
        let u = uniforms(light, 80.0, 4);
        let r = receivers(&u, light);
        assert_eq!(GrassShadowView::new(&u, 0, light, None), None);
        assert_eq!(GrassShadowView::new(&u, 0, light, Some(cut(&r))), None);
        assert_eq!(cast_box([u.light_vps[0]], Some(cut(&[]))), None);
    }

    // A high sun keeps the receivers at the cast distance; a low one stretches
    // them by the tallest blade's shadow, no farther than the grass's last
    // cascade.
    #[test]
    fn the_receivers_reach_as_far_as_a_blade_s_shadow() {
        let u = uniforms([0.0, 1.0, 0.0], 320.0, 4);
        let overhead = grass_receiver_depth(&u, [0.0, 1.0, 0.0], TALLEST);
        assert_eq!(overhead, GRASS_SHADOW_DISTANCE);
        let low = normalize_clamped([0.0, 0.25, 1.0], 1e-6);
        let d = grass_receiver_depth(&u, low, TALLEST);
        assert!((d - (GRASS_SHADOW_DISTANCE + 4.0)).abs() < 1e-3, "{d}");
        let last = 31 - grass_cascades(&u).leading_zeros() as usize;
        let limit = u.cascade_splits[last];
        assert_eq!(grass_receiver_depth(&u, [0.0, 0.0, 1.0], TALLEST), limit);
        assert_eq!(grass_receiver_depth(&u, [0.0, -0.5, 1.0], TALLEST), limit);
    }

    // Whatever a casting cascade's box holds, the uncut cast box holds too, so
    // no blade a kept cascade draws is culled before it is placed.
    #[test]
    fn the_cast_box_holds_every_casting_cascade() {
        for light in [[0.3, 0.6, 0.2], [0.0, 0.2, 1.0], [-0.7, 0.3, -0.4]] {
            for reach in [20.0, 80.0, 320.0] {
                let u = uniforms(light, reach, 4);
                let mask = grass_cascades(&u);
                let cast = GrassShadowView::new(&u, mask, light, None).unwrap().vp;
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

    // Every point of a casting cascade toward the light from a receiver in
    // view is a caster of that receiver, and the cut cast box holds it, so no
    // blade whose shadow lands in view is culled before it is placed.
    #[test]
    fn the_cut_cast_box_holds_every_caster_of_a_receiver_in_view() {
        let steps = [0.0, 0.25, 0.5, 0.75, 1.0];
        for light in [
            [0.3, 0.6, 0.2],
            [0.0, 0.2, 1.0],
            [-0.7, 0.3, -0.4],
            [0.0, 1.0, 0.0],
        ] {
            let toward = normalize_clamped(light, 1e-6);
            for reach in [20.0, 80.0, 320.0] {
                let u = uniforms(light, reach, 4);
                let corners = receivers(&u, light);
                let mask = grass_cascades(&u);
                let cast = GrassShadowView::new(&u, mask, light, Some(cut(&corners)))
                    .unwrap()
                    .vp;
                let mut held = 0;
                for &a in &steps {
                    for &b in &steps {
                        for &t in &steps {
                            let receiver = in_slice(&corners, [a, b, t]);
                            for k in 0..20 {
                                let s = k as f32 * 2.0;
                                let p = core::array::from_fn(|i| receiver[i] + toward[i] * s);
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
                }
                assert!(held > 0);
            }
        }
    }

    // The cut leaves out blades the casting cascades hold whose shadows fall
    // away from the view.
    #[test]
    fn the_cut_cast_box_leaves_out_casters_of_nothing_in_view() {
        let light = [0.0, 0.6, -1.0];
        let u = uniforms(light, 80.0, 4);
        let mask = grass_cascades(&u);
        let r = receivers(&u, light);
        let cast = GrassShadowView::new(&u, mask, light, Some(cut(&r)))
            .unwrap()
            .vp;
        let behind = [0.0, 0.3, 3.0];
        let held = (0..NUM_SHADOW_CASCADES)
            .any(|i| mask & (1 << i) != 0 && inside(&u.light_vps[i], behind, 0.0));
        assert!(held);
        assert!(!inside(&cast, behind, 0.0));
    }

    #[test]
    fn blades_in_view_near_the_camera_cast() {
        for f in [view([0.3, 0.6, 0.2], 80.0), cut_view([0.3, 0.6, 0.2], 80.0)] {
            let f = f.frustum();
            assert!(casts(&f, CAMERA, [0.0, 0.0, -4.0], 0.5, 0.03));
            assert!(casts(&f, CAMERA, [1.5, 0.0, -1.0], 0.5, 0.03));
        }
    }

    // Past the first cascade, a blade still casts into the cascade its
    // shadow falls in.
    #[test]
    fn blades_beyond_the_first_cascade_cast() {
        let light = [0.3, 0.6, 0.2];
        let u = uniforms(light, 80.0, 4);
        let blade = [0.0, 0.0, -15.0];
        assert!(!inside(&u.light_vps[0], blade, 0.0));
        for f in [view(light, 80.0), cut_view(light, 80.0)] {
            assert!(casts(&f.frustum(), CAMERA, blade, 0.5, 0.03));
        }
    }

    // A blade just behind the camera is off screen, yet the cascades' boxes,
    // which bound spheres around their slices, hold it: it casts into a
    // cascade the schedule may keep, so its shadow does not pop when the
    // camera turns.
    #[test]
    fn blades_off_screen_inside_the_cascades_cast() {
        let f = view([0.3, 0.6, 0.2], 80.0).frustum();
        assert!(casts(&f, CAMERA, [0.0, 0.0, 1.0], 0.5, 0.03));
        assert!(casts(&f, CAMERA, [-3.0, 0.0, 0.5], 0.5, 0.03));
    }

    // Lit from low over the camera's shoulder, the light frustum stretches
    // toward the sun: blades up-sun cast onto the slice the camera sees.
    #[test]
    fn a_low_sun_reaches_blades_up_sun() {
        let f = view([0.0, 0.2, 1.0], 80.0).frustum();
        assert!(casts(&f, CAMERA, [0.0, 0.0, 6.0], 0.5, 0.03));
    }

    // Lit from low behind the camera, a blade behind it, off screen, throws
    // its shadow forward into view, so it casts into a cascade drawn for this
    // frame alone.
    #[test]
    fn blades_off_screen_whose_shadows_fall_in_view_cast() {
        let f = cut_view([0.0, 0.2, 1.0], 80.0).frustum();
        assert!(casts(&f, CAMERA, [0.0, 0.0, 0.3], 0.5, 0.03));
    }

    // Lit from in front, the same blade's shadow falls away behind the camera.
    #[test]
    fn blades_off_screen_whose_shadows_fall_away_do_not_cast() {
        let f = cut_view([0.0, 0.6, -1.0], 80.0).frustum();
        assert!(!casts(&f, CAMERA, [0.0, 0.0, 3.0], 0.5, 0.03));
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
}
