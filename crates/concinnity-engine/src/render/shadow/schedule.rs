//! Cross-backend cascade re-render scheduling for the cascaded shadow map. The
//! shadow pass re-rasterizes all scene geometry into every cascade slice, so it
//! is one of the heaviest passes; `ShadowUpdate::Hybrid` amortizes the far
//! cascades across frames (the cascades covering the near field every frame, one
//! far cascade round-robin) while keeping each slice primed before it is
//! sampled. Shared by all three backends so the policy lives once next to the
//! CSM math in `csm.rs`.

use crate::components::ShadowUpdate;
use crate::gfx::render_types::{NUM_SHADOW_CASCADES, ShadowUniforms};
use crate::render::backend::init::ShadowCadence;
use crate::render::shadow::csm::{self, ShadowUniformInputs};
use concinnity_core::math::SaturatingFrom;

// The near field of a camera that follows nothing: what moves at its feet and
// just ahead of it.
const FREE_NEAR_FIELD_DEPTH: f32 = 4.0;

// How far a follow camera's near field reaches, as a multiple of the view depth
// of what it orbits: the followed subject runs up to a quarter of the boom ahead
// of the pivot, and the ground around it shows its shadow.
const FOLLOW_NEAR_FIELD_SCALE: f32 = 1.5;

/// The camera a frame's cascades are fit to.
#[derive(Clone, Copy, Debug)]
pub struct CascadeCamera {
    /// World-to-view matrix, column-major.
    pub view: [[f32; 4]; 4],
    /// World-space position.
    pub position: [f32; 3],
    /// Vertical field of view, in radians.
    pub fov_y_rad: f32,
    /// Viewport width over height.
    pub aspect: f32,
    /// Near plane.
    pub near: f32,
    /// How far the camera sees, which caps the shadow distance; `None` for no
    /// limit.
    pub view_distance: Option<f32>,
    /// View depth of what the camera orbits when it follows something, or
    /// `None` for a free camera.
    pub focus_depth: Option<f32>,
}

/// The directional light a renderer's cascades are cast from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CascadeLight {
    /// Unit vector pointing toward the light.
    pub dir_to_source: [f32; 3],
    /// Per-cascade shadow map edge, in texels.
    pub map_size: u32,
}

/// Round-robin clock + primed-set state for the cascade re-render schedule. One
/// per renderer; `next_mask` advances it once per frame and returns which
/// cascades to re-render. The caller refreshes only those cascades' light VPs and
/// re-rasterizes only those slices, so skipped cascades keep the depth + VP they
/// were last rendered with.
#[derive(Debug, Default)]
pub(crate) struct ShadowCascadeScheduler {
    // Advances once per shadow update; selects which far cascade Hybrid mode
    // refreshes this frame.
    clock: u32,
    // Bit `i` set once cascade `i` has been rendered since the scheduler was
    // created. Unprimed cascades are force-rendered so a slice is never sampled
    // before it holds valid depth.
    primed_mask: u32,
}

impl ShadowCascadeScheduler {
    /// Choose which cascades to re-render this frame and advance the round-robin
    /// clock. Delegates the selection to the pure `select_cascade_mask` and
    /// applies its side effects: advance the clock and record the newly primed
    /// set. `near_cascades` leading cascades re-render every frame under
    /// `Hybrid`. Bit `i` set in the result means cascade `i` re-renders this
    /// frame.
    pub(crate) fn next_mask(
        &mut self,
        update: ShadowUpdate,
        active_cascades: u32,
        near_cascades: u32,
    ) -> u32 {
        let (mask, primed) = select_cascade_mask(
            update,
            self.clock,
            self.primed_mask,
            active_cascades,
            near_cascades,
        );
        self.clock = self.clock.wrapping_add(1);
        self.primed_mask = primed;
        mask
    }

    /// Render every cascade on the next frame, as before any had been.
    pub(crate) fn reprime(&mut self) {
        self.primed_mask = 0;
    }

    /// Advance the schedule one frame and refresh `uniforms` for it: the splits,
    /// fade and cascade count always, and the light view-projection of each cascade
    /// that re-renders this frame. A skipped cascade keeps the projection its
    /// slice was last rendered with, so every slice is sampled the way it was
    /// drawn. Returns the re-render mask.
    pub(crate) fn refresh(
        &mut self,
        uniforms: &mut ShadowUniforms,
        cadence: &ShadowCadence,
        light: CascadeLight,
        camera: CascadeCamera,
    ) -> u32 {
        let fresh = csm::compute_shadow_uniforms(ShadowUniformInputs {
            view: camera.view,
            cam_pos: camera.position,
            fov_y_rad: camera.fov_y_rad,
            aspect: camera.aspect,
            near: camera.near,
            shadow_distance: shadow_distance(cadence.distance, camera.view_distance),
            light_dir_to_source: light.dir_to_source,
            shadow_map_size: light.map_size,
            active_cascades: cadence.cascades,
        });
        let near = csm::cascades_sampled_within(&fresh, near_field_depth(camera.focus_depth));
        let mask = self.next_mask(cadence.update, cadence.cascades, near.count_ones());
        uniforms.cascade_splits = fresh.cascade_splits;
        uniforms.active_cascades = fresh.active_cascades;
        uniforms.fade_start = fresh.fade_start;
        for (i, vp) in uniforms.light_vps.iter_mut().enumerate() {
            if mask & (1u32 << i) != 0 {
                *vp = fresh.light_vps[i];
            }
        }
        mask
    }
}

// The authored shadow distance, capped by the camera's view distance when it
// has one: nothing past it is drawn, so nothing there needs a shadow.
fn shadow_distance(authored: u32, view_distance: Option<f32>) -> f32 {
    let authored = authored as f32;
    view_distance.map_or(authored, |d| authored.min(d))
}

// The view depth out to which `Hybrid` redraws every cascade each frame. A
// caster moving there, such as a followed subject, would otherwise leave its
// shadow lagging behind it until its cascade's turn in the rotation comes
// around.
fn near_field_depth(focus_depth: Option<f32>) -> f32 {
    focus_depth.map_or(FREE_NEAR_FIELD_DEPTH, |depth| {
        (depth * FOLLOW_NEAR_FIELD_SCALE).max(FREE_NEAR_FIELD_DEPTH)
    })
}

// Pure cascade-selection step, split out of `next_mask` so the priming +
// round-robin policy is unit-testable without renderer state.
//
// Given the update policy, the current round-robin clock, and which cascades
// have already been primed, returns `(render_mask, new_primed_mask)`:
//
//   - `scheduled` is the policy's steady-state set (EveryFrame = all; Hybrid =
//     the `near_cascades` leading cascades plus one round-robin far cascade, or
//     all when every cascade is near).
//   - any cascade not yet primed is force-rendered this frame so its slice is
//     never sampled before it holds valid depth. Because a cascade's bit is set
//     in `primed` the first frame it renders and cleared only by `reprime`,
//     priming renders each cascade once on first access and once after each
//     `reprime`.
fn select_cascade_mask(
    update: ShadowUpdate,
    clock: u32,
    primed: u32,
    active_cascades: u32,
    near_cascades: u32,
) -> (u32, u32) {
    // Only the first `active` cascades exist this frame; the rest are never
    // scheduled (and never sampled -- the shader bounds its lookup by the same
    // count). `primed` may carry bits for cascades that were active under a
    // higher count; `& all` masks them off so they never force a render.
    let active = active_cascades.clamp(1, u32::saturating_from(NUM_SHADOW_CASCADES));
    let all = (1u32 << active) - 1;
    let scheduled = match update {
        ShadowUpdate::EveryFrame => all,
        ShadowUpdate::Hybrid => {
            let near_count = near_cascades.clamp(1, active);
            let far_count = active - near_count;
            if far_count == 0 {
                all
            } else {
                let near = (1u32 << near_count) - 1;
                near | (1u32 << (near_count + clock % far_count))
            }
        }
    };
    let unprimed = all & !primed;
    let mask = (scheduled | unprimed) & all;
    (mask, primed | mask)
}

#[cfg(test)]
mod tests {
    use super::{
        CascadeCamera, CascadeLight, FREE_NEAR_FIELD_DEPTH, ShadowCadence, ShadowCascadeScheduler,
        near_field_depth, select_cascade_mask, shadow_distance,
    };
    use crate::components::ShadowUpdate;
    use crate::gfx::render_types::{NUM_SHADOW_CASCADES, ShadowUniforms};
    use crate::render::shadow::csm;
    use crate::transform::IDENTITY;
    use concinnity_core::math::SaturatingFrom;

    fn camera(x: f32, view_distance: Option<f32>) -> CascadeCamera {
        let mut view = IDENTITY;
        view[3][0] = -x;
        CascadeCamera {
            view,
            position: [x, 0.0, 0.0],
            fov_y_rad: 1.0,
            aspect: 16.0 / 9.0,
            near: 0.1,
            view_distance,
            focus_depth: None,
        }
    }

    fn following(focus_depth: f32) -> CascadeCamera {
        CascadeCamera {
            focus_depth: Some(focus_depth),
            ..camera(0.0, None)
        }
    }

    const LIGHT: CascadeLight = CascadeLight {
        dir_to_source: [0.3, 1.0, 0.2],
        map_size: 1024,
    };

    fn cadence_of(update: ShadowUpdate, distance: u32) -> ShadowCadence {
        ShadowCadence {
            update,
            distance,
            cascades: u32::saturating_from(NUM_SHADOW_CASCADES),
        }
    }

    fn expected(camera: CascadeCamera, cadence: &ShadowCadence) -> ShadowUniforms {
        csm::compute_shadow_uniforms(csm::ShadowUniformInputs {
            view: camera.view,
            cam_pos: camera.position,
            fov_y_rad: camera.fov_y_rad,
            aspect: camera.aspect,
            near: camera.near,
            shadow_distance: shadow_distance(cadence.distance, camera.view_distance),
            light_dir_to_source: LIGHT.dir_to_source,
            shadow_map_size: LIGHT.map_size,
            active_cascades: cadence.cascades,
        })
    }

    #[test]
    fn refresh_keeps_the_projection_of_every_skipped_cascade() {
        let cadence = cadence_of(ShadowUpdate::Hybrid, 80);
        let mut scheduler = ShadowCascadeScheduler::default();
        let mut uniforms = csm::empty_shadow_uniforms();
        let first = scheduler.refresh(&mut uniforms, &cadence, LIGHT, camera(0.0, Some(500.0)));
        assert_eq!(first, ALL, "the first frame primes every cascade");

        let before = uniforms;
        let moved = camera(25.0, Some(500.0));
        let mask = scheduler.refresh(&mut uniforms, &cadence, LIGHT, moved);
        assert_ne!(mask, ALL);
        let fresh = expected(moved, &cadence);
        let stale = (0..NUM_SHADOW_CASCADES)
            .any(|i| mask & (1 << i) == 0 && fresh.light_vps[i] != before.light_vps[i]);
        assert!(stale, "the move changes a skipped cascade's projection");
        for i in 0..NUM_SHADOW_CASCADES {
            let want = if mask & (1 << i) != 0 {
                fresh.light_vps[i]
            } else {
                before.light_vps[i]
            };
            assert_eq!(uniforms.light_vps[i], want, "cascade {i}");
        }
        assert_eq!(uniforms.cascade_splits, fresh.cascade_splits);
        assert_eq!(uniforms.active_cascades, fresh.active_cascades);
        assert_eq!(uniforms.fade_start.to_bits(), fresh.fade_start.to_bits());
    }

    #[test]
    fn refresh_caps_the_shadow_distance_at_the_view_distance() {
        let cadence = cadence_of(ShadowUpdate::EveryFrame, 400);
        let short = camera(0.0, Some(60.0));
        let mut uniforms = csm::empty_shadow_uniforms();
        ShadowCascadeScheduler::default().refresh(&mut uniforms, &cadence, LIGHT, short);
        let capped = expected(short, &cadence_of(ShadowUpdate::EveryFrame, 60));
        assert_eq!(uniforms.cascade_splits, capped.cascade_splits);
        assert!((uniforms.cascade_splits[NUM_SHADOW_CASCADES - 1] - 60.0).abs() < 1e-3);
    }

    #[test]
    fn an_unlimited_camera_keeps_the_authored_shadow_distance() {
        let cadence = cadence_of(ShadowUpdate::EveryFrame, 400);
        let mut uniforms = csm::empty_shadow_uniforms();
        ShadowCascadeScheduler::default().refresh(
            &mut uniforms,
            &cadence,
            LIGHT,
            camera(0.0, None),
        );
        assert!((uniforms.cascade_splits[NUM_SHADOW_CASCADES - 1] - 400.0).abs() < 1e-3);
        let beyond = camera(0.0, Some(1.0e4));
        ShadowCascadeScheduler::default().refresh(&mut uniforms, &cadence, LIGHT, beyond);
        assert!((uniforms.cascade_splits[NUM_SHADOW_CASCADES - 1] - 400.0).abs() < 1e-3);
    }

    #[test]
    fn refresh_returns_the_schedule_it_advanced() {
        let cadence = cadence_of(ShadowUpdate::Hybrid, 80);
        let mut refreshed = ShadowCascadeScheduler::default();
        let mut stepped = ShadowCascadeScheduler::default();
        let mut uniforms = csm::empty_shadow_uniforms();
        let near = csm::cascades_sampled_within(
            &expected(camera(0.0, Some(500.0)), &cadence),
            FREE_NEAR_FIELD_DEPTH,
        )
        .count_ones();
        for _ in 0..8 {
            let got = refreshed.refresh(&mut uniforms, &cadence, LIGHT, camera(0.0, Some(500.0)));
            assert_eq!(
                got,
                stepped.next_mask(cadence.update, cadence.cascades, near)
            );
        }
    }

    #[test]
    fn the_near_field_follows_what_the_camera_orbits() {
        assert_eq!(near_field_depth(None), FREE_NEAR_FIELD_DEPTH);
        assert_eq!(near_field_depth(Some(6.0)), 9.0);
        assert_eq!(near_field_depth(Some(40.0)), 60.0);
        assert_eq!(near_field_depth(Some(0.5)), FREE_NEAR_FIELD_DEPTH);
    }

    // Every cascade the shader reads within the near field re-renders every
    // frame, whatever the reach and however far back the camera follows.
    #[test]
    fn hybrid_redraws_every_cascade_read_within_the_near_field() {
        for reach in [20, 80, 160, 210, 320, 1000, 4000] {
            let cadence = cadence_of(ShadowUpdate::Hybrid, reach);
            for focus in [None, Some(3.0), Some(6.0), Some(20.0)] {
                let cam = CascadeCamera {
                    focus_depth: focus,
                    ..camera(0.0, None)
                };
                let fresh = expected(cam, &cadence);
                let near = csm::cascades_sampled_within(&fresh, near_field_depth(focus));
                let mut scheduler = ShadowCascadeScheduler::default();
                let mut uniforms = csm::empty_shadow_uniforms();
                for frame in 0..6 {
                    let mask = scheduler.refresh(&mut uniforms, &cadence, LIGHT, cam);
                    assert_eq!(mask & near, near, "{reach} m, {focus:?}, frame {frame}");
                }
            }
        }
    }

    // At 210 m the second cascade starts at 8.5 m but blends in from 7.2 m,
    // where a subject 6 m ahead of the camera trails with the pivot's lag.
    #[test]
    fn hybrid_keeps_a_followed_subject_in_a_current_cascade() {
        let cadence = cadence_of(ShadowUpdate::Hybrid, 210);
        let splits = expected(following(6.0), &cadence).cascade_splits;
        assert!(splits[0] > 8.0 && splits[0] < 9.0, "{splits:?}");
        let mut scheduler = ShadowCascadeScheduler::default();
        let mut uniforms = csm::empty_shadow_uniforms();
        for _ in 0..6 {
            let mask = scheduler.refresh(&mut uniforms, &cadence, LIGHT, following(6.0));
            assert_eq!(mask & 0b11, 0b11);
        }
    }

    // A free camera at a long reach keeps the far rotation: one near cascade
    // and one far one a frame.
    #[test]
    fn a_free_camera_at_a_long_reach_amortizes_the_far_cascades() {
        let cadence = cadence_of(ShadowUpdate::Hybrid, 320);
        let mut scheduler = ShadowCascadeScheduler::default();
        let mut uniforms = csm::empty_shadow_uniforms();
        scheduler.refresh(&mut uniforms, &cadence, LIGHT, camera(0.0, None));
        for _ in 0..6 {
            let mask = scheduler.refresh(&mut uniforms, &cadence, LIGHT, camera(0.0, None));
            assert_eq!(mask & 1, 1);
            assert_eq!(mask.count_ones(), 2);
        }
    }

    #[test]
    fn hybrid_redraws_every_near_cascade_and_rotates_the_rest() {
        let active = u32::saturating_from(NUM_SHADOW_CASCADES);
        let mut union = 0u32;
        for clock in 0..(active - 2) {
            let (mask, _) = select_cascade_mask(ShadowUpdate::Hybrid, clock, ALL, active, 2);
            assert_eq!(mask & 0b11, 0b11, "clock {clock}");
            assert_eq!((mask & !0b11).count_ones(), 1, "clock {clock}");
            union |= mask;
        }
        assert_eq!(union, ALL);
    }

    #[test]
    fn hybrid_redraws_all_when_every_cascade_is_near() {
        let active = u32::saturating_from(NUM_SHADOW_CASCADES);
        for clock in 0..3 {
            let (mask, _) = select_cascade_mask(ShadowUpdate::Hybrid, clock, ALL, active, active);
            assert_eq!(mask, ALL);
        }
    }

    const ALL: u32 = (1u32 << NUM_SHADOW_CASCADES) - 1;

    #[test]
    fn first_access_primes_every_cascade_at_once() {
        // From the unprimed state both policies render all cascades on the
        // very first frame, so no slice is sampled before it holds depth.
        for update in [ShadowUpdate::Hybrid, ShadowUpdate::EveryFrame] {
            let (mask, primed) =
                select_cascade_mask(update, 0, 0, u32::saturating_from(NUM_SHADOW_CASCADES), 1);
            assert_eq!(mask, ALL, "{update:?} should prime all cascades on frame 0");
            assert_eq!(primed, ALL, "every cascade is primed after frame 0");
        }
    }

    #[test]
    fn fewer_active_cascades_only_schedule_the_live_slots() {
        // With active_cascades = 2, only cascades 0 and 1 are ever scheduled --
        // the inactive 2,3 never render even if their primed bits are stale from
        // a higher count. EveryFrame renders both; Hybrid (far_count = 1) also
        // renders both (no far cascade to amortize with only one far slot).
        let live = 0b11u32;
        for update in [ShadowUpdate::Hybrid, ShadowUpdate::EveryFrame] {
            // Stale primed bits for the now-inactive cascades must not leak in.
            let (mask, primed) = select_cascade_mask(update, 0, ALL, 2, 1);
            assert_eq!(mask, live, "{update:?} scheduled an inactive cascade");
            assert_eq!(
                primed & !live,
                ALL & !live,
                "inactive primed bits untouched"
            );
        }
        // active = 1: only the near cascade ever renders.
        let (mask, _) = select_cascade_mask(ShadowUpdate::Hybrid, 3, 0, 1, 1);
        assert_eq!(mask, 0b1);
    }

    #[test]
    fn priming_never_re_renders_a_cascade() {
        // Once primed, Hybrid drops back to its steady-state set: the near
        // cascade plus exactly one far cascade. The full-prime render only
        // happens while a cascade is still unprimed, never again.
        let far_count = (u32::saturating_from(NUM_SHADOW_CASCADES) - 1).max(1);
        for clock in 0..(far_count * 3) {
            let (mask, primed) = select_cascade_mask(
                ShadowUpdate::Hybrid,
                clock,
                ALL,
                u32::saturating_from(NUM_SHADOW_CASCADES),
                1,
            );
            assert_eq!(primed, ALL, "already-primed set is unchanged");
            assert_eq!(mask & 1, 1, "near cascade refreshes every frame");
            let extra = (mask & !1u32).count_ones();
            assert_eq!(extra, 1, "exactly one far cascade refreshes per frame");
        }
    }

    #[test]
    fn hybrid_round_robins_every_far_cascade() {
        // Over `far_count` consecutive frames the steady-state Hybrid mask
        // touches each far cascade exactly once.
        let far_count = (u32::saturating_from(NUM_SHADOW_CASCADES) - 1).max(1);
        let mut union = 0u32;
        for clock in 0..far_count {
            let (mask, _) = select_cascade_mask(
                ShadowUpdate::Hybrid,
                clock,
                ALL,
                u32::saturating_from(NUM_SHADOW_CASCADES),
                1,
            );
            union |= mask;
        }
        assert_eq!(union, ALL, "every cascade is refreshed within one round");
    }

    #[test]
    fn every_frame_always_renders_all() {
        for clock in 0..5 {
            let (mask, primed) = select_cascade_mask(
                ShadowUpdate::EveryFrame,
                clock,
                ALL,
                u32::saturating_from(NUM_SHADOW_CASCADES),
                1,
            );
            assert_eq!(mask, ALL);
            assert_eq!(primed, ALL);
        }
    }

    #[test]
    fn priming_is_monotonic_and_one_shot_per_cascade() {
        // Drive the policy frame by frame from the unprimed state and assert
        // the primed set only grows, and any cascade rendered purely for
        // priming (in the mask but not in that frame's scheduled set) is
        // rendered at most once across the whole run.
        let mut primed = 0u32;
        let mut prime_renders = [0u32; NUM_SHADOW_CASCADES];
        for clock in 0..(u32::saturating_from(NUM_SHADOW_CASCADES) + 4) {
            let far_count = (u32::saturating_from(NUM_SHADOW_CASCADES) - 1).max(1);
            let scheduled_near_far = 1u32 | (1u32 << (1 + clock % far_count));
            let before = primed;
            let (mask, next) = select_cascade_mask(
                ShadowUpdate::Hybrid,
                clock,
                primed,
                u32::saturating_from(NUM_SHADOW_CASCADES),
                1,
            );
            assert_eq!(next & before, before, "primed set must never lose a bit");
            for (c, count) in prime_renders.iter_mut().enumerate() {
                let bit = 1u32 << c;
                let primed_only = (mask & bit != 0) && (scheduled_near_far & bit == 0);
                if primed_only {
                    *count += 1;
                }
            }
            primed = next;
        }
        for (c, count) in prime_renders.iter().enumerate() {
            assert!(
                *count <= 1,
                "cascade {c} was force-primed {count} times (>1)"
            );
        }
        assert_eq!(primed, ALL, "all cascades primed after the run");
    }

    #[test]
    fn scheduler_advances_clock_and_accumulates_primes() {
        // The struct wrapper primes everything on frame 0, then steady-states
        // to near + one far cascade and rotates the far cascade each frame.
        let mut sched = ShadowCascadeScheduler::default();
        assert_eq!(
            sched.next_mask(
                ShadowUpdate::Hybrid,
                u32::saturating_from(NUM_SHADOW_CASCADES),
                1,
            ),
            ALL,
            "frame 0 primes all"
        );
        let far_count = (u32::saturating_from(NUM_SHADOW_CASCADES) - 1).max(1);
        let mut union = 0u32;
        for _ in 0..far_count {
            let mask = sched.next_mask(
                ShadowUpdate::Hybrid,
                u32::saturating_from(NUM_SHADOW_CASCADES),
                1,
            );
            assert_eq!(mask & 1, 1, "near cascade refreshes every frame");
            assert_eq!((mask & !1u32).count_ones(), 1, "one far cascade per frame");
            union |= mask;
        }
        assert_eq!(union, ALL, "every cascade refreshed within one round");
    }
}
