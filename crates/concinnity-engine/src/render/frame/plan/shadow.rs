//! Which shadow slices and views re-render each frame, and the projections
//! they render through.

use crate::components::ShadowUpdate;
use crate::environment::planet::Rebase;
use crate::gfx::render_types::{CloudShadowLanes, FarShadowLanes, ShadowUniforms};
use crate::render::backend::init::{ShadowCadence, ShadowMapSizes};
use crate::render::environment::grass::shadow::{GrassShadowView, grass_cascades};
use crate::render::shadow::csm::empty_shadow_uniforms;
use crate::render::shadow::far::{FarFrame, FarGrid, FarShadowScheduler, FarTick};
use crate::render::shadow::schedule::{
    CascadeCamera, CascadeLight, ShadowCascadeScheduler, shadow_distance,
};
use crate::render::shadow::spot::SpotShadowScheduler;

/// One frame's inputs to the shadow schedules.
#[derive(Clone, Copy, Debug)]
pub struct ShadowFrame {
    /// The shadow maps the renderer built.
    pub maps: ShadowMapSizes,
    /// Unit vector toward the shadowed light.
    pub dir_to_source: [f32; 3],
    /// The camera the sun's shadows are fit to.
    pub camera: CascadeCamera,
    /// Seconds since the world started.
    pub elapsed: f32,
    /// How many spot lights cast shadows.
    pub spots: u32,
}

/// The cascade, far and spot shadow schedules and the sun shadows' carried
/// uniforms.
pub struct ShadowPlan {
    /// The live schedule.
    pub cadence: ShadowCadence,
    /// The cascade and far projections: a cascade skipped this frame keeps the
    /// one its slice was last rendered with, and the far view the one it was
    /// last drawn through, so the shaded passes sample each consistently. The
    /// empty set while shadows are off.
    pub uniforms: ShadowUniforms,
    /// The cascades re-rendering this frame, one bit each.
    pub cascade_mask: u32,
    /// The re-rendering cascades a world's grass casts into, one bit each.
    pub grass_mask: u32,
    /// Whether the far view redraws this frame.
    pub far_draw: bool,
    /// The spot slices re-rendering this frame, one bit each.
    pub spot_mask: u32,
    cascades: ShadowCascadeScheduler,
    far: FarShadowScheduler,
    spots: SpotShadowScheduler,
}

impl ShadowPlan {
    /// A schedule following `cadence`, with nothing rendered yet.
    pub fn new(cadence: ShadowCadence) -> Self {
        Self {
            cadence,
            uniforms: empty_shadow_uniforms(),
            cascade_mask: 0,
            grass_mask: 0,
            far_draw: false,
            spot_mask: 0,
            cascades: ShadowCascadeScheduler::default(),
            far: FarShadowScheduler::default(),
            spots: SpotShadowScheduler::default(),
        }
    }

    /// The cascades this frame's grass casts into, lit from `to_light`, or
    /// `None` when it casts into none.
    pub fn grass_view(&self, to_light: [f32; 3]) -> Option<GrassShadowView> {
        GrassShadowView::new(&self.uniforms, self.grass_mask, to_light)
    }

    /// Where this frame's cloud shadows lie, or [`CloudShadowLanes::OFF`].
    pub fn set_clouds(&mut self, clouds: CloudShadowLanes) {
        self.uniforms.clouds = clouds;
    }

    /// Carry the cascade and far projections into the frame the world moved
    /// to, so a slice or a far map kept after the move is sampled where it was
    /// drawn.
    pub fn rebase(&mut self, rebase: &Rebase) {
        for vp in &mut self.uniforms.light_vps {
            *vp = rebase.reproject(*vp);
        }
        let far = &mut self.uniforms.far;
        if far.center[3] > 0.0 {
            far.light_vp = rebase.reproject(far.light_vp);
            let [x, y, z, reach] = far.center;
            let [x, y, z] = rebase.apply_point([x, y, z]);
            far.center = [x, y, z, reach];
        }
        self.far.rebase(rebase);
    }

    /// Note that the casters inside the world-space box `bounds` changed, so a
    /// far map that reaches them redraws. `None` is a change anywhere.
    pub fn casters_changed(&mut self, bounds: Option<([f32; 3], [f32; 3])>) {
        self.far.casters_changed(bounds);
    }

    /// Render every cascade and the far view on the next frame, for a view
    /// that does not continue the last: a kept slice or far map would cover
    /// where the camera was.
    pub fn reprime(&mut self) {
        self.cascades.reprime();
        self.far.reprime();
    }

    /// Start the spot slices over, so each re-primes before it is sampled.
    pub fn reset_spots(&mut self) {
        self.spots = SpotShadowScheduler::default();
    }

    /// Advance one frame: the cascades and the far view of the shadow maps the
    /// renderer built, and the shadowed spots.
    pub fn refresh(&mut self, frame: ShadowFrame) {
        if frame.maps.cascades > 0 {
            let light = CascadeLight {
                dir_to_source: frame.dir_to_source,
                map_size: frame.maps.cascades,
            };
            self.cascade_mask =
                self.cascades
                    .refresh(&mut self.uniforms, &self.cadence, light, frame.camera);
            self.grass_mask = self.cascade_mask & grass_cascades(&self.uniforms);
        }
        self.refresh_far(&frame);
        let every_frame = matches!(self.cadence.update, ShadowUpdate::EveryFrame);
        self.spot_mask = self.spots.next_mask(every_frame, frame.spots as usize);
    }

    // Advance the far view one frame. It runs only beside the cascades it
    // takes over from, and covers no farther than the camera sees.
    fn refresh_far(&mut self, frame: &ShadowFrame) {
        let camera = &frame.camera;
        let reach = shadow_distance(self.cadence.far_distance, camera.view_distance);
        let grid = if frame.maps.cascades > 0 {
            FarGrid::new(frame.dir_to_source, reach, frame.maps.far)
        } else {
            None
        };
        let far = self.far.next(FarTick {
            grid,
            camera: camera.position,
            elapsed: frame.elapsed,
            refresh_frames: self.cadence.far_refresh_frames,
        });
        self.far_draw = match far {
            FarFrame::Off => {
                self.uniforms.far = FarShadowLanes::OFF;
                false
            }
            FarFrame::Keep => false,
            FarFrame::Draw(lanes) => {
                self.uniforms.far = lanes;
                true
            }
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::IDENTITY;

    fn frame(
        maps: ShadowMapSizes,
        dir_to_source: [f32; 3],
        camera: CascadeCamera,
        spots: u32,
    ) -> ShadowFrame {
        ShadowFrame {
            maps,
            dir_to_source,
            camera,
            elapsed: 0.0,
            spots,
        }
    }

    fn camera() -> CascadeCamera {
        CascadeCamera {
            view: IDENTITY,
            position: [0.0; 3],
            fov_y_rad: 1.0,
            aspect: 1.0,
            near: 0.1,
            view_distance: None,
            focus_depth: None,
        }
    }

    fn cadence() -> ShadowCadence {
        ShadowCadence {
            update: ShadowUpdate::EveryFrame,
            distance: 100,
            cascades: 4,
            far_distance: 0,
            far_refresh_frames: 2,
        }
    }

    const CASCADES: ShadowMapSizes = ShadowMapSizes {
        cascades: 1024,
        far: 0,
    };

    const WITH_FAR: ShadowMapSizes = ShadowMapSizes {
        cascades: 1024,
        far: 1024,
    };

    fn far_cadence() -> ShadowCadence {
        ShadowCadence {
            far_distance: 3000,
            ..hybrid()
        }
    }

    #[test]
    fn without_a_map_the_cascades_stay_empty_and_unrendered() {
        let mut plan = ShadowPlan::new(cadence());
        plan.refresh(frame(
            ShadowMapSizes::default(),
            [0.0, 1.0, 0.0],
            camera(),
            0,
        ));
        assert_eq!(plan.cascade_mask, 0);
        assert_eq!(
            plan.uniforms.active_cascades,
            empty_shadow_uniforms().active_cascades
        );
    }

    fn hybrid() -> ShadowCadence {
        ShadowCadence {
            update: ShadowUpdate::Hybrid,
            ..cadence()
        }
    }

    // A slice skipped after a rebase projects a carried point where its
    // render saw the point before the move.
    #[test]
    fn a_rebase_carries_the_skipped_cascades() {
        let mut plan = ShadowPlan::new(hybrid());
        plan.refresh(frame(CASCADES, [0.3, 1.0, 0.2], camera(), 0));
        let drawn = plan.uniforms.light_vps;
        let rebase = Rebase {
            rotation: concinnity_core::math::quat_from_axis_angle([1.0, 0.0, 0.0], 0.02),
            translation: [-900.0, 4.0, 10.0],
        };
        plan.rebase(&rebase);
        plan.refresh(frame(CASCADES, [0.3, 1.0, 0.2], camera(), 0));
        assert_eq!(
            plan.cascade_mask, 0b1011,
            "the near field plus one far cascade"
        );
        let clip = |m: [[f32; 4]; 4], p: [f32; 3]| -> [f32; 4] {
            core::array::from_fn(|i| m[0][i] * p[0] + m[1][i] * p[1] + m[2][i] * p[2] + m[3][i])
        };
        let p = [3.0, -1.0, -20.0];
        let cascade = 2;
        let before = clip(drawn[cascade], p);
        let after = clip(plan.uniforms.light_vps[cascade], rebase.apply_point(p));
        for i in 0..4 {
            assert!((before[i] - after[i]).abs() < 1e-2, "{before:?} {after:?}");
        }
    }

    #[test]
    fn a_reprime_renders_every_cascade_once() {
        let mut plan = ShadowPlan::new(hybrid());
        plan.refresh(frame(CASCADES, [0.0, 1.0, 0.0], camera(), 0));
        assert_eq!(plan.cascade_mask, 0b1111);
        plan.refresh(frame(CASCADES, [0.0, 1.0, 0.0], camera(), 0));
        assert_eq!(plan.cascade_mask.count_ones(), 3);
        plan.reprime();
        plan.refresh(frame(CASCADES, [0.0, 1.0, 0.0], camera(), 0));
        assert_eq!(plan.cascade_mask, 0b1111);
        plan.refresh(frame(CASCADES, [0.0, 1.0, 0.0], camera(), 0));
        assert_eq!(plan.cascade_mask.count_ones(), 3);
    }

    // The grass draws into the cascades near the camera, and only on the
    // frames that re-render them.
    #[test]
    fn the_grass_casts_into_the_near_cascades_that_re_render() {
        let mut plan = ShadowPlan::new(hybrid());
        plan.refresh(frame(CASCADES, [0.3, 1.0, 0.2], camera(), 0));
        let near = grass_cascades(&plan.uniforms);
        assert_eq!(plan.grass_mask, near);
        assert!(near.count_ones() > 1, "{near:#b}");
        for _ in 0..4 {
            plan.refresh(frame(CASCADES, [0.3, 1.0, 0.2], camera(), 0));
            assert_eq!(plan.grass_mask, plan.cascade_mask & near);
        }
        let mut off = ShadowPlan::new(hybrid());
        off.refresh(frame(
            ShadowMapSizes::default(),
            [0.3, 1.0, 0.2],
            camera(),
            0,
        ));
        assert_eq!(off.grass_mask, 0);
    }

    #[test]
    fn every_frame_renders_every_cascade_and_spot() {
        let mut plan = ShadowPlan::new(cadence());
        plan.refresh(frame(CASCADES, [0.0, 1.0, 0.0], camera(), 2));
        assert_eq!(plan.cascade_mask, 0b1111);
        assert_eq!(plan.spot_mask, 0b11);
        assert_eq!(plan.uniforms.active_cascades, 4);
    }

    #[test]
    fn without_a_far_distance_or_map_the_far_view_stays_off() {
        for (cadence, maps) in [
            (hybrid(), WITH_FAR),
            (far_cadence(), CASCADES),
            (
                far_cadence(),
                ShadowMapSizes {
                    cascades: 0,
                    far: 1024,
                },
            ),
        ] {
            let mut plan = ShadowPlan::new(cadence);
            plan.refresh(frame(maps, [0.3, 1.0, 0.2], camera(), 0));
            assert!(!plan.far_draw);
            assert_eq!(plan.uniforms.far, FarShadowLanes::OFF);
        }
    }

    // The far view draws on the frame it first covers anything, so it is
    // never sampled before it holds depth, and turning it off clears what the
    // shaded passes read.
    #[test]
    fn the_far_view_primes_before_it_is_sampled() {
        let mut plan = ShadowPlan::new(far_cadence());
        plan.refresh(frame(WITH_FAR, [0.3, 1.0, 0.2], camera(), 0));
        assert!(plan.far_draw);
        assert_eq!(plan.uniforms.far.center[3], 3000.0);
        let drawn = plan.uniforms.far;
        plan.refresh(frame(WITH_FAR, [0.3, 1.0, 0.2], camera(), 0));
        assert!(!plan.far_draw, "a still camera keeps the map");
        assert_eq!(plan.uniforms.far, drawn);
        plan.cadence.far_distance = 0;
        plan.refresh(frame(WITH_FAR, [0.3, 1.0, 0.2], camera(), 0));
        assert_eq!(plan.uniforms.far, FarShadowLanes::OFF);
        plan.cadence.far_distance = 3000;
        plan.refresh(frame(WITH_FAR, [0.3, 1.0, 0.2], camera(), 0));
        assert!(plan.far_draw);
    }

    #[test]
    fn the_far_view_covers_no_farther_than_the_camera_sees() {
        let mut plan = ShadowPlan::new(far_cadence());
        let near_sighted = CascadeCamera {
            view_distance: Some(1200.0),
            ..camera()
        };
        plan.refresh(frame(WITH_FAR, [0.3, 1.0, 0.2], near_sighted, 0));
        assert_eq!(plan.uniforms.far.center[3], 1200.0);
    }

    #[test]
    fn a_reprime_redraws_the_far_view() {
        let mut plan = ShadowPlan::new(far_cadence());
        plan.refresh(frame(WITH_FAR, [0.3, 1.0, 0.2], camera(), 0));
        plan.refresh(frame(WITH_FAR, [0.3, 1.0, 0.2], camera(), 0));
        assert!(!plan.far_draw);
        plan.reprime();
        plan.refresh(frame(WITH_FAR, [0.3, 1.0, 0.2], camera(), 0));
        assert!(plan.far_draw);
    }

    // A far map kept after a rebase projects a carried point where its draw
    // saw the point, and measures its coverage from the carried center.
    #[test]
    fn a_rebase_carries_the_far_view() {
        let mut plan = ShadowPlan::new(far_cadence());
        plan.refresh(frame(WITH_FAR, [0.3, 1.0, 0.2], camera(), 0));
        let drawn = plan.uniforms.far;
        let rebase = Rebase {
            rotation: concinnity_core::math::quat_from_axis_angle([0.0, 1.0, 0.0], 0.3),
            translation: [-900.0, 4.0, 10.0],
        };
        plan.rebase(&rebase);
        let far = plan.uniforms.far;
        let clip = |m: [[f32; 4]; 4], p: [f32; 3]| -> [f32; 4] {
            core::array::from_fn(|i| m[0][i] * p[0] + m[1][i] * p[1] + m[2][i] * p[2] + m[3][i])
        };
        let p = [700.0, -12.0, -1500.0];
        let before = clip(drawn.light_vp, p);
        let after = clip(far.light_vp, rebase.apply_point(p));
        for i in 0..4 {
            assert!((before[i] - after[i]).abs() < 1e-3, "{before:?} {after:?}");
        }
        let center = rebase.apply_point([drawn.center[0], drawn.center[1], drawn.center[2]]);
        for (carried, want) in far.center.iter().zip(center) {
            assert!((carried - want).abs() < 1e-2, "{far:?}");
        }
        assert_eq!(far.center[3], drawn.center[3]);
        assert_eq!(
            (far.fade_start, far.compare_offset, far.texel),
            (drawn.fade_start, drawn.compare_offset, drawn.texel)
        );
        // The camera and light carried by the same move keep the map.
        let carried = CascadeCamera {
            position: rebase.apply_point(camera().position),
            ..camera()
        };
        let light = rebase.apply_vector([0.3, 1.0, 0.2]);
        plan.refresh(frame(WITH_FAR, light, carried, 0));
        assert!(!plan.far_draw);
    }
}
