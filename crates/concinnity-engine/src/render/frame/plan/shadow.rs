//! Which shadow slices re-render each frame, and the cascade projections
//! they render through.

use crate::components::ShadowUpdate;
use crate::gfx::render_types::{CloudShadowLanes, ShadowUniforms};
use crate::planet::Rebase;
use crate::render::backend::init::ShadowCadence;
use crate::render::shadow::csm::empty_shadow_uniforms;
use crate::render::shadow::schedule::{CascadeCamera, CascadeLight, ShadowCascadeScheduler};
use crate::render::shadow::spot::SpotShadowScheduler;

/// The cascade and spot shadow schedules and the cascades' carried uniforms.
pub struct ShadowPlan {
    /// The live cascade schedule.
    pub cadence: ShadowCadence,
    /// The cascade projections: a cascade skipped this frame keeps the one its
    /// slice was last rendered with, so the main pass samples it consistently.
    /// The empty set while shadows are off.
    pub uniforms: ShadowUniforms,
    /// The cascades re-rendering this frame, one bit each.
    pub cascade_mask: u32,
    /// The spot slices re-rendering this frame, one bit each.
    pub spot_mask: u32,
    cascades: ShadowCascadeScheduler,
    spots: SpotShadowScheduler,
}

impl ShadowPlan {
    /// A schedule following `cadence`, with nothing rendered yet.
    pub fn new(cadence: ShadowCadence) -> Self {
        Self {
            cadence,
            uniforms: empty_shadow_uniforms(),
            cascade_mask: 0,
            spot_mask: 0,
            cascades: ShadowCascadeScheduler::default(),
            spots: SpotShadowScheduler::default(),
        }
    }

    /// Where this frame's cloud shadows lie, or [`CloudShadowLanes::OFF`].
    pub fn set_clouds(&mut self, clouds: CloudShadowLanes) {
        self.uniforms.clouds = clouds;
    }

    /// Carry the cascade projections into the frame the world moved to, so a
    /// slice skipped after the move is sampled where it was drawn.
    pub fn rebase(&mut self, rebase: &Rebase) {
        for vp in &mut self.uniforms.light_vps {
            *vp = rebase.reproject(*vp);
        }
    }

    /// Render every cascade on the next frame, for a view that does not
    /// continue the last: a skipped slice would cover where the camera was.
    pub fn reprime_cascades(&mut self) {
        self.cascades.reprime();
    }

    /// Start the spot slices over, so each re-primes before it is sampled.
    pub fn reset_spots(&mut self) {
        self.spots = SpotShadowScheduler::default();
    }

    /// Advance one frame: the cascades of a `map_size` map lit from `light`
    /// seen by `camera`, or none without a map, and `spots` shadowed spots.
    pub fn refresh(
        &mut self,
        map_size: Option<u32>,
        dir_to_source: [f32; 3],
        camera: CascadeCamera,
        spots: u32,
    ) {
        if let Some(map_size) = map_size {
            let light = CascadeLight {
                dir_to_source,
                map_size,
            };
            self.cascade_mask =
                self.cascades
                    .refresh(&mut self.uniforms, &self.cadence, light, camera);
        }
        let every_frame = matches!(self.cadence.update, ShadowUpdate::EveryFrame);
        self.spot_mask = self.spots.next_mask(every_frame, spots as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::IDENTITY;

    fn camera() -> CascadeCamera {
        CascadeCamera {
            view: IDENTITY,
            position: [0.0; 3],
            fov_y_rad: 1.0,
            aspect: 1.0,
            near: 0.1,
            view_distance: None,
        }
    }

    fn cadence() -> ShadowCadence {
        ShadowCadence {
            update: ShadowUpdate::EveryFrame,
            distance: 100,
            cascades: 4,
        }
    }

    #[test]
    fn without_a_map_the_cascades_stay_empty_and_unrendered() {
        let mut plan = ShadowPlan::new(cadence());
        plan.refresh(None, [0.0, 1.0, 0.0], camera(), 0);
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
        plan.refresh(Some(1024), [0.3, 1.0, 0.2], camera(), 0);
        let drawn = plan.uniforms.light_vps;
        let rebase = Rebase {
            rotation: concinnity_core::math::quat_from_axis_angle([1.0, 0.0, 0.0], 0.02),
            translation: [-900.0, 4.0, 10.0],
        };
        plan.rebase(&rebase);
        plan.refresh(Some(1024), [0.3, 1.0, 0.2], camera(), 0);
        assert_eq!(plan.cascade_mask, 0b0101);
        let clip = |m: [[f32; 4]; 4], p: [f32; 3]| -> [f32; 4] {
            core::array::from_fn(|i| m[0][i] * p[0] + m[1][i] * p[1] + m[2][i] * p[2] + m[3][i])
        };
        let p = [3.0, -1.0, -20.0];
        for cascade in [1, 3] {
            let before = clip(drawn[cascade], p);
            let after = clip(plan.uniforms.light_vps[cascade], rebase.apply_point(p));
            for i in 0..4 {
                assert!((before[i] - after[i]).abs() < 1e-2, "{before:?} {after:?}");
            }
        }
    }

    #[test]
    fn a_reprime_renders_every_cascade_once() {
        let mut plan = ShadowPlan::new(hybrid());
        plan.refresh(Some(1024), [0.0, 1.0, 0.0], camera(), 0);
        assert_eq!(plan.cascade_mask, 0b1111);
        plan.refresh(Some(1024), [0.0, 1.0, 0.0], camera(), 0);
        assert_eq!(plan.cascade_mask.count_ones(), 2);
        plan.reprime_cascades();
        plan.refresh(Some(1024), [0.0, 1.0, 0.0], camera(), 0);
        assert_eq!(plan.cascade_mask, 0b1111);
        plan.refresh(Some(1024), [0.0, 1.0, 0.0], camera(), 0);
        assert_eq!(plan.cascade_mask.count_ones(), 2);
    }

    #[test]
    fn every_frame_renders_every_cascade_and_spot() {
        let mut plan = ShadowPlan::new(cadence());
        plan.refresh(Some(1024), [0.0, 1.0, 0.0], camera(), 2);
        assert_eq!(plan.cascade_mask, 0b1111);
        assert_eq!(plan.spot_mask, 0b11);
        assert_eq!(plan.uniforms.active_cascades, 4);
    }
}
