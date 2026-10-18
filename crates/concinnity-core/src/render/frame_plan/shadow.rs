//! Which shadow slices re-render each frame, and the cascade projections
//! they render through.

use crate::components::ShadowUpdate;
use crate::gfx::render_types::ShadowUniforms;
use crate::render::backend_init::ShadowCadence;
use crate::render::csm::empty_shadow_uniforms;
use crate::render::shadow_schedule::{CascadeCamera, CascadeLight, ShadowCascadeScheduler};
use crate::render::spot_shadow::SpotShadowScheduler;

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

    #[test]
    fn every_frame_renders_every_cascade_and_spot() {
        let mut plan = ShadowPlan::new(cadence());
        plan.refresh(Some(1024), [0.0, 1.0, 0.0], camera(), 2);
        assert_eq!(plan.cascade_mask, 0b1111);
        assert_eq!(plan.spot_mask, 0b11);
        assert_eq!(plan.uniforms.active_cascades, 4);
    }
}
