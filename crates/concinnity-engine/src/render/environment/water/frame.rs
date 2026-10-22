//! A surface's per-draw block as built, and the body that moves it each
//! frame. Every backend keeps one per surface and writes `params_at` into the
//! frame's copy of the surface's params.

use crate::ecs::SimTiming;
use crate::planet::{LocalFrame, Rebase};
use crate::render::environment::grass::map_then;
use crate::render::uniforms::WaterParams;
use crate::render::uniforms::grass::GRASS_IDENTITY_MAP;
use crate::water::WaterBody;

/// A surface's params as built at init, and the body whose waves, ripples
/// and ocean tiles move them each frame. A planet sea also keeps where the
/// simulated frame sits on the planet, and where the wind's gusts lie in it,
/// carried through every move of the frame.
#[derive(Debug, Clone)]
pub struct WaterMotion {
    body: WaterBody,
    params: WaterParams,
    frame: LocalFrame,
    wind_map: [[f32; 4]; 2],
}

impl WaterMotion {
    /// The motion of `body`, whose params select its planar reflection when
    /// `planar` (see [`WaterParams::planar_lane`]).
    pub fn new(body: &WaterBody, planar: bool) -> Self {
        Self {
            body: body.clone(),
            params: WaterParams::from_body(body, planar),
            frame: LocalFrame::AUTHORED,
            wind_map: GRASS_IDENTITY_MAP,
        }
    }

    /// Carry the surface into the frame the world moved to: a planet sea's
    /// waves and gusts stay where they were on the planet. A rectangle does
    /// not move with the frame.
    pub fn rebase(&mut self, rebase: &Rebase) {
        if self.body.sea.is_some() {
            self.frame = self.frame.followed(rebase);
            self.wind_map = map_then(self.wind_map, &rebase.inverse_matrix());
        }
    }

    /// The body it moves.
    pub fn body(&self) -> &WaterBody {
        &self.body
    }

    /// The params as built.
    pub fn params(&self) -> &WaterParams {
        &self.params
    }

    /// The params `time` seconds into the world, seen from a camera at
    /// `camera`, which an ocean's grid and lookups are anchored to.
    pub fn params_at(&self, time: f64, camera: [f32; 3]) -> WaterParams {
        let mut params = self.params;
        match self.body.sea_motion_at(time, &self.frame, camera) {
            Some(sea) => params.advance_sea(&self.body, &sea, self.wind_map),
            None => params.advance(&self.body.motion_at(time, self.body.anchor(camera))),
        }
        params
    }
}

/// The instant the world's water is drawn at: the simulated instant the
/// frame's blended poses show. Whatever floats is drawn on the surface it was
/// simulated against, and the water moves only while the simulation does.
pub fn water_time(timing: &SimTiming) -> f64 {
    timing.drawn_time()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{WaterSurface, Wind};
    use crate::water::ocean::OceanQuality;
    use crate::wind::WindField;

    // A pond moves the same wherever the camera is; an ocean's anchor
    // follows the camera.
    #[test]
    fn only_an_oceans_params_follow_the_camera() {
        let wind = WindField::new(Some(&Wind {
            strength: 9.0,
            ..Wind::default()
        }));
        let pond = WaterMotion::new(&WaterBody::resolve(&WaterSurface::default(), &wind), false);
        let a = pond.params_at(5.0, [0.0; 3]);
        let b = pond.params_at(5.0, [300.0, 2.0, -40.0]);
        assert_eq!(bytemuck::bytes_of(&a), bytemuck::bytes_of(&b));

        let sea = WaterSurface {
            extent: [3000.0, 3000.0],
            fetch_meters: 1.0e5,
            ..Default::default()
        };
        let (bodies, _) = WaterBody::resolve_all([&sea], &wind, OceanQuality::FULL, None);
        let ocean = WaterMotion::new(&bodies[0], false);
        let a = ocean.params_at(5.0, [0.0; 3]);
        let b = ocean.params_at(5.0, [300.0, 2.0, -40.0]);
        assert_ne!(a.anchor, b.anchor);
        assert_eq!(a.ocean.layers, b.ocean.layers);
        assert!(a.ocean.layers[0] > 0);
        // The built params are the start of the world seen from the center.
        assert_eq!(
            bytemuck::bytes_of(&ocean.params_at(0.0, sea.center)),
            bytemuck::bytes_of(ocean.params())
        );
    }

    // A planet sea's gusts are laid out where the grass's are: the same rows,
    // carried through the same moves. A rectangle ignores the moves.
    #[test]
    fn a_planet_sea_carries_its_gusts_like_the_grass() {
        use crate::components::WaterShape;
        use crate::render::environment::grass::GrassHistory;
        let wind = WindField::new(Some(&Wind::default()));
        let sea = WaterSurface {
            shape: WaterShape::PlanetSea,
            sea_level: 10.0,
            ..Default::default()
        };
        let planet = crate::components::Planet::default().shape();
        let (bodies, _) = WaterBody::resolve_all([&sea], &wind, OceanQuality::FULL, Some(&planet));
        let mut motion = WaterMotion::new(&bodies[0], false);
        let mut grass = GrassHistory::default();
        let mut pond =
            WaterMotion::new(&WaterBody::resolve(&WaterSurface::default(), &wind), false);
        let (_, rebase) = LocalFrame::AUTHORED.recentered([700.0, 4.0, -700.0], [0.0, 1.0, 0.0]);
        let before = pond.params_at(3.0, [700.0, 4.0, -700.0]);
        for _ in 0..2 {
            motion.rebase(&rebase);
            grass.rebase(&rebase);
            pond.rebase(&rebase);
        }
        assert_eq!(motion.wind_map, grass.wind_map());
        assert_ne!(motion.wind_map, GRASS_IDENTITY_MAP);
        let after = pond.params_at(3.0, [700.0, 4.0, -700.0]);
        assert_eq!(bytemuck::bytes_of(&before), bytemuck::bytes_of(&after));
        let p = motion.params_at(3.0, [0.0, 30.0, 0.0]);
        assert_eq!(p.sea.wind_map, motion.wind_map);
        assert!(p.sea.up[3] > 50_000.0);
    }
}
