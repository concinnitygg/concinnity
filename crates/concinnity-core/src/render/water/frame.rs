//! A surface's per-draw block as built, and the body that moves it each
//! frame. Every backend keeps one per surface and writes `params_at` into the
//! frame's copy of the surface's params.

use super::WaterBody;
use crate::render::uniforms::WaterParams;

/// A surface's params as built at init, and the body whose waves, ripples
/// and ocean tiles move them each frame.
#[derive(Debug, Clone)]
pub struct WaterMotion {
    body: WaterBody,
    params: WaterParams,
}

impl WaterMotion {
    /// The motion of `body`, whose params select its planar reflection when
    /// `planar` (see [`WaterParams::planar_lane`]).
    pub fn new(body: &WaterBody, planar: bool) -> Self {
        Self {
            body: body.clone(),
            params: WaterParams::from_body(body, planar),
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
        params.advance(&self.body.motion_at(time, self.body.anchor(camera)));
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{WaterSurface, Wind};
    use crate::render::water::ocean::OceanQuality;
    use crate::render::wind::WindField;

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
        let (bodies, _) = WaterBody::resolve_all([&sea], &wind, OceanQuality::FULL);
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
}
