//! The atmosphere's reflections: the sky drawn into a small cube and
//! prefiltered on the GPU by the reflection-probe kernels, into the cube the
//! specular term samples in place of an environment map's.
//!
//! Redrawing it every frame would cost a full prefilter per frame for a sky
//! that changes slowly, so the whole cube, its sky and every prefiltered mip,
//! is redrawn only in a frame the sky has moved enough since the last draw. A
//! sky that keeps moving (a fast climb or descent, a sunset seen from it) is
//! then redrawn as it moves. Its mips are never convolved from different
//! skies, which a rough reflection would show as a flicker each time the
//! sharper and blurrier mips it blends caught up at different frames.

use super::frame::AtmosphereFrame;
use super::lut::SKY_CUBE_FACE;
use super::settings::AtmosphereSettings;
use super::surface::SurfaceBelow;
use crate::render::probe::capture::PrefilterPlan;
use crate::render::probe::radiance::SKY_CUBE_STORAGE_SCALE;
use concinnity_core::math::asin;
use concinnity_core::math::vec3::{dot, length, scale};

/// What the sky cube is prefiltered with: the probes' faces and firefly
/// clamp, at half their GGX samples (the sky has no fine detail, and the
/// sun's disc is left to the light's own highlight), over a cube stored at
/// [`SKY_CUBE_STORAGE_SCALE`]. The faces match an environment map's because
/// the specular tap biases the hardware's level of detail, which a smaller
/// cube would shift toward its sharp mips.
pub const SKY_CUBE_PLAN: PrefilterPlan =
    PrefilterPlan::new(SKY_CUBE_FACE, 64, 12.0).stored_at(SKY_CUBE_STORAGE_SCALE);

/// How far the sun, or the ground's up, may turn in the cube's frame, in
/// degrees, before the cube is redrawn: about two minutes of an Earth day,
/// and under a sun's width.
pub const SKY_CUBE_TURN_DEG: f32 = 0.5;

/// How far the sun may rise or sink against the ground's up, in degrees,
/// before the cube is redrawn while the sun is between the heights of
/// [`SKY_CUBE_LOW_SUN_DEG`]. A low sun brightens or dims the whole sky by
/// about a percent for each tenth of a degree it moves, so each redraw steps
/// the reflections by a fifth of a percent, under one 8-bit level.
pub const SKY_CUBE_LOW_SUN_TURN_DEG: f32 = 0.02;

/// The sun's heights above the horizon, in degrees, between which the cube
/// follows [`SKY_CUBE_LOW_SUN_TURN_DEG`]: from the end of twilight up through
/// the low sun after sunrise and before sunset.
pub const SKY_CUBE_LOW_SUN_DEG: [f32; 2] = [-18.0, 10.0];

/// How far the camera may climb or sink before the cube is redrawn: this
/// many kilometers plus [`SKY_CUBE_HEIGHT_FRACTION`] of its height.
pub const SKY_CUBE_HEIGHT_KM: f32 = 0.02;

/// See [`SKY_CUBE_HEIGHT_KM`].
pub const SKY_CUBE_HEIGHT_FRACTION: f32 = 0.05;

// How long the clouds may drift and change, in seconds, before the cube is
// redrawn.
const SKY_CUBE_CLOUD_SECONDS: f32 = 4.0;

/// What the sky in the cube depends on: the sun and the ground's up, both in
/// the cube's own frame (the sky rotation's), the camera's height, the
/// ground around the camera below the horizon, and the clouds' clock.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkyCubeKey {
    sun: [f32; 3],
    up: [f32; 3],
    height: f32,
    surface: Option<SurfaceBelow>,
    clouds: Option<f32>,
}

impl SkyCubeKey {
    /// The key of `frame`'s sky, the cube read through `sky_rot`
    /// (`SkyOrientation::sample_rows`).
    pub fn new(
        settings: &AtmosphereSettings,
        frame: &AtmosphereFrame,
        sky_rot: &[[f32; 4]; 3],
    ) -> Self {
        let into_cube = |v: [f32; 3]| -> [f32; 3] {
            core::array::from_fn(|i| dot([sky_rot[i][0], sky_rot[i][1], sky_rot[i][2]], v))
        };
        let up = scale(frame.camera_km, 1.0 / length(frame.camera_km).max(1e-6));
        Self {
            sun: into_cube(frame.sun_direction),
            up: into_cube(up),
            height: frame.camera_height(settings),
            surface: frame.surface,
            clouds: frame.clouds.map(|c| c.time),
        }
    }

    // Whether the sky has moved far enough from `last` to redraw the cube.
    fn moved_from(&self, last: &Self) -> bool {
        let turn = concinnity_core::math::cos(SKY_CUBE_TURN_DEG.to_radians());
        let tolerance = SKY_CUBE_HEIGHT_KM + SKY_CUBE_HEIGHT_FRACTION * last.height.abs();
        let surface = match (&self.surface, &last.surface) {
            (Some(s), Some(l)) => s.moved_from(l),
            (s, l) => s.is_some() != l.is_some(),
        };
        let clouds = match (self.clouds, last.clouds) {
            (Some(t), Some(l)) => (t - l).abs() > SKY_CUBE_CLOUD_SECONDS,
            (c, l) => c.is_some() != l.is_some(),
        };
        dot(self.sun, last.sun) < turn
            || dot(self.up, last.up) < turn
            || self.sun_moved_low(last)
            || (self.height - last.height).abs() > tolerance
            || surface
            || clouds
    }

    // The sun's height above the horizon, in degrees.
    fn sun_height_deg(&self) -> f32 {
        asin(dot(self.sun, self.up).clamp(-1.0, 1.0)).to_degrees()
    }

    // Whether the sun, low now or at `last`, has risen or sunk past the low
    // sun's tolerance since.
    fn sun_moved_low(&self, last: &Self) -> bool {
        let [lowest, highest] = SKY_CUBE_LOW_SUN_DEG;
        let (now, then) = (self.sun_height_deg(), last.sun_height_deg());
        let low = |h: f32| (lowest..=highest).contains(&h);
        (low(now) || low(then)) && (now - then).abs() > SKY_CUBE_LOW_SUN_TURN_DEG
    }
}

/// When the sky cube is redrawn.
#[derive(Debug, Clone, Copy, Default)]
pub struct SkyCubeSchedule {
    drawn: Option<SkyCubeKey>,
}

impl SkyCubeSchedule {
    /// Whether this frame redraws the whole cube for a sky with `key`: the
    /// first time, and once `key` has moved from the last drawn.
    pub fn plan(&mut self, key: SkyCubeKey) -> bool {
        let redraw = self.drawn.is_none_or(|last| key.moved_from(&last));
        if redraw {
            self.drawn = Some(key);
        }
        redraw
    }

    /// Forget the cube, so the next frame redraws it: after the tables it is
    /// drawn from were rebuilt from new sources.
    pub fn reset(&mut self) {
        self.drawn = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Atmosphere, DirectionalLight};
    use crate::environment::sky::SkyOrientation;

    fn key(sun: [f32; 3], camera_km: [f32; 3], rows: [[f32; 4]; 3]) -> SkyCubeKey {
        let s = AtmosphereSettings::resolve(&Atmosphere::default(), None);
        let light = DirectionalLight {
            direction: sun,
            ..Default::default()
        };
        SkyCubeKey::new(&s, &AtmosphereFrame::new(&s, camera_km, &light), &rows)
    }

    fn sun_at(deg: f32) -> [f32; 3] {
        let a = deg.to_radians();
        [
            concinnity_core::math::cos(a),
            concinnity_core::math::sin(a),
            0.0,
        ]
    }

    const GROUND: [f32; 3] = [0.0, 6360.002, 0.0];

    // The first frame draws the cube; a still sky then costs nothing.
    #[test]
    fn the_first_frame_draws_and_a_still_sky_is_left() {
        let mut s = SkyCubeSchedule::default();
        let k = key(sun_at(30.0), GROUND, SkyOrientation::IDENTITY_ROWS);
        assert!(s.plan(k));
        for _ in 0..10 {
            assert!(!s.plan(k));
        }
    }

    // A sun past the threshold is redrawn at once, and a sun short of it is
    // measured from the last draw, not the last frame.
    #[test]
    fn a_moved_sun_is_redrawn_from_the_last_draw() {
        let mut s = SkyCubeSchedule::default();
        s.plan(key(sun_at(30.0), GROUND, SkyOrientation::IDENTITY_ROWS));
        let step = SKY_CUBE_TURN_DEG * 0.6;
        assert!(!s.plan(key(
            sun_at(30.0 + step),
            GROUND,
            SkyOrientation::IDENTITY_ROWS
        )));
        assert!(s.plan(key(
            sun_at(30.0 + 2.0 * step),
            GROUND,
            SkyOrientation::IDENTITY_ROWS
        )));
        assert!(!s.plan(key(
            sun_at(30.0 + 2.0 * step),
            GROUND,
            SkyOrientation::IDENTITY_ROWS
        )));
    }

    // A low sun rising at a twentieth of a degree a frame, as from a fast flight
    // toward dawn, is redrawn every frame.
    #[test]
    fn a_low_sun_rising_every_frame_is_drawn_every_frame() {
        let mut s = SkyCubeSchedule::default();
        for frame in 0..20 {
            let sun = sun_at(-2.0 + 0.05 * frame as f32);
            assert!(s.plan(key(sun, GROUND, SkyOrientation::IDENTITY_ROWS)));
        }
    }

    // Turning the sky moves the sun and the up in the cube's frame alike;
    // climbing redraws once past the height tolerance.
    #[test]
    fn a_turned_sky_or_a_climb_redraws() {
        let rows = |deg| SkyOrientation::new([0.0, 0.0, 1.0], deg).sample_rows();
        let base = key(sun_at(30.0), GROUND, rows(0.0));
        assert!(key(sun_at(30.0), GROUND, rows(1.0)).moved_from(&base));
        assert!(!key(sun_at(30.0), GROUND, rows(0.2)).moved_from(&base));
        let up = |m: f32| [0.0, 6360.0 + m / 1000.0, 0.0];
        let low = key(sun_at(30.0), up(2.0), rows(0.0));
        assert!(!key(sun_at(30.0), up(15.0), rows(0.0)).moved_from(&low));
        assert!(key(sun_at(30.0), up(30.0), rows(0.0)).moved_from(&low));
        let high = key(sun_at(30.0), up(10_000.0), rows(0.0));
        assert!(!key(sun_at(30.0), up(10_400.0), rows(0.0)).moved_from(&high));
        assert!(key(sun_at(30.0), up(10_600.0), rows(0.0)).moved_from(&high));
    }

    // A low sun redraws the cube on a smaller rise than a high one, and a
    // sun deep below the horizon on none.
    #[test]
    fn a_low_sun_redraws_on_a_smaller_rise() {
        let at = |deg| key(sun_at(deg), GROUND, SkyOrientation::IDENTITY_ROWS);
        assert!(at(2.03).moved_from(&at(2.0)));
        assert!(at(-5.0).moved_from(&at(-5.03)));
        assert!(!at(2.01).moved_from(&at(2.0)));
        assert!(!at(30.2).moved_from(&at(30.0)));
        assert!(!at(-30.2).moved_from(&at(-30.0)));
    }

    // The ground below the horizon turning from land to sea redraws; the same
    // ground does not.
    #[test]
    fn a_changed_surface_redraws() {
        use super::super::surface::{SeaLook, SurfacePalette, SurfaceView};
        use crate::render::environment::ground::GroundCovers;
        let shape = crate::components::Planet::default().shape();
        let view = SurfaceView {
            camera: [0.0, shape.radius + shape.highest_height() + 10.0, 0.0],
            east: [1.0, 0.0, 0.0],
            east_rendered: [1.0, 0.0, 0.0],
        };
        let below = |level: Option<f32>| {
            let sea = level.map(|level| SeaLook {
                level,
                look: crate::environment::water::DistantLook {
                    body: [0.02, 0.05, 0.15],
                    fresnel_power: 5.0,
                    alpha: 0.1,
                },
            });
            let palette = SurfacePalette::new([0.1; 3], GroundCovers::default(), None, sea);
            Some(SurfaceBelow::sample(&palette, &shape, &view))
        };
        let base = key(sun_at(30.0), GROUND, SkyOrientation::IDENTITY_ROWS);
        let land = SkyCubeKey {
            surface: below(None),
            ..base
        };
        let flooded = SkyCubeKey {
            surface: below(Some(1000.0)),
            ..base
        };
        assert!(land.moved_from(&base));
        assert!(flooded.moved_from(&land));
        assert!(!land.moved_from(&land));
    }

    // Clouds coming or going redraw the cube, and so does their drifting once
    // a few seconds have passed.
    #[test]
    fn drifting_clouds_redraw_the_cube() {
        let base = key(sun_at(30.0), GROUND, SkyOrientation::IDENTITY_ROWS);
        let at = |t: f32| SkyCubeKey {
            clouds: Some(t),
            ..base
        };
        assert!(at(0.0).moved_from(&base));
        assert!(!at(1.0).moved_from(&at(0.0)));
        assert!(at(5.0).moved_from(&at(0.0)));
    }

    // A reset redraws the cube on the next frame.
    #[test]
    fn a_reset_redraws() {
        let mut s = SkyCubeSchedule::default();
        let k = key(sun_at(30.0), GROUND, SkyOrientation::IDENTITY_ROWS);
        s.plan(k);
        s.reset();
        assert!(s.plan(k));
    }

    #[test]
    fn the_plan_covers_the_cube() {
        assert_eq!(
            SKY_CUBE_PLAN.face_size(),
            PrefilterPlan::RUNTIME.face_size()
        );
        // 512 down to 4.
        assert_eq!(SKY_CUBE_PLAN.mips(), 8);
    }
}
