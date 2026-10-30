//! The atmosphere's reflections: the sky drawn into a small cube and
//! prefiltered on the GPU by the reflection-probe kernels, into the cube the
//! specular term samples in place of an environment map's.
//!
//! Redrawing it every frame would cost a full prefilter per frame for a sky
//! that changes slowly, so it is redrawn only once the sky has moved enough
//! since the last draw, and the redraw is spread over frames: the sky and the
//! prefilter's source pyramid in one frame, then one rough mip per frame. The
//! first fill does it all at once, so nothing ever samples an empty cube.

use super::frame::AtmosphereFrame;
use super::lut::SKY_CUBE_FACE;
use super::settings::AtmosphereSettings;
use super::surface::SurfaceBelow;
use crate::render::probe::capture::PrefilterPlan;
use crate::render::probe::radiance::SKY_CUBE_STORAGE_SCALE;
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
            || (self.height - last.height).abs() > tolerance
            || surface
            || clouds
    }
}

/// What one frame records into the sky cube.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SkyCubeWork {
    /// Draw the sky into the capture, copy it into the cube's mirror mip and
    /// build the capture's pyramid.
    pub capture: bool,
    /// The first rough mip to convolve.
    pub ggx_first: u32,
    /// One past the last.
    pub ggx_end: u32,
}

impl SkyCubeWork {
    /// The rough mips to convolve, in order.
    pub fn ggx_mips(&self) -> core::ops::Range<u32> {
        self.ggx_first..self.ggx_end
    }

    /// Whether the frame records anything.
    pub fn is_empty(&self) -> bool {
        !self.capture && self.ggx_first >= self.ggx_end
    }
}

/// When the sky cube is drawn and convolved.
#[derive(Debug, Clone, Copy)]
pub struct SkyCubeSchedule {
    mips: u32,
    drawn: Option<SkyCubeKey>,
    // The next rough mip of the redraw in progress; `mips` once it is done.
    next: u32,
}

impl SkyCubeSchedule {
    /// A schedule over a cube of `mips` levels that has never been drawn.
    pub fn new(mips: u32) -> Self {
        Self {
            mips,
            drawn: None,
            next: mips,
        }
    }

    /// This frame's work for a sky with `key`: the whole cube the first
    /// time; the next rough mip while a redraw is in progress; the sky and
    /// the pyramid once `key` has moved from the last drawn; else nothing.
    pub fn plan(&mut self, key: SkyCubeKey) -> SkyCubeWork {
        match self.drawn {
            None => {
                self.drawn = Some(key);
                self.next = self.mips;
                SkyCubeWork {
                    capture: true,
                    ggx_first: 1,
                    ggx_end: self.mips,
                }
            }
            Some(_) if self.next < self.mips => {
                let mip = self.next;
                self.next += 1;
                SkyCubeWork {
                    capture: false,
                    ggx_first: mip,
                    ggx_end: mip + 1,
                }
            }
            Some(last) if key.moved_from(&last) => {
                self.drawn = Some(key);
                self.next = 1;
                SkyCubeWork {
                    capture: true,
                    ggx_first: 1,
                    ggx_end: 1,
                }
            }
            Some(_) => SkyCubeWork::default(),
        }
    }

    /// Forget the cube, so the next frame fills it whole: after the tables
    /// it is drawn from were rebuilt from new sources.
    pub fn reset(&mut self) {
        self.drawn = None;
        self.next = self.mips;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Atmosphere, DirectionalLight};
    use crate::sky::SkyOrientation;

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
    const MIPS: u32 = SKY_CUBE_PLAN.mips();

    // The first frame does the whole cube; a still sky then costs nothing.
    #[test]
    fn the_first_fill_is_whole_and_a_still_sky_is_left() {
        let mut s = SkyCubeSchedule::new(MIPS);
        let k = key(sun_at(30.0), GROUND, SkyOrientation::IDENTITY_ROWS);
        let first = s.plan(k);
        assert!(first.capture);
        assert_eq!(first.ggx_mips(), 1..MIPS);
        for _ in 0..10 {
            assert!(s.plan(k).is_empty());
        }
    }

    // A sun past the threshold is redrawn over frames: the sky and pyramid,
    // then one rough mip a frame, in order, and nothing once done.
    #[test]
    fn a_moved_sun_is_redrawn_one_mip_a_frame() {
        let mut s = SkyCubeSchedule::new(MIPS);
        s.plan(key(sun_at(30.0), GROUND, SkyOrientation::IDENTITY_ROWS));
        let near = key(
            sun_at(30.0 + SKY_CUBE_TURN_DEG * 0.5),
            GROUND,
            SkyOrientation::IDENTITY_ROWS,
        );
        assert!(s.plan(near).is_empty());
        let moved = key(sun_at(31.0), GROUND, SkyOrientation::IDENTITY_ROWS);
        let start = s.plan(moved);
        assert!(start.capture && start.ggx_mips().is_empty());
        for mip in 1..MIPS {
            let w = s.plan(moved);
            assert!(!w.capture);
            assert_eq!(w.ggx_mips(), mip..mip + 1);
        }
        assert!(s.plan(moved).is_empty());
    }

    // A move during a redraw waits for it to finish, then starts another.
    #[test]
    fn a_redraw_finishes_before_the_next() {
        let mut s = SkyCubeSchedule::new(MIPS);
        s.plan(key(sun_at(30.0), GROUND, SkyOrientation::IDENTITY_ROWS));
        s.plan(key(sun_at(31.0), GROUND, SkyOrientation::IDENTITY_ROWS));
        let later = key(sun_at(40.0), GROUND, SkyOrientation::IDENTITY_ROWS);
        for _ in 1..MIPS {
            assert!(!s.plan(later).capture);
        }
        assert!(s.plan(later).capture);
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
                deep_color: [0.02, 0.05, 0.15],
                fresnel_power: 5.0,
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

    // A reset fills the cube whole again.
    #[test]
    fn a_reset_refills() {
        let mut s = SkyCubeSchedule::new(MIPS);
        let k = key(sun_at(30.0), GROUND, SkyOrientation::IDENTITY_ROWS);
        s.plan(k);
        s.reset();
        assert_eq!(s.plan(k).ggx_mips(), 1..MIPS);
    }

    #[test]
    fn the_plan_covers_the_cube() {
        assert_eq!(
            SKY_CUBE_PLAN.face_size(),
            PrefilterPlan::RUNTIME.face_size()
        );
        // 512 down to 4.
        assert_eq!(MIPS, 8);
    }
}
