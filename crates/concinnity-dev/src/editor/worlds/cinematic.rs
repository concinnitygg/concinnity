// src/editor/worlds/cinematic.rs
//
// The start screen's attract camera: a small cycle of slow shots over the
// world the sidebar is previewing, with a black fade between them. Pure state
// and pure math -- the shot clock, the framing derived from the previewed
// world's bounds, the pose each shot holds at a moment, and the fade envelope
// -- plus the one full-viewport sprite the fade is drawn with. The hook's
// `hook/cinematic_drive.rs` owns the wiring: the frame dt, the world's camera,
// and when the cycle runs at all.
//
// The camera it produces never reaches the authored world: it is written onto
// the live preview's `Camera3D` each frame and the world's own pose is put
// back the moment the screen hands the session a world.

use super::super::framing::{CameraPose, bounding_sphere, fit_distance};
use super::super::registry::ID_BASE;
use super::super::widget;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// Reserved id family: the next free block after the Worlds panel's (0xC000).
pub(crate) const FADE: AssetId = AssetId(ID_BASE + 0xD000);

// How long a moving shot holds, and how long every shot's fade in and out
// take. A shot spends most of its time clear: the fades are the punctuation,
// not the show.
const SHOT_SECS: f32 = 13.0;
// The spin covers a whole turn, so it takes twice as long to stay as calm as
// the shots that only cross a slice of one.
const SPIN_SECS: f32 = 26.0;
const FADE_SECS: f32 = 1.1;
// A hitch (a rebuild, a window drag) must not skip a shot.
const MAX_DT: f32 = 0.1;

// How far outside the fitted framing distance each moving shot sits.
const ORBIT_DISTANCE: f32 = 1.15;
const DRIFT_DISTANCE: f32 = 1.2;
// Camera elevation above the bounds centre, as an angle off the horizontal.
// Slightly raised: a look down over the world reads as an establishing shot,
// while a level camera reads as a screenshot.
const ELEVATION: f32 = 0.24;
// Orbit sweep rate in radians per second. A shot covers well under a quarter
// turn, which is the difference between ambient motion and a demo reel.
const ORBIT_RATE: f32 = 0.045;
// How far the drift trucks sideways over its shot, in bounding-sphere radii.
const DRIFT_SPAN: f32 = 0.5;
// Where each shot starts its sweep, so the cycle does not replay one angle.
const ORBIT_START: f32 = 0.6;
const DRIFT_AZIMUTH: f32 = 4.1;

// Bounds smaller than this have nothing to frame: an empty world, or one whose
// props all collapsed to a point.
const MIN_RADIUS: f32 = 1.0e-3;
// The largest sphere a shot frames, in world units. Standing back far enough
// to fit a whole street (or a world whose bounds a sky dome inflates) puts the
// camera outside the scene, looking at the back of its sky. Past this the
// shots hold their scale and let the world overrun the frame, which is what an
// establishing shot of a large place looks like anyway.
const MAX_FRAMED_RADIUS: f32 = 18.0;

// One shot of the cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Shot {
    // A wide sweep around the world.
    Orbit,
    // A full turn on the spot, from the world's own camera: the view the
    // author set up, looking around from where they left it.
    Spin,
    // A lateral truck across the world, holding it centred.
    Drift,
}

impl Shot {
    // How long this shot holds before the cycle hands over.
    const fn secs(self) -> f32 {
        match self {
            Shot::Spin => SPIN_SECS,
            _ => SHOT_SECS,
        }
    }
}

// The cycle, in order. Each hands over to the next through black.
const CYCLE: [Shot; 3] = [Shot::Orbit, Shot::Spin, Shot::Drift];

// What the shots frame: the previewed world's bounds centre, the sphere radius
// they frame it at (capped, see `MAX_FRAMED_RADIUS`), the distance that fits
// that sphere in the view, and the pose the world's own camera holds (which
// the spin turns on the spot from).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Framing {
    pub center: [f32; 3],
    pub radius: f32,
    pub distance: f32,
    pub home: CameraPose,
}

impl Framing {
    // The framing for a world-space AABB, or `None` when there is nothing to
    // frame: an empty or degenerate bounds, or one a stale transform left
    // non-finite. A world with no framing gets no cinematic at all, which is
    // what leaves the seeded empty scene on its own camera.
    pub(crate) fn new(
        mn: [f32; 3],
        mx: [f32; 3],
        fov_y_radians: f32,
        aspect: f32,
        home: CameraPose,
    ) -> Option<Framing> {
        if !mn.iter().chain(mx.iter()).all(|c| c.is_finite()) {
            return None;
        }
        let (bounds_center, extent) = bounding_sphere(mn, mx);
        if extent <= MIN_RADIUS {
            return None;
        }
        let radius = extent.min(MAX_FRAMED_RADIUS);
        let distance = fit_distance(radius, fov_y_radians, aspect);
        if !distance.is_finite() {
            return None;
        }
        // A world too large to frame whole is framed on what its own camera
        // faces rather than on the middle of its bounds: the centre of a
        // street (or of a sky dome that inflated the bounds) is a point in the
        // air with nothing at it, while the author's view is of something.
        let center = match extent > MAX_FRAMED_RADIUS {
            true => ahead_of(&home, distance),
            false => bounds_center,
        };
        Some(Framing {
            center,
            radius,
            distance,
            home,
        })
    }
}

// The running cycle: which shot is up and how far into it the clock is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Cinematic {
    shot: usize,
    elapsed: f32,
}

impl Cinematic {
    // A fresh cycle, at the first shot's opening black.
    pub(crate) fn new() -> Self {
        Cinematic {
            shot: 0,
            elapsed: 0.0,
        }
    }

    // Step the clock by a frame. `dt` is clamped, so a stalled frame costs the
    // shot a moment rather than jumping past the next one.
    pub(crate) fn advance(&mut self, dt: f32) {
        self.elapsed += dt.clamp(0.0, MAX_DT);
        while self.elapsed >= self.shot().secs() {
            self.elapsed -= self.shot().secs();
            self.shot = (self.shot + 1) % CYCLE.len();
        }
    }

    pub(crate) fn shot(&self) -> Shot {
        CYCLE[self.shot]
    }

    // The fade's alpha this moment: opaque at a shot boundary, clear through
    // the shot's body. Both ends of a shot are black, so a handover is one
    // continuous dip rather than two.
    pub(crate) fn fade_alpha(&self) -> f32 {
        let (t, secs) = (self.elapsed, self.shot().secs());
        let alpha = if t < FADE_SECS {
            1.0 - t / FADE_SECS
        } else if t > secs - FADE_SECS {
            (t - (secs - FADE_SECS)) / FADE_SECS
        } else {
            0.0
        };
        alpha.clamp(0.0, 1.0)
    }

    // The camera pose this moment's shot holds.
    pub(crate) fn pose(&self, f: &Framing) -> CameraPose {
        let t = self.elapsed;
        match self.shot() {
            Shot::Orbit => ring_pose(f, ORBIT_START + ORBIT_RATE * t, f.distance * ORBIT_DISTANCE),
            // The world's own camera, turning where it stands: the position
            // and the tilt the author set, carried once around.
            Shot::Spin => CameraPose {
                yaw: f.home.yaw + std::f32::consts::TAU * (t / SPIN_SECS),
                ..f.home
            },
            Shot::Drift => {
                let s = t / SHOT_SECS - 0.5;
                let base = ring_pose(f, DRIFT_AZIMUTH, f.distance * DRIFT_DISTANCE);
                let right = right_of(DRIFT_AZIMUTH);
                let offset = DRIFT_SPAN * f.radius * s;
                let position = [
                    base.position[0] + right[0] * offset,
                    base.position[1],
                    base.position[2] + right[2] * offset,
                ];
                aimed(position, f.center)
            }
        }
    }
}

// The point `distance` ahead of a camera at its own height: where its view
// lands, taken horizontally so a camera tipped at the sky or the floor still
// anchors a shot in the world.
fn ahead_of(home: &CameraPose, distance: f32) -> [f32; 3] {
    let (sin, cos) = home.yaw.sin_cos();
    [
        home.position[0] - sin * distance,
        home.position[1],
        home.position[2] - cos * distance,
    ]
}

// A pose on the ring `distance` from the centre at `azimuth`, raised by the
// standing elevation and aimed back at the centre.
fn ring_pose(f: &Framing, azimuth: f32, distance: f32) -> CameraPose {
    let horizontal = distance * ELEVATION.cos();
    let position = [
        f.center[0] + azimuth.sin() * horizontal,
        f.center[1] + distance * ELEVATION.sin(),
        f.center[2] + azimuth.cos() * horizontal,
    ];
    aimed(position, f.center)
}

// The horizontal unit vector at right angles to an azimuth, which is the
// direction the drift trucks along.
fn right_of(azimuth: f32) -> [f32; 3] {
    [azimuth.cos(), 0.0, -azimuth.sin()]
}

// The pose at `position` looking at `target`, in the free-fly yaw/pitch basis
// (`framing::forward` is its inverse).
fn aimed(position: [f32; 3], target: [f32; 3]) -> CameraPose {
    let d = [
        target[0] - position[0],
        target[1] - position[1],
        target[2] - position[2],
    ];
    let flat = (d[0] * d[0] + d[2] * d[2]).sqrt();
    CameraPose {
        position,
        yaw: (-d[0]).atan2(-d[2]),
        pitch: d[1].atan2(flat),
    }
}

// Draw the fade over the whole window at `alpha`, or hide it entirely. Black
// rather than a tint: a shot hands over through darkness, not through a dim.
// It is a UI-layer sprite, drawn over the tonemapped image, so it never reaches
// the scene the auto-exposure meters.
pub(crate) fn apply(world: &mut World, vp: [f32; 2], alpha: f32) {
    let tint = [0.0, 0.0, 0.0, alpha.clamp(0.0, 1.0)];
    widget::place_sprite(world, FADE, [0.0, 0.0, vp[0], vp[1]], tint, alpha > 0.0);
}

pub(crate) fn hide(world: &mut World) {
    widget::set_sprite_visible(world, FADE, false);
}

#[cfg(test)]
mod tests {
    use super::super::super::framing::forward;
    use super::*;

    const FOV: f32 = std::f32::consts::FRAC_PI_3;
    const ASPECT: f32 = 16.0 / 9.0;

    // The world's own camera: off to one side of the box, level, facing -Z.
    const HOME: CameraPose = CameraPose {
        position: [1.0, 1.7, 9.0],
        yaw: 0.3,
        pitch: -0.05,
    };

    fn framing() -> Framing {
        Framing::new([-4.0, 0.0, -4.0], [4.0, 3.0, 4.0], FOV, ASPECT, HOME)
            .expect("a real box frames")
    }

    // Wind the clock to the opening of `shot`.
    fn at(shot: Shot) -> Cinematic {
        let mut c = Cinematic::new();
        while c.shot() != shot {
            c.advance(MAX_DT);
        }
        c
    }

    // Distance from a pose to the framed centre.
    fn range(p: &CameraPose, f: &Framing) -> f32 {
        let d = [
            p.position[0] - f.center[0],
            p.position[1] - f.center[1],
            p.position[2] - f.center[2],
        ];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }

    // How far off the framed centre a pose is pointing, in world units at the
    // centre's range.
    fn aim_error(p: &CameraPose, f: &Framing) -> f32 {
        let fw = forward(p.yaw, p.pitch);
        let d = range(p, f);
        (0..3)
            .map(|a| (p.position[a] + fw[a] * d - f.center[a]).abs())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn the_cycle_walks_every_shot_and_wraps() {
        let mut c = Cinematic::new();
        assert_eq!(c.shot(), Shot::Orbit);
        let step = |c: &mut Cinematic| {
            // The clamp caps a single step, so a shot is crossed in frames.
            let held = c.shot();
            while c.shot() == held {
                c.advance(MAX_DT);
            }
        };
        step(&mut c);
        assert_eq!(c.shot(), Shot::Spin);
        step(&mut c);
        assert_eq!(c.shot(), Shot::Drift);
        step(&mut c);
        assert_eq!(c.shot(), Shot::Orbit, "the cycle wraps");
        // The spin covers a whole turn, so it is given longer to stay calm.
        assert!(Shot::Spin.secs() > Shot::Orbit.secs());
    }

    #[test]
    fn a_hitch_cannot_skip_a_shot() {
        let mut c = Cinematic::new();
        // A frame that took a whole cycle still advances one clamped step.
        c.advance(3.0 * SHOT_SECS);
        assert_eq!(c.shot(), Shot::Orbit);
        assert!((c.elapsed - MAX_DT).abs() < 1e-6);
        // Nor does time run backwards on a negative dt.
        c.advance(-5.0);
        assert!((c.elapsed - MAX_DT).abs() < 1e-6);
    }

    #[test]
    fn the_fade_is_black_at_every_boundary_and_clear_between() {
        let mut c = Cinematic::new();
        assert_eq!(c.fade_alpha(), 1.0, "a cycle opens on black");
        // The clamp caps a step, so the fade is crossed in frames.
        while c.elapsed < FADE_SECS * 0.5 {
            c.advance(MAX_DT);
        }
        let alpha = c.fade_alpha();
        assert!((0.4..0.6).contains(&alpha), "fading in: {alpha}");
        // Clear through the body of the shot.
        while c.elapsed < SHOT_SECS * 0.5 {
            c.advance(MAX_DT);
        }
        assert_eq!(c.fade_alpha(), 0.0);
        // Opaque again as the shot hands over, and the next opens on black:
        // one continuous dip, not two.
        let mut closing = 0.0;
        while c.shot() == Shot::Orbit {
            closing = c.fade_alpha();
            c.advance(MAX_DT);
        }
        assert!(closing > 0.9, "the shot closes on black: {closing}");
        assert_eq!(c.shot(), Shot::Spin);
        assert!(c.fade_alpha() > 0.9, "and the next opens on it");
    }

    // Switching worlds mid-fade cannot leave the screen stuck black: the new
    // cycle starts opaque and clears within one fade.
    #[test]
    fn a_reset_mid_fade_starts_over_and_clears() {
        let mut c = Cinematic::new();
        while c.elapsed < SHOT_SECS - FADE_SECS * 0.5 {
            c.advance(MAX_DT);
        }
        assert!(c.fade_alpha() > 0.0, "mid fade-out");

        let mut fresh = Cinematic::new();
        assert_eq!(fresh.fade_alpha(), 1.0);
        assert_eq!(fresh.shot(), Shot::Orbit, "a new preview opens on shot one");
        let mut last = 1.0;
        while fresh.elapsed < FADE_SECS {
            fresh.advance(MAX_DT);
            let alpha = fresh.fade_alpha();
            assert!(alpha <= last, "the fade only ever clears: {alpha} > {last}");
            last = alpha;
        }
        assert_eq!(fresh.fade_alpha(), 0.0);
    }

    #[test]
    fn framing_needs_real_bounds() {
        // A point-sized (or empty) world has nothing to frame.
        assert_eq!(Framing::new([1.0; 3], [1.0; 3], FOV, ASPECT, HOME), None);
        // Nor does one a stale transform left non-finite.
        assert_eq!(
            Framing::new([f32::NEG_INFINITY; 3], [1.0; 3], FOV, ASPECT, HOME),
            None
        );
        assert_eq!(
            Framing::new([0.0; 3], [f32::NAN; 3], FOV, ASPECT, HOME),
            None
        );

        // A world far larger than a shot can frame is framed at the cap, on
        // what its own camera faces, so no shot stands out beyond the scene
        // (or its sky) to fit it all in.
        let street = Framing::new([-200.0, 0.0, -30.0], [200.0, 40.0, 30.0], FOV, ASPECT, HOME)
            .expect("a street frames");
        assert_eq!(street.radius, MAX_FRAMED_RADIUS);
        assert!(
            street.distance < 4.0 * MAX_FRAMED_RADIUS,
            "and stands a shot's distance out, not the street's: {}",
            street.distance
        );
        assert_eq!(
            street.center[1], HOME.position[1],
            "the subject is at the camera's own height, never up in the sky"
        );
        let fw = forward(HOME.yaw, 0.0);
        for a in [0, 2] {
            let ahead = HOME.position[a] + fw[a] * street.distance;
            assert!(
                (street.center[a] - ahead).abs() < 1e-3,
                "and straight ahead of it"
            );
        }

        let f = framing();
        assert!(f.radius < MAX_FRAMED_RADIUS, "a small world frames whole");
        assert_eq!(f.center, [0.0, 1.5, 0.0], "on the bounds it fits");
        assert!(
            f.distance > f.radius,
            "the camera stands outside the bounds"
        );
        assert_eq!(f.home, HOME, "and the world's own camera is carried");
    }

    // The shots that move around the world all keep it in frame. The spin is
    // not one of them: it stands where the world's own camera stands and looks
    // wherever that camera was pointed.
    #[test]
    fn the_moving_shots_stand_outside_the_bounds_and_look_at_them() {
        let f = framing();
        let mut c = Cinematic::new();
        for _ in 0..((Shot::Orbit.secs() + SPIN_SECS + Shot::Drift.secs()) / MAX_DT) as usize {
            if c.shot() != Shot::Spin {
                let p = c.pose(&f);
                assert!(
                    range(&p, &f) > f.radius,
                    "{:?} sits inside the bounds",
                    c.shot()
                );
                assert!(
                    p.position[1] > f.center[1],
                    "{:?} looks down on the world",
                    c.shot()
                );
                assert!(aim_error(&p, &f) < 1e-3, "{:?} loses the centre", c.shot());
            }
            c.advance(MAX_DT);
        }
    }

    #[test]
    fn the_orbit_sweeps_one_way_at_a_calm_rate() {
        let f = framing();
        let mut c = Cinematic::new();
        let start = c.pose(&f);
        let mut last = start.yaw;
        let mut swept = 0.0_f32;
        while c.elapsed + MAX_DT < Shot::Orbit.secs() {
            c.advance(MAX_DT);
            let yaw = c.pose(&f).yaw;
            let step = shortest(yaw - last);
            assert!(step >= 0.0, "the sweep never reverses: {step}");
            swept += step.abs();
            last = yaw;
        }
        // A shot covers a slice of a turn, not a lap.
        assert!(swept > 0.1, "the orbit moves: {swept}");
        assert!(swept < std::f32::consts::PI * 0.5, "and slowly: {swept}");
        // The radius is held: this shot circles, it does not close in.
        let end = c.pose(&f);
        assert!((range(&start, &f) - range(&end, &f)).abs() < 1e-3);
    }

    // The spin holds the world's own camera and turns it once around: the
    // author's vantage, looking about from where they left it.
    #[test]
    fn the_spin_turns_once_where_the_worlds_camera_stands() {
        let f = framing();
        let mut c = at(Shot::Spin);
        assert_eq!(c.pose(&f).position, HOME.position, "it opens at home");
        let mut swept = 0.0_f32;
        let mut last = c.pose(&f).yaw;
        while c.shot() == Shot::Spin {
            c.advance(MAX_DT);
            if c.shot() != Shot::Spin {
                break;
            }
            let p = c.pose(&f);
            assert_eq!(p.position, HOME.position, "the camera never leaves home");
            assert_eq!(p.pitch, HOME.pitch, "nor changes the tilt it was set at");
            let step = shortest(p.yaw - last);
            assert!(step >= 0.0, "the turn never reverses: {step}");
            swept += step;
            last = p.yaw;
        }
        // One full turn over the shot, and calmly: well under the orbit's rate
        // per second despite covering far more ground.
        let turn = std::f32::consts::TAU;
        assert!(
            (swept - turn).abs() < 0.05,
            "a full turn and no more: {swept}"
        );
        assert!(turn / SPIN_SECS < 0.3, "at a calm rate");
    }

    #[test]
    fn the_drift_trucks_sideways_holding_the_centre() {
        let f = framing();
        let mut c = at(Shot::Drift);
        let opened = c.pose(&f);
        while c.elapsed + MAX_DT < Shot::Drift.secs() {
            c.advance(MAX_DT);
        }
        let closed = c.pose(&f);
        let travelled = ((closed.position[0] - opened.position[0]).powi(2)
            + (closed.position[2] - opened.position[2]).powi(2))
        .sqrt();
        let span = DRIFT_SPAN * f.radius;
        assert!(
            travelled > span * 0.9 && travelled <= span + 1e-3,
            "the truck covers its span: {travelled} of {span}"
        );
        assert!(
            (closed.position[1] - opened.position[1]).abs() < 1e-6,
            "and stays level"
        );
        assert!(aim_error(&closed, &f) < 1e-3);
    }

    fn shortest(delta: f32) -> f32 {
        use std::f32::consts::{PI, TAU};
        (delta + PI).rem_euclid(TAU) - PI
    }
}
