// One reset decision per frame from the view, the frame's clock and the
// causes reported since the last frame.

use super::HistoryResetCauses;
use super::motion::{MotionReference, ViewStep};
use crate::ecs::Entity;
use crate::ecs::asset_id::AssetId;
use crate::transform::Mat4;

/// How many simulation ticks past the frame it was raised for a camera cut
/// waits for the camera to move before it lands anyway.
const MAX_PENDING_TICKS: u32 = 2;
/// How many frames a camera cut waits at most, for a world whose simulation is
/// not running.
const MAX_PENDING_FRAMES: u32 = 16;

/// A frame's view as the tracker compares it with the previous one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistoryView {
    /// The entity carrying the camera the frame draws from, if any. Drawing
    /// from a different camera is a cut.
    pub camera: Option<Entity>,
    /// World-space camera position, before any render-origin rebase, so a
    /// rebase is not mistaken for a teleport.
    pub position: [f32; 3],
    /// The camera's view matrix; only its rotation is read.
    pub view: Mat4,
    /// Vertical field of view in radians.
    pub fov_y_radians: f32,
    /// The active scene, or `None` in a world without scenes.
    pub scene: Option<AssetId>,
}

impl HistoryView {
    fn is_finite(&self) -> bool {
        self.position.iter().all(|v| v.is_finite())
            && self.view.iter().flatten().all(|v| v.is_finite())
            && self.fov_y_radians.is_finite()
    }
}

/// How much time a frame covered.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameClock {
    /// Seconds since the previous frame, which keep running while the world
    /// is paused.
    pub dt: f32,
    /// Fixed simulation ticks run this frame: none while the world is paused,
    /// and none on a frame drawn between two ticks.
    pub ticks: u32,
    /// Seconds each tick advances.
    pub tick_dt: f32,
}

impl FrameClock {
    /// Seconds of simulation this frame ran.
    pub fn sim_seconds(&self) -> f32 {
        self.ticks as f32 * self.tick_dt
    }
}

/// A raised camera cut waiting for the camera to move.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PendingCut {
    frames: u32,
    ticks: u32,
}

/// The previous frame's view, the camera's recent motion and the causes
/// reported since, turned into one reset decision per frame.
///
/// A reported [`CAMERA_CUT`](HistoryResetCauses::CAMERA_CUT) is the primary
/// signal. It lands on the first frame the camera moves off the path its
/// recent motion predicts, which may come after the frame it was raised for
/// when the system raising it runs before the one that moves the camera. It
/// lands anyway once [`MAX_PENDING_TICKS`] more ticks run without a move (or
/// [`MAX_PENDING_FRAMES`] frames, in a world that is not simulating), and any
/// reset in the meantime satisfies it.
///
/// Without one, a frame is still a cut when its step strays from the
/// predicted one, in position, rotation (roll included) or zoom, further than
/// the camera's recent rates explain, so a steadily fast camera keeps its
/// history and a sudden jump does not. A change of camera, a non-finite pose
/// and the frame after one are cuts too.
#[derive(Clone, Debug, Default)]
pub struct HistoryResetTracker {
    prev: Option<HistoryView>,
    motion: MotionReference,
    // The step of the last frame that reset. The step after it may be the
    // camera carrying on at that rate, launched straight to speed, rather
    // than a second jump.
    after_reset: Option<ViewStep>,
    pending_cut: Option<PendingCut>,
    requested: HistoryResetCauses,
    reissued: HistoryResetCauses,
}

impl HistoryResetTracker {
    /// Report causes found outside the view, applied to the next observed
    /// frame; a camera cut waits for the camera to move.
    pub fn request(&mut self, causes: HistoryResetCauses) {
        self.requested = self.requested.union(causes);
    }

    /// A frame that decided `causes` was dropped before it drew: apply them
    /// to the next observed frame as they are.
    pub fn reissue(&mut self, causes: HistoryResetCauses) {
        self.reissued = self.reissued.union(causes);
    }

    /// Compare `view` with the previous frame's and return why this frame's
    /// history is invalid. The first frame resets nothing: there is no
    /// history yet.
    pub fn observe(&mut self, view: HistoryView, clock: FrameClock) -> HistoryResetCauses {
        let requested = core::mem::take(&mut self.requested);
        let mut causes = core::mem::take(&mut self.reissued)
            .union(requested.without(HistoryResetCauses::CAMERA_CUT));
        let raised =
            requested.contains(HistoryResetCauses::CAMERA_CUT) && self.pending_cut.is_none();
        if raised {
            self.pending_cut = Some(PendingCut::default());
        }
        let Some(prev) = self.prev.replace(view) else {
            self.pending_cut = None;
            return HistoryResetCauses::NONE;
        };
        if prev.scene != view.scene {
            causes = causes.union(HistoryResetCauses::SCENE_SWITCH);
        }
        if prev.camera != view.camera {
            self.motion = MotionReference::default();
            return self.discontinuity(causes);
        }
        if !prev.is_finite() || !view.is_finite() {
            return self.discontinuity(causes);
        }
        let step = ViewStep::between(&prev, &view, clock.dt);
        if step.as_ref().is_some_and(|s| self.is_jump(s)) {
            causes = causes.union(HistoryResetCauses::CAMERA_CUT);
        }
        if self.lands_pending_cut(step.as_ref(), clock, raised, causes.any()) {
            causes = causes.union(HistoryResetCauses::CAMERA_CUT);
        }
        match step {
            // An unchanged view over simulated time is a camera that has
            // stopped, and its rates decay by that time. With no ticks run (a
            // paused world, or a frame drawn between two ticks) it holds.
            None if clock.ticks > 0 => {
                self.motion.rest(clock.sim_seconds());
                self.after_reset = None;
            }
            None => {}
            // A reset frame's step is a jump, not motion: the rates stay as
            // they were before it.
            Some(step) if causes.any() => self.after_reset = Some(step),
            // A step while a raised cut waits may hold the jump: it is not
            // learned as motion either.
            Some(_) if self.pending_cut.is_some() => {}
            Some(step) => {
                if let Some(jump) = self.after_reset.take()
                    && self.motion.is_cut(&step)
                {
                    self.motion = MotionReference::launched(&jump);
                }
                self.motion.absorb(&step);
            }
        }
        causes
    }

    // A frame that cannot be compared with the one before it.
    fn discontinuity(&mut self, causes: HistoryResetCauses) -> HistoryResetCauses {
        self.after_reset = None;
        self.pending_cut = None;
        causes.union(HistoryResetCauses::CAMERA_CUT)
    }

    // A step off the predicted path, and, just after a reset, off the path of
    // a camera carrying on at the reset step's rate as well.
    fn is_jump(&self, step: &ViewStep) -> bool {
        self.motion.is_cut(step)
            && self
                .after_reset
                .as_ref()
                .is_none_or(|jump| MotionReference::launched(jump).is_cut(step))
    }

    // Whether a pending cut lands this frame: the frame resets anyway, the
    // camera left its predicted path, or the cut has waited as long as it
    // may. The frame it was raised for does not count toward the wait.
    fn lands_pending_cut(
        &mut self,
        step: Option<&ViewStep>,
        clock: FrameClock,
        raised: bool,
        resetting: bool,
    ) -> bool {
        let Some(mut waited) = self.pending_cut else {
            return false;
        };
        if !raised {
            waited.ticks += clock.ticks;
            waited.frames += 1;
        }
        let lands = resetting
            || step.is_some_and(|s| self.motion.departs(s))
            || waited.ticks >= MAX_PENDING_TICKS
            || waited.frames >= MAX_PENDING_FRAMES;
        self.pending_cut = (!lands).then_some(waited);
        lands
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::camera::view_matrix;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::f32::consts::{FRAC_PI_2, TAU};
    use core::num::NonZeroU32;

    const FOV: f32 = 1.2;
    const CUT: HistoryResetCauses = HistoryResetCauses::CAMERA_CUT;

    #[derive(Clone, Copy)]
    struct Pose {
        position: [f32; 3],
        yaw: f32,
        pitch: f32,
        roll: f32,
        fov: f32,
    }

    impl Pose {
        fn at(position: [f32; 3]) -> Self {
            Self {
                position,
                yaw: 0.0,
                pitch: 0.0,
                roll: 0.0,
                fov: FOV,
            }
        }

        fn yawed(self, yaw: f32) -> Self {
            Self { yaw, ..self }
        }

        fn view(self) -> HistoryView {
            HistoryView {
                camera: Some(Entity::new(1, NonZeroU32::MIN)),
                position: self.position,
                view: rolled(view_matrix(self.position, self.yaw, self.pitch), self.roll),
                fov_y_radians: self.fov,
                scene: Some(AssetId(1)),
            }
        }
    }

    const TICK: f32 = crate::ecs::SimTiming::TICK_DT;

    // A frame of `dt` seconds over which the simulation ran its share of ticks
    // (at least one, at most its catch-up limit).
    fn ran(dt: f32) -> FrameClock {
        let max = crate::ecs::SimTiming::MAX_TICKS_PER_FRAME;
        ticks_over(dt, ((dt / TICK).round() as u32).clamp(1, max))
    }

    fn ticks_over(dt: f32, ticks: u32) -> FrameClock {
        FrameClock {
            dt,
            ticks,
            tick_dt: TICK,
        }
    }

    // A view matrix turned about its own viewing axis.
    fn rolled(mut view: Mat4, roll: f32) -> Mat4 {
        let (s, c) = crate::math::sin_cos(roll);
        for col in &mut view {
            let (right, up) = (col[0], col[1]);
            col[0] = c * right + s * up;
            col[1] = -s * right + c * up;
        }
        view
    }

    fn play(
        tracker: &mut HistoryResetTracker,
        frames: &[(Pose, FrameClock)],
    ) -> Vec<HistoryResetCauses> {
        frames
            .iter()
            .map(|&(pose, clock)| tracker.observe(pose.view(), clock))
            .collect()
    }

    fn cut_frames(causes: &[HistoryResetCauses]) -> Vec<usize> {
        causes
            .iter()
            .enumerate()
            .filter(|(_, c)| c.contains(CUT))
            .map(|(i, _)| i)
            .collect()
    }

    fn cuts(frames: &[(Pose, FrameClock)]) -> Vec<usize> {
        cut_frames(&play(&mut HistoryResetTracker::default(), frames))
    }

    fn still(pose: Pose, n: usize, dt: f32) -> Vec<(Pose, FrameClock)> {
        vec![(pose, ran(dt)); n]
    }

    // A camera that pulls away from rest at 300 units/s^2, cruises at 300
    // units/s along -Z for `cruise` seconds and turns at 90 degrees/s while
    // it does, sampled at `fps`.
    fn fast_drive(fps: f32, cruise: f32) -> Vec<(Pose, FrameClock)> {
        let dt = 1.0 / fps;
        let accel_end = 1.0;
        let frames = ((accel_end + cruise) * fps).round() as usize;
        let distance = |t: f32| {
            if t < accel_end {
                150.0 * t * t
            } else {
                150.0 + 300.0 * (t - accel_end)
            }
        };
        (0..=frames)
            .map(|i| {
                let t = i as f32 * dt;
                let pose =
                    Pose::at([0.0, 1.0, -distance(t)]).yawed((t - accel_end).max(0.0) * FRAC_PI_2);
                (pose, ran(dt))
            })
            .collect()
    }

    // Carry on from the last frame along its own step `n` more times, each
    // moved by `shift`.
    fn carry_on(frames: &mut Vec<(Pose, FrameClock)>, n: usize, shift: [f32; 3]) {
        let len = frames.len();
        let (a, b) = (frames[len - 2].0, frames[len - 1].0);
        let clock = frames[len - 1].1;
        let step = [
            b.position[0] - a.position[0],
            b.position[1] - a.position[1],
            b.position[2] - a.position[2],
        ];
        for i in 1..=n {
            let mut pose = b;
            for axis in 0..3 {
                pose.position[axis] += step[axis] * i as f32 + shift[axis];
            }
            pose.yaw = b.yaw + (b.yaw - a.yaw) * i as f32;
            frames.push((pose, clock));
        }
    }

    #[test]
    fn the_first_frame_resets_nothing_even_when_asked() {
        let mut tracker = HistoryResetTracker::default();
        tracker.request(HistoryResetCauses::SETTINGS_CHANGE.union(CUT));
        let causes = tracker.observe(Pose::at([0.0; 3]).view(), ran(1.0 / 60.0));
        assert_eq!(causes, HistoryResetCauses::NONE);
        // Nor does the cut wait for a later one.
        assert!(
            !tracker
                .observe(Pose::at([0.0; 3]).view(), ran(1.0 / 60.0))
                .any()
        );
    }

    // The first step after load is judged against a camera at rest rather
    // than taken as the rate the camera moves at.
    #[test]
    fn the_first_step_is_judged() {
        let dt = 1.0 / 60.0;
        let causes = cuts(&[
            (Pose::at([0.0; 3]), ran(dt)),
            (Pose::at([50.0, 0.0, 0.0]), ran(dt)),
        ]);
        assert_eq!(causes, [1]);
    }

    #[test]
    fn steady_fast_travel_is_never_a_cut_at_any_frame_rate() {
        for fps in [20.0, 30.0, 60.0] {
            assert_eq!(cuts(&fast_drive(fps, 4.0)), [] as [usize; 0], "{fps} fps");
        }
    }

    // A 60 Hz simulation drawn at 50 fps lands one step on some frames and two
    // on others, so the distance per frame doubles while the frame time holds.
    #[test]
    fn a_fixed_step_camera_sampled_unevenly_is_never_a_cut() {
        let tick = 1.0 / 60.0;
        let mut frames = Vec::new();
        let mut ticks = 0u32;
        for frame in 0..400 {
            // Pull away from rest over the first two seconds.
            let speed = 300.0 * (frame as f32 / 100.0).min(1.0);
            ticks += if frame % 5 == 0 { 2 } else { 1 };
            let z = frames
                .last()
                .map_or(0.0, |(p, _): &(Pose, FrameClock)| p.position[2]);
            let moved = if frame % 5 == 0 { 2.0 } else { 1.0 } * tick * speed;
            frames.push((Pose::at([0.0, 1.0, z - moved]), ran(0.02)));
        }
        assert!(ticks > 0);
        assert_eq!(cuts(&frames), [] as [usize; 0]);
    }

    // The simulation catches up over a long frame only as far as it may, so
    // the camera covers what the capped catch-up runs; the allowance is capped
    // the same way.
    #[test]
    fn a_hitch_on_a_fast_camera_is_not_a_cut() {
        let mut frames = fast_drive(30.0, 2.0);
        let &(last, clock) = frames.last().expect("frames");
        let mut after = last;
        let ran_for = super::super::motion::MAX_STEP_SECONDS;
        after.position[2] -= 300.0 * ran_for;
        after.yaw += FRAC_PI_2 * ran_for;
        frames.push((after, ran(0.4)));
        let mut resumed = after;
        resumed.position[2] -= 300.0 * clock.dt;
        resumed.yaw += FRAC_PI_2 * clock.dt;
        frames.push((resumed, clock));
        assert_eq!(cuts(&frames), [] as [usize; 0]);
    }

    // A hitch covers no more motion than the simulation could run, so a
    // jump hidden in one is still a jump.
    #[test]
    fn a_teleport_inside_a_hitch_is_a_cut() {
        let mut frames = fast_drive(30.0, 2.0);
        let &(last, _) = frames.last().expect("frames");
        let mut after = last;
        after.position[2] -= 300.0 * 2.0;
        frames.push((after, ran(2.0)));
        assert_eq!(cuts(&frames), [frames.len() - 1]);
    }

    #[test]
    fn a_teleport_mid_travel_is_one_cut() {
        let mut frames = fast_drive(30.0, 2.0);
        let cut_at = frames.len();
        carry_on(&mut frames, 30, [400.0, 0.0, 0.0]);
        // The camera carries on at the speed it had, so only the jump cuts.
        assert_eq!(cuts(&frames), [cut_at]);
    }

    // Moving forward at 10 units a frame, a 20 unit jump sideways is no
    // longer than two frames of travel, but nowhere near where the camera
    // was headed.
    #[test]
    fn a_lateral_teleport_while_moving_is_a_cut() {
        let dt = 1.0 / 30.0;
        let mut frames = Vec::new();
        let mut z = 0.0;
        for i in 0..60 {
            z -= 10.0 * (i as f32 / 30.0).min(1.0);
            frames.push((Pose::at([0.0, 0.0, z]), ran(dt)));
        }
        assert_eq!(cuts(&frames), [] as [usize; 0]);
        let cut_at = frames.len();
        carry_on(&mut frames, 10, [20.0, 0.0, 0.0]);
        assert_eq!(cuts(&frames), [cut_at]);
    }

    // Two jumps on consecutive frames: the second is not the camera carrying
    // on from the first.
    #[test]
    fn a_two_stage_teleport_cuts_twice() {
        let dt = 1.0 / 60.0;
        let mut frames = still(Pose::at([0.0; 3]), 5, dt);
        frames.push((Pose::at([400.0, 0.0, 0.0]), ran(dt)));
        frames.push((Pose::at([400.0, 0.0, 300.0]), ran(dt)));
        frames.extend(still(Pose::at([400.0, 0.0, 300.0]), 5, dt));
        assert_eq!(cuts(&frames), [5, 6]);
    }

    // A jump is not motion: a 400 unit respawn leaves the camera judged as
    // the still camera it was, so a later small jump still cuts.
    #[test]
    fn a_cut_then_a_further_jump_is_a_cut() {
        let dt = 1.0 / 60.0;
        let mut frames = still(Pose::at([0.0; 3]), 5, dt);
        frames.extend(still(Pose::at([400.0, 0.0, 0.0]), 10, dt));
        frames.push((Pose::at([405.0, 0.0, 0.0]), ran(dt)));
        assert_eq!(cuts(&frames), [5, 15]);
    }

    // A leap from rest straight to 300 units/s is a discontinuity, but only
    // the leap: the frames after it move at the rate the camera now holds.
    #[test]
    fn a_camera_launched_straight_to_speed_cuts_once() {
        let dt = 1.0 / 30.0;
        let mut frames = still(Pose::at([0.0; 3]), 10, dt);
        frames.extend((1..60).map(|i| (Pose::at([0.0, 0.0, -10.0 * i as f32]), ran(dt))));
        assert_eq!(cuts(&frames), [10]);
    }

    #[test]
    fn a_snap_turn_is_a_cut_and_a_fast_steady_turn_is_not() {
        let dt = 1.0 / 30.0;
        // 360 degrees/s, eased in over half a second.
        let mut yaw = 0.0;
        let turning: Vec<_> = (0..90)
            .map(|i| {
                yaw += TAU * dt * (i as f32 / 15.0).min(1.0);
                (Pose::at([0.0; 3]).yawed(yaw), ran(dt))
            })
            .collect();
        assert_eq!(cuts(&turning), [] as [usize; 0]);

        let rest = Pose::at([0.0; 3]);
        assert_eq!(
            cuts(&[(rest, ran(dt)), (rest.yawed(FRAC_PI_2), ran(dt))]),
            [1]
        );
    }

    // Orbiting at 90 degrees/s, a snap back the other way is a turn no larger
    // than the camera's usual one, but in the wrong direction.
    #[test]
    fn a_reverse_snap_while_orbiting_is_a_cut() {
        let dt = 1.0 / 60.0;
        let rate = FRAC_PI_2;
        let orbit = |yaw: f32| {
            let (s, c) = crate::math::sin_cos(yaw);
            Pose::at([10.0 * s, 2.0, 10.0 * c]).yawed(yaw)
        };
        let mut frames: Vec<_> = (0..120)
            .map(|i| (orbit(i as f32 * rate * dt), ran(dt)))
            .collect();
        assert_eq!(cuts(&frames), [] as [usize; 0]);
        let back = 119.0 * rate * dt - 50f32.to_radians();
        let cut_at = frames.len();
        frames.push((orbit(back), ran(dt)));
        assert_eq!(cuts(&frames), [cut_at]);
    }

    #[test]
    fn a_pitch_snap_is_a_cut() {
        let dt = 1.0 / 60.0;
        let rest = Pose::at([0.0; 3]);
        let down = Pose {
            pitch: -1.2,
            ..rest
        };
        assert_eq!(cuts(&[(rest, ran(dt)), (down, ran(dt))]), [1]);
    }

    #[test]
    fn a_roll_snap_is_a_cut() {
        let dt = 1.0 / 60.0;
        let rest = Pose::at([0.0; 3]);
        let rolled = Pose {
            roll: FRAC_PI_2,
            ..rest
        };
        assert_eq!(cuts(&[(rest, ran(dt)), (rolled, ran(dt))]), [1]);
    }

    #[test]
    fn a_smooth_zoom_is_not_a_cut_and_a_snap_is() {
        let dt = 1.0 / 60.0;
        // 70 degrees down to 20 over half a second.
        let (wide, narrow) = (70f32.to_radians(), 20f32.to_radians());
        let zoom: Vec<_> = (0..=30)
            .map(|i| {
                let fov = wide * crate::math::powf(narrow / wide, i as f32 / 30.0);
                (
                    Pose {
                        fov,
                        ..Pose::at([0.0; 3])
                    },
                    ran(dt),
                )
            })
            .collect();
        assert_eq!(cuts(&zoom), [] as [usize; 0]);

        let open = Pose {
            fov: wide,
            ..Pose::at([0.0; 3])
        };
        let zoomed = Pose {
            fov: wide / 2.0,
            ..open
        };
        assert_eq!(cuts(&[(open, ran(dt)), (zoomed, ran(dt))]), [1]);
    }

    // A paused world hands the tracker the same view while the frame clock
    // runs; the camera resumes at speed without a cut.
    #[test]
    fn a_pause_holds_the_recent_motion() {
        let mut frames = fast_drive(30.0, 1.0);
        let &(last, clock) = frames.last().expect("frames");
        let paused = ticks_over(clock.dt, 0);
        frames.extend(vec![(last, paused); 1800]);
        let mut resumed = last;
        resumed.position[2] -= 300.0 * clock.dt;
        resumed.yaw += FRAC_PI_2 * clock.dt;
        frames.push((resumed, clock));
        assert_eq!(cuts(&frames), [] as [usize; 0]);
    }

    // A camera that stopped while the world ran forgets how fast it was: a
    // jump a minute later is judged as a jump from rest.
    #[test]
    fn a_stop_decays_the_recent_motion() {
        let mut frames = fast_drive(30.0, 1.0);
        let &(last, clock) = frames.last().expect("frames");
        frames.extend(vec![(last, clock); 1800]);
        let mut respawned = last;
        respawned.position[0] += 25.0;
        frames.push((respawned, clock));
        assert_eq!(cuts(&frames), [frames.len() - 1]);
    }

    #[test]
    fn a_turn_snap_after_a_stopped_flick_is_a_cut() {
        let dt = 1.0 / 60.0;
        let mut yaw = 0.0;
        let mut frames = Vec::new();
        // A flick at 20 degrees a frame, eased in, then held still.
        for i in 0..20 {
            yaw += 20f32.to_radians() * (i as f32 / 5.0).min(1.0);
            frames.push((Pose::at([0.0; 3]).yawed(yaw), ran(dt)));
        }
        assert_eq!(cuts(&frames), [] as [usize; 0]);
        frames.extend(still(Pose::at([0.0; 3]).yawed(yaw), 600, dt));
        frames.push((Pose::at([0.0; 3]).yawed(yaw + FRAC_PI_2), ran(dt)));
        assert_eq!(cuts(&frames), [frames.len() - 1]);
    }

    // A raised cut lands on the first frame the camera moves, not on the
    // frame it was raised for when the camera moves a frame later.
    #[test]
    fn a_raised_cut_waits_for_the_camera_to_move() {
        let dt = 1.0 / 60.0;
        let mut tracker = HistoryResetTracker::default();
        let home = Pose::at([0.0; 3]).view();
        tracker.observe(home, ran(dt));
        tracker.request(CUT);
        assert!(!tracker.observe(home, ran(dt)).any());
        // Under the fallback's floors, so only the raised cut can land here.
        let nudged = Pose::at([0.5, 0.0, 0.0]).view();
        assert_eq!(tracker.observe(nudged, ran(dt)), CUT);
        assert!(!tracker.observe(nudged, ran(dt)).any());
    }

    // A raised cut whose camera moved the same frame lands right away.
    #[test]
    fn a_raised_cut_with_the_move_lands_at_once() {
        let dt = 1.0 / 60.0;
        let mut tracker = HistoryResetTracker::default();
        tracker.observe(Pose::at([0.0; 3]).view(), ran(dt));
        tracker.request(CUT);
        assert_eq!(
            tracker.observe(Pose::at([0.5, 0.0, 0.0]).view(), ran(dt)),
            CUT
        );
    }

    // A raised cut while the camera moves steadily waits for the move off
    // its path, not just any move.
    #[test]
    fn a_raised_cut_on_a_moving_camera_waits_for_the_jump() {
        let dt = 1.0 / 30.0;
        let mut frames: Vec<_> = (0..60)
            .map(|i| (Pose::at([0.0, 0.0, -0.1 * i as f32]), ran(dt)))
            .collect();
        let mut tracker = HistoryResetTracker::default();
        play(&mut tracker, &frames);
        tracker.request(CUT);
        carry_on(&mut frames, 1, [0.0; 3]);
        let on_path = frames.last().expect("frames").0;
        assert!(!tracker.observe(on_path.view(), ran(dt)).any());
        carry_on(&mut frames, 1, [1.0, 0.0, 0.0]);
        let jumped = frames.last().expect("frames").0;
        assert_eq!(tracker.observe(jumped.view(), ran(dt)), CUT);
    }

    // A raised cut whose camera never moves lands anyway once it has waited
    // as long as it may.
    #[test]
    fn a_raised_cut_lands_after_its_wait() {
        let dt = 1.0 / 60.0;
        let mut tracker = HistoryResetTracker::default();
        let home = Pose::at([0.0; 3]).view();
        tracker.observe(home, ran(dt));
        tracker.request(CUT);
        // The frame it was raised for, then the ticks it may wait.
        assert!(!tracker.observe(home, ran(dt)).any());
        for _ in 0..MAX_PENDING_TICKS - 1 {
            assert!(!tracker.observe(home, ran(dt)).any());
        }
        assert_eq!(tracker.observe(home, ran(dt)), CUT);
        assert!(!tracker.observe(home, ran(dt)).any());
    }

    // In a world that is not simulating, a raised cut waits a bounded number
    // of frames.
    #[test]
    fn a_raised_cut_in_a_paused_world_lands_after_its_frame_cap() {
        let mut tracker = HistoryResetTracker::default();
        let home = Pose::at([0.0; 3]).view();
        let paused = ticks_over(1.0 / 60.0, 0);
        tracker.observe(home, paused);
        tracker.request(CUT);
        assert!(!tracker.observe(home, paused).any());
        for _ in 0..MAX_PENDING_FRAMES - 1 {
            assert!(!tracker.observe(home, paused).any());
        }
        assert_eq!(tracker.observe(home, paused), CUT);
    }

    // Drawn at 240 Hz over a 60 Hz simulation, a camera the simulation moves
    // a tick after the cut is raised moves several frames later: the wait is
    // counted in ticks, so the cut is still waiting for it.
    #[test]
    fn a_raised_cut_waits_through_frames_between_ticks() {
        let dt = 1.0 / 240.0;
        let mut tracker = HistoryResetTracker::default();
        let home = Pose::at([0.0; 3]).view();
        tracker.observe(home, ticks_over(dt, 1));
        tracker.request(CUT);
        for _ in 0..4 {
            assert!(!tracker.observe(home, ticks_over(dt, 0)).any());
        }
        let nudged = Pose::at([0.5, 0.0, 0.0]).view();
        assert_eq!(tracker.observe(nudged, ticks_over(dt, 1)), CUT);
    }

    // A frame that resets for another reason satisfies a waiting cut, so the
    // camera's later move does not reset again.
    #[test]
    fn any_reset_satisfies_a_raised_cut() {
        let dt = 1.0 / 60.0;
        let mut tracker = HistoryResetTracker::default();
        let home = Pose::at([0.0; 3]).view();
        tracker.observe(home, ran(dt));
        tracker.request(CUT.union(HistoryResetCauses::SETTINGS_CHANGE));
        let causes = tracker.observe(home, ran(dt));
        assert!(causes.contains(HistoryResetCauses::SETTINGS_CHANGE));
        assert!(causes.contains(CUT));
        let nudged = Pose::at([0.5, 0.0, 0.0]).view();
        assert!(!tracker.observe(nudged, ran(dt)).any());
    }

    // At 30 m/s a 0.3 m jump is shorter than a frame's travel, but well off
    // the predicted step: the raised cut lands on it, and the jump is not
    // learned as speed.
    #[test]
    fn a_short_raised_cut_on_a_fast_camera_lands_on_its_jump() {
        let dt = 1.0 / 60.0;
        let mut frames = Vec::new();
        let mut z = 0.0;
        for i in 0..120 {
            z -= 0.5 * (i as f32 / 60.0).min(1.0);
            frames.push((Pose::at([0.0, 0.0, z]), ran(dt)));
        }
        let mut tracker = HistoryResetTracker::default();
        assert_eq!(cut_frames(&play(&mut tracker, &frames)), [] as [usize; 0]);
        tracker.request(CUT);
        carry_on(&mut frames, 1, [0.0; 3]);
        let on_path = frames.last().expect("frames").0;
        assert!(!tracker.observe(on_path.view(), ran(dt)).any());
        carry_on(&mut frames, 1, [0.3, 0.0, 0.0]);
        let jumped = frames.last().expect("frames").0;
        assert_eq!(tracker.observe(jumped.view(), ran(dt)), CUT);
        carry_on(&mut frames, 1, [0.3, 0.0, 0.0]);
        let after = frames.last().expect("frames").0;
        assert!(!tracker.observe(after.view(), ran(dt)).any());
    }

    // Drawn at 240 Hz over a 60 Hz simulation, a camera that stopped from 300
    // m/s forgets its speed over simulated time, not over the frames that
    // happened to run a tick: 1.5 s later, a 15 m respawn on a long frame is
    // a jump. Decayed by render time alone it would keep enough of its speed
    // to pass.
    #[test]
    fn a_stop_decays_by_simulated_time_at_a_high_frame_rate() {
        let dt = 1.0 / 240.0;
        let mut frames = Vec::new();
        let mut z = 0.0;
        for frame in 0..960u32 {
            let ticks = u32::from(frame % 4 == 0);
            let speed = 300.0 * (frame as f32 / 480.0).min(1.0);
            z -= speed * TICK * ticks as f32;
            frames.push((Pose::at([0.0, 0.0, z]), ticks_over(dt, ticks)));
        }
        let mut tracker = HistoryResetTracker::default();
        assert_eq!(cut_frames(&play(&mut tracker, &frames)), [] as [usize; 0]);
        let stopped = Pose::at([0.0, 0.0, z]);
        for frame in 0..360u32 {
            tracker.observe(stopped.view(), ticks_over(dt, u32::from(frame % 4 == 0)));
        }
        let respawned = Pose::at([15.0, 0.0, z]);
        assert_eq!(tracker.observe(respawned.view(), ticks_over(0.4, 5)), CUT);
    }

    // The documented limit of judging by the predicted step: a 360 m/s path
    // at 30 fps that reverses, or turns a 90 degree corner, in one frame reads
    // as a cut.
    #[test]
    fn a_reversal_or_sharp_corner_at_extreme_speed_reads_as_a_cut() {
        let dt = 1.0 / 30.0;
        let mut frames = Vec::new();
        let mut z = 0.0;
        for i in 0..90 {
            z -= 12.0 * (i as f32 / 60.0).min(1.0);
            frames.push((Pose::at([0.0, 0.0, z]), ran(dt)));
        }
        assert_eq!(cuts(&frames), [] as [usize; 0]);
        let mut reversed = frames.clone();
        reversed.push((Pose::at([0.0, 0.0, z + 12.0]), ran(dt)));
        assert_eq!(cuts(&reversed), [frames.len()]);
        let mut cornered = frames.clone();
        cornered.push((Pose::at([12.0, 0.0, z]), ran(dt)));
        assert_eq!(cuts(&cornered), [frames.len()]);
    }

    // A frame timed at zero still moves at the camera's usual rate; it is
    // judged as if it took as long as the last.
    #[test]
    fn a_zero_time_frame_on_a_fast_camera_is_not_a_cut() {
        let dt = 1.0 / 30.0;
        let mut frames = Vec::new();
        let mut z = 0.0;
        for i in 0..90 {
            z -= 12.0 * (i as f32 / 60.0).min(1.0);
            frames.push((Pose::at([0.0, 0.0, z]), ran(dt)));
        }
        frames.push((Pose::at([0.0, 0.0, z - 12.0]), ticks_over(0.0, 0)));
        frames.push((Pose::at([0.0, 0.0, z - 24.0]), ran(dt)));
        assert_eq!(cuts(&frames), [] as [usize; 0]);
    }

    // Mouse look applied unscaled over a near-zero frame does not teach a
    // huge turn rate that would hide a snap turn after it.
    #[test]
    fn a_turn_over_a_near_zero_frame_does_not_hide_a_snap() {
        let dt = 1.0 / 60.0;
        let step = 5f32.to_radians();
        let mut frames: Vec<_> = (0..30)
            .map(|i| (Pose::at([0.0; 3]).yawed(step * i as f32), ran(dt)))
            .collect();
        frames.push((Pose::at([0.0; 3]).yawed(step * 30.0), ticks_over(1e-5, 0)));
        frames.push((Pose::at([0.0; 3]).yawed(step * 31.0), ran(dt)));
        assert_eq!(cuts(&frames), [] as [usize; 0]);
        frames.push((Pose::at([0.0; 3]).yawed(step * 31.0 + FRAC_PI_2), ran(dt)));
        assert_eq!(cuts(&frames), [frames.len() - 1]);
    }

    #[test]
    fn a_reissued_cause_lands_on_the_next_frame_as_it_is() {
        let dt = 1.0 / 60.0;
        let mut tracker = HistoryResetTracker::default();
        let home = Pose::at([0.0; 3]).view();
        tracker.observe(home, ran(dt));
        tracker.reissue(CUT);
        assert_eq!(tracker.observe(home, ran(dt)), CUT);
        assert!(!tracker.observe(home, ran(dt)).any());
    }

    #[test]
    fn a_non_finite_pose_and_the_return_from_it_are_cuts() {
        let dt = 1.0 / 60.0;
        let bad = Pose::at([f32::NAN, 0.0, 0.0]);
        let good = Pose::at([0.0; 3]);
        let frames = [
            (good, ran(dt)),
            (bad, ran(dt)),
            (good, ran(dt)),
            (good, ran(dt)),
        ];
        assert_eq!(cuts(&frames), [1, 2]);
    }

    #[test]
    fn drawing_from_another_camera_is_a_cut() {
        let dt = 1.0 / 60.0;
        let mut tracker = HistoryResetTracker::default();
        let first = Pose::at([0.0; 3]).view();
        tracker.observe(first, ran(dt));
        let other = HistoryView {
            camera: Some(Entity::new(2, NonZeroU32::MIN)),
            ..first
        };
        assert_eq!(tracker.observe(other, ran(dt)), CUT);
        assert!(!tracker.observe(other, ran(dt)).any());
    }

    #[test]
    fn a_scene_switch_is_its_own_cause() {
        let dt = 1.0 / 60.0;
        let mut tracker = HistoryResetTracker::default();
        let base = Pose::at([0.0; 3]).view();
        tracker.observe(base, ran(dt));
        let switched = HistoryView {
            scene: Some(AssetId(2)),
            ..base
        };
        assert_eq!(
            tracker.observe(switched, ran(dt)),
            HistoryResetCauses::SCENE_SWITCH
        );
        assert!(!tracker.observe(switched, ran(dt)).any());
    }

    // A camera that persists across a scene switch and moves with it does not
    // learn the move as its speed: a later small jump still cuts.
    #[test]
    fn a_scene_switch_that_moves_the_camera_teaches_no_speed() {
        let dt = 1.0 / 60.0;
        let mut tracker = HistoryResetTracker::default();
        let base = Pose::at([0.0; 3]).view();
        tracker.observe(base, ran(dt));
        tracker.observe(base, ran(dt));
        // Under the fallback's floor, so only the scene change resets.
        let moved = HistoryView {
            scene: Some(AssetId(2)),
            ..Pose::at([1.5, 0.0, 0.0]).view()
        };
        assert_eq!(
            tracker.observe(moved, ran(dt)),
            HistoryResetCauses::SCENE_SWITCH
        );
        for _ in 0..5 {
            assert!(!tracker.observe(moved, ran(dt)).any());
        }
        let jumped = HistoryView {
            position: [6.0, 0.0, 0.0],
            ..moved
        };
        assert_eq!(tracker.observe(jumped, ran(dt)), CUT);
    }

    #[test]
    fn requested_causes_other_than_a_cut_land_on_the_next_frame_only() {
        let dt = 1.0 / 60.0;
        let mut tracker = HistoryResetTracker::default();
        let base = Pose::at([0.0; 3]).view();
        tracker.observe(base, ran(dt));
        tracker.request(HistoryResetCauses::SETTINGS_CHANGE);
        let jumped = Pose::at([50.0, 0.0, 0.0]).view();
        let causes = tracker.observe(jumped, ran(dt));
        assert!(causes.contains(HistoryResetCauses::SETTINGS_CHANGE));
        assert!(causes.contains(CUT));
        assert!(!tracker.observe(jumped, ran(dt)).any());
    }
}
