// The fallback cut test: one frame's change of view judged against the step
// the camera's own recent motion predicts, so a camera moving fast but
// steadily never reads as a cut while a sudden jump, in any direction, does.
//
// The limit of judging by the predicted step: a camera that reverses or turns
// a sharp corner at speed in one frame strays from its prediction by up to
// twice a frame's travel, against an allowance of one frame's travel plus the
// floor, so it reads as a cut. A 360 m/s path at 30 fps (12 m a frame) cuts on
// a reversal (24 m off) and on a 90 degree corner (17 m off). The engine's own
// controllers smooth their velocity, so only an extreme scripted path meets
// this, and one extra reset costs less than a missed cut.

use super::tracker::HistoryView;
use crate::ecs::SimTiming;
use crate::math::{acos, exp2, ln, sqrt};
use crate::transform::Mat4;

/// How far a step may stray from its prediction, in meters (world units),
/// before it is a cut whatever the camera's speed.
pub(super) const CUT_DISTANCE_METERS: f32 = 2.0;
/// How far a step may turn away from its prediction (45 degrees), roll
/// included, before it is a cut whatever the camera's turn rate.
pub(super) const CUT_TURN_RADIANS: f32 = core::f32::consts::FRAC_PI_4;
/// How far a step may zoom away from its prediction before it is a cut: the
/// natural log of a 1.25 field-of-view ratio.
pub(super) const CUT_ZOOM: f32 = 0.223_143_55;
/// How many times its recent rate a moving camera may stray from its predicted
/// step. A fixed-step simulation sampled at an uneven frame rate strays by one
/// step's motion, half what its peak rate covers in a frame.
const RATE_TOLERANCE: f32 = 1.0;
/// How quickly a recent rate is forgotten once the camera slows or stops.
const RATE_HALF_LIFE_SECONDS: f32 = 0.5;
/// The longest a frame may count for: the most simulated time one frame can
/// run, so a hitch allows no more motion than the simulation could produce.
pub(super) const MAX_STEP_SECONDS: f32 = SimTiming::MAX_TICKS_PER_FRAME as f32 * SimTiming::TICK_DT;
/// The most a prediction stretches the last step to cover a longer frame.
const MAX_STEP_STRETCH: f32 = 5.0;
/// The shortest a step may count for. A step timed shorter (mouse look applied
/// unscaled over a near-zero frame) counts for as long as the last one did, so
/// neither its prediction nor the rates it teaches blow up.
const MIN_STEP_SECONDS: f32 = 0.001;
/// How far a step must stray from its prediction to count as moving off it,
/// for a pending cut waiting on the camera: a quarter of a frame's travel at
/// the recent rate, plus these floors. A fixed-step simulation sampled
/// unevenly strays by a whole step's travel on the frames that run an extra
/// tick, so a cut raised on such a camera can land a frame before its move.
const DEPARTURE_RATE_FRACTION: f32 = 0.25;
const DEPARTURE_METERS: f32 = 0.001;
const DEPARTURE_RADIANS: f32 = 0.001;

type Mat3 = [[f32; 3]; 3];

const IDENTITY3: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// How the view changed between two frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ViewStep {
    // World units moved.
    translation: [f32; 3],
    // The rotation taking the previous view orientation to this one.
    rotation: Mat3,
    // Natural log of the field-of-view ratio.
    zoom: f32,
    // Seconds the step took, at most `MAX_STEP_SECONDS`.
    seconds: f32,
}

impl Default for ViewStep {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
            rotation: IDENTITY3,
            zoom: 0.0,
            seconds: 0.0,
        }
    }
}

impl ViewStep {
    /// The change from `prev` to `cur` over `dt` seconds, or `None` when the
    /// view is exactly the same.
    pub(super) fn between(prev: &HistoryView, cur: &HistoryView, dt: f32) -> Option<Self> {
        if prev.position == cur.position
            && prev.view == cur.view
            && prev.fov_y_radians == cur.fov_y_radians
        {
            return None;
        }
        let zoom = if prev.fov_y_radians > 0.0 && cur.fov_y_radians > 0.0 {
            ln(cur.fov_y_radians / prev.fov_y_radians)
        } else {
            0.0
        };
        Some(Self {
            translation: sub(cur.position, prev.position),
            rotation: mul(
                &orientation(&cur.view),
                &transpose(&orientation(&prev.view)),
            ),
            zoom,
            seconds: frame_seconds(dt),
        })
    }
}

/// `dt` as the heuristic counts it: never negative, never longer than the
/// simulation could run in one frame.
pub(super) fn frame_seconds(dt: f32) -> f32 {
    if dt.is_nan() {
        0.0
    } else {
        dt.clamp(0.0, MAX_STEP_SECONDS)
    }
}

/// How far a step strayed from its prediction.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Deviation {
    distance: f32,
    angle: f32,
    zoom: f32,
}

/// The camera's last continuous step, which predicts the next, and its recent
/// rates of travel, turn and zoom: each the highest it has lately moved at,
/// decaying once it slows or stops.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct MotionReference {
    last: ViewStep,
    speed: f32,
    turn_rate: f32,
    zoom_rate: f32,
}

impl MotionReference {
    /// A camera that just took `step`, as if it were its ordinary motion.
    fn moving_like(step: &ViewStep) -> Self {
        let mut reference = Self::default();
        reference.absorb(step);
        reference
    }

    /// A camera that set off with `jump`: it predicts the same step again,
    /// with half the slack the jump's own rate would allow, so only a camera
    /// that carries on much as it began passes.
    pub(super) fn launched(jump: &ViewStep) -> Self {
        let mut reference = Self::moving_like(jump);
        reference.speed *= 0.5;
        reference.turn_rate *= 0.5;
        reference.zoom_rate *= 0.5;
        reference
    }

    /// Whether `step` strays from the predicted one further than the cut
    /// floors plus what the recent rates allow.
    pub(super) fn is_cut(&self, step: &ViewStep) -> bool {
        self.strays(
            step,
            RATE_TOLERANCE,
            CUT_DISTANCE_METERS,
            CUT_TURN_RADIANS,
            CUT_ZOOM,
        )
    }

    /// Whether `step` leaves the predicted path by more than a fraction of a
    /// frame's ordinary travel: where a cut raised before the camera moved is
    /// waiting to land.
    pub(super) fn departs(&self, step: &ViewStep) -> bool {
        self.strays(
            step,
            DEPARTURE_RATE_FRACTION,
            DEPARTURE_METERS,
            DEPARTURE_RADIANS,
            DEPARTURE_RADIANS,
        )
    }

    fn strays(
        &self,
        step: &ViewStep,
        tolerance: f32,
        distance: f32,
        angle: f32,
        zoom: f32,
    ) -> bool {
        let d = self.deviation(step);
        let dt = self.seconds_of(step);
        d.distance > distance + tolerance * self.speed * dt
            || d.angle > angle + tolerance * self.turn_rate * dt
            || d.zoom > zoom + tolerance * self.zoom_rate * dt
    }

    /// Fold a continuous step into the prediction and the recent rates.
    pub(super) fn absorb(&mut self, step: &ViewStep) {
        let dt = self.seconds_of(step);
        let decay = decay(dt);
        let distance = length(step.translation);
        let angle = rotation_angle(&step.rotation, &IDENTITY3);
        self.speed = (distance / dt).max(self.speed * decay);
        self.turn_rate = (angle / dt).max(self.turn_rate * decay);
        self.zoom_rate = (step.zoom.abs() / dt).max(self.zoom_rate * decay);
        self.last = ViewStep {
            seconds: dt,
            ..*step
        };
    }

    // How long `step` counts for: its own time, or, timed shorter than
    // `MIN_STEP_SECONDS`, as long as the last step took.
    fn seconds_of(&self, step: &ViewStep) -> f32 {
        if step.seconds >= MIN_STEP_SECONDS {
            step.seconds
        } else {
            self.last.seconds.max(MIN_STEP_SECONDS)
        }
    }

    /// The camera held still for `dt` seconds of running simulation: it
    /// predicts stillness, and its recent rates decay.
    pub(super) fn rest(&mut self, dt: f32) {
        let dt = frame_seconds(dt);
        let decay = decay(dt);
        self.speed *= decay;
        self.turn_rate *= decay;
        self.zoom_rate *= decay;
        self.last = ViewStep::default();
    }

    // The step predicted by the last one, stretched to `step`'s duration for
    // the translation and the zoom. The rotation repeats as it was.
    fn deviation(&self, step: &ViewStep) -> Deviation {
        let stretch = if self.last.seconds > 0.0 {
            (self.seconds_of(step) / self.last.seconds).min(MAX_STEP_STRETCH)
        } else {
            0.0
        };
        let predicted = scale(self.last.translation, stretch);
        Deviation {
            distance: length(sub(step.translation, predicted)),
            angle: rotation_angle(&step.rotation, &self.last.rotation),
            zoom: (step.zoom - self.last.zoom * stretch).abs(),
        }
    }
}

fn decay(dt: f32) -> f32 {
    exp2(-dt / RATE_HALF_LIFE_SECONDS)
}

// The rotation block of a view matrix (stored column-major) as rows.
fn orientation(view: &Mat4) -> Mat3 {
    let mut m = [[0.0; 3]; 3];
    for (row, out) in m.iter_mut().enumerate() {
        for (col, cell) in out.iter_mut().enumerate() {
            *cell = view[col][row];
        }
    }
    m
}

fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut m = [[0.0; 3]; 3];
    for (row, out) in m.iter_mut().enumerate() {
        for (col, cell) in out.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[row][k] * b[k][col]).sum();
        }
    }
    m
}

fn transpose(a: &Mat3) -> Mat3 {
    let mut m = [[0.0; 3]; 3];
    for (row, out) in m.iter_mut().enumerate() {
        for (col, cell) in out.iter_mut().enumerate() {
            *cell = a[col][row];
        }
    }
    m
}

// The angle of the rotation taking `b` to `a`, from the trace of `a * b^T`,
// read as the inner product of the two.
fn rotation_angle(a: &Mat3, b: &Mat3) -> f32 {
    let mut trace = 0.0;
    for row in 0..3 {
        for col in 0..3 {
            trace += a[row][col] * b[row][col];
        }
    }
    acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn length(a: [f32; 3]) -> f32 {
    sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::camera::view_matrix;

    const DT: f32 = 1.0 / 60.0;

    fn view(position: [f32; 3], yaw: f32, fov_y_radians: f32) -> HistoryView {
        HistoryView {
            camera: None,
            position,
            view: view_matrix(position, yaw, 0.0),
            fov_y_radians,
            scene: None,
        }
    }

    fn step(from: HistoryView, to: HistoryView) -> ViewStep {
        ViewStep::between(&from, &to, DT).expect("moved")
    }

    #[test]
    fn an_identical_view_is_no_step() {
        let a = view([1.0, 2.0, 3.0], 0.4, 1.0);
        assert_eq!(ViewStep::between(&a, &a, DT), None);
    }

    #[test]
    fn a_step_measures_translation_rotation_and_zoom() {
        let a = view([0.0; 3], 0.0, 1.0);
        let b = view([3.0, 4.0, 0.0], 0.5, 2.0);
        let s = step(a, b);
        assert_eq!(s.translation, [3.0, 4.0, 0.0]);
        assert!((rotation_angle(&s.rotation, &IDENTITY3) - 0.5).abs() < 1e-3);
        assert!((s.zoom - core::f32::consts::LN_2).abs() < 1e-5);
    }

    #[test]
    fn a_long_frame_counts_only_what_the_simulation_could_run() {
        assert_eq!(frame_seconds(2.0), MAX_STEP_SECONDS);
        assert_eq!(frame_seconds(-1.0), 0.0);
        assert_eq!(frame_seconds(f32::NAN), 0.0);
        assert_eq!(frame_seconds(DT), DT);
    }

    // The floor is in meters: the near plane a camera draws with does not
    // move it.
    #[test]
    fn a_camera_at_rest_is_judged_against_the_floors() {
        let rest = MotionReference::default();
        let a = view([0.0; 3], 0.0, 1.0);
        let short = view([CUT_DISTANCE_METERS * 0.9, 0.0, 0.0], 0.0, 1.0);
        let long = view([CUT_DISTANCE_METERS * 1.1, 0.0, 0.0], 0.0, 1.0);
        assert!(!rest.is_cut(&step(a, short)));
        assert!(rest.is_cut(&step(a, long)));
        let turned = view([0.0; 3], CUT_TURN_RADIANS * 1.1, 1.0);
        assert!(rest.is_cut(&step(a, turned)));
    }

    // Moving forward at 10 units a frame, a sideways jump half again as long
    // as a frame's travel is far off the predicted step though no longer than
    // two frames of it.
    #[test]
    fn a_jump_off_the_predicted_path_is_a_cut_even_at_the_usual_distance() {
        let a = view([0.0; 3], 0.0, 1.0);
        let b = view([0.0, 0.0, -10.0], 0.0, 1.0);
        let reference = MotionReference::moving_like(&step(a, b));
        let ahead = view([0.0, 0.0, -20.0], 0.0, 1.0);
        let aside = view([15.0, 0.0, -10.0], 0.0, 1.0);
        assert!(!reference.is_cut(&step(b, ahead)));
        assert!(reference.is_cut(&step(b, aside)));
    }

    #[test]
    fn departing_needs_only_a_small_move_off_the_prediction() {
        let rest = MotionReference::default();
        let a = view([0.0; 3], 0.0, 1.0);
        assert!(rest.departs(&step(a, view([0.01, 0.0, 0.0], 0.0, 1.0))));
        let b = view([0.0, 0.0, -10.0], 0.0, 1.0);
        let moving = MotionReference::moving_like(&step(a, b));
        assert!(!moving.departs(&step(b, view([0.0, 0.0, -20.0], 0.0, 1.0))));
    }

    #[test]
    fn rest_decays_the_rates_and_predicts_stillness() {
        let a = view([0.0; 3], 0.0, 1.0);
        let b = view([0.0, 0.0, -10.0], 0.0, 1.0);
        let mut reference = MotionReference::moving_like(&step(a, b));
        let speed = reference.speed;
        // Ten frames of a tenth of a half-life each, inside the cap.
        for _ in 0..10 {
            reference.rest(RATE_HALF_LIFE_SECONDS / 10.0);
        }
        assert!((reference.speed - speed * 0.5).abs() < speed * 1e-3);
        assert_eq!(reference.last, ViewStep::default());
    }

    // A step timed near zero counts as long as the last one, so a turn
    // applied over it teaches an ordinary rate rather than a huge one.
    #[test]
    fn a_step_without_time_counts_as_long_as_the_last() {
        let a = view([0.0; 3], 0.0, 1.0);
        let b = view([0.0; 3], 0.1, 1.0);
        let mut reference = MotionReference::moving_like(&step(a, b));
        let rate = reference.turn_rate;
        let c = view([0.0; 3], 0.2, 1.0);
        reference.absorb(&ViewStep::between(&b, &c, 1e-6).expect("turned"));
        assert!((reference.turn_rate - rate).abs() < rate * 1e-2);
    }
}
