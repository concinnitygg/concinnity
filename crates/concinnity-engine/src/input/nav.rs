// src/input/nav.rs
//
// Pure navigation-pulse shaping: the held d-pad directions and the left-stick
// deflection become one-frame directional pulses with hold auto-repeat, for
// menu focus movement. No OS or clock access -- the caller feeds held state
// and a frame dt, so the fold is driven synthetically in tests.

use crate::components::NavDirection;

// Seconds a direction must stay held before it starts repeating.
const REPEAT_DELAY: f32 = 0.4;
// Seconds between repeat pulses while a direction stays held.
const REPEAT_INTERVAL: f32 = 0.12;
// Stick deflection that begins a navigation press, well past any sane
// deadzone.
const STICK_PRESS: f32 = 0.5;
// Deflection the stick must fall back under to release a press; the gap keeps
// a wavering hold from re-pulsing.
const STICK_RELEASE: f32 = 0.35;

// Held d-pad directions, in `[up, down, left, right]` order.
pub(crate) type DpadHeld = [bool; 4];

// Auto-repeat state for the navigation pulse: which direction is being
// commanded (d-pad or stick) and how long until the next repeat fires.
#[derive(Debug, Default)]
pub(crate) struct NavRepeat {
    commanded: Option<NavDirection>,
    until_repeat: f32,
    // The stick's latched direction, held until the deflection falls back
    // under `STICK_RELEASE` (hysteresis, so drift near the press threshold
    // never machine-guns pulses).
    stick_latched: Option<NavDirection>,
}

impl NavRepeat {
    // Advance one frame and return the pulse to publish, if any. `dpad` is the
    // held d-pad state, `stick` the left-stick vector in the movement
    // convention (+y forward/up), `dt` the frame time in seconds.
    pub(crate) fn step(
        &mut self,
        dpad: DpadHeld,
        stick: [f32; 2],
        dt: f32,
    ) -> Option<NavDirection> {
        self.stick_latched = self.stick_direction(stick);
        let commanded = dpad_direction(dpad).or(self.stick_latched);

        if commanded != self.commanded {
            self.commanded = commanded;
            self.until_repeat = REPEAT_DELAY;
            return commanded;
        }
        let held = commanded?;
        self.until_repeat -= dt;
        if self.until_repeat <= 0.0 {
            self.until_repeat += REPEAT_INTERVAL;
            // A frame spanning several intervals still emits one pulse; the
            // timer never owes a burst.
            if self.until_repeat < 0.0 {
                self.until_repeat = REPEAT_INTERVAL;
            }
            return Some(held);
        }
        None
    }

    // The stick's commanded direction under hysteresis: a latched direction
    // holds until its component falls under the release threshold; otherwise
    // the dominant component past the press threshold latches.
    fn stick_direction(&self, stick: [f32; 2]) -> Option<NavDirection> {
        if let Some(dir) = self.stick_latched
            && component(stick, dir) > STICK_RELEASE
        {
            return Some(dir);
        }
        let (x, y) = (stick[0], stick[1]);
        let dir = if y.abs() >= x.abs() {
            if y > 0.0 {
                NavDirection::Up
            } else {
                NavDirection::Down
            }
        } else if x > 0.0 {
            NavDirection::Right
        } else {
            NavDirection::Left
        };
        (component(stick, dir) >= STICK_PRESS).then_some(dir)
    }
}

// The stick deflection along a direction (positive when deflected that way).
fn component(stick: [f32; 2], dir: NavDirection) -> f32 {
    match dir {
        NavDirection::Up => stick[1],
        NavDirection::Down => -stick[1],
        NavDirection::Left => -stick[0],
        NavDirection::Right => stick[0],
    }
}

// The held d-pad direction, or `None`. With several held (rare), the first in
// `[up, down, left, right]` order wins.
fn dpad_direction(dpad: DpadHeld) -> Option<NavDirection> {
    const ORDER: [NavDirection; 4] = [
        NavDirection::Up,
        NavDirection::Down,
        NavDirection::Left,
        NavDirection::Right,
    ];
    dpad.iter()
        .zip(ORDER)
        .find(|(held, _)| **held)
        .map(|(_, dir)| dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDLE: DpadHeld = [false; 4];
    const DOWN: DpadHeld = [false, true, false, false];

    #[test]
    fn press_pulses_once_and_repeats_after_the_delay() {
        let mut nav = NavRepeat::default();
        assert_eq!(nav.step(DOWN, [0.0, 0.0], 0.016), Some(NavDirection::Down));
        // Held short of the delay: silent.
        assert_eq!(nav.step(DOWN, [0.0, 0.0], REPEAT_DELAY - 0.05), None);
        // Crossing the delay fires a repeat, then the interval paces further
        // repeats.
        assert_eq!(nav.step(DOWN, [0.0, 0.0], 0.1), Some(NavDirection::Down));
        assert_eq!(nav.step(DOWN, [0.0, 0.0], REPEAT_INTERVAL / 2.0), None);
        assert_eq!(
            nav.step(DOWN, [0.0, 0.0], REPEAT_INTERVAL),
            Some(NavDirection::Down)
        );
    }

    #[test]
    fn release_clears_and_a_new_press_pulses_immediately() {
        let mut nav = NavRepeat::default();
        nav.step(DOWN, [0.0, 0.0], 0.016);
        assert_eq!(nav.step(IDLE, [0.0, 0.0], 0.016), None);
        assert_eq!(nav.step(DOWN, [0.0, 0.0], 0.016), Some(NavDirection::Down));
    }

    #[test]
    fn direction_change_pulses_without_waiting_for_the_delay() {
        let mut nav = NavRepeat::default();
        nav.step(DOWN, [0.0, 0.0], 0.016);
        let up: DpadHeld = [true, false, false, false];
        assert_eq!(nav.step(up, [0.0, 0.0], 0.016), Some(NavDirection::Up));
    }

    #[test]
    fn stick_press_pulses_and_holds_through_hysteresis() {
        let mut nav = NavRepeat::default();
        // Under the press threshold: nothing.
        assert_eq!(nav.step(IDLE, [0.0, -0.4], 0.016), None);
        // Past it: one pulse (stick -y is Down in the movement convention).
        assert_eq!(nav.step(IDLE, [0.0, -0.6], 0.016), Some(NavDirection::Down));
        // Sagging into the hysteresis band keeps the hold (no re-pulse, no
        // release), so the repeat timer keeps running.
        assert_eq!(nav.step(IDLE, [0.0, -0.4], 0.016), None);
        assert_eq!(
            nav.step(IDLE, [0.0, -0.4], REPEAT_DELAY),
            Some(NavDirection::Down)
        );
        // Falling under the release threshold clears; the next deliberate
        // deflection pulses again.
        assert_eq!(nav.step(IDLE, [0.0, -0.2], 0.016), None);
        assert_eq!(nav.step(IDLE, [0.0, -0.9], 0.016), Some(NavDirection::Down));
    }

    #[test]
    fn stick_picks_the_dominant_axis() {
        let mut nav = NavRepeat::default();
        assert_eq!(nav.step(IDLE, [0.8, 0.3], 0.016), Some(NavDirection::Right));
        let mut nav = NavRepeat::default();
        assert_eq!(nav.step(IDLE, [0.3, 0.8], 0.016), Some(NavDirection::Up));
    }

    #[test]
    fn dpad_wins_over_the_stick() {
        let mut nav = NavRepeat::default();
        assert_eq!(nav.step(DOWN, [0.0, 0.9], 0.016), Some(NavDirection::Down));
    }

    #[test]
    fn a_long_frame_emits_a_single_pulse() {
        let mut nav = NavRepeat::default();
        nav.step(DOWN, [0.0, 0.0], 0.016);
        // One huge frame: exactly one repeat, and the timer does not go into
        // debt to fire a burst on the frames after.
        assert_eq!(nav.step(DOWN, [0.0, 0.0], 5.0), Some(NavDirection::Down));
        assert_eq!(nav.step(DOWN, [0.0, 0.0], 0.016), None);
    }
}
