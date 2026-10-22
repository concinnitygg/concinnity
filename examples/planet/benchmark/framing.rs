// How the camera films the marble along the flight: shot by shot, where it
// looks from behind the marble and how it frames it.

use super::flight::smoothstep;

/// What a [`Shot`]'s yaw and pitch are measured from.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Aim {
    /// The course ahead and the horizon below the marble, so a shot aimed at
    /// the horizon stays on it from the ground to orbit.
    Course,
    /// The sun, so the shot keeps it in view as it moves.
    Sun,
}

/// One framing the camera reaches at a moment of the run.
pub(super) struct Shot {
    /// Seconds into the run, the warm-up left out.
    pub(super) seconds: f64,
    pub(super) aim: Aim,
    /// Where the camera looks, in degrees left of what it aims by.
    pub(super) yaw: f64,
    /// Where the camera looks, in degrees above what it aims by.
    pub(super) pitch: f64,
    /// How far behind the marble the camera sits, in meters.
    pub(super) distance: f64,
    /// Where the marble sits in the frame, in degrees right of and above the
    /// center of view.
    pub(super) offset: [f64; 2],
}

const fn shot(seconds: f64, yaw: f64, pitch: f64, distance: f64, offset: [f64; 2]) -> Shot {
    Shot {
        seconds,
        aim: Aim::Course,
        yaw,
        pitch,
        distance,
        offset,
    }
}

const fn sun_shot(seconds: f64, yaw: f64, pitch: f64, distance: f64, offset: [f64; 2]) -> Shot {
    Shot {
        aim: Aim::Sun,
        ..shot(seconds, yaw, pitch, distance, offset)
    }
}

// Low behind the marble in the grass, round to its side against the sun and
// back, over the beach, round with the floating crates as they pass on the
// right, out to sea, up past it at the sun, round to face the whole planet
// from orbit, ahead to the terminator, down at the dark disc, low through the
// night, back at the setting moon over the sea, and ahead into the sunrise.
// The camera moves from one shot to the next eased in and out, and holds
// between.
pub(super) const SHOTS: [Shot; 26] = [
    shot(0.0, 0.0, -4.0, 3.2, [0.0, -6.0]),
    shot(3.0, 0.0, -4.0, 3.0, [0.0, -6.0]),
    shot(7.0, -75.0, -5.0, 3.6, [8.0, -5.0]),
    shot(10.0, -75.0, -5.0, 3.6, [8.0, -5.0]),
    shot(15.0, 0.0, -5.0, 3.2, [0.0, -6.0]),
    shot(19.0, 0.0, -6.0, 3.4, [0.0, -6.0]),
    shot(22.5, -35.0, -13.0, 4.6, [8.0, 3.0]),
    shot(24.0, -35.0, -13.0, 4.6, [8.0, 3.0]),
    shot(26.5, 10.0, -14.0, 5.0, [10.0, 4.0]),
    shot(28.8, 10.0, -14.0, 5.0, [10.0, 4.0]),
    shot(29.8, -50.0, -18.0, 4.6, [14.0, 6.0]),
    shot(31.0, -50.0, -18.0, 4.6, [14.0, 6.0]),
    shot(33.5, 0.0, -4.0, 3.2, [0.0, -5.0]),
    shot(41.0, 0.0, -2.0, 3.6, [0.0, -6.0]),
    sun_shot(47.0, 14.0, -16.0, 4.0, [-6.0, -8.0]),
    sun_shot(53.0, 14.0, -16.0, 4.5, [-6.0, -8.0]),
    shot(60.0, -180.0, 3.0, 5.0, [-10.0, 12.0]),
    shot(67.0, -180.0, 3.0, 5.0, [-10.0, 12.0]),
    shot(72.0, 0.0, 2.0, 5.0, [8.0, 12.0]),
    shot(94.0, 0.0, 2.0, 4.5, [8.0, 10.0]),
    shot(102.0, 0.0, -10.0, 3.6, [0.0, -6.0]),
    shot(103.0, -1.0, -9.5, 3.6, [0.0, -6.0]),
    shot(106.0, 165.0, 4.0, 4.0, [10.0, -14.0]),
    shot(108.5, 165.0, 4.0, 4.0, [10.0, -14.0]),
    shot(112.0, -12.0, -4.0, 3.4, [0.0, -6.0]),
    shot(128.0, -12.0, -4.0, 3.2, [0.0, -6.0]),
];

/// The two shots the camera is between `seconds` into the run, and how far
/// from the first to the second it has moved, eased in and out.
pub(super) fn at(seconds: f64) -> (&'static Shot, &'static Shot, f64) {
    let next = SHOTS
        .iter()
        .position(|s| s.seconds > seconds)
        .unwrap_or(SHOTS.len());
    let (from, to) = match next {
        0 => (&SHOTS[0], &SHOTS[0]),
        n if n == SHOTS.len() => (&SHOTS[n - 1], &SHOTS[n - 1]),
        n => (&SHOTS[n - 1], &SHOTS[n]),
    };
    let span = to.seconds - from.seconds;
    let k = if span > 0.0 {
        smoothstep((seconds - from.seconds) / span)
    } else {
        1.0
    };
    (from, to, k)
}
