// How the camera films the subject along the tour: shot by shot, where it
// looks from behind the subject and how it frames it, each shot keyed to a
// moment of a phase rather than of the run, so it follows the tour however
// its phases are timed.

use alloc::vec::Vec;

use super::timetable::{Phase, phase_start, smoothstep};

/// One framing the camera reaches at a moment of a phase.
pub(super) struct Shot {
    /// The phase the shot is timed in.
    pub(super) phase: &'static str,
    /// Seconds from the phase's start, or before its end when negative.
    pub(super) at: f64,
    /// Where the camera looks, in degrees left of the course ahead.
    pub(super) yaw: f64,
    /// Where the camera looks, in degrees above the horizon below the
    /// subject, so a shot aimed at the horizon stays on it from the ground to
    /// orbit.
    pub(super) pitch: f64,
    /// How far behind the subject the camera sits, in meters.
    pub(super) distance: f64,
    /// Where the subject sits in the frame, in degrees right of and above the
    /// center of view.
    pub(super) offset: [f64; 2],
}

const fn shot(
    phase: &'static str,
    at: f64,
    yaw: f64,
    pitch: f64,
    distance: f64,
    offset: [f64; 2],
) -> Shot {
    Shot {
        phase,
        at,
        yaw,
        pitch,
        distance,
        offset,
    }
}

// Low behind the subject in the grass, round to its side against the sun and
// back, then up over the beach as it reaches the water.
const SKIM: [Shot; 8] = [
    shot("settle", 0.0, 0.0, -4.0, 3.2, [0.0, -6.0]),
    shot("meadow", 0.0, 0.0, -4.0, 3.0, [0.0, -6.0]),
    shot("meadow", 4.0, -75.0, -5.0, 3.6, [8.0, -5.0]),
    shot("meadow", 7.0, -75.0, -5.0, 3.6, [8.0, -5.0]),
    shot("meadow", 12.0, 0.0, -5.0, 3.2, [0.0, -6.0]),
    shot("meadow", -3.0, 0.0, -6.0, 3.4, [0.0, -6.0]),
    shot("shore", 0.5, -35.0, -13.0, 4.6, [8.0, 3.0]),
    shot("shore", -2.0, -35.0, -13.0, 4.6, [8.0, 3.0]),
];

// The same through the grass while the subject rolls itself, up over the
// beach as it comes down to the water, looking down on it as it floats, high
// behind it as it is towed up to speed so its wake spreads below, and low at
// its side for the spray off its bow.
const ROLL: [Shot; 13] = [
    shot("settle", 0.0, 0.0, -4.0, 3.2, [0.0, -6.0]),
    shot("roll", 0.0, 0.0, -4.0, 3.0, [0.0, -6.0]),
    shot("roll", 4.0, -75.0, -5.0, 3.6, [8.0, -5.0]),
    shot("roll", 7.0, -75.0, -5.0, 3.6, [8.0, -5.0]),
    shot("roll", 12.0, 0.0, -5.0, 3.2, [0.0, -6.0]),
    shot("roll", -8.0, 0.0, -6.0, 3.4, [0.0, -6.0]),
    shot("roll", -5.0, -35.0, -13.0, 4.6, [8.0, 3.0]),
    shot("roll", -3.0, -35.0, -13.0, 4.6, [8.0, 3.0]),
    shot("float", 1.0, 10.0, -16.0, 5.0, [10.0, 4.0]),
    shot("float", -1.0, 10.0, -16.0, 5.0, [10.0, 4.0]),
    shot("wake", 1.5, 8.0, -22.0, 9.0, [0.0, 8.0]),
    shot("wake", 4.0, 8.0, -22.0, 9.0, [0.0, 8.0]),
    shot("wake", -1.5, -55.0, -14.0, 6.5, [6.0, -2.0]),
];

// Out to sea behind the subject, then looking ever further down on it as it
// climbs, the ground and then the whole planet below it, held over the planet
// through the orbit, ahead to the terminator on the way down, low through the
// night, back at the moon over the water, and ahead into the sunrise. The
// camera moves from one shot to the next eased in and out, and holds between.
const FLIGHT: [Shot; 13] = [
    shot("sea", 1.5, 0.0, -4.0, 3.2, [0.0, -5.0]),
    shot("climb", 1.0, 0.0, -2.0, 3.6, [0.0, -6.0]),
    shot("climb", 7.0, 0.0, -20.0, 4.0, [0.0, -8.0]),
    shot("climb", 13.0, 0.0, -14.0, 4.5, [6.0, -14.0]),
    shot("orbit", 2.0, 0.0, -9.5, 4.5, [8.0, -16.0]),
    shot("orbit", 14.0, 0.0, -9.5, 4.5, [-6.0, -16.0]),
    shot("descent", 8.0, 0.0, 2.0, 4.5, [8.0, 10.0]),
    shot("night", 2.0, 0.0, -10.0, 3.6, [0.0, -6.0]),
    shot("night", 3.0, -1.0, -9.5, 3.6, [0.0, -6.0]),
    shot("night", 6.0, 165.0, 4.0, 4.0, [10.0, -14.0]),
    shot("night", 8.5, 165.0, 4.0, 4.0, [10.0, -14.0]),
    shot("night", 12.0, -12.0, -4.0, 3.4, [0.0, -6.0]),
    shot("sunrise", 10.0, -12.0, -4.0, 3.2, [0.0, -6.0]),
];

/// The shots of a tour laid out on its clock, in time order.
pub(super) struct Framing {
    shots: Vec<(f64, &'static Shot)>,
}

impl Framing {
    /// The shots for `phases`, rolled or skimmed over the ground, each at its
    /// moment of its phase.
    pub(super) fn new(phases: &[&Phase], rolls: bool) -> Self {
        let ground: &'static [Shot] = if rolls { &ROLL } else { &SKIM };
        let mut shots: Vec<(f64, &'static Shot)> = ground
            .iter()
            .chain(FLIGHT.iter())
            .filter_map(|shot| {
                let start = phase_start(phases, shot.phase)?;
                let length = phases.iter().find(|p| p.name == shot.phase)?.seconds;
                let at = if shot.at < 0.0 {
                    length + shot.at
                } else {
                    shot.at
                };
                Some((start + at.clamp(0.0, length), shot))
            })
            .collect();
        shots.sort_by(|a, b| a.0.total_cmp(&b.0));
        Self { shots }
    }

    /// The two shots the camera is between `seconds` into the tour, and how
    /// far from the first to the second it has moved, eased in and out.
    pub(super) fn at(&self, seconds: f64) -> (&'static Shot, &'static Shot, f64) {
        let shots = &self.shots;
        let held = (0.0, &FLIGHT[0]);
        let next = shots
            .iter()
            .position(|s| s.0 > seconds)
            .unwrap_or(shots.len());
        let from = next
            .checked_sub(1)
            .and_then(|i| shots.get(i))
            .or_else(|| shots.first())
            .copied()
            .unwrap_or(held);
        let to = shots.get(next).copied().unwrap_or(from);
        let span = to.0 - from.0;
        let k = if span > 0.0 {
            smoothstep((seconds - from.0) / span)
        } else {
            1.0
        };
        (from.1, to.1, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::planet_tour::timetable::phases;

    // Every shot lands inside its own phase, in time order, a shot timed
    // before its phase's end counted back from that end.
    #[test]
    fn shots_land_in_their_phases() {
        for (rolls, sea) in [(false, true), (true, true), (true, false)] {
            let phases = phases(rolls, sea);
            let framing = Framing::new(&phases, rolls);
            assert!(framing.shots.windows(2).all(|w| w[0].0 <= w[1].0));
            for (at, shot) in &framing.shots {
                let start = phase_start(&phases, shot.phase).unwrap();
                let length = phases
                    .iter()
                    .find(|p| p.name == shot.phase)
                    .unwrap()
                    .seconds;
                assert!(
                    (start..=start + length).contains(at),
                    "{} at {at}",
                    shot.phase
                );
            }
        }
        let roll = Framing::new(&phases(true, true), true);
        let (from, to, k) = roll.at(0.0);
        assert_eq!((from.phase, to.phase, k), ("settle", "roll", 0.0));
        let (last, _, _) = roll.at(1.0e6);
        assert_eq!(last.phase, "sunrise");
    }
}
