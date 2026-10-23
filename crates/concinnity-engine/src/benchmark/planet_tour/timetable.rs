// The tour's timetable: how high over the ground and how fast the subject
// goes at each moment, and how far around the planet that carries it.

use alloc::vec::Vec;

use concinnity_core::math::SaturatingFrom;

/// One stretch of the tour, named as a phase of the report.
pub(super) struct Phase {
    pub(super) name: &'static str,
    pub(super) seconds: f64,
    /// Height of the subject over the ground the phase reaches, in meters.
    pub(super) height: f64,
    /// Speed along the course the phase reaches, in meters per second.
    pub(super) speed: f64,
    /// How long, from the phase's start, the subject takes to get from the
    /// previous phase's height and speed to this one's. 0 holds them.
    pub(super) change_seconds: f64,
}

// Skimming the meadow the subject clears the tallest blades by a hand; over
// the surf and the sea it clears the waves. A rolling subject is set down a
// little over the ground and floats at the water's surface.
const MEADOW_HEIGHT: f64 = 1.3;
const SEA_HEIGHT: f64 = 2.0;
const ROLL_HEIGHT: f64 = 1.0;
const FLOAT_HEIGHT: f64 = 0.0;
const ORBIT_HEIGHT: f64 = 150_000.0;
const ORBIT_SPEED: f64 = 6_000.0;
const NIGHT_HEIGHT: f64 = 200.0;
const NIGHT_SPEED: f64 = 2_000.0;
const SKIM_SPEED: f64 = 18.0;
const ROLL_SPEED: f64 = 15.0;
const PADDLE_SPEED: f64 = 2.0;
/// The pace the subject is towed up to through the water before it lifts off,
/// in meters per second.
pub(super) const WAKE_SPEED: f64 = 8.0;

const fn phase(
    name: &'static str,
    seconds: f64,
    height: f64,
    speed: f64,
    change_seconds: f64,
) -> Phase {
    Phase {
        name,
        seconds,
        height,
        speed,
        change_seconds,
    }
}

// A held settle, then a skim through the grass, across the beach and the surf.
const SKIM: [Phase; 3] = [
    phase("settle", 3.0, MEADOW_HEIGHT, 0.0, 0.0),
    phase("meadow", 19.0, MEADOW_HEIGHT, SKIM_SPEED, 3.0),
    phase("shore", 10.0, SEA_HEIGHT, SKIM_SPEED, 2.0),
];

// A held settle, then the subject rolls itself through the grass and down the
// beach into the surf, floats and paddles out, and is towed up to speed
// through the water, plowing a wake. The tour picks it up as the sea phase
// opens.
const ROLL: [Phase; 4] = [
    phase("settle", 3.0, ROLL_HEIGHT, 0.0, 0.0),
    phase("roll", 27.0, ROLL_HEIGHT, ROLL_SPEED, 3.0),
    phase("float", 4.0, FLOAT_HEIGHT, PADDLE_SPEED, 1.0),
    phase("wake", 8.0, FLOAT_HEIGHT, WAKE_SPEED, 2.0),
];

// Out over the sea, a climb toward the sun that speeds up out of the air, a
// pass in orbit over the day side and across the terminator, a descent into
// the night, a fast skim through it, and a landing on the meadow of the next
// continent as the sun comes up. The climb and the descent change height and
// speed by a constant ratio at their fastest, the way a lift-off or a landing
// feels even.
const FLIGHT: [Phase; 6] = [
    phase("sea", 8.0, SEA_HEIGHT, 30.0, 4.0),
    phase("climb", 18.0, ORBIT_HEIGHT, ORBIT_SPEED, 18.0),
    phase("orbit", 28.0, ORBIT_HEIGHT, ORBIT_SPEED, 0.0),
    phase("descent", 14.0, NIGHT_HEIGHT, NIGHT_SPEED, 14.0),
    phase("night", 18.0, NIGHT_HEIGHT, NIGHT_SPEED, 0.0),
    phase("sunrise", 16.0, MEADOW_HEIGHT, SKIM_SPEED, 10.0),
];

/// The tour's phases in order: the ground stretch, rolled or skimmed, then
/// the flight. A rolling subject floats and plows a wake only on a planet
/// with a `sea`.
pub(super) fn phases(rolls: bool, sea: bool) -> Vec<&'static Phase> {
    let ground: &[Phase] = if rolls { &ROLL } else { &SKIM };
    ground
        .iter()
        .filter(|phase| sea || !matches!(phase.name, "float" | "wake"))
        .chain(FLIGHT.iter())
        .collect()
}

/// The least height over dry ground the path keeps for a rolling subject,
/// so the stretches it floats in never bury it where it is aground.
pub(super) fn ground_clearance(rolls: bool) -> f64 {
    if rolls { ROLL_HEIGHT } else { 0.0 }
}

/// Seconds from the start of the tour to the start of the phase `name`, or
/// `None` for a phase the tour has not.
pub(super) fn phase_start(phases: &[&Phase], name: &str) -> Option<f64> {
    let at = phases.iter().position(|p| p.name == name)?;
    Some(phases[..at].iter().map(|p| p.seconds).sum())
}

/// A moment of the flight.
#[derive(Clone, Copy)]
pub(super) struct Fix {
    pub(super) seconds: f64,
    /// Angle traveled around the planet's center along the course, in radians.
    pub(super) arc: f64,
    /// Height of the marble over the ground, in meters.
    pub(super) height: f64,
}

/// The tour's moments every `step` seconds from the start to the end of
/// `phases`, over a planet of `radius`, phase boundaries among them, starting
/// `arc` radians along the course.
pub(super) fn fixes(phases: &[&Phase], radius: f64, step: f64, arc: f64) -> Vec<Fix> {
    let mut prev = Fix {
        seconds: 0.0,
        arc,
        height: phases.first().map_or(0.0, |p| p.height),
    };
    let mut fixes = alloc::vec![prev];
    let mut start = 0.0;
    for (phase, at) in phases.iter().enumerate() {
        let steps = usize::saturating_from(libm::round(at.seconds / step).max(1.0));
        let dt = at.seconds / steps as f64;
        for i in 1..=steps {
            let (from, to) = ((i - 1) as f64 * dt, i as f64 * dt);
            let (height, _) = state(phases, phase, to);
            prev = Fix {
                seconds: start + to,
                arc: prev.arc + arc_between(phases, phase, from, to, radius),
                height,
            };
            fixes.push(prev);
        }
        start += at.seconds;
    }
    fixes
}

// Height and speed `local` seconds into `phase`.
fn state(phases: &[&Phase], phase: usize, local: f64) -> (f64, f64) {
    let to = phases[phase];
    let Some(from) = phase.checked_sub(1).map(|p| phases[p]) else {
        return (to.height, to.speed);
    };
    let k = if to.change_seconds > 0.0 {
        smoothstep(local / to.change_seconds)
    } else {
        1.0
    };
    (
        blend(from.height, to.height, k),
        blend(from.speed, to.speed, k),
    )
}

// The angle around the center the course covers `from` to `to` seconds into
// `phase`, by Simpson's rule: the speed is along the course at the camera's
// height, so its rate of turn about the center falls as it climbs.
fn arc_between(phases: &[&Phase], phase: usize, from: f64, to: f64, radius: f64) -> f64 {
    const PARTS: usize = 8;
    let rate = |t: f64| {
        let (height, speed) = state(phases, phase, t);
        speed / (radius + height)
    };
    let h = (to - from) / PARTS as f64;
    let inner: f64 = (1..PARTS)
        .map(|i| rate(from + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 })
        .sum();
    (rate(from) + inner + rate(to)) * h / 3.0
}

// `from` to `to` by `k`: by a constant ratio where both are positive, so a
// climb from meters to kilometers spends as long on each tenfold, else
// evenly.
fn blend(from: f64, to: f64, k: f64) -> f64 {
    if k >= 1.0 || from == to {
        to
    } else if from > 0.0 && to > 0.0 {
        libm::exp(libm::log(from) + (libm::log(to) - libm::log(from)) * k)
    } else {
        from + (to - from) * k
    }
}

pub(super) fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}
