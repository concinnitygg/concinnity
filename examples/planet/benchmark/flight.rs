// The run's timetable: how high over the ground and how fast the marble goes
// at each moment, and how far around the planet that carries it.

use concinnity_core::math::SaturatingFrom;

/// One stretch of the flight, named as a phase of the report.
pub(super) struct Phase {
    pub(super) name: &'static str,
    pub(super) seconds: f64,
    /// Height of the marble over the ground the phase reaches, in meters.
    pub(super) height: f64,
    /// Speed along the course the phase reaches, in meters per second.
    pub(super) speed: f64,
    /// How long, from the phase's start, the marble takes to get from the
    /// previous phase's height and speed to this one's. 0 holds them.
    pub(super) change_seconds: f64,
}

// Over the meadow the marble clears the tallest blades by a hand; over the
// surf and the sea it clears the waves.
const MEADOW_HEIGHT: f64 = 1.3;
const SEA_HEIGHT: f64 = 2.0;
const ORBIT_HEIGHT: f64 = 150_000.0;
const ORBIT_SPEED: f64 = 6_000.0;
const NIGHT_HEIGHT: f64 = 200.0;
const NIGHT_SPEED: f64 = 2_000.0;
const SKIM_SPEED: f64 = 18.0;

// A held settle on the meadow, a skim through the grass, across the beach and
// the surf and out over the sea, a climb toward the sun that speeds up out of
// the air, a pass in orbit over the day side and across the terminator, a
// descent into the night, a fast skim through it, and a landing on the
// meadow of the next continent as the sun comes up. The climb and the descent
// change height and speed by a constant ratio at their fastest, the way a
// lift-off or a landing feels even.
pub(super) const PHASES: [Phase; 9] = [
    Phase {
        name: "settle",
        seconds: 3.0,
        height: MEADOW_HEIGHT,
        speed: 0.0,
        change_seconds: 0.0,
    },
    Phase {
        name: "meadow",
        seconds: 19.0,
        height: MEADOW_HEIGHT,
        speed: SKIM_SPEED,
        change_seconds: 3.0,
    },
    Phase {
        name: "shore",
        seconds: 10.0,
        height: SEA_HEIGHT,
        speed: SKIM_SPEED,
        change_seconds: 2.0,
    },
    Phase {
        name: "sea",
        seconds: 8.0,
        height: SEA_HEIGHT,
        speed: 30.0,
        change_seconds: 4.0,
    },
    Phase {
        name: "climb",
        seconds: 18.0,
        height: ORBIT_HEIGHT,
        speed: ORBIT_SPEED,
        change_seconds: 18.0,
    },
    Phase {
        name: "orbit",
        seconds: 18.0,
        height: ORBIT_HEIGHT,
        speed: ORBIT_SPEED,
        change_seconds: 0.0,
    },
    Phase {
        name: "descent",
        seconds: 14.0,
        height: NIGHT_HEIGHT,
        speed: NIGHT_SPEED,
        change_seconds: 14.0,
    },
    Phase {
        name: "night",
        seconds: 26.0,
        height: NIGHT_HEIGHT,
        speed: NIGHT_SPEED,
        change_seconds: 0.0,
    },
    Phase {
        name: "sunrise",
        seconds: 16.0,
        height: MEADOW_HEIGHT,
        speed: SKIM_SPEED,
        change_seconds: 10.0,
    },
];

/// A moment of the flight.
#[derive(Clone, Copy)]
pub(super) struct Fix {
    pub(super) seconds: f64,
    /// Angle traveled around the planet's center along the course, in radians.
    pub(super) arc: f64,
    /// Height of the marble over the ground, in meters.
    pub(super) height: f64,
}

/// The flight's moments every `step` seconds from the start to the end, over
/// a planet of `radius`, phase boundaries among them, starting `arc` radians
/// along the course.
pub(super) fn fixes(radius: f64, step: f64, arc: f64) -> Vec<Fix> {
    let mut fixes = vec![Fix {
        seconds: 0.0,
        arc,
        height: PHASES[0].height,
    }];
    let mut start = 0.0;
    for (phase, at) in PHASES.iter().enumerate() {
        let steps = usize::saturating_from((at.seconds / step).round().max(1.0));
        let dt = at.seconds / steps as f64;
        for i in 1..=steps {
            let prev = *fixes.last().expect("starts with a fix");
            let (from, to) = ((i - 1) as f64 * dt, i as f64 * dt);
            let (height, _) = state(phase, to);
            fixes.push(Fix {
                seconds: start + to,
                arc: prev.arc + arc_between(phase, from, to, radius),
                height,
            });
        }
        start += at.seconds;
    }
    fixes
}

// Height and speed `local` seconds into `phase`.
fn state(phase: usize, local: f64) -> (f64, f64) {
    let to = &PHASES[phase];
    let Some(from) = phase.checked_sub(1).map(|p| &PHASES[p]) else {
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
fn arc_between(phase: usize, from: f64, to: f64, radius: f64) -> f64 {
    const PARTS: usize = 8;
    let rate = |t: f64| {
        let (height, speed) = state(phase, t);
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
        (from.ln() + (to.ln() - from.ln()) * k).exp()
    } else {
        from + (to - from) * k
    }
}

pub(super) fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}
