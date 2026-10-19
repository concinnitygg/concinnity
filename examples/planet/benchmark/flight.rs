// The run's timetable: how high over the ground and how fast the camera goes
// at each moment, and how far around the planet that carries it.

use concinnity_core::math::SaturatingFrom;

/// One stretch of the flight, named as a phase of the report.
pub(super) struct Phase {
    pub(super) name: &'static str,
    pub(super) seconds: f64,
    /// Height over the ground the phase reaches, in meters.
    pub(super) height: f64,
    /// Speed along the course the phase reaches, in meters per second.
    pub(super) speed: f64,
    /// How long, from the phase's start, the camera takes to get from the
    /// previous phase's height and speed to this one's. 0 holds them.
    pub(super) change_seconds: f64,
    /// Whether the camera looks at the sky's sun rather than down the course.
    pub(super) watches_sun: bool,
}

const GROUND_HEIGHT: f64 = 2.5;
const ORBIT_HEIGHT: f64 = 150_000.0;
const LOW_HEIGHT: f64 = 100.0;

// A held settle, a run over the meadow, a climb that speeds up out of the air,
// a pass in orbit, a descent to skim the ground far to the north, and a
// landing back to the meadow's height and pace to watch the sun rise. The
// climb and the descent change height and speed by a constant ratio at their
// fastest, the way a lift-off or a landing feels even.
pub(super) const PHASES: [Phase; 7] = [
    Phase {
        name: "settle",
        seconds: 4.0,
        height: GROUND_HEIGHT,
        speed: 0.0,
        change_seconds: 0.0,
        watches_sun: false,
    },
    Phase {
        name: "ground",
        seconds: 22.0,
        height: GROUND_HEIGHT,
        speed: 30.0,
        change_seconds: 3.0,
        watches_sun: false,
    },
    Phase {
        name: "climb",
        seconds: 20.0,
        height: ORBIT_HEIGHT,
        speed: 6_000.0,
        change_seconds: 20.0,
        watches_sun: true,
    },
    Phase {
        name: "orbit",
        seconds: 14.0,
        height: ORBIT_HEIGHT,
        speed: 6_000.0,
        change_seconds: 0.0,
        watches_sun: true,
    },
    Phase {
        name: "descent",
        seconds: 14.0,
        height: LOW_HEIGHT,
        speed: 600.0,
        change_seconds: 14.0,
        watches_sun: true,
    },
    Phase {
        name: "low",
        seconds: 38.0,
        height: LOW_HEIGHT,
        speed: 600.0,
        change_seconds: 0.0,
        watches_sun: true,
    },
    Phase {
        name: "dawn",
        seconds: 30.0,
        height: GROUND_HEIGHT,
        speed: 30.0,
        change_seconds: 8.0,
        watches_sun: true,
    },
];

/// A moment of the flight.
#[derive(Clone, Copy)]
pub(super) struct Fix {
    pub(super) seconds: f64,
    /// Angle traveled around the planet's center along the course, in radians.
    pub(super) arc: f64,
    /// Height over the ground, in meters.
    pub(super) height: f64,
    /// Index into [`PHASES`] of the phase the moment falls in.
    pub(super) phase: usize,
}

/// The flight's moments every `step` seconds from the start to the end, over
/// a planet of `radius`, phase boundaries among them.
pub(super) fn fixes(radius: f64, step: f64) -> Vec<Fix> {
    let mut fixes = vec![Fix {
        seconds: 0.0,
        arc: 0.0,
        height: PHASES[0].height,
        phase: 0,
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
                phase,
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
