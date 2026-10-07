//! The things the camera circles, and where each one stands.
//!
//! One ordered table drives both halves of the world: a station declares the
//! assets it is made of and how far off it wants to be looked at from, and the
//! camera path is derived from the same rows, so a station cannot be built
//! without a stretch of path named after it.

mod glass;
mod ground;
mod instances;
mod lights;
mod particles;
mod physics;
mod shadows;
mod swarm;
mod water;

use concinnity::cook::WorldBuilder;

/// One station: how far the camera holds off it, how it is looked at, and what
/// it is made of.
pub(crate) struct Station {
    /// The name the report attributes this station's frames to.
    pub(crate) segment: &'static str,
    /// The radius the camera circles it at. Wide enough to hold the whole
    /// station in frame, since that is the view the segment measures.
    pub(crate) radius: f32,
    /// The elevation the camera holds while it circles.
    pub(crate) pitch_deg: f32,
    /// Declares the station's assets into the world.
    pub(crate) declare: fn(&mut WorldBuilder, [f32; 3]),
}

/// The corridor, station by station, in the order the camera meets them.
///
/// Every station stands on the center line. The camera reaches each one along
/// a half circle of that station's own radius, and the half circles alternate
/// sides, so consecutive ones meet where their shared tangent is and the path
/// runs on without a straight stretch between them.
pub(crate) const STATIONS: &[Station] = &[
    Station {
        segment: "instances",
        radius: 24.0,
        pitch_deg: -7.0,
        declare: instances::declare,
    },
    Station {
        segment: "shadows",
        radius: 34.0,
        pitch_deg: -5.0,
        declare: shadows::declare,
    },
    Station {
        segment: "lights",
        radius: 26.0,
        pitch_deg: -5.0,
        declare: lights::declare,
    },
    Station {
        segment: "particles",
        radius: 20.0,
        pitch_deg: 2.0,
        declare: particles::declare,
    },
    Station {
        segment: "glass",
        radius: 22.0,
        pitch_deg: -5.0,
        declare: glass::declare,
    },
    Station {
        segment: physics::SEGMENT,
        radius: physics::RADIUS,
        pitch_deg: -11.0,
        declare: physics::declare,
    },
    Station {
        segment: swarm::SEGMENT,
        radius: 22.0,
        pitch_deg: -8.0,
        declare: swarm::declare,
    },
    Station {
        segment: "water",
        radius: 30.0,
        pitch_deg: -8.0,
        declare: water::declare,
    },
];

/// Where the camera's path begins, on the center line ahead of the first
/// station.
pub(crate) const START_Z: f32 = 0.0;

/// Where station `index` stands.
///
/// Each one sits its own radius plus its neighbor's behind the last, which is
/// the spacing that makes their half circles meet.
pub(crate) fn center(index: usize) -> [f32; 3] {
    let mut z = START_Z;
    for (i, station) in STATIONS.iter().enumerate().take(index + 1) {
        z -= station.radius;
        if i < index {
            z -= station.radius;
        }
    }
    [0.0, 0.0, z]
}

/// Which side of the center line station `index` is circled on: `1` bulges
/// toward +X and `-1` toward -X. They alternate, which is what makes one half
/// circle leave along the tangent the next arrives on.
pub(crate) fn side(index: usize) -> f32 {
    if index.is_multiple_of(2) { 1.0 } else { -1.0 }
}

/// How far along -Z the path reaches, past the last station.
pub(crate) fn corridor_end() -> f32 {
    let last = STATIONS.len() - 1;
    center(last)[2] - STATIONS[last].radius
}

/// Declare the corridor everything stands on, then every station on it.
pub(crate) fn declare_all(world: &mut WorldBuilder) {
    ground::declare(world);
    for (index, station) in STATIONS.iter().enumerate() {
        (station.declare)(world, center(index));
    }
}

/// Index `i` of `count` laid out about zero at `spacing`. An even count
/// straddles zero and an odd one is centered on it.
pub(crate) fn spread(i: usize, count: usize, spacing: f32) -> f32 {
    (i as f32 - (count as f32 - 1.0) * 0.5) * spacing
}
