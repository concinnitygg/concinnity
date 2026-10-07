//! The path the camera flies, derived from the stations it circles.
//!
//! Two lists played against one clock: where the camera is, and where it is
//! looking. The camera reaches each station along a half circle of that
//! station's own radius, so every frame of a segment sees the station from the
//! same distance, and it keeps moving throughout, so temporal antialiasing and
//! reprojection stay in the state they hold during play rather than converging
//! on a still frame.
//!
//! A camera track leg is a straight run, so each half circle is a fan of
//! chords. Only the first names the segment; the rest carry it on.

use concinnity::components::Ease;
use concinnity::components::{CameraTravel, CameraTurn};

use crate::stations::{self, STATIONS};

/// How high above the floor the camera flies.
pub(crate) const HEIGHT: f32 = 7.0;
/// How fast the camera travels, in world units per second. Constant, so a wide
/// station is circled at the same pace as a narrow one and takes longer.
pub(crate) const SPEED: f32 = 20.0;
/// The opening hold, before the camera sets off. Longer than the report's
/// discard, so the measured frames all belong to a station.
pub(crate) const SETTLE_SECONDS: f32 = 3.0;

/// The name the opening hold reports under.
pub(crate) const SETTLE_SEGMENT: &str = "settle";

// How many straight runs one half circle is cut into. Enough that a chord
// departs from the arc it stands in for by well under a world unit.
const CHORDS: usize = 12;

// How far round a half circle the camera keeps the station in the middle of
// the frame before it turns forward onto the next one. Past this point the
// station is behind the camera's shoulder and would be out of frame whatever
// the heading did.
const FACING_UNTIL: f32 = 0.70;

/// The path, leg by leg. The first leg of each half circle carries the segment
/// name its frames are reported under, so a regression names the station it is
/// at.
pub(crate) fn travel_legs() -> Vec<CameraTravel> {
    let mut legs = vec![CameraTravel {
        seconds: SETTLE_SECONDS,
        segment: SETTLE_SEGMENT.to_string(),
        ..Default::default()
    }];
    for (index, station) in STATIONS.iter().enumerate() {
        for chord in 0..CHORDS {
            let from = arc_point(index, chord);
            let to = arc_point(index, chord + 1);
            let step = [to[0] - from[0], 0.0, to[2] - from[2]];
            let distance = (step[0] * step[0] + step[2] * step[2]).sqrt();
            legs.push(CameraTravel {
                direction: step,
                distance,
                speed: SPEED,
                ease: Ease::Linear,
                segment: match chord {
                    0 => station.segment.to_string(),
                    _ => String::new(),
                },
                ..Default::default()
            });
        }
    }
    legs
}

/// The head turn, one per travel leg and timed to it.
pub(crate) fn turn_legs() -> Vec<CameraTurn> {
    let travel = travel_legs();
    let mut runs = travel[1..].iter();
    let mut legs = vec![CameraTurn {
        seconds: SETTLE_SECONDS,
        ..Default::default()
    }];
    for (index, station) in STATIONS.iter().enumerate() {
        for chord in 0..CHORDS {
            let run = runs.next().expect("a run for every chord");
            let at = (chord + 1) as f32 / CHORDS as f32;
            legs.push(CameraTurn {
                yaw_deg: Some(arc_yaw(stations::side(index), at)),
                pitch_deg: Some(station.pitch_deg),
                seconds: leg_seconds(run),
                ease: Ease::Linear,
                ..Default::default()
            });
        }
    }
    legs
}

/// When the leg named `segment` begins and ends, in seconds of simulated time
/// from the world start.
///
/// The track advances on the fixed simulation clock, so these are the same
/// seconds on every machine, and a world asking where the camera is can ask the
/// clock instead.
pub(crate) fn segment_window(segment: &str) -> (f32, f32) {
    let mut opened = None;
    let mut at = 0.0;
    for leg in travel_legs() {
        if leg.segment == segment {
            opened = Some(at);
        } else if !leg.segment.is_empty() && opened.is_some() {
            break;
        }
        at += leg_seconds(&leg);
    }
    match opened {
        Some(from) => (from, at),
        None => panic!("no leg is named {segment}"),
    }
}

// Where the camera stands `chord` runs into station `index`'s half circle.
//
// The arc starts on the center line ahead of the station and ends on it
// behind, bulging to the station's own side in between.
fn arc_point(index: usize, chord: usize) -> [f32; 3] {
    let station = &STATIONS[index];
    let center = stations::center(index);
    let angle = core::f32::consts::PI * chord as f32 / CHORDS as f32;
    [
        center[0] + stations::side(index) * station.radius * angle.sin(),
        HEIGHT,
        center[2] + station.radius * angle.cos(),
    ]
}

// The heading `at` the way round a half circle bulging toward `side`.
//
// Level with the corridor at both ends, where the next station stands straight
// ahead, and pinned on the station being circled in between. Coming back level
// is what lets one arc hand over to the next without the camera whipping round.
fn arc_yaw(side: f32, at: f32) -> f32 {
    let facing = 180.0 * at;
    if at <= FACING_UNTIL {
        return side * facing;
    }
    let back = (1.0 - at) / (1.0 - FACING_UNTIL);
    side * 180.0 * FACING_UNTIL * back * back * (3.0 - 2.0 * back)
}

// A leg that names a speed is timed by the distance it covers; one that names
// neither holds for its own duration.
fn leg_seconds(leg: &CameraTravel) -> f32 {
    if leg.speed > 0.0 {
        leg.distance / leg.speed
    } else {
        leg.seconds
    }
}
