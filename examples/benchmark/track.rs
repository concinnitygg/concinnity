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
// The arc starts on the centre line ahead of the station and ends on it
// behind, bulging to the station's own side in between.
fn arc_point(index: usize, chord: usize) -> [f32; 3] {
    let station = &STATIONS[index];
    let centre = stations::centre(index);
    let angle = core::f32::consts::PI * chord as f32 / CHORDS as f32;
    [
        centre[0] + stations::side(index) * station.radius * angle.sin(),
        HEIGHT,
        centre[2] + station.radius * angle.cos(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn run_seconds() -> f32 {
        travel_legs().iter().map(leg_seconds).sum()
    }

    // The corridor runs one way. Each half circle bulges off the centre line
    // and comes back to it further along, so the camera meets each station once
    // and the segments come out in path order.
    #[test]
    fn every_half_circle_ends_further_down_the_corridor_than_it_began() {
        for index in 0..STATIONS.len() {
            let start = arc_point(index, 0);
            let end = arc_point(index, CHORDS);
            assert!(
                end[2] < start[2],
                "{} doubles back",
                STATIONS[index].segment
            );
            assert!((start[0]).abs() < 1e-3, "an arc starts off the centre line");
            assert!((end[0]).abs() < 1e-3, "an arc ends off the centre line");
        }
    }

    // A chord is a straight run standing in for a piece of the arc. Too few and
    // the camera cuts the corner, and the station is nearer than the radius
    // says it should be.
    #[test]
    fn a_chord_stands_in_for_its_arc_to_within_a_world_unit() {
        let half = core::f32::consts::PI / (2.0 * CHORDS as f32);
        for station in STATIONS {
            let sagitta = station.radius * (1.0 - half.cos());
            assert!(
                sagitta < 1.0,
                "{} cuts {sagitta} off its arc",
                station.segment
            );
        }
    }

    // Every station gets a stretch of path named after it, and the opening hold
    // gets one of its own, or the report has frames it cannot attribute.
    #[test]
    fn the_legs_name_the_settle_and_then_every_station_in_order() {
        let named: Vec<String> = travel_legs()
            .iter()
            .filter(|leg| !leg.segment.is_empty())
            .map(|leg| leg.segment.clone())
            .collect();
        let mut expected = vec![SETTLE_SEGMENT.to_string()];
        expected.extend(STATIONS.iter().map(|s| s.segment.to_string()));
        assert_eq!(named, expected);
    }

    // The opening hold holds still: it is the only leg that names a duration
    // rather than a distance.
    #[test]
    fn the_opening_hold_is_the_only_leg_that_stands_still() {
        let legs = travel_legs();
        assert_eq!(legs[0].segment, SETTLE_SEGMENT);
        assert_eq!(legs[0].distance, 0.0);
        assert_eq!(legs[0].seconds, SETTLE_SECONDS);
        assert!(legs[1..].iter().all(|leg| leg.speed > 0.0));
    }

    // The two tracks are played against one clock, so a turn that ran longer or
    // shorter than its leg would have the camera looking somewhere else on
    // arrival.
    #[test]
    fn each_turn_lasts_exactly_as_long_as_the_leg_it_rides() {
        let travel = travel_legs();
        let turns = turn_legs();
        assert_eq!(travel.len(), turns.len());
        for (leg, turn) in travel.iter().zip(turns.iter()) {
            let seconds = leg_seconds(leg);
            assert!(
                (turn.seconds - seconds).abs() < 1e-4,
                "{} turns for {} and travels for {seconds}",
                leg.segment,
                turn.seconds,
            );
        }
    }

    // The camera looks straight at the station from the side of it, halfway
    // round. That is the view the segment is measuring.
    #[test]
    fn the_camera_faces_the_station_squarely_halfway_round_it() {
        assert_eq!(arc_yaw(1.0, 0.5), 90.0);
        assert_eq!(arc_yaw(-1.0, 0.5), -90.0);
    }

    // Both ends of an arc look down the corridor, which is where the station it
    // is handing over to stands. A heading that ended anywhere else would have
    // the camera whip round between two stations.
    #[test]
    fn an_arc_hands_over_level_with_the_corridor() {
        for side in [1.0, -1.0] {
            assert_eq!(arc_yaw(side, 0.0), 0.0);
            assert!(arc_yaw(side, 1.0).abs() < 1e-4);
        }
        // The opening hold names no heading, which is what makes it a hold, and
        // the camera is authored looking down the corridor already.
        assert_eq!(turn_legs()[0].yaw_deg, None);
    }

    // The station stays inside the frame for as long as the heading tracks it,
    // and the heading only stops tracking once it is past the shoulder.
    #[test]
    fn the_station_is_in_frame_for_most_of_its_own_arc() {
        let mut in_frame = 0;
        let steps = 100;
        for step in 0..steps {
            let at = step as f32 / steps as f32;
            let off = (180.0 * at - arc_yaw(1.0, at)).abs();
            if off < 45.0 {
                in_frame += 1;
            }
        }
        assert!(in_frame >= 70, "the station is in frame for {in_frame}%");
    }

    // A run short enough to sit through and long enough that every segment
    // holds a distribution rather than a handful of frames.
    #[test]
    fn the_run_lasts_between_twenty_and_forty_seconds() {
        let seconds = run_seconds();
        assert!((20.0..=40.0).contains(&seconds), "the run takes {seconds}s");
    }
}
