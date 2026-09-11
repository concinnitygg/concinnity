// Scripted camera path schema: the authored legs and the keys they bake into.

use alloc::string::String;
use alloc::vec::Vec;

use crate::ecs::asset_id::AssetId;
use crate::math::vec3;

/// How a [CameraTrack](#cameratrack) leg paces the run between its start and
/// its end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ease {
    /// Constant rate for the whole leg.
    #[default]
    Linear,
    /// Starts at rest and reaches full rate at the end.
    In,
    /// Starts at full rate and comes to rest at the end.
    Out,
    /// Starts and ends at rest.
    InOut,
}

impl Ease {
    /// Progress `t` in `0..=1` remapped by this curve, itself in `0..=1`. The
    /// remap spans the whole leg either way, so an eased leg still covers its
    /// full distance in its full duration; only the rate along the way differs.
    pub fn eval(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Ease::Linear => t,
            Ease::In => t * t,
            Ease::Out => t * (2.0 - t),
            Ease::InOut => t * t * (3.0 - 2.0 * t),
        }
    }
}

/// One straight run on a [CameraTrack](#cameratrack)'s travel track.
///
/// The direction is world space rather than camera relative, so a turn running
/// over the same span does not bend the path: the two tracks stay independent,
/// which is what lets them be read separately.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CameraTravel {
    /// World-space direction of the run. Need not be unit length.
    pub direction: [f32; 3],
    /// How far the run travels along `direction`, in world units.
    pub distance: f32,
    /// Travel rate in world units per second, which fixes the leg's duration
    /// as `distance` divided by it. An eased leg still covers `distance` in
    /// that duration, so this is the average rate rather than the peak.
    pub speed: f32,
    /// Duration in seconds, overriding the one `distance` and `speed` imply. A
    /// leg with only this set holds the camera still for that long.
    pub seconds: f32,
    /// How the run is paced.
    pub ease: Ease,
    /// Names the segment of the path this leg opens, for whatever reads where
    /// the track has reached. An empty label continues whichever segment the
    /// previous leg was in.
    pub segment: String,
}

/// One turn on a [CameraTrack](#cameratrack)'s turn track.
///
/// The angles are absolute headings rather than deltas, so a leg reads as the
/// direction the camera ends up looking. Yaw takes the shorter way round.
/// Either angle may be left out to keep the one the camera already holds,
/// which is what makes a leg with only `seconds` a hold rather than a turn to
/// zero.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CameraTurn {
    /// Heading the turn ends at, in degrees, or unset to hold the current one.
    /// `0` looks toward -Z.
    pub yaw_deg: Option<f32>,
    /// Elevation the turn ends at, in degrees, or unset to hold the current
    /// one. Positive looks up.
    pub pitch_deg: Option<f32>,
    /// Turn rate in degrees per second, taken over whichever of yaw and pitch
    /// has further to go so the two arrive together. The leg's duration
    /// follows from it, and cannot be resolved until the world starts and the
    /// heading the first leg turns away from is known.
    pub degrees_per_second: f32,
    /// Duration in seconds, overriding the one `degrees_per_second` implies. A
    /// leg with only this set holds the heading for that long.
    pub seconds: f32,
    /// How the turn is paced.
    pub ease: Ease,
}

/// Authored fields of a `CameraTrack`; the baked keys are not declared.
///
/// ```rust
/// # use concinnity_core::components::cook::CameraTrack as CameraTrackArgs;
/// # use concinnity_core::components::{CameraTravel, CameraTurn, Ease};
/// CameraTrackArgs {
///     travel: vec![CameraTravel {
///         direction: [0.0, 0.0, -1.0],
///         distance: 20.0,
///         speed: 4.0,
///         ease: Ease::InOut,
///         segment: "approach".into(),
///         ..Default::default()
///     }],
///     turn: vec![CameraTurn {
///         yaw_deg: Some(90.0),
///         degrees_per_second: 30.0,
///         ..Default::default()
///     }],
/// };
/// ```
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CameraTrackArgs {
    /// Where the camera goes, leg by leg.
    pub travel: Vec<CameraTravel>,
    /// Where the camera looks, leg by leg. Played against the same clock as
    /// `travel` and independent of it, so the camera can turn one way while
    /// travelling another.
    pub turn: Vec<CameraTurn>,
}

/// A point the travel track reaches, baked from a
/// [CameraTravel](#cameratravel) leg.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CameraTravelKey {
    /// Offset from the camera's starting position reached at `end_seconds`.
    pub offset: [f32; 3],
    /// Seconds from the track's start at which the camera arrives here.
    pub end_seconds: f32,
    /// How the run up to this point is paced.
    pub ease: Ease,
    /// Index into [`CameraTrack::segments`], or `None` while no leg has yet
    /// opened one.
    pub segment: Option<u32>,
}

/// A heading the turn track reaches, baked from a [CameraTurn](#cameraturn)
/// leg. Its duration is resolved at world start rather than here, because the
/// first leg's turn is measured from the camera's authored heading.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CameraTurnKey {
    /// Heading reached, in degrees, or `None` to hold the current one.
    pub yaw_deg: Option<f32>,
    /// Elevation reached, in degrees, or `None` to hold the current one.
    pub pitch_deg: Option<f32>,
    /// Turn rate the duration follows from when `seconds` is zero.
    pub degrees_per_second: f32,
    /// Authored duration in seconds, or `0` to take it from the rate.
    pub seconds: f32,
    /// How the turn is paced.
    pub ease: Ease,
}

/// Drives the world's [Camera3D](#camera3d) along a scripted path, so a
/// fly-through visits the same poses on every machine that runs it.
///
/// One per world, and its presence takes the camera away from the input
/// controller: a world declaring a track is driven by the track, whatever
/// `controller` the camera carries.
///
/// The track has two independent lists played against one clock. `travel` is
/// where the camera goes and `turn` is where it looks, so a leg of each runs
/// at the same time and the camera can turn toward one thing while travelling
/// toward another. Each list runs from the camera's authored pose; when one
/// runs out the camera holds that list's last value while the other finishes.
///
/// The clock is the fixed simulation step, not the frame delta. A machine that
/// renders half as fast visits the same poses at the same track times and
/// simply samples fewer of them, which is what makes two runs comparable. The
/// track freezes while a world-pausing screen is open, like the rest of the
/// simulation.
///
/// ```rust
/// # use concinnity_core::components::{CameraTrack, CameraTravel, Ease};
/// # use concinnity_core::components::cook::CameraTrack as CameraTrackArgs;
/// CameraTrack::bake(CameraTrackArgs {
///     travel: vec![CameraTravel {
///         direction: [1.0, 0.0, 0.0],
///         distance: 10.0,
///         speed: 5.0,
///         ..Default::default()
///     }],
///     ..Default::default()
/// });
/// ```
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CameraTrack {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// The travel track, as offsets from the camera's starting position.
    pub travel: Vec<CameraTravelKey>,
    /// The turn track, as absolute headings.
    pub turn: Vec<CameraTurnKey>,
    /// The path's segment names, in the order the travel legs open them. A
    /// key's `segment` indexes this list.
    pub segments: Vec<String>,
}

impl CameraTrack {
    /// Translate the authored legs into the runtime keys: resolve each travel
    /// leg's duration and accumulate its offset, gather the segment names, and
    /// carry the turn legs through for the world start to time.
    pub fn bake(args: CameraTrackArgs) -> Self {
        let args = crate::components::validate::camera_track(args);
        let mut travel = Vec::with_capacity(args.travel.len());
        let mut segments: Vec<String> = Vec::new();
        let mut offset = [0.0_f32; 3];
        let mut end_seconds = 0.0_f32;
        let mut segment = None;
        for leg in &args.travel {
            if !leg.segment.is_empty() {
                let at = segments.iter().position(|s| *s == leg.segment);
                segment = Some(match at {
                    Some(i) => i as u32,
                    None => {
                        segments.push(leg.segment.clone());
                        (segments.len() - 1) as u32
                    }
                });
            }
            end_seconds += travel_seconds(leg);
            offset = vec3::add(
                offset,
                vec3::scale(vec3::vec3_normalise(leg.direction), leg.distance),
            );
            travel.push(CameraTravelKey {
                offset,
                end_seconds,
                ease: leg.ease,
                segment,
            });
        }
        let turn = args
            .turn
            .iter()
            .map(|leg| CameraTurnKey {
                yaw_deg: leg.yaw_deg,
                pitch_deg: leg.pitch_deg,
                degrees_per_second: leg.degrees_per_second,
                seconds: leg.seconds,
                ease: leg.ease,
            })
            .collect();
        Self {
            asset_id: AssetId::default(),
            travel,
            turn,
            segments,
        }
    }
}

// How long a travel leg runs: its authored duration, or the one its distance
// and speed imply.
pub(crate) fn travel_seconds(leg: &CameraTravel) -> f32 {
    if leg.seconds > 0.0 {
        leg.seconds
    } else if leg.speed > 0.0 {
        leg.distance / leg.speed
    } else {
        0.0
    }
}

impl crate::ecs::Component for CameraTrack {
    const NAME: &'static str = "CameraTrack";

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }

    fn from_baked(bytes: &[u8]) -> Result<Self, crate::error::CnError> {
        Ok(crate::blob::decode_exact(bytes)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    fn travel(direction: [f32; 3], distance: f32, speed: f32) -> CameraTravel {
        CameraTravel {
            direction,
            distance,
            speed,
            ..Default::default()
        }
    }

    fn baked(travel: Vec<CameraTravel>, turn: Vec<CameraTurn>) -> CameraTrack {
        CameraTrack::bake(CameraTrackArgs { travel, turn })
    }

    #[test]
    fn easing_spans_the_whole_leg_whichever_curve_it_uses() {
        // Every curve has to reach both ends, or a leg would not cover its
        // distance; only the rate along the way is allowed to differ.
        for ease in [Ease::Linear, Ease::In, Ease::Out, Ease::InOut] {
            assert_eq!(ease.eval(0.0), 0.0, "{ease:?}");
            assert_eq!(ease.eval(1.0), 1.0, "{ease:?}");
            assert!((0.0..=1.0).contains(&ease.eval(0.5)), "{ease:?}");
        }
        assert_eq!(Ease::Linear.eval(0.25), 0.25);
        assert!(Ease::In.eval(0.25) < 0.25);
        assert!(Ease::Out.eval(0.25) > 0.25);
        assert_eq!(Ease::InOut.eval(0.5), 0.5);
    }

    #[test]
    fn easing_clamps_progress_from_outside_the_leg() {
        assert_eq!(Ease::InOut.eval(-1.0), 0.0);
        assert_eq!(Ease::InOut.eval(2.0), 1.0);
    }

    #[test]
    fn ease_names_parse_in_snake_case() {
        let e = |s: &str| serde_json::from_str::<Ease>(s).unwrap();
        assert_eq!(e(r#""linear""#), Ease::Linear);
        assert_eq!(e(r#""in""#), Ease::In);
        assert_eq!(e(r#""out""#), Ease::Out);
        assert_eq!(e(r#""in_out""#), Ease::InOut);
        assert_eq!(serde_json::to_string(&Ease::InOut).unwrap(), r#""in_out""#);
    }

    #[test]
    fn an_empty_track_bakes_to_nothing() {
        let t = baked(vec![], vec![]);
        assert!(t.travel.is_empty());
        assert!(t.turn.is_empty());
        assert!(t.segments.is_empty());
    }

    #[test]
    fn a_travel_leg_takes_its_distance_divided_by_its_speed() {
        let t = baked(vec![travel([0.0, 0.0, -1.0], 20.0, 4.0)], vec![]);
        assert_eq!(t.travel[0].end_seconds, 5.0);
        assert_eq!(t.travel[0].offset, [0.0, 0.0, -20.0]);
    }

    #[test]
    fn an_authored_duration_overrides_the_one_speed_implies() {
        let leg = CameraTravel {
            seconds: 3.0,
            ..travel([1.0, 0.0, 0.0], 20.0, 4.0)
        };
        let t = baked(vec![leg], vec![]);
        assert_eq!(t.travel[0].end_seconds, 3.0);
        // The distance is unchanged: the override moves the clock, not the path.
        assert_eq!(t.travel[0].offset, [20.0, 0.0, 0.0]);
    }

    #[test]
    fn a_leg_with_neither_a_speed_nor_a_duration_takes_no_time() {
        let t = baked(vec![travel([1.0, 0.0, 0.0], 20.0, 0.0)], vec![]);
        assert_eq!(t.travel[0].end_seconds, 0.0);
    }

    #[test]
    fn a_hold_leg_carries_the_clock_without_moving_the_camera() {
        let hold = CameraTravel {
            seconds: 2.5,
            ..Default::default()
        };
        let t = baked(vec![travel([1.0, 0.0, 0.0], 4.0, 2.0), hold], vec![]);
        assert_eq!(t.travel[1].end_seconds, 4.5);
        assert_eq!(t.travel[1].offset, t.travel[0].offset);
    }

    #[test]
    fn offsets_accumulate_along_normalized_directions() {
        // The direction is not unit length, so the leg would overshoot if the
        // bake scaled by it rather than by its direction.
        let t = baked(
            vec![
                travel([0.0, 0.0, -2.0], 10.0, 1.0),
                travel([3.0, 0.0, 0.0], 6.0, 1.0),
            ],
            vec![],
        );
        assert_eq!(t.travel[0].offset, [0.0, 0.0, -10.0]);
        assert_eq!(t.travel[1].offset, [6.0, 0.0, -10.0]);
        assert_eq!(t.travel[1].end_seconds, 16.0);
    }

    #[test]
    fn segments_collect_in_first_appearance_order() {
        let named = |name: &str| CameraTravel {
            segment: name.to_string(),
            ..travel([1.0, 0.0, 0.0], 1.0, 1.0)
        };
        let t = baked(
            vec![
                named("shadows"),
                travel([1.0, 0.0, 0.0], 1.0, 1.0),
                named("rays"),
                named("shadows"),
            ],
            vec![],
        );
        assert_eq!(t.segments, ["shadows", "rays"]);
        assert_eq!(t.travel[0].segment, Some(0));
        // An unlabelled leg stays in the segment the one before it opened.
        assert_eq!(t.travel[1].segment, Some(0));
        assert_eq!(t.travel[2].segment, Some(1));
        // A repeated label re-enters its segment rather than declaring another.
        assert_eq!(t.travel[3].segment, Some(0));
    }

    #[test]
    fn legs_before_the_first_label_are_in_no_segment() {
        let t = baked(vec![travel([1.0, 0.0, 0.0], 1.0, 1.0)], vec![]);
        assert_eq!(t.travel[0].segment, None);
        assert!(t.segments.is_empty());
    }

    #[test]
    fn turn_legs_carry_through_untimed() {
        // The first leg's sweep is measured from the camera's own heading, so
        // the bake cannot lay the turn track out on the clock.
        let t = baked(
            vec![],
            vec![CameraTurn {
                yaw_deg: Some(90.0),
                pitch_deg: Some(-10.0),
                degrees_per_second: 45.0,
                ease: Ease::Out,
                ..Default::default()
            }],
        );
        assert_eq!(t.turn[0].yaw_deg, Some(90.0));
        assert_eq!(t.turn[0].pitch_deg, Some(-10.0));
        assert_eq!(t.turn[0].degrees_per_second, 45.0);
        assert_eq!(t.turn[0].seconds, 0.0);
        assert_eq!(t.turn[0].ease, Ease::Out);
    }

    #[test]
    fn a_degenerate_direction_becomes_a_hold_of_the_duration_it_implied() {
        // Zeroing the leg outright would pull every later leg earlier on the
        // clock, silently rewriting a path the author laid out around it.
        let t = baked(vec![travel([0.0, 0.0, 0.0], 12.0, 4.0)], vec![]);
        assert_eq!(t.travel[0].end_seconds, 3.0);
        assert_eq!(t.travel[0].offset, [0.0; 3]);
    }

    #[test]
    fn negative_and_non_finite_quantities_are_clamped_away() {
        let t = baked(
            vec![CameraTravel {
                direction: [1.0, f32::NAN, 0.0],
                distance: -5.0,
                speed: f32::INFINITY,
                seconds: -1.0,
                ..Default::default()
            }],
            vec![CameraTurn {
                yaw_deg: Some(f32::NAN),
                pitch_deg: Some(10.0),
                degrees_per_second: -30.0,
                seconds: f32::NEG_INFINITY,
                ..Default::default()
            }],
        );
        assert_eq!(t.travel[0].offset, [0.0; 3]);
        assert_eq!(t.travel[0].end_seconds, 0.0);
        // A target that is not a real angle is dropped, so the leg holds.
        assert_eq!(t.turn[0].yaw_deg, None);
        assert_eq!(t.turn[0].pitch_deg, Some(10.0));
        assert_eq!(t.turn[0].degrees_per_second, 0.0);
        assert_eq!(t.turn[0].seconds, 0.0);
    }

    #[test]
    fn an_authored_track_parses_and_round_trips_through_postcard() {
        let args: CameraTrackArgs = serde_json::from_str(
            r#"{"travel":[{"direction":[0,0,-1],"distance":8,"speed":2,
                           "ease":"in_out","segment":"approach"}],
                "turn":[{"yaw_deg":45,"degrees_per_second":30}]}"#,
        )
        .unwrap();
        assert_eq!(args.travel[0].ease, Ease::InOut);
        assert_eq!(args.travel[0].segment, "approach");
        assert_eq!(args.turn[0].yaw_deg, Some(45.0));

        let track = CameraTrack::bake(args);
        let bytes = postcard::to_allocvec(&track).unwrap();
        let back: CameraTrack = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back, track);
        assert_eq!(back.segments, ["approach"]);
        assert_eq!(back.travel[0].end_seconds, 4.0);
    }
}
