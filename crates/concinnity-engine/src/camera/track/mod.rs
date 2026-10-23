//! The scripted camera path: the keys a world's
//! [`CameraTrack`](crate::components::CameraTrack) baked into, the system that
//! plays them, and where it has reached.
//!
//! The track's two lists are played against one clock and never read each
//! other, so traveling and turning compose without either bending the other.
//! That clock is the fixed simulation step, which is what makes a run
//! repeatable: a slower machine samples the same path at the same track times,
//! just at fewer of them.

mod status;
mod system;
mod timeline;

use crate::components::{Prop, Vocabulary};
use crate::ecs::Ref;
use alloc::vec::Vec;
use concinnity_core::math::vec3;
pub(crate) use status::CameraTrackStatus;
pub(crate) use system::CameraTrackSystem;

/// How a [`CameraTrack`](#cameratrack) leg paces the run between its start and
/// its end.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize, Vocabulary,
)]
#[serde(rename_all = "snake_case")]
pub enum Ease {
    /// Constant rate for the whole leg.
    #[default]
    #[vocab("linear")]
    Linear,
    /// Starts at rest and reaches full rate at the end.
    #[vocab("in")]
    In,
    /// Starts at full rate and comes to rest at the end.
    #[vocab("out")]
    Out,
    /// Starts and ends at rest.
    #[vocab("in_out")]
    InOut,
}

impl Ease {
    /// Progress `t` in `0..=1` remapped by this curve, itself in `0..=1`. The
    /// remap spans the whole leg either way, so an eased leg still covers its
    /// full distance in its full duration; only the rate along the way differs.
    pub fn eval(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::In => t * t,
            Self::Out => t * (2.0 - t),
            Self::InOut => t * t * (3.0 - 2.0 * t),
        }
    }
}

/// One straight run on a [`CameraTrack`](#cameratrack)'s travel track.
///
/// The direction is world space rather than camera relative, so a turn running
/// over the same span does not bend the path: the two tracks stay independent,
/// which is what lets them be read separately.
#[derive(
    Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields,
)]
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
    /// Jump to where the leg ends instead of running to it. The leg takes no
    /// time, whatever `speed` or `seconds` say, and the first frame drawn from
    /// the new pose starts afresh rather than blending with the frames before
    /// the jump, as at a shot change. A leg that takes no time anyway is a cut
    /// as soon as it moves the camera.
    pub cut: bool,
}

/// One turn on a [`CameraTrack`](#cameratrack)'s turn track.
///
/// The angles are absolute headings rather than deltas, so a leg reads as the
/// direction the camera ends up looking. Yaw takes the shorter way round.
/// Either angle may be left out to keep the one the camera already holds,
/// which is what makes a leg with only `seconds` a hold rather than a turn to
/// zero.
#[derive(
    Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields,
)]
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
    /// Snap to the leg's heading instead of turning to it. The leg takes no
    /// time, whatever `degrees_per_second` or `seconds` say, and the first
    /// frame drawn from the new heading starts afresh rather than blending with
    /// the frames before the snap, as at a shot change. A leg that takes no
    /// time anyway is a cut as soon as it turns the camera.
    pub cut: bool,
}

/// One change of framing on a [`CameraTrack`](#cameratrack)'s boom track: how
/// far the camera sits from the track's `subject` and where the subject sits
/// in the frame.
///
/// Either value may be left out to keep the one the boom already holds. Until
/// the first leg that sets a value the boom holds that value, so a track can
/// open on the framing its first leg names.
#[derive(
    Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields,
)]
#[serde(default)]
pub struct CameraBoom {
    /// Distance from the subject the camera ends at, in world units, or unset
    /// to hold the current one.
    pub distance: Option<f32>,
    /// Where the subject sits in the frame the leg ends at, in degrees right
    /// of and above the center of view, or unset to hold the current place.
    /// `[0, -10]` keeps the subject a little below the center.
    pub offset_deg: Option<[f32; 2]>,
    /// Duration in seconds. A leg with only this set holds the framing for
    /// that long.
    pub seconds: f32,
    /// How the change is paced.
    pub ease: Ease,
    /// Snap to the leg's framing instead of moving to it, as at a shot change.
    /// A leg that takes no time anyway is a cut as soon as it changes the
    /// framing.
    pub cut: bool,
}

/// Drives the world's [`Camera3D`](#camera3d) along a scripted path, so a
/// fly-through visits the same poses on every machine that runs it.
///
/// One per world, and its presence takes the camera away from the input
/// controller: a world declaring a track is driven by the track, whatever
/// `controller` the camera carries. Nor does the physics collide the camera
/// while a track drives it, so the path is flown exactly as written.
///
/// The track has two independent lists played against one clock. `travel` is
/// where the camera goes and `turn` is where it looks, so a leg of each runs
/// at the same time and the camera can turn toward one thing while traveling
/// toward another. Each list runs from the camera's authored pose; when one
/// runs out the camera holds that list's last value while the other finishes.
///
/// A leg marked `cut` jumps rather than runs, which is how a track changes
/// shot: the frame drawn from the new pose does not blend with the old one.
///
/// A track with a `subject` films a [Prop](#prop) from behind a boom, as a
/// third-person camera does. `travel` then carries the subject rather than the
/// camera, `turn` is still where the camera looks, and a third list, `boom`,
/// sets how far back the camera sits and where in the frame the subject
/// appears. The camera is placed from the subject on every frame, so the
/// subject holds its place in the frame exactly while either moves, and a
/// sweep of `turn` circles the camera round it. Without boom legs the camera
/// is placed at its authored distance from the subject, on the line of view
/// through it, from the first frame.
///
/// The subject must be a Prop with no `parent`, since travel writes its
/// position in world space. The track owns its position: a subject with a
/// [`PropBody`](#propbody) moves kinematically along the track while the
/// track has it, still pushing whatever it runs into, and no
/// [`RollingDrive`](#rollingdrive) steers it. A subject that is missing,
/// parented or later removed leaves the track driving the camera itself from
/// wherever it stands, with a warning.
///
/// The clock is the fixed simulation step, not the frame delta. A machine that
/// renders half as fast visits the same poses at the same track times and
/// simply samples fewer of them, which is what makes two runs comparable. The
/// track freezes while a world-pausing screen is open, like the rest of the
/// simulation.
///
/// ```rust
/// # use concinnity_engine::components::cook::CameraTrack as CameraTrackArgs;
/// # use concinnity_engine::components::{CameraTravel, CameraTurn, Ease};
/// CameraTrackArgs {
///     travel: vec![CameraTravel {
///         direction: [0.0, 0.0, -1.0],
///         distance: 20.0,
///         speed: 4.0,
///         ease: Ease::InOut,
///         ..Default::default()
///     }],
///     turn: vec![CameraTurn {
///         yaw_deg: Some(90.0),
///         degrees_per_second: 30.0,
///         ..Default::default()
///     }],
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields,
)]
#[serde(default)]
pub struct CameraTrackArgs {
    /// Where the camera goes, leg by leg.
    pub travel: Vec<CameraTravel>,
    /// Where the camera looks, leg by leg. Played against the same clock as
    /// `travel` and independent of it, so the camera can turn one way while
    /// traveling another.
    pub turn: Vec<CameraTurn>,
    /// The [Prop](#prop) the track carries and films, or unset for a track
    /// that carries the camera itself.
    pub subject: Option<Ref<Prop>>,
    /// How the camera frames the `subject`, leg by leg, on the same clock as
    /// the other two lists. Used only with a `subject`.
    pub boom: Vec<CameraBoom>,
}

/// A point the travel track reaches, baked from a
/// [`CameraTravel`](#cameratravel) leg.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
)]
pub struct CameraTravelKey {
    /// How far the leg moves the camera: from where the previous leg left it
    /// to where it stands at `end_seconds`. Each leg carries its own
    /// displacement rather than one from the track's start, so a long track
    /// keeps every leg as precise as a short one.
    pub delta: [f32; 3],
    /// Seconds from the track's start at which the camera arrives here.
    pub end_seconds: f32,
    /// How the run up to this point is paced.
    pub ease: Ease,
    /// The camera jumps here rather than running: the key takes no time and
    /// the frame that reaches it starts afresh.
    pub cut: bool,
}

/// A heading the turn track reaches, baked from a [`CameraTurn`](#cameraturn)
/// leg. Its duration is resolved at world start rather than here, because the
/// first leg's turn is measured from the camera's authored heading.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
)]
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
    /// The camera snaps to this heading rather than turning: the key takes no
    /// time and the frame that reaches it starts afresh.
    pub cut: bool,
}

/// A framing the boom track reaches, baked from a [`CameraBoom`](#cameraboom)
/// leg. Unset values are resolved at world start, against the framing the
/// camera starts in.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
)]
pub struct CameraBoomKey {
    /// Distance from the subject reached, or `None` to hold the current one.
    pub distance: Option<f32>,
    /// Place of the subject in the frame reached, in degrees right and up,
    /// or `None` to hold the current one.
    pub offset_deg: Option<[f32; 2]>,
    /// Seconds from the track's start at which the framing is reached.
    pub end_seconds: f32,
    /// How the change up to this key is paced.
    pub ease: Ease,
    /// The camera snaps to this framing rather than moving to it.
    pub cut: bool,
}

/// The runtime `CameraTrack`: the keys
/// [`cook::CameraTrack`](crate::components::cook::CameraTrack)'s legs bake into.
///
/// ```rust
/// # use concinnity_engine::components::{CameraTrack, CameraTravel, Ease};
/// # use concinnity_engine::components::cook::CameraTrack as CameraTrackArgs;
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
#[derive(
    Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields,
)]
pub struct CameraTrack {
    /// The travel track, leg by leg, each a displacement from where the one
    /// before it ended.
    pub travel: Vec<CameraTravelKey>,
    /// The turn track, as absolute headings.
    pub turn: Vec<CameraTurnKey>,
    /// The prop the track carries and films, if any.
    pub subject: Option<Ref<Prop>>,
    /// The boom track, as the framings it reaches.
    pub boom: Vec<CameraBoomKey>,
}

impl CameraTrack {
    /// Translate the authored legs into the runtime keys: resolve each travel
    /// leg's duration and displacement, and carry the turn legs through for the
    /// world start to time.
    pub fn bake(args: CameraTrackArgs) -> Self {
        let args = crate::components::validate::camera_track(args);
        let mut travel = Vec::with_capacity(args.travel.len());
        let mut end_seconds = 0.0_f32;
        for leg in &args.travel {
            let seconds = travel_seconds(leg);
            end_seconds += seconds;
            travel.push(CameraTravelKey {
                delta: vec3::scale(vec3::normalize(leg.direction), leg.distance),
                end_seconds,
                ease: leg.ease,
                cut: leg.cut || (seconds == 0.0 && leg.distance > 0.0),
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
                cut: leg.cut,
            })
            .collect();
        let mut end_seconds = 0.0_f32;
        let boom = args
            .boom
            .iter()
            .map(|leg| {
                let seconds = if leg.cut { 0.0 } else { leg.seconds };
                end_seconds += seconds;
                CameraBoomKey {
                    distance: leg.distance,
                    offset_deg: leg.offset_deg,
                    end_seconds,
                    ease: leg.ease,
                    cut: leg.cut,
                }
            })
            .collect();
        Self {
            travel,
            turn,
            subject: args.subject,
            boom,
        }
    }
}

// How long a travel leg runs: its authored duration, or the one its distance
// and speed imply.
pub(crate) fn travel_seconds(leg: &CameraTravel) -> f32 {
    if leg.cut {
        0.0
    } else if leg.seconds > 0.0 {
        leg.seconds
    } else if leg.speed > 0.0 {
        leg.distance / leg.speed
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        CameraTrack::bake(CameraTrackArgs {
            travel,
            turn,
            ..Default::default()
        })
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
    }

    #[test]
    fn a_travel_leg_takes_its_distance_divided_by_its_speed() {
        let t = baked(vec![travel([0.0, 0.0, -1.0], 20.0, 4.0)], vec![]);
        assert_eq!(t.travel[0].end_seconds, 5.0);
        assert_eq!(t.travel[0].delta, [0.0, 0.0, -20.0]);
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
        assert_eq!(t.travel[0].delta, [20.0, 0.0, 0.0]);
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
        assert_eq!(t.travel[1].delta, [0.0; 3]);
    }

    #[test]
    fn each_leg_moves_its_own_distance_along_its_normalized_direction() {
        // The direction is not unit length, so the leg would overshoot if the
        // bake scaled by it rather than by its direction.
        let t = baked(
            vec![
                travel([0.0, 0.0, -2.0], 10.0, 1.0),
                travel([3.0, 0.0, 0.0], 6.0, 1.0),
            ],
            vec![],
        );
        assert_eq!(t.travel[0].delta, [0.0, 0.0, -10.0]);
        assert_eq!(t.travel[1].delta, [6.0, 0.0, 0.0]);
        assert_eq!(t.travel[1].end_seconds, 16.0);
    }

    #[test]
    fn a_cut_leg_takes_no_time_and_keeps_its_distance() {
        let cut = CameraTravel {
            cut: true,
            seconds: 4.0,
            ..travel([0.0, 0.0, 1.0], 12.0, 2.0)
        };
        let t = baked(vec![travel([1.0, 0.0, 0.0], 2.0, 1.0), cut], vec![]);
        assert!(!t.travel[0].cut);
        assert!(t.travel[1].cut);
        assert_eq!(t.travel[1].end_seconds, t.travel[0].end_seconds);
        assert_eq!(t.travel[1].delta, [0.0, 0.0, 12.0]);
    }

    // A leg that moves the camera in no time is a jump whether or not the
    // author called it one; a hold that takes no time moves nothing.
    #[test]
    fn an_instant_move_is_a_cut_and_an_instant_hold_is_not() {
        let t = baked(
            vec![travel([1.0, 0.0, 0.0], 5.0, 0.0), CameraTravel::default()],
            vec![],
        );
        assert!(t.travel[0].cut);
        assert!(!t.travel[1].cut);
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
        assert!(!t.turn[0].cut);
    }

    #[test]
    fn boom_legs_are_laid_out_on_the_clock_and_keep_their_subject() {
        let t = CameraTrack::bake(CameraTrackArgs {
            subject: Some(Ref::new(crate::ecs::asset_id::AssetId(3))),
            boom: vec![
                CameraBoom {
                    distance: Some(4.0),
                    seconds: 2.0,
                    ..Default::default()
                },
                CameraBoom {
                    offset_deg: Some([5.0, -8.0]),
                    seconds: 3.0,
                    ease: Ease::InOut,
                    ..Default::default()
                },
                CameraBoom {
                    distance: Some(9.0),
                    seconds: 6.0,
                    cut: true,
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        assert_eq!(t.subject, Some(Ref::new(crate::ecs::asset_id::AssetId(3))));
        let ends: Vec<f32> = t.boom.iter().map(|k| k.end_seconds).collect();
        assert_eq!(ends, [2.0, 5.0, 5.0], "a cut takes no time");
        assert_eq!(t.boom[0].distance, Some(4.0));
        assert_eq!(t.boom[0].offset_deg, None);
        assert_eq!(t.boom[1].offset_deg, Some([5.0, -8.0]));
        assert_eq!(t.boom[1].ease, Ease::InOut);
        assert!(t.boom[2].cut);
    }

    #[test]
    fn a_degenerate_direction_becomes_a_hold_of_the_duration_it_implied() {
        // Zeroing the leg outright would pull every later leg earlier on the
        // clock, silently rewriting a path the author laid out around it.
        let t = baked(vec![travel([0.0, 0.0, 0.0], 12.0, 4.0)], vec![]);
        assert_eq!(t.travel[0].end_seconds, 3.0);
        assert_eq!(t.travel[0].delta, [0.0; 3]);
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
        let boom = CameraTrack::bake(CameraTrackArgs {
            boom: vec![CameraBoom {
                distance: Some(-2.0),
                offset_deg: Some([f32::NAN, 4.0]),
                seconds: f32::NAN,
                ..Default::default()
            }],
            ..Default::default()
        });
        assert_eq!(boom.boom[0].distance, Some(0.0));
        assert_eq!(
            boom.boom[0].offset_deg, None,
            "an offset that is not a real angle holds"
        );
        assert_eq!(boom.boom[0].end_seconds, 0.0);
        assert_eq!(t.travel[0].delta, [0.0; 3]);
        assert_eq!(t.travel[0].end_seconds, 0.0);
        // A target that is not a real angle is dropped, so the leg holds.
        assert_eq!(t.turn[0].yaw_deg, None);
        assert_eq!(t.turn[0].pitch_deg, Some(10.0));
        assert_eq!(t.turn[0].degrees_per_second, 0.0);
        assert_eq!(t.turn[0].seconds, 0.0);
    }
}
