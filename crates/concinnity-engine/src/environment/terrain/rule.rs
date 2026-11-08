//! Where on the ground a cover belongs: a [`GroundRule`] of altitude and slope.

use concinnity_core::math::{cos, sqrt};

/// Where on the ground something belongs: between two altitudes and between
/// two slopes, fading in and out over soft edges.
///
/// `altitude` is measured in meters above a [Planet](#planet)'s `radius`, its
/// sea level, and `slope` in degrees from level ground. Either end of either
/// band may be left unset (`null`), which leaves the band open on that side;
/// a rule with every end unset holds everywhere.
///
/// At each set bound the rule fades out over `altitude_edge` meters or
/// `slope_edge` degrees. The boundary also wanders by about as much again,
/// along a pattern fixed to the ground and the same every run, so two covers
/// meeting at one altitude interlock in ragged tongues rather than along a
/// contour line. Rules that meet at the same bound with the same edge share
/// that pattern, so where one ends the other begins.
///
/// ```rust
/// # use concinnity_engine::components::GroundRule;
/// // A beach: from 4 m below sea level to 2.5 m above it, on gentle ground.
/// GroundRule {
///     altitude: [Some(-4.0), Some(2.5)],
///     slope: [None, Some(25.0)],
///     altitude_edge: 0.8,
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct GroundRule {
    /// Lowest and highest altitude, in meters above sea level. Unset ends are
    /// open.
    pub altitude: [Option<f32>; 2],
    /// Gentlest and steepest slope, in degrees from level, within [0, 90].
    /// Unset ends are open.
    pub slope: [Option<f32>; 2],
    /// How wide the fade at an altitude bound is, in meters. At least 0.01.
    #[asset(default = 1.0)]
    pub altitude_edge: f32,
    /// How wide the fade at a slope bound is, in degrees. At least 0.1.
    #[asset(default = 5.0)]
    pub slope_edge: f32,
}

/// Narrowest altitude fade a rule keeps, in meters.
pub const MIN_ALTITUDE_EDGE: f32 = 0.01;

/// Narrowest slope fade a rule keeps, in degrees.
pub const MIN_SLOPE_EDGE: f32 = 0.1;

// The farthest an altitude bound reaches, in meters: an infinite bound stands
// beyond any ground at this.
const ALTITUDE_LIMIT: f32 = 1.0e12;

impl GroundRule {
    /// The rule that holds everywhere.
    pub const EVERYWHERE: Self = Self {
        altitude: [None, None],
        slope: [None, None],
        altitude_edge: 1.0,
        slope_edge: 5.0,
    };

    /// This rule with every value made usable: NaN bounds unset, infinite
    /// altitudes beyond any ground (a lowest altitude of `+inf` holds
    /// nowhere), slopes within [0, 90] degrees, a band given highest first
    /// put in order, and the edges no narrower than [`MIN_ALTITUDE_EDGE`] and
    /// [`MIN_SLOPE_EDGE`].
    #[must_use]
    pub fn validated(self) -> Self {
        let finite = |v: Option<f32>| v.filter(|v| !v.is_nan());
        let ordered = |[lo, hi]: [Option<f32>; 2]| match (lo, hi) {
            (Some(a), Some(b)) if a > b => [Some(b), Some(a)],
            _ => [lo, hi],
        };
        let edge = |v: f32, min: f32, default: f32| {
            if v.is_finite() { v.max(min) } else { default }
        };
        Self {
            altitude: ordered(
                self.altitude
                    .map(|v| finite(v).map(|v| v.clamp(-ALTITUDE_LIMIT, ALTITUDE_LIMIT))),
            ),
            slope: ordered(self.slope.map(|v| finite(v).map(|v| v.clamp(0.0, 90.0)))),
            altitude_edge: edge(self.altitude_edge, MIN_ALTITUDE_EDGE, 1.0),
            slope_edge: edge(self.slope_edge, MIN_SLOPE_EDGE, 5.0),
        }
    }
}

impl GroundRule {
    /// The rule as ramps, validated first: `[altitude, slope]`, each the
    /// `[k, c]` of its lower bound then its upper, so a bound's ramp at a
    /// measure `m` is `m * k + c` (see [`steepness`] for slopes).
    pub fn ramps(&self) -> [[f32; 4]; 2] {
        let r = self.validated();
        let [a0, a1] = altitude_ramp(r.altitude[0], r.altitude_edge, true);
        let [a2, a3] = altitude_ramp(r.altitude[1], r.altitude_edge, false);
        let [s0, s1] = slope_ramp(r.slope[0], r.slope_edge, true);
        let [s2, s3] = slope_ramp(r.slope[1], r.slope_edge, false);
        [[a0, a1, a2, a3], [s0, s1, s2, s3]]
    }
}

/// An open bound's ramp offset, so the ramp is always full. Matches
/// `GROUND_RULE_OPEN` in the shaders.
pub const GROUND_RULE_OPEN: f32 = 3.0;

/// The most the edge noise shifts a ramp, in units of its edge. Matches
/// `GROUND_RULE_WANDER` in the shaders.
pub const GROUND_RULE_WANDER: f32 = 1.5;

const OPEN: [f32; 2] = [0.0, GROUND_RULE_OPEN];
const NOWHERE: [f32; 2] = [0.0, -GROUND_RULE_OPEN];

// The ramp `[k, c]` over a measure in meters that rises across `bound` toward the
// inside, `rising` for a lower bound, over `edge` meters.
fn altitude_ramp(bound: Option<f32>, edge: f32, rising: bool) -> [f32; 2] {
    let Some(b) = bound else {
        return OPEN;
    };
    let k = if rising { 1.0 } else { -1.0 } / edge;
    [k, 0.5 - b * k]
}

// The ramp over the steepness across a slope `bound` in degrees, `gentlest`
// for the lower bound, over `edge` degrees. A bound at level ground
// (gentlest) or at a cliff (steepest) leaves the band open; one at the other
// end (gentlest at a cliff, steepest at level) closes it.
fn slope_ramp(bound: Option<f32>, edge: f32, gentlest: bool) -> [f32; 2] {
    let Some(b) = bound else {
        return OPEN;
    };
    if (gentlest && b <= 0.0) || (!gentlest && b >= 90.0) {
        return OPEN;
    }
    if (gentlest && b >= 90.0) || (!gentlest && b <= 0.0) {
        return NOWHERE;
    }
    // Half weight at the bound itself, narrowed near level ground or a cliff
    // so the fade stays within [0, 90] degrees.
    let half = (edge * 0.5).min(b).min(90.0 - b);
    let width = steepness(b + half) - steepness(b - half);
    let k = if gentlest { 1.0 } else { -1.0 } / width;
    [k, 0.5 - steepness(b) * k]
}

/// How steep ground `degrees` from level is, the measure slope rules ramp
/// over: `sqrt(1 - cos)`, from 0 on level ground to 1 on a cliff, close to
/// proportional to the angle. The shaders take it from the cosine between the
/// ground's normal and up.
pub fn steepness(degrees: f32) -> f32 {
    sqrt((1.0 - cos(degrees.to_radians())).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_open_rule_ramps_full_everywhere() {
        let open = [0.0, GROUND_RULE_OPEN, 0.0, GROUND_RULE_OPEN];
        assert_eq!(GroundRule::EVERYWHERE.ramps(), [open, open]);
        let band = GroundRule {
            altitude: [Some(-1.0), Some(3.0)],
            altitude_edge: 2.0,
            ..GroundRule::default()
        }
        .ramps()[0];
        assert_eq!(band[1] - band[0], 0.5);
        assert_eq!(3.0 * band[2] + band[3], 0.5);
    }

    #[test]
    fn steepness_runs_from_level_to_a_cliff() {
        assert_eq!(steepness(0.0), 0.0);
        assert!((steepness(90.0) - 1.0).abs() < 1e-6);
        assert!(steepness(30.0) < steepness(60.0));
    }

    #[test]
    fn the_default_holds_everywhere() {
        assert_eq!(GroundRule::default(), GroundRule::EVERYWHERE);
    }

    #[test]
    fn validation_orders_bands_and_drops_non_finite_bounds() {
        let r = GroundRule {
            altitude: [Some(5.0), Some(-3.0)],
            slope: [Some(f32::NAN), Some(120.0)],
            altitude_edge: f32::NAN,
            slope_edge: -2.0,
        }
        .validated();
        assert_eq!(r.altitude, [Some(-3.0), Some(5.0)]);
        assert_eq!(r.slope, [None, Some(90.0)]);
        assert_eq!(r.altitude_edge, 1.0);
        assert_eq!(r.slope_edge, MIN_SLOPE_EDGE);
        let far = GroundRule {
            altitude: [Some(f32::INFINITY), Some(f32::NEG_INFINITY)],
            altitude_edge: 0.0,
            ..GroundRule::default()
        }
        .validated();
        assert_eq!(far.altitude, [Some(-ALTITUDE_LIMIT), Some(ALTITUDE_LIMIT)]);
        assert_eq!(far.altitude_edge, MIN_ALTITUDE_EDGE);
        let nowhere = GroundRule {
            altitude: [Some(f32::INFINITY), None],
            ..GroundRule::default()
        }
        .validated();
        assert_eq!(nowhere.altitude, [Some(ALTITUDE_LIMIT), None]);
    }

    #[test]
    fn a_rule_round_trips_through_json_with_open_ends() {
        let r: GroundRule =
            serde_json::from_str(r#"{"altitude":[null,3.0],"slope_edge":2.0}"#).unwrap();
        assert_eq!(r.altitude, [None, Some(3.0)]);
        assert_eq!(r.slope, [None, None]);
        assert_eq!(r.altitude_edge, 1.0);
        assert_eq!(r.slope_edge, 2.0);
        let back: GroundRule = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(back, r);
    }
}
