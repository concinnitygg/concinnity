// What a character move is allowed to do: how steep a surface it may stand on,
// how high a lip it may climb, and whether gravity governs it at all. Separate
// from `SimConfig` because none of it tunes a step -- two worlds with the same
// solver can want different characters -- and because the slope limit is
// authored in degrees while every comparison against it is on a normal's
// upward component, a conversion worth doing in one place.

use crate::math::cos;
use crate::physics::sim::math::Vec3;

/// Smallest upward normal component that still counts as a surface rather than
/// a wall, for a mover whose slope limit is switched off. A wall's normal is
/// zero here, so a limit of `0` still does not make walls walkable.
const UPWARD: f32 = 1.0e-3;

/// Tuning for a character move.
///
/// [`CharacterConfig::default`] is a gravity-bound character on ordinary
/// terrain: a 45 degree climb limit and a knee-high step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CharacterConfig {
    /// Steepest surface, in degrees from horizontal, the mover treats as
    /// ground. Anything steeper is a wall it slides down rather than climbs.
    /// `0` disables the limit, leaving every upward-facing surface walkable.
    pub max_slope_deg: f32,
    /// Tallest obstacle a grounded mover climbs onto instead of stopping at,
    /// and the furthest it stays attached to the ground stepping off a lip.
    /// `0` disables both.
    pub step_height: f32,
    /// Whether the mover is gravity-bound. A grounded character climbs steps
    /// and stays attached to the ground; a free-flying camera does neither.
    pub grounded: bool,
}

impl Default for CharacterConfig {
    fn default() -> Self {
        CharacterConfig {
            max_slope_deg: 45.0,
            step_height: 0.3,
            grounded: true,
        }
    }
}

impl CharacterConfig {
    /// Tune a character by climb limit, step height, and whether gravity
    /// governs it.
    pub(crate) fn new(max_slope_deg: f32, step_height: f32, grounded: bool) -> Self {
        CharacterConfig {
            max_slope_deg,
            step_height,
            grounded,
        }
    }

    /// Smallest upward normal component a surface can have and still be stood
    /// on.
    pub(crate) fn min_ground_normal_y(&self) -> f32 {
        if self.max_slope_deg <= 0.0 {
            return UPWARD;
        }
        // A NaN limit falls out of `max` as the disabled limit rather than as
        // a comparison that admits everything.
        let radians = self.max_slope_deg * (core::f32::consts::PI / 180.0);
        cos(radians).max(UPWARD)
    }

    /// Whether a surface with this contact normal can be walked on rather than
    /// slid down.
    pub(crate) fn is_walkable(&self, normal: Vec3) -> bool {
        normal.y >= self.min_ground_normal_y()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::sin;
    use crate::physics::sim::math::vec3;

    // A normal that many degrees off vertical, in the xy plane.
    fn tilted(degrees: f32) -> Vec3 {
        let radians = degrees * (core::f32::consts::PI / 180.0);
        vec3(sin(radians), cos(radians), 0.0)
    }

    #[test]
    fn a_slope_at_the_limit_is_walkable_and_one_past_it_is_not() {
        let config = CharacterConfig::new(45.0, 0.3, true);
        assert!(config.is_walkable(Vec3::Y));
        assert!(config.is_walkable(tilted(44.0)));
        assert!(!config.is_walkable(tilted(46.0)));
        assert!(
            !config.is_walkable(vec3(1.0, 0.0, 0.0)),
            "a wall is not ground"
        );
        assert!(!config.is_walkable(-Vec3::Y), "a ceiling is not ground");
    }

    #[test]
    fn a_disabled_limit_walks_every_upward_face_but_still_not_a_wall() {
        for max_slope_deg in [0.0, -10.0] {
            let config = CharacterConfig::new(max_slope_deg, 0.3, true);
            assert!(config.is_walkable(tilted(80.0)), "{max_slope_deg}");
            assert!(!config.is_walkable(vec3(1.0, 0.0, 0.0)), "{max_slope_deg}");
            assert!(!config.is_walkable(tilted(95.0)), "{max_slope_deg}");
        }
    }

    // A limit at or past vertical means the same thing as no limit, rather
    // than a cosine that has gone negative and made ceilings walkable.
    #[test]
    fn a_limit_at_vertical_or_beyond_stops_at_upward_facing() {
        for max_slope_deg in [90.0, 120.0] {
            let config = CharacterConfig::new(max_slope_deg, 0.3, true);
            assert!(config.min_ground_normal_y() > 0.0, "{max_slope_deg}");
            assert!(config.is_walkable(tilted(89.0)), "{max_slope_deg}");
            assert!(!config.is_walkable(tilted(91.0)), "{max_slope_deg}");
        }
    }

    #[test]
    fn the_default_is_a_grounded_character_on_ordinary_terrain() {
        let config = CharacterConfig::default();
        assert!(config.grounded);
        assert!(config.step_height > 0.0);
        assert!(config.is_walkable(tilted(30.0)));
        assert!(!config.is_walkable(tilted(60.0)));
    }
}
