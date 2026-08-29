// What is left of a move once it has met something. All of it is arithmetic on
// two vectors, with no world to ask, which is why it sits apart from the
// resolve loop that drives it: the awkward cases -- a slope too steep to climb,
// two walls meeting at a crease -- are the ones worth testing directly rather
// than through a scene built to provoke them.

use crate::physics::sim::math::{Vec3, vec3};

/// The part of `motion` that survives a contact with a surface whose normal is
/// `normal`: the motion projected onto the contact plane, so the mover slides
/// rather than stops.
///
/// A surface too steep to stand on must not lift the mover: if sliding along
/// it would carry the mover upward, that climb is dropped and only the
/// horizontal part is kept, which is what makes a steep slope a wall while
/// leaving gravity free to pull the mover back down it.
pub(crate) fn deflect(motion: Vec3, normal: Vec3, walkable: bool) -> Vec3 {
    let slid = motion - normal * motion.dot(normal);
    if walkable || slid.y <= 0.0 {
        return slid;
    }
    horizontal(slid)
}

/// The part of `motion` that survives two contacts at once.
///
/// Sliding along either plane alone would drive the mover into the other, so
/// the only direction left is the crease the two make. Two opposed planes have
/// no crease, and a move into that wedge is over.
pub(crate) fn crease(motion: Vec3, first: Vec3, second: Vec3) -> Vec3 {
    let along = first.cross(second).normalize_or_zero();
    along * motion.dot(along)
}

/// Whether sliding along `normal` would drive the mover back into a surface
/// already met at `previous`.
pub(crate) fn re_entrant(slid: Vec3, previous: Vec3) -> bool {
    slid.dot(previous) < 0.0
}

/// The part of a vector that lies in the ground plane.
pub(crate) fn horizontal(v: Vec3) -> Vec3 {
    vec3(v.x, 0.0, v.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{cos, sin};

    // The normal of a hill that many degrees steep, rising toward +x: a move
    // along +x climbs it.
    fn hill(degrees: f32) -> Vec3 {
        let radians = degrees * (core::f32::consts::PI / 180.0);
        vec3(-sin(radians), cos(radians), 0.0)
    }

    #[test]
    fn a_move_straight_into_a_wall_has_nothing_left() {
        let slid = deflect(vec3(0.0, 0.0, 1.0), -Vec3::Z, false);
        assert!(slid.length() < 1.0e-6, "{slid:?}");
    }

    #[test]
    fn a_move_at_an_angle_to_a_wall_keeps_the_part_along_it() {
        let slid = deflect(vec3(1.0, 0.0, 1.0), -Vec3::Z, false);
        assert!((slid - vec3(1.0, 0.0, 0.0)).length() < 1.0e-6, "{slid:?}");
    }

    // A wall takes nothing out of gravity: a mover pressed against one still
    // falls down it.
    #[test]
    fn a_wall_leaves_a_falling_move_falling() {
        let slid = deflect(vec3(0.0, -1.0, 1.0), -Vec3::Z, false);
        assert!((slid - vec3(0.0, -1.0, 0.0)).length() < 1.0e-6, "{slid:?}");
    }

    #[test]
    fn a_walkable_slope_lets_the_move_climb_it() {
        let slid = deflect(vec3(1.0, -0.1, 0.0), hill(20.0), true);
        assert!(slid.y > 0.0, "walking into a ramp goes up it: {slid:?}");
    }

    // The same slope past the limit: the mover may keep going along it and may
    // be pulled down it, but it may not be carried up.
    #[test]
    fn a_slope_past_the_limit_never_carries_the_move_upward() {
        let climbing = deflect(vec3(1.0, -0.1, 0.0), hill(60.0), false);
        assert!(climbing.y <= 0.0, "{climbing:?}");
        assert!(
            climbing.z.abs() < 1.0e-6,
            "still along the slope: {climbing:?}"
        );

        let falling = deflect(vec3(0.0, -1.0, 0.0), hill(60.0), false);
        assert!(falling.y < 0.0, "gravity still slides it down: {falling:?}");
        assert!(falling.x < 0.0, "and away from the hill: {falling:?}");
    }

    #[test]
    fn a_crease_leaves_only_the_line_the_two_planes_share() {
        // Two walls meeting at a right angle: the crease is vertical.
        let along = crease(vec3(1.0, -1.0, 1.0), -Vec3::Z, -Vec3::X);
        assert!(
            along.x.abs() < 1.0e-6 && along.z.abs() < 1.0e-6,
            "{along:?}"
        );
        assert!((along.y + 1.0).abs() < 1.0e-6, "{along:?}");
    }

    #[test]
    fn two_opposed_walls_leave_nothing() {
        let along = crease(vec3(1.0, 0.0, 0.0), Vec3::X, -Vec3::X);
        assert_eq!(along, Vec3::ZERO);
    }

    #[test]
    fn re_entry_is_a_slide_that_turns_back_into_what_was_already_met() {
        assert!(re_entrant(vec3(0.0, 0.0, 1.0), -Vec3::Z));
        assert!(!re_entrant(vec3(0.0, 0.0, -1.0), -Vec3::Z));
        assert!(
            !re_entrant(vec3(1.0, 0.0, 0.0), -Vec3::Z),
            "along it is fine"
        );
    }

    #[test]
    fn horizontal_drops_the_vertical_part() {
        assert_eq!(horizontal(vec3(1.0, 5.0, -2.0)), vec3(1.0, 0.0, -2.0));
    }
}
