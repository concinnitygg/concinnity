// Which bodies moved far enough in one step to need a sweep.
//
// The measure is the body's thinnest width rather than its width along the
// motion. The thinnest is never the larger of the two, so a gate built on it
// fires at least as early as one built on the exact width, and it is settled
// once when the body is built rather than by a support call per body per
// step.
//
// That is what makes the gate safe to set well below one width: a body only
// passes through something if it moves further in a step than its own width
// plus the surface's, so any motion able to tunnel has already armed the
// sweep.

use crate::physics::ColliderShape;

use crate::physics::sim::math::Vec3;

/// The shape's smallest width, which is the least of itself it can leave
/// behind on the way past something.
pub(crate) fn min_extent(shape: &ColliderShape) -> f32 {
    match *shape {
        ColliderShape::Ball { radius } => 2.0 * radius.abs(),
        // The caps are as wide as the cylinder, so the thin way across a
        // capsule is the same whichever end is leading.
        ColliderShape::Capsule { radius, .. } => 2.0 * radius.abs(),
        ColliderShape::Cuboid { half_extents } => {
            let thinnest = half_extents
                .iter()
                .fold(f32::INFINITY, |least, &half| least.min(half.abs()));
            2.0 * thinnest
        }
    }
}

/// Whether one step's motion is large enough that the step's own contact test
/// could miss what the body went through. `width` is the body's cached
/// [`min_extent`].
pub(crate) fn is_fast(width: f32, motion: Vec3, ratio: f32) -> bool {
    if width <= 0.0 || ratio <= 0.0 {
        return false;
    }
    let threshold = ratio * width;
    motion.length_squared() > threshold * threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::sim::math::vec3;

    const RATIO: f32 = 0.5;
    const TICK: f32 = 1.0 / 60.0;

    const BALL: ColliderShape = ColliderShape::Ball { radius: 0.5 };
    const CRATE: ColliderShape = ColliderShape::Cuboid {
        half_extents: [0.4, 0.4, 0.4],
    };
    const PANEL: ColliderShape = ColliderShape::Cuboid {
        half_extents: [2.0, 2.0, 0.02],
    };
    const CAPSULE: ColliderShape = ColliderShape::Capsule {
        half_height: 0.6,
        radius: 0.3,
    };

    #[test]
    fn the_width_is_the_thin_way_across_each_shape() {
        assert_eq!(min_extent(&BALL), 1.0);
        assert_eq!(min_extent(&CRATE), 0.8);
        assert!((min_extent(&PANEL) - 0.04).abs() < 1.0e-6);
        assert_eq!(min_extent(&CAPSULE), 0.6);
    }

    // The cheap path is the one nearly every body takes. A character walking,
    // a prop tumbling, a crate falling for a second: none of them are worth a
    // sweep, and the gate has to say so.
    #[test]
    fn ordinary_speeds_do_not_arm_the_sweep() {
        for speed in [1.0, 4.0, 8.0, 20.0] {
            let motion = vec3(0.0, -speed * TICK, 0.0);
            assert!(
                !is_fast(min_extent(&CRATE), motion, RATIO),
                "a 0.8-wide crate at {speed} units per second travels {} per tick",
                speed * TICK
            );
        }
        assert!(
            !is_fast(min_extent(&CAPSULE), vec3(8.0 * TICK, 0.0, 0.0), RATIO),
            "a character walking is not a candidate"
        );
    }

    #[test]
    fn a_body_crossing_a_unit_per_tick_arms_the_sweep() {
        assert!(is_fast(min_extent(&BALL), vec3(0.0, 0.0, -1.0), RATIO));
        assert!(is_fast(
            min_extent(&CRATE),
            vec3(0.0, -60.0 * TICK, 0.0),
            RATIO
        ));
    }

    // The property the whole gate rests on: tunnelling needs more motion than
    // the mover's own width, and at a ratio of one or less the gate is
    // already armed by then. Checked at the ratio the default config ships.
    #[test]
    fn nothing_can_tunnel_before_the_gate_fires() {
        for shape in [BALL, CRATE, PANEL, CAPSULE] {
            let width = min_extent(&shape);
            // The least motion that could carry the shape past something of
            // no thickness at all.
            let motion = vec3(width * 1.000_1, 0.0, 0.0);
            assert!(
                is_fast(width, motion, RATIO),
                "{shape:?} can cross its own {width} wide unswept"
            );
            assert!(is_fast(width, motion, 1.0), "{shape:?} at the ratio cap");
        }
    }

    #[test]
    fn a_shape_with_no_width_and_a_disabled_ratio_both_decline() {
        let point = ColliderShape::Ball { radius: 0.0 };
        assert!(!is_fast(min_extent(&point), vec3(100.0, 0.0, 0.0), RATIO));
        assert!(!is_fast(min_extent(&BALL), vec3(100.0, 0.0, 0.0), 0.0));
        assert!(!is_fast(min_extent(&BALL), vec3(100.0, 0.0, 0.0), -1.0));
    }

    #[test]
    fn the_gate_reads_the_whole_motion_and_not_one_axis_of_it() {
        // Each axis is under the threshold; together they are over it.
        let motion = vec3(0.3, 0.3, 0.3);
        assert!(motion.x < 0.5 * min_extent(&BALL));
        assert!(
            is_fast(min_extent(&BALL), motion, RATIO),
            "{}",
            motion.length()
        );
    }
}
