// The drag that brings a floating body to the water's speed. Pure, so a
// step's outcome is the same however it is grouped.

use crate::components::Buoyancy;
use concinnity_core::math::exp;

/// The body's linear and angular velocity after `dt` seconds of drag,
/// `share` of it submerged in water moving at `water`: each decays toward the
/// water's by exactly the exponential the drag rate sets, so no step can
/// overshoot it however long it is.
pub(super) fn drag(
    linear: [f32; 3],
    angular: [f32; 3],
    water: [f32; 3],
    share: f32,
    buoyancy: Buoyancy,
    dt: f32,
) -> ([f32; 3], [f32; 3]) {
    let share = share.clamp(0.0, 1.0);
    let keep = exp(-buoyancy.linear_drag * share * dt);
    let spin = exp(-buoyancy.angular_drag * share * dt);
    (
        core::array::from_fn(|i| water[i] + (linear[i] - water[i]) * keep),
        angular.map(|w| w * spin),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Drag brings the body to the water's speed and never past it, at any
    // step, and does nothing out of the water.
    #[test]
    fn drag_never_overshoots() {
        let b = Buoyancy {
            linear_drag: 50.0,
            angular_drag: 50.0,
        };
        let water = [1.0, 0.0, -2.0];
        for dt in [1.0 / 60.0, 0.5, 10.0] {
            let (v, w) = drag([5.0, 3.0, 0.0], [4.0, 0.0, -1.0], water, 1.0, b, dt);
            for i in 0..3 {
                let before = [5.0, 3.0, 0.0][i] - water[i];
                let after = v[i] - water[i];
                assert!(after * before >= 0.0 && after.abs() <= before.abs(), "{dt}");
            }
            assert!(w[0] <= 4.0 && w[0] >= 0.0);
        }
        let dry = drag([5.0, 3.0, 0.0], [4.0, 0.0, -1.0], water, 0.0, b, 1.0);
        assert_eq!(dry, ([5.0, 3.0, 0.0], [4.0, 0.0, -1.0]));
    }
}
