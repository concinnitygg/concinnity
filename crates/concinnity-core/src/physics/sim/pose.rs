// The two rotation conversions the simulation's boundary is written in. They
// live here rather than in a caller so a pose blended outside the simulation
// uses the same YXZ decomposition the bodies inside it were built with.

use super::math::Quat;

/// The `[x, y, z, w]` rotation quaternion for engine Euler degrees
/// `[pitch, yaw, roll]`, applied in YXZ order.
///
/// Rotations are blended in quaternion space, so a caller interpolating poses
/// converts its authored angles once here rather than per frame.
pub fn quat_from_euler_deg(euler_deg: [f32; 3]) -> [f32; 4] {
    Quat::from_euler_deg(euler_deg).to_xyzw()
}

/// Engine Euler degrees `[pitch, yaw, roll]` for an `[x, y, z, w]` rotation
/// quaternion, which need not be normalised.
///
/// The decomposition is lossy at +-90 degrees of pitch, where yaw and roll
/// fold into one angle: the whole rotation is reported as yaw.
pub fn euler_deg_from_quat(q: [f32; 4]) -> [f32; 3] {
    Quat::from_xyzw(q).to_euler_deg()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rotation_round_trips_away_from_gimbal_lock() {
        // Pitch kept well clear of +-90 deg so the YXZ decomposition is unique.
        for euler in [[0.0, 0.0, 0.0], [12.0, 45.0, -30.0], [-20.0, 170.0, 60.0]] {
            let back = euler_deg_from_quat(quat_from_euler_deg(euler));
            for axis in 0..3 {
                let diff = (back[axis] - euler[axis]).rem_euclid(360.0);
                let diff = diff.min(360.0 - diff);
                assert!(diff < 0.01, "axis {axis}: {back:?} != {euler:?}");
            }
        }
    }

    #[test]
    fn the_identity_quaternion_is_zero_euler() {
        assert_eq!(euler_deg_from_quat([0.0, 0.0, 0.0, 1.0]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn an_unnormalised_quaternion_decomposes_the_same_way() {
        let q = quat_from_euler_deg([12.0, 45.0, -30.0]);
        let scaled = [q[0] * 4.0, q[1] * 4.0, q[2] * 4.0, q[3] * 4.0];
        let (a, b) = (euler_deg_from_quat(q), euler_deg_from_quat(scaled));
        for axis in 0..3 {
            assert!((a[axis] - b[axis]).abs() < 0.01, "axis {axis}: {a:?} {b:?}");
        }
    }
}
