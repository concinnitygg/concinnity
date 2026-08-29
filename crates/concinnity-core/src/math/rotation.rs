//! The engine's rotation convention, in one place: unit quaternions as
//! `[x, y, z, w]`, and the YXZ Euler decomposition every authored
//! `[pitch, yaw, roll]` triple is read and written through.
//!
//! Both directions live here rather than with a caller because the two sides
//! that need them are far apart: the cook turns imported quaternions into
//! authored angles, and the simulation turns stepped body rotations back into
//! the same. A second implementation would let those two disagree about what a
//! rotation means.

use crate::math::{atan2, sin_cos, sqrt};

/// Unit quaternion `(x, y, z, w)` representing a rotation.
pub type Quat = [f32; 4];

/// Hamilton product `a * b`.
fn mul(a: Quat, b: Quat) -> Quat {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

/// Rotation of `angle_rad` about a canonical axis, `axis` being 0, 1 or 2.
fn about_axis(axis: usize, angle_rad: f32) -> Quat {
    let (s, c) = sin_cos(angle_rad * 0.5);
    let mut q = [0.0, 0.0, 0.0, c];
    q[axis] = s;
    q
}

/// A rotation quaternion scaled to unit length, or the identity when `q` is too
/// short to have a direction.
pub fn quat_normalize(q: Quat) -> Quat {
    let len = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if len < 1e-12 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
}

/// The rotation quaternion for engine Euler degrees `[pitch, yaw, roll]`,
/// applied yaw then pitch then roll.
pub fn quat_from_euler_yxz_deg(euler_deg: [f32; 3]) -> Quat {
    let [pitch, yaw, roll] = euler_deg;
    mul(
        mul(
            about_axis(1, yaw.to_radians()),
            about_axis(0, pitch.to_radians()),
        ),
        about_axis(2, roll.to_radians()),
    )
}

/// Engine Euler degrees `[pitch, yaw, roll]` for a rotation quaternion, which
/// need not be normalised.
///
/// The decomposition is lossy at +-90 degrees of pitch, where yaw and roll fold
/// into one angle: the whole rotation is reported as yaw.
pub fn euler_yxz_deg_from_quat(q: Quat) -> [f32; 3] {
    let [x, y, z, w] = quat_normalize(q);
    // The column-major rotation-matrix entries the YXZ decomposition reads.
    let m21 = 2.0 * (y * z - w * x);
    let m20 = 2.0 * (x * z + w * y);
    let m22 = 1.0 - 2.0 * (x * x + y * y);
    let m01 = 2.0 * (x * y + w * z);
    let m11 = 1.0 - 2.0 * (x * x + z * z);
    let m10 = 2.0 * (x * y - w * z);
    let m00 = 1.0 - 2.0 * (y * y + z * z);

    let sp = (-m21).clamp(-1.0, 1.0);
    // cos(pitch) is taken straight from the matrix (the column-2 length in the
    // XZ plane) rather than `sqrt(1 - sp*sp)`, which loses nearly all its
    // precision to catastrophic cancellation as pitch approaches +-90 degrees.
    let cp = sqrt(m20 * m20 + m22 * m22);
    let pitch = atan2(sp, cp);
    let (yaw, roll) = if cp > 1e-4 {
        (atan2(m20, m22), atan2(m01, m11))
    } else {
        (atan2(sp * m10, m00), 0.0)
    };
    [pitch.to_degrees(), yaw.to_degrees(), roll.to_degrees()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < tol)
    }

    #[test]
    fn euler_round_trips_away_from_gimbal_lock() {
        for e in [
            [0.0, 0.0, 0.0],
            [12.0, -34.0, 56.0],
            [-80.0, 170.0, -170.0],
            [45.0, 90.0, -45.0],
        ] {
            let got = euler_yxz_deg_from_quat(quat_from_euler_yxz_deg(e));
            assert!(close(got, e, 1e-2), "{e:?} round-tripped to {got:?}");
        }
    }

    #[test]
    fn pitch_at_gimbal_lock_pins_roll_to_zero() {
        for pitch in [-90.0, 90.0] {
            let got = euler_yxz_deg_from_quat(quat_from_euler_yxz_deg([pitch, 30.0, 40.0]));
            assert!((got[0] - pitch).abs() < 1e-2, "pitch became {}", got[0]);
            assert_eq!(got[2], 0.0, "roll should fold into yaw");
        }
    }

    #[test]
    fn a_degenerate_quaternion_normalises_to_the_identity() {
        assert_eq!(quat_normalize([0.0; 4]), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(euler_yxz_deg_from_quat([0.0; 4]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn the_quaternion_built_is_already_unit_length() {
        for e in [[0.0, 0.0, 0.0], [12.0, -34.0, 56.0], [-89.0, 179.0, 61.0]] {
            let q = quat_from_euler_yxz_deg(e);
            let len2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
            assert!((len2 - 1.0).abs() < 1e-5, "{e:?} gave length^2 {len2}");
        }
    }
}
