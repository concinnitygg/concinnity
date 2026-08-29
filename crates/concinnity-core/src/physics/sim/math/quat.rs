// Body orientation. Quaternions rather than matrices because integrating an
// angular velocity and renormalising afterwards is exact and cheap here, and
// because the pose crossing the crate boundary is already a quaternion.
//
// The Euler conversions use the engine's YXZ order in degrees, the same order
// `Prop::model_matrix` builds its rotation with, so a pose set from authored
// Euler angles reads back as the angles that were set.

use super::vec3::{Vec3, vec3};
use crate::math::sqrt;
#[cfg(test)]
use crate::math::{cos, sin};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Quat {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) z: f32,
    pub(crate) w: f32,
}

impl Default for Quat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Quat {
    pub(crate) const IDENTITY: Quat = Quat {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    pub(crate) const fn from_xyzw(v: [f32; 4]) -> Self {
        Quat {
            x: v[0],
            y: v[1],
            z: v[2],
            w: v[3],
        }
    }

    pub(crate) const fn to_xyzw(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    // Test-only since `from_euler_deg` began delegating to `crate::math`.
    #[cfg(test)]
    pub(crate) fn from_axis_angle(axis: Vec3, angle_rad: f32) -> Self {
        let half = angle_rad * 0.5;
        let s = sin(half);
        Quat {
            x: axis.x * s,
            y: axis.y * s,
            z: axis.z * s,
            w: cos(half),
        }
    }

    /// Engine Euler degrees `[pitch, yaw, roll]`, applied yaw then pitch then
    /// roll.
    pub(crate) fn from_euler_deg(euler_deg: [f32; 3]) -> Self {
        Self::from_xyzw(crate::math::quat_from_euler_yxz_deg(euler_deg))
    }

    /// Decompose back into engine Euler degrees `[pitch, yaw, roll]`.
    pub(crate) fn to_euler_deg(self) -> [f32; 3] {
        crate::math::euler_yxz_deg_from_quat(self.to_xyzw())
    }

    pub(crate) fn mul(self, rhs: Quat) -> Quat {
        Quat {
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
        }
    }

    pub(crate) fn conjugate(self) -> Quat {
        Quat {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    pub(crate) fn normalize(self) -> Quat {
        let len_sq = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        if len_sq <= f32::MIN_POSITIVE {
            return Quat::IDENTITY;
        }
        let inv = 1.0 / sqrt(len_sq);
        Quat {
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
            w: self.w * inv,
        }
    }

    pub(crate) fn rotate(self, v: Vec3) -> Vec3 {
        // v + 2w(q x v) + 2(q x (q x v)), the standard expansion that avoids
        // building a matrix for a single vector.
        let q = vec3(self.x, self.y, self.z);
        let t = q.cross(v) * 2.0;
        v + t * self.w + q.cross(t)
    }

    pub(crate) fn inverse_rotate(self, v: Vec3) -> Vec3 {
        self.conjugate().rotate(v)
    }

    /// Advance an orientation by `angular_velocity` (radians per second) over
    /// `dt`, renormalising so repeated steps do not drift off the unit sphere.
    pub(crate) fn integrate(self, angular_velocity: Vec3, dt: f32) -> Quat {
        let w = Quat {
            x: angular_velocity.x,
            y: angular_velocity.y,
            z: angular_velocity.z,
            w: 0.0,
        };
        let d = w.mul(self);
        let half = dt * 0.5;
        Quat {
            x: self.x + d.x * half,
            y: self.y + d.y * half,
            z: self.z + d.z * half,
            w: self.w + d.w * half,
        }
        .normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1.0e-5
    }

    // A thousand first-order steps accumulate a little truncation error; this
    // is the tolerance a full turn integrated in pieces lands inside.
    fn close_after_many_steps(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1.0e-4
    }

    #[test]
    fn identity_rotates_nothing() {
        let v = vec3(1.0, 2.0, 3.0);
        assert!(close(Quat::IDENTITY.rotate(v), v));
        assert_eq!(Quat::IDENTITY.to_euler_deg(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn a_quarter_turn_about_y_maps_x_onto_negative_z() {
        let q = Quat::from_euler_deg([0.0, 90.0, 0.0]);
        assert!(close(q.rotate(Vec3::X), vec3(0.0, 0.0, -1.0)));
        assert!(close(q.inverse_rotate(vec3(0.0, 0.0, -1.0)), Vec3::X));
    }

    #[test]
    fn euler_round_trips_away_from_gimbal_lock() {
        for euler in [
            [0.0, 0.0, 0.0],
            [12.0, 45.0, -30.0],
            [-20.0, 170.0, 60.0],
            [80.0, -100.0, 15.0],
        ] {
            let back = Quat::from_euler_deg(euler).to_euler_deg();
            for axis in 0..3 {
                let diff = (back[axis] - euler[axis]).rem_euclid(360.0);
                let diff = diff.min(360.0 - diff);
                assert!(diff < 0.01, "axis {axis}: {back:?} != {euler:?}");
            }
        }
    }

    #[test]
    fn gimbal_lock_folds_roll_into_yaw_without_producing_nan() {
        let euler = Quat::from_euler_deg([90.0, 30.0, 20.0]).to_euler_deg();
        assert!(euler.iter().all(|a| a.is_finite()), "{euler:?}");
        assert!((euler[0] - 90.0).abs() < 0.05, "{euler:?}");
        assert_eq!(euler[2], 0.0);
    }

    #[test]
    fn multiplication_composes_rotations_in_order() {
        let yaw = Quat::from_euler_deg([0.0, 90.0, 0.0]);
        let combined = yaw.mul(yaw);
        assert!(close(combined.rotate(Vec3::X), -Vec3::X));
        assert!(close(yaw.mul(yaw.conjugate()).rotate(Vec3::X), Vec3::X));
    }

    // Integration must both turn the body and keep the quaternion normalised:
    // a thousand steps of spin should still be a unit quaternion.
    #[test]
    fn integrating_a_spin_stays_normalised() {
        let mut q = Quat::IDENTITY;
        let spin = vec3(0.0, core::f32::consts::TAU, 0.0);
        for _ in 0..1000 {
            q = q.integrate(spin, 1.0 / 1000.0);
        }
        let len = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        assert!((len - 1.0).abs() < 1.0e-5, "len {len}");
        // A full turn about Y lands back where it started.
        assert!(
            close_after_many_steps(q.rotate(Vec3::X), Vec3::X),
            "{:?}",
            q.rotate(Vec3::X)
        );
    }

    #[test]
    fn xyzw_round_trips() {
        let q = Quat::from_euler_deg([10.0, 20.0, 30.0]);
        assert_eq!(Quat::from_xyzw(q.to_xyzw()), q);
    }
}
