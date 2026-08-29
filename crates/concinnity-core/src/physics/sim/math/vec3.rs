// The simulation's 3-vector. It exists rather than a general math crate being
// pulled in because the simulation needs exactly one guarantee from its
// arithmetic: that the same inputs produce the same bits on every platform.
// Every operation here is plain f32 add/sub/mul/div or a `libm` call, both of
// which are specified exactly.

use crate::math::sqrt;
use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct Vec3 {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) z: f32,
}

/// Shorthand constructor.
pub(crate) const fn vec3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3 { x, y, z }
}

impl Vec3 {
    pub(crate) const ZERO: Vec3 = vec3(0.0, 0.0, 0.0);
    pub(crate) const X: Vec3 = vec3(1.0, 0.0, 0.0);
    pub(crate) const Y: Vec3 = vec3(0.0, 1.0, 0.0);
    pub(crate) const Z: Vec3 = vec3(0.0, 0.0, 1.0);

    pub(crate) const fn splat(v: f32) -> Self {
        vec3(v, v, v)
    }

    pub(crate) const fn from_array(v: [f32; 3]) -> Self {
        vec3(v[0], v[1], v[2])
    }

    pub(crate) const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    pub(crate) const fn axis(index: usize) -> Self {
        match index {
            0 => Self::X,
            1 => Self::Y,
            _ => Self::Z,
        }
    }

    pub(crate) const fn get(self, index: usize) -> f32 {
        match index {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }

    pub(crate) fn set(&mut self, index: usize, value: f32) {
        match index {
            0 => self.x = value,
            1 => self.y = value,
            _ => self.z = value,
        }
    }

    pub(crate) fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub(crate) fn cross(self, other: Self) -> Self {
        vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub(crate) fn length_squared(self) -> f32 {
        self.dot(self)
    }

    pub(crate) fn length(self) -> f32 {
        sqrt(self.length_squared())
    }

    /// Unit vector, or `ZERO` when the vector is too short to have a direction.
    pub(crate) fn normalize_or_zero(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq <= f32::MIN_POSITIVE {
            return Self::ZERO;
        }
        self * (1.0 / sqrt(len_sq))
    }

    /// Unit vector, falling back to `fallback` when there is no direction.
    pub(crate) fn normalize_or(self, fallback: Self) -> Self {
        let normalized = self.normalize_or_zero();
        if normalized == Self::ZERO {
            fallback
        } else {
            normalized
        }
    }

    pub(crate) fn abs(self) -> Self {
        vec3(self.x.abs(), self.y.abs(), self.z.abs())
    }

    pub(crate) fn min(self, other: Self) -> Self {
        vec3(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    pub(crate) fn max(self, other: Self) -> Self {
        vec3(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }

    pub(crate) fn clamp(self, low: Self, high: Self) -> Self {
        self.max(low).min(high)
    }

    /// Index of the component with the largest absolute value.
    pub(crate) fn max_abs_axis(self) -> usize {
        let a = self.abs();
        if a.x >= a.y && a.x >= a.z {
            0
        } else if a.y >= a.z {
            1
        } else {
            2
        }
    }

    /// Any unit vector perpendicular to `self`, which must be unit length.
    /// Picking the smaller component to cross against keeps the result well
    /// conditioned whatever direction `self` points.
    pub(crate) fn any_perpendicular(self) -> Self {
        if self.x.abs() <= self.y.abs() && self.x.abs() <= self.z.abs() {
            self.cross(Self::X).normalize_or(Self::Y)
        } else if self.y.abs() <= self.z.abs() {
            self.cross(Self::Y).normalize_or(Self::Z)
        } else {
            self.cross(Self::Z).normalize_or(Self::X)
        }
    }

    pub(crate) fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 {
        vec3(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 {
        vec3(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: f32) -> Vec3 {
        vec3(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Vec3 {
        vec3(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, rhs: f32) -> Vec3 {
        vec3(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        vec3(-self.x, -self.y, -self.z)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Vec3) {
        *self = *self + rhs;
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Vec3) {
        *self = *self - rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_and_cross_follow_the_right_hand_rule() {
        assert_eq!(Vec3::X.cross(Vec3::Y), Vec3::Z);
        assert_eq!(Vec3::Y.cross(Vec3::Z), Vec3::X);
        assert_eq!(Vec3::Z.cross(Vec3::X), Vec3::Y);
        assert_eq!(Vec3::X.dot(Vec3::Y), 0.0);
        assert_eq!(vec3(1.0, 2.0, 3.0).dot(vec3(4.0, 5.0, 6.0)), 32.0);
    }

    #[test]
    fn length_matches_the_pythagorean_answer() {
        assert_eq!(vec3(3.0, 4.0, 0.0).length(), 5.0);
        assert_eq!(vec3(3.0, 4.0, 0.0).length_squared(), 25.0);
    }

    #[test]
    fn normalizing_a_zero_vector_yields_zero_rather_than_nan() {
        assert_eq!(Vec3::ZERO.normalize_or_zero(), Vec3::ZERO);
        assert_eq!(Vec3::ZERO.normalize_or(Vec3::Y), Vec3::Y);
        let n = vec3(0.0, -2.0, 0.0).normalize_or(Vec3::Y);
        assert_eq!(n, vec3(0.0, -1.0, 0.0));
    }

    #[test]
    fn a_perpendicular_is_perpendicular_and_unit_length_for_every_axis() {
        for dir in [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            -Vec3::X,
            vec3(0.6, 0.8, 0.0),
            vec3(0.0, 0.6, -0.8),
        ] {
            let p = dir.any_perpendicular();
            assert!((p.dot(dir)).abs() < 1.0e-6, "{dir:?} -> {p:?}");
            assert!((p.length() - 1.0).abs() < 1.0e-6, "{dir:?} -> {p:?}");
        }
    }

    #[test]
    fn component_access_round_trips_through_index() {
        let mut v = vec3(1.0, 2.0, 3.0);
        assert_eq!([v.get(0), v.get(1), v.get(2)], [1.0, 2.0, 3.0]);
        v.set(1, 9.0);
        assert_eq!(v, vec3(1.0, 9.0, 3.0));
        assert_eq!(vec3(1.0, -9.0, 3.0).max_abs_axis(), 1);
        assert_eq!(Vec3::axis(2), Vec3::Z);
    }

    #[test]
    fn arithmetic_matches_component_wise_expectations() {
        let a = vec3(1.0, 2.0, 3.0);
        let b = vec3(0.5, -1.0, 2.0);
        assert_eq!(a + b, vec3(1.5, 1.0, 5.0));
        assert_eq!(a - b, vec3(0.5, 3.0, 1.0));
        assert_eq!(a * 2.0, vec3(2.0, 4.0, 6.0));
        assert_eq!(a * b, vec3(0.5, -2.0, 6.0));
        assert_eq!(a / 2.0, vec3(0.5, 1.0, 1.5));
        assert_eq!(-a, vec3(-1.0, -2.0, -3.0));
        assert_eq!(a.min(b), vec3(0.5, -1.0, 2.0));
        assert_eq!(a.max(b), vec3(1.0, 2.0, 3.0));
        assert_eq!(b.abs(), vec3(0.5, 1.0, 2.0));
        assert_eq!(a.clamp(Vec3::ZERO, Vec3::splat(2.0)), vec3(1.0, 2.0, 2.0));
    }
}
