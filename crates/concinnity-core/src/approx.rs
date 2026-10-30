//! Comparing floats within a tolerance, for tests. Every crate's tests used to
//! carry their own `close` or `approx`, so the shared one ships behind the
//! `test-support` feature (and in this crate's own test build) beside
//! [`crate::bench`].
//!
//! A tolerance bounds each component on its own, so a vector is close when no
//! component is further off than `tol`. A NaN anywhere is never close.

use core::fmt::Debug;

/// A float, or an array of them, compared component by component.
pub trait Approx: Copy + Debug {
    /// The float a tolerance is given in.
    type Scalar: Copy + Debug;

    /// Whether every component is within `tol` of the matching one in `want`.
    fn within(self, want: Self, tol: Self::Scalar) -> bool;

    /// Whether every component is within `tol` of the matching one in `want`,
    /// scaled by that wanted component's magnitude when it exceeds 1.
    fn within_rel(self, want: Self, tol: Self::Scalar) -> bool;
}

macro_rules! approx_float {
    ($t:ty) => {
        impl Approx for $t {
            type Scalar = $t;

            fn within(self, want: Self, tol: $t) -> bool {
                (self - want).abs() <= tol
            }

            fn within_rel(self, want: Self, tol: $t) -> bool {
                (self - want).abs() <= tol * want.abs().max(1.0)
            }
        }
    };
}

approx_float!(f32);
approx_float!(f64);

impl<T: Approx, const N: usize> Approx for [T; N] {
    type Scalar = T::Scalar;

    fn within(self, want: Self, tol: T::Scalar) -> bool {
        self.into_iter().zip(want).all(|(a, b)| a.within(b, tol))
    }

    fn within_rel(self, want: Self, tol: T::Scalar) -> bool {
        self.into_iter()
            .zip(want)
            .all(|(a, b)| a.within_rel(b, tol))
    }
}

/// Whether `got` is within `tol` of `want` in every component.
pub fn close<T: Approx>(got: T, want: T, tol: T::Scalar) -> bool {
    got.within(want, tol)
}

/// Whether `got` is within `tol` of `want` in every component, the tolerance
/// growing with the wanted magnitude past 1.
pub fn close_rel<T: Approx>(got: T, want: T, tol: T::Scalar) -> bool {
    got.within_rel(want, tol)
}

/// Assert [`close`], naming both values and the tolerance.
///
/// # Panics
///
/// When `got` is not within `tol` of `want`.
#[track_caller]
pub fn assert_close<T: Approx>(got: T, want: T, tol: T::Scalar) {
    assert!(
        close(got, want, tol),
        "got {got:?}, want {want:?} within {tol:?}"
    );
}

/// Assert [`close_rel`], naming both values and the tolerance.
///
/// # Panics
///
/// When `got` is not within the scaled `tol` of `want`.
#[track_caller]
pub fn assert_close_rel<T: Approx>(got: T, want: T, tol: T::Scalar) {
    assert!(
        close_rel(got, want, tol),
        "got {got:?}, want {want:?} within {tol:?} relative"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scalar_is_close_up_to_the_tolerance() {
        assert!(close(1.0_f32, 1.0005, 1e-3));
        assert!(close(0.5_f64, 0.75, 0.25));
        assert!(!close(1.0_f32, 1.01, 1e-3));
    }

    #[test]
    fn every_component_must_be_close() {
        assert!(close([1.0_f32, 2.0, 3.0], [1.0, 2.0, 3.0001], 1e-3));
        assert!(!close([1.0_f32, 2.1, 3.0], [1.0, 2.0, 3.0], 1e-3));
    }

    #[test]
    fn nested_arrays_compare_every_entry() {
        let a = [[1.0_f32, 0.0], [0.0, 1.0]];
        assert!(close(a, [[1.0, 0.0], [0.0, 1.00001]], 1e-4));
        assert!(!close(a, [[1.0, 0.0], [0.5, 1.0]], 1e-4));
    }

    #[test]
    fn nan_is_never_close() {
        assert!(!close(f32::NAN, f32::NAN, 1.0));
        assert!(!close([0.0, f64::NAN], [0.0, 0.0], 1.0));
        assert!(!close_rel(f32::NAN, 1.0, 1.0));
    }

    #[test]
    fn the_relative_tolerance_scales_past_one() {
        assert!(close_rel(1000.5_f32, 1000.0, 1e-3));
        assert!(!close(1000.5_f32, 1000.0, 1e-3));
        assert!(!close_rel(0.01_f32, 0.0, 1e-3));
    }

    #[test]
    fn the_assertions_pass_on_close_values() {
        assert_close([1.0_f32, 2.0], [1.0, 2.0], 0.0);
        assert_close_rel(2.0e6_f64, 2.0e6 + 1.0, 1e-6);
    }

    #[test]
    #[should_panic(expected = "within")]
    fn assert_close_names_the_values_when_they_differ() {
        assert_close(1.0_f32, 2.0, 0.5);
    }

    #[test]
    #[should_panic(expected = "relative")]
    fn assert_close_rel_names_the_values_when_they_differ() {
        assert_close_rel(10.0_f32, 20.0, 1e-3);
    }
}
