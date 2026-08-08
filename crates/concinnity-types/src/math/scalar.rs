// The f32 transcendentals `core` does not carry, forwarded to libm.
//
// Free functions rather than an extension trait: `cargo test` links the test
// harness against std, whose inherent f32 methods win method resolution over
// any trait, so a trait would go unused in exactly the build that checks it.

/// Square root.
pub fn sqrt(x: f32) -> f32 {
    libm::sqrtf(x)
}

/// Sine of an angle in radians.
pub fn sin(x: f32) -> f32 {
    libm::sinf(x)
}

/// Cosine of an angle in radians.
pub fn cos(x: f32) -> f32 {
    libm::cosf(x)
}

/// Sine and cosine of an angle in radians.
pub fn sin_cos(x: f32) -> (f32, f32) {
    libm::sincosf(x)
}

/// Tangent of an angle in radians.
pub fn tan(x: f32) -> f32 {
    libm::tanf(x)
}

/// Arc sine, in radians.
pub fn asin(x: f32) -> f32 {
    libm::asinf(x)
}

/// Arc cosine, in radians.
pub fn acos(x: f32) -> f32 {
    libm::acosf(x)
}

/// Arc tangent of `y / x`, in radians, using both signs to pick the quadrant.
pub fn atan2(y: f32, x: f32) -> f32 {
    libm::atan2f(y, x)
}

/// Largest integer at or below `x`.
pub fn floor(x: f32) -> f32 {
    libm::floorf(x)
}

/// Fractional part, keeping the sign of `x`.
pub fn fract(x: f32) -> f32 {
    x - libm::truncf(x)
}

/// `e` raised to `x`.
pub fn exp(x: f32) -> f32 {
    libm::expf(x)
}

/// 2 raised to `x`.
pub fn exp2(x: f32) -> f32 {
    libm::exp2f(x)
}

/// `x` raised to `n`.
pub fn powf(x: f32, n: f32) -> f32 {
    libm::powf(x, n)
}

/// Least nonnegative remainder of `x (mod rhs)`.
pub fn rem_euclid(x: f32, rhs: f32) -> f32 {
    let r = libm::fmodf(x, rhs);
    if r < 0.0 { r + libm::fabsf(rhs) } else { r }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Each must agree with the std implementation it stands in for: the compute
    // crate above this one is std-linked and calls std's version on the same
    // values, so a divergence would be a seam between the two halves of a split
    // that is supposed to be behaviour-preserving.
    #[track_caller]
    fn approx(got: f32, want: f32) {
        assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
    }

    #[test]
    fn transcendentals_match_std() {
        for &x in &[0.0f32, 0.5, 1.0, 2.5, 7.0] {
            approx(sqrt(x), f32::sqrt(x));
            approx(exp(x), f32::exp(x));
            approx(exp2(x), f32::exp2(x));
            approx(powf(x, 1.5), f32::powf(x, 1.5));
        }
        for &x in &[-2.5f32, -0.75, 0.0, 0.3, 1.2, 3.0] {
            approx(sin(x), f32::sin(x));
            approx(cos(x), f32::cos(x));
            approx(tan(x), f32::tan(x));
            approx(floor(x), f32::floor(x));
            approx(fract(x), f32::fract(x));
            approx(atan2(x, 2.0), f32::atan2(x, 2.0));
            let (s, c) = sin_cos(x);
            approx(s, f32::sin(x));
            approx(c, f32::cos(x));
        }
        for &x in &[-1.0f32, -0.5, 0.0, 0.5, 1.0] {
            approx(asin(x), f32::asin(x));
            approx(acos(x), f32::acos(x));
        }
    }

    // The one function with no libm counterpart, so the formula is ours: a
    // negative dividend must come back in [0, |rhs|), not negative like fmod.
    #[test]
    fn rem_euclid_matches_std_across_signs() {
        for &(a, b) in &[
            (7.5f32, 2.0f32),
            (-7.5, 2.0),
            (7.5, -2.0),
            (-7.5, -2.0),
            (0.0, 3.0),
            (-0.25, 1.0),
        ] {
            approx(rem_euclid(a, b), f32::rem_euclid(a, b));
            assert!(rem_euclid(a, b) >= 0.0, "{a} rem_euclid {b}");
        }
    }
}
