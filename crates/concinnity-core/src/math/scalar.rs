// The f32 transcendentals `core` does not carry, forwarded to libm.
//
// Free functions rather than an extension trait: `cargo test` links the test
// harness against std, whose inherent f32 methods win method resolution over
// any trait, so a trait would go unused in exactly the build that checks it.

/// Square root.
pub fn sqrt(x: f32) -> f32 {
    libm::sqrtf(x)
}

/// Length of the hypotenuse of a right triangle with legs `x` and `y`,
/// without the intermediate overflow of `sqrt(x * x + y * y)`.
pub fn hypot(x: f32, y: f32) -> f32 {
    libm::hypotf(x, y)
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

/// Smallest integer at or above `x`.
pub fn ceil(x: f32) -> f32 {
    libm::ceilf(x)
}

/// Nearest integer, with halves rounded away from zero.
pub fn round(x: f32) -> f32 {
    libm::roundf(x)
}

/// Integer part, discarding the fraction and keeping the sign of `x`.
pub fn trunc(x: f32) -> f32 {
    libm::truncf(x)
}

/// Fractional part, keeping the sign of `x`.
pub fn fract(x: f32) -> f32 {
    x - trunc(x)
}

/// `e` raised to `x`.
pub fn exp(x: f32) -> f32 {
    libm::expf(x)
}

/// 2 raised to `x`.
pub fn exp2(x: f32) -> f32 {
    libm::exp2f(x)
}

/// Natural logarithm.
pub fn ln(x: f32) -> f32 {
    libm::logf(x)
}

/// Base-2 logarithm.
pub fn log2(x: f32) -> f32 {
    libm::log2f(x)
}

/// `x` raised to `n`.
pub fn powf(x: f32, n: f32) -> f32 {
    libm::powf(x, n)
}

/// `x` raised to the integer power `n`, by squaring. Matches the expansion
/// `f32::powi` lowers to rather than routing through [`powf`], which would
/// round differently.
pub fn powi(x: f32, n: i32) -> f32 {
    let mut base = x;
    let mut exp = n;
    let mut acc = 1.0;
    loop {
        if exp & 1 != 0 {
            acc *= base;
        }
        // Truncating division walks the magnitude's bits for a negative `n`
        // too, so the reciprocal below is the only place the sign is read.
        exp /= 2;
        if exp == 0 {
            break;
        }
        base *= base;
    }
    if n < 0 { 1.0 / acc } else { acc }
}

/// `x * y + z` with a single rounding.
pub fn mul_add(x: f32, y: f32, z: f32) -> f32 {
    libm::fmaf(x, y, z)
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

    // Relative form, for the functions whose range runs well past the absolute
    // tolerance above (powi of a large base, hypot of large legs).
    #[track_caller]
    fn approx_rel(got: f32, want: f32) {
        let scale = want.abs().max(1.0);
        assert!((got - want).abs() <= 1e-6 * scale, "got {got}, want {want}");
    }

    #[test]
    fn transcendentals_match_std() {
        for &x in &[0.0f32, 0.5, 1.0, 2.5, 7.0] {
            approx(sqrt(x), f32::sqrt(x));
            approx(exp(x), f32::exp(x));
            approx(exp2(x), f32::exp2(x));
            approx(powf(x, 1.5), f32::powf(x, 1.5));
            approx_rel(hypot(x, 3.0), f32::hypot(x, 3.0));
        }
        for &x in &[0.25f32, 0.5, 1.0, 2.5, 7.0, 1000.0] {
            approx(ln(x), f32::ln(x));
            approx(log2(x), f32::log2(x));
        }
        for &x in &[-2.5f32, -0.75, 0.0, 0.3, 1.2, 3.0] {
            approx(sin(x), f32::sin(x));
            approx(cos(x), f32::cos(x));
            approx(tan(x), f32::tan(x));
            approx(floor(x), f32::floor(x));
            approx(ceil(x), f32::ceil(x));
            approx(round(x), f32::round(x));
            approx(trunc(x), f32::trunc(x));
            approx(fract(x), f32::fract(x));
            approx(atan2(x, 2.0), f32::atan2(x, 2.0));
            approx(mul_add(x, 2.5, -1.25), f32::mul_add(x, 2.5, -1.25));
            let (s, c) = sin_cos(x);
            approx(s, f32::sin(x));
            approx(c, f32::cos(x));
        }
        for &x in &[-1.0f32, -0.5, 0.0, 0.5, 1.0] {
            approx(asin(x), f32::asin(x));
            approx(acos(x), f32::acos(x));
        }
    }

    // The rounding family splits on the half and on the sign, which is exactly
    // where floor / ceil / round / trunc stop agreeing with one another.
    #[test]
    fn rounding_matches_std_at_the_halves_and_across_signs() {
        for &x in &[-2.5f32, -1.5, -0.5, -0.25, 0.0, 0.25, 0.5, 1.5, 2.5] {
            approx(floor(x), f32::floor(x));
            approx(ceil(x), f32::ceil(x));
            approx(round(x), f32::round(x));
            approx(trunc(x), f32::trunc(x));
            approx(fract(x), f32::fract(x));
        }
    }

    // powi has no libm counterpart, so the squaring loop is ours: it must agree
    // with std's across both signs of the exponent and at the zero exponent,
    // where the accumulator alone decides the answer.
    #[test]
    fn powi_matches_std_across_exponent_signs() {
        for &x in &[-3.0f32, -0.5, 0.5, 1.0, 2.0, 7.5] {
            for n in -6i32..=6 {
                approx_rel(powi(x, n), f32::powi(x, n));
            }
        }
        assert_eq!(powi(0.0, 0), 1.0);
        assert_eq!(powi(5.0, 1), 5.0);
    }

    // mul_add must round once, not twice. `(2^23 + 1)^2` needs 47 bits, so the
    // rounded product loses the trailing 1 and the unfused expression cancels
    // to zero; only the fused form keeps it.
    #[test]
    fn mul_add_rounds_once() {
        let x = 8_388_609.0f32;
        let z = -(x * x);
        assert_eq!(mul_add(x, x, z), f32::mul_add(x, x, z));
        assert_eq!(mul_add(x, x, z), 1.0);
        assert_eq!(x * x + z, 0.0);
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
