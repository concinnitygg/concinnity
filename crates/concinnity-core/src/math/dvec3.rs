//! Double-precision 3-vector math, for positions an f32 runs out of digits
//! for: a planet tens of kilometers across placed to the centimeter, or a
//! path stepped across one. The f32 twins are in [`vec3`](super::vec3).

/// A double-precision 3-vector.
pub type DVec3 = [f64; 3];

/// Component-wise sum.
pub fn add(a: DVec3, b: DVec3) -> DVec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Component-wise difference.
pub fn sub(a: DVec3, b: DVec3) -> DVec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `v` times `s`.
pub fn scale(v: DVec3, s: f64) -> DVec3 {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Dot product.
pub fn dot(a: DVec3, b: DVec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product, right-handed.
pub fn cross(a: DVec3, b: DVec3) -> DVec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Euclidean length.
pub fn length(v: DVec3) -> f64 {
    libm::sqrt(dot(v, v))
}

/// `v` scaled to unit length, or `fallback` when it has no direction.
pub fn normalize_or(v: DVec3, fallback: DVec3) -> DVec3 {
    let len = length(v);
    if len > 1e-300 && len.is_finite() {
        scale(v, 1.0 / len)
    } else {
        fallback
    }
}

/// An f32 vector widened to double precision.
pub fn from_f32(v: [f32; 3]) -> DVec3 {
    [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]
}

/// A double-precision vector narrowed to f32.
pub fn to_f32(v: DVec3) -> [f32; 3] {
    [v[0] as f32, v[1] as f32, v[2] as f32]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_basics_agree_with_hand_values() {
        let (a, b) = ([1.0, 2.0, 3.0], [4.0, -5.0, 6.0]);
        assert_eq!(add(a, b), [5.0, -3.0, 9.0]);
        assert_eq!(sub(a, b), [-3.0, 7.0, -3.0]);
        assert_eq!(scale(a, 2.0), [2.0, 4.0, 6.0]);
        assert_eq!(dot(a, b), 12.0);
        assert_eq!(cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]);
        assert_eq!(length([3.0, 4.0, 12.0]), 13.0);
    }

    #[test]
    fn normalizing_falls_back_without_a_direction() {
        assert_eq!(
            normalize_or([0.0, 0.0, 5.0], [1.0, 0.0, 0.0]),
            [0.0, 0.0, 1.0]
        );
        assert_eq!(normalize_or([0.0; 3], [1.0, 0.0, 0.0]), [1.0, 0.0, 0.0]);
        assert_eq!(
            normalize_or([f64::NAN; 3], [0.0, 1.0, 0.0]),
            [0.0, 1.0, 0.0]
        );
    }

    // A planet-sized offset keeps a centimeter that an f32 drops, and narrows
    // back to the nearest f32.
    #[test]
    fn double_precision_keeps_what_f32_drops() {
        let far = add([50_000_000.0, 0.0, 0.0], [0.01, 0.0, 0.0]);
        assert!((far[0] - 50_000_000.01).abs() < 1e-6);
        assert_eq!(to_f32(far), [50_000_000.0, 0.0, 0.0]);
        assert_eq!(from_f32([0.5, -2.0, 8.0]), [0.5, -2.0, 8.0]);
    }
}
