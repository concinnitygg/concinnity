//! The crate's shared 3-component vector math. Every module that needs a dot,
//! cross, normalize, or component-wise op reaches for these rather than
//! redeclaring them.
//!
//! Normalization takes its degenerate-input rule from the caller, because it
//! differs by site: [`try_normalize`] hands back `None` below a caller-chosen
//! length, [`normalize_or`] substitutes a fallback direction, and
//! [`normalize_clamped`] floors the length so the result stays finite.

use crate::math::sqrt;

/// Dot product.
pub fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product.
pub fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Component-wise difference, `a - b`.
pub fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Component-wise sum, `a + b`.
pub fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Every component scaled by `s`.
pub fn scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Squared Euclidean length.
pub fn length_sq(v: [f32; 3]) -> f32 {
    dot(v, v)
}

/// Euclidean length.
pub fn length(v: [f32; 3]) -> f32 {
    sqrt(dot(v, v))
}

/// Unit-length `v`, or `None` when its length is not a finite value above
/// `min_len`.
pub fn try_normalize(v: [f32; 3], min_len: f32) -> Option<[f32; 3]> {
    let len = length(v);
    (len > min_len && len.is_finite()).then(|| [v[0] / len, v[1] / len, v[2] / len])
}

/// Unit-length `v`, or `fallback` when its length is not a finite value above
/// `min_len`.
pub fn normalize_or(v: [f32; 3], min_len: f32, fallback: [f32; 3]) -> [f32; 3] {
    try_normalize(v, min_len).unwrap_or(fallback)
}

/// `v` divided by its length floored at `min_len`: unit length for any vector
/// longer than `min_len`, and a short but finite one below it.
pub fn normalize_clamped(v: [f32; 3], min_len: f32) -> [f32; 3] {
    let len = length(v).max(min_len);
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Component-wise linear interpolation from `a` to `b`.
pub fn lerp(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Accumulate `src` into `dst` in place. Used by the smooth-normal passes, which
/// sum every incident face normal per vertex before normalizing once.
pub fn vec3_add(dst: &mut [f32; 3], src: [f32; 3]) {
    dst[0] += src[0];
    dst[1] += src[1];
    dst[2] += src[2];
}

/// Unit-length `n`, falling back to `+Y` when it is too short to have a
/// direction.
pub fn vec3_normalize(n: [f32; 3]) -> [f32; 3] {
    normalize_or(n, 1e-6, [0.0, 1.0, 0.0])
}

/// Newell-style face normal from three CCW positions. Shared with the cook
/// generators' smooth-normal pass.
pub fn vec3_face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    vec3_normalize(cross(sub(b, a), sub(c, a)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_is_right_handed() {
        assert_eq!(cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]);
        assert_eq!(cross([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn dot_and_length_agree() {
        let v = [3.0, 4.0, 0.0];
        assert_eq!(dot(v, v), 25.0);
        assert_eq!(length_sq(v), 25.0);
        assert_eq!(length(v), 5.0);
    }

    #[test]
    fn try_normalize_refuses_a_vector_no_longer_than_the_threshold() {
        assert_eq!(try_normalize([0.0, 3.0, 4.0], 1e-6), Some([0.0, 0.6, 0.8]));
        assert_eq!(try_normalize([0.0; 3], 1e-6), None);
        assert_eq!(try_normalize([1e-5, 0.0, 0.0], 1e-5), None);
        assert_eq!(try_normalize([2e-5, 0.0, 0.0], 1e-5), Some([1.0, 0.0, 0.0]));
        // A zero threshold still refuses the zero vector.
        assert_eq!(try_normalize([0.0; 3], 0.0), None);
        assert_eq!(try_normalize([f32::NAN, 0.0, 0.0], 1e-6), None);
        assert_eq!(try_normalize([f32::INFINITY, 0.0, 0.0], 1e-6), None);
    }

    #[test]
    fn normalize_or_substitutes_the_fallback() {
        let z = [0.0, 0.0, 1.0];
        assert_eq!(normalize_or([0.0, 0.0, 0.0], 1e-6, z), z);
        assert_eq!(normalize_or([2.0, 0.0, 0.0], 1e-6, z), [1.0, 0.0, 0.0]);
        assert_eq!(normalize_or([1e-7, 0.0, 0.0], 1e-6, z), z);
        assert_eq!(normalize_or([1e-7, 0.0, 0.0], 1e-9, z), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn normalize_clamped_stays_finite_on_a_degenerate_vector() {
        assert_eq!(normalize_clamped([0.0, 0.0, 4.0], 1e-6), [0.0, 0.0, 1.0]);
        assert_eq!(normalize_clamped([0.0; 3], 1e-6), [0.0; 3]);
        let tiny = normalize_clamped([1e-8, 0.0, 0.0], 1e-6);
        assert!((tiny[0] - 1e-2).abs() < 1e-6, "{tiny:?}");
    }

    #[test]
    fn component_ops_are_component_wise() {
        assert_eq!(add([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), [5.0, 7.0, 9.0]);
        assert_eq!(sub([4.0, 5.0, 6.0], [1.0, 2.0, 3.0]), [3.0, 3.0, 3.0]);
        assert_eq!(scale([1.0, 2.0, 3.0], 2.0), [2.0, 4.0, 6.0]);
        assert_eq!(lerp([0.0, 0.0, 0.0], [2.0, 4.0, 6.0], 0.5), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn vec3_add_accumulates_in_place() {
        let mut acc = [1.0, 1.0, 1.0];
        vec3_add(&mut acc, [1.0, 2.0, 3.0]);
        vec3_add(&mut acc, [1.0, 2.0, 3.0]);
        assert_eq!(acc, [3.0, 5.0, 7.0]);
    }

    #[test]
    fn normalize_falls_back_on_a_degenerate_vector() {
        assert_eq!(vec3_normalize([0.0, 0.0, 0.0]), [0.0, 1.0, 0.0]);
        assert_eq!(vec3_normalize([0.0, 0.0, 2.0]), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn face_normal_of_a_ccw_triangle_points_up() {
        let n = vec3_face_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(n[1] > 0.99, "expected +Y, got {n:?}");
    }
}
