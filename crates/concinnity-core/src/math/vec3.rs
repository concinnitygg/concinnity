//! The crate's shared 3-component vector math. Every module that needs a dot,
//! cross, or component-wise op reaches for these rather than redeclaring them.
//!
//! Normalisation deliberately stays with its caller: the degenerate-input rule
//! differs by site (a fallback axis vs `None` vs a clamped length), and folding
//! those together would change behavior at grazing inputs.

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

/// Euclidean length.
pub fn length(v: [f32; 3]) -> f32 {
    sqrt(dot(v, v))
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
/// sum every incident face normal per vertex before normalising once.
pub fn vec3_add(dst: &mut [f32; 3], src: [f32; 3]) {
    dst[0] += src[0];
    dst[1] += src[1];
    dst[2] += src[2];
}

/// Unit-length `n`, falling back to `+Y` when it is too short to have a
/// direction.
pub fn vec3_normalise(n: [f32; 3]) -> [f32; 3] {
    let len = length(n);
    if len < 1e-6 {
        [0.0, 1.0, 0.0]
    } else {
        scale(n, 1.0 / len)
    }
}

/// Newell-style face normal from three CCW positions. Shared with the cook
/// generators' smooth-normal pass.
pub fn vec3_face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    vec3_normalise(cross(sub(b, a), sub(c, a)))
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
        assert_eq!(length(v), 5.0);
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
    fn normalise_falls_back_on_a_degenerate_vector() {
        assert_eq!(vec3_normalise([0.0, 0.0, 0.0]), [0.0, 1.0, 0.0]);
        assert_eq!(vec3_normalise([0.0, 0.0, 2.0]), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn face_normal_of_a_ccw_triangle_points_up() {
        let n = vec3_face_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(n[1] > 0.99, "expected +Y, got {n:?}");
    }
}
