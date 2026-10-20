// Interpolation helpers shared by the bakes, the camera track and the render prep.

/// Linear interpolation from `a` to `b`; `t` is not clamped.
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Component-wise [`lerp`] over a fixed-size array.
pub fn lerp_array<const N: usize>(a: [f32; N], b: [f32; N], t: f32) -> [f32; N] {
    core::array::from_fn(|i| lerp(a[i], b[i], t))
}

/// Hermite smoothstep of `x` between `edge0` and `edge1`, clamped to `[0, 1]`.
///
/// Equal edges step at the edge: `0` below it, `1` at or above it.
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge0 == edge1 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lerp_hits_both_ends_and_the_midpoint() {
        assert_eq!(lerp(2.0, 6.0, 0.0), 2.0);
        assert_eq!(lerp(2.0, 6.0, 1.0), 6.0);
        assert_eq!(lerp(2.0, 6.0, 0.5), 4.0);
    }

    #[test]
    fn lerp_array_interpolates_each_component() {
        assert_eq!(lerp_array([0.0, 10.0], [4.0, 20.0], 0.25), [1.0, 12.5]);
    }

    #[test]
    fn smoothstep_clamps_outside_the_edges() {
        assert_eq!(smoothstep(1.0, 3.0, 0.0), 0.0);
        assert_eq!(smoothstep(1.0, 3.0, 4.0), 1.0);
        assert_eq!(smoothstep(1.0, 3.0, 2.0), 0.5);
    }

    #[test]
    fn smoothstep_with_equal_edges_steps_at_the_edge() {
        assert_eq!(smoothstep(2.0, 2.0, 1.0), 0.0);
        assert_eq!(smoothstep(2.0, 2.0, 2.0), 1.0);
        assert_eq!(smoothstep(2.0, 2.0, 3.0), 1.0);
    }
}
