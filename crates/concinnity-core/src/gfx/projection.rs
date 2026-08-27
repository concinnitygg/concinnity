//! The engine's one perspective projection, in the convention every backend
//! already agreed on: right-handed, looking down `-z`, with depth mapped to
//! `[0, 1]` after the perspective divide. Metal, Vulkan, and DirectX all sample
//! depth that way, so the same matrix is valid for all three; Vulkan and D3D12
//! compensate for their Y-down NDC with a negative-height viewport rather than
//! by flipping the projection.

use crate::gfx::transform::Mat4;
use crate::math::tan;

// Floor applied to the half-FOV tangent. A zero or near-zero vertical FOV would
// otherwise divide by zero and fill the matrix with infinities.
const MIN_HALF_FOV_TAN: f32 = 1.0e-6;

/// Right-handed perspective projection with depth in `[0, 1]`. `fov_y_radians`
/// is the full vertical field of view; `aspect` is width over height.
pub fn perspective_rh(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let ys = 1.0 / tan(fov_y_radians * 0.5).max(MIN_HALF_FOV_TAN);
    let xs = ys / aspect;
    let zs = far / (near - far);
    [
        [xs, 0.0, 0.0, 0.0],
        [0.0, ys, 0.0, 0.0],
        [0.0, 0.0, zs, -1.0],
        [0.0, 0.0, zs * near, 0.0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transform(m: Mat4, p: [f32; 3]) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for (row, o) in out.iter_mut().enumerate() {
            *o = m[0][row] * p[0] + m[1][row] * p[1] + m[2][row] * p[2] + m[3][row];
        }
        out
    }

    #[test]
    fn near_and_far_map_to_zero_and_one() {
        let p = perspective_rh(75.0f32.to_radians(), 1.6, 0.1, 500.0);
        let near = transform(p, [0.0, 0.0, -0.1]);
        let far = transform(p, [0.0, 0.0, -500.0]);
        assert!((near[2] / near[3]).abs() < 1e-4);
        assert!((far[2] / far[3] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn the_frustum_edge_lands_on_the_ndc_boundary() {
        // At a 90 degree vertical FOV and square aspect the frustum edge sits
        // at x = -z, so an edge point projects exactly onto NDC x = 1.
        let p = perspective_rh(90.0f32.to_radians(), 1.0, 0.1, 50.0);
        let edge = transform(p, [10.0, 0.0, -10.0]);
        assert!((edge[0] / edge[3] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn a_degenerate_fov_stays_finite() {
        let p = perspective_rh(0.0, 1.0, 0.1, 100.0);
        assert!(p.iter().flatten().all(|v| v.is_finite()));
    }

    // The backends and the shadow builders each reached this matrix their own
    // way before it moved here: through std's tangent rather than the libm shim,
    // and (in the shadow builders) through a different arrangement of the same
    // algebra. Reassociating a float product is not free and the two tangents
    // need not agree to the last bit, so pin the gap: every element must land
    // within one ulp of the other ordering.
    #[test]
    fn the_reassociated_form_agrees_to_within_one_ulp() {
        fn other(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
            let t = f32::tan(fov_y_radians * 0.5).max(MIN_HALF_FOV_TAN);
            let fmn = far - near;
            [
                [1.0 / (aspect * t), 0.0, 0.0, 0.0],
                [0.0, 1.0 / t, 0.0, 0.0],
                [0.0, 0.0, -far / fmn, -1.0],
                [0.0, 0.0, -(far * near) / fmn, 0.0],
            ]
        }
        for fov_deg in [10.0f32, 45.0, 60.0, 75.0, 90.0, 140.0, 179.0] {
            for aspect in [0.5f32, 1.0, 1.6, 2.35, 3.0] {
                for near in [0.01f32, 0.1, 1.0] {
                    for far in [10.0f32, 500.0, 10_000.0] {
                        let fov = fov_deg.to_radians();
                        let a = perspective_rh(fov, aspect, near, far);
                        let b = other(fov, aspect, near, far);
                        for col in 0..4 {
                            for row in 0..4 {
                                let ulps = (i64::from(a[col][row].to_bits())
                                    - i64::from(b[col][row].to_bits()))
                                .abs();
                                assert!(
                                    ulps <= 1,
                                    "fov={fov_deg} aspect={aspect} near={near} far={far} \
                                     [{col}][{row}]: {} vs {} ({ulps} ulps)",
                                    a[col][row],
                                    b[col][row]
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
