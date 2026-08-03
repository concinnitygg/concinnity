// src/gfx/camera.rs
//
// Pure math for a right-handed first-person camera. All functions are
// stateless -- callers own position, yaw, and pitch directly (e.g. on
// Camera3D) and pass them in as needed.

use crate::geometry::vec3::cross;

// Floor applied to a viewport aspect ratio before it reaches a shader, so a
// zero-height viewport cannot divide by zero in the ray reconstruction.
pub const MIN_ASPECT: f32 = 1.0e-3;

// The two view-ray reconstruction terms every screen-space pass sends its
// shader: the half-FOV tangent and the floored aspect.
pub fn view_ray_scale(fov_y_radians: f32, aspect: f32) -> (f32, f32) {
    ((fov_y_radians * 0.5).tan(), aspect.max(MIN_ASPECT))
}

// Camera-to-world from the view rotation and the world camera position. The
// rotation already lives in `inv_view_rot`'s 3x3; only the translation column
// has to be filled in.
pub fn camera_to_world(inv_view_rot: [[f32; 4]; 4], cam_pos: [f32; 3]) -> [[f32; 4]; 4] {
    let mut inv_view = inv_view_rot;
    inv_view[3] = [cam_pos[0], cam_pos[1], cam_pos[2], 1.0];
    inv_view
}

// Build a column-major view matrix from position, yaw, and pitch.
//
// Convention matches the GLSL mat4 layout used by the shader UBOs:
// view[column][row], right-handed, Y-up.

pub fn view_matrix(position: [f32; 3], yaw: f32, pitch: f32) -> [[f32; 4]; 4] {
    let (sin_yaw, cos_yaw) = yaw.sin_cos();
    let (sin_pitch, cos_pitch) = pitch.sin_cos();

    let fwd = [-sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch];
    let right = normalize(cross(fwd, [0.0, 1.0, 0.0]));
    let up = cross(right, fwd);

    let [rx, ry, rz] = right;
    let [ux, uy, uz] = up;
    let [fx, fy, fz] = fwd;
    let [px, py, pz] = position;

    [
        [rx, ux, -fx, 0.0],
        [ry, uy, -fy, 0.0],
        [rz, uz, -fz, 0.0],
        [
            -(rx * px + ry * py + rz * pz),
            -(ux * px + uy * py + uz * pz),
            fx * px + fy * py + fz * pz,
            1.0,
        ],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-7 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[test]
    fn view_matrix_at_origin_with_zero_angles_is_identity() {
        let m = view_matrix([0.0, 0.0, 0.0], 0.0, 0.0);
        let id = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        for c in 0..4 {
            for r in 0..4 {
                assert!(
                    (m[c][r] - id[c][r]).abs() < 1e-5,
                    "m[{c}][{r}] = {} expected {}",
                    m[c][r],
                    id[c][r]
                );
            }
        }
    }

    #[test]
    fn view_matrix_basis_is_orthonormal() {
        let m = view_matrix([1.0, 2.0, 3.0], 0.7, -0.3);
        // The 3x3 rotation part's rows are the right / up / -forward basis; each
        // is unit length and the three are mutually orthogonal.
        let basis = |r: usize| [m[0][r], m[1][r], m[2][r]];
        for r in 0..3 {
            assert!(
                (dot(basis(r), basis(r)) - 1.0).abs() < 1e-4,
                "row {r} not unit"
            );
        }
        assert!(dot(basis(0), basis(1)).abs() < 1e-4);
        assert!(dot(basis(0), basis(2)).abs() < 1e-4);
        assert!(dot(basis(1), basis(2)).abs() < 1e-4);
        assert_eq!(m[0][3], 0.0);
        assert_eq!(m[3][3], 1.0);
    }

    #[test]
    fn normalize_falls_back_for_a_zero_vector() {
        assert_eq!(normalize([0.0, 0.0, 0.0]), [0.0, 0.0, 1.0]);
        let n = normalize([3.0, 0.0, 0.0]);
        assert!((n[0] - 1.0).abs() < 1e-6);
        assert_eq!([n[1], n[2]], [0.0, 0.0]);
    }

    #[test]
    fn cross_is_right_handed() {
        assert_eq!(cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]);
    }
}
