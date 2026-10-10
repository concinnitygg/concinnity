//! Pure math for a right-handed first-person camera. All functions are
//! stateless -- callers own position, yaw, and pitch directly (e.g. on
//! Camera3D) and pass them in as needed.

use crate::math::vec3::{cross, normalize_or};
use crate::math::{sin_cos, tan};

// Floor applied to a viewport aspect ratio before it reaches a shader, so a
// zero-height viewport cannot divide by zero in the ray reconstruction.
pub(crate) const MIN_ASPECT: f32 = 1.0e-3;

// The two view-ray reconstruction terms every screen-space pass sends its
// shader: the half-FOV tangent and the floored aspect.
pub(crate) fn view_ray_scale(fov_y_radians: f32, aspect: f32) -> (f32, f32) {
    (tan(fov_y_radians * 0.5), aspect.max(MIN_ASPECT))
}

// Camera-to-world from the view rotation and the world camera position. The
// rotation already lives in `inv_view_rot`'s 3x3; only the translation column
// has to be filled in.
pub(crate) fn camera_to_world(inv_view_rot: [[f32; 4]; 4], cam_pos: [f32; 3]) -> [[f32; 4]; 4] {
    let mut inv_view = inv_view_rot;
    inv_view[3] = [cam_pos[0], cam_pos[1], cam_pos[2], 1.0];
    inv_view
}

/// Build a column-major view matrix from position, yaw, and pitch.
///
/// Convention matches the GLSL mat4 layout used by the shader UBOs:
/// `view[column][row]`, right-handed, Y-up.
pub fn view_matrix(position: [f32; 3], yaw: f32, pitch: f32) -> [[f32; 4]; 4] {
    let (sin_yaw, cos_yaw) = sin_cos(yaw);
    let (sin_pitch, cos_pitch) = sin_cos(pitch);

    let fwd = [-sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch];
    let right = normalize_or(cross(fwd, [0.0, 1.0, 0.0]), 1e-7, [0.0, 0.0, 1.0]);
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

/// [`view_matrix`] with yaw turning about `up` and pitch measured from the
/// plane square to it, rather than about and from `+Y`: the `+Y` camera
/// tilted onto `up` along the shortest arc. A camera standing on a planet sees
/// its horizon level this way wherever it stands.
pub fn view_matrix_about(position: [f32; 3], yaw: f32, pitch: f32, up: [f32; 3]) -> [[f32; 4]; 4] {
    let m = view_matrix([0.0; 3], yaw, pitch);
    // The view's rows are the camera's right, up and back axes.
    let row = |r: usize| [m[0][r], m[1][r], m[2][r]];
    let [right, cam_up, back] = [0, 1, 2].map(|r| tilt_to_up(up, row(r)));
    let axes = [right, cam_up, back];
    let mut out = [[0.0; 4]; 4];
    for (r, axis) in axes.iter().enumerate() {
        for c in 0..3 {
            out[c][r] = axis[c];
        }
        out[3][r] = -(axis[0] * position[0] + axis[1] * position[1] + axis[2] * position[2]);
    }
    out[3][3] = 1.0;
    out
}

/// `v`, given in a frame whose up is `+Y`, turned into the frame whose up is
/// `up` along the shortest arc between the two. A level direction stays
/// level, and nothing turns about `up` itself.
pub fn tilt_to_up(up: [f32; 3], v: [f32; 3]) -> [f32; 3] {
    let u = normalize_or(up, 1e-7, [0.0, 1.0, 0.0]);
    let c = u[1];
    if c >= 1.0 {
        return v;
    }
    if c <= -1.0 + 1e-6 {
        return [v[0], -v[1], -v[2]];
    }
    // Rodrigues about k = Y x up = (u.z, 0, -u.x): v c + (k x v) + k (k.v) / (1 + c).
    let (kx, kz) = (u[2], -u[0]);
    let kv = kx * v[0] + kz * v[2];
    let f = kv / (1.0 + c);
    [
        v[0] * c - kz * v[1] + kx * f,
        v[1] * c + (kz * v[0] - kx * v[2]),
        v[2] * c + kx * v[1] + kz * f,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::vec3::dot;

    fn close(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < tol)
    }

    // Tilting onto +Y is no tilt at all, so a level world's view is unchanged.
    #[test]
    fn the_view_about_plus_y_is_the_plain_view() {
        let a = view_matrix([1.0, 2.0, 3.0], 0.7, -0.2);
        let b = view_matrix_about([1.0, 2.0, 3.0], 0.7, -0.2, [0.0, 1.0, 0.0]);
        for c in 0..4 {
            for r in 0..4 {
                assert!((a[c][r] - b[c][r]).abs() < 1e-6);
            }
        }
    }

    // The tilt lands +Y on `up` and keeps a direction square to it level.
    #[test]
    fn the_tilt_lands_plus_y_on_up() {
        let up = normalize_or([0.02, 1.0, -0.01], 1e-7, [0.0; 3]);
        assert!(close(tilt_to_up(up, [0.0, 1.0, 0.0]), up, 1e-6));
        let level = tilt_to_up(up, [0.0, 0.0, -1.0]);
        assert!(dot(level, up).abs() < 1e-6);
        assert!((dot(level, level) - 1.0).abs() < 1e-6);
    }

    // A camera with zero pitch looks square to its own up, and its view's up
    // row is that up.
    #[test]
    fn a_level_camera_about_a_tilted_up_looks_square_to_it() {
        let up = normalize_or([0.1, 1.0, 0.05], 1e-7, [0.0; 3]);
        let m = view_matrix_about([5.0, 0.0, -2.0], 1.1, 0.0, up);
        let back = [m[0][2], m[1][2], m[2][2]];
        let view_up = [m[0][1], m[1][1], m[2][1]];
        assert!(dot(back, up).abs() < 1e-6);
        assert!(close(view_up, up, 1e-6));
        // The eye maps to the view's origin.
        let p = [5.0, 0.0, -2.0];
        let eye: [f32; 3] =
            core::array::from_fn(|r| m[0][r] * p[0] + m[1][r] * p[1] + m[2][r] * p[2] + m[3][r]);
        assert!(close(eye, [0.0; 3], 1e-5), "{eye:?}");
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
    fn looking_straight_up_falls_back_to_a_z_right_axis() {
        let m = view_matrix([0.0; 3], 0.0, core::f32::consts::FRAC_PI_2);
        assert_eq!([m[0][0], m[1][0], m[2][0]], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn cross_is_right_handed() {
        assert_eq!(cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]);
    }
}
