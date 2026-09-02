use crate::gfx::transform::quat_to_mat3;
use crate::math::{euler_yxz_deg_from_quat, quat_from_axis_angle};

/// Where the celestial sphere has turned to, published once per tick by
/// [`SkyRotationSystem`](super::SkyRotationSystem) and read by every consumer
/// of the sky.
///
/// A world with no `SkyRotation` never publishes one; a reader that finds none
/// uses [`SkyOrientation::default`], which is the identity.
#[derive(Debug, Clone, Copy)]
pub struct SkyOrientation {
    /// The current angle in degrees, carried so the system can resume from it.
    pub angle_deg: f32,
    // Column-major rotation taking a direction from the sky's baked frame into
    // world space. Held rather than recomputed because every consumer wants it.
    rotation: [[f32; 3]; 3],
}

impl Default for SkyOrientation {
    fn default() -> Self {
        Self {
            angle_deg: 0.0,
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }
}

impl SkyOrientation {
    /// The sample rows of a sky that does not turn: what a world with no
    /// `SkyRotation` uploads, and what a backend holds until one is pushed.
    pub const IDENTITY_ROWS: [[f32; 4]; 3] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];

    /// The orientation `angle_deg` about `axis`.
    ///
    /// The sign is the one a planet's own spin gives its sky: with `axis` along
    /// `+X`, a positive angle carries a body from `+Z` up over `+Y` and down
    /// toward `-Z`. That is a negative turn about the pole in the right-handed
    /// sense, which is what an observer on the surface actually sees.
    pub fn new(axis: [f32; 3], angle_deg: f32) -> Self {
        let q = quat_from_axis_angle(axis, -angle_deg.to_radians());
        Self {
            angle_deg,
            rotation: quat_to_mat3(q),
        }
    }

    /// `dir` carried by the sky's rotation: an authored direction in the baked
    /// frame, answered in world space.
    pub fn rotate(&self, dir: [f32; 3]) -> [f32; 3] {
        let r = &self.rotation;
        core::array::from_fn(|i| r[0][i] * dir[0] + r[1][i] * dir[1] + r[2][i] * dir[2])
    }

    /// The rows a shader multiplies a world-space direction by to reach the
    /// environment cubemap's baked frame: the inverse rotation, one `float4`
    /// per row with an unused `w`.
    pub fn sample_rows(&self) -> [[f32; 4]; 3] {
        let r = &self.rotation;
        core::array::from_fn(|i| [r[i][0], r[i][1], r[i][2], 0.0])
    }

    /// The same rotation as engine Euler degrees, for the transform the
    /// component's entity carries.
    pub fn euler_deg(&self) -> [f32; 3] {
        euler_yxz_deg_from_quat(crate::gfx::transform::quat_from_mat3(self.rotation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-4)
    }

    // The sign convention the whole feature hangs on: a positive angle about
    // the default pole runs a body from behind the viewer, overhead, and down
    // in front of it.
    #[test]
    fn a_positive_angle_carries_a_body_from_plus_z_over_plus_y_to_minus_z() {
        let axis = [1.0, 0.0, 0.0];
        let body = [0.0, 0.0, 1.0];
        assert!(close(SkyOrientation::new(axis, 0.0).rotate(body), body));
        assert!(close(
            SkyOrientation::new(axis, 90.0).rotate(body),
            [0.0, 1.0, 0.0]
        ));
        assert!(close(
            SkyOrientation::new(axis, 180.0).rotate(body),
            [0.0, 0.0, -1.0]
        ));
        // Partway up on the first quarter turn, never below the horizon.
        let quarter = SkyOrientation::new(axis, 45.0).rotate(body);
        assert!(quarter[1] > 0.0 && quarter[2] > 0.0, "{quarter:?}");
    }

    // The shader rows undo the rotation, so a direction carried by the sky and
    // then sampled through the rows lands back where the cube was baked.
    #[test]
    fn the_sample_rows_invert_the_rotation() {
        let sky = SkyOrientation::new([0.3, 1.0, -0.4], 71.0);
        let baked = [0.2, -0.5, 0.84];
        let world = sky.rotate(baked);
        let rows = sky.sample_rows();
        let back: [f32; 3] = core::array::from_fn(|i| {
            rows[i][0] * world[0] + rows[i][1] * world[1] + rows[i][2] * world[2]
        });
        assert!(close(back, baked), "{back:?}");
        assert!(rows.iter().all(|r| r[3] == 0.0));
    }

    // The Euler angles the pivot entity carries describe the same rotation the
    // lights and the cubemap use, so a parented prop tracks its light.
    #[test]
    fn the_euler_angles_describe_the_same_rotation() {
        let sky = SkyOrientation::new([1.0, 0.0, 0.0], 120.0);
        let dir = [0.0, 0.0, 1.0];
        let by_matrix = sky.rotate(dir);
        let m = crate::gfx::transform::trs_matrix([0.0; 3], sky.euler_deg(), [1.0; 3]);
        let by_euler: [f32; 3] =
            core::array::from_fn(|i| m[0][i] * dir[0] + m[1][i] * dir[1] + m[2][i] * dir[2]);
        assert!(close(by_matrix, by_euler), "{by_matrix:?} {by_euler:?}");
    }

    // No rotation is the identity, which is what a world with no SkyRotation
    // renders and lights with.
    #[test]
    fn the_default_orientation_moves_nothing() {
        let sky = SkyOrientation::default();
        assert_eq!(sky.angle_deg, 0.0);
        assert_eq!(sky.sample_rows(), SkyOrientation::IDENTITY_ROWS);
        assert!(close(sky.rotate([0.3, -0.7, 0.2]), [0.3, -0.7, 0.2]));
        assert!(close(sky.euler_deg(), [0.0; 3]));
    }
}
