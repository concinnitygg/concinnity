// What a move of the simulated frame does to everything in it, and the record
// of those moves the systems holding positions catch up from.

use crate::math::{
    Quat, euler_yxz_deg_from_quat, quat_from_euler_yxz_deg, quat_mul, quat_normalize, quat_rotate,
};
use crate::transform::{Mat4, mat4_mul};

/// A rigid move from one simulated frame into the next: `p' = R p + t`.
///
/// Applied to every position, velocity, rotation and remembered previous
/// pose at once, nothing moves relative to anything else, so the move is
/// invisible.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rebase {
    /// The rotation `R`, as a unit quaternion `[x, y, z, w]`.
    pub rotation: Quat,
    /// The translation `t`, applied after the rotation.
    pub translation: [f32; 3],
}

impl Rebase {
    /// The move that leaves everything where it is.
    pub const IDENTITY: Self = Self {
        rotation: [0.0, 0.0, 0.0, 1.0],
        translation: [0.0; 3],
    };

    /// A point carried into the new frame.
    pub fn apply_point(&self, p: [f32; 3]) -> [f32; 3] {
        let r = quat_rotate(self.rotation, p);
        [
            r[0] + self.translation[0],
            r[1] + self.translation[1],
            r[2] + self.translation[2],
        ]
    }

    /// A direction or velocity carried into the new frame.
    pub fn apply_vector(&self, v: [f32; 3]) -> [f32; 3] {
        quat_rotate(self.rotation, v)
    }

    /// An orientation carried into the new frame.
    pub fn apply_rotation(&self, q: Quat) -> Quat {
        quat_normalize(quat_mul(self.rotation, q))
    }

    /// Engine Euler degrees carried into the new frame.
    pub fn apply_euler_deg(&self, euler_deg: [f32; 3]) -> [f32; 3] {
        euler_yxz_deg_from_quat(self.apply_rotation(quat_from_euler_yxz_deg(euler_deg)))
    }

    /// The move as a column-major matrix.
    pub fn matrix(&self) -> Mat4 {
        let x = quat_rotate(self.rotation, [1.0, 0.0, 0.0]);
        let y = quat_rotate(self.rotation, [0.0, 1.0, 0.0]);
        let z = quat_rotate(self.rotation, [0.0, 0.0, 1.0]);
        let t = self.translation;
        [
            [x[0], x[1], x[2], 0.0],
            [y[0], y[1], y[2], 0.0],
            [z[0], z[1], z[2], 0.0],
            [t[0], t[1], t[2], 1.0],
        ]
    }

    /// The move back out of the new frame, as a column-major matrix.
    pub fn inverse_matrix(&self) -> Mat4 {
        self.inverse().matrix()
    }

    /// The move back out of the new frame.
    pub fn inverse(&self) -> Self {
        let [x, y, z, w] = self.rotation;
        let rotation = [-x, -y, -z, w];
        let t = quat_rotate(rotation, self.translation);
        Self {
            rotation,
            translation: [-t[0], -t[1], -t[2]],
        }
    }

    /// This move followed by `next`.
    pub fn then(&self, next: &Self) -> Self {
        Self {
            rotation: quat_normalize(quat_mul(next.rotation, self.rotation)),
            translation: next.apply_point(self.translation),
        }
    }

    /// A matrix that maps points of the old frame (a model matrix, or a view
    /// matrix's inverse) carried into the new frame: `M * m`.
    pub fn apply_matrix(&self, m: Mat4) -> Mat4 {
        mat4_mul(self.matrix(), m)
    }

    /// A matrix that takes points of the old frame somewhere (a view or a
    /// view-projection) made to take the same points in the new frame:
    /// `m * M^-1`.
    pub fn reproject(&self, m: Mat4) -> Mat4 {
        mat4_mul(m, self.inverse_matrix())
    }
}

/// Whether the camera has strayed far enough from the frame's origin that the
/// frame should move to it.
pub fn needs_rebase(camera: [f32; 3], distance: f32) -> bool {
    let [x, y, z] = camera;
    x * x + y * y + z * z > distance * distance
}

// How many past moves the record keeps. A reader steps every frame and the
// frame moves at most once per frame, so a reader is never more than one
// behind; the rest is slack.
const KEPT: usize = 8;

/// Every move of the simulated frame so far, as a count and the most recent
/// moves: a resource published by whatever moves the frame.
///
/// Each system holding positions of its own keeps a [`RebaseCursor`] and
/// applies what it has not yet seen before it touches them.
#[derive(Debug, Clone)]
pub struct FrameRebases {
    count: u64,
    recent: [Rebase; KEPT],
}

impl Default for FrameRebases {
    fn default() -> Self {
        Self {
            count: 0,
            recent: [Rebase::IDENTITY; KEPT],
        }
    }
}

impl FrameRebases {
    /// Record one more move.
    pub fn push(&mut self, rebase: Rebase) {
        self.recent[(self.count % KEPT as u64) as usize] = rebase;
        self.count += 1;
    }

    /// How many moves have been recorded.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// The moves after the first `seen`, composed into one, or `None` when
    /// there are none. A reader more than the kept moves behind gets the kept
    /// ones only.
    pub fn since(&self, seen: u64) -> Option<Rebase> {
        if seen >= self.count {
            return None;
        }
        let first = seen.max(self.count.saturating_sub(KEPT as u64));
        let mut total = Rebase::IDENTITY;
        for n in first..self.count {
            total = total.then(&self.recent[(n % KEPT as u64) as usize]);
        }
        Some(total)
    }
}

/// How far one reader has caught up with the [`FrameRebases`].
#[derive(Debug, Clone, Copy, Default)]
pub struct RebaseCursor {
    seen: u64,
}

impl RebaseCursor {
    /// Start caught up with `rebases`, so a reader built after some moves
    /// does not apply them to positions already in the current frame.
    pub fn at(rebases: Option<&FrameRebases>) -> Self {
        Self {
            seen: rebases.map_or(0, FrameRebases::count),
        }
    }

    /// The moves this reader has not yet applied, marking them applied.
    pub fn take(&mut self, rebases: Option<&FrameRebases>) -> Option<Rebase> {
        let rebases = rebases?;
        let pending = rebases.since(self.seen);
        self.seen = rebases.count();
        pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::quat_from_axis_angle;
    use crate::transform::trs_matrix;

    fn close(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < tol)
    }

    fn turn() -> Rebase {
        Rebase {
            rotation: quat_from_axis_angle([1.0, 0.0, 0.3], 0.02),
            translation: [-800.0, 6.0, 420.0],
        }
    }

    fn apply(m: Mat4, p: [f32; 3]) -> [f32; 4] {
        core::array::from_fn(|i| m[0][i] * p[0] + m[1][i] * p[1] + m[2][i] * p[2] + m[3][i])
    }

    #[test]
    fn the_matrix_moves_points_like_the_rebase() {
        let r = turn();
        let p = [810.0, -5.0, -400.0];
        let m = apply(r.matrix(), p);
        assert!(close([m[0], m[1], m[2]], r.apply_point(p), 1e-3));
        let back = r.inverse().apply_point(r.apply_point(p));
        assert!(close(back, p, 1e-3), "{back:?}");
    }

    // A view-projection reprojected into the new frame sees a carried point
    // exactly where the old one saw the point before it was carried.
    #[test]
    fn a_reprojected_view_sees_a_carried_point_where_it_was() {
        let r = turn();
        let view = trs_matrix([3.0, -2.0, 10.0], [5.0, 30.0, 0.0], [1.0; 3]);
        let p = [805.0, -4.0, -410.0];
        let before = apply(view, p);
        let after = apply(r.reproject(view), r.apply_point(p));
        for i in 0..4 {
            assert!((before[i] - after[i]).abs() < 2e-3, "{before:?} {after:?}");
        }
    }

    // A model matrix carried into the new frame places its geometry where the
    // carried points are.
    #[test]
    fn a_carried_model_matrix_places_geometry_in_the_new_frame() {
        let r = turn();
        let model = trs_matrix([800.0, -6.0, -420.0], [0.0, 45.0, 10.0], [2.0; 3]);
        let local = [0.5, 1.0, -0.25];
        let world = apply(model, local);
        let carried = apply(r.apply_matrix(model), local);
        let expected = r.apply_point([world[0], world[1], world[2]]);
        assert!(close([carried[0], carried[1], carried[2]], expected, 1e-3));
    }

    #[test]
    fn euler_angles_turn_with_the_frame() {
        let r = turn();
        let euler = [10.0, 70.0, -5.0];
        let turned = quat_from_euler_yxz_deg(r.apply_euler_deg(euler));
        let v = [0.0, 0.0, -1.0];
        let a = quat_rotate(turned, v);
        let b = r.apply_vector(quat_rotate(quat_from_euler_yxz_deg(euler), v));
        assert!(close(a, b, 1e-5));
    }

    #[test]
    fn composition_matches_applying_in_turn() {
        let a = turn();
        let b = Rebase {
            rotation: quat_from_axis_angle([0.0, 0.2, 1.0], -0.015),
            translation: [500.0, -3.0, 900.0],
        };
        let p = [12.0, 3.0, -40.0];
        let both = a.then(&b).apply_point(p);
        assert!(close(both, b.apply_point(a.apply_point(p)), 1e-3));
    }

    #[test]
    fn the_rebase_trigger_is_a_distance_from_the_origin() {
        assert!(!needs_rebase([600.0, 0.0, 700.0], 1000.0));
        assert!(needs_rebase([800.0, 10.0, 700.0], 1000.0));
        assert!(needs_rebase([0.0, 1001.0, 0.0], 1000.0));
    }

    #[test]
    fn a_cursor_applies_each_move_once() {
        let mut rebases = FrameRebases::default();
        let mut cursor = RebaseCursor::at(Some(&rebases));
        assert!(cursor.take(Some(&rebases)).is_none());
        rebases.push(turn());
        let taken = cursor.take(Some(&rebases)).expect("one move pending");
        let p = [1.0, 2.0, 3.0];
        assert!(close(taken.apply_point(p), turn().apply_point(p), 1e-4));
        assert!(cursor.take(Some(&rebases)).is_none(), "already applied");
        assert!(cursor.take(None).is_none());
    }

    // A reader that missed two moves gets both, in order.
    #[test]
    fn a_lagging_cursor_gets_the_moves_composed() {
        let mut rebases = FrameRebases::default();
        let mut cursor = RebaseCursor::at(Some(&rebases));
        let a = turn();
        let b = a.inverse();
        rebases.push(a);
        rebases.push(b);
        let total = cursor.take(Some(&rebases)).expect("two moves pending");
        assert!(close(
            total.apply_point([5.0, 6.0, 7.0]),
            [5.0, 6.0, 7.0],
            1e-3
        ));
        // A cursor started after the moves sees none of them.
        assert!(
            RebaseCursor::at(Some(&rebases))
                .take(Some(&rebases))
                .is_none()
        );
    }
}
