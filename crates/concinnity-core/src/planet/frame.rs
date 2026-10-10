// The local frame a planet world simulates in: an anchor held in double
// precision, and the single-precision frame around it whose +Y is the radial
// up at the anchor.

use super::dvec::{self, DMat3, DVec3};
use super::rebase::Rebase;

/// Where the simulated frame sits in the world it was authored in.
///
/// The world is authored, and starts, in one frame. Positions far from it lose
/// precision in single-precision floats, so a planet world keeps simulating in
/// a frame that moves with the camera: its origin is a point in the authored
/// frame, held in double precision, and its axes are the authored axes turned
/// so that `+Y` points away from the planet's center at that point. Everything
/// the engine simulates and renders stays in this frame, in single precision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalFrame {
    origin: DVec3,
    // Column-major: column `i` is local axis `i` in authored coordinates.
    basis: DMat3,
}

impl Default for LocalFrame {
    fn default() -> Self {
        Self::AUTHORED
    }
}

impl LocalFrame {
    /// The frame the world was authored in.
    pub const AUTHORED: Self = Self {
        origin: [0.0; 3],
        basis: dvec::DMAT3_IDENTITY,
    };

    /// The frame's origin in authored coordinates.
    pub fn origin(&self) -> DVec3 {
        self.origin
    }

    /// The frame's local point `p` in authored coordinates.
    pub fn to_world(&self, p: [f32; 3]) -> DVec3 {
        dvec::add(self.origin, dvec::mul_vec(&self.basis, dvec::from_f32(p)))
    }

    /// The authored point `w` in this frame, in double precision.
    pub fn to_local_f64(&self, w: DVec3) -> DVec3 {
        dvec::mul_transpose_vec(&self.basis, dvec::sub(w, self.origin))
    }

    /// The authored point `w` in this frame.
    pub fn to_local(&self, w: DVec3) -> [f32; 3] {
        dvec::to_f32(self.to_local_f64(w))
    }

    /// The authored direction `d` in this frame.
    pub fn dir_to_local(&self, d: DVec3) -> [f32; 3] {
        dvec::to_f32(dvec::mul_transpose_vec(&self.basis, d))
    }

    /// The local direction `d` in authored coordinates.
    pub fn dir_to_world(&self, d: [f32; 3]) -> DVec3 {
        dvec::mul_vec(&self.basis, dvec::from_f32(d))
    }

    /// The rotation taking an authored direction into this frame, column-major.
    pub fn rotation_from_world(&self) -> [[f32; 3]; 3] {
        let b = &self.basis;
        // The transpose of the basis.
        core::array::from_fn(|col| core::array::from_fn(|row| b[row][col] as f32))
    }

    /// The model matrix placing geometry whose vertices are relative to the
    /// authored point `origin`, with authored axes, in this frame.
    pub fn model_at(&self, origin: DVec3) -> crate::transform::Mat4 {
        let r = self.rotation_from_world();
        let t = self.to_local(origin);
        [
            [r[0][0], r[0][1], r[0][2], 0.0],
            [r[1][0], r[1][1], r[1][2], 0.0],
            [r[2][0], r[2][1], r[2][2], 0.0],
            [t[0], t[1], t[2], 1.0],
        ]
    }

    /// The frame recentered on local point `at`, its `+Y` turned onto the
    /// authored direction `up`, and the transform that carries a point of this
    /// frame into the new one.
    ///
    /// The axes turn along the shortest arc from the old `+Y` to the new one,
    /// so nothing spins about the new up: a heading measured from the frame's
    /// axes carries across unchanged.
    pub fn recentered(&self, at: [f32; 3], up: DVec3) -> (Self, Rebase) {
        let up_local =
            dvec::normalize_or(dvec::mul_transpose_vec(&self.basis, up), [0.0, 1.0, 0.0]);
        let turn = dvec::arc_from_up(up_local);
        let frame = Self {
            origin: self.to_world(at),
            basis: dvec::orthonormalize(&dvec::mul(&self.basis, &turn)),
        };
        // new = turn^T (old - at)
        let inverse: DMat3 = core::array::from_fn(|c| core::array::from_fn(|r| turn[r][c]));
        let shift = dvec::mul_vec(&inverse, dvec::from_f32(at));
        let rebase = Rebase {
            rotation: dvec::quat_of(&inverse),
            translation: dvec::to_f32(dvec::scale(shift, -1.0)),
        };
        (frame, rebase)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < tol)
    }

    const CENTER: DVec3 = [0.0, -50_000.0, 0.0];

    fn up_at(frame: &LocalFrame, p: [f32; 3]) -> DVec3 {
        dvec::normalize_or(dvec::sub(frame.to_world(p), CENTER), [0.0, 1.0, 0.0])
    }

    #[test]
    fn the_authored_frame_is_the_identity() {
        let f = LocalFrame::AUTHORED;
        assert_eq!(f.to_world([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]);
        assert_eq!(f.to_local([4.0, 5.0, 6.0]), [4.0, 5.0, 6.0]);
    }

    // Recentering puts the chosen point at the origin with the radial up on
    // +Y, and the rebase carries every point exactly as the two frames say.
    #[test]
    fn recentering_puts_the_point_at_the_origin_under_its_own_up() {
        let old = LocalFrame::AUTHORED;
        let at = [900.0, -8.1, -450.0];
        let (new, rebase) = old.recentered(at, up_at(&old, at));
        assert!(close(new.to_local(old.to_world(at)), [0.0; 3], 1e-4));
        let up = new.dir_to_local(up_at(&old, at));
        assert!(close(up, [0.0, 1.0, 0.0], 1e-6), "{up:?}");
        for p in [[0.0, 0.0, 0.0], [905.0, -7.0, -440.0], [-30.0, 12.0, 80.0]] {
            let by_frames = new.to_local(old.to_world(p));
            let by_rebase = rebase.apply_point(p);
            assert!(
                close(by_frames, by_rebase, 2e-3),
                "{by_frames:?} {by_rebase:?}"
            );
        }
    }

    // Many rebases along a long walk keep the axes orthonormal and keep a
    // point's authored position where it was.
    #[test]
    fn a_long_walk_of_rebases_loses_nothing() {
        let mut frame = LocalFrame::AUTHORED;
        let landmark: DVec3 = [3_000.0, -95.0, 7_000.0];
        let mut local = frame.to_local(landmark);
        for _ in 0..40 {
            let at = [800.0, 0.0, 600.0];
            let (next, rebase) = frame.recentered(at, up_at(&frame, at));
            local = rebase.apply_point(local);
            frame = next;
            let drift = dvec::length(dvec::sub(frame.to_world(local), landmark));
            assert!(drift < 0.2, "{drift}");
            let b = &frame.basis;
            assert!((dvec::length(b[0]) - 1.0).abs() < 1e-12);
            assert!(dvec::dot(b[0], b[1]).abs() < 1e-12);
        }
        // Forty kilometers on, the frame's own up is still the radial up.
        let up = frame.dir_to_local(up_at(&frame, [0.0; 3]));
        assert!(close(up, [0.0, 1.0, 0.0], 1e-6));
    }

    // Geometry placed by `model_at` lands where the frame puts its authored
    // points.
    #[test]
    fn a_model_matrix_places_authored_geometry_in_the_frame() {
        let old = LocalFrame::AUTHORED;
        let at = [1_500.0, -22.0, 300.0];
        let (frame, _) = old.recentered(at, up_at(&old, at));
        let origin: DVec3 = [1_480.0, -20.0, 310.0];
        let m = frame.model_at(origin);
        let offset = [2.0, 0.5, -3.0];
        let placed: [f32; 3] = core::array::from_fn(|i| {
            m[0][i] * offset[0] + m[1][i] * offset[1] + m[2][i] * offset[2] + m[3][i]
        });
        let expected = frame.to_local(dvec::add(origin, dvec::from_f32(offset)));
        assert!(close(placed, expected, 1e-3), "{placed:?} {expected:?}");
    }

    #[test]
    fn the_world_rotation_is_the_inverse_basis() {
        let old = LocalFrame::AUTHORED;
        let at = [2_000.0, -40.0, 0.0];
        let (new, _) = old.recentered(at, up_at(&old, at));
        let r = new.rotation_from_world();
        let d = [0.0, 0.0, 1.0];
        let by_rows: [f32; 3] = core::array::from_fn(|i| r[0][i] * 0.0 + r[1][i] * 0.0 + r[2][i]);
        assert!(close(by_rows, new.dir_to_local(d), 1e-6));
        let back = new.dir_to_world(new.dir_to_local([0.3, 0.4, 0.5]));
        assert!(close(dvec::to_f32(back), [0.3, 0.4, 0.5], 1e-6));
    }
}
