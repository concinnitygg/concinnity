// A 3x3 matrix, held for the two jobs the simulation needs one for: the
// world-space inverse inertia tensor, and the effective-mass block a joint
// solves three coupled rows through. Every shape here has principal axes
// aligned with its local frame, so inertia is stored as a diagonal and only
// becomes a full matrix once rotated into world space.

use super::quat::Quat;
use super::vec3::Vec3;

/// Column-major: `cols[i]` is the image of basis vector `i`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Mat3 {
    pub(crate) cols: [Vec3; 3],
}

impl Default for Mat3 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl Mat3 {
    pub(crate) const ZERO: Mat3 = Mat3 {
        cols: [Vec3::ZERO; 3],
    };

    #[cfg(test)]
    pub(crate) const IDENTITY: Mat3 = Mat3 {
        cols: [Vec3::X, Vec3::Y, Vec3::Z],
    };

    pub(crate) const fn from_cols(x: Vec3, y: Vec3, z: Vec3) -> Self {
        Mat3 { cols: [x, y, z] }
    }

    #[cfg(test)]
    pub(crate) const fn from_diagonal(d: Vec3) -> Self {
        Mat3 {
            cols: [
                Vec3 {
                    x: d.x,
                    y: 0.0,
                    z: 0.0,
                },
                Vec3 {
                    x: 0.0,
                    y: d.y,
                    z: 0.0,
                },
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: d.z,
                },
            ],
        }
    }

    pub(crate) fn from_quat(q: Quat) -> Self {
        Mat3 {
            cols: [q.rotate(Vec3::X), q.rotate(Vec3::Y), q.rotate(Vec3::Z)],
        }
    }

    /// `R * diag(d) * R^T`: a diagonal tensor expressed in the frame `q`
    /// names. Built directly from the rotated axes rather than by multiplying
    /// three matrices.
    pub(crate) fn diagonal_conjugated(q: Quat, d: Vec3) -> Self {
        let r = Self::from_quat(q);
        let scaled = [r.cols[0] * d.x, r.cols[1] * d.y, r.cols[2] * d.z];
        // Row i of R^T is column i of R, so the product's column j is
        // sum_k scaled[k] * R.cols[k].get(j).
        Mat3 {
            cols: [
                scaled[0] * r.cols[0].x + scaled[1] * r.cols[1].x + scaled[2] * r.cols[2].x,
                scaled[0] * r.cols[0].y + scaled[1] * r.cols[1].y + scaled[2] * r.cols[2].y,
                scaled[0] * r.cols[0].z + scaled[1] * r.cols[1].z + scaled[2] * r.cols[2].z,
            ],
        }
    }

    pub(crate) fn mul_vec3(&self, v: Vec3) -> Vec3 {
        self.cols[0] * v.x + self.cols[1] * v.y + self.cols[2] * v.z
    }

    pub(crate) fn add(&self, other: Mat3) -> Mat3 {
        Mat3 {
            cols: [
                self.cols[0] + other.cols[0],
                self.cols[1] + other.cols[1],
                self.cols[2] + other.cols[2],
            ],
        }
    }

    /// The inverse, or [`Mat3::ZERO`] when there is none.
    ///
    /// A singular block is the honest answer for a pair nothing can move: an
    /// inverse of zero produces an impulse of zero, where a `None` would make
    /// every caller restate the same degenerate case.
    pub(crate) fn inverse(&self) -> Mat3 {
        let [a, b, c] = self.cols;
        let (r0, r1, r2) = (b.cross(c), c.cross(a), a.cross(b));
        let determinant = a.dot(r0);
        if determinant.abs() <= f32::MIN_POSITIVE {
            return Mat3::ZERO;
        }
        let inv = 1.0 / determinant;
        // The cofactor rows scaled by the determinant are the inverse's rows,
        // so they transpose into its columns.
        Mat3 {
            cols: [
                super::vec3::vec3(r0.x, r1.x, r2.x) * inv,
                super::vec3::vec3(r0.y, r1.y, r2.y) * inv,
                super::vec3::vec3(r0.z, r1.z, r2.z) * inv,
            ],
        }
    }

    #[cfg(test)]
    pub(crate) fn transpose(&self) -> Mat3 {
        Mat3 {
            cols: [
                super::vec3::vec3(self.cols[0].x, self.cols[1].x, self.cols[2].x),
                super::vec3::vec3(self.cols[0].y, self.cols[1].y, self.cols[2].y),
                super::vec3::vec3(self.cols[0].z, self.cols[1].z, self.cols[2].z),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::vec3::vec3;
    use super::*;

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1.0e-5
    }

    #[test]
    fn identity_and_diagonal_scale_their_axes() {
        assert!(close(
            Mat3::IDENTITY.mul_vec3(vec3(1.0, 2.0, 3.0)),
            vec3(1.0, 2.0, 3.0)
        ));
        let d = Mat3::from_diagonal(vec3(2.0, 3.0, 4.0));
        assert!(close(d.mul_vec3(vec3(1.0, 1.0, 1.0)), vec3(2.0, 3.0, 4.0)));
        assert_eq!(Mat3::ZERO.mul_vec3(vec3(1.0, 2.0, 3.0)), Vec3::ZERO);
    }

    #[test]
    fn a_sum_of_matrices_adds_columnwise() {
        let a = Mat3::from_diagonal(vec3(1.0, 2.0, 3.0));
        let b = Mat3::from_cols(Vec3::Y, Vec3::Z, Vec3::X);
        let sum = a.add(b);
        assert!(close(sum.cols[0], vec3(1.0, 1.0, 0.0)));
        assert!(close(sum.cols[1], vec3(0.0, 2.0, 1.0)));
        assert!(close(sum.cols[2], vec3(1.0, 0.0, 3.0)));
    }

    // The property the solver relies on: whatever the block, multiplying by
    // its inverse gives back the vector that went in.
    #[test]
    fn an_inverse_undoes_the_matrix_it_came_from() {
        let rotation = Mat3::from_quat(Quat::from_euler_deg([25.0, -40.0, 65.0]));
        for m in [
            Mat3::IDENTITY,
            Mat3::from_diagonal(vec3(2.0, 0.5, 4.0)),
            Mat3::diagonal_conjugated(
                Quat::from_euler_deg([10.0, 20.0, 30.0]),
                vec3(1.0, 3.0, 7.0),
            ),
            rotation,
        ] {
            let inverse = m.inverse();
            for v in [Vec3::X, Vec3::Y, Vec3::Z, vec3(0.4, -1.2, 3.0)] {
                assert!(close(inverse.mul_vec3(m.mul_vec3(v)), v), "{m:?} {v:?}");
            }
        }
    }

    // A block nothing can move has no inverse, and zero is the answer that
    // makes the impulse it scales come out zero too.
    #[test]
    fn a_singular_matrix_inverts_to_zero() {
        assert_eq!(Mat3::ZERO.inverse(), Mat3::ZERO);
        // Two equal columns: rank two, no inverse.
        let flat = Mat3::from_cols(Vec3::X, Vec3::X, Vec3::Z);
        assert_eq!(flat.inverse(), Mat3::ZERO);
    }

    #[test]
    fn from_quat_matches_rotating_the_vector_directly() {
        let q = Quat::from_euler_deg([15.0, 40.0, -25.0]);
        let m = Mat3::from_quat(q);
        for v in [Vec3::X, Vec3::Y, Vec3::Z, vec3(1.0, -2.0, 0.5)] {
            assert!(close(m.mul_vec3(v), q.rotate(v)), "{v:?}");
        }
    }

    // A rotated sphere tensor is still the same multiple of the identity: the
    // conjugation must not change an isotropic tensor.
    #[test]
    fn conjugating_an_isotropic_tensor_leaves_it_alone() {
        let q = Quat::from_euler_deg([33.0, -71.0, 12.0]);
        let m = Mat3::diagonal_conjugated(q, Vec3::splat(2.5));
        for v in [Vec3::X, Vec3::Y, Vec3::Z, vec3(0.3, 0.4, -0.5)] {
            assert!(
                close(m.mul_vec3(v), v * 2.5),
                "{v:?} -> {:?}",
                m.mul_vec3(v)
            );
        }
    }

    // The conjugated tensor must agree with applying R^T, the diagonal, then R
    // in three separate steps.
    #[test]
    fn conjugation_agrees_with_the_three_step_product() {
        let q = Quat::from_euler_deg([20.0, 65.0, -40.0]);
        let d = vec3(1.0, 4.0, 9.0);
        let m = Mat3::diagonal_conjugated(q, d);
        for v in [Vec3::X, Vec3::Y, Vec3::Z, vec3(-1.0, 2.0, 3.0)] {
            let expected = q.rotate(q.inverse_rotate(v) * d);
            assert!(close(m.mul_vec3(v), expected), "{v:?}");
        }
    }

    // Inertia tensors are symmetric, so the conjugation must produce one.
    #[test]
    fn a_conjugated_diagonal_is_symmetric() {
        let m = Mat3::diagonal_conjugated(
            Quat::from_euler_deg([12.0, 34.0, 56.0]),
            vec3(1.0, 2.0, 3.0),
        );
        let t = m.transpose();
        for i in 0..3 {
            assert!(close(m.cols[i], t.cols[i]), "column {i}");
        }
    }
}
