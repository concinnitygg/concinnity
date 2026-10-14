//! The six faces of the cube around a planet, each carrying its own copy of
//! the sea's wave layout. A face maps the sphere onto its plane by the
//! stereographic projection from the opposite face's center: conformal, so a
//! wave keeps its shape wherever it lies, and stretched by `2 / (1 + cos a)`
//! at `a` from the face's center, so by at most a quarter at the cube's
//! corners. Two faces sharing an edge stretch equally along it.

use crate::planet::{DVec3, dvec};

/// One face of the cube: its normal and the two axes its plane is laid out
/// along, in the planet's authored axes. The four faces around `+Y` continue
/// its axes across their shared edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeaFace {
    // `[u, normal, v]`, a right-handed basis (`u x normal = v`).
    axes: [DVec3; 3],
}

const X: DVec3 = [1.0, 0.0, 0.0];
const Y: DVec3 = [0.0, 1.0, 0.0];
const Z: DVec3 = [0.0, 0.0, 1.0];

fn neg(v: DVec3) -> DVec3 {
    dvec::scale(v, -1.0)
}

impl SeaFace {
    /// The face whose normal is `axis` (0 = X, 1 = Y, 2 = Z), on its
    /// `positive` side.
    pub fn of_axis(axis: usize, positive: bool) -> Self {
        let axes = match (axis, positive) {
            (0, true) => [neg(Y), X, Z],
            (0, false) => [Y, neg(X), Z],
            (1, true) => [X, Y, Z],
            (1, false) => [X, neg(Y), neg(Z)],
            (2, true) => [X, Z, neg(Y)],
            _ => [X, neg(Z), Y],
        };
        Self { axes }
    }

    /// The face the direction `dir` from the planet's center passes through.
    pub fn holding(dir: DVec3) -> Self {
        let a = dir.map(f64::abs);
        let axis = if a[0] >= a[1] && a[0] >= a[2] {
            0
        } else if a[1] >= a[2] {
            1
        } else {
            2
        };
        Self::of_axis(axis, dir[axis] >= 0.0)
    }

    /// `[u, normal, v]` in authored axes.
    pub fn axes(&self) -> [DVec3; 3] {
        self.axes
    }

    /// The authored direction `d` in the face's own axes `[u, normal, v]`.
    pub fn coords(&self, d: DVec3) -> DVec3 {
        self.axes.map(|a| dvec::dot(a, d))
    }

    /// Where the unit direction `d` lands on the face's plane, in meters on
    /// a sphere of `radius`: `2 radius (u, v) / (1 + normal)`.
    pub fn project(&self, radius: f64, d: DVec3) -> [f64; 2] {
        let [u, n, v] = self.coords(d);
        let k = 2.0 * radius / (1.0 + n);
        [k * u, k * v]
    }

    /// The unit direction that lands on `s` (see [`SeaFace::project`]).
    pub fn unproject(&self, radius: f64, s: [f64; 2]) -> DVec3 {
        let (a, b) = (s[0] / (2.0 * radius), s[1] / (2.0 * radius));
        let q = a * a + b * b;
        let f = [2.0 * a, 1.0 - q, 2.0 * b].map(|c| c / (1.0 + q));
        let [u, n, v] = self.axes;
        dvec::add(
            dvec::add(dvec::scale(u, f[0]), dvec::scale(n, f[1])),
            dvec::scale(v, f[2]),
        )
    }

    /// Meters on the face's plane per meter along the sphere at `d`.
    pub fn stretch(&self, d: DVec3) -> f64 {
        2.0 / (1.0 + self.coords(d)[1])
    }

    /// The unit directions along the sphere at `d` in which the face's `u`
    /// and `v` grow, in authored axes. They are perpendicular, since the map
    /// is conformal.
    pub fn tangents(&self, d: DVec3) -> [DVec3; 2] {
        let [x, n, z] = self.coords(d);
        let [u_axis, n_axis, v_axis] = self.axes;
        let along = |a: DVec3| {
            let flat = dvec::sub(a, dvec::scale(d, dvec::dot(a, d)));
            dvec::normalize_or(flat, u_axis)
        };
        // The gradients of u / (1 + n) and v / (1 + n).
        let gu = dvec::sub(dvec::scale(u_axis, 1.0 + n), dvec::scale(n_axis, x));
        let gv = dvec::sub(dvec::scale(v_axis, 1.0 + n), dvec::scale(n_axis, z));
        [along(gu), along(gv)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all() -> impl Iterator<Item = SeaFace> {
        (0..6).map(|i| SeaFace::of_axis(i / 2, i % 2 == 0))
    }

    fn close(a: DVec3, b: DVec3, tol: f64) -> bool {
        dvec::length(dvec::sub(a, b)) < tol
    }

    // Every face is a right-handed orthonormal basis whose normal points out
    // of the cube.
    #[test]
    fn every_face_is_a_right_handed_basis() {
        for (i, face) in all().enumerate() {
            let [u, n, v] = face.axes();
            assert!(close(dvec::cross(u, n), v, 1e-12), "{i}");
            assert_eq!(n[i / 2], if i % 2 == 0 { 1.0 } else { -1.0 }, "{i}");
            assert_eq!(SeaFace::holding(dvec::scale(n, 3.0)), face);
        }
    }

    // A face's center maps to its origin with no stretch, and every point
    // maps back onto itself.
    #[test]
    fn the_projection_round_trips() {
        let r = 50_000.0;
        for face in all() {
            let [_, n, _] = face.axes();
            assert!(face.project(r, n).iter().all(|c| c.abs() < 1e-9));
            assert!((face.stretch(n) - 1.0).abs() < 1e-12);
            for d in [[0.3, 0.9, -0.2], [0.7, 0.7, 0.1], [-0.5, 0.6, 0.6]] {
                let d = dvec::normalize_or(d, Y);
                let back = face.unproject(r, face.project(r, d));
                if dvec::dot(d, n) > 0.0 {
                    assert!(close(back, d, 1e-12), "{back:?} {d:?}");
                }
            }
        }
    }

    // The tangents are the directions the projection grows along, at the
    // face's stretch: a small step along either moves the projected point by
    // the stretch along that axis only.
    #[test]
    fn the_tangents_follow_the_projection() {
        let r = 50_000.0;
        let face = SeaFace::of_axis(1, true);
        for d in [[0.0, 1.0, 0.0], [0.6, 0.75, 0.2], [-0.3, 0.8, -0.5]] {
            let d = dvec::normalize_or(d, Y);
            let [tu, tv] = face.tangents(d);
            assert!(dvec::dot(tu, tv).abs() < 1e-9);
            assert!(dvec::dot(tu, d).abs() < 1e-9);
            let k = face.stretch(d);
            let s0 = face.project(r, d);
            let step = 0.01;
            for (t, axis) in [(tu, 0), (tv, 1)] {
                let moved = dvec::normalize_or(dvec::add(d, dvec::scale(t, step / r)), Y);
                let s1 = face.project(r, moved);
                let ds = [s1[0] - s0[0], s1[1] - s0[1]];
                assert!((ds[axis] - k * step).abs() < 1e-6, "{ds:?} {k}");
                assert!(ds[1 - axis].abs() < 1e-6, "{ds:?}");
            }
        }
    }

    // Two faces sharing an edge stretch alike on it, and the four faces
    // around +Y carry its u and v across their edges.
    #[test]
    fn neighbors_meet_alike_at_their_edge() {
        let top = SeaFace::of_axis(1, true);
        for (axis, positive, u_or_v) in [(0, true, 0), (0, false, 0), (2, true, 1), (2, false, 1)] {
            let side = SeaFace::of_axis(axis, positive);
            let [_, n, _] = side.axes();
            let edge = dvec::normalize_or(dvec::add(Y, n), Y);
            assert!((top.stretch(edge) - side.stretch(edge)).abs() < 1e-12);
            let a = top.tangents(edge)[u_or_v];
            let b = side.tangents(edge)[u_or_v];
            assert!(close(a, b, 1e-9), "{axis} {positive}: {a:?} {b:?}");
        }
    }
}
