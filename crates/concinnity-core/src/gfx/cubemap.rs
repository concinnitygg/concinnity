//! The engine's cube-face convention, in one place: which way a face texel
//! looks, and the camera basis that captures that face.
//!
//! Face order is `0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z`, with a texel at `(u, v)` in
//! `[-1, 1]` looking along [`face_dir`]. Three sites have to agree on it -- the
//! cook's equirect resampler, the IBL convolutions that read a baked cube, and
//! the reflection probe that renders one -- and a face silently flipped or
//! rotated between them is the classic cube-capture bug, so the direction table
//! and the basis derived from it sit together and are pinned against each other
//! by a test below.

use crate::gfx::projection::normalize3;

/// Direction a face texel at `(u, v)` in `[-1, 1]` looks along, not normalised.
///
/// # Panics
///
/// Panics when `face` is not in `0..6`.
pub fn face_dir(face: usize, u: f32, v: f32) -> [f32; 3] {
    match face {
        0 => [1.0, -v, -u],
        1 => [-1.0, -v, u],
        2 => [u, 1.0, v],
        3 => [u, -1.0, -v],
        4 => [u, -v, 1.0],
        5 => [-u, -v, -1.0],
        _ => unreachable!("invalid cube face index {face}"),
    }
}

/// Unit direction for cube texel `(x, y)` of a `face_size` square face.
pub fn texel_dir(face: usize, x: u32, y: u32, face_size: u32) -> [f32; 3] {
    let u = (x as f32 + 0.5) / face_size as f32 * 2.0 - 1.0;
    let v = (y as f32 + 0.5) / face_size as f32 * 2.0 - 1.0;
    normalize3(face_dir(face, u, v))
}

/// Per-face camera basis `[right, up, forward]` in world space, derived so that
/// a 90-degree view down `-forward` reproduces [`face_dir`]: `right` is
/// `d/du` of the face direction, `up` is `-d/dv`.
pub const FACE_BASIS: [[[f32; 3]; 3]; 6] = [
    [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], // 0 +X
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], // 1 -X
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], // 2 +Y
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], // 3 -Y
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  // 4 +Z
    [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], // 5 -Z
];

#[cfg(test)]
mod tests {
    use super::*;

    // The two tables are one convention written twice, so derive each face
    // direction from its basis and require the table to agree: a texel at
    // (u, v) looks along forward + u*right - v*up.
    #[test]
    fn the_face_basis_reproduces_the_direction_table() {
        for (face, &[r, up, f]) in FACE_BASIS.iter().enumerate() {
            for (u, v) in [
                (0.0f32, 0.0f32),
                (0.5, 0.0),
                (0.0, 0.5),
                (-0.6, 0.3),
                (0.7, -0.4),
            ] {
                let from_basis = [
                    f[0] + u * r[0] - v * up[0],
                    f[1] + u * r[1] - v * up[1],
                    f[2] + u * r[2] - v * up[2],
                ];
                let d = face_dir(face, u, v);
                for axis in 0..3 {
                    assert!(
                        (d[axis] - from_basis[axis]).abs() < 1e-6,
                        "face {face} ({u},{v}) axis {axis}: {d:?} vs {from_basis:?}"
                    );
                }
            }
        }
    }

    const AXES: [[f32; 3]; 6] = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];

    #[test]
    fn a_face_centre_looks_straight_down_its_axis() {
        for (face, axis) in AXES.iter().enumerate() {
            assert_eq!(face_dir(face, 0.0, 0.0), *axis, "face {face}");
        }
    }

    // An even face size has no texel exactly on the centre, so the nearest one
    // only leans along the axis; what must hold is that it leans the right way.
    #[test]
    fn a_centre_texel_leans_along_its_face_axis() {
        for (face, axis) in AXES.iter().enumerate() {
            let d = texel_dir(face, 4, 4, 8);
            let along = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2];
            assert!(
                along > 0.9,
                "face {face}: {d:?} projects {along} onto {axis:?}"
            );
        }
    }

    #[test]
    fn every_texel_direction_is_unit_length() {
        for face in 0..6 {
            for y in 0..4 {
                for x in 0..4 {
                    let d = texel_dir(face, x, y, 4);
                    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    assert!((len - 1.0).abs() < 1e-5, "face {face} ({x},{y}) len {len}");
                }
            }
        }
    }
}
