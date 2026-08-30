// Large inside-facing cube used as a skybox.
//
// Wound CCW from the interior. The floor face is included to prevent the clear
// colour showing at the horizon or beyond terrain edges.
//
// The blue channel is set to 2.0 (outside the [0, 1] scene range) so the
// fragment shader can identify sky vertices with a simple threshold and skip
// diffuse lighting on them.
//
// UV layout is present for legacy compatibility; sky rendering now uses the
// view-direction rather than UV so the values do not affect the output.

use alloc::vec::Vec;

use super::Vert;

// A skybox face: four corner positions, the inward normal, then the four UVs
// (one per corner).
type SkyboxFace = (
    [f32; 3],
    [f32; 3],
    [f32; 3],
    [f32; 3],
    [f32; 3],
    [[f32; 2]; 4],
);

/// Build an inside-facing skybox cube with half-extent `size` on all axes.
/// Keep `size` below the camera's far plane so the sky is not clipped.
pub fn build_skybox(size: f32) -> (Vec<Vert>, Vec<u16>) {
    let s = size;
    let color = [1.0f32, 1.0, 2.0];

    let mut verts: Vec<Vert> = Vec::new();
    let mut idxs: Vec<u16> = Vec::new();

    let faces: &[SkyboxFace] = &[
        // ceiling (+Y, viewed from below) -- CCW from interior requires reversed vertex order
        (
            [-s, s, s],
            [-s, s, -s],
            [s, s, -s],
            [s, s, s],
            [0.0, -1.0, 0.0],
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        ),
        // floor (-Y, viewed from above) -- CCW from interior requires reversed vertex order
        (
            [-s, -s, -s],
            [-s, -s, s],
            [s, -s, s],
            [s, -s, -s],
            [0.0, 1.0, 0.0],
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        ),
        // north wall (+Z)
        (
            [s, -s, s],
            [-s, -s, s],
            [-s, s, s],
            [s, s, s],
            [0.0, 0.0, -1.0],
            [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        ),
        // south wall (-Z)
        (
            [-s, -s, -s],
            [s, -s, -s],
            [s, s, -s],
            [-s, s, -s],
            [0.0, 0.0, 1.0],
            [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        ),
        // east wall (+X)
        (
            [s, -s, -s],
            [s, -s, s],
            [s, s, s],
            [s, s, -s],
            [-1.0, 0.0, 0.0],
            [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        ),
        // west wall (-X)
        (
            [-s, -s, s],
            [-s, -s, -s],
            [-s, s, -s],
            [-s, s, s],
            [1.0, 0.0, 0.0],
            [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        ),
    ];

    for (a, b, c, d, normal, uvs) in faces {
        let base = verts.len() as u16;
        verts.extend_from_slice(&[
            (*a, *normal, color, uvs[0]),
            (*b, *normal, color, uvs[1]),
            (*c, *normal, color, uvs[2]),
            (*d, *normal, color, uvs[3]),
        ]);
        idxs.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    (verts, idxs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skybox_is_six_quads_at_the_requested_half_extent() {
        let (verts, idxs) = build_skybox(490.0);
        assert_eq!(verts.len(), 6 * 4);
        assert_eq!(idxs.len(), 6 * 6);
        assert!(idxs.iter().all(|&i| (i as usize) < verts.len()));
        for (pos, ..) in &verts {
            for axis in pos {
                assert_eq!(axis.abs(), 490.0, "corner off the cube: {pos:?}");
            }
        }
    }

    #[test]
    fn skybox_normals_face_the_interior() {
        let (verts, _) = build_skybox(4.0);
        for (pos, normal, ..) in &verts {
            let dot = pos[0] * normal[0] + pos[1] * normal[1] + pos[2] * normal[2];
            assert_eq!(dot, -4.0, "normal {normal:?} is not inward at {pos:?}");
        }
    }

    #[test]
    fn skybox_vertex_colour_flags_sky_with_an_out_of_range_blue() {
        let (verts, _) = build_skybox(1.0);
        assert!(
            verts
                .iter()
                .all(|(_, _, color, _)| *color == [1.0, 1.0, 2.0])
        );
    }
}
