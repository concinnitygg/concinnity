// Room interior geometry.
//
// Each face is two CCW triangles. Vertex colour distinguishes surfaces:
// warm dark grey floor, light grey ceiling, slightly different greys per wall.
// UV coordinates tile at one repeat per metre.

use alloc::vec::Vec;

use super::Vert;

/// Build room geometry from explicit extents.
///
/// Returns `(vertices, indices)` where each vertex is `(pos, normal, color, uv)`.
/// Winding is CCW when viewed from inside. UV coordinates tile at one repeat
/// per metre. Normals point inward so diffuse lighting is correct for a camera
/// inside the room.
pub fn build_room_geometry(
    half_width: f32,
    half_depth: f32,
    floor_y: f32,
    ceiling_y: f32,
) -> (Vec<Vert>, Vec<u16>) {
    let mut verts: Vec<Vert> = Vec::new();
    let mut idxs: Vec<u16> = Vec::new();

    let (xn, xp) = (-half_width, half_width);
    let (yn, yp) = (floor_y, ceiling_y);
    let (zn, zp) = (-half_depth, half_depth);

    let width = half_width * 2.0;
    let depth = half_depth * 2.0;
    let height = yp - yn;

    let mut push_quad = |a: [f32; 3],
                         b: [f32; 3],
                         c: [f32; 3],
                         d: [f32; 3],
                         normal: [f32; 3],
                         color: [f32; 3],
                         uv_a: [f32; 2],
                         uv_b: [f32; 2],
                         uv_c: [f32; 2],
                         uv_d: [f32; 2]| {
        let base = verts.len() as u16;
        verts.extend_from_slice(&[
            (a, normal, color, uv_a),
            (b, normal, color, uv_b),
            (c, normal, color, uv_c),
            (d, normal, color, uv_d),
        ]);
        idxs.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    };

    let floor_color = [0.25, 0.22, 0.20];
    let ceiling_color = [0.70, 0.70, 0.72];
    let wall_n_color = [0.45, 0.45, 0.48];
    let wall_s_color = [0.40, 0.40, 0.43];
    let wall_e_color = [0.50, 0.48, 0.46];
    let wall_w_color = [0.42, 0.44, 0.46];

    // floor: normal points up into the room
    push_quad(
        [xn, yn, zn],
        [xp, yn, zn],
        [xp, yn, zp],
        [xn, yn, zp],
        [0.0, 1.0, 0.0],
        floor_color,
        [0.0, 0.0],
        [width, 0.0],
        [width, depth],
        [0.0, depth],
    );

    // ceiling: normal points down into the room
    push_quad(
        [xn, yp, zp],
        [xp, yp, zp],
        [xp, yp, zn],
        [xn, yp, zn],
        [0.0, -1.0, 0.0],
        ceiling_color,
        [0.0, 0.0],
        [width, 0.0],
        [width, depth],
        [0.0, depth],
    );

    // north wall (+Z face); normal points toward -Z
    push_quad(
        [xp, yn, zp],
        [xn, yn, zp],
        [xn, yp, zp],
        [xp, yp, zp],
        [0.0, 0.0, -1.0],
        wall_n_color,
        [0.0, height],
        [width, height],
        [width, 0.0],
        [0.0, 0.0],
    );

    // south wall (-Z face); normal points toward +Z
    push_quad(
        [xn, yn, zn],
        [xp, yn, zn],
        [xp, yp, zn],
        [xn, yp, zn],
        [0.0, 0.0, 1.0],
        wall_s_color,
        [0.0, height],
        [width, height],
        [width, 0.0],
        [0.0, 0.0],
    );

    // east wall (+X face); normal points toward -X
    push_quad(
        [xp, yn, zn],
        [xp, yn, zp],
        [xp, yp, zp],
        [xp, yp, zn],
        [-1.0, 0.0, 0.0],
        wall_e_color,
        [0.0, height],
        [depth, height],
        [depth, 0.0],
        [0.0, 0.0],
    );

    // west wall (-X face); normal points toward +X
    push_quad(
        [xn, yn, zp],
        [xn, yn, zn],
        [xn, yp, zn],
        [xn, yp, zp],
        [1.0, 0.0, 0.0],
        wall_w_color,
        [0.0, height],
        [depth, height],
        [depth, 0.0],
        [0.0, 0.0],
    );

    (verts, idxs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(verts: &[Vert]) -> ([f32; 3], [f32; 3]) {
        let mut mn = [f32::INFINITY; 3];
        let mut mx = [f32::NEG_INFINITY; 3];
        for (pos, ..) in verts {
            for k in 0..3 {
                mn[k] = mn[k].min(pos[k]);
                mx[k] = mx[k].max(pos[k]);
            }
        }
        (mn, mx)
    }

    #[test]
    fn room_geometry_is_six_quads_spanning_the_requested_extents() {
        let (verts, idxs) = build_room_geometry(3.0, 4.0, 0.0, 2.5);
        assert_eq!(verts.len(), 6 * 4);
        assert_eq!(idxs.len(), 6 * 6);
        assert!(idxs.iter().all(|&i| (i as usize) < verts.len()));
        let (mn, mx) = bounds(&verts);
        assert_eq!(mn, [-3.0, 0.0, -4.0]);
        assert_eq!(mx, [3.0, 2.5, 4.0]);
    }

    #[test]
    fn room_normals_point_into_the_interior() {
        // Every face's normal is a unit axis pointing at the room centre, which
        // for an origin-centred box means normal . position is negative.
        let (verts, _) = build_room_geometry(3.0, 4.0, -1.0, 1.0);
        for (pos, normal, ..) in &verts {
            let len =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-6, "normal {normal:?} is not unit");
            let dot = pos[0] * normal[0] + pos[1] * normal[1] + pos[2] * normal[2];
            assert!(dot < 0.0, "normal {normal:?} is not inward at {pos:?}");
        }
    }

    #[test]
    fn room_uvs_tile_once_per_metre() {
        let (verts, _) = build_room_geometry(3.0, 4.0, 0.0, 2.5);
        // The floor quad is emitted first and tiles across the full 6x8 extent.
        let floor_uvs: Vec<[f32; 2]> = verts[..4].iter().map(|v| v.3).collect();
        assert_eq!(
            floor_uvs,
            alloc::vec![[0.0, 0.0], [6.0, 0.0], [6.0, 8.0], [0.0, 8.0]]
        );
        // The north wall (third quad) tiles width by ceiling height.
        let wall_uvs: Vec<[f32; 2]> = verts[8..12].iter().map(|v| v.3).collect();
        assert_eq!(
            wall_uvs,
            alloc::vec![[0.0, 2.5], [6.0, 2.5], [6.0, 0.0], [0.0, 0.0]]
        );
    }
}
