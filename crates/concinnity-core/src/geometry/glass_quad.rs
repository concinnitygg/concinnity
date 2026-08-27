//! src/geometry/glass_quad.rs: flat rectangular quad for a GlassPanel.
//!
//! Builds a single 4-vertex / 6-index quad centred at `centre`, facing
//! `normal`, sized by `half_size` (half-width along the panel tangent,
//! half-height along its bitangent). The tangent frame is derived from the
//! normal so the panel can face any direction. Per-vertex normals are the
//! (constant) panel normal; the fragment shader flips it toward the viewer so
//! the panel is two-sided.

use crate::math::vec3::{cross, length};
use alloc::vec;
use alloc::vec::Vec;

type Verts = Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 2])>;

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = length(v);
    if len < 1e-6 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

/// Build the quad geometry for one glass panel. Returns 4 vertices (in the
/// shared `(pos, normal, color, uv)` layout the mesh builders use) and 6
/// indices (two triangles). `color` is a white placeholder; the glass
/// fragment shader ignores per-vertex colour.
/// Orthonormal (width, height) axes spanning the plane of `n`, which must be
/// unit length. Shared with the rectangular area light so a panel and a light
/// with the same normal agree on which way is "across".
pub fn plane_basis(n: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    // A world up reference that is not parallel to the normal.
    let up_ref = if n[1].abs() > 0.99 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let tangent = normalize(cross(up_ref, n));
    let bitangent = cross(n, tangent); // already unit
    (tangent, bitangent)
}

/// Build the quad for a glass panel: its vertices and triangle indices.
pub fn build_glass_quad(
    centre: [f32; 3],
    normal: [f32; 3],
    half_size: [f32; 2],
) -> (Verts, Vec<u16>) {
    let n = normalize(normal);
    let hw = half_size[0].max(1e-3);
    let hh = half_size[1].max(1e-3);

    let (tangent, bitangent) = plane_basis(n);

    let corner = |su: f32, sv: f32| -> [f32; 3] {
        [
            centre[0] + tangent[0] * su * hw + bitangent[0] * sv * hh,
            centre[1] + tangent[1] * su * hw + bitangent[1] * sv * hh,
            centre[2] + tangent[2] * su * hw + bitangent[2] * sv * hh,
        ]
    };

    let color = [1.0f32, 1.0, 1.0];
    let verts: Verts = vec![
        (corner(-1.0, -1.0), n, color, [0.0, 0.0]),
        (corner(1.0, -1.0), n, color, [1.0, 0.0]),
        (corner(1.0, 1.0), n, color, [1.0, 1.0]),
        (corner(-1.0, 1.0), n, color, [0.0, 1.0]),
    ];
    // Two triangles, CCW when viewed from the +normal side. The transparent
    // pass renders with no face culling, so winding only sets the default
    // front face; the shader is two-sided regardless.
    let idxs: Vec<u16> = vec![0, 1, 2, 0, 2, 3];

    (verts, idxs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::vec3::dot;

    #[test]
    fn quad_has_four_verts_six_indices() {
        let (v, i) = build_glass_quad([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [2.0, 1.0]);
        assert_eq!(v.len(), 4);
        assert_eq!(i.len(), 6);
    }

    #[test]
    fn vertices_are_coplanar_with_panel_normal() {
        let centre = [1.0, 2.0, -3.0];
        let normal = [0.0, 0.0, 1.0];
        let (v, _) = build_glass_quad(centre, normal, [2.0, 1.5]);
        for (pos, vn, _, _) in &v {
            // Each corner lies in the plane through centre with the panel normal.
            let rel = [pos[0] - centre[0], pos[1] - centre[1], pos[2] - centre[2]];
            assert!(dot(rel, normal).abs() < 1e-5);
            assert_eq!(*vn, normal);
        }
    }

    #[test]
    fn half_size_controls_extent() {
        // Axis-aligned panel facing +Z: width spans X, height spans Y.
        let (v, _) = build_glass_quad([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [3.0, 1.0]);
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for (pos, _, _, _) in &v {
            min_x = min_x.min(pos[0]);
            max_x = max_x.max(pos[0]);
            min_y = min_y.min(pos[1]);
            max_y = max_y.max(pos[1]);
        }
        assert!((max_x - min_x - 6.0).abs() < 1e-4);
        assert!((max_y - min_y - 2.0).abs() < 1e-4);
    }

    #[test]
    fn degenerate_normal_falls_back() {
        let (v, _) = build_glass_quad([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0]);
        // Falls back to +Z normal; quad is still well-formed.
        assert_eq!(v.len(), 4);
        for (_, vn, _, _) in &v {
            assert_eq!(*vn, [0.0, 0.0, 1.0]);
        }
    }
}
