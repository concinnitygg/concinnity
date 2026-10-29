//! The axis-aligned cube a pass rasterizes to cover a volume: a decal's box or a
//! raymarched SDF volume's bounds. The vertex shader scales and places it, and
//! the pass culls one face set so each covered pixel shades once.

/// Corners of the cube centered on the origin with half-width `half_extent`.
/// [`CUBE_INDICES`] indexes them.
pub const fn cube_corners(half_extent: f32) -> [[f32; 3]; 8] {
    let h = half_extent;
    [
        [-h, -h, -h],
        [h, -h, -h],
        [h, h, -h],
        [-h, h, -h],
        [-h, -h, h],
        [h, -h, h],
        [h, h, h],
        [-h, h, h],
    ]
}

/// The cube's 12 triangles over [`cube_corners`], wound counter-clockwise seen
/// from outside.
#[rustfmt::skip]
pub const CUBE_INDICES: [u16; 36] = [
    0, 2, 1,  0, 3, 2, // -Z
    4, 5, 6,  4, 6, 7, // +Z
    0, 4, 7,  0, 7, 3, // -X
    1, 2, 6,  1, 6, 5, // +X
    0, 1, 5,  0, 5, 4, // -Y
    3, 7, 6,  3, 6, 2, // +Y
];

#[cfg(test)]
mod tests {
    use super::*;

    // Every triangle's normal must point away from the cube's center.
    #[test]
    fn every_triangle_winds_outward() {
        let c = cube_corners(1.0);
        for tri in CUBE_INDICES.chunks(3) {
            let [a, b, d] = [c[tri[0] as usize], c[tri[1] as usize], c[tri[2] as usize]];
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ad = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
            let n = [
                ab[1] * ad[2] - ab[2] * ad[1],
                ab[2] * ad[0] - ab[0] * ad[2],
                ab[0] * ad[1] - ab[1] * ad[0],
            ];
            let centroid = [a[0] + b[0] + d[0], a[1] + b[1] + d[1], a[2] + b[2] + d[2]];
            let outward = n[0] * centroid[0] + n[1] * centroid[1] + n[2] * centroid[2];
            assert!(outward > 0.0, "triangle {tri:?} winds inward");
        }
    }

    #[test]
    fn corners_sit_at_the_half_extent() {
        for corner in cube_corners(0.5) {
            assert!(corner.iter().all(|v| v.abs() == 0.5));
        }
    }
}
