// src/geometry/terrain.rs: subdivided terrain grid with deterministic height displacement.
//
// The grid spans [-half_width, half_width] × [-half_depth, half_depth] with
// (subdivisions+1)^2 vertices. Heights are computed by three octaves of value
// noise driven by lcg_hash so output is identical across builds. Smooth vertex
// normals are computed by accumulating face normals from all sharing triangles.
//
// Parameters:
//   half_width   -- half the terrain extent along X (default 64.0)
//   half_depth   -- half the terrain extent along Z (default 64.0)
//   subdivisions -- grid resolution per axis (default 64, clamped 4..=255)
//   amplitude    -- peak height above the base plane in metres (default 4.0)

type Verts = Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 2])>;

pub(super) fn build_terrain(args: &serde_json::Value) -> Result<(Verts, Vec<u16>), String> {
    let half_width = args
        .get("half_width")
        .and_then(|v| v.as_f64())
        .unwrap_or(64.0) as f32;
    let half_depth = args
        .get("half_depth")
        .and_then(|v| v.as_f64())
        .unwrap_or(64.0) as f32;
    let subdivisions = args
        .get("subdivisions")
        .and_then(|v| v.as_u64())
        .unwrap_or(64)
        .clamp(4, 255) as usize;
    let amplitude = args
        .get("amplitude")
        .and_then(|v| v.as_f64())
        .unwrap_or(4.0) as f32;

    let cols = subdivisions + 1;
    let rows = subdivisions + 1;

    if cols * rows > 65536 {
        return Err(format!(
            "terrain subdivisions {} produces {} vertices, exceeding the u16 limit; use subdivisions ≤ 255",
            subdivisions,
            cols * rows
        ));
    }

    let color = [0.55f32, 0.62, 0.42];

    // pass 1: compute all positions
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        for col in 0..cols {
            let s = col as f32 / subdivisions as f32;
            let t = row as f32 / subdivisions as f32;
            let x = -half_width + s * half_width * 2.0;
            let z = -half_depth + t * half_depth * 2.0;
            let y = terrain_height(col as u32, row as u32, subdivisions as u32, amplitude);
            positions.push([x, y, z]);
        }
    }

    // pass 2: accumulate face normals at each vertex
    let mut normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; cols * rows];
    for row in 0..subdivisions {
        for col in 0..subdivisions {
            let tl = row * cols + col;
            let tr = tl + 1;
            let bl = tl + cols;
            let br = bl + 1;
            let n1 = super::vec3_face_normal(positions[tl], positions[bl], positions[tr]);
            super::vec3_add(&mut normals[tl], n1);
            super::vec3_add(&mut normals[bl], n1);
            super::vec3_add(&mut normals[tr], n1);
            let n2 = super::vec3_face_normal(positions[tr], positions[bl], positions[br]);
            super::vec3_add(&mut normals[tr], n2);
            super::vec3_add(&mut normals[bl], n2);
            super::vec3_add(&mut normals[br], n2);
        }
    }

    let mut idxs: Vec<u16> = Vec::with_capacity(subdivisions * subdivisions * 6);
    let mut verts: Verts = Vec::with_capacity(cols * rows);

    for i in 0..cols * rows {
        let [x, y, z] = positions[i];
        let normal = super::vec3_normalise(normals[i]);
        verts.push(([x, y, z], normal, color, [x, z]));
    }

    for row in 0..subdivisions {
        for col in 0..subdivisions {
            let tl = (row * cols + col) as u16;
            let tr = tl + 1;
            let bl = tl + cols as u16;
            let br = bl + 1;
            idxs.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }

    Ok((verts, idxs))
}

// Returns the Y displacement for lattice position (col, row).
// Three octaves of deterministic value noise give coarse hills, medium bumps,
// and fine surface variation.
fn terrain_height(col: u32, row: u32, subdivisions: u32, amplitude: f32) -> f32 {
    let octaves: &[(u32, f32)] = &[(1, 1.00), (3, 0.40), (9, 0.15)];

    let mut sum = 0.0f32;
    let mut weight_sum = 0.0f32;

    for &(divisor, weight) in octaves {
        let scale = (subdivisions / divisor).max(1);
        let gx = col / scale;
        let gy = row / scale;
        let fx = (col % scale) as f32 / scale as f32;
        let fy = (row % scale) as f32 / scale as f32;

        let h00 = lattice_val(gx, gy);
        let h10 = lattice_val(gx + 1, gy);
        let h01 = lattice_val(gx, gy + 1);
        let h11 = lattice_val(gx + 1, gy + 1);
        let top = h00 + (h10 - h00) * fx;
        let bot = h01 + (h11 - h01) * fx;
        sum += (top + (bot - top) * fy) * weight;
        weight_sum += weight;
    }

    let normalised = sum / weight_sum;
    (normalised - 0.05).max(0.0) * amplitude
}

fn lattice_val(x: u32, y: u32) -> f32 {
    let h = lcg_hash(x.wrapping_mul(1619).wrapping_add(y.wrapping_mul(31337)));
    (h & 0xFF) as f32 / 255.0
}

fn lcg_hash(mut v: u32) -> u32 {
    v = v.wrapping_mul(1664525).wrapping_add(1013904223);
    v ^= v >> 16;
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_counts_and_extents_follow_the_subdivision_count() {
        let args = serde_json::json!({
            "half_width": 10.0, "half_depth": 5.0, "subdivisions": 8, "amplitude": 3.0,
        });
        let (verts, idxs) = build_terrain(&args).unwrap();
        assert_eq!(verts.len(), 9 * 9);
        assert_eq!(idxs.len(), 8 * 8 * 6);
        assert!(idxs.iter().all(|&i| (i as usize) < verts.len()));

        let mut mn = [f32::INFINITY; 3];
        let mut mx = [f32::NEG_INFINITY; 3];
        for (pos, ..) in &verts {
            for k in 0..3 {
                mn[k] = mn[k].min(pos[k]);
                mx[k] = mx[k].max(pos[k]);
            }
        }
        assert_eq!((mn[0], mx[0]), (-10.0, 10.0));
        assert_eq!((mn[2], mx[2]), (-5.0, 5.0));
        // Heights stay inside [0, amplitude] and vary across the grid.
        assert!(mn[1] >= 0.0 && mx[1] <= 3.0);
        assert!(mx[1] > mn[1], "expected height variation, got flat {mx:?}");
    }

    #[test]
    fn every_vertex_normal_is_unit_length() {
        let args = serde_json::json!({"subdivisions": 6, "amplitude": 5.0});
        let (verts, _) = build_terrain(&args).unwrap();
        for (_, normal, ..) in &verts {
            let len =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "normal {normal:?} is not unit");
            assert!(normal[1] > 0.0, "terrain normal points down: {normal:?}");
        }
    }

    #[test]
    fn zero_amplitude_produces_a_flat_grid() {
        let args = serde_json::json!({"subdivisions": 4, "amplitude": 0.0});
        let (verts, _) = build_terrain(&args).unwrap();
        assert!(verts.iter().all(|(pos, ..)| pos[1] == 0.0));
        assert!(verts.iter().all(|(_, n, ..)| *n == [0.0, 1.0, 0.0]));
    }

    #[test]
    fn subdivisions_clamp_to_the_supported_range() {
        // Below the floor clamps to 4 (5x5 lattice)...
        let (small, _) = build_terrain(&serde_json::json!({"subdivisions": 0})).unwrap();
        assert_eq!(small.len(), 5 * 5);
        // ...and above the ceiling clamps to 255, the largest grid that still
        // indexes with u16.
        let (large, idxs) = build_terrain(&serde_json::json!({"subdivisions": 4096})).unwrap();
        assert_eq!(large.len(), 256 * 256);
        assert_eq!(idxs.len(), 255 * 255 * 6);
    }

    #[test]
    fn terrain_height_is_deterministic_and_scales_with_amplitude() {
        let a = terrain_height(3, 7, 32, 4.0);
        assert_eq!(a, terrain_height(3, 7, 32, 4.0));
        assert!((terrain_height(3, 7, 32, 8.0) - a * 2.0).abs() < 1e-5);
        // The noise is floored at the base plane, never negative.
        for col in 0..32u32 {
            for row in 0..32u32 {
                assert!(terrain_height(col, row, 32, 4.0) >= 0.0);
            }
        }
    }

    #[test]
    fn lattice_values_are_normalised_and_position_dependent() {
        for x in 0..16u32 {
            for y in 0..16u32 {
                let v = lattice_val(x, y);
                assert!((0.0..=1.0).contains(&v), "lattice_val({x},{y}) = {v}");
            }
        }
        assert_ne!(lattice_val(0, 0), lattice_val(1, 0));
        assert_ne!(lcg_hash(0), lcg_hash(1));
    }
}
