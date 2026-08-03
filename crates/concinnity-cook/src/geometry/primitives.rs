// src/geometry/primitives.rs: box, cylinder, plane, sphere generators.

type Verts = Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 2])>;
type GeomResult = Result<(Verts, Vec<u16>), String>;

// A box face: four corner positions, the outward normal, then the face width
// and height (used to scale UV tiling).
type BoxFace = ([f32; 3], [f32; 3], [f32; 3], [f32; 3], [f32; 3], f32, f32);

// Builds an axis-aligned box from half_extents [x, y, z].
//
// All six faces are included, wound CCW from the outside. Each face carries
// its outward-facing normal. UV coordinates tile once across each face,
// scaling with the face dimensions (one repeat per metre).
pub(super) fn build_box(args: &serde_json::Value) -> GeomResult {
    let he =
        super::parse_f32x3(args.get("half_extents"), "half_extents").unwrap_or([0.5, 0.5, 0.5]);
    let [hx, hy, hz] = he;

    let color = [0.75f32, 0.74, 0.72];

    let mut verts: Verts = Vec::new();
    let mut idxs: Vec<u16> = Vec::new();

    let faces: &[BoxFace] = &[
        // +Y (top)
        (
            [-hx, hy, hz],
            [hx, hy, hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [0.0, 1.0, 0.0],
            hx * 2.0,
            hz * 2.0,
        ),
        // -Y (bottom)
        (
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, -hy, hz],
            [-hx, -hy, hz],
            [0.0, -1.0, 0.0],
            hx * 2.0,
            hz * 2.0,
        ),
        // +Z (front)
        (
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
            [0.0, 0.0, 1.0],
            hx * 2.0,
            hy * 2.0,
        ),
        // -Z (back)
        (
            [hx, -hy, -hz],
            [-hx, -hy, -hz],
            [-hx, hy, -hz],
            [hx, hy, -hz],
            [0.0, 0.0, -1.0],
            hx * 2.0,
            hy * 2.0,
        ),
        // +X (right)
        (
            [hx, -hy, hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [hx, hy, hz],
            [1.0, 0.0, 0.0],
            hz * 2.0,
            hy * 2.0,
        ),
        // -X (left)
        (
            [-hx, -hy, -hz],
            [-hx, -hy, hz],
            [-hx, hy, hz],
            [-hx, hy, -hz],
            [-1.0, 0.0, 0.0],
            hz * 2.0,
            hy * 2.0,
        ),
    ];

    for (a, b, c, d, normal, u_max, v_max) in faces {
        let base = verts.len() as u16;
        verts.extend_from_slice(&[
            (*a, *normal, color, [0.0, 0.0]),
            (*b, *normal, color, [*u_max, 0.0]),
            (*c, *normal, color, [*u_max, *v_max]),
            (*d, *normal, color, [0.0, *v_max]),
        ]);
        idxs.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    Ok((verts, idxs))
}

// Builds an upright cylinder from radius, height, and segment count.
//
// Centred on the origin: bottom cap at y = -height/2, top cap at y = +height/2.
// Sides use cylindrical UV projection; caps use planar UV.
// segment_count defaults to 16 if omitted.
pub(super) fn build_cylinder(args: &serde_json::Value) -> GeomResult {
    let radius = args.get("radius").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
    let height = args.get("height").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let segments = args
        .get("segments")
        .and_then(|v| v.as_u64())
        .unwrap_or(16)
        .max(3) as usize;

    let half_h = height / 2.0;
    let side_color = [0.70f32, 0.68, 0.66];
    let cap_color = [0.60f32, 0.58, 0.56];

    let mut verts: Verts = Vec::new();
    let mut idxs: Vec<u16> = Vec::new();

    // sides: two rings (bottom then top)
    let side_base = verts.len() as u16;
    for ring in 0..=1 {
        let y = if ring == 0 { -half_h } else { half_h };
        for i in 0..segments {
            let t = i as f32 / segments as f32;
            let angle = t * std::f32::consts::TAU;
            let nx = angle.cos();
            let nz = angle.sin();
            let u = t * std::f32::consts::TAU * radius;
            let v = if ring == 0 { height } else { 0.0 };
            verts.push((
                [nx * radius, y, nz * radius],
                [nx, 0.0, nz],
                side_color,
                [u, v],
            ));
        }
    }
    for i in 0..segments {
        let next = (i + 1) % segments;
        let b = side_base + i as u16;
        let bn = side_base + next as u16;
        let t = b + segments as u16;
        let tn = bn + segments as u16;
        idxs.extend_from_slice(&[b, bn, tn, tn, t, b]);
    }

    // top cap; normal = +Y
    let top_base = verts.len() as u16;
    verts.push(([0.0, half_h, 0.0], [0.0, 1.0, 0.0], cap_color, [0.5, 0.5]));
    for i in 0..segments {
        let t = i as f32 / segments as f32;
        let angle = t * std::f32::consts::TAU;
        let x = angle.cos() * radius;
        let z = angle.sin() * radius;
        verts.push((
            [x, half_h, z],
            [0.0, 1.0, 0.0],
            cap_color,
            [0.5 + x / (radius * 2.0), 0.5 + z / (radius * 2.0)],
        ));
    }
    for i in 0..segments {
        let next = (i + 1) % segments;
        idxs.extend_from_slice(&[
            top_base,
            top_base + 1 + next as u16,
            top_base + 1 + i as u16,
        ]);
    }

    // bottom cap; normal = -Y
    let bot_base = verts.len() as u16;
    verts.push(([0.0, -half_h, 0.0], [0.0, -1.0, 0.0], cap_color, [0.5, 0.5]));
    for i in 0..segments {
        let t = i as f32 / segments as f32;
        let angle = t * std::f32::consts::TAU;
        let x = angle.cos() * radius;
        let z = angle.sin() * radius;
        verts.push((
            [x, -half_h, z],
            [0.0, -1.0, 0.0],
            cap_color,
            [0.5 + x / (radius * 2.0), 0.5 + z / (radius * 2.0)],
        ));
    }
    for i in 0..segments {
        let next = (i + 1) % segments;
        idxs.extend_from_slice(&[
            bot_base,
            bot_base + 1 + i as u16,
            bot_base + 1 + next as u16,
        ]);
    }

    Ok((verts, idxs))
}

// Builds a flat horizontal plane from half_width and half_depth.
//
// Lies in the XZ plane at Y = 0, facing up. UV tiles at one repeat per metre.
pub(super) fn build_plane(args: &serde_json::Value) -> GeomResult {
    let half_width = args
        .get("half_width")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
    let half_depth = args
        .get("half_depth")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;

    let normal = [0.0f32, 1.0, 0.0];
    let color = [0.80f32, 0.79, 0.78];
    let w = half_width * 2.0;
    let d = half_depth * 2.0;

    let verts = vec![
        ([-half_width, 0.0, -half_depth], normal, color, [0.0, 0.0]),
        ([half_width, 0.0, -half_depth], normal, color, [w, 0.0]),
        ([half_width, 0.0, half_depth], normal, color, [w, d]),
        ([-half_width, 0.0, half_depth], normal, color, [0.0, d]),
    ];
    let idxs = vec![0u16, 1, 2, 2, 3, 0];

    Ok((verts, idxs))
}

// Builds a UV sphere from radius, ring count, and segment count.
//
// Centred on the origin; poles at Y = ±radius. UV mapping is spherical.
// Normal at every point equals normalise(pos).
//
// Parameters:
//   radius   -- sphere radius (default 1.0)
//   rings    -- latitudinal divisions between the poles (default 12, min 2)
//   segments -- longitudinal divisions around the equator (default 16, min 3)
pub(super) fn build_sphere(args: &serde_json::Value) -> GeomResult {
    let radius = args.get("radius").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let rings = (args
        .get("rings")
        .and_then(|v| v.as_u64())
        .unwrap_or(12)
        .max(2)) as usize;
    let segments = (args
        .get("segments")
        .and_then(|v| v.as_u64())
        .unwrap_or(16)
        .max(3)) as usize;

    let vert_count = (rings + 1) * (segments + 1) + 2;
    if vert_count > 65536 {
        return Err(format!(
            "sphere rings={} segments={} produces {} vertices, exceeding the u16 limit",
            rings, segments, vert_count
        ));
    }

    let color = [0.82f32, 0.80, 0.78];
    let mut verts: Verts = Vec::new();
    let mut idxs: Vec<u16> = Vec::new();

    // theta depends only on the segment, so the ring loop would otherwise
    // recompute the same sin/cos pair once per ring.
    let ring_angles: Vec<(f32, f32)> = (0..=segments)
        .map(|seg| {
            let theta = std::f32::consts::TAU * seg as f32 / segments as f32;
            theta.sin_cos()
        })
        .collect();

    for ring in 0..=rings {
        let phi = std::f32::consts::PI * (ring as f32 + 1.0) / (rings as f32 + 1.0);
        let (sin_phi, cos_phi) = phi.sin_cos();
        for (seg, &(sin_theta, cos_theta)) in ring_angles.iter().enumerate() {
            let nx = sin_phi * cos_theta;
            let ny = cos_phi;
            let nz = sin_phi * sin_theta;
            let u = seg as f32 / segments as f32;
            let v = (ring as f32 + 1.0) / (rings as f32 + 1.0);
            verts.push((
                [nx * radius, ny * radius, nz * radius],
                [nx, ny, nz],
                color,
                [u, v],
            ));
        }
    }

    // north pole cap
    let pole_n = verts.len() as u16;
    verts.push(([0.0, radius, 0.0], [0.0, 1.0, 0.0], color, [0.5, 0.0]));
    for seg in 0..segments {
        idxs.extend_from_slice(&[pole_n, seg as u16, (seg + 1) as u16]);
    }

    // south pole cap
    let pole_s = verts.len() as u16;
    verts.push(([0.0, -radius, 0.0], [0.0, -1.0, 0.0], color, [0.5, 1.0]));
    let last_ring_start = ((rings - 1) * (segments + 1)) as u16;
    for seg in 0..segments {
        idxs.extend_from_slice(&[
            pole_s,
            last_ring_start + (seg + 1) as u16,
            last_ring_start + seg as u16,
        ]);
    }

    // quads between adjacent rings
    for ring in 0..rings - 1 {
        let row0 = (ring * (segments + 1)) as u16;
        let row1 = row0 + (segments + 1) as u16;
        for seg in 0..segments {
            let tl = row0 + seg as u16;
            let tr = row0 + (seg + 1) as u16;
            let bl = row1 + seg as u16;
            let br = row1 + (seg + 1) as u16;
            idxs.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }

    Ok((verts, idxs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(verts: &Verts) -> ([f32; 3], [f32; 3]) {
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

    fn length(v: [f32; 3]) -> f32 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    // Indices address real vertices and no triangle collapses to a line.
    fn assert_triangles_are_well_formed(verts: &Verts, idxs: &[u16]) {
        assert_eq!(idxs.len() % 3, 0);
        assert!(idxs.iter().all(|&i| (i as usize) < verts.len()));
        for tri in idxs.chunks_exact(3) {
            let a = verts[tri[0] as usize].0;
            let b = verts[tri[1] as usize].0;
            let c = verts[tri[2] as usize].0;
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            assert!(length(cross) > 1e-9, "degenerate triangle {tri:?}");
        }
    }

    #[test]
    fn box_defaults_to_a_unit_cube_of_six_quads() {
        let (verts, idxs) = build_box(&serde_json::json!({})).unwrap();
        assert_eq!(verts.len(), 6 * 4);
        assert_eq!(idxs.len(), 6 * 6);
        assert_triangles_are_well_formed(&verts, &idxs);
        let (mn, mx) = bounds(&verts);
        assert_eq!(mn, [-0.5, -0.5, -0.5]);
        assert_eq!(mx, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn box_half_extents_set_the_bounds_and_uv_tiling() {
        let args = serde_json::json!({"half_extents": [2.0, 3.0, 4.0]});
        let (verts, idxs) = build_box(&args).unwrap();
        assert_triangles_are_well_formed(&verts, &idxs);
        let (mn, mx) = bounds(&verts);
        assert_eq!(mn, [-2.0, -3.0, -4.0]);
        assert_eq!(mx, [2.0, 3.0, 4.0]);
        // The +Y face is emitted first and tiles x by z; the +Z face (third)
        // tiles x by y.
        assert_eq!(verts[2].3, [4.0, 8.0]);
        assert_eq!(verts[10].3, [4.0, 6.0]);
    }

    #[test]
    fn box_carries_one_outward_axis_normal_per_face() {
        let (verts, _) = build_box(&serde_json::json!({})).unwrap();
        let mut seen: Vec<[f32; 3]> = Vec::new();
        for face in verts.chunks_exact(4) {
            let n = face[0].1;
            assert!((length(n) - 1.0).abs() < 1e-6);
            // The normal points away from the origin at every corner of its face.
            for (pos, normal, ..) in face {
                assert_eq!(*normal, n);
                let dot = pos[0] * n[0] + pos[1] * n[1] + pos[2] * n[2];
                assert!(dot > 0.0, "normal {n:?} is not outward at {pos:?}");
            }
            assert!(!seen.contains(&n), "duplicate face normal {n:?}");
            seen.push(n);
        }
        assert_eq!(seen.len(), 6);
    }

    #[test]
    fn cylinder_counts_follow_the_segment_count() {
        // Sides are two rings of `segments`; each cap adds a centre plus a ring.
        let args = serde_json::json!({"radius": 2.0, "height": 4.0, "segments": 8});
        let (verts, idxs) = build_cylinder(&args).unwrap();
        assert_eq!(verts.len(), 2 * 8 + 2 * (1 + 8));
        assert_eq!(idxs.len(), 8 * 6 + 2 * (8 * 3));
        assert_triangles_are_well_formed(&verts, &idxs);

        let (mn, mx) = bounds(&verts);
        assert_eq!((mn[1], mx[1]), (-2.0, 2.0));
        for (pos, ..) in &verts {
            let radial = (pos[0] * pos[0] + pos[2] * pos[2]).sqrt();
            assert!(radial <= 2.0 + 1e-5, "vertex outside the radius: {pos:?}");
        }
    }

    #[test]
    fn cylinder_side_normals_are_horizontal_and_radial() {
        let args = serde_json::json!({"radius": 3.0, "height": 1.0, "segments": 6});
        let (verts, _) = build_cylinder(&args).unwrap();
        for (pos, normal, ..) in &verts[..12] {
            assert_eq!(normal[1], 0.0);
            assert!((length(*normal) - 1.0).abs() < 1e-5);
            // The side normal is the vertex's own outward radial direction.
            assert!((normal[0] * 3.0 - pos[0]).abs() < 1e-4);
            assert!((normal[2] * 3.0 - pos[2]).abs() < 1e-4);
        }
        // Cap centres sit on the axis with the cap's axial normal.
        assert_eq!(verts[12].0, [0.0, 0.5, 0.0]);
        assert_eq!(verts[12].1, [0.0, 1.0, 0.0]);
        assert_eq!(verts[19].0, [0.0, -0.5, 0.0]);
        assert_eq!(verts[19].1, [0.0, -1.0, 0.0]);
    }

    #[test]
    fn cylinder_defaults_to_sixteen_segments_and_clamps_below_three() {
        let (default_verts, _) = build_cylinder(&serde_json::json!({})).unwrap();
        assert_eq!(default_verts.len(), 2 * 16 + 2 * (1 + 16));
        let (mn, mx) = bounds(&default_verts);
        assert_eq!((mn[1], mx[1]), (-0.5, 0.5));

        let (thin, idxs) = build_cylinder(&serde_json::json!({"segments": 1})).unwrap();
        assert_eq!(thin.len(), 2 * 3 + 2 * (1 + 3));
        assert_triangles_are_well_formed(&thin, &idxs);
    }

    #[test]
    fn plane_is_one_upward_facing_quad_tiled_per_metre() {
        let args = serde_json::json!({"half_width": 3.0, "half_depth": 5.0});
        let (verts, idxs) = build_plane(&args).unwrap();
        assert_eq!(verts.len(), 4);
        assert_eq!(idxs, vec![0, 1, 2, 2, 3, 0]);
        assert_triangles_are_well_formed(&verts, &idxs);
        assert!(verts.iter().all(|(pos, ..)| pos[1] == 0.0));
        assert!(verts.iter().all(|(_, n, ..)| *n == [0.0, 1.0, 0.0]));
        let (mn, mx) = bounds(&verts);
        assert_eq!(mn, [-3.0, 0.0, -5.0]);
        assert_eq!(mx, [3.0, 0.0, 5.0]);
        assert_eq!(verts[2].3, [6.0, 10.0]);
    }

    #[test]
    fn plane_defaults_to_a_two_metre_square() {
        let (verts, _) = build_plane(&serde_json::json!({})).unwrap();
        let (mn, mx) = bounds(&verts);
        assert_eq!(mn, [-1.0, 0.0, -1.0]);
        assert_eq!(mx, [1.0, 0.0, 1.0]);
    }

    #[test]
    fn sphere_counts_follow_the_ring_and_segment_counts() {
        let args = serde_json::json!({"radius": 2.0, "rings": 4, "segments": 6});
        let (verts, idxs) = build_sphere(&args).unwrap();
        assert_eq!(verts.len(), (4 + 1) * (6 + 1) + 2);
        // Two pole fans plus a quad band between each adjacent ring pair.
        assert_eq!(idxs.len(), 2 * 6 * 3 + (4 - 1) * 6 * 6);
        assert_triangles_are_well_formed(&verts, &idxs);
    }

    #[test]
    fn every_sphere_vertex_sits_on_the_radius_with_a_radial_normal() {
        let args = serde_json::json!({"radius": 3.0, "rings": 5, "segments": 8});
        let (verts, _) = build_sphere(&args).unwrap();
        for (pos, normal, ..) in &verts {
            assert!((length(*pos) - 3.0).abs() < 1e-4, "off-radius {pos:?}");
            assert!((length(*normal) - 1.0).abs() < 1e-5);
            for k in 0..3 {
                assert!((normal[k] * 3.0 - pos[k]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn sphere_clamps_rings_and_segments_to_a_buildable_minimum() {
        let (verts, idxs) = build_sphere(&serde_json::json!({"rings": 0, "segments": 0})).unwrap();
        assert_eq!(verts.len(), (2 + 1) * (3 + 1) + 2);
        assert_triangles_are_well_formed(&verts, &idxs);
    }

    #[test]
    fn sphere_rejects_a_tessellation_past_the_u16_index_limit() {
        let args = serde_json::json!({"rings": 255, "segments": 255});
        let err = build_sphere(&args).unwrap_err();
        assert!(err.contains("65538 vertices"), "got: {err}");
        assert!(err.contains("u16"), "got: {err}");
    }
}

// No winding-direction unit tests live here. A previous iteration added an
// "outward winding" check and silently flipped every primitive's index order
// to make it pass, which produced a renderer-visible regression because
// the rasterizer pipeline empirically expects the *opposite* winding for
// the sphere / cylinder-side / plane triangles. The mesh as a whole is not
// even uniformly wound (cylinder caps and the box wind outward; everything
// else winds inward), and the project's Metal pipeline doesn't enable
// back-face culling, so any per-primitive winding test is misleading. Keep
// the index orders in this file untouched unless you have re-validated the
// full render with the showcase world.
