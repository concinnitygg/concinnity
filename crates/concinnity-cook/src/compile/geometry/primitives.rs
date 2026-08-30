// JSON arg parsing for the box, cylinder, plane, and sphere generators; the
// geometry itself is `concinnity_core::geometry`.

type Verts = Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 2])>;
type GeomResult = Result<(Verts, Vec<u16>), String>;

pub(super) fn build_box(args: &serde_json::Value) -> GeomResult {
    let he =
        super::parse_f32x3(args.get("half_extents"), "half_extents").unwrap_or([0.5, 0.5, 0.5]);
    Ok(concinnity_core::geometry::build_box(he))
}

pub(super) fn build_cylinder(args: &serde_json::Value) -> GeomResult {
    let radius = args.get("radius").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
    let height = args.get("height").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let segments = args.get("segments").and_then(|v| v.as_u64()).unwrap_or(16) as u32;
    Ok(concinnity_core::geometry::build_cylinder(
        radius, height, segments,
    ))
}

pub(super) fn build_plane(args: &serde_json::Value) -> GeomResult {
    let half_width = args
        .get("half_width")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
    let half_depth = args
        .get("half_depth")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
    Ok(concinnity_core::geometry::build_plane(
        half_width, half_depth,
    ))
}

pub(super) fn build_sphere(args: &serde_json::Value) -> GeomResult {
    let radius = args.get("radius").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let rings = args.get("rings").and_then(|v| v.as_u64()).unwrap_or(12) as u32;
    let segments = args.get("segments").and_then(|v| v.as_u64()).unwrap_or(16) as u32;
    concinnity_core::geometry::build_sphere(radius, rings, segments)
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

    #[test]
    fn box_defaults_to_a_unit_cube() {
        let (verts, idxs) = build_box(&serde_json::json!({})).unwrap();
        assert_eq!(verts.len(), 6 * 4);
        assert_eq!(idxs.len(), 6 * 6);
        let (mn, mx) = bounds(&verts);
        assert_eq!(mn, [-0.5, -0.5, -0.5]);
        assert_eq!(mx, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn box_half_extents_set_the_bounds_and_uv_tiling() {
        let args = serde_json::json!({"half_extents": [2.0, 3.0, 4.0]});
        let (verts, _) = build_box(&args).unwrap();
        let (mn, mx) = bounds(&verts);
        assert_eq!(mn, [-2.0, -3.0, -4.0]);
        assert_eq!(mx, [2.0, 3.0, 4.0]);
        // The +Y face is emitted first and tiles x by z; the +Z face (third)
        // tiles x by y.
        assert_eq!(verts[2].3, [4.0, 8.0]);
        assert_eq!(verts[10].3, [4.0, 6.0]);
    }

    #[test]
    fn cylinder_defaults_to_sixteen_segments_and_clamps_below_three() {
        let (default_verts, _) = build_cylinder(&serde_json::json!({})).unwrap();
        assert_eq!(default_verts.len(), 2 * 16 + 2 * (1 + 16));
        let (mn, mx) = bounds(&default_verts);
        assert_eq!((mn[1], mx[1]), (-0.5, 0.5));

        let (thin, _) = build_cylinder(&serde_json::json!({"segments": 1})).unwrap();
        assert_eq!(thin.len(), 2 * 3 + 2 * (1 + 3));
    }

    #[test]
    fn cylinder_args_set_radius_height_and_segments() {
        let args = serde_json::json!({"radius": 2.0, "height": 4.0, "segments": 8});
        let (verts, idxs) = build_cylinder(&args).unwrap();
        assert_eq!(verts.len(), 2 * 8 + 2 * (1 + 8));
        assert_eq!(idxs.len(), 8 * 6 + 2 * (8 * 3));
        let (mn, mx) = bounds(&verts);
        assert_eq!((mn[1], mx[1]), (-2.0, 2.0));
    }

    #[test]
    fn plane_defaults_to_a_two_metre_square() {
        let (verts, _) = build_plane(&serde_json::json!({})).unwrap();
        let (mn, mx) = bounds(&verts);
        assert_eq!(mn, [-1.0, 0.0, -1.0]);
        assert_eq!(mx, [1.0, 0.0, 1.0]);
    }

    #[test]
    fn plane_args_set_the_extents() {
        let args = serde_json::json!({"half_width": 3.0, "half_depth": 5.0});
        let (verts, idxs) = build_plane(&args).unwrap();
        assert_eq!(verts.len(), 4);
        assert_eq!(idxs, vec![0, 1, 2, 2, 3, 0]);
        let (mn, mx) = bounds(&verts);
        assert_eq!(mn, [-3.0, 0.0, -5.0]);
        assert_eq!(mx, [3.0, 0.0, 5.0]);
    }

    #[test]
    fn sphere_defaults_clamp_to_a_buildable_minimum() {
        let (verts, _) = build_sphere(&serde_json::json!({"rings": 0, "segments": 0})).unwrap();
        assert_eq!(verts.len(), (2 + 1) * (3 + 1) + 2);
    }

    #[test]
    fn sphere_rejects_a_tessellation_past_the_u16_index_limit() {
        let args = serde_json::json!({"rings": 255, "segments": 255});
        let err = build_sphere(&args).unwrap_err();
        assert!(err.contains("65538 vertices"), "got: {err}");
        assert!(err.contains("u16"), "got: {err}");
    }

    #[test]
    fn sphere_args_set_the_tessellation() {
        let args = serde_json::json!({"radius": 2.0, "rings": 4, "segments": 6});
        let (verts, idxs) = build_sphere(&args).unwrap();
        assert_eq!(verts.len(), (4 + 1) * (6 + 1) + 2);
        assert_eq!(idxs.len(), 2 * 6 * 3 + (4 - 1) * 6 * 6);
    }
}
