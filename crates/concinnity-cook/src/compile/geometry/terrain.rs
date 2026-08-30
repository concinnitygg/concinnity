// JSON arg parsing for the terrain generator; the geometry itself is
// `concinnity_core::geometry::build_terrain`.

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
        .min(u64::from(u32::MAX)) as u32;
    let amplitude = args
        .get("amplitude")
        .and_then(|v| v.as_f64())
        .unwrap_or(4.0) as f32;
    concinnity_core::geometry::build_terrain(half_width, half_depth, subdivisions, amplitude)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrain_args_set_extents_and_resolution() {
        let args = serde_json::json!({
            "half_width": 10.0, "half_depth": 5.0, "subdivisions": 8, "amplitude": 3.0,
        });
        let (verts, idxs) = build_terrain(&args).unwrap();
        assert_eq!(verts.len(), 9 * 9);
        assert_eq!(idxs.len(), 8 * 8 * 6);
    }

    #[test]
    fn terrain_defaults_to_a_sixty_four_subdivision_grid() {
        let (verts, _) = build_terrain(&serde_json::json!({})).unwrap();
        assert_eq!(verts.len(), 65 * 65);
    }

    #[test]
    fn oversized_subdivisions_clamp_to_the_u16_limit() {
        let (large, idxs) = build_terrain(&serde_json::json!({"subdivisions": 4096})).unwrap();
        assert_eq!(large.len(), 256 * 256);
        assert_eq!(idxs.len(), 255 * 255 * 6);
    }
}
