// JSON arg parsing for the skybox generator; the geometry itself is
// `concinnity_core::geometry::build_skybox`.

type Verts = Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 2])>;

pub(super) fn build_skybox(args: &serde_json::Value) -> Result<(Verts, Vec<u16>), String> {
    let size = args.get("size").and_then(|v| v.as_f64()).unwrap_or(490.0) as f32;
    Ok(concinnity_core::geometry::build_skybox(size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skybox_defaults_to_a_490_metre_half_extent() {
        let (verts, idxs) = build_skybox(&serde_json::json!({})).unwrap();
        assert_eq!(verts.len(), 6 * 4);
        assert_eq!(idxs.len(), 6 * 6);
        for (pos, ..) in &verts {
            for axis in pos {
                assert_eq!(axis.abs(), 490.0, "corner off the cube: {pos:?}");
            }
        }
    }

    #[test]
    fn skybox_size_arg_scales_every_corner() {
        let (verts, _) = build_skybox(&serde_json::json!({"size": 12.5})).unwrap();
        for (pos, ..) in &verts {
            for axis in pos {
                assert_eq!(axis.abs(), 12.5);
            }
        }
    }
}
