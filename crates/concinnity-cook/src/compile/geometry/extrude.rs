// JSON arg parsing for the extrude generator (a 2D profile in the XZ plane
// extruded along Y, authored by the macOS Mesh Editor); the geometry itself is
// `concinnity_core::geometry::build_extrude`.

type Verts = Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 2])>;
type GeomResult = Result<(Verts, Vec<u16>), String>;

pub(super) fn build_extrude(args: &serde_json::Value) -> GeomResult {
    let profile_raw = args
        .get("profile")
        .and_then(|v| v.as_array())
        .ok_or("extrude requires a `profile` array of [x, z] pairs")?;

    let mut profile: Vec<[f32; 2]> = Vec::with_capacity(profile_raw.len());
    for (i, p) in profile_raw.iter().enumerate() {
        let arr = p
            .as_array()
            .ok_or_else(|| format!("profile[{i}] must be a 2-element [x, z] array"))?;
        if arr.len() < 2 {
            return Err(format!(
                "profile[{i}] must have 2 elements, got {}",
                arr.len()
            ));
        }
        let x = arr[0]
            .as_f64()
            .ok_or_else(|| format!("profile[{i}][0] must be a number"))? as f32;
        let z = arr[1]
            .as_f64()
            .ok_or_else(|| format!("profile[{i}][1] must be a number"))? as f32;
        profile.push([x, z]);
    }

    let height = args.get("height").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let corner_radius = args
        .get("corner_radius")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let corner_segments = args
        .get("corner_segments")
        .and_then(|v| v.as_u64())
        .unwrap_or(8) as u32;

    concinnity_core::geometry::build_extrude(&profile, height, corner_radius, corner_segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extrude_args(profile: serde_json::Value, extras: serde_json::Value) -> serde_json::Value {
        let mut obj = serde_json::Map::new();
        obj.insert("generator".into(), "extrude".into());
        obj.insert("profile".into(), profile);
        if let Some(map) = extras.as_object() {
            for (k, v) in map {
                obj.insert(k.clone(), v.clone());
            }
        }
        serde_json::Value::Object(obj)
    }

    #[test]
    fn build_extrude_square() {
        let profile = serde_json::json!([[-1, -1], [1, -1], [1, 1], [-1, 1]]);
        let (verts, idxs) =
            build_extrude(&extrude_args(profile, serde_json::json!({"height": 2.0}))).unwrap();
        assert!(!verts.is_empty());
        assert!(!idxs.is_empty());
        assert_eq!(idxs.len() % 3, 0);
        // Square: top + bottom (4 each) + 4 side walls (4 verts each) = 24 verts.
        assert_eq!(verts.len(), 24);
    }

    #[test]
    fn build_extrude_rejects_too_few_points() {
        let profile = serde_json::json!([[0, 0], [1, 0]]);
        let err = build_extrude(&extrude_args(profile, serde_json::json!({}))).unwrap_err();
        assert!(err.contains("at least 3"));
    }

    #[test]
    fn build_extrude_requires_a_profile_array() {
        let err = build_extrude(&serde_json::json!({"height": 1.0})).unwrap_err();
        assert!(err.contains("`profile` array"), "got: {err}");
        let err = build_extrude(&serde_json::json!({"profile": 3})).unwrap_err();
        assert!(err.contains("`profile` array"), "got: {err}");
    }

    #[test]
    fn build_extrude_rejects_malformed_profile_points() {
        let cases: [(serde_json::Value, &str); 4] = [
            (
                serde_json::json!([[0, 0], [1, 0], "nope"]),
                "profile[2] must be a 2-element",
            ),
            (
                serde_json::json!([[0, 0], [1, 0], [1]]),
                "profile[2] must have 2 elements, got 1",
            ),
            (
                serde_json::json!([[0, 0], ["x", 0], [1, 1]]),
                "profile[1][0] must be a number",
            ),
            (
                serde_json::json!([[0, 0], [0, "z"], [1, 1]]),
                "profile[1][1] must be a number",
            ),
        ];
        for (profile, expected) in cases {
            let err = build_extrude(&extrude_args(profile, serde_json::json!({}))).unwrap_err();
            assert!(err.contains(expected), "expected '{expected}', got: {err}");
        }
    }

    #[test]
    fn build_extrude_rejects_non_positive_height() {
        let profile = serde_json::json!([[0, 0], [1, 0], [1, 1]]);
        let err =
            build_extrude(&extrude_args(profile, serde_json::json!({"height": 0.0}))).unwrap_err();
        assert!(err.contains("positive"), "got: {err}");
    }

    #[test]
    fn build_extrude_rejects_negative_corner_radius() {
        let profile = serde_json::json!([[0, 0], [1, 0], [1, 1]]);
        let err = build_extrude(&extrude_args(
            profile,
            serde_json::json!({"corner_radius": -0.5}),
        ))
        .unwrap_err();
        assert!(err.contains("non-negative"), "got: {err}");
    }

    #[test]
    fn corner_rounding_adds_arc_points_on_convex_corners() {
        let profile = serde_json::json!([[-1, -1], [1, -1], [1, 1], [-1, 1]]);
        let sharp = build_extrude(&extrude_args(
            profile.clone(),
            serde_json::json!({"corner_segments": 4}),
        ))
        .unwrap()
        .0
        .len();
        let rounded = build_extrude(&extrude_args(
            profile,
            serde_json::json!({"corner_radius": 0.2, "corner_segments": 4}),
        ))
        .unwrap()
        .0
        .len();
        assert!(rounded > sharp, "rounded {rounded} <= sharp {sharp}");
    }
}
