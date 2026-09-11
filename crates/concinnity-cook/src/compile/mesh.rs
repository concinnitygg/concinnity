//! Build-time mesh payload entry point. Most generators compile with no source
//! decode and delegate straight to `crate::compile::geometry`; the
//! `heightfield` generator needs its source image decoded (the runtime crate
//! links no image decoders), so this crate decodes it here and hands the
//! pixels to `crate::compile::geometry::compile_heightfield_payload`.

/// Compile a Mesh / ProceduralMesh component's JSON args into a packed binary
/// payload. The `heightfield` generator's source PNG is decoded here in the
/// build crate; every other generator delegates to
/// `crate::compile::geometry::compile_mesh_payload`.
pub fn compile_mesh_payload(
    args: &serde_json::Value,
    assets_dir: Option<&std::path::Path>,
) -> Result<Vec<u8>, String> {
    if args.get("generator").and_then(|v| v.as_str()) == Some("heightfield") {
        let source = args
            .get("source")
            .and_then(|v| v.as_str())
            .ok_or("heightfield generator requires a `source` PNG path")?;
        let (w, h, rgba) = crate::compile::texture::decode_source(source, 0, assets_dir)
            .map_err(|e| format!("heightfield: {e}"))?;
        crate::compile::geometry::compile_heightfield_payload(args, w, h, rgba)
    } else {
        crate::compile::geometry::compile_mesh_payload(args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heightfield_end_to_end_via_tmp_png() {
        // Verify the build-side path decodes a PNG, generates the grid, and
        // bakes a collider trailer whose heights equal the rendered mesh's
        // per-vertex Y.
        let tree = concinnity_testing::TempTree::new();
        // A diagonal ramp 0..63 across the 8x8 grid.
        let pixels: Vec<u8> = (0..64u8).collect();
        let path = tree.write(
            "heightfield.png",
            concinnity_testing::fixtures::png::gray(8, 8, &pixels),
        );

        let args = serde_json::json!({
            "generator": "heightfield",
            "half_width": 5.0,
            "half_depth": 5.0,
            "subdivisions": 4,
            "source": path.to_str().unwrap(),
            "elevation_min": 0.0,
            "elevation_max": 10.0,
        });
        let payload = compile_mesh_payload(&args, None).expect("compiles");

        let grid = concinnity_core::gfx::mesh_payload::deserialise_heightfield(&payload)
            .expect("parse")
            .expect("heightfield trailer present");
        assert_eq!((grid.rows, grid.cols), (5, 5));
        let (mesh_verts, _, _) =
            concinnity_core::gfx::mesh_payload::deserialise_with_lods(&payload)
                .expect("render path");
        assert_eq!(grid.heights.len(), mesh_verts.len());
        for (h, v) in grid.heights.iter().zip(&mesh_verts) {
            assert_eq!(*h, v.pos[1]);
        }
        // Elevation is bracketed by the configured range and varies.
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for v in &mesh_verts {
            min_y = min_y.min(v.pos[1]);
            max_y = max_y.max(v.pos[1]);
        }
        assert!(min_y >= 0.0);
        assert!(max_y <= 10.0);
        assert!(
            max_y > min_y,
            "expected variation but got flat at {}",
            max_y
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn heightfield_missing_source_errors() {
        let args = serde_json::json!({ "generator": "heightfield", "elevation_max": 1.0 });
        let err = compile_mesh_payload(&args, None).unwrap_err();
        assert!(err.contains("source"), "got: {err}");
    }

    #[test]
    fn heightfield_unreadable_source_errors_with_the_heightfield_prefix() {
        let args = serde_json::json!({
            "generator": "heightfield",
            "source": "/no/such/terrain.png",
            "elevation_max": 1.0,
        });
        let err = compile_mesh_payload(&args, None).unwrap_err();
        assert!(err.starts_with("heightfield: "), "got: {err}");
        assert!(err.contains("/no/such/terrain.png"), "got: {err}");
    }

    #[test]
    fn non_heightfield_generator_delegates_to_the_generator_module() {
        let args = serde_json::json!({ "generator": "sphere", "radius": 1.0 });
        assert!(compile_mesh_payload(&args, None).is_ok());
    }
}
