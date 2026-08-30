// JSON arg parsing for the heightfield generator (a terrain grid driven by a
// grayscale heightmap image); the geometry itself is
// `concinnity_core::geometry::build_heightfield_from_pixels`. The source image
// is decoded by this crate before the pixels are passed in -- the runtime
// crates link no image decoders.

type Verts = Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 2])>;

pub(super) fn build_heightfield_from_pixels(
    args: &serde_json::Value,
    img_w: u32,
    img_h: u32,
    rgba: &[u8],
) -> Result<(Verts, Vec<u16>), String> {
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
    let elevation_min = args
        .get("elevation_min")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let elevation_max = args
        .get("elevation_max")
        .and_then(|v| v.as_f64())
        .ok_or("heightfield generator requires `elevation_max`")? as f32;

    concinnity_core::geometry::build_heightfield_from_pixels(
        &concinnity_core::geometry::HeightfieldField {
            half_width,
            half_depth,
            subdivisions,
            elevation_min,
            elevation_max,
        },
        img_w,
        img_h,
        rgba,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // A `w`x`h` grayscale-RGBA buffer whose red channel ramps 0..255 across X
    // so the generated mesh has real elevation variation to sample.
    fn ramp_rgba(w: u32, h: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..h {
            for x in 0..w {
                let v = if w > 1 { (x * 255 / (w - 1)) as u8 } else { 0 };
                out.extend_from_slice(&[v, v, v, 255]);
            }
        }
        out
    }

    #[test]
    fn requires_elevation_max() {
        let args = serde_json::json!({ "subdivisions": 3 });
        let rgba = ramp_rgba(4, 4);
        let err = build_heightfield_from_pixels(&args, 4, 4, &rgba).unwrap_err();
        assert!(err.contains("elevation_max"), "got: {}", err);
    }

    #[test]
    fn rejects_zero_extent_image() {
        let args = serde_json::json!({ "subdivisions": 3, "elevation_max": 1.0 });
        let err = build_heightfield_from_pixels(&args, 0, 0, &[]).unwrap_err();
        assert!(err.contains("zero extent"), "got: {}", err);
    }

    #[test]
    fn a_single_pixel_heightmap_produces_a_flat_mesh() {
        let args = serde_json::json!({
            "half_width": 1.0,
            "half_depth": 1.0,
            "subdivisions": 4,
            "elevation_min": 2.0,
            "elevation_max": 9.0,
        });
        let rgba = ramp_rgba(1, 1);
        let (verts, _) = build_heightfield_from_pixels(&args, 1, 1, &rgba).expect("builds");
        // The lone pixel's red channel is 0, so every sample lands on the
        // elevation floor and the surface stays level.
        assert!(verts.iter().all(|(pos, ..)| pos[1] == 2.0));
        assert!(verts.iter().all(|(_, n, ..)| *n == [0.0, 1.0, 0.0]));
    }

    #[test]
    fn subdivisions_clamp_to_the_supported_range() {
        let args = |subdiv: u64| serde_json::json!({"subdivisions": subdiv, "elevation_max": 1.0});
        let rgba = ramp_rgba(4, 4);
        let (small, _) = build_heightfield_from_pixels(&args(0), 4, 4, &rgba).unwrap();
        assert_eq!(small.len(), 5 * 5);
        // 255 is the largest grid that still indexes with u16.
        let (large, idxs) = build_heightfield_from_pixels(&args(4096), 4, 4, &rgba).unwrap();
        assert_eq!(large.len(), 256 * 256);
        assert_eq!(idxs.len(), 255 * 255 * 6);
    }
}
