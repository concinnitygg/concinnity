//! Compiles a CubemapTexture component's args into the binary payload that the
//! renderer reads at runtime. A cubemap is six square HDR faces stored as
//! RGBA32F in face-major order (face 0 → face 5, each face row-major top-down).
//!
//! Source format: equirectangular Radiance HDR (.hdr / RGBE). The
//! equirect is resampled at build time into six cube faces using bilinear
//! interpolation in HDR space.
//!
//! Payload format (little-endian):
//!   u32  magic     = b"CUBE" = 0x45425543
//!   u32  face_size
//!   u32  mip_count = 1
//!   u32  format_id = 0  (RGBA32F)
//!   6 * face_size * face_size * 4 * 4 bytes  raw RGBA32F, face-major
//!
//! Face order matches the standard cube convention used by Metal / Vulkan / DX:
//!   0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z, 5: -Z

use crate::hdr::{
    CUBE_FORMAT_RGBA32F, CUBE_PAYLOAD_HEADER_BYTES, CUBE_PAYLOAD_MAGIC, equirect_to_cube,
};

// The `source` path a `CubemapTexture`'s args declare, checked for extension.
fn cubemap_source(args: &serde_json::Value) -> Result<&str, String> {
    let source = args.get("source").and_then(|v| v.as_str()).unwrap_or("");
    if source.is_empty() {
        return Err("CubemapTexture requires a `source` path".into());
    }
    if !source.to_ascii_lowercase().ends_with(".hdr") {
        return Err(format!(
            "CubemapTexture source '{}' must be a Radiance .hdr file",
            source
        ));
    }
    Ok(source)
}

// Validate that args specify either a supported source extension or omit it.
pub(crate) fn validate_cubemap_args(args: &serde_json::Value) -> Result<(), String> {
    cubemap_source(args)?;
    let face_size = args
        .get("face_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(256);
    if !(8..=4096).contains(&face_size) {
        return Err(format!(
            "CubemapTexture face_size {} out of range (8..=4096)",
            face_size
        ));
    }
    if !face_size.is_power_of_two() {
        return Err(format!(
            "CubemapTexture face_size {} must be a power of two",
            face_size
        ));
    }
    Ok(())
}

// Compile a CubemapTexture component's JSON args into a packed binary payload.
pub(crate) fn compile_cubemap_payload(args: &serde_json::Value) -> Result<Vec<u8>, String> {
    validate_cubemap_args(args)?;
    let source = cubemap_source(args)?;
    let face_size = args
        .get("face_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(256) as u32;

    let hdr = crate::hdr::load_file(source)?;
    let faces = equirect_to_cube(&hdr, face_size);
    Ok(serialise_faces(face_size, &faces))
}

fn serialise_faces(face_size: u32, faces: &[Vec<f32>; 6]) -> Vec<u8> {
    let face_floats = (face_size as usize) * (face_size as usize) * 4;
    let mut buf = Vec::with_capacity(CUBE_PAYLOAD_HEADER_BYTES + 6 * face_floats * 4);
    buf.extend_from_slice(&CUBE_PAYLOAD_MAGIC.to_le_bytes());
    buf.extend_from_slice(&face_size.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&CUBE_FORMAT_RGBA32F.to_le_bytes());
    for face in faces {
        debug_assert_eq!(face.len(), face_floats);
        buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(face));
    }
    buf
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdr::{HdrImage, deserialise};

    #[test]
    fn payload_round_trip_via_deserialise() {
        let pixel = [0.7f32, 0.3, 0.2];
        let hdr = HdrImage {
            width: 16,
            height: 8,
            pixels: vec![pixel; 16 * 8],
        };
        let faces = equirect_to_cube(&hdr, 8);
        let blob = serialise_faces(8, &faces);
        let (face_size, face_bytes) = deserialise(&blob).expect("deserialise");
        assert_eq!(face_size, 8);
        assert_eq!(face_bytes.len(), 6 * 8 * 8 * 4 * 4);
        // First face, first pixel:
        let p0 = f32::from_le_bytes(face_bytes[0..4].try_into().unwrap());
        let p1 = f32::from_le_bytes(face_bytes[4..8].try_into().unwrap());
        let p2 = f32::from_le_bytes(face_bytes[8..12].try_into().unwrap());
        let p3 = f32::from_le_bytes(face_bytes[12..16].try_into().unwrap());
        assert!((p0 - pixel[0]).abs() < 1e-4);
        assert!((p1 - pixel[1]).abs() < 1e-4);
        assert!((p2 - pixel[2]).abs() < 1e-4);
        assert!((p3 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn validate_cubemap_args_rejects_bad_face_size() {
        let args = serde_json::json!({ "source": "foo.hdr", "face_size": 300 });
        let err = validate_cubemap_args(&args).unwrap_err();
        assert!(err.contains("power of two"), "got: {}", err);
    }

    #[test]
    fn validate_cubemap_args_requires_hdr_extension() {
        let args = serde_json::json!({ "source": "foo.png" });
        let err = validate_cubemap_args(&args).unwrap_err();
        assert!(err.contains(".hdr"), "got: {}", err);
    }

    #[test]
    fn validate_cubemap_args_accepts_defaults() {
        let args = serde_json::json!({ "source": "studio.hdr" });
        validate_cubemap_args(&args).expect("defaults should validate");
    }

    #[test]
    fn validate_cubemap_args_requires_a_source() {
        let err = validate_cubemap_args(&serde_json::json!({})).unwrap_err();
        assert!(err.contains("requires a `source`"), "got: {err}");
    }

    #[test]
    fn validate_cubemap_args_rejects_out_of_range_face_size() {
        // 4 is a power of two but below the 8..=4096 range.
        let args = serde_json::json!({ "source": "studio.hdr", "face_size": 4 });
        let err = validate_cubemap_args(&args).unwrap_err();
        assert!(err.contains("out of range"), "got: {err}");
    }

    #[test]
    fn compile_cubemap_payload_surfaces_a_missing_hdr() {
        // A directory-qualified path resolves verbatim; opening it fails.
        let args = serde_json::json!({ "source": "/no/such/dir/missing.hdr", "face_size": 8 });
        let err = compile_cubemap_payload(&args).unwrap_err();
        assert!(err.contains("failed to open HDR"), "got: {err}");
    }

    fn write_hdr(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> String {
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).expect("write hdr");
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn compile_cubemap_payload_resamples_a_source_hdr_into_six_faces() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rgb = [0.5f32, 0.25, 0.125];
        let src = write_hdr(
            &dir,
            "studio.hdr",
            &crate::hdr::test_fixtures::solid_hdr_blob(16, 8, rgb),
        );
        let args = serde_json::json!({ "source": src, "face_size": 8 });
        let payload = compile_cubemap_payload(&args).expect("compile");

        let (face_size, face_bytes) = deserialise(&payload).expect("deserialise");
        assert_eq!(face_size, 8);
        assert_eq!(face_bytes.len(), 6 * 8 * 8 * 4 * 4);
        // A solid equirect resamples to the same colour on every face.
        for (i, texel) in face_bytes.chunks_exact(4).enumerate() {
            let v = f32::from_le_bytes(texel.try_into().unwrap());
            let want = if i % 4 == 3 { 1.0 } else { rgb[i % 4] };
            assert!((v - want).abs() < 0.01, "float {i} was {v}, want {want}");
        }
    }

    #[test]
    fn compile_cubemap_payload_surfaces_a_corrupt_hdr() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_hdr(&dir, "broken.hdr", b"not a radiance file\n");
        let args = serde_json::json!({ "source": src, "face_size": 8 });
        let err = compile_cubemap_payload(&args).unwrap_err();
        assert!(err.contains("failed to decode HDR"), "got: {err}");
        assert!(err.contains("missing Radiance magic"), "got: {err}");
    }
}
