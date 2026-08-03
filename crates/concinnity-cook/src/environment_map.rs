// src/environment_map.rs
//
// Compiles an EnvironmentMap component's args into a payload bundling two
// precomputed IBL cubemaps:
//
//   - **Irradiance cubemap.** Low-resolution (8x8 per face by default)
//     cosine-weighted hemisphere integral of the source. Used by the shader's
//     diffuse ambient term: `diffuse = (1-F)(1-metallic) * irradiance * albedo / π`.
//   - **Prefiltered radiance cubemap.** A mip chain where mip 0 = source and
//     mip N = source convolved with the GGX lobe at roughness = N / (mip_count - 1).
//     Used with the Karis env-BRDF analytic fit (already in every fragment shader
//     as `env_brdf_approx`) for the specular ambient term.
//
// A BRDF LUT is deliberately NOT shipped: the Karis polynomial fit
// (`env_brdf_approx` in default.metal / default_frag.hlsl / FRAG_GLSL) replaces
// it analytically. That keeps one binding slot free and dodges a build step.
//
// Source format: equirectangular Radiance HDR (.hdr), same as CubemapTexture.
// Sampling: Hammersley QMC + GGX importance sampling for prefilter, uniform
// (phi, theta) grid for irradiance.
//
// Payload format (little-endian):
//   u32  magic              = b"ENVM" = 0x4D564E45
//   u32  format_id          = 0  (RGBA32F)
//   u32  irradiance_face    (e.g. 8)
//   u32  prefilter_face     (mip 0 size, e.g. 512)
//   u32  prefilter_mips     (e.g. 5)
//   u32  _pad
//   ... irradiance cube         (6 * irradiance_face² * 16 bytes)
//   ... prefilter mip 0         (6 * prefilter_face² * 16 bytes)
//   ... prefilter mip 1         (6 * (prefilter_face/2)² * 16 bytes)
//   ...
//   ... prefilter mip (mips-1)  (6 * (prefilter_face >> (mips-1))² * 16 bytes)
//
// Face order matches CubemapTexture: +X, -X, +Y, -Y, +Z, -Z.

use serde::Deserialize;

use crate::hdr::{HdrImage, equirect_to_cube};
use concinnity_core::assets::EnvironmentMap;
use concinnity_core::build::environment_map::{
    DEFAULT_IRRADIANCE_PHI_SAMPLES, DEFAULT_IRRADIANCE_THETA_SAMPLES, compute_irradiance,
    compute_prefilter, max_mip_count, resolve_hdr_source, serialise_payload,
};

// Validation + entry point
//
// The three tunables (prefilter/irradiance face size, prefilter sample count)
// have a single source of truth: the `EnvironmentMap` `Default` impl in
// concinnity-core. Args are deserialised through that struct, so a field absent
// from the JSONL inherits the core default instead of a constant duplicated here.

fn resolve_args(args: &serde_json::Value) -> Result<EnvironmentMap, String> {
    let params: EnvironmentMap = Deserialize::deserialize(args)
        .map_err(|e| format!("invalid EnvironmentMap args: {}", e))?;
    match (params.source.is_empty(), params.generator.is_empty()) {
        (true, true) => return Err("EnvironmentMap requires either `source` or `generator`".into()),
        (false, false) => {
            return Err("EnvironmentMap takes either `source` or `generator`, not both".into());
        }
        (false, true) => {
            if !params.source.to_ascii_lowercase().ends_with(".hdr") {
                return Err(format!(
                    "EnvironmentMap source '{}' must be a Radiance .hdr file",
                    params.source
                ));
            }
        }
        (true, false) => match params.generator.as_str() {
            "sky" => {}
            other => return Err(format!("unknown EnvironmentMap generator '{}'", other)),
        },
    }
    let prefilter_face = params.prefilter_face_size;
    if !(16..=1024).contains(&prefilter_face) || !prefilter_face.is_power_of_two() {
        return Err(format!(
            "EnvironmentMap prefilter_face_size {} must be a power of two in 16..=1024",
            prefilter_face
        ));
    }
    let irradiance_face = params.irradiance_face_size;
    if !(8..=128).contains(&irradiance_face) || !irradiance_face.is_power_of_two() {
        return Err(format!(
            "EnvironmentMap irradiance_face_size {} must be a power of two in 8..=128",
            irradiance_face
        ));
    }
    if !params.prefilter_clamp.is_finite() || params.prefilter_clamp < 0.0 {
        return Err(format!(
            "EnvironmentMap prefilter_clamp {} must be a finite value >= 0 (0 disables it)",
            params.prefilter_clamp
        ));
    }
    Ok(params)
}

pub fn validate_environment_map_args(args: &serde_json::Value) -> Result<(), String> {
    resolve_args(args).map(|_| ())
}

pub fn compile_environment_map_payload(args: &serde_json::Value) -> Result<Vec<u8>, String> {
    let params = resolve_args(args)?;
    let prefilter_face = params.prefilter_face_size;
    let irradiance_face = params.irradiance_face_size;
    let prefilter_samples = params.prefilter_samples;

    let hdr = if !params.source.is_empty() {
        // A bare filename (no directory component) is resolved via the same
        // asset-search the build pipeline uses for shader sources: search
        // .concinnity/assets/ recursively, falling back to the raw path so an
        // absolute or relative path also works.
        let resolved = resolve_hdr_source(&params.source);
        crate::hdr::load_file(&resolved)?
    } else {
        match params.generator.as_str() {
            "sky" => generate_sky_equirect(),
            other => return Err(format!("unknown EnvironmentMap generator '{}'", other)),
        }
    };
    Ok(bake_payload(
        &hdr,
        prefilter_face,
        irradiance_face,
        prefilter_samples,
        params.prefilter_clamp,
    ))
}

// Convolve an equirectangular source into the serialised IBL payload (header +
// irradiance + prefilter mips). The single bake both the build pass and the
// hot-reload decode run, so a preview can never diverge from the built asset.
fn bake_payload(
    hdr: &HdrImage,
    prefilter_face: u32,
    irradiance_face: u32,
    prefilter_samples: u32,
    prefilter_clamp: f32,
) -> Vec<u8> {
    let source_cube = equirect_to_cube(hdr, prefilter_face);
    let prefilter_mips = max_mip_count(prefilter_face);
    let irradiance = compute_irradiance(
        &source_cube,
        prefilter_face,
        irradiance_face,
        DEFAULT_IRRADIANCE_PHI_SAMPLES,
        DEFAULT_IRRADIANCE_THETA_SAMPLES,
    );
    let prefilter = compute_prefilter(
        &source_cube,
        prefilter_face,
        prefilter_mips,
        prefilter_samples,
        prefilter_clamp,
        // Imported environment map: mip 0 IS the on-screen skybox, keep it unclamped.
        false,
    );
    serialise_payload(
        irradiance_face,
        prefilter_face,
        prefilter_mips,
        &irradiance,
        &prefilter,
    )
}

// Decode an EnvironmentMap source path the same way
// `compile_environment_map_payload` does at build time, returning the
// serialised payload (header + irradiance + prefilter mips). Exposed for the
// asset hot-reload path (`cn debug` only), which the editor drives; production
// reads the compiled payload from a blob locator instead. `prefilter_face`,
// `irradiance_face`, and `prefilter_samples` should be the values from the
// declared `EnvironmentMap` asset so the decode produces the same texture sizes
// as the build pass. The convolutions are CPU-bound and take seconds at default
// sizes: the caller pays this on the render thread.
pub fn decode_source(
    source: &str,
    prefilter_face: u32,
    irradiance_face: u32,
    prefilter_samples: u32,
    prefilter_clamp: f32,
) -> Result<Vec<u8>, String> {
    let resolved = resolve_hdr_source(source);
    let hdr = crate::hdr::load_file(&resolved)?;
    Ok(bake_payload(
        &hdr,
        prefilter_face,
        irradiance_face,
        prefilter_samples,
        prefilter_clamp,
    ))
}

// Synthetic equirectangular HDR for the `generator: "sky"` source. Same
// palette as the 2D `generate_sky` texture generator, extended to a full
// sphere: top half is zenith → mid → horizon, bottom half is solid horizon
// (no ground term yet, IBL only). Slightly super-1.0 values toward the sun
// direction give the prefilter convolution something HDR-like to chew on.
fn generate_sky_equirect() -> HdrImage {
    let width = 256u32;
    let height = 128u32;
    // Linear-light approximations of the procedural sky palette.
    let zenith = [0.012, 0.105, 0.526];
    let mid = [0.142, 0.355, 0.708];
    let horizon = [0.563, 0.726, 0.857];
    // Sun direction in equirect UV space: roughly south, 30° elevation.
    let sun_u = 0.25_f32;
    let sun_v = 0.35_f32;
    let sun_color = [3.0, 2.6, 2.1];
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for row in 0..height {
        let v = row as f32 / (height - 1) as f32;
        // Map v to a "sky elevation" t in [0, 1]: 0 at horizon, 1 at zenith.
        // Top half v∈[0, 0.5] maps to zenith→horizon, bottom half stays flat at horizon.
        let t = if v < 0.5 { 1.0 - v * 2.0 } else { 0.0 };
        let base = if t > 0.5 {
            let s = (t - 0.5) * 2.0;
            [
                lerp(mid[0], zenith[0], s),
                lerp(mid[1], zenith[1], s),
                lerp(mid[2], zenith[2], s),
            ]
        } else {
            let s = t * 2.0;
            let warm = (1.0 - s).powi(2) * 0.07;
            [
                lerp(horizon[0], mid[0], s) + warm * 0.5,
                lerp(horizon[1], mid[1], s) + warm * 0.25,
                lerp(horizon[2], mid[2], s),
            ]
        };
        for col in 0..width {
            let u = col as f32 / (width - 1) as f32;
            // Soft circular sun: gaussian-ish bump in equirect UV space.
            let du = (u - sun_u).abs();
            let du = du.min(1.0 - du); // wrap horizontally
            let dv = v - sun_v;
            let d2 = du * du + dv * dv;
            let sun_amt = (-d2 / 0.0006).exp();
            let r = base[0] + sun_color[0] * sun_amt;
            let g = base[1] + sun_color[1] * sun_amt;
            let b = base[2] + sun_color[2] * sun_amt;
            pixels.push([r, g, b]);
        }
    }
    HdrImage {
        width,
        height,
        pixels,
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::build::environment_map::deserialise;

    #[test]
    fn validate_environment_map_args_requires_source_or_generator() {
        let args = serde_json::json!({});
        let err = validate_environment_map_args(&args).unwrap_err();
        assert!(err.contains("source") || err.contains("generator"));
    }

    #[test]
    fn validate_environment_map_args_rejects_non_hdr() {
        let args = serde_json::json!({ "source": "studio.png" });
        let err = validate_environment_map_args(&args).unwrap_err();
        assert!(err.contains(".hdr"));
    }

    #[test]
    fn validate_environment_map_args_accepts_sky_generator() {
        let args = serde_json::json!({ "generator": "sky" });
        validate_environment_map_args(&args).expect("sky generator should validate");
    }

    #[test]
    fn validate_environment_map_args_rejects_both_source_and_generator() {
        let args = serde_json::json!({ "source": "x.hdr", "generator": "sky" });
        let err = validate_environment_map_args(&args).unwrap_err();
        assert!(err.contains("not both"));
    }

    #[test]
    fn validate_environment_map_args_rejects_out_of_range_prefilter_face() {
        // 8 is a power of two but below the 16..=1024 prefilter range.
        let args = serde_json::json!({ "generator": "sky", "prefilter_face_size": 8 });
        let err = validate_environment_map_args(&args).unwrap_err();
        assert!(err.contains("prefilter_face_size"), "got: {err}");
    }

    #[test]
    fn validate_environment_map_args_rejects_out_of_range_irradiance_face() {
        // 4 is a power of two but below the 8..=128 irradiance range.
        let args = serde_json::json!({ "generator": "sky", "irradiance_face_size": 4 });
        let err = validate_environment_map_args(&args).unwrap_err();
        assert!(err.contains("irradiance_face_size"), "got: {err}");
    }

    #[test]
    fn validate_environment_map_args_rejects_negative_prefilter_clamp() {
        let args = serde_json::json!({ "generator": "sky", "prefilter_clamp": -1.0 });
        let err = validate_environment_map_args(&args).unwrap_err();
        assert!(err.contains("prefilter_clamp"), "got: {err}");
    }

    #[test]
    fn validate_environment_map_args_rejects_unknown_generator() {
        let args = serde_json::json!({ "generator": "aurora" });
        let err = validate_environment_map_args(&args).unwrap_err();
        assert!(
            err.contains("unknown EnvironmentMap generator"),
            "got: {err}"
        );
    }

    #[test]
    fn validate_environment_map_args_rejects_mistyped_args() {
        let args = serde_json::json!({ "generator": "sky", "prefilter_face_size": "big" });
        let err = validate_environment_map_args(&args).unwrap_err();
        assert!(err.contains("invalid EnvironmentMap args"), "got: {err}");
    }

    #[test]
    fn compile_environment_map_payload_surfaces_a_corrupt_hdr() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("broken.hdr");
        std::fs::write(&path, b"not a radiance file\n").expect("write hdr");
        let args = serde_json::json!({ "source": path.to_str().unwrap() });
        let err = compile_environment_map_payload(&args).unwrap_err();
        assert!(err.contains("failed to decode HDR"), "got: {err}");
        assert!(err.contains("missing Radiance magic"), "got: {err}");
    }

    #[test]
    fn compile_environment_map_payload_surfaces_a_missing_hdr() {
        // A directory-qualified path resolves verbatim, so the load fails on open.
        let args = serde_json::json!({ "source": "/no/such/dir/missing.hdr" });
        let err = compile_environment_map_payload(&args).unwrap_err();
        assert!(err.contains("HDR"), "got: {err}");
    }

    #[test]
    fn sky_generator_compiles_into_full_payload() {
        let args = serde_json::json!({
            "generator": "sky",
            "prefilter_face_size": 16,
            "irradiance_face_size": 8,
            "prefilter_samples": 32,
        });
        let blob = compile_environment_map_payload(&args).expect("compile");
        let view = deserialise(&blob).expect("deserialise");
        assert_eq!(view.irradiance_face, 8);
        assert_eq!(view.prefilter_face, 16);
        // Prefilter mips for face_size 16: 16, 8, 4 → 3 levels.
        assert_eq!(view.prefilter_mip_bytes.len(), 3);
    }

    // Build a minimal uncompressed Radiance HDR blob of `width × height` solid
    // (r, g, b) pixels: the tiny encoder the `decode_source` tests feed in.
    fn synth_rgbe(r: f32, g: f32, b: f32) -> [u8; 4] {
        let maxv = r.max(g).max(b);
        if maxv < 1e-32 {
            return [0, 0, 0, 0];
        }
        let bits = maxv.to_bits();
        let raw_exp = ((bits >> 23) & 0xff) as i32;
        let exp = raw_exp - 126;
        let mantissa_bits = (bits & 0x7f_ffff) | (126 << 23);
        let mantissa = f32::from_bits(mantissa_bits);
        let scale = (mantissa * 256.0) / maxv;
        [
            (r * scale) as u8,
            (g * scale) as u8,
            (b * scale) as u8,
            (exp + 128) as u8,
        ]
    }

    fn raw_hdr_blob(width: u32, height: u32, rgb: [f32; 3]) -> Vec<u8> {
        let pixel = synth_rgbe(rgb[0], rgb[1], rgb[2]);
        let mut blob = Vec::new();
        blob.extend_from_slice(b"#?RADIANCE\n");
        blob.extend_from_slice(b"FORMAT=32-bit_rle_rgbe\n\n");
        blob.extend_from_slice(format!("-Y {} +X {}\n", height, width).as_bytes());
        for _ in 0..(width * height) {
            blob.extend_from_slice(&pixel);
        }
        blob
    }

    #[test]
    fn decode_source_missing_file_errors() {
        let err = decode_source("/definitely/does/not/exist.hdr", 16, 8, 16, 0.0)
            .expect_err("should fail");
        assert!(
            err.contains("failed to open") || err.contains("No such file"),
            "got: {}",
            err
        );
    }

    #[test]
    fn decode_source_of_an_unlit_hdr_integrates_to_zero_radiance() {
        // A source with no radiance anywhere convolves to a black irradiance
        // cube; only the alpha channel carries a value.
        let tmp = std::env::temp_dir().join(format!(
            "concinnity_envmap_black_test_{}.hdr",
            std::process::id()
        ));
        std::fs::write(&tmp, raw_hdr_blob(16, 8, [0.0, 0.0, 0.0])).expect("write hdr");
        let payload = decode_source(tmp.to_str().unwrap(), 16, 8, 16, 0.0).expect("decode");
        let _ = std::fs::remove_file(&tmp);
        let view = deserialise(&payload).expect("deserialise");
        for (i, texel) in view.irradiance_bytes.chunks_exact(4).enumerate() {
            if i % 4 == 3 {
                continue; // alpha
            }
            let v = f32::from_le_bytes(texel.try_into().unwrap());
            assert_eq!(v, 0.0, "irradiance float {i} was {v}");
        }
    }

    #[test]
    fn decode_source_round_trips_through_deserialise() {
        // Write a tiny solid-colour HDR into a tempfile, decode it, and verify
        // the resulting payload deserialises with the requested sizes.
        let tmp = std::env::temp_dir().join(format!(
            "concinnity_envmap_decode_test_{}.hdr",
            std::process::id()
        ));
        std::fs::write(&tmp, raw_hdr_blob(16, 8, [0.6, 0.3, 0.15])).expect("write hdr");
        let payload = decode_source(tmp.to_str().unwrap(), 16, 8, 16, 0.0).expect("decode");
        let _ = std::fs::remove_file(&tmp);
        let view = deserialise(&payload).expect("deserialise");
        assert_eq!(view.irradiance_face, 8);
        assert_eq!(view.prefilter_face, 16);
        // mip chain for face_size 16: 16, 8, 4 → 3 levels.
        assert_eq!(view.prefilter_mip_bytes.len(), 3);
        assert_eq!(view.prefilter_mip_bytes.len(), 3);
        assert_eq!(view.irradiance_bytes.len(), 6 * 8 * 8 * 4 * 4);
    }
}
