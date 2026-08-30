//! Compiles an EnvironmentMap component's args into a payload bundling two
//! precomputed IBL cubemaps:
//!
//!   - **Irradiance cubemap.** Low-resolution (32x32 per face by default)
//!     cosine-weighted hemisphere integral of the source. Used by the shader's
//!     diffuse ambient term: `diffuse = (1-F)(1-metallic) * irradiance * albedo / π`.
//!   - **Prefiltered radiance cubemap.** A mip chain where mip 0 = source and
//!     mip N = source convolved with the GGX lobe at roughness = N / (mip_count - 1).
//!     Used with the Karis env-BRDF analytic fit (already in every fragment shader
//!     as `env_brdf_approx`) for the specular ambient term.
//!
//! A BRDF LUT is deliberately NOT shipped: the Karis polynomial fit
//! (`env_brdf_approx` in main.metal / main_frag.hlsl / FRAG_GLSL) replaces
//! it analytically. That keeps one binding slot free and dodges a build step.
//!
//! Source format: equirectangular Radiance HDR (.hdr), same as CubemapTexture.
//! Sampling: Hammersley QMC + GGX importance sampling for prefilter, uniform
//! (phi, theta) grid for irradiance.
//!
//! The convolutions themselves are [`bake`], which decomposes them into
//! independent output rows so a caller can spread the work over its own thread
//! pool. This module is the payload format around them.
//!
//! Payload format (little-endian):
//!   u32  magic              = b"ENVM" = 0x4D564E45
//!   u32  format_id          = 0  (RGBA32F)
//!   u32  irradiance_face    (e.g. 32)
//!   u32  prefilter_face     (mip 0 size, e.g. 512)
//!   u32  prefilter_mips     (e.g. 5)
//!   u32  _pad
//!   ... irradiance cube         (6 * irradiance_face² * 16 bytes)
//!   ... prefilter mip 0         (6 * prefilter_face² * 16 bytes)
//!   ... prefilter mip 1         (6 * (prefilter_face/2)² * 16 bytes)
//!   ...
//!   ... prefilter mip (mips-1)  (6 * (prefilter_face >> (mips-1))² * 16 bytes)
//!
//! Face order matches CubemapTexture: +X, -X, +Y, -Y, +Z, -Z.

pub mod bake;
pub mod schedule;
pub mod source;

use crate::decode::{ByteReader, checked_product};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

pub use bake::{
    CubeBake, DEFAULT_IRRADIANCE_PHI_SAMPLES, DEFAULT_IRRADIANCE_THETA_SAMPLES, FaceRow,
    compute_irradiance, compute_prefilter, face_rows, prefilter_mip0, prefilter_roughness,
};
pub use schedule::{RowScheduler, Serial};

pub(crate) const ENVMAP_PAYLOAD_MAGIC: u32 = u32::from_le_bytes(*b"ENVM");
pub(crate) const ENVMAP_FORMAT_RGBA32F: u32 = 0;
pub(crate) const ENVMAP_PAYLOAD_HEADER_BYTES: usize = 24;

/// Check an `EnvironmentMap`'s baked dimensions: both cube edges must be
/// powers of two inside the range the shader's mip assumptions hold for, and
/// the reflection clamp must be a finite non-negative gain. Shared by the cook
/// pipeline and the [`bake`](crate::bake) builder so a world is refused the
/// same way whichever declared it.
pub fn check_sizes(map: &crate::components::EnvironmentMap) -> Result<(), String> {
    let prefilter_face = map.prefilter_face_size;
    if !(16..=1024).contains(&prefilter_face) || !prefilter_face.is_power_of_two() {
        return Err(format!(
            "EnvironmentMap prefilter_face_size {} must be a power of two in 16..=1024",
            prefilter_face
        ));
    }
    let irradiance_face = map.irradiance_face_size;
    if !(8..=128).contains(&irradiance_face) || !irradiance_face.is_power_of_two() {
        return Err(format!(
            "EnvironmentMap irradiance_face_size {} must be a power of two in 8..=128",
            irradiance_face
        ));
    }
    if !map.prefilter_clamp.is_finite() || map.prefilter_clamp < 0.0 {
        return Err(format!(
            "EnvironmentMap prefilter_clamp {} must be a finite value >= 0 (0 disables it)",
            map.prefilter_clamp
        ));
    }
    Ok(())
}

/// Number of mip levels for a square cube face of `face_size` pixels. The
/// smallest mip is clamped to 4×4 to keep the prefilter convolution sensible
/// at high roughness.
pub const fn max_mip_count(face_size: u32) -> u32 {
    let mut mips = 0u32;
    let mut s = face_size;
    while s >= 4 {
        mips += 1;
        s /= 2;
    }
    mips
}

// Payload codec

/// Pack the baked irradiance and prefilter cubes into a blob payload.
pub fn serialise_payload(
    irradiance_face: u32,
    prefilter_face: u32,
    prefilter_mips: u32,
    irradiance: &[Vec<f32>; 6],
    prefilter: &[[Vec<f32>; 6]],
) -> Vec<u8> {
    debug_assert_eq!(prefilter.len(), prefilter_mips as usize);
    let mut total = ENVMAP_PAYLOAD_HEADER_BYTES + 6 * (irradiance_face as usize).pow(2) * 4 * 4;
    for mip in 0..prefilter_mips {
        let s = (prefilter_face >> mip) as usize;
        total += 6 * s * s * 4 * 4;
    }
    let mut buf = Vec::with_capacity(total);
    buf.extend_from_slice(&ENVMAP_PAYLOAD_MAGIC.to_le_bytes());
    buf.extend_from_slice(&ENVMAP_FORMAT_RGBA32F.to_le_bytes());
    buf.extend_from_slice(&irradiance_face.to_le_bytes());
    buf.extend_from_slice(&prefilter_face.to_le_bytes());
    buf.extend_from_slice(&prefilter_mips.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // pad
    for face in irradiance {
        buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(face));
    }
    for mip in prefilter {
        for face in mip {
            buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(face));
        }
    }
    buf
}

/// Metadata read from a serialised EnvironmentMap payload. The byte ranges
/// point into the payload buffer so the runtime can upload them directly.
#[derive(Debug)]
pub struct EnvMapView<'a> {
    /// Irradiance cube edge in pixels.
    pub irradiance_face: u32,
    /// Prefilter cube edge in pixels at mip 0.
    pub prefilter_face: u32,
    /// The six irradiance faces, RGBA32F.
    pub irradiance_bytes: &'a [u8],
    /// One slice per prefilter mip, ordered mip 0 → mip N-1.
    pub prefilter_mip_bytes: Vec<&'a [u8]>,
}

// Deserialise a packed EnvironmentMap payload back into byte-range views into
// the buffer. The runtime upload path uses this to feed the per-face slices
// to the GPU without copying. Called by every backend at init time, and by
// the Metal hot-reload path via `update_environment_map`.
// Bytes the six RGBA32F faces of a cube with edge length `edge` occupy.
fn cube_face_bytes(label: &str, edge: u32) -> Result<usize, String> {
    checked_product(label, &[6, edge as usize, edge as usize, 4, 4])
}

/// Read a packed payload back as byte-range views into `bytes`.
pub fn deserialise(bytes: &[u8]) -> Result<EnvMapView<'_>, String> {
    let mut r = ByteReader::open_payload(
        bytes,
        ENVMAP_PAYLOAD_MAGIC,
        ENVMAP_PAYLOAD_HEADER_BYTES,
        "envmap",
    )?;
    let format = r.u32()?;
    if format != ENVMAP_FORMAT_RGBA32F {
        return Err(format!("envmap format_id {} unsupported", format));
    }
    let irradiance_face = r.u32()?;
    let prefilter_face = r.u32()?;
    let prefilter_mips = r.u32()?;
    if prefilter_mips == 0 || prefilter_mips > 12 {
        return Err(format!(
            "envmap prefilter_mips {} out of range",
            prefilter_mips
        ));
    }
    // Face edges are payload-supplied, so each section's footprint is checked
    // before it is used as a length. The seek skips the header's trailing pad.
    r.seek(ENVMAP_PAYLOAD_HEADER_BYTES)?;
    let irradiance_bytes = r.take(cube_face_bytes("envmap irradiance", irradiance_face)?)?;
    let mut prefilter_mip_bytes = Vec::with_capacity(prefilter_mips as usize);
    for mip in 0..prefilter_mips {
        let edge = prefilter_face >> mip;
        let mip_size = cube_face_bytes("envmap prefilter mip", edge)?;
        prefilter_mip_bytes.push(r.take(mip_size)?);
    }
    Ok(EnvMapView {
        irradiance_face,
        prefilter_face,
        irradiance_bytes,
        prefilter_mip_bytes,
    })
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_cube(face_size: u32, color: [f32; 3]) -> [Vec<f32>; 6] {
        let f = face_size as usize;
        core::array::from_fn(|_| {
            let mut face = Vec::with_capacity(f * f * 4);
            for _ in 0..f * f {
                face.extend_from_slice(&[color[0], color[1], color[2], 1.0]);
            }
            face
        })
    }

    #[test]
    fn payload_round_trip() {
        let source = solid_cube(8, [0.6, 0.4, 0.2]);
        let irr = compute_irradiance(&source, 8, 4, 32, 8);
        let prefilter = compute_prefilter(&source, 8, 2, 16, 0.0, false);
        let blob = serialise_payload(4, 8, 2, &irr, &prefilter);
        let view = deserialise(&blob).expect("deserialise");
        assert_eq!(view.irradiance_face, 4);
        assert_eq!(view.prefilter_face, 8);
        assert_eq!(view.prefilter_mip_bytes.len(), 2);
        assert_eq!(view.irradiance_bytes.len(), 6 * 4 * 4 * 4 * 4);
        assert_eq!(view.prefilter_mip_bytes[0].len(), 6 * 8 * 8 * 4 * 4);
        assert_eq!(view.prefilter_mip_bytes[1].len(), 6 * 4 * 4 * 4 * 4);
    }

    #[test]
    fn max_mip_count_clamps_at_four_pixels() {
        assert_eq!(max_mip_count(256), 7); // 256, 128, 64, 32, 16, 8, 4
        assert_eq!(max_mip_count(16), 3); // 16, 8, 4
        assert_eq!(max_mip_count(8), 2); // 8, 4
        assert_eq!(max_mip_count(4), 1); // 4
    }

    // A header with plausible fields but no body behind them.
    fn header_only(irradiance_face: u32, prefilter_face: u32, prefilter_mips: u32) -> Vec<u8> {
        let mut buf = ENVMAP_PAYLOAD_MAGIC.to_le_bytes().to_vec();
        buf.extend_from_slice(&ENVMAP_FORMAT_RGBA32F.to_le_bytes());
        buf.extend_from_slice(&irradiance_face.to_le_bytes());
        buf.extend_from_slice(&prefilter_face.to_le_bytes());
        buf.extend_from_slice(&prefilter_mips.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    #[test]
    fn rejects_a_payload_shorter_than_the_header() {
        let full = header_only(4, 8, 2);
        for len in 0..ENVMAP_PAYLOAD_HEADER_BYTES {
            assert!(deserialise(&full[..len]).is_err(), "len {} decoded", len);
        }
    }

    #[test]
    fn rejects_a_bad_magic() {
        let mut bytes = header_only(4, 8, 2);
        bytes[..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        assert!(deserialise(&bytes).is_err());
    }

    #[test]
    fn rejects_a_payload_truncated_in_the_irradiance_section() {
        let mut bytes = header_only(4, 8, 2);
        bytes.extend(core::iter::repeat_n(0u8, 6 * 4 * 4 * 4 * 4 - 8));
        let err = deserialise(&bytes).unwrap_err();
        assert!(err.contains("unexpected end"), "{}", err);
    }

    #[test]
    fn rejects_a_payload_truncated_in_a_prefilter_mip() {
        let mut bytes = header_only(4, 8, 2);
        bytes.extend(core::iter::repeat_n(0u8, 6 * 4 * 4 * 4 * 4));
        bytes.extend(core::iter::repeat_n(0u8, 6 * 8 * 8 * 4 * 4));
        // Second mip (4x4 faces) is missing entirely.
        let err = deserialise(&bytes).unwrap_err();
        assert!(err.contains("unexpected end"), "{}", err);
    }

    // A face edge near u32::MAX makes `6 * edge * edge * 16` wrap; the wrapped
    // product would be small enough to pass a length check and hand out a
    // slice unrelated to the real section.
    #[test]
    fn rejects_an_irradiance_face_that_overflows_its_footprint() {
        let bytes = header_only(u32::MAX, 8, 2);
        let err = deserialise(&bytes).unwrap_err();
        assert!(err.contains("overflow"), "{}", err);
    }

    #[test]
    fn rejects_a_prefilter_face_that_overflows_its_footprint() {
        let mut bytes = header_only(4, u32::MAX, 2);
        bytes.extend(core::iter::repeat_n(0u8, 6 * 4 * 4 * 4 * 4));
        let err = deserialise(&bytes).unwrap_err();
        assert!(err.contains("overflow"), "{}", err);
    }

    #[test]
    fn rejects_out_of_range_mip_counts() {
        assert!(deserialise(&header_only(4, 8, 0)).is_err());
        assert!(deserialise(&header_only(4, 8, 13)).is_err());
    }
}
