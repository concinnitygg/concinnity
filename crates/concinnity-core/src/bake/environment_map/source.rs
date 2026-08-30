//! Equirectangular sources for the IBL bake: the in-memory image type, the
//! resampler that turns one into the six cube faces the convolutions read,
//! the built-in `sky` generator, and the full source-to-payload bake. Decoding
//! a source *file* into an [`HdrImage`] is the cook crate's job; nothing here
//! touches I/O.

use alloc::vec;
use alloc::vec::Vec;

use super::bake::{CubeBake, prefilter_mip0};
use super::schedule::RowScheduler;
use super::{
    DEFAULT_IRRADIANCE_PHI_SAMPLES, DEFAULT_IRRADIANCE_THETA_SAMPLES, max_mip_count,
    prefilter_roughness, serialise_payload,
};
use crate::math::{acos, atan2, exp, floor, powi, sqrt};

/// An equirectangular radiance image: linear RGB rows, top-down.
#[derive(Debug, Clone)]
pub struct HdrImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major top-down linear RGB triples, `width * height` of them.
    pub pixels: Vec<[f32; 3]>,
}

/// Resample an equirectangular HDR image into six square cube faces of
/// `face_size` pixels. Output is RGBA32F (alpha = 1.0) row-major top-down,
/// matching the Metal / Vulkan / DX cube convention.
pub fn equirect_to_cube(hdr: &HdrImage, face_size: u32) -> [Vec<f32>; 6] {
    let f = face_size as usize;
    let mut faces: [Vec<f32>; 6] = core::array::from_fn(|_| vec![0.0; f * f * 4]);
    for (face, face_buf) in faces.iter_mut().enumerate() {
        for y in 0..f {
            for x in 0..f {
                // Map pixel center to NDC [-1, 1].
                let u = (x as f32 + 0.5) / face_size as f32 * 2.0 - 1.0;
                let v = (y as f32 + 0.5) / face_size as f32 * 2.0 - 1.0;
                let dir = face_uv_to_dir(face, u, v);
                let sample = sample_equirect(hdr, dir);
                let off = (y * f + x) * 4;
                face_buf[off] = sample[0];
                face_buf[off + 1] = sample[1];
                face_buf[off + 2] = sample[2];
                face_buf[off + 3] = 1.0;
            }
        }
    }
    faces
}

// Convert a face index + face UV in NDC [-1, 1] to a world-space direction.
// Face order: 0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z.
fn face_uv_to_dir(face: usize, u: f32, v: f32) -> [f32; 3] {
    let d = match face {
        0 => [1.0, -v, -u],
        1 => [-1.0, -v, u],
        2 => [u, 1.0, v],
        3 => [u, -1.0, -v],
        4 => [u, -v, 1.0],
        5 => [-u, -v, -1.0],
        _ => unreachable!("invalid cube face index {}", face),
    };
    normalize3(d)
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = length(v).max(1e-20);
    [v[0] / l, v[1] / l, v[2] / l]
}

fn length(v: [f32; 3]) -> f32 {
    sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
}

fn sample_equirect(hdr: &HdrImage, dir: [f32; 3]) -> [f32; 3] {
    let phi = atan2(dir[2], dir[0]); // [-π, π]
    let theta = acos(dir[1].clamp(-1.0, 1.0)); // [0, π]
    let u = phi / (2.0 * core::f32::consts::PI) + 0.5;
    let v = theta / core::f32::consts::PI;
    let fx = u * hdr.width as f32 - 0.5;
    let fy = v * hdr.height as f32 - 0.5;
    let x0 = floor(fx) as i32;
    let y0 = floor(fy) as i32;
    let dx = fx - x0 as f32;
    let dy = fy - y0 as f32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let w00 = (1.0 - dx) * (1.0 - dy);
    let w10 = dx * (1.0 - dy);
    let w01 = (1.0 - dx) * dy;
    let w11 = dx * dy;
    let p00 = fetch_wrap(hdr, x0, y0);
    let p10 = fetch_wrap(hdr, x1, y0);
    let p01 = fetch_wrap(hdr, x0, y1);
    let p11 = fetch_wrap(hdr, x1, y1);
    [
        p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11,
        p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11,
        p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11,
    ]
}

fn fetch_wrap(hdr: &HdrImage, x: i32, y: i32) -> [f32; 3] {
    // Horizontal wrap (longitude), vertical clamp (latitude poles).
    let w = hdr.width as i32;
    let h = hdr.height as i32;
    let xw = x.rem_euclid(w);
    let yc = y.clamp(0, h - 1);
    hdr.pixels[(yc * w + xw) as usize]
}

/// Convolve an equirectangular source into the serialised IBL payload:
/// header, irradiance cube, prefilter mips. One bake serves the cook
/// pipeline, the editor's hot-reload preview, and a runtime bake, so no path
/// can diverge from the built asset; `rows` spreads each convolution's
/// independent output rows over whatever the caller has
/// ([`super::schedule::Serial`] without a pool).
pub fn bake_payload<S: RowScheduler>(
    hdr: &HdrImage,
    prefilter_face: u32,
    irradiance_face: u32,
    prefilter_samples: u32,
    prefilter_clamp: f32,
    rows: &S,
) -> Vec<u8> {
    let source_cube = equirect_to_cube(hdr, prefilter_face);
    let prefilter_mips = max_mip_count(prefilter_face);
    let irradiance = CubeBake::irradiance(
        &source_cube,
        prefilter_face,
        irradiance_face,
        DEFAULT_IRRADIANCE_PHI_SAMPLES,
        DEFAULT_IRRADIANCE_THETA_SAMPLES,
    )
    .bake(rows);
    // The source-resolution mip 0 IS the on-screen skybox, keep it unclamped.
    let mut prefilter = Vec::with_capacity(prefilter_mips as usize);
    prefilter.push(prefilter_mip0(
        &source_cube,
        prefilter_face,
        prefilter_clamp,
        false,
    ));
    for mip in 1..prefilter_mips {
        prefilter.push(
            CubeBake::ggx(
                &source_cube,
                prefilter_face,
                prefilter_face >> mip,
                prefilter_roughness(mip, prefilter_mips),
                prefilter_samples,
                prefilter_clamp,
            )
            .bake(rows),
        );
    }
    serialise_payload(
        irradiance_face,
        prefilter_face,
        prefilter_mips,
        &irradiance,
        &prefilter,
    )
}

/// Synthetic equirectangular HDR for the `generator: "sky"` source. Same
/// palette as the 2D `generate_sky` texture generator, extended to a full
/// sphere: top half is zenith → mid → horizon, bottom half is solid horizon
/// (no ground term yet, IBL only). Slightly super-1.0 values toward the sun
/// direction give the prefilter convolution something HDR-like to chew on.
pub fn generate_sky_equirect() -> HdrImage {
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
            let warm = powi(1.0 - s, 2) * 0.07;
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
            let sun_amt = exp(-d2 / 0.0006);
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

#[cfg(test)]
mod tests {
    use super::super::deserialise;
    use super::super::schedule::Serial;
    use super::*;

    #[test]
    fn sky_generator_bakes_into_a_full_payload_serially() {
        let hdr = generate_sky_equirect();
        assert_eq!((hdr.width, hdr.height), (256, 128));
        let payload = bake_payload(&hdr, 16, 8, 32, 12.0, &Serial);
        let view = deserialise(&payload).expect("deserialise");
        assert_eq!(view.irradiance_face, 8);
        assert_eq!(view.prefilter_face, 16);
        // Prefilter mips for face_size 16: 16, 8, 4 → 3 levels.
        assert_eq!(view.prefilter_mip_bytes.len(), 3);
    }

    #[test]
    fn equirect_solid_color_produces_solid_cube() {
        let pixel = [0.8f32, 0.4, 0.1];
        let hdr = HdrImage {
            width: 32,
            height: 16,
            pixels: vec![pixel; 32 * 16],
        };
        let faces = equirect_to_cube(&hdr, 16);
        for (idx, face) in faces.iter().enumerate() {
            assert_eq!(face.len(), 16 * 16 * 4);
            for px in face.chunks_exact(4) {
                assert!((px[0] - pixel[0]).abs() < 1e-4, "face {} R", idx);
                assert!((px[1] - pixel[1]).abs() < 1e-4, "face {} G", idx);
                assert!((px[2] - pixel[2]).abs() < 1e-4, "face {} B", idx);
                assert!((px[3] - 1.0).abs() < 1e-6, "face {} A", idx);
            }
        }
    }

    #[test]
    fn equirect_red_seam_lights_only_the_minus_x_face() {
        // Paint a four-pixel-wide red band on the equirect straddling the
        // longitude = ±π seam (columns {30, 31, 0, 1} for a 32-wide image).
        // The -X face is centered on that longitude; +X is on the opposite
        // side and should see almost no red.
        let mut pixels = vec![[0.0f32; 3]; 32 * 16];
        for y in 0..16 {
            for &x in &[30usize, 31, 0, 1] {
                pixels[y * 32 + x] = [10.0, 0.0, 0.0];
            }
        }
        let hdr = HdrImage {
            width: 32,
            height: 16,
            pixels,
        };
        let faces = equirect_to_cube(&hdr, 16);
        let mean_red = |face: &[f32]| -> f32 {
            let n = face.len() / 4;
            face.chunks_exact(4).map(|p| p[0]).sum::<f32>() / n as f32
        };
        let plus_x = mean_red(&faces[0]);
        let minus_x = mean_red(&faces[1]);
        assert!(
            minus_x > 5.0 * plus_x.max(0.001),
            "-X mean red ({}) should dwarf +X mean red ({})",
            minus_x,
            plus_x
        );
    }

    #[test]
    #[should_panic(expected = "invalid cube face index 6")]
    fn face_uv_to_dir_rejects_an_out_of_range_face() {
        let _ = face_uv_to_dir(6, 0.0, 0.0);
    }
}
